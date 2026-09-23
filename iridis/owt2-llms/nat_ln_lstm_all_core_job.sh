#!/bin/bash
#SBATCH --job-name=nat_ln_lstm_owt2
#SBATCH --partition=i7_h200
#SBATCH --account=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
################################################################################


N_GPUS=2          # Must match --gres=gpu:N above
N_LAYER=12        # Paper uses 24 for Table 2 ablation
N_REPEAT=8        # Number of block repeats
ITERATIONS=40000  # Training steps
BATCH_SIZE=32       # Per-GPU micro-batch 
ACC_STEPS=2      # Gradient accumulation 
CKPT_FREQ=2000    # Save and evaluate every 2000 steps. This controls evaluation frequency as well
N_EMBD=384

# Reserved layers:
N_LAYER_BEGIN=2
N_LAYER_END=1

# ========================= END CONFIGURATION ================================
set -euo pipefail
LSTM_PERSISTENT_CELL=true
CELL_MODE="persistent"

if [ "$LSTM_PERSISTENT_CELL" = true ]; then
    CELL_FLAG=(--lstm_persistent_cell)
else
    CELL_MODE="reset_each_repeat"
    CELL_FLAG=(--no-lstm_persistent_cell)
fi

# Login-node self-submission only.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"

    ENV_FILE="$REPO_DIR/iridis/env.sh"
    if [ ! -f "$ENV_FILE" ]; then
        echo "ERROR: env.sh not found: $ENV_FILE" >&2
        exit 1
    fi

    source "$ENV_FILE"
    RUN_DIR="$(next_run_dir "$PACKAGE_DIR")"

    echo "Repository: $REPO_DIR"
    echo "Logs:       $RUN_DIR"

    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR" \
        "$0" "$@"
fi

# Compute-node execution. Use the path exported by the wrapper.
if [ -z "${REPO_DIR:-}" ]; then
    echo "ERROR: REPO_DIR was not exported by the submission wrapper." >&2
    echo "Submit with: bash iridis/owt2-llms/nat_ln_lstm_all_core_job.sh" >&2
    exit 1
fi

if [ ! -f "$REPO_DIR/iridis/env.sh" ]; then
    echo "ERROR: invalid REPO_DIR: $REPO_DIR" >&2
    exit 1
fi

source "$REPO_DIR/iridis/env.sh"
cd "$REPO_DIR"

# --- Scratch-based output dirs (off home quota) ---
EXPS_DIR="/scratch/ab3u21/exps"
mkdir -p "$EXPS_DIR" "$DATA_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"


# --- Environment ---
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"
EXPNAME="nat_real_ln_lstm_ut_all_core_pc_40k_ndim_${N_EMBD}_${CELL_MODE}_beg_${N_LAYER_BEGIN}_mid_${N_REPEAT}_end_${N_LAYER_END}"
for arg in "$@"; do
    case "$arg" in
        --exp_name|--exp_name=*|--model|--model=*|--dataset|--dataset=*|--results_base_folder|--results_base_folder=*)
            echo "ERROR: $arg is fixed by this job script." >&2
            exit 2
            ;;
    esac
done
export WANDB_MODE=disabled
unset WANDB_RESUME WANDB_RUN_ID WANDB_NAME

cd "$REPO_DIR"

# --- GPU diagnostics ---
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Verify dataset ---
if [ ! -f "$DATA_DIR/openwebtext2/train.bin" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR/openwebtext2/train.bin"
    echo "  Run: bash iridis/download-dataset/job.sh"
    echo "  Then: bash iridis/extract-tokenize-dataset/job.sh"
    exit 1
fi

# --- Build training args ---
# NOTE: cotformer_full_depth does NOT use depth_embedding, depth_random_method,
# or min_repeat. The model always runs all n_repeat iterations unconditionally.
TRAIN_ARGS=(
    --config_format base
    --model ln_lstm_ut_all_core
    --n_embd "$N_EMBD"
    --n_head 6
    --n_layer "$N_LAYER"
    --n_repeat "$N_REPEAT"
    --batch_size "$BATCH_SIZE"
    --sequence_length 256
    --acc_steps "$ACC_STEPS"
    --dropout 0.0
    --iterations "$ITERATIONS"
    --dataset owt2
    --data_dir "$DATA_DIR"
    --lr 1e-3
    --weight_decay 0.1
    --warmup_percent 0.2
    --eval_freq "$CKPT_FREQ"
    --seed 0
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --save_checkpoint_freq "$CKPT_FREQ"
    --results_base_folder "$EXPS_DIR"
    --use_pretrained auto
    --lstm_forget_gate learned
    --lstm_control_input previous_and_proposed
    "${CELL_FLAG[@]}"
    "$@"
)

if (( $# > 0 )); then
    digest=$(printf '%s\0' "${TRAIN_ARGS[@]}" | sha256sum)
    digest=${digest%% *}
    EXPNAME="nat_ln_lstm_ut_all_core_cli_${digest:0:16}"
fi
TRAIN_ARGS+=(--exp_name "$EXPNAME")

# --- Launch ---
if [ "$N_GPUS" -gt 1 ]; then
    export OMP_NUM_THREADS=1
    RDZV_HOST=$(hostname)
    RDZV_PORT=$(expr 10000 + $(echo -n "$SLURM_JOB_ID" | tail -c 4))

    echo "Launching DDP training: ${N_GPUS} GPUs, RDZV ${RDZV_HOST}:${RDZV_PORT}"
    torchrun \
        --nproc_per_node="$N_GPUS" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT}" \
        main.py "${TRAIN_ARGS[@]}" \
        --distributed_backend nccl
else
    echo "Launching single-GPU training"
    python main.py "${TRAIN_ARGS[@]}"
fi

EXIT_CODE=$?

echo "========================================="
echo " Training finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo ""
echo " Checkpoints: $EXPS_DIR/owt2/ln_lstm_ut_all_core/$EXPNAME"
echo ""
echo " If training incomplete, resubmit:"
echo "   bash iridis/owt2-llms/nat_ln_lstm_all_core_job.sh with the same arguments"
echo "========================================="

exit $EXIT_CODE
