#!/bin/bash
#SBATCH --job-name=ca30_rule30
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
################################################################################
# Shared Rule 30 variable-update launcher.
#
# Prefer a model preset:
#   bash iridis/ca-rule30/job_but.sh
#   bash iridis/ca-rule30/job_cotformer.sh
#
# Direct use and trailing CLI overrides are also supported:
#   bash iridis/ca-rule30/job.sh --model but_full_depth --n_embd 128
################################################################################

# ========================= CONFIGURATION ====================================

PARTITION="${PARTITION:-ecsstudents_l4}"
ACCOUNT="${ACCOUNT:-ecsstudents}"
N_GPUS="${N_GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

MODEL="${MODEL:-but_full_depth}"
ATTENTION_MODE="${ATTENTION_MODE:-bidirectional}"
POSITIONAL_ENCODER="${POSITIONAL_ENCODER:-rotary}"
N_LAYER="${N_LAYER:-1}"
# Variable CA pairs override this fallback repeat count during train/evaluation.
N_REPEAT="${N_REPEAT:-1}"
N_LAYER_BEGIN="${N_LAYER_BEGIN:-0}"
N_LAYER_END="${N_LAYER_END:-0}"
N_EMBD="${N_EMBD:-64}"
N_HEAD="${N_HEAD:-4}"
NUM_CELLS="${NUM_CELLS:-64}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-100000}"
VAL_SAMPLES="${VAL_SAMPLES:-4096}"
TEST_SAMPLES="${TEST_SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ACC_STEPS="${ACC_STEPS:-1}"
ITERATIONS="${ITERATIONS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-250}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"

OPT="${OPT:-adamw}"
SCHEDULER="${SCHEDULER:-none}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-0.9}"
DROPOUT="${DROPOUT:-0.0}"
# Model initialization and training-row generation use independent defaults.
# Holding DATA_SEED fixed while sweeping SEED isolates initialization variance.
SEED="${SEED:-0}"
DATA_SEED="${DATA_SEED:-1}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1}"

CA_TRAIN_PAIRS="${CA_TRAIN_PAIRS:-1:1 2:2 3:3}"
# Empty values let ca_main.py derive the largest configured pair.
BEST_PAIR="${BEST_PAIR-}"
CA_EXTRAPOLATION_VAL_PAIRS="${CA_EXTRAPOLATION_VAL_PAIRS-4:4 5:5}"
EXTRAPOLATION_BEST_PAIR="${EXTRAPOLATION_BEST_PAIR-}"
STRICT_MIN_ID_CELL_ACCURACY="${STRICT_MIN_ID_CELL_ACCURACY:-0.99}"
STRICT_MIN_ID_EXACT_ACCURACY="${STRICT_MIN_ID_EXACT_ACCURACY:-0.95}"
CA_FINAL_EVAL_PAIRS="${CA_FINAL_EVAL_PAIRS-8:8}"
DIAGNOSTIC_MAX_REPEATS="${DIAGNOSTIC_MAX_REPEATS:-8}"
DIAGNOSTIC_EXAMPLES="${DIAGNOSTIC_EXAMPLES:-1}"

# Parse values that affect shell-side naming/reporting. The complete argument
# list is still forwarded to Python, where later occurrences take precedence.
CLI_ARGS=("$@")
ARG_INDEX=0
while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ]; do
    case "${CLI_ARGS[$ARG_INDEX]}" in
        --model)
            MODEL="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --attention_mode)
            ATTENTION_MODE="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --positional_encoder)
            POSITIONAL_ENCODER="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --n_layer)
            N_LAYER="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --n_repeat)
            N_REPEAT="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --n_layer_begin)
            N_LAYER_BEGIN="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --n_layer_end)
            N_LAYER_END="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --n_embd)
            N_EMBD="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --n_head)
            N_HEAD="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --batch_size)
            BATCH_SIZE="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --acc_steps)
            ACC_STEPS="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --iterations)
            ITERATIONS="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --eval_freq)
            EVAL_FREQ="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --opt)
            OPT="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --scheduler)
            SCHEDULER="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --lr)
            LR="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --weight_decay)
            WEIGHT_DECAY="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --grad_clip)
            GRAD_CLIP="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --dropout)
            DROPOUT="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --seed)
            SEED="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --data_seed)
            DATA_SEED="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --ca_shuffle_seed)
            SHUFFLE_SEED="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --ca_extrapolation_min_id_cell_accuracy)
            STRICT_MIN_ID_CELL_ACCURACY="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --ca_extrapolation_min_id_exact_sequence_accuracy)
            STRICT_MIN_ID_EXACT_ACCURACY="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --ca_best_pair)
            BEST_PAIR="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --ca_extrapolation_best_pair)
            EXTRAPOLATION_BEST_PAIR="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --ca_train_pairs|--ca_extrapolation_val_pairs|--ca_final_eval_pairs)
            LIST_OPTION="${CLI_ARGS[$ARG_INDEX]}"
            ARG_INDEX=$((ARG_INDEX + 1))
            PAIR_OVERRIDE=()
            while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ] \
                && [[ "${CLI_ARGS[$ARG_INDEX]}" != --* ]]; do
                PAIR_OVERRIDE+=("${CLI_ARGS[$ARG_INDEX]}")
                ARG_INDEX=$((ARG_INDEX + 1))
            done
            case "$LIST_OPTION" in
                --ca_train_pairs)
                    CA_TRAIN_PAIRS="${PAIR_OVERRIDE[*]}"
                    ;;
                --ca_extrapolation_val_pairs)
                    CA_EXTRAPOLATION_VAL_PAIRS="${PAIR_OVERRIDE[*]}"
                    ;;
                --ca_final_eval_pairs)
                    CA_FINAL_EVAL_PAIRS="${PAIR_OVERRIDE[*]}"
                    ;;
            esac
            ;;
        --exp_name)
            EXP_NAME="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        *)
            ARG_INDEX=$((ARG_INDEX + 1))
            ;;
    esac
done

MODEL_RUN_NAME="${MODEL_RUN_NAME:-$MODEL}"
MODEL_TAG="${MODEL//[^a-zA-Z0-9_-]/_}"
TRAIN_PAIR_TAG="${CA_TRAIN_PAIRS//:/x}"
TRAIN_PAIR_TAG="${TRAIN_PAIR_TAG// /_}"
EXP_NAME="${EXP_NAME:-ca30_${MODEL_TAG}_embd_${N_EMBD}_beg_${N_LAYER_BEGIN}_mid_${N_LAYER}x${N_REPEAT}_end_${N_LAYER_END}_pairs_${TRAIN_PAIR_TAG}_opt_${OPT}_lr_${LR}_wd_${WEIGHT_DECAY}_gc_${GRAD_CLIP}_mseed_${SEED}_dseed_${DATA_SEED}}"

read -r -a TRAIN_PAIR_ARGS <<< "$CA_TRAIN_PAIRS"
read -r -a EXTRAPOLATION_PAIR_ARGS <<< "$CA_EXTRAPOLATION_VAL_PAIRS"
read -r -a FINAL_PAIR_ARGS <<< "$CA_FINAL_EVAL_PAIRS"

TRAIN_PAIR_CLI=(--ca_train_pairs "${TRAIN_PAIR_ARGS[@]}")
BEST_PAIR_CLI=()
if [ -n "$BEST_PAIR" ]; then
    BEST_PAIR_CLI=(--ca_best_pair "$BEST_PAIR")
fi
EXTRAPOLATION_PAIR_CLI=()
if [ "${#EXTRAPOLATION_PAIR_ARGS[@]}" -gt 0 ]; then
    EXTRAPOLATION_PAIR_CLI=(
        --ca_extrapolation_val_pairs
        "${EXTRAPOLATION_PAIR_ARGS[@]}"
    )
fi
EXTRAPOLATION_BEST_CLI=()
if [ -n "$EXTRAPOLATION_BEST_PAIR" ]; then
    EXTRAPOLATION_BEST_CLI=(
        --ca_extrapolation_best_pair
        "$EXTRAPOLATION_BEST_PAIR"
    )
fi
FINAL_PAIR_CLI=()
if [ "${#FINAL_PAIR_ARGS[@]}" -gt 0 ]; then
    FINAL_PAIR_CLI=(--ca_final_eval_pairs "${FINAL_PAIR_ARGS[@]}")
fi

# ========================= END CONFIGURATION ================================

next_ca_run_dir() {
    local runs_root="$1"
    local model_name="$2"
    local safe_model_name
    local last_number
    local next_number
    local candidate

    mkdir -p "$runs_root"
    safe_model_name=$(printf '%s' "$model_name" | tr -c '[:alnum:]_-' '_')
    [ -n "$safe_model_name" ] || safe_model_name="model"
    last_number=$(
        find "$runs_root" -mindepth 1 -maxdepth 1 -type d \
            -name 'run_[0-9]*__*' -printf '%f\n' \
            | sed -n 's/^run_\([0-9][0-9]*\)__.*/\1/p' \
            | sort -n \
            | tail -1
    )
    next_number=$(( ${last_number:--1} + 1 ))
    while true; do
        candidate="$runs_root/run_${next_number}__${safe_model_name}"
        if mkdir "$candidate" 2>/dev/null; then
            printf '%s\n' "$candidate"
            return
        fi
        next_number=$((next_number + 1))
    done
}

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUNS_DIR="${CA_RUNS_DIR:-$PACKAGE_DIR/runs}"
    RUN_DIR=$(next_ca_run_dir "$RUNS_DIR" "$MODEL_RUN_NAME")
    JOB_NAME="ca30_${MODEL_RUN_NAME//[^a-zA-Z0-9_-]/_}"

    echo "=== Rule 30 training submission ==="
    echo "  Model:            $MODEL"
    echo "  Run directory:    $RUN_DIR"
    echo "  Attention:        $ATTENTION_MODE"
    echo "  Positional enc:   $POSITIONAL_ENCODER"
    echo "  Train pairs:      $CA_TRAIN_PAIRS"
    echo "  Extrap val pairs: ${CA_EXTRAPOLATION_VAL_PAIRS:-none}"
    echo "  Final eval pairs: ${CA_FINAL_EVAL_PAIRS:-none}"
    echo "  Optimizer:        $OPT"
    echo "  LR / WD / clip:   $LR / $WEIGHT_DECAY / $GRAD_CLIP"
    echo "  Model seed:       $SEED"
    echo "  Data seed:        $DATA_SEED"
    echo "  Shuffle seed:     $SHUFFLE_SEED"
    echo "  Experiment:       $EXP_NAME"
    echo ""
    exec "$SBATCH_BIN" \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --gres="gpu:$N_GPUS" \
        --mem="$MEMORY" \
        --time="$TIME_LIMIT" \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$PACKAGE_DIR/job.sh" "$@"
fi

set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
    echo "Tip: launch with 'bash iridis/ca-rule30/job_*.sh'."
fi

source "$REPO_DIR/iridis/env.sh"
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(job_output_dir)
    echo "WARNING: RUN_DIR was not prepared by the shared launcher."
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/output.log") 2> >(tee -a "$RUN_DIR/error.log" >&2)

EXPS_DIR="/scratch/ab3u21/exps/cellular-automaton"
mkdir -p "$EXPS_DIR" "$HF_HOME" "$TIKTOKEN_CACHE_DIR" "$WANDB_DIR"

module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

echo "========================================="
echo " Rule 30 Variable-Update Training"
echo " User:             $USER"
echo " Node:             $(hostname)"
echo " CPUs:             $SLURM_CPUS_PER_TASK"
echo " GPUs:             $N_GPUS"
echo " Job ID:           $SLURM_JOB_ID"
echo " Model:            $MODEL"
echo " Attention:        $ATTENTION_MODE"
echo " Positional enc:   $POSITIONAL_ENCODER"
echo " Architecture:     ${N_LAYER}L, fallback repeats=$N_REPEAT"
echo " Width:            d=$N_EMBD h=$N_HEAD"
echo " CA cells:         $NUM_CELLS"
echo " Train/val/test:   $TRAIN_SAMPLES/$VAL_SAMPLES/$TEST_SAMPLES"
echo " Batch size:       $BATCH_SIZE x $ACC_STEPS"
echo " Train pairs:      $CA_TRAIN_PAIRS"
echo " Best pair:        ${BEST_PAIR:-derived by ca_main.py}"
echo " Extrap val pairs: ${CA_EXTRAPOLATION_VAL_PAIRS:-none}"
echo " Extrap best pair: ${EXTRAPOLATION_BEST_PAIR:-derived by ca_main.py}"
echo " Strict ID gate:   cell>=$STRICT_MIN_ID_CELL_ACCURACY exact>=$STRICT_MIN_ID_EXACT_ACCURACY"
echo " Final eval pairs: ${CA_FINAL_EVAL_PAIRS:-none}"
echo " Diagnostic depth: $DIAGNOSTIC_MAX_REPEATS"
echo " Optimizer:        $OPT"
echo " Scheduler:        $SCHEDULER"
echo " Learning rate:    $LR"
echo " Weight decay:     $WEIGHT_DECAY"
echo " Gradient clip:    $GRAD_CLIP"
echo " Dropout:          $DROPOUT"
echo " Iterations:       $ITERATIONS"
echo " Evaluation:       every $EVAL_FREQ steps, at most $EVAL_MAX_BATCHES batches"
echo " Data workers:     $NUM_WORKERS"
echo " Model seed:       $SEED"
echo " Data seed:        $DATA_SEED"
echo " Shuffle seed:     $SHUFFLE_SEED"
echo " Experiment:       $EXP_NAME"
echo " Results:          $EXPS_DIR"
echo " Run directory:    $RUN_DIR"
echo " Started:          $(date)"
echo "========================================="

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

TRAIN_ARGS=(
    --config_format base
    --model "$MODEL"
    --attention_mode "$ATTENTION_MODE"
    --positional_encoder "$POSITIONAL_ENCODER"
    --n_embd "$N_EMBD"
    --n_head "$N_HEAD"
    --n_layer "$N_LAYER"
    --n_repeat "$N_REPEAT"
    --n_layer_begin "$N_LAYER_BEGIN"
    --n_layer_end "$N_LAYER_END"
    --sequence_length "$NUM_CELLS"
    --batch_size "$BATCH_SIZE"
    --acc_steps "$ACC_STEPS"
    --dropout "$DROPOUT"
    --iterations "$ITERATIONS"
    --opt "$OPT"
    --scheduler "$SCHEDULER"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --grad_clip "$GRAD_CLIP"
    --eval_freq "$EVAL_FREQ"
    --seed "$SEED"
    --data_seed "$DATA_SEED"
    --ca_shuffle_seed "$SHUFFLE_SEED"
    --results_base_folder "$EXPS_DIR"
    --exp_name "$EXP_NAME"
    --ca_run_dir "$RUN_DIR"
    --use_pretrained None
    --ca_data_mode materialized
    --ca_train_num_cells "$NUM_CELLS"
    --ca_eval_num_cells "$NUM_CELLS"
    --ca_steps 1
    "${TRAIN_PAIR_CLI[@]}"
    "${BEST_PAIR_CLI[@]}"
    "${EXTRAPOLATION_PAIR_CLI[@]}"
    "${EXTRAPOLATION_BEST_CLI[@]}"
    --ca_extrapolation_best_metric cell_accuracy
    --ca_extrapolation_min_id_cell_accuracy "$STRICT_MIN_ID_CELL_ACCURACY"
    --ca_extrapolation_min_id_exact_sequence_accuracy "$STRICT_MIN_ID_EXACT_ACCURACY"
    --ca_train_samples "$TRAIN_SAMPLES"
    --ca_val_samples "$VAL_SAMPLES"
    --ca_test_samples "$TEST_SAMPLES"
    --ca_bernoulli_p 0.5
    --ca_eval_batch_size "$BATCH_SIZE"
    --ca_eval_max_batches "$EVAL_MAX_BATCHES"
    "${FINAL_PAIR_CLI[@]}"
    --ca_final_eval_max_batches "$EVAL_MAX_BATCHES"
    --ca_repeat_diagnostic_max_repeats "$DIAGNOSTIC_MAX_REPEATS"
    --ca_repeat_diagnostic_examples "$DIAGNOSTIC_EXAMPLES"
    --ca_repeat_diagnostic_max_batches "$EVAL_MAX_BATCHES"
    --ca_num_workers "$NUM_WORKERS"
    --ca_best_metric exact_sequence_accuracy
    --ca_save_every "$ITERATIONS"
    --ca_log_every 25
    "$@"
)

if /usr/bin/time -v python cellular_automaton/ca_main.py "${TRAIN_ARGS[@]}"; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

echo "========================================="
echo " Rule 30 run finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo " Results:   $EXPS_DIR/rule30/$MODEL/$EXP_NAME"
echo " Logs:      $RUN_DIR"
echo "========================================="

exit "$EXIT_CODE"
