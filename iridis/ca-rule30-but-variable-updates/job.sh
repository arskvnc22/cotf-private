#!/bin/bash
#SBATCH --job-name=ca30_but_var
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
################################################################################
# Variable-update Rule 30 training with a bidirectional Block Universal
# Transformer. Each optimizer step uses one homogeneous CA_STEPS:MODEL_REPEATS
# pair. The pairs alternate in a deterministic round-robin schedule.
#
# Usage:
#   cd ~/CoTFormer && bash iridis/ca-rule30-but-variable-updates/job.sh
#
# Configuration can be overridden with environment variables, for example:
#   CA_TRAIN_PAIRS="1:1 2:2 3:3" BEST_PAIR=3:3 ITERATIONS=5000 \
#     bash iridis/ca-rule30-but-variable-updates/job.sh
################################################################################

# ========================= CONFIGURATION ====================================

N_GPUS=1
MODEL="but_full_depth"
ATTENTION_MODE="bidirectional"
POSITIONAL_ENCODER="rotary"
N_LAYER="${N_LAYER:-1}"
# This is the fallback model depth. --ca_train_pairs and --ca_final_eval_pairs
# override it for their respective train/evaluation forwards.
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
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-0.9}"
SEED="${SEED:-0}"
CA_TRAIN_PAIRS="${CA_TRAIN_PAIRS:-1:1 2:2 3:3 }"
BEST_PAIR="${BEST_PAIR:-3:3}"
CA_EXTRAPOLATION_VAL_PAIRS="${CA_EXTRAPOLATION_VAL_PAIRS:-4:4 5:5}"
EXTRAPOLATION_BEST_PAIR="${EXTRAPOLATION_BEST_PAIR:-5:5}"
STRICT_MIN_ID_CELL_ACCURACY="${STRICT_MIN_ID_CELL_ACCURACY:-0.99}"
STRICT_MIN_ID_EXACT_ACCURACY="${STRICT_MIN_ID_EXACT_ACCURACY:-0.95}"
CA_FINAL_EVAL_PAIRS="${CA_FINAL_EVAL_PAIRS:- 8:8}"
DIAGNOSTIC_MAX_REPEATS="${DIAGNOSTIC_MAX_REPEATS:-8}"
DIAGNOSTIC_EXAMPLES="${DIAGNOSTIC_EXAMPLES:-1}"

# Keep automatically generated names accurate when these settings are supplied
# as trailing CLI overrides instead of environment variables.
CLI_ARGS=("$@")
ARG_INDEX=0
while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ]; do
    case "${CLI_ARGS[$ARG_INDEX]}" in
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
        --seed)
            SEED="${CLI_ARGS[$((ARG_INDEX + 1))]}"
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
        --ca_train_pairs)
            ARG_INDEX=$((ARG_INDEX + 1))
            PAIR_OVERRIDE=()
            while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ] \
                && [[ "${CLI_ARGS[$ARG_INDEX]}" != --* ]]; do
                PAIR_OVERRIDE+=("${CLI_ARGS[$ARG_INDEX]}")
                ARG_INDEX=$((ARG_INDEX + 1))
            done
            if [ "${#PAIR_OVERRIDE[@]}" -gt 0 ]; then
                CA_TRAIN_PAIRS="${PAIR_OVERRIDE[*]}"
            fi
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

# Convert pair syntax such as "1:1 2:2" into a path-friendly "1x1_2x2" tag.
TRAIN_PAIR_TAG="${CA_TRAIN_PAIRS//:/x}"
TRAIN_PAIR_TAG="${TRAIN_PAIR_TAG// /_}"
EXP_NAME="${EXP_NAME:-R2_ca_30_but_embd_${N_EMBD}_beg_${N_LAYER_BEGIN}_>layrep_${N_LAYER}_X_${N_REPEAT}_>end${N_LAYER_END}_variable_updates_pairs_${TRAIN_PAIR_TAG}_lr_${LR}_wd_${WEIGHT_DECAY}_gc_${GRAD_CLIP}_seed_${SEED}}"

read -r -a TRAIN_PAIR_ARGS <<< "$CA_TRAIN_PAIRS"
read -r -a EXTRAPOLATION_PAIR_ARGS <<< "$CA_EXTRAPOLATION_VAL_PAIRS"
read -r -a FINAL_PAIR_ARGS <<< "$CA_FINAL_EVAL_PAIRS"

# ========================= END CONFIGURATION ================================

if [ -z "$SLURM_JOB_ID" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"

    RUN_DIR=$(next_run_dir "$PACKAGE_DIR")
    echo "=== Rule 30 BUT variable-update training ==="
    echo "  Partition:       ecsstudents_l4"
    echo "  GPUs:            $N_GPUS"
    echo "  Attention:       $ATTENTION_MODE"
    echo "  Positional enc:  $POSITIONAL_ENCODER"
    echo "  Train pairs:     $CA_TRAIN_PAIRS"
    echo "  Best pair:       $BEST_PAIR"
    echo "  Learning rate:   $LR"
    echo "  Weight decay:    $WEIGHT_DECAY"
    echo "  Gradient clip:   $GRAD_CLIP"
    echo "  Extrap val:      $CA_EXTRAPOLATION_VAL_PAIRS"
    echo "  Strict ID gate:  cell>=$STRICT_MIN_ID_CELL_ACCURACY exact>=$STRICT_MIN_ID_EXACT_ACCURACY"
    echo "  Test pairs:      $CA_FINAL_EVAL_PAIRS"
    echo "  Training steps:  $ITERATIONS"
    echo "  Experiment:      $EXP_NAME"
    echo "  Logs:            $RUN_DIR/"
    echo ""
    exec sbatch \
        --output="$RUN_DIR/slurm_%j.out" \
        --error="$RUN_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR" \
        "$0" "$@"
fi

set -eo pipefail
export PYTHONUNBUFFERED=1

if [ -z "$REPO_DIR" ]; then
    REPO_DIR="$HOME/CoTFormer"
    echo "WARNING: REPO_DIR not set -- falling back to $REPO_DIR"
    echo "Tip: use 'bash job.sh' instead of 'sbatch job.sh'"
fi

source "$REPO_DIR/iridis/env.sh"
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(job_output_dir)
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
echo " Rule 30 BUT Variable-Update Training"
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
echo " Best pair:        $BEST_PAIR"
echo " Extrap val pairs: $CA_EXTRAPOLATION_VAL_PAIRS"
echo " Strict ID gate:   cell>=$STRICT_MIN_ID_CELL_ACCURACY exact>=$STRICT_MIN_ID_EXACT_ACCURACY"
echo " Final eval pairs: $CA_FINAL_EVAL_PAIRS"
echo " Diagnostic depth: $DIAGNOSTIC_MAX_REPEATS"
echo " Learning rate:    $LR"
echo " Weight decay:     $WEIGHT_DECAY"
echo " Iterations:       $ITERATIONS"
echo " Evaluation:       every $EVAL_FREQ steps, at most $EVAL_MAX_BATCHES batches"
echo " Data workers:     $NUM_WORKERS"
echo " Gradient clip:    $GRAD_CLIP"
echo " Seed:             $SEED"
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
    --dropout 0.0
    --iterations "$ITERATIONS"
    --scheduler none
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --grad_clip "$GRAD_CLIP"
    --eval_freq "$EVAL_FREQ"
    --seed "$SEED"
    --results_base_folder "$EXPS_DIR"
    --exp_name "$EXP_NAME"
    --use_pretrained None
    --ca_data_mode materialized
    --ca_train_num_cells "$NUM_CELLS"
    --ca_eval_num_cells "$NUM_CELLS"
    --ca_steps 1
    --ca_train_pairs
    "${TRAIN_PAIR_ARGS[@]}"
    --ca_best_pair "$BEST_PAIR"
    --ca_extrapolation_val_pairs
    "${EXTRAPOLATION_PAIR_ARGS[@]}"
    --ca_extrapolation_best_pair "$EXTRAPOLATION_BEST_PAIR"
    --ca_extrapolation_best_metric cell_accuracy
    --ca_extrapolation_min_id_cell_accuracy "$STRICT_MIN_ID_CELL_ACCURACY"
    --ca_extrapolation_min_id_exact_sequence_accuracy "$STRICT_MIN_ID_EXACT_ACCURACY"
    --ca_train_samples "$TRAIN_SAMPLES"
    --ca_val_samples "$VAL_SAMPLES"
    --ca_test_samples "$TEST_SAMPLES"
    --ca_bernoulli_p 0.5
    --ca_eval_batch_size "$BATCH_SIZE"
    --ca_eval_max_batches "$EVAL_MAX_BATCHES"
    --ca_final_eval_pairs
    "${FINAL_PAIR_ARGS[@]}"
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
echo " Rule 30 variable-update run finished: $(date)"
echo " Exit code: $EXIT_CODE"
echo " Results:   $EXPS_DIR/rule30/$MODEL/$EXP_NAME"
echo " Logs:      $RUN_DIR"
echo "========================================="

exit "$EXIT_CODE"
