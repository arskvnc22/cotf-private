#!/bin/bash
#SBATCH --job-name=ca30_but_eval
#SBATCH --partition=ecsstudents_l4
#SBATCH --account=ecsstudents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

################################################################################
# Evaluate selected checkpoints from an existing Rule 30 BUT or LSTM-UT run.
#
#   bash iridis/ca-rule30/job_but_eval.sh \
#     --source-run-id run_121__ca_lstm_r12_zero \
#     --checkpoint-type best_extrapolation_strict \
#     --max-repeats 100
#########################################################################################

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash iridis/ca-rule30/job_but_eval.sh --source-run-id RUN_ID [options]

Options:
  --source-run-id ID       Source directory basename under ca-rule30/runs.
  --source-run-dir PATH    Explicit source run directory.
  --output-root PATH       Scratch root for the full evaluation summary.
  --checkpoint-type TYPE   Evaluate one checkpoint: best_id,
                           best_extrapolation_strict,
                           best_extrapolation_unconstrained, or
                           best_average_extrap.
  --max-repeats N          Evaluate repeat depths 1 through N against
                           Rule 30 horizons 0 through N.
  -h, --help               Show this help.

Without --checkpoint-type, the evaluator runs the original three checkpoints:
best_id, best_extrapolation_strict, and best_extrapolation_unconstrained.
PARTITION, ACCOUNT, N_GPUS, CPUS_PER_TASK, MEMORY, TIME_LIMIT, and SBATCH_BIN
override the Slurm defaults.
EOF
}
option_value() {
    local option="$1"
    local value="${2-}"
    if [ -z "$value" ] || [[ "$value" == --* ]]; then
        echo "$option requires a value." >&2
        exit 2
    fi
}

PARTITION="${PARTITION:-ecsstudents_l4}"
ACCOUNT="${ACCOUNT:-ecsstudents}"
N_GPUS="${N_GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ab3u21/exps/cellular-automaton/standalone-eval}"
MAX_REPEATS="${MAX_REPEATS:-}"
CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-}"

CLI_ARGS=("$@")
ARG_INDEX=0
while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ]; do
    OPTION="${CLI_ARGS[$ARG_INDEX]}"
    case "$OPTION" in
        -h|--help)
            usage
            exit 0
            ;;
        --source-run-id)
            option_value "$OPTION" "${CLI_ARGS[$((ARG_INDEX + 1))]-}"
            SOURCE_RUN_ID="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ;;
        --source-run-dir)
            option_value "$OPTION" "${CLI_ARGS[$((ARG_INDEX + 1))]-}"
            SOURCE_RUN_DIR="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ;;
        --output-root)
            option_value "$OPTION" "${CLI_ARGS[$((ARG_INDEX + 1))]-}"
            OUTPUT_ROOT="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ;;
        --max-repeats)
            option_value "$OPTION" "${CLI_ARGS[$((ARG_INDEX + 1))]-}"
            MAX_REPEATS="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ;;
        --checkpoint-type)
            option_value "$OPTION" "${CLI_ARGS[$((ARG_INDEX + 1))]-}"
            CHECKPOINT_TYPE="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ;;
        *)
            echo "Unknown option: $OPTION" >&2
            usage >&2
            exit 2
            ;;
    esac
    ARG_INDEX=$((ARG_INDEX + 2))
done

if [ -z "$SOURCE_RUN_ID" ]; then
    echo "--source-run-id is required." >&2
    exit 2
fi
if ! [[ "$SOURCE_RUN_ID" =~ ^run_[0-9]+__(but_full_depth|ca_lstm_r(8|12)_zero)$ ]]; then
    echo "SOURCE_RUN_ID must identify a BUT or r8/r12 LSTM-UT run." >&2
    exit 2
fi

next_eval_run_dir() {
    local runs_root="$1"
    local last_number
    local next_number
    local candidate
    mkdir -p "$runs_root"
    last_number=$(
        find "$runs_root" -mindepth 1 -maxdepth 1 -type d \
            -name 'run_[0-9]*__*' -printf '%f\n' \
            | sed -n 's/^run_\([0-9][0-9]*\)__.*/\1/p' \
            | sort -n \
            | tail -1
    )
    next_number=$(( ${last_number:--1} + 1 ))
    while true; do
        candidate="$runs_root/run_${next_number}__${SOURCE_RUN_ID#*__}_eval"
        if mkdir "$candidate" 2>/dev/null; then
            printf '%s\n' "$candidate"
            return
        fi
        next_number=$((next_number + 1))
    done
}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO_DIR="$(cd "$PACKAGE_DIR/../.." && pwd)"
    source "$REPO_DIR/iridis/env.sh"
    SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$PACKAGE_DIR/runs/$SOURCE_RUN_ID}"
    RUN_DIR=$(next_eval_run_dir "$PACKAGE_DIR/runs")

    echo "=== Rule 30 standalone checkpoint evaluation submission ==="
    echo "  Source run:     $SOURCE_RUN_ID"
    echo "  Source dir:     $SOURCE_RUN_DIR"
    echo "  Output run:     $RUN_DIR"
    echo "  Artifact root:  $OUTPUT_ROOT"

    exec "$SBATCH_BIN" \
        --job-name=ca30_but_eval \
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
        --mail-type=END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL,REPO_DIR="$REPO_DIR",RUN_DIR="$RUN_DIR",SOURCE_RUN_ID="$SOURCE_RUN_ID",SOURCE_RUN_DIR="$SOURCE_RUN_DIR",OUTPUT_ROOT="$OUTPUT_ROOT",MAX_REPEATS="$MAX_REPEATS",CHECKPOINT_TYPE="$CHECKPOINT_TYPE" \
        "$PACKAGE_DIR/job_but_eval.sh" "${CLI_ARGS[@]}"
fi

source "$REPO_DIR/iridis/env.sh"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

ARTIFACT_DIR="$OUTPUT_ROOT/$SOURCE_RUN_ID/job_${SLURM_JOB_ID}"
echo "=== Rule 30 BUT standalone evaluation ==="
echo "  Source run:     $SOURCE_RUN_ID"
echo "  Source dir:     $SOURCE_RUN_DIR"
echo "  Output run:     $RUN_DIR"
echo "  Artifact dir:   $ARTIFACT_DIR"
echo "  Checkpoints:    best_id, best_extrapolation_strict, best_extrapolation_unconstrained"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
EVAL_OPTIONS=()
if [[ -n "$MAX_REPEATS" ]]; then
    EVAL_OPTIONS+=(--max-repeats "$MAX_REPEATS")
fi
if [[ -n "$CHECKPOINT_TYPE" ]]; then
    EVAL_OPTIONS+=(--checkpoint-type "$CHECKPOINT_TYPE")
fi
/usr/bin/time -v python -m cellular_automaton.ca_checkpoint_eval \
    --source-run-dir "$SOURCE_RUN_DIR" \
    --output-run-dir "$RUN_DIR" \
    --artifact-dir "$ARTIFACT_DIR" \
    --device cuda:0 \
    "${EVAL_OPTIONS[@]}"

echo "=== Standalone evaluation completed ==="
echo "  Analyzer run:  $RUN_DIR"
echo "  Full summary:  $ARTIFACT_DIR/summary.json"
