#!/usr/bin/env bash
# Self-submitting DCA delayed-recall cache-intervention evaluation.
#
# Default run:
#   bash iridis/dca-30/job_dca_intervene_eval.sh
#
# Override the scientific inputs with environment variables, for example:
#   SOURCE_RUN_ID=run_71__dca_cotf_h12_recall1_allquery_ckpt \
#   EXPECTED_STEP=5000 \
#   HORIZONS="6 12" \
#   INTERVENTION_DEPTHS="6 12" \
#   MAX_BATCHES=32 \
#   bash iridis/dca-30/job_dca_intervene_eval.sh
#
# An empty INTERVENTION_DEPTHS value evaluates every valid depth 1..H for
# each horizon H. The three supported interventions run by default.

set -euo pipefail

# ------------------------- Scientific configuration -------------------------

SOURCE_RUN_ID="${SOURCE_RUN_ID:-run_70__dca_cotf_h12_recall1_allquery_ckpt}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_delayed_recall.pt}"
EXPECTED_STEP="${EXPECTED_STEP:-5000}"
HORIZONS="${HORIZONS:-6 12}"
INTERVENTION_DEPTHS="${INTERVENTION_DEPTHS:-}"
NUM_RECALL_REPEATS="${NUM_RECALL_REPEATS:-}"
INTERVENTIONS="${INTERVENTIONS:-target-value-corruption target-repeat-only target-repeat-masked}"
VALUE_PERMUTATION_OFFSET="${VALUE_PERMUTATION_OFFSET:-1}"
DIAGNOSTIC_SPLIT="${DIAGNOSTIC_SPLIT:-validation}"
DIAGNOSTIC_DATA_MODE="${DIAGNOSTIC_DATA_MODE:-manifest}"
MAX_BATCHES="${MAX_BATCHES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
BACKEND_CHECK_POLICY="${BACKEND_CHECK_POLICY:-once-per-horizon}"

NUM_EXAMPLES="${NUM_EXAMPLES:-0}"
COLLECT_ATTENTION="${COLLECT_ATTENTION:-1}"

# --------------------------- Slurm configuration ----------------------------

PARTITION="${PARTITION:-ecsstudents_l4}"
ACCOUNT="${ACCOUNT:-ecsstudents}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

if (( $# != 0 )); then
    echo "This launcher uses environment-variable overrides, not arguments." >&2
    echo "See the examples at the top of $0." >&2
    exit 2
fi

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! [[ "$EXPECTED_STEP" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "EXPECTED_STEP must be a non-negative integer." >&2
    exit 2
fi
for value in "$VALUE_PERMUTATION_OFFSET" "$MAX_BATCHES"; do
    if ! is_positive_integer "$value"; then
        echo "Permutation offset and max batches must be positive integers." >&2
        exit 2
    fi
done
if [[ -n "$EVAL_BATCH_SIZE" ]] && ! is_positive_integer "$EVAL_BATCH_SIZE"; then
      echo "EVAL_BATCH_SIZE must be empty or a positive integer." >&2
      exit 2
fi
case "$BACKEND_CHECK_POLICY" in
    all|once-per-horizon|none) ;;
    *)
        echo "BACKEND_CHECK_POLICY must be all, once-per-horizon, or none." >&2
        exit 2
        ;;
esac

if [[ -n "$NUM_RECALL_REPEATS" ]] && ! is_positive_integer "$NUM_RECALL_REPEATS"; then
    echo "NUM_RECALL_REPEATS must be empty or a positive integer." >&2
    exit 2
fi
if ! [[ "$NUM_EXAMPLES" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "NUM_EXAMPLES must be a non-negative integer." >&2
    exit 2
fi
if [[ "$COLLECT_ATTENTION" != 0 && "$COLLECT_ATTENTION" != 1 ]]; then
    echo "COLLECT_ATTENTION must be 0 or 1." >&2
    exit 2
fi
case "$DIAGNOSTIC_SPLIT" in
    validation|test) ;;
    *)
        echo "DIAGNOSTIC_SPLIT must be validation or test." >&2
        exit 2
        ;;
esac
case "$DIAGNOSTIC_DATA_MODE" in
    manifest|indexed|materialized) ;;
    *)
        echo "DIAGNOSTIC_DATA_MODE must be manifest, indexed, or materialized." >&2
        exit 2
        ;;
esac

read -r -a HORIZON_ARGS <<< "$HORIZONS"
read -r -a DEPTH_ARGS <<< "$INTERVENTION_DEPTHS"
read -r -a INTERVENTION_ARGS <<< "$INTERVENTIONS"
if (( ${#HORIZON_ARGS[@]} == 0 )); then
    echo "HORIZONS must contain at least one positive integer." >&2
    exit 2
fi
for value in "${HORIZON_ARGS[@]}" "${DEPTH_ARGS[@]}"; do
    if ! is_positive_integer "$value"; then
        echo "Horizons and intervention depths must be positive integers." >&2
        exit 2
    fi
done

# Slurm executes a spool copy of this script. Resolve the repository before
# submission and export it explicitly for the worker process.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    DCA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export DCA_REPO_DIR="$(cd "$DCA_SCRIPT_DIR/../.." && pwd)"
    source "$DCA_REPO_DIR/iridis/env.sh"

    DCA_SOURCE_RUN_DIR="$DCA_REPO_DIR/iridis/dca-30/runs/$SOURCE_RUN_ID"
    if [[ ! -f "$DCA_SOURCE_RUN_DIR/run_manifest.json" ]]; then
        echo "Source run manifest does not exist: $DCA_SOURCE_RUN_DIR/run_manifest.json" >&2
        exit 2
    fi

    export DCA_RUN_LOG_DIR="$DCA_REPO_DIR/iridis/dca-30/intervene-eval/$SOURCE_RUN_ID"
    mkdir -p "$DCA_RUN_LOG_DIR"

    export SOURCE_RUN_ID CHECKPOINT_NAME EXPECTED_STEP HORIZONS
    export INTERVENTION_DEPTHS NUM_RECALL_REPEATS INTERVENTIONS
    export VALUE_PERMUTATION_OFFSET DIAGNOSTIC_SPLIT DIAGNOSTIC_DATA_MODE
    export MAX_BATCHES NUM_EXAMPLES COLLECT_ATTENTION
    export PARTITION ACCOUNT CPUS_PER_TASK MEMORY TIME_LIMIT SBATCH_BIN
    export MAX_BATCHES NUM_EXAMPLES COLLECT_ATTENTION EVAL_BATCH_SIZE
    export BACKEND_CHECK_POLICY


    echo "Submitting DCA recall-cache interventions on one L4."
    echo "  Source run:          $SOURCE_RUN_ID"
    echo "  Checkpoint:          $CHECKPOINT_NAME (step $EXPECTED_STEP)"
    echo "  Horizons:            ${HORIZON_ARGS[*]}"
    if (( ${#DEPTH_ARGS[@]} == 0 )); then
        echo "  Intervention depths: every valid depth per horizon"
    else
        echo "  Intervention depths: ${DEPTH_ARGS[*]}"
    fi
    echo "  Interventions:       ${INTERVENTION_ARGS[*]}"
    if [[ -z "$NUM_RECALL_REPEATS" ]]; then
        echo "  Recall passes:        source manifest default"
    else
        echo "  Recall passes:        $NUM_RECALL_REPEATS"
    fi
    echo "  Validation batches:  $MAX_BATCHES"
    echo "  Slurm logs:           $DCA_RUN_LOG_DIR"

    exec "$SBATCH_BIN" \
        --job-name=dca_intervene_eval \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$CPUS_PER_TASK" \
        --gres=gpu:1 \
        --mem="$MEMORY" \
        --time="$TIME_LIMIT" \
        --chdir="$DCA_REPO_DIR" \
        --output="$DCA_RUN_LOG_DIR/slurm_%j.out" \
        --error="$DCA_RUN_LOG_DIR/slurm_%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="$NOTIFY_EMAIL" \
        --export=ALL \
        "$DCA_SCRIPT_DIR/job_dca_intervene_eval.sh"
fi

# ----------------------------- Worker process -------------------------------

DCA_REPO_DIR="${DCA_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$DCA_REPO_DIR" || ! -f "$DCA_REPO_DIR/iridis/env.sh" ]]; then
    echo "Cannot locate the repository; launch this file with bash." >&2
    exit 2
fi

source "$DCA_REPO_DIR/iridis/env.sh"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$DCA_REPO_DIR:${PYTHONPATH:-}"
cd "$DCA_REPO_DIR"

RUN_MANIFEST="$DCA_REPO_DIR/iridis/dca-30/runs/$SOURCE_RUN_ID/run_manifest.json"
if [[ ! -f "$RUN_MANIFEST" ]]; then
    echo "Run manifest does not exist: $RUN_MANIFEST" >&2
    exit 2
fi

DCA_OUTPUT_ROOT="${DCA_OUTPUT_ROOT:-$RESULTS_DIR/dca-intervene-eval/artifacts}"
DCA_OUTPUT_DIR="$DCA_OUTPUT_ROOT/$SOURCE_RUN_ID/job_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$DCA_OUTPUT_DIR")"
if [[ -e "$DCA_OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output: $DCA_OUTPUT_DIR" >&2
    exit 2
fi

EVALUATOR_ARGS=(
    --run-manifest "$RUN_MANIFEST"
    --expected-run-id "$SOURCE_RUN_ID"
    --checkpoint-name "$CHECKPOINT_NAME"
    --expected-step "$EXPECTED_STEP"
    --device cuda:0
    --horizons "${HORIZON_ARGS[@]}"
    --interventions "${INTERVENTION_ARGS[@]}"
    --value-permutation-offset "$VALUE_PERMUTATION_OFFSET"
    --diagnostic-split "$DIAGNOSTIC_SPLIT"
    --diagnostic-data-mode "$DIAGNOSTIC_DATA_MODE"
    --max-batches "$MAX_BATCHES"
    --num-examples "$NUM_EXAMPLES"
    --output-dir "$DCA_OUTPUT_DIR"
    --backend-check-policy "$BACKEND_CHECK_POLICY"

)
if [[ -n "$NUM_RECALL_REPEATS" ]]; then
    EVALUATOR_ARGS+=(--num-recall-repeats "$NUM_RECALL_REPEATS")
fi
if (( ${#DEPTH_ARGS[@]} > 0 )); then
    EVALUATOR_ARGS+=(--intervention-depths "${DEPTH_ARGS[@]}")
fi
if [[ "$COLLECT_ATTENTION" == 0 ]]; then
    EVALUATOR_ARGS+=(--no-collect-attention)
fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then
    EVALUATOR_ARGS+=(--eval-batch-size "$EVAL_BATCH_SIZE")
fi


echo "========================================================================"
echo " DCA model-level recall-cache intervention evaluation"
echo "========================================================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Host:                $(hostname)"
echo "Source run:          $SOURCE_RUN_ID"
echo "Checkpoint:          $CHECKPOINT_NAME"
echo "Expected step:       $EXPECTED_STEP"
echo "Horizons:            ${HORIZON_ARGS[*]}"
if (( ${#DEPTH_ARGS[@]} == 0 )); then
    echo "Intervention depths: every valid depth per horizon"
else
    echo "Intervention depths: ${DEPTH_ARGS[*]}"
fi
if [[ -z "$NUM_RECALL_REPEATS" ]]; then
    echo "Recall passes:       source manifest default"
else
    echo "Recall passes:       $NUM_RECALL_REPEATS"
fi
if [[ -z "$EVAL_BATCH_SIZE" ]]; then
    echo "Evaluation batch size: source manifest default"
else
    echo "Evaluation batch size: $EVAL_BATCH_SIZE"
fi
echo "Backend check policy: $BACKEND_CHECK_POLICY"


echo "Interventions:       ${INTERVENTION_ARGS[*]}"
echo "Split/data mode:     $DIAGNOSTIC_SPLIT / $DIAGNOSTIC_DATA_MODE"
echo "Maximum batches:     $MAX_BATCHES"
echo "Attention reporting: $COLLECT_ATTENTION"
echo "Output directory:    $DCA_OUTPUT_DIR"
echo "Started:             $(date --iso-8601=seconds)"
echo "------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "========================================================================"
echo

# The evaluator itself prints two compact result tables to this Slurm output:
# backend equivalence first, then intervention deltas from the manual baseline.
python3 -m cellular_automaton.dca_intervene_eval "${EVALUATOR_ARGS[@]}"

echo
echo "========================================================================"
echo " Evaluation completed"
echo " Main machine-readable and indented report: $DCA_OUTPUT_DIR/summary.json"
echo " Finished: $(date --iso-8601=seconds)"
echo "========================================================================"
