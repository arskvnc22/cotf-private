#!/bin/bash
################################################################################
# Delayed-recall Rule 30 preset for DCA-BUT.
#
# Slurm owns stdout/stderr for this preset. It therefore writes only
# slurm_JOBID.out and slurm_JOBID.err, without duplicate output.log/error.log.
################################################################################
export MEMORY="${MEMORY:-32G}"
export TIME_LIMIT="${TIME_LIMIT:-03:00:00}"

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SHARED_JOB="$SCRIPT_DIR/job.sh"
export CA_RUNS_DIR="${CA_RUNS_DIR:-$SCRIPT_DIR/runs}"
export CA_PYTHON_ENTRYPOINT="cellular_automaton/dca_main.py"
export MIRROR_JOB_LOGS=0
export REPEAT_CACHE_WINDOW=""
export CA_EXTRAPOLATION_VAL_PAIRS=""
export CA_FINAL_EVAL_PAIRS=""
export DIAGNOSTIC_MAX_REPEATS=""
export MODEL_RUN_NAME="${MODEL_RUN_NAME:-dca_but_delayed}"
echo "$SCRIPT_DIR"
echo "$SCRIPT_DIR"
echo "$SCRIPT_DIR"

EFFECTIVE_EVAL_FREQ="${EVAL_FREQ:-250}"
CLI_ARGS=("$@")
ARG_INDEX=0
while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ]; do
    if [ "${CLI_ARGS[$ARG_INDEX]}" = "--eval_freq" ]; then
        EFFECTIVE_EVAL_FREQ="${CLI_ARGS[$((ARG_INDEX + 1))]}"
        ARG_INDEX=$((ARG_INDEX + 2))
    else
        ARG_INDEX=$((ARG_INDEX + 1))
    fi
done

PRESET_ARGS=(
    --model "${MODEL:-dca_but}"
    --attention_mode "${ATTENTION_MODE:-bidirectional}"
    --positional_encoder "${POSITIONAL_ENCODER:-rotary}"
    --ca_delayed_percentage "${CA_DELAYED_PERCENTAGE:-50}"
    --query_horizon_policy "${QUERY_HORIZON_POLICY:-uniform}"
    --ca_delayed_best_metric "${CA_DELAYED_BEST_METRIC:-cell_accuracy}"
    --ca_save_every "${CA_SAVE_EVERY:-$EFFECTIVE_EVAL_FREQ}"
    --ca_exact_data_resume
)

exec bash "$SHARED_JOB" "${PRESET_ARGS[@]}" "$@"
