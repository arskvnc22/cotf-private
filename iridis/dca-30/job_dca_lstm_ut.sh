#!/bin/bash
################################################################################
# Delayed-recall Rule 30 LSTM Universal Transformer preset.
#
# Raw long-term-memory diagnostics are enabled by default for evolution repeat
# 1 and recall step 1 while querying repeat 1. Override the diagnostic lists
# with trailing Python arguments, or disable all default memory diagnostics with
# DCA_LSTM_SAVE_MEMORY=0.
#
# Example ablation:
#   bash iridis/dca-30/job_dca_lstm_ut.sh \
#       --lstm_forget_gate none \
#       --lstm_control_input proposed \
#       --dca_lstm_memory_evolution_repeats 1 6 12 \
#       --dca_lstm_memory_query_repeats 1 6 12 \
#       --seed 3
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
export MODEL_RUN_NAME="${MODEL_RUN_NAME:-dca_lstm_ut_delayed}"

EFFECTIVE_EVAL_FREQ="${EVAL_FREQ:-250}"
EFFECTIVE_FORGET_GATE="${LSTM_FORGET_GATE:-learned}"
EFFECTIVE_CONTROL_INPUT="${LSTM_CONTROL_INPUT:-previous_and_proposed}"
EFFECTIVE_SEED="${SEED:-1}"
EFFECTIVE_EXP_NAME="${EXP_NAME:-}"

CLI_ARGS=("$@")
ARG_INDEX=0
while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ]; do
    case "${CLI_ARGS[$ARG_INDEX]}" in
        --eval_freq)
            EFFECTIVE_EVAL_FREQ="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --lstm_forget_gate)
            EFFECTIVE_FORGET_GATE="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --lstm_control_input)
            EFFECTIVE_CONTROL_INPUT="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --seed)
            EFFECTIVE_SEED="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        --exp_name)
            EFFECTIVE_EXP_NAME="${CLI_ARGS[$((ARG_INDEX + 1))]}"
            ARG_INDEX=$((ARG_INDEX + 2))
            ;;
        *)
            ARG_INDEX=$((ARG_INDEX + 1))
            ;;
    esac
done

if [ -z "$EFFECTIVE_EXP_NAME" ]; then
    EFFECTIVE_EXP_NAME="dca30_lstm_ut_fg_${EFFECTIVE_FORGET_GATE}_control_${EFFECTIVE_CONTROL_INPUT}_seed_${EFFECTIVE_SEED}"
fi

PRESET_ARGS=(
    --model "${MODEL:-dca_lstm_ut}"
    --attention_mode "${ATTENTION_MODE:-bidirectional}"
    --positional_encoder "${POSITIONAL_ENCODER:-rotary}"
    --lstm_forget_gate "$EFFECTIVE_FORGET_GATE"
    --lstm_control_input "$EFFECTIVE_CONTROL_INPUT"
    --ca_delayed_percentage "${CA_DELAYED_PERCENTAGE:-50}"
    --query_horizon_policy "${QUERY_HORIZON_POLICY:-uniform}"
    --ca_delayed_best_metric "${CA_DELAYED_BEST_METRIC:-cell_accuracy}"
    --ca_save_every "${CA_SAVE_EVERY:-$EFFECTIVE_EVAL_FREQ}"
    --ca_exact_data_resume
    --exp_name "$EFFECTIVE_EXP_NAME"
)

DCA_LSTM_SAVE_MEMORY="${DCA_LSTM_SAVE_MEMORY:-1}"
if [ "$DCA_LSTM_SAVE_MEMORY" = "1" ]; then
    MEMORY_EVOLUTION_REPEATS="${DCA_LSTM_MEMORY_EVOLUTION_REPEATS:-1}"
    MEMORY_QUERY_REPEATS="${DCA_LSTM_MEMORY_QUERY_REPEATS:-1}"
    MEMORY_RECALL_STEPS="${DCA_LSTM_MEMORY_RECALL_STEPS:-1}"
    MEMORY_EXAMPLES="${DCA_LSTM_MEMORY_EXAMPLES:-2}"

    read -r -a MEMORY_EVOLUTION_ARGS <<< "$MEMORY_EVOLUTION_REPEATS"
    read -r -a MEMORY_QUERY_ARGS <<< "$MEMORY_QUERY_REPEATS"
    read -r -a MEMORY_RECALL_ARGS <<< "$MEMORY_RECALL_STEPS"

    PRESET_ARGS+=(
        --dca_lstm_memory_evolution_repeats "${MEMORY_EVOLUTION_ARGS[@]}"
        --dca_lstm_memory_query_repeats "${MEMORY_QUERY_ARGS[@]}"
        --dca_lstm_memory_recall_steps "${MEMORY_RECALL_ARGS[@]}"
        --dca_lstm_memory_examples "$MEMORY_EXAMPLES"
    )
elif [ "$DCA_LSTM_SAVE_MEMORY" != "0" ]; then
    echo "DCA_LSTM_SAVE_MEMORY must be 0 or 1." >&2
    exit 2
fi

exec bash "$SHARED_JOB" "${PRESET_ARGS[@]}" "$@"
