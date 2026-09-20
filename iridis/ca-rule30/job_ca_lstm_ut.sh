#!/bin/bash
################################################################################
# Rule 30 LSTM Universal Transformer preset.
#
# The default experiment name separates the two architectural ablations and
# model seeds so their checkpoint directories cannot collide. Preset arguments
# come first; trailing user arguments still win:
#   bash iridis/ca-rule30/job_ca_lstm_ut.sh \
#       --lstm_forget_gate none \
#       --lstm_control_input proposed \
#       --seed 3
################################################################################
export REPEAT_CACHE_WINDOW=""

set -e
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"

EFFECTIVE_FORGET_GATE="${LSTM_FORGET_GATE:-learned}"
EFFECTIVE_CONTROL_INPUT="${LSTM_CONTROL_INPUT:-previous_and_proposed}"
EFFECTIVE_SEED="${SEED:-1}"
EFFECTIVE_EXP_NAME="${EXP_NAME:-}"

# These values participate in shell-side naming as well as Python parsing.
# Read their trailing CLI overrides before constructing the default name.
CLI_ARGS=("$@")
ARG_INDEX=0
while [ "$ARG_INDEX" -lt "${#CLI_ARGS[@]}" ]; do
    case "${CLI_ARGS[$ARG_INDEX]}" in
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
    EFFECTIVE_EXP_NAME="ca30_lstm_ut_fg_${EFFECTIVE_FORGET_GATE}_control_${EFFECTIVE_CONTROL_INPUT}_seed_${EFFECTIVE_SEED}"
fi

PRESET_ARGS=(
    --model "${MODEL:-lstm_ut_bidir}"
    --attention_mode "${ATTENTION_MODE:-bidirectional}"
    --positional_encoder "${POSITIONAL_ENCODER:-rotary}"
    --lstm_forget_gate "$EFFECTIVE_FORGET_GATE"
    --lstm_control_input "$EFFECTIVE_CONTROL_INPUT"
    --exp_name "$EFFECTIVE_EXP_NAME"
)

exec bash "$PACKAGE_DIR/job.sh" "${PRESET_ARGS[@]}" "$@"
