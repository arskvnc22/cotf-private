#!/bin/bash
################################################################################
# Rule 30 Block Universal Transformer preset.
#
# Preset arguments come first; user arguments come last and therefore win:
#   bash iridis/ca-rule30/job_but.sh --n_embd 128 --seed 3
################################################################################

set -e
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"

PRESET_ARGS=(
    --model "${MODEL:-but_full_depth}"
    --attention_mode "${ATTENTION_MODE:-bidirectional}"
    --positional_encoder "${POSITIONAL_ENCODER:-rotary}"
)

exec bash "$PACKAGE_DIR/job.sh" "${PRESET_ARGS[@]}" "$@"
