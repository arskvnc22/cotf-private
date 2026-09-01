#!/bin/bash
################################################################################
# Rule 30 CoTFormer preset.
#
# This selects the repository's cache-retaining CoTFormer implementation.
# Preset arguments come first; user arguments come last and therefore win:
#   bash iridis/ca-rule30/job_cotformer.sh --n_embd 128 --seed 3
#
# The launcher is ready for the CA interface, but fixed_cot_attn must still be
# given runtime-repeat/all-cell/bidirectional support before CA training works.
################################################################################

set -e
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"

PRESET_ARGS=(
    --model "${MODEL:-ca_cotf}"
    --attention_mode "${ATTENTION_MODE:-bidirectional}"
    --positional_encoder "${POSITIONAL_ENCODER:-rotary}"
)

exec bash "$PACKAGE_DIR/job.sh" "${PRESET_ARGS[@]}" "$@"
