#!/usr/bin/env bash
# Run with bash on the login node to submit one L4 baseline evaluation:
#   bash iridis/dca-30/job_dca_recall_eval.sh
# No evaluation or submission is performed by merely sourcing iridis/env.sh.
set -euo pipefail

if (( $# > 0 )); then
    echo "Usage: bash iridis/dca-30/job_dca_recall_eval.sh" >&2
    exit 2
fi

# Slurm copies the script to its spool, so resolve and export the repository
# location before submission instead of deriving it from the worker's script.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    DCA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export DCA_REPO_DIR="$(cd "$DCA_SCRIPT_DIR/../.." && pwd)"

    echo "Submitting run 56 baseline recall evaluation on one L4."
    echo "Checkpoint: best_delayed_recall.pt, step 10000"
    echo "Evolution horizon: 30; all queries; two recall passes"
    echo "Validation: four batches; 16 examples per query"
    echo "Logs: $DCA_SCRIPT_DIR/recall_eval_%j.{out,err}"

    exec sbatch \
        --job-name=dca_recall_eval \
        --partition=ecsstudents_l4 \
        --account=ecsstudents \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --gres=gpu:1 \
        --mem=32G \
        --time=03:00:00 \
        --chdir="$DCA_REPO_DIR" \
        --output="$DCA_SCRIPT_DIR/recall_eval_%j.out" \
        --error="$DCA_SCRIPT_DIR/recall_eval_%j.err" \
        --export=ALL \
        "$DCA_SCRIPT_DIR/job_dca_recall_eval.sh"
fi

# A direct sbatch invocation from the repository root can also resolve paths;
# the self-submitting route above explicitly supplies DCA_REPO_DIR.
DCA_REPO_DIR="${DCA_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$DCA_REPO_DIR" || ! -f "$DCA_REPO_DIR/iridis/env.sh" ]]; then
    echo "Cannot locate repository; launch with bash from the login node." >&2
    exit 2
fi

source "$DCA_REPO_DIR/iridis/env.sh"
module load conda
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PREFIX"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$DCA_REPO_DIR:${PYTHONPATH:-}"
cd "$DCA_REPO_DIR"

DCA_RUN_ID=run_56__dca_cotf_h30_recall2_allquery_ckpt
DCA_OUTPUT_ROOT="$DCA_REPO_DIR/iridis/dca-30/outputs/recall-audit/baselines"
DCA_ARTIFACT_NAME="run56-h30-recall2-baseline-job-${SLURM_JOB_ID}"

echo "Job: $SLURM_JOB_ID; host: $(hostname)"
echo "Artifact: $DCA_OUTPUT_ROOT/$DCA_RUN_ID/$DCA_ARTIFACT_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Omitting --query-repeats requests every evolution repeat 1..30, including
# age zero. Every query still evolves to 30, then decodes after two recall
# passes. Baseline preserves the checkpoint's attention and cache policy.
python3 -m cellular_automaton.dca_att_eval \
    --run-manifest "$DCA_REPO_DIR/iridis/dca-30/runs/$DCA_RUN_ID/run_manifest.json" \
    --expected-run-id "$DCA_RUN_ID" \
    --checkpoint-name best_delayed_recall.pt \
    --expected-step 10000 \
    --device cuda:0 \
    --evaluate-delayed-recall \
    --num-repeats 30 \
    --num-recall-repeats 2 \
    --conditions baseline \
    --diagnostic-split validation \
    --max-batches 4 \
    --num-examples 16 \
    --output-root "$DCA_OUTPUT_ROOT" \
    --artifact-name "$DCA_ARTIFACT_NAME"
