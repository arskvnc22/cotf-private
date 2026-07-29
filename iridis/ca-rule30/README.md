# Rule 30 Iridis launchers

All new Rule 30 Slurm outputs are placed in one flat `runs/` directory. Run
names contain the resolved model name:

```text
runs/run_0__but_full_depth/
runs/run_1__fixed_cot_attn/
```

Once Python starts, every run directory contains:

```text
run_manifest.json   resolved model, data, optimizer, pair, seed, and Slurm metadata
eval_metrics.jsonl  normalized scalar metrics, refreshed after every evaluation
notes.md            human-editable hypothesis and observations
slurm_*.out/.err    scheduler output
output.log          mirrored standard output
error.log           mirrored standard error
```

`eval_metrics.jsonl` is long-form: each line is one scalar measurement with
explicit fields for split, checkpoint type, training step, row length, Rule 30
steps, model repeats, repeat-state endpoints, metric name, and value. It is
atomically rewritten from `training_stats.json`, so resuming a run does not
duplicate earlier evaluations.

Use a model preset:

```bash
bash iridis/ca-rule30/job_but.sh
bash iridis/ca-rule30/job_cotformer.sh
```

`job_cotformer.sh` currently prepares the `fixed_cot_attn` configuration and
output naming only. That model still needs the CA runtime-repeat, all-cell
logit, and bidirectional-attention changes before the job can train.

Each preset calls the shared `job.sh`. Preset arguments are placed before user
arguments, so trailing overrides win:

```bash
bash iridis/ca-rule30/job_but.sh \
  --n_embd 128 \
  --seed 3 \
  --data_seed 17 \
  --ca_tags reproduction run28 \
  --ca_note "Confirming the four-depth extrapolation result" \
  --ca_train_pairs 2:2 4:4 \
  --ca_extrapolation_val_pairs 1:1 3:3 5:5 6:6 7:7 \
  --ca_final_eval_pairs 8:8
```

Environment overrides remain supported:

```bash
SEED=3 DATA_SEED=17 LR=3e-4 WEIGHT_DECAY=0.05 \
  bash iridis/ca-rule30/job_but.sh
```

Trailing CLI values take precedence over environment values and preset
values. `SEED`/`--seed` controls model initialization, while
`DATA_SEED`/`--data_seed` controls the generated training rows and DataLoader
shuffle. Their defaults are independently fixed at 0 and 1, so a model-seed
sweep holds the data realization constant unless explicitly changed. Both
seeds appear in the generated experiment name and run manifest. The existing
Rule 30 launcher directories and historical run outputs are intentionally left
in place.

## Analysis

The analyzer searches manifests recursively, so experiment structure comes
from metadata rather than directory names. It has no pandas dependency.

List runs and filter on any manifest field:

```bash
python -m cellular_automaton.ca_analyze list \
  --runs iridis/ca-rule30/runs \
  --where model=but_full_depth \
  --where annotations.tags~reproduction \
  --where training_pairs='1:1|2:2|4:4' \
  --where weight_decay=0.1
```

Create a dynamic pair table for the unconstrained extrapolation checkpoint:

```bash
python -m cellular_automaton.ca_analyze table \
  --runs iridis/ca-rule30/runs \
  --metric cell_accuracy \
  --split final_test \
  --role internal_repeat_extrapolation \
  --checkpoint best_extrapolation_unconstrained \
  --average-over seed,data_seed \
  --output-dir iridis/ca-rule30/reports/but_seed_comparison
```

Rows are kept separate unless every non-averaged configuration field agrees.
The table therefore will not silently combine different optimizers,
architectures, training pairs, validation/test seeds, or other meaningful
settings. Aggregates include the individual values, mean, sample variance and
standard deviation, median, quartiles, minimum, and maximum. Use
`--average-over seed` to average model initialization while holding the data
seed fixed, or `--average-over data_seed` for the converse.

Plot final accuracy by evaluated pair, extrapolation accuracy over training,
and adjacent-repeat hidden-state similarity:

```bash
python -m cellular_automaton.ca_analyze plot-horizon \
  --runs iridis/ca-rule30/runs \
  --checkpoint best_extrapolation_unconstrained

python -m cellular_automaton.ca_analyze plot-training \
  --runs iridis/ca-rule30/runs \
  --where model=but_full_depth

python -m cellular_automaton.ca_analyze plot-state \
  --runs iridis/ca-rule30/runs \
  --metric cosine_similarity
```

Each plot command saves the image, its plotted CSV/JSON data, and an
`analysis_manifest.json` containing the exact filters and selected runs under
`iridis/ca-rule30/reports/`. Seed traces are faint; the main curves show the
mean with a one-standard-deviation band. Aggregation uses measurements at
their actual evaluation steps and does not interpolate missing steps.

## Training exposure accounting

Every consumed microbatch now contributes its actual local batch size,
multiplied by the distributed world size, rather than assuming every batch is
full. `training_stats.json`, `eval_metrics.jsonl`, periodic Slurm output, and
WandB logging record:

```text
materialized_training_rows
total_optimizer_steps
total_microbatches
total_examples_seen
total_cells_seen
equivalent_dataset_passes
optimizer_steps, microbatches, examples and cells for each STEPS:REPEATS pair
example_fraction and optimizer_step_fraction for each pair
```

`materialized_training_rows` is the size of the fixed generated pool;
`total_examples_seen` counts repeated presentations as the loader cycles over
that pool. Exposure snapshots use the `training` split with roles
`training_exposure` and `training_exposure_by_pair`. For example:

```bash
python -m cellular_automaton.ca_analyze table \
  --runs iridis/ca-rule30/runs \
  --split training \
  --role training_exposure_by_pair \
  --metric examples_seen \
  --checkpoint none \
  --length any
```

New per-step records support exact reconstruction after resume. If an older
checkpoint is resumed, historical steps that predate these counters are
estimated using the former full-batch formula and reported with
`accounting_exact=0` and a nonzero `legacy_approximated_steps`; all newly
consumed batches remain exact.

Run `python -m cellular_automaton.ca_analyze COMMAND --help` for all selectors.
Historical runs without `run_manifest.json` are not guessed from filenames;
they need to be rerun or explicitly migrated before this analyzer includes
them.
