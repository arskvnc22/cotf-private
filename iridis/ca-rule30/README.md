# Rule 30 Iridis launchers

All new Rule 30 Slurm outputs are placed in one flat `runs/` directory. Run
names contain the resolved model name:

```text
runs/run_0__but_full_depth/
runs/run_1__fixed_cot_attn/
```

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
  --ca_train_pairs 2:2 4:4 \
  --ca_extrapolation_val_pairs 1:1 3:3 5:5 6:6 7:7 \
  --ca_final_eval_pairs 8:8
```

Environment overrides remain supported:

```bash
LR=3e-4 WEIGHT_DECAY=0.05 bash iridis/ca-rule30/job_but.sh
```

Trailing CLI values take precedence over environment values and preset
values. The existing Rule 30 launcher directories and historical run outputs
are intentionally left in place.
