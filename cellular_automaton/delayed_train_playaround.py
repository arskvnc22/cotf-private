"""Run the delayed-CA trainer end to end with a tiny DCA BUT on CPU."""

from __future__ import annotations

import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models
import distributed
from cellular_automaton.ca_delayed_train import train_ca as train_delayed_ca
from cellular_automaton.dca_main import (
    apply_ca_task_config,
    get_args,
    make_ca_dataloaders,
    seed_global_training_rngs,
)

def main():
    with tempfile.TemporaryDirectory(prefix="delayed_ca_debug_") as output_dir:
        sys.argv = [
            sys.argv[0],
            "--device", "cpu",
            "--model", "dca_but",
            "--attention_mode", "bidirectional",
            "--n_layer", "1",
            "--n_head", "1",
            "--n_embd", "16",
            "--sequence_length", "8",
            "--n_repeat", "2",
            "--batch_size", "2",
            "--acc_steps", "1",
            "--iterations", "12",
            "--eval_freq", "1",
            "--scheduler", "none",
            "--use_pretrained", "None",
            "--ca_train_num_cells", "8",
            "--ca_eval_num_cells", "8",
            "--ca_train_samples", "8",
            "--ca_val_samples", "4",
            "--ca_test_samples", "4",
            "--results_base_folder", output_dir,
            "--ca_train_pairs",
            "1:1",
            "2:2",
            "3:3",

            "--ca_delayed_percentage", "50",
            "--query_horizon_policy", "uniform",
            "--ca_max_relative_age", "3"
        ]
        args = get_args()
        backend = distributed.make_backend_from_args(args)
        args = backend.get_adjusted_args_for_process(args)
        args.device = torch.device("cpu")
        args.quiet_trainer_output = False
        with redirect_stdout(StringIO()):
            seed_global_training_rngs(args)
            apply_ca_task_config(args, backend)
            args.ca_repeat_diagnostic_max_repeats = None
            args.ca_repeat_diagnostic_horizons = None
            args.direction_embedding = True
            args.world_size = backend.get_world_size()
            train_loader, eval_loaders, test_loaders = make_ca_dataloaders(
                args, backend
            )
            model = models.make_model_from_args(args).to(args.device)
            model = backend.transform_model(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = None
        checkpoint_dir = Path(output_dir) / "run"
        checkpoint_dir.mkdir()

        train_delayed_ca(
            model,
            optimizer,
            scheduler,
            train_loader,
            eval_loaders,
            test_loaders,
            args,
            backend,
            checkpoint_dir,
        )
if __name__ == "__main__":
    main()
