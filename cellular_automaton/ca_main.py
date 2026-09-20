"""Entry point for the Rule 30 cellular-automaton experiments.

This module owns task-level configuration and runtime setup, then hands the
constructed model and loaders to the dedicated CA trainer.
"""

import argparse
import copy
import inspect
import random
import sys
from pathlib import Path

# Running ``python cellular_automaton/ca_main.py`` makes this directory, rather
# than the repository root, the first import location.  Add the root explicitly
# so shared packages such as config and distributed resolve without requiring a
# job script to set PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import config
import distributed
import models
from optim.runner_utils import (
    load_training_checkpoint,
    make_optimizer,
    make_scheduler,
    resolve_resume_checkpoint,
    sanitize_for_json,
)

try:
    from .ca_gen import MaterializedRule30Dataset, Rule30Dataset
    from .ca_reporting import (
        MANIFEST_FILENAME,
        build_run_manifest,
        ensure_notes_file,
        read_json,
        update_run_manifest_status,
        write_run_manifest,
    )
    from .ca_train import train_ca
except ImportError:
    # Support direct execution as ``python cellular_automaton/ca_main.py``.
    from ca_gen import MaterializedRule30Dataset, Rule30Dataset
    from ca_reporting import (
        MANIFEST_FILENAME,
        build_run_manifest,
        ensure_notes_file,
        read_json,
        update_run_manifest_status,
        write_run_manifest,
    )
    from ca_train import train_ca


CA_TASK_NAME = "rule30"
CA_VOCAB_SIZE = 2


def ca_step_repeat_pair(value):
    """Parse a CA horizon/model-depth pair written as CA_STEPS:MODEL_REPEATS."""
    try:
        ca_steps, num_repeats = (int(part) for part in value.split(":"))
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "Expected CA_STEPS:MODEL_REPEATS, for example 8:8."
        )
    if ca_steps <= 0 or num_repeats <= 0:
        raise argparse.ArgumentTypeError("CA steps and model repeats must be positive.")
    return ca_steps, num_repeats


def get_args():
    """Parse CA-specific arguments together with the shared model arguments."""
    # Discover the requested shared configuration format without letting this
    # preliminary parser consume --help or any task/model argument.
    format_parser = argparse.ArgumentParser(add_help=False)
    format_parser.add_argument(
        "--config_format",
        default="base",
        choices=config.registered_formats(),
    )
    format_args, _ = format_parser.parse_known_args()

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format",
        default="base",
        choices=config.registered_formats(),
    )
    def positive_int(value):
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("Expected a positive integer.")
        return parsed
    parser.add_argument(
        "--repeat_cache_window",
        "--repeat-cache-window",
        dest="repeat_cache_window",
        type=positive_int,
        default=None,

    )

    # Task shape.  Training length is separate from sequence_length because
    # sequence_length is the largest row the model can accept, while training
    # can deliberately expose it to shorter rows.
    parser.add_argument("--ca_train_num_cells", type=int, default=64)
    parser.add_argument(
        "--ca_eval_num_cells",
        type=int,
        nargs="+",
        default=[64, 128, 256],
    )
    parser.add_argument(
        "--ca_steps",
        type=int,
        default=1,
        help="Number of Rule 30 updates between each input row and target row.",
    )
    parser.add_argument(
        "--attention_implementation",
        choices=("sdpa", "manual"),
        default="sdpa",
        help=(
            "Attention backend used by models that support an explicit choice. "
            "Manual attention remains separate from diagnostic collection."
        ),
    )
    parser.add_argument(
        "--lstm_forget_gate",
        choices=("learned", "none"),
        default="learned",
        help=(
            "LSTM-UT retention ablation. 'learned' uses a learned forget gate; "
            "'none' retains the previous cell exactly before the additive write."
        ),
    )
    parser.add_argument(
        "--lstm_control_input",
        choices=("previous_and_proposed", "proposed"),
        default="previous_and_proposed",
        help=(
            "Inputs used by the LSTM-UT forget, write, proposal, and hidden gates."
        ),
    )

    parser.add_argument(
        "--ca_train_pairs",
        type=ca_step_repeat_pair,
        nargs="+",
        default=None,
        metavar="STEPS:REPEATS",
        help=(
            "Variable-horizon training pairs. One pair is used for an entire "
            "optimizer step, for example 1:1 2:2 4:4. When omitted, the "
            "existing fixed --ca_steps/--n_repeat path is unchanged."
        ),
    )
    parser.add_argument(
        "--ca_best_pair",
        type=ca_step_repeat_pair,
        default=None,
        metavar="STEPS:REPEATS",
        help=(
            "Training pair used for best-checkpoint selection. Defaults to the "
            "pair with the largest CA horizon, then largest repeat count."
        ),
    )
    parser.add_argument(
        "--ca_boundary",
        choices=["periodic"],
        default="periodic",
        help="Boundary convention used to generate Rule 30 targets.",
    )

    # Dataset construction.  The validation seed is separate so validation is
    # fixed and cannot accidentally reproduce the indexed training examples.
    parser.add_argument("--ca_train_samples", type=int, default=100_000)
    parser.add_argument("--ca_val_samples", type=int, default=10_000)
    parser.add_argument("--ca_val_seed", type=int, default=1_000_003)
    parser.add_argument(
        "--ca_test_samples",
        type=int,
        default=None,
        help="Final-test rows per length; defaults to --ca_val_samples.",
    )
    parser.add_argument(
        "--ca_test_seed",
        type=int,
        default=2_000_003,
        help="Independent seed for untouched final-test rows.",
    )
    parser.add_argument("--ca_bernoulli_p", type=float, default=0.5)
    parser.add_argument(
        "--ca_data_mode",
        choices=["materialized", "indexed"],
        default="materialized",
        help=(
            "materialized generates each fixed split once in memory; indexed "
            "retains the older deterministic per-item generation path."
        ),
    )
    parser.add_argument(
        "--ca_exact_data_resume",
        action="store_true",
        help=(
            "Restore the precise shuffle position after a checkpoint. By "
            "default a resumed run starts a fresh DataLoader pass."
        ),
    )
    parser.add_argument(
        "--ca_shuffle_seed",
        type=int,
        default=None,
        help=(
            "Seed controlling training-row order. Defaults to --data_seed so "
            "existing experiments retain their current ordering."
        ),
    )
    parser.add_argument("--ca_num_workers", type=int, default=0)
    parser.add_argument("--ca_eval_batch_size", type=int, default=None)
    parser.add_argument("--ca_eval_max_batches", type=int, default=None)
    parser.add_argument("--ca_save_every", type=int, default=None)
    parser.add_argument("--ca_log_every", type=int, default=None)
    parser.add_argument(
        "--ca_best_length",
        type=int,
        default=None,
        help="Validation row length used to select the best checkpoint.",
    )
    parser.add_argument(
        "--ca_best_metric",
        choices=["exact_sequence_accuracy", "cell_accuracy", "loss"],
        default="exact_sequence_accuracy",
    )
    parser.add_argument(
        "--ca_extrapolation_val_pairs",
        type=ca_step_repeat_pair,
        nargs="*",
        default=[],
        metavar="STEPS:REPEATS",
        help=(
            "Unseen pairs evaluated at every normal evaluation step for "
            "extrapolation checkpoint selection, for example 3:3."
        ),
    )
    parser.add_argument(
        "--ca_extrapolation_best_pair",
        type=ca_step_repeat_pair,
        default=None,
        metavar="STEPS:REPEATS",
        help=(
            "Extrapolation-validation pair used to select both the strict and "
            "unconstrained best-extrapolation checkpoints."
        ),
    )
    parser.add_argument(
        "--ca_extrapolation_best_metric",
        choices=["exact_sequence_accuracy", "cell_accuracy", "loss"],
        default="cell_accuracy",
    )
    parser.add_argument(
        "--ca_extrapolation_min_id_cell_accuracy",
        type=float,
        default=0.99,
        help=(
            "Minimum cell accuracy on every training pair before strict "
            "extrapolation checkpoint selection. The unconstrained selector "
            "ignores this threshold."
        ),
    )
    parser.add_argument(
        "--ca_extrapolation_min_id_exact_sequence_accuracy",
        type=float,
        default=0.95,
        help=(
            "Minimum exact-row accuracy on every training pair before strict "
            "extrapolation checkpoint selection. The unconstrained selector "
            "ignores this threshold."
        ),
    )
    parser.add_argument(
        "--ca_repeat_diagnostic_max_repeats",
        type=int,
        default=None,
        help=(
            "Run repeat diagnostics from one through this depth at every eval. "
            "For variable training, defaults to the largest configured pair."
        ),
    )
    parser.add_argument(
        "--ca_repeat_diagnostic_horizons",
        type=int,
        nargs="*",
        default=None,
        metavar="STEPS",
        help="True Rule 30 horizons used as matrix columns; defaults to 0..max repeats.",
    )
    parser.add_argument(
        "--ca_repeat_diagnostic_examples",
        type=int,
        default=1,
        help="Fixed decoded/target rows printed at every evaluation step.",
    )
    parser.add_argument(
        "--ca_repeat_diagnostic_max_batches",
        type=int,
        default=None,
        help="Optional diagnostic-only batch cap; defaults to --ca_eval_max_batches.",
    )
    parser.add_argument(
        "--ca_diagnostic_length",
        type=int,
        default=None,
        help="Validation row length used for optional fixed-CoT diagnostics.",
    )
    parser.add_argument(
        "--ca_log_fixed_cot_diagnostics",
        action="store_true",
        help="Collect model-specific repeat/attention diagnostics when available.",
    )
    parser.add_argument(
        "--ca_final_eval_pairs",
        type=ca_step_repeat_pair,
        nargs="*",
        default=[],
        metavar="STEPS:REPEATS",
        help="One-forward internal extrapolation settings, such as 2:2 8:8.",
    )
    parser.add_argument(
        "--ca_final_external_steps",
        type=int,
        nargs="*",
        default=[],
        metavar="STEPS",
        help="Target CA horizons for repeated full-model rollout.",
    )
    parser.add_argument("--ca_final_eval_max_batches", type=int, default=None)
    parser.add_argument(
        "--ca_run_dir",
        type=str,
        default=None,
        help=(
            "Optional scheduler-side run directory. When provided, resolved "
            "metadata, normalized metrics, and notes are written beside the "
            "Slurm logs."
        ),
    )
    parser.add_argument(
    "--ca_delayed_percentage",
    type=int,
    default=50,
    )

    parser.add_argument(
        "--query_horizon_policy",
        choices=("uniform",),
        default="uniform",
    )
    parser.add_argument(
        "--ca_tags",
        nargs="*",
        default=[],
        help="Searchable free-form tags stored in the run manifest.",
    )
    parser.add_argument(
        "--ca_note",
        type=str,
        default=None,
        help="Short searchable experiment note stored in the run manifest.",
    )

    # The existing configuration format adds all shared optimizer, model,
    # logging, and distributed arguments before the one final parse.  Keeping a
    # single final parse also makes --help show both shared and CA options.
    return config.parse_args_with_format(
        format=format_args.config_format,
        base_parser=parser,
        args=None,
        namespace=None,
    )


def print_master(distributed_backend, message):
    """Print once under DDP and normally in a single-process run."""
    if distributed_backend.is_master_process():
        print(message)


def maybe_make_sampler(dataset, shuffle, seed, *, distributed):
    """Partition a dataset under DDP; otherwise let DataLoader shuffle it."""
    if (
        distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    return None


def make_loader(
    dataset,
    *,
    batch_size,
    shuffle,
    seed,
    num_workers,
    pin_memory,
    distributed=False,
):
    """Construct a reproducibly ordered loader for an already-fixed dataset."""
    sampler = maybe_make_sampler(
        dataset, shuffle=shuffle, seed=seed, distributed=distributed
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False, # if ddp DistributedSampler owns shuffling
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def make_ca_fixed_loaders(args, *, base_seed, num_samples):
    """Build deterministic validation/test loaders from resolved CA arguments."""
    pin_memory = args.device.type == "cuda"
    dataset_class = (
        MaterializedRule30Dataset
        if args.ca_data_mode == "materialized"
        else Rule30Dataset
    )
    batch_size = args.ca_eval_batch_size or args.batch_size
    loaders = {}
    for num_cells in args.ca_eval_num_cells:
        # Including the length in the split seed keeps lengths deterministic
        # without making their rows prefixes of one another.
        split_seed = int(base_seed) + int(num_cells)
        dataset = dataset_class(
            num_samples=num_samples,
            num_cells=num_cells,
            steps=args.ca_steps,
            bernoulli_p=args.ca_bernoulli_p,
            seed=split_seed,
        )
        loaders[num_cells] = make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=split_seed,
            num_workers=args.ca_num_workers,
            pin_memory=pin_memory,
            distributed=False,
        )
    return loaders


def make_ca_dataloaders(args, distributed_backend):
    """Build training, fixed validation, and independent final-test loaders."""
    train_data_seed = int(args.data_seed)
    shuffle_seed = int(args.ca_shuffle_seed)
    pin_memory = args.device.type == "cuda"
    dataset_class = (
        MaterializedRule30Dataset
        if args.ca_data_mode == "materialized"
        else Rule30Dataset
    )

    train_dataset = dataset_class(
        num_samples=args.ca_train_samples,
        num_cells=args.ca_train_num_cells,
        steps=args.ca_steps,
        bernoulli_p=args.ca_bernoulli_p,
        seed=train_data_seed,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=shuffle_seed,
        num_workers=args.ca_num_workers,
        pin_memory=pin_memory,
        distributed=True,
    )

    eval_loaders = make_ca_fixed_loaders(
        args,
        base_seed=args.ca_val_seed,
        num_samples=args.ca_val_samples,
    )
    test_loaders = make_ca_fixed_loaders(
        args,
        base_seed=args.ca_test_seed,
        num_samples=args.ca_test_samples,
    )

    print_master(
        distributed_backend,
        f"Built {len(train_dataset)} {args.ca_data_mode} training examples at length "
        f"{args.ca_train_num_cells}, fixed validation splits, and independent "
        f"final-test splits at lengths {list(eval_loaders)}"
        + (
            f"; variable training pairs are {args.ca_train_pairs}."
            if args.ca_train_pairs
            else "."
        ),
    )
    if args.ca_data_mode == "materialized":
        storage_bytes = train_dataset.storage_bytes + sum(
            loader.dataset.storage_bytes for loader in eval_loaders.values()
        ) + sum(
            loader.dataset.storage_bytes for loader in test_loaders.values()
        )
        print_master(
            distributed_backend,
            f"Materialized CA splits occupy {storage_bytes / (1024 ** 2):.1f} MiB "
            "of CPU tensor storage.",
        )
    return train_loader, eval_loaders, test_loaders


def apply_ca_task_config(args, distributed_backend):
    """Validate CA arguments and derive shared model configuration fields."""
    if args.ca_train_num_cells <= 0:
        raise ValueError("--ca_train_num_cells must be positive.")
    if not args.ca_eval_num_cells or any(length <= 0 for length in args.ca_eval_num_cells):
        raise ValueError("--ca_eval_num_cells must contain positive lengths.")
    if args.ca_steps <= 0:
        raise ValueError("--ca_steps must be positive.")
    if args.ca_train_pairs:
        if len(set(args.ca_train_pairs)) != len(args.ca_train_pairs):
            raise ValueError("--ca_train_pairs cannot contain duplicate pairs.")
        if args.ca_best_pair is None:
            args.ca_best_pair = max(args.ca_train_pairs)
        elif args.ca_best_pair not in args.ca_train_pairs:
            raise ValueError("--ca_best_pair must be included in --ca_train_pairs.")
        if args.ca_final_external_steps:
            raise ValueError(
                "--ca_final_external_steps is not yet defined for variable-horizon "
                "training. Use --ca_final_eval_pairs to test additional internal "
                "step/repeat settings."
            )
        if args.ca_log_fixed_cot_diagnostics:
            raise ValueError(
                "Fixed-CoT diagnostics are not yet supported with "
                "--ca_train_pairs."
            )
    elif args.ca_best_pair is not None:
        raise ValueError("--ca_best_pair requires --ca_train_pairs.")
    if args.ca_extrapolation_val_pairs:
        if not args.ca_train_pairs:
            raise ValueError(
                "--ca_extrapolation_val_pairs requires --ca_train_pairs."
            )
        if len(set(args.ca_extrapolation_val_pairs)) != len(
            args.ca_extrapolation_val_pairs
        ):
            raise ValueError("Extrapolation-validation pairs cannot be duplicated.")
        overlap = set(args.ca_train_pairs) & set(args.ca_extrapolation_val_pairs)
        if overlap:
            raise ValueError(
                "Extrapolation-validation pairs must be unseen during training; "
                f"overlap: {sorted(overlap)}."
            )
        if args.ca_extrapolation_best_pair is None:
            args.ca_extrapolation_best_pair = max(args.ca_extrapolation_val_pairs)
        elif args.ca_extrapolation_best_pair not in args.ca_extrapolation_val_pairs:
            raise ValueError(
                "--ca_extrapolation_best_pair must be included in "
                "--ca_extrapolation_val_pairs."
            )
        test_overlap = set(args.ca_extrapolation_val_pairs) & set(
            args.ca_final_eval_pairs
        )
        if test_overlap:
            raise ValueError(
                "Pairs used for extrapolation checkpoint selection cannot also "
                f"be final-test pairs; overlap: {sorted(test_overlap)}."
            )
    elif args.ca_extrapolation_best_pair is not None:
        raise ValueError(
            "--ca_extrapolation_best_pair requires --ca_extrapolation_val_pairs."
        )
    for name, value in (
        ("--ca_extrapolation_min_id_cell_accuracy", args.ca_extrapolation_min_id_cell_accuracy),
        (
            "--ca_extrapolation_min_id_exact_sequence_accuracy",
            args.ca_extrapolation_min_id_exact_sequence_accuracy,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")

    if args.ca_test_samples is None:
        args.ca_test_samples = args.ca_val_samples
    if args.ca_shuffle_seed is None:
        args.ca_shuffle_seed = int(args.data_seed)
    if (
        args.ca_train_samples <= 0
        or args.ca_val_samples <= 0
        or args.ca_test_samples <= 0
    ):
        raise ValueError("CA train, validation, and test sample counts must be positive.")
    if args.ca_test_seed == args.ca_val_seed:
        raise ValueError("--ca_test_seed must differ from --ca_val_seed.")
    if not 0.0 <= args.ca_bernoulli_p <= 1.0:
        raise ValueError("--ca_bernoulli_p must lie in [0, 1].")
    if args.ca_num_workers < 0:
        raise ValueError("--ca_num_workers cannot be negative.")
    if args.ca_eval_batch_size is not None and args.ca_eval_batch_size <= 0:
        raise ValueError("--ca_eval_batch_size must be positive when provided.")
    if args.ca_eval_max_batches is not None and args.ca_eval_max_batches <= 0:
        raise ValueError("--ca_eval_max_batches must be positive when provided.")
    if (
        args.ca_repeat_diagnostic_max_batches is not None
        and args.ca_repeat_diagnostic_max_batches <= 0
    ):
        raise ValueError(
            "--ca_repeat_diagnostic_max_batches must be positive when provided."
        )
    if args.ca_repeat_diagnostic_examples < 0:
        raise ValueError("--ca_repeat_diagnostic_examples cannot be negative.")
    if args.ca_save_every is not None and args.ca_save_every <= 0:
        raise ValueError("--ca_save_every must be positive when provided.")
    if args.ca_log_every is not None and args.ca_log_every <= 0:
        raise ValueError("--ca_log_every must be positive when provided.")
    if args.eval_freq <= 0:
        raise ValueError("--eval_freq must be positive.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")
    if args.acc_steps <= 0:
        raise ValueError("--acc_steps must be positive.")
    if args.compile:
        raise ValueError(
            "The CA runner does not yet support --compile because resumable "
            "state names must remain stable."
        )
    if any(steps <= 0 for steps in args.ca_final_external_steps):
        raise ValueError("--ca_final_external_steps must contain positive values.")
    if (
        args.ca_final_eval_max_batches is not None
        and args.ca_final_eval_max_batches <= 0
    ):
        raise ValueError("--ca_final_eval_max_batches must be positive when provided.")
    invalid_external_steps = [
        steps
        for steps in args.ca_final_external_steps
        if steps % args.ca_steps != 0
    ]
    if invalid_external_steps:
        raise ValueError(
            "Every final external CA horizon must be divisible by --ca_steps; "
            f"invalid values: {invalid_external_steps}."
        )

    configured_pairs = tuple(args.ca_train_pairs or ()) + tuple(
        args.ca_extrapolation_val_pairs
    ) + tuple(args.ca_final_eval_pairs)
    if args.ca_repeat_diagnostic_max_repeats is None and configured_pairs:
        args.ca_repeat_diagnostic_max_repeats = max(
            repeats for _, repeats in configured_pairs
        )
    if args.ca_repeat_diagnostic_max_repeats is not None:
        if args.ca_repeat_diagnostic_max_repeats <= 0:
            raise ValueError("--ca_repeat_diagnostic_max_repeats must be positive.")
        if args.ca_repeat_diagnostic_horizons is None:
            args.ca_repeat_diagnostic_horizons = list(
                range(args.ca_repeat_diagnostic_max_repeats + 1)
            )
        elif (
            not args.ca_repeat_diagnostic_horizons
            or any(value < 0 for value in args.ca_repeat_diagnostic_horizons)
            or len(set(args.ca_repeat_diagnostic_horizons))
            != len(args.ca_repeat_diagnostic_horizons)
        ):
            raise ValueError(
                "--ca_repeat_diagnostic_horizons must contain unique, "
                "non-negative horizons."
            )
        missing_extrapolation = [
            pair
            for pair in args.ca_extrapolation_val_pairs
            if pair[0] not in args.ca_repeat_diagnostic_horizons
            or pair[1] > args.ca_repeat_diagnostic_max_repeats
        ]
        if missing_extrapolation:
            raise ValueError(
                "Repeat diagnostics must cover every extrapolation-validation "
                f"pair; missing: {missing_extrapolation}."
            )
    elif args.ca_repeat_diagnostic_horizons is not None:
        raise ValueError(
            "--ca_repeat_diagnostic_horizons requires "
            "--ca_repeat_diagnostic_max_repeats."
        )

    if args.ca_best_length is None:
        args.ca_best_length = args.ca_train_num_cells
    if args.ca_best_length not in args.ca_eval_num_cells:
        raise ValueError(
            "--ca_best_length must be included in --ca_eval_num_cells; got "
            f"{args.ca_best_length} versus {args.ca_eval_num_cells}."
        )
    if args.ca_diagnostic_length is None:
        args.ca_diagnostic_length = args.ca_best_length
    if args.ca_diagnostic_length not in args.ca_eval_num_cells:
        raise ValueError(
            "--ca_diagnostic_length must be included in --ca_eval_num_cells."
        )

    required_sequence_length = max(
        args.ca_train_num_cells,
        *args.ca_eval_num_cells,
    )
    if args.sequence_length < required_sequence_length:
        print_master(
            distributed_backend,
            f"Raising sequence_length from {args.sequence_length} to "
            f"{required_sequence_length} so every requested CA row fits.",
        )
        args.sequence_length = required_sequence_length

    if args.vocab_size != CA_VOCAB_SIZE:
        print_master(
            distributed_backend,
            f"Setting vocab_size to {CA_VOCAB_SIZE} for binary Rule 30 cells.",
        )
        args.vocab_size = CA_VOCAB_SIZE

    # Dataset names are also used to organize checkpoint/result directories.
    args.dataset = CA_TASK_NAME


def seed_global_training_rngs(args):
    """Seed the random sources currently used by the repository."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def main(args):
    """Build and train a Rule 30 experiment using the CA-specific runner."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    reporting_run_dir = None
    try:
        args = distributed_backend.get_adjusted_args_for_process(args)
        args.device = torch.device(args.device)
        if args.device.type == "cuda":
            torch.cuda.set_device(args.device)

        seed_global_training_rngs(args)
        apply_ca_task_config(args, distributed_backend)
        args.world_size = distributed_backend.get_world_size()
        checkpoint_dir = Path(
            args.results_base_folder,
            args.dataset,
            args.model,
            args.exp_name,
        )

        if args.ca_run_dir is not None and distributed_backend.is_master_process():
            reporting_run_dir = Path(args.ca_run_dir)
            manifest_path = reporting_run_dir / MANIFEST_FILENAME
            existing_manifest = (
                read_json(manifest_path) if manifest_path.is_file() else None
            )
            manifest = build_run_manifest(
                sanitize_for_json(vars(args)),
                reporting_run_dir,
                checkpoint_dir,
                status="running",
                existing=existing_manifest,
            )
            write_run_manifest(reporting_run_dir, manifest)
            ensure_notes_file(reporting_run_dir)

        if (checkpoint_dir / "summary.json").is_file():
            print_master(
                distributed_backend,
                f"Already found completed experiment '{checkpoint_dir}'. Skipping.",
            )
            if reporting_run_dir is not None:
                update_run_manifest_status(reporting_run_dir, "skipped")
            return
        if distributed_backend.is_master_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        distributed_backend.sync()

        train_loader, eval_loaders, test_loaders = make_ca_dataloaders(
            args, distributed_backend
        )

        model = models.make_model_from_args(args).to(args.device)
        model = distributed_backend.transform_model(model)
        optimizer = make_optimizer(
            args, model, distributed_backend, device_type=args.device.type
        )
        scheduler = make_scheduler(args, optimizer)
        raw_model = distributed_backend.get_raw_model(model)
        if (
            (
                args.ca_train_pairs
                or args.ca_extrapolation_val_pairs
                or args.ca_final_eval_pairs
                or args.ca_repeat_diagnostic_max_repeats
            )
            and "num_repeats" not in inspect.signature(raw_model.forward).parameters
        ):
            raise ValueError(
                "Variable CA training/evaluation pairs require a model whose "
                "forward method accepts num_repeats."
            )

        resume_path = resolve_resume_checkpoint(args, checkpoint_dir)
        if resume_path is not None and not resume_path.is_file():
            raise FileNotFoundError(f"CA checkpoint does not exist: {resume_path}")
        start_step, data_state = load_training_checkpoint(
            resume_path,
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            distributed_backend=distributed_backend,
            device=args.device,
        )
        if start_step > args.iterations:
            raise ValueError(
                f"Checkpoint step {start_step} exceeds --iterations={args.iterations}."
            )

        if distributed_backend.is_master_process() and args.wandb:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.exp_name,
                config=copy.deepcopy(sanitize_for_json(vars(args))),
                entity=args.wandb_entity,
            )

        print_master(distributed_backend, "Resolved Rule 30 experiment configuration:")
        print_master(
            distributed_backend,
            {
                "task": args.dataset,
                "train_num_cells": args.ca_train_num_cells,
                "eval_num_cells": args.ca_eval_num_cells,
                "ca_steps": args.ca_steps,
                "train_pairs": args.ca_train_pairs,
                "best_pair": args.ca_best_pair,
                "boundary": args.ca_boundary,
                "bernoulli_p": args.ca_bernoulli_p,
                "data_mode": args.ca_data_mode,
                "exact_data_resume": args.ca_exact_data_resume,
                "shuffle_seed": args.ca_shuffle_seed,
                "train_samples": args.ca_train_samples,
                "val_samples": args.ca_val_samples,
                "val_seed": args.ca_val_seed,
                "test_samples": args.ca_test_samples,
                "test_seed": args.ca_test_seed,
                "extrapolation_val_pairs": args.ca_extrapolation_val_pairs,
                "extrapolation_best_pair": args.ca_extrapolation_best_pair,
                "extrapolation_strict_min_id_cell_accuracy": (
                    args.ca_extrapolation_min_id_cell_accuracy
                ),
                "extrapolation_strict_min_id_exact_sequence_accuracy": (
                    args.ca_extrapolation_min_id_exact_sequence_accuracy
                ),
                "repeat_diagnostic_max_repeats": args.ca_repeat_diagnostic_max_repeats,
                "repeat_diagnostic_horizons": args.ca_repeat_diagnostic_horizons,
                "final_eval_pairs": args.ca_final_eval_pairs,
                "final_external_steps": args.ca_final_external_steps,
                "final_eval_max_batches": args.ca_final_eval_max_batches,
                "model_sequence_length": args.sequence_length,
                "vocab_size": args.vocab_size,
                "model": args.model,
                "n_repeat": args.n_repeat,
                "best_length": args.ca_best_length,
                "best_metric": args.ca_best_metric,
                "start_step": start_step,
                "device": str(args.device),
            },
        )
        print_master(
            distributed_backend,
            f"Training from step {start_step} ({len(train_loader)} training "
            f"batches per epoch; {len(eval_loaders)} fixed validation loaders; "
            f"pairs={args.ca_train_pairs or 'fixed'}).",
        )
        train_ca(
            model,
            optimizer,
            scheduler,
            train_loader,
            eval_loaders,
            test_loaders,
            args,
            distributed_backend,
            checkpoint_dir,
            start_step=start_step,
            data_state=data_state,
        )
        if distributed_backend.is_master_process() and args.wandb:
            import wandb

            wandb.finish()
        if reporting_run_dir is not None:
            update_run_manifest_status(reporting_run_dir, "completed")
    except BaseException as error:
        if reporting_run_dir is not None:
            try:
                update_run_manifest_status(
                    reporting_run_dir,
                    "failed",
                    error=error,
                )
            except Exception as reporting_error:
                print(
                    "WARNING: failed to record run failure in the manifest: "
                    f"{reporting_error}",
                    file=sys.stderr,
                )
        raise
    finally:
        distributed_backend.finalize()


if __name__ == "__main__":
    main(get_args())
