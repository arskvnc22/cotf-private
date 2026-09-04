"""Single-GPU entry point for the initial SoCA learnability dry run."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import config
import distributed
import models
from optim.runner_utils import make_optimizer, make_scheduler, sanitize_for_json

from cellular_automaton.ca_main import make_loader, seed_global_training_rngs
from cellular_automaton.soca_gen import MaterializedSOCADataset, SOCADataset
from cellular_automaton.soca_train import soca_train


SOCA_TASK_NAME = "soca_rule30_local"
SOCA_INPUT_VOCAB_SIZE = 4
SOCA_OUTPUT_VOCAB_SIZE = 2


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative integer.")
    return value


def get_args(argv=None):
    """Parse SoCA dry-run arguments together with shared model arguments."""
    format_parser = argparse.ArgumentParser(add_help=False)
    format_parser.add_argument(
        "--config_format",
        default="base",
        choices=config.registered_formats(),
    )
    format_args, _ = format_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format",
        default="base",
        choices=config.registered_formats(),
    )
    parser.add_argument("--soca_train_samples", type=_positive_int, default=65_536)
    parser.add_argument("--soca_val_samples", type=_positive_int, default=4_096)
    parser.add_argument("--soca_val_seed", type=int, default=1_000_003)
    parser.add_argument("--soca_schedule_seed", type=int, default=2_000_003)
    parser.add_argument("--soca_shuffle_seed", type=int, default=None)
    parser.add_argument("--soca_bernoulli_p", type=float, default=0.5)
    parser.add_argument(
        "--soca_schedule_protocol",
        choices=("separate_directions", "uniform_single_switch"),
        default="separate_directions",
        help=(
            "separate_directions preserves the original forward-only/reverse-only "
            "training distribution; uniform_single_switch samples every persistent "
            "first-reversal category uniformly."
        ),
    )
    parser.add_argument(
        "--soca_reversal_weights",
        type=float,
        nargs="+",
        default=None,
        metavar="WEIGHT",
        help=(
            "Weights for first reverse repeat 1..R+1. Values must match the "
            "selected soca_schedule_protocol."
        ),
    )
    parser.add_argument(
        "--soca_data_mode",
        choices=("materialized", "indexed"),
        default="materialized",
    )
    parser.add_argument("--soca_num_workers", type=_non_negative_int, default=0)
    parser.add_argument("--soca_eval_batch_size", type=_positive_int, default=None)
    parser.add_argument("--soca_eval_max_batches", type=_positive_int, default=None)
    parser.add_argument("--soca_copy_loss_weight", type=float, default=1.0)
    parser.add_argument("--soca_log_every", type=_positive_int, default=None)
    parser.add_argument("--soca_save_every", type=_positive_int, default=None)
    parser.add_argument("--soca_exact_data_resume", action="store_true")
    parser.add_argument(
        "--repeat_cache_window",
        type=_positive_int,
        default=None,
        help="Reserved for later cache experiments; must be omitted here.",
    )

    task_defaults = argparse.Namespace(
        model="but_soca",
        sequence_length=64,
        n_repeat=4,
        n_layer=4,
        n_head=4,
        n_embd=128,
        n_layer_begin=0,
        n_layer_end=0,
        attention_mode="bidirectional",
        dropout=0.0,
        use_pretrained=None,
    )
    return config.parse_args_with_format(
        format=format_args.config_format,
        base_parser=parser,
        args=argv,
        namespace=task_defaults,
    )


def resolve_soca_task_config(args):
    """Validate and freeze the scientific conditions of the dry run."""
    if args.distributed_backend is not None:
        raise ValueError("The initial SoCA learnability run is single GPU only.")
    if args.model != "but_soca":
        raise ValueError("The initial learnability run requires model='but_soca'.")
    if args.sequence_length <= 0 or args.sequence_length % 2 != 0:
        raise ValueError("sequence_length must be a positive even total length N.")
    if args.n_repeat <= 0:
        raise ValueError("n_repeat must be positive.")
    if args.n_layer_end != 0:
        raise ValueError("Per-repeat SoCA supervision requires n_layer_end == 0.")
    if args.attention_mode != "bidirectional":
        raise ValueError("The SoCA local-rule run requires bidirectional attention.")
    if args.repeat_cache_window is not None or args.lm_cache != "none":
        raise ValueError("The initial BUT learnability run does not use a cache.")
    if not 0.0 <= args.soca_bernoulli_p <= 1.0:
        raise ValueError("soca_bernoulli_p must lie in [0, 1].")
    if args.soca_copy_loss_weight != 1.0:
        raise ValueError(
            "The initial learnability run fixes soca_copy_loss_weight at 1.0."
        )
    if args.eval_freq <= 0 or args.iterations <= 0 or args.acc_steps <= 0:
        raise ValueError("eval_freq, iterations, and acc_steps must be positive.")
    if args.compile:
        raise ValueError("The initial SoCA dry run does not support --compile.")
    if args.soca_exact_data_resume:
        raise ValueError("Exact resume is outside the initial learnability run.")
    if args.use_pretrained not in {None, False, "None", "none"}:
        raise ValueError("Checkpoint resume is outside the initial learnability run.")

    if args.soca_reversal_weights is None:
        if args.soca_schedule_protocol == "separate_directions":
            args.soca_reversal_weights = (
                [1.0] + [0.0] * (args.n_repeat - 1) + [1.0]
            )
        else:
            args.soca_reversal_weights = [1.0] * (args.n_repeat + 1)
    else:
        args.soca_reversal_weights = [
            float(value) for value in args.soca_reversal_weights
        ]
    if len(args.soca_reversal_weights) != args.n_repeat + 1:
        raise ValueError(
            "soca_reversal_weights must have R+1 entries for first reverse "
            "repeat 1..R+1."
        )
    if any(value < 0 for value in args.soca_reversal_weights):
        raise ValueError("soca_reversal_weights cannot be negative.")
    if args.soca_reversal_weights[0] <= 0 or args.soca_reversal_weights[-1] <= 0:
        raise ValueError(
            "Both reverse-only and forward-only endpoint weights must be positive."
        )
    if args.soca_schedule_protocol == "separate_directions":
        if any(value != 0 for value in args.soca_reversal_weights[1:-1]):
            raise ValueError(
                "separate_directions permits only reverse-only and forward-only "
                "schedule weights."
            )
    else:
        first_weight = args.soca_reversal_weights[0]
        if any(
            value <= 0 or value != first_weight
            for value in args.soca_reversal_weights
        ):
            raise ValueError(
                "uniform_single_switch requires equal positive weight for every "
                "first-reversal category."
            )

    if args.soca_shuffle_seed is None:
        args.soca_shuffle_seed = int(args.data_seed)
    args.input_vocab_size = SOCA_INPUT_VOCAB_SIZE
    args.output_vocab_size = SOCA_OUTPUT_VOCAB_SIZE
    args.direction_embedding = True
    args.vocab_size = SOCA_OUTPUT_VOCAB_SIZE
    args.dataset = SOCA_TASK_NAME
    args.world_size = 1
    return args


def make_soca_dataloaders(args):
    """Build configured training and paired fixed validation conditions."""
    dataset_class = (
        MaterializedSOCADataset
        if args.soca_data_mode == "materialized"
        else SOCADataset
    )
    pin_memory = args.device.type == "cuda"
    train_dataset = dataset_class(
        num_samples=args.soca_train_samples,
        sequence_length=args.sequence_length,
        num_repeats=args.n_repeat,
        bernoulli_p=args.soca_bernoulli_p,
        seed=args.data_seed,
        schedule_seed=args.soca_schedule_seed,
        reversal_weights=args.soca_reversal_weights,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.soca_shuffle_seed,
        num_workers=args.soca_num_workers,
        pin_memory=pin_memory,
        distributed=False,
    )

    validation_batch_size = args.soca_eval_batch_size or args.batch_size
    common_validation = {
        "num_samples": args.soca_val_samples,
        "sequence_length": args.sequence_length,
        "num_repeats": args.n_repeat,
        "bernoulli_p": args.soca_bernoulli_p,
        "seed": args.soca_val_seed,
        "schedule_seed": args.soca_schedule_seed,
    }
    validation_datasets = {
        "forward_only": dataset_class(
            **common_validation,
            first_reverse_repeat=args.n_repeat + 1,
        ),
        "reverse_only": dataset_class(
            **common_validation,
            first_reverse_repeat=1,
        ),
    }
    if args.soca_schedule_protocol == "uniform_single_switch":
        validation_datasets.update(
            {
                f"switch_at_{first_reverse_repeat}": dataset_class(
                    **common_validation,
                    first_reverse_repeat=first_reverse_repeat,
                )
                for first_reverse_repeat in range(2, args.n_repeat + 1)
            }
        )
    validation_loaders = {
        condition: make_loader(
            dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            seed=args.soca_val_seed,
            num_workers=args.soca_num_workers,
            pin_memory=pin_memory,
            distributed=False,
        )
        for condition, dataset in validation_datasets.items()
    }
    return train_loader, validation_loaders


def main(args):
    """Build and run the single-GPU SoCA learnability experiment."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
    seed_global_training_rngs(args)
    resolve_soca_task_config(args)

    backend = distributed.make_backend_from_args(args)
    checkpoint_dir = Path(
        args.results_base_folder,
        args.dataset,
        args.model,
        args.exp_name,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loaders = make_soca_dataloaders(args)
    model = models.make_model_from_args(args).to(args.device)
    model = backend.transform_model(model)
    optimizer = make_optimizer(
        args,
        model,
        backend,
        device_type=args.device.type,
    )
    scheduler = make_scheduler(args, optimizer)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=copy.deepcopy(sanitize_for_json(vars(args))),
            entity=args.wandb_entity,
        )

    print("Resolved SoCA local-transition dry run:", flush=True)
    print(
        json.dumps(
            {
                "sequence_length": args.sequence_length,
                "component_row_length": args.sequence_length // 2,
                "num_repeats": args.n_repeat,
                "schedule_protocol": args.soca_schedule_protocol,
                "training_schedule_weights": args.soca_reversal_weights,
                "validation_conditions": list(validation_loaders),
                "train_samples": args.soca_train_samples,
                "validation_samples_per_condition": args.soca_val_samples,
                "copy_loss_weight": args.soca_copy_loss_weight,
                "input_vocab_size": args.input_vocab_size,
                "output_vocab_size": args.output_vocab_size,
                "attention_mode": args.attention_mode,
                "model": args.model,
                "device": str(args.device),
                "checkpoint_dir": str(checkpoint_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    try:
        return soca_train(
            model,
            optimizer,
            scheduler,
            train_loader,
            validation_loaders,
            None,
            args,
            backend,
            checkpoint_dir,
        )
    finally:
        if args.wandb:
            import wandb

            wandb.finish()
        backend.finalize()


if __name__ == "__main__":
    main(get_args())
