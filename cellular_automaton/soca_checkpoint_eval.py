"""Evaluate a frozen SoCA checkpoint on switches and repeat extrapolation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import models
from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_main import make_loader, seed_global_training_rngs
from cellular_automaton.ca_reporting import write_json
from cellular_automaton.soca_eval import evaluate_soca_model
from cellular_automaton.soca_gen import MaterializedSOCADataset
from cellular_automaton.soca_main import get_args, resolve_soca_task_config


@dataclass(frozen=True)
class SOCAEvaluationCondition:
    """One persistent-direction schedule evaluated at one repeat horizon."""

    horizon: int
    first_reverse_repeat: int

    @property
    def schedule_class(self):
        if self.first_reverse_repeat == 1:
            return "reverse_only"
        if self.first_reverse_repeat == self.horizon + 1:
            return "forward_only"
        return "forward_then_reverse"

    @property
    def key(self):
        if self.schedule_class == "forward_then_reverse":
            label = f"switch_at_{self.first_reverse_repeat}"
        else:
            label = self.schedule_class
        return f"horizon_{self.horizon}/{label}"

    @property
    def schedule(self):
        return [
            0 if repeat < self.first_reverse_repeat else 1
            for repeat in range(1, self.horizon + 1)
        ]

    @property
    def schedule_text(self):
        return "".join("R" if value else "F" for value in self.schedule)

    def metadata(self):
        return {
            **asdict(self),
            "key": self.key,
            "schedule_class": self.schedule_class,
            "schedule": self.schedule,
            "schedule_text": self.schedule_text,
        }


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--horizons",
        type=_positive_int,
        nargs="+",
        default=[4, 8, 12],
    )
    parser.add_argument("--trained_repeats", type=_positive_int, default=4)
    parser.add_argument("--eval_samples", type=_positive_int, default=4096)
    parser.add_argument("--eval_batch_size", type=_positive_int, default=128)
    parser.add_argument("--eval_max_batches", type=_positive_int, default=None)
    parser.add_argument("--eval_num_workers", type=_non_negative_int, default=4)
    parser.add_argument("--eval_seed", type=int, default=1_000_003)
    parser.add_argument("--bernoulli_p", type=float, default=0.5)
    parser.add_argument("--copy_loss_weight", type=float, default=1.0)

    parser.add_argument("--n_layer", type=_positive_int, default=4)
    parser.add_argument("--n_layer_begin", type=_non_negative_int, default=0)
    parser.add_argument("--n_layer_end", type=_non_negative_int, default=0)
    parser.add_argument("--n_embd", type=_positive_int, default=128)
    parser.add_argument("--n_head", type=_positive_int, default=4)
    parser.add_argument("--sequence_length", type=_positive_int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--positional_encoder", default="rotary")
    parser.add_argument("--attention_mode", default="bidirectional")
    parser.add_argument("--dtype", default=None)
    return parser.parse_args(argv)


def build_soca_evaluation_conditions(
    horizons: Sequence[int],
) -> list[SOCAEvaluationCondition]:
    """Return every persistent single-switch schedule at each horizon."""
    normalized = [int(horizon) for horizon in horizons]
    if not normalized or any(horizon <= 0 for horizon in normalized):
        raise ValueError("At least one positive evaluation horizon is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Evaluation horizons must be unique.")

    return [
        SOCAEvaluationCondition(horizon, first_reverse_repeat)
        for horizon in normalized
        for first_reverse_repeat in range(1, horizon + 2)
    ]


def make_soca_condition_dataset(
    condition: SOCAEvaluationCondition,
    *,
    num_samples: int,
    sequence_length: int,
    bernoulli_p: float,
    seed: int,
):
    """Materialize one condition; equal seeds give paired initial states."""
    return MaterializedSOCADataset(
        num_samples=num_samples,
        sequence_length=sequence_length,
        num_repeats=condition.horizon,
        bernoulli_p=bernoulli_p,
        seed=seed,
        schedule_seed=seed,
        first_reverse_repeat=condition.first_reverse_repeat,
    )


def make_soca_condition_loader(condition, args):
    dataset = make_soca_condition_dataset(
        condition,
        num_samples=args.eval_samples,
        sequence_length=args.sequence_length,
        bernoulli_p=args.bernoulli_p,
        seed=args.eval_seed,
    )
    return make_loader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.eval_seed,
        num_workers=args.eval_num_workers,
        pin_memory=args.device.type == "cuda",
        distributed=False,
    )


def build_model_args(args):
    """Reconstruct the training architecture using the normal SoCA parser."""
    argv = [
        "--device",
        str(args.device),
        "--model",
        "but_soca",
        "--attention_mode",
        args.attention_mode,
        "--positional_encoder",
        args.positional_encoder,
        "--n_layer",
        str(args.n_layer),
        "--n_repeat",
        str(args.trained_repeats),
        "--n_layer_begin",
        str(args.n_layer_begin),
        "--n_layer_end",
        str(args.n_layer_end),
        "--n_embd",
        str(args.n_embd),
        "--n_head",
        str(args.n_head),
        "--sequence_length",
        str(args.sequence_length),
        "--dropout",
        str(args.dropout),
        "--seed",
        str(args.model_seed),
        "--data_seed",
        str(args.eval_seed),
        "--batch_size",
        str(args.eval_batch_size),
        "--iterations",
        "1",
        "--eval_freq",
        "1",
        "--soca_copy_loss_weight",
        str(args.copy_loss_weight),
        "--use_pretrained",
        "None",
    ]
    if args.dtype is not None:
        argv.extend(("--dtype", args.dtype))
    model_args = get_args(argv)
    model_args.device = torch.device(model_args.device)
    return resolve_soca_task_config(model_args)


def load_soca_checkpoint(model, checkpoint_path, device):
    """Strictly load a resumable SoCA training checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise ValueError(f"SOCA checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SOCA checkpoint must contain a mapping.")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("SOCA checkpoint has no model state dictionary.")
    step = checkpoint.get("itr")
    if hasattr(step, "item"):
        step = step.item()
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise ValueError("SOCA checkpoint must contain an integer itr.")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return {
        "path": str(checkpoint_path.resolve()),
        "step": int(step),
    }


def _autocast_context(args):
    if args.device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(
        device_type=args.device.type,
        dtype=args.dtype,
    )


def _compact_metric_group(group):
    if group is None:
        return None

    def binary(component):
        values = group.get(component)
        if values is None:
            return None
        return {
            "cell_accuracy": values["cell_accuracy"],
            "exact_row_accuracy": values["exact_row_accuracy"],
            "total_bit_errors": values["total_bit_errors"],
        }

    return {
        "state": binary("state"),
        "computed": binary("computed"),
        "copied": binary("copied"),
        "joint_pair_cell_accuracy": group["joint_pair_cell_accuracy"],
        "joint_exact_pair_row_accuracy": group[
            "joint_exact_pair_row_accuracy"
        ],
        "cross_entropy_loss": group["cross_entropy_loss"],
        "configured_loss": group["configured_loss"],
        "transitions": group["transitions"],
    }


def compact_condition_summary(condition, metrics, trained_repeats):
    by_repeat = {
        key: _compact_metric_group(values)
        for key, values in metrics["by_repeat"].items()
    }
    extrapolated = {
        key: values
        for key, values in by_repeat.items()
        if int(key.removeprefix("repeat_")) > int(trained_repeats)
    }
    switch_key = (
        f"repeat_{condition.first_reverse_repeat}"
        if condition.schedule_class == "forward_then_reverse"
        else None
    )
    return {
        **condition.metadata(),
        "trained_repeats": int(trained_repeats),
        "is_repeat_extrapolation": condition.horizon > int(trained_repeats),
        "overall": _compact_metric_group(metrics["overall"]),
        "switch_repeat_metrics": (
            by_repeat.get(switch_key) if switch_key is not None else None
        ),
        "by_repeat": by_repeat,
        "extrapolated_repeats": extrapolated,
        "by_reversal_offset": {
            key: _compact_metric_group(values)
            for key, values in metrics["by_reversal_offset"].items()
        },
    }


@torch.no_grad()
def evaluate_soca_protocol(forward_context, conditions, args):
    """Evaluate all conditions using paired deterministic initial states."""
    full_metrics = {}
    summaries = []
    for condition_index, condition in enumerate(conditions, start=1):
        print(
            f"[{condition_index}/{len(conditions)}] {condition.key} "
            f"schedule={condition.schedule_text}",
            flush=True,
        )
        loader = make_soca_condition_loader(condition, args)
        metrics = evaluate_soca_model(
            forward_context,
            loader,
            args.device,
            copy_loss_weight=args.copy_loss_weight,
            target_steps_per_repeat=1,
            max_batches=args.eval_max_batches,
            ctx=_autocast_context(args),
        )
        full_metrics[condition.key] = {
            "condition": condition.metadata(),
            "metrics": metrics,
        }
        summary = compact_condition_summary(
            condition,
            metrics,
            args.trained_repeats,
        )
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "condition": condition.key,
                    "overall": summary["overall"],
                    "switch_repeat_metrics": summary[
                        "switch_repeat_metrics"
                    ],
                }
            ),
            flush=True,
        )
    return full_metrics, summaries


def _protocol_metadata(args, conditions):
    return {
        "hypotheses": [
            "zero-shot composition of separately trained forward and reverse operators",
            "recurrent-depth extrapolation beyond the trained repeat count",
        ],
        "trained_repeats": int(args.trained_repeats),
        "evaluation_horizons": [int(value) for value in args.horizons],
        "num_conditions": len(conditions),
        "schedule_semantics": (
            "persistent forward-then-reverse; reversal is applied on "
            "first_reverse_repeat"
        ),
        "paired_initial_states": True,
        "target_steps_per_model_repeat": 1,
        "eval_samples_per_condition": int(args.eval_samples),
        "eval_seed": int(args.eval_seed),
        "bernoulli_p": float(args.bernoulli_p),
        "sequence_length": int(args.sequence_length),
        "component_row_length": int(args.sequence_length // 2),
        "input_vocab_size": 4,
        "output_vocab_size": 2,
        "cache_enabled": False,
        "parameter_updates": False,
    }


def run_standalone_evaluation(args):
    if not 0.0 <= args.bernoulli_p <= 1.0:
        raise ValueError("bernoulli_p must lie in [0, 1].")
    if args.copy_loss_weight != 1.0:
        raise ValueError("This protocol keeps copy_loss_weight fixed at 1.0.")
    if args.sequence_length % 2 != 0:
        raise ValueError("sequence_length must be even.")

    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
    model_args = build_model_args(args)
    args.dtype = model_args.dtype
    seed_global_training_rngs(model_args)

    model = models.make_model_from_args(model_args)
    checkpoint_metadata = load_soca_checkpoint(
        model,
        args.checkpoint,
        args.device,
    )
    forward_context = CAForwardContext(
        model,
        CAForwardPolicy.from_args(model_args),
    )
    conditions = build_soca_evaluation_conditions(args.horizons)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Evaluation output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    protocol = _protocol_metadata(args, conditions)
    full_report = {
        "schema_version": 1,
        "status": "running",
        "checkpoint": checkpoint_metadata,
        "protocol": protocol,
        "conditions": {},
    }
    full_path = output_dir / "full_metrics.json"
    summary_path = output_dir / "summary.json"
    write_json(full_path, full_report)

    try:
        full_metrics, summaries = evaluate_soca_protocol(
            forward_context,
            conditions,
            args,
        )
        full_report["conditions"] = full_metrics
        full_report["status"] = "completed"
        summary_report = {
            "schema_version": 1,
            "status": "completed",
            "checkpoint": checkpoint_metadata,
            "protocol": protocol,
            "conditions": summaries,
        }
        write_json(full_path, full_report)
        write_json(summary_path, summary_report)
    except BaseException as error:
        full_report["status"] = "failed"
        full_report["error"] = f"{type(error).__name__}: {error}"
        write_json(full_path, full_report)
        raise

    return {
        "checkpoint": checkpoint_metadata,
        "conditions": len(conditions),
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "full_metrics": str(full_path),
    }


def main(argv=None):
    result = run_standalone_evaluation(parse_args(argv))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
