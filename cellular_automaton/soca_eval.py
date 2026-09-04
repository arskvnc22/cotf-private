"""Evaluation for reversible second-order Rule 30 models."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .ca_forward import CAForwardContext
from .soca_gen import FORWARD, REVERSE, soca_step
from .soca_train import encode_soca_inputs


@dataclass
class SOCABatchTrace:
    """Everything needed to score one evaluated batch.

    sequence_length is the total token length N. The previous and current
    component rows therefore each have length N / 2.
    """

    initial_state: torch.Tensor                 # [B, N]
    direction_schedule: torch.Tensor            # [B, R]
    target_steps_per_repeat: torch.Tensor        # [B, R]
    targets_by_repeat: torch.Tensor              # [B, R, N]
    repeat_logits: torch.Tensor                  # [B, R, N, 2]
    average_depth: torch.Tensor | float | None = None

    @property
    def predictions_by_repeat(self) -> torch.Tensor:
        return self.repeat_logits.argmax(dim=-1)


def normalize_target_steps(
    target_steps_per_repeat: int | Sequence[int] | torch.Tensor | None,
    direction_schedule: torch.Tensor,
) -> torch.Tensor:
    """Return non-negative semantic transition counts with shape [B, R].

    The direction is supplied separately by direction_schedule.

    Examples
    --------
    Ordinary SOCA:
        direction_schedule      = [F, F, R, R]
        target_steps_per_repeat = [1, 1, 1, 1]

    Three-step retrieval on the first reverse-conditioned loop:
        direction_schedule      = [F, F, F, R]
        target_steps_per_repeat = [1, 1, 1, 3]
    """
    if direction_schedule.ndim != 2:
        raise ValueError("direction_schedule must have shape [B, R].")

    batch_size, num_repeats = direction_schedule.shape
    device = direction_schedule.device

    if target_steps_per_repeat is None:
        return torch.ones(
            (batch_size, num_repeats),
            dtype=torch.long,
            device=device,
        )

    steps = torch.as_tensor(
        target_steps_per_repeat,
        dtype=torch.long,
        device=device,
    )

    if steps.ndim == 0:
        steps = steps.expand(batch_size, num_repeats)
    elif steps.ndim == 1:
        if steps.shape[0] != num_repeats:
            raise ValueError(
                "One-dimensional target_steps_per_repeat must have length R."
            )
        steps = steps.unsqueeze(0).expand(batch_size, -1)
    elif tuple(steps.shape) != (batch_size, num_repeats):
        raise ValueError(
            "target_steps_per_repeat must be scalar, [R], or [B, R]; "
            f"got {tuple(steps.shape)}."
        )

    if not bool(torch.all(steps >= 0).item()):
        raise ValueError("target_steps_per_repeat cannot contain negative values.")

    return steps


@torch.no_grad()
def rollout_soca_targets_by_model_repeat(
    initial_state: torch.Tensor,
    direction_schedule: torch.Tensor,
    target_steps_per_repeat: int | Sequence[int] | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one exact target after every model recurrent loop.

    A model loop may correspond to zero, one, or several exact SOCA
    transitions. Per-example transition counts are supported.

    Returns
    -------
    targets_by_repeat:
        Exact raw binary targets with shape [B, R, N].
    normalized_steps:
        The resolved semantic transition counts with shape [B, R].
    """
    if initial_state.ndim != 2:
        raise ValueError("initial_state must have shape [B, N].")
    if initial_state.shape[-1] % 2 != 0:
        raise ValueError("SOCA sequence length N must be even.")
    if direction_schedule.ndim != 2:
        raise ValueError("direction_schedule must have shape [B, R].")
    if direction_schedule.shape[0] != initial_state.shape[0]:
        raise ValueError("State and schedule batch sizes do not match.")
    if not bool(torch.all((initial_state == 0) | (initial_state == 1)).item()):
        raise ValueError("initial_state must contain only binary values.")
    if not bool(
        torch.all(
            (direction_schedule == FORWARD)
            | (direction_schedule == REVERSE)
        ).item()
    ):
        raise ValueError("direction_schedule must contain only FORWARD or REVERSE.")

    steps = normalize_target_steps(
        target_steps_per_repeat,
        direction_schedule,
    )

    current = initial_state
    targets = []

    for repeat_index in range(direction_schedule.shape[1]):
        direction = direction_schedule[:, repeat_index]
        step_counts = steps[:, repeat_index]
        maximum_steps = int(step_counts.max().item())

        for atomic_step in range(maximum_steps):
            transitioned = soca_step(current, direction)
            active = step_counts.gt(atomic_step).unsqueeze(-1)
            current = torch.where(active, transitioned, current)

        targets.append(current)

    return torch.stack(targets, dim=1), steps


@torch.no_grad()
def build_soca_batch_trace(
    forward_context: CAForwardContext,
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str,
    *,
    target_steps_per_repeat: int | Sequence[int] | torch.Tensor | None = None,
    model_forward_kwargs: Mapping[str, Any] | None = None,
    ctx=None,
) -> SOCABatchTrace:
    """Run one batch and return a validated, intervention-neutral trace."""
    model = forward_context.model
    device = torch.device(device)
    non_blocking = device.type == "cuda"

    initial_state = batch["state_sequence"].to(
        device,
        dtype=torch.long,
        non_blocking=non_blocking,
    )
    direction_schedule = batch["direction_schedule"].to(
        device,
        dtype=torch.long,
        non_blocking=non_blocking,
    )

    batch_target_steps = batch.get(
        "target_steps_per_repeat",
        target_steps_per_repeat,
    )
    targets_by_repeat, normalized_steps = (
        rollout_soca_targets_by_model_repeat(
            initial_state,
            direction_schedule,
            batch_target_steps,
        )
    )

    input_ids = encode_soca_inputs(
        initial_state,
        model.config.input_vocab_size,
    )

    if int(model.config.output_vocab_size) != 2:
        raise ValueError(
            "This evaluator expects the SOCA LM head to have two binary classes."
        )

    forward_kwargs = dict(model_forward_kwargs or {})
    reserved = {
        "direction_schedule",
        "num_repeats",
        "return_repeat_logits",
        "get_logits",
    }
    conflicts = reserved.intersection(forward_kwargs)
    if conflicts:
        raise ValueError(
            "model_forward_kwargs cannot override evaluator-owned arguments: "
            + ", ".join(sorted(conflicts))
        )

    forward_kwargs.update(
        {
            "direction_schedule": direction_schedule,
            "num_repeats": direction_schedule.shape[1],
            "return_repeat_logits": True,
            "get_logits": False,
        }
    )

    ctx = ctx or nullcontext()
    with ctx:
        outputs = forward_context.call(input_ids, **forward_kwargs)

    if not isinstance(outputs, Mapping):
        raise TypeError("SOCA models must return a mapping of outputs.")

    repeat_logits = outputs.get("repeat_logits")
    if repeat_logits is None:
        raise KeyError("SOCA model output is missing repeat_logits.")

    expected_shape = (
        initial_state.shape[0],
        direction_schedule.shape[1],
        initial_state.shape[1],
        2,
    )
    if tuple(repeat_logits.shape) != expected_shape:
        raise ValueError(
            "Expected repeat_logits with shape "
            f"{expected_shape}, got {tuple(repeat_logits.shape)}."
        )

    return SOCABatchTrace(
        initial_state=initial_state,
        direction_schedule=direction_schedule,
        target_steps_per_repeat=normalized_steps,
        targets_by_repeat=targets_by_repeat,
        repeat_logits=repeat_logits,
        average_depth=outputs.get("average_depth"),
    )


def _safe_ratio(numerator: int | float, denominator: int | float):
    if denominator == 0:
        return None
    return numerator / denominator


def _matthews_correlation(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / denominator


def _new_binary_counters() -> dict[str, Any]:
    return {
        "rows": 0,
        "exact_rows": 0,
        "positions": 0,
        "correct_positions": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "predicted_ones": 0,
        "target_ones": 0,
        "position_correct": None,
        "position_total": None,
    }


def _update_binary_counters(
    counters: dict[str, Any],
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if predictions.shape != targets.shape:
        raise ValueError("Binary predictions and targets do not align.")
    if predictions.ndim < 2:
        raise ValueError("Binary metric rows require at least two dimensions.")

    width = predictions.shape[-1]
    predictions = predictions.reshape(-1, width)
    targets = targets.reshape(-1, width)

    if predictions.shape[0] == 0:
        return

    correct = predictions.eq(targets)
    target_zero = targets.eq(0)
    target_one = targets.eq(1)
    predicted_zero = predictions.eq(0)
    predicted_one = predictions.eq(1)

    counters["rows"] += predictions.shape[0]
    counters["exact_rows"] += int(correct.all(dim=-1).sum().item())
    counters["positions"] += predictions.numel()
    counters["correct_positions"] += int(correct.sum().item())
    counters["tn"] += int((predicted_zero & target_zero).sum().item())
    counters["fp"] += int((predicted_one & target_zero).sum().item())
    counters["fn"] += int((predicted_zero & target_one).sum().item())
    counters["tp"] += int((predicted_one & target_one).sum().item())
    counters["predicted_ones"] += int(predicted_one.sum().item())
    counters["target_ones"] += int(target_one.sum().item())

    position_correct = correct.sum(dim=0).detach().cpu().to(torch.long)
    if counters["position_correct"] is None:
        counters["position_correct"] = torch.zeros(width, dtype=torch.long)
        counters["position_total"] = torch.zeros(width, dtype=torch.long)
    elif counters["position_correct"].numel() != width:
        raise ValueError("One metric group cannot mix different row lengths.")

    counters["position_correct"] += position_correct
    counters["position_total"] += predictions.shape[0]


def _finalize_binary_counters(counters: Mapping[str, Any]):
    if counters["rows"] == 0:
        return None

    position_accuracy = [
        _safe_ratio(int(correct), int(total))
        for correct, total in zip(
            counters["position_correct"],
            counters["position_total"],
        )
    ]

    target_zeros = counters["tn"] + counters["fp"]
    target_ones = counters["tp"] + counters["fn"]

    return {
        "cell_accuracy": _safe_ratio(
            counters["correct_positions"],
            counters["positions"],
        ),
        "exact_row_accuracy": _safe_ratio(
            counters["exact_rows"],
            counters["rows"],
        ),
        "matthews_correlation": _matthews_correlation(
            counters["tn"],
            counters["fp"],
            counters["fn"],
            counters["tp"],
        ),
        "zero_accuracy": _safe_ratio(counters["tn"], target_zeros),
        "one_accuracy": _safe_ratio(counters["tp"], target_ones),
        "target_one_rate": _safe_ratio(
            counters["target_ones"],
            counters["positions"],
        ),
        "predicted_one_rate": _safe_ratio(
            counters["predicted_ones"],
            counters["positions"],
        ),
        "correct_cells": counters["correct_positions"],
        "total_cells": counters["positions"],
        "exact_rows": counters["exact_rows"],
        "total_rows": counters["rows"],
        "total_bit_errors": (
            counters["positions"] - counters["correct_positions"]
        ),
        "position_accuracy": position_accuracy,
    }


def _new_metric_group() -> dict[str, Any]:
    return {
        "state": _new_binary_counters(),
        "previous": _new_binary_counters(),
        "current": _new_binary_counters(),
        "computed": _new_binary_counters(),
        "copied": _new_binary_counters(),
        "transitions": 0,
        "unit_transitions": 0,
        "joint_pair_cells": 0,
        "correct_joint_pair_cells": 0,
        "exact_pair_rows": 0,
        "loss_sum": 0.0,
        "loss_positions": 0,
        "computed_loss_sum": 0.0,
        "computed_loss_positions": 0,
        "copied_loss_sum": 0.0,
        "copied_loss_positions": 0,
        "average_depth_sum": 0.0,
        "average_depth_count": 0,
    }


def _update_metric_group(
    group: dict[str, Any],
    trace: SOCABatchTrace,
    selection: torch.Tensor,
) -> None:
    """Update one slice selected over [B, R].

    Computed/copied metrics are only defined when a model repeat represents
    exactly one atomic SOCA transition. Multi-step jump targets are still
    included in state, previous, current and joint-pair metrics.
    """
    if selection.shape != trace.direction_schedule.shape:
        raise ValueError("Metric selection must have shape [B, R].")
    if not bool(selection.any().item()):
        return

    logits = trace.repeat_logits[selection]                    # [M, N, 2]
    targets = trace.targets_by_repeat[selection]               # [M, N]
    predictions = logits.argmax(dim=-1)                        # [M, N]
    directions = trace.direction_schedule[selection]           # [M]
    target_steps = trace.target_steps_per_repeat[selection]     # [M]

    sequence_length = predictions.shape[-1]
    if sequence_length % 2 != 0:
        raise ValueError("SOCA sequence length must be even.")
    row_length = sequence_length // 2

    predicted_previous = predictions[:, :row_length]
    predicted_current = predictions[:, row_length:]
    target_previous = targets[:, :row_length]
    target_current = targets[:, row_length:]

    _update_binary_counters(group["state"], predictions, targets)
    _update_binary_counters(
        group["previous"],
        predicted_previous,
        target_previous,
    )
    _update_binary_counters(
        group["current"],
        predicted_current,
        target_current,
    )

    pair_correct = (
        predicted_previous.eq(target_previous)
        & predicted_current.eq(target_current)
    )
    group["transitions"] += predictions.shape[0]
    group["joint_pair_cells"] += pair_correct.numel()
    group["correct_joint_pair_cells"] += int(pair_correct.sum().item())
    group["exact_pair_rows"] += int(pair_correct.all(dim=-1).sum().item())

    element_loss = F.cross_entropy(
        logits.reshape(-1, 2),
        targets.reshape(-1),
        reduction="none",
    ).reshape(predictions.shape[0], sequence_length)

    group["loss_sum"] += float(element_loss.sum().item())
    group["loss_positions"] += element_loss.numel()

    unit_transition = target_steps.eq(1)
    if bool(unit_transition.any().item()):
        unit_previous_prediction = predicted_previous[unit_transition]
        unit_current_prediction = predicted_current[unit_transition]
        unit_previous_target = target_previous[unit_transition]
        unit_current_target = target_current[unit_transition]
        unit_direction = directions[unit_transition].bool()

        computed_prediction = torch.where(
            unit_direction[:, None],
            unit_previous_prediction,
            unit_current_prediction,
        )
        computed_target = torch.where(
            unit_direction[:, None],
            unit_previous_target,
            unit_current_target,
        )
        copied_prediction = torch.where(
            unit_direction[:, None],
            unit_current_prediction,
            unit_previous_prediction,
        )
        copied_target = torch.where(
            unit_direction[:, None],
            unit_current_target,
            unit_previous_target,
        )

        _update_binary_counters(
            group["computed"],
            computed_prediction,
            computed_target,
        )
        _update_binary_counters(
            group["copied"],
            copied_prediction,
            copied_target,
        )

        unit_loss = element_loss[unit_transition]
        previous_loss = unit_loss[:, :row_length]
        current_loss = unit_loss[:, row_length:]

        computed_loss = torch.where(
            unit_direction[:, None],
            previous_loss,
            current_loss,
        )
        copied_loss = torch.where(
            unit_direction[:, None],
            current_loss,
            previous_loss,
        )

        group["unit_transitions"] += int(unit_transition.sum().item())
        group["computed_loss_sum"] += float(computed_loss.sum().item())
        group["computed_loss_positions"] += computed_loss.numel()
        group["copied_loss_sum"] += float(copied_loss.sum().item())
        group["copied_loss_positions"] += copied_loss.numel()

    if trace.average_depth is not None:
        depth = float(
            torch.as_tensor(trace.average_depth)
            .detach()
            .cpu()
            .float()
            .item()
        )
        selected_examples = int(selection.any(dim=1).sum().item())
        group["average_depth_sum"] += depth * selected_examples
        group["average_depth_count"] += selected_examples


def _finalize_metric_group(
    group: Mapping[str, Any],
    *,
    copy_loss_weight: float,
) -> dict[str, Any]:
    computed_loss = _safe_ratio(
        group["computed_loss_sum"],
        group["computed_loss_positions"],
    )
    copied_loss = _safe_ratio(
        group["copied_loss_sum"],
        group["copied_loss_positions"],
    )

    # This reproduces the configured computed + lambda_copy * copied loss only
    # when every selected target is an atomic one-step transition.
    configured_loss = None
    if (
        group["transitions"] > 0
        and group["unit_transitions"] == group["transitions"]
        and computed_loss is not None
        and copied_loss is not None
    ):
        configured_loss = computed_loss + copy_loss_weight * copied_loss

    return {
        "state": _finalize_binary_counters(group["state"]),
        "previous": _finalize_binary_counters(group["previous"]),
        "current": _finalize_binary_counters(group["current"]),
        "computed": _finalize_binary_counters(group["computed"]),
        "copied": _finalize_binary_counters(group["copied"]),
        "joint_pair_cell_accuracy": _safe_ratio(
            group["correct_joint_pair_cells"],
            group["joint_pair_cells"],
        ),
        "joint_exact_pair_row_accuracy": _safe_ratio(
            group["exact_pair_rows"],
            group["transitions"],
        ),
        "cross_entropy_loss": _safe_ratio(
            group["loss_sum"],
            group["loss_positions"],
        ),
        "computed_loss": computed_loss,
        "copied_loss": copied_loss,
        "configured_loss": configured_loss,
        "transitions": group["transitions"],
        "unit_transitions": group["unit_transitions"],
        "average_depth": _safe_ratio(
            group["average_depth_sum"],
            group["average_depth_count"],
        ),
    }


def new_soca_eval_counters() -> dict[str, Any]:
    return {
        "num_batches": 0,
        "examples": 0,
        "overall": _new_metric_group(),
        "by_repeat": {},
        "by_direction": {},
        "by_transition": {},
        "by_schedule_class": {},
        "by_reversal_repeat": {},
        "by_reversal_offset": {},
    }


def _group(groups: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in groups:
        groups[key] = _new_metric_group()
    return groups[key]


def _scored_repeat_mask(
    direction_schedule: torch.Tensor,
    scored_repeats: Sequence[int] | None,
) -> torch.Tensor:
    batch_size, num_repeats = direction_schedule.shape

    if scored_repeats is None:
        return torch.ones(
            (batch_size, num_repeats),
            dtype=torch.bool,
            device=direction_schedule.device,
        )

    mask = torch.zeros(
        (batch_size, num_repeats),
        dtype=torch.bool,
        device=direction_schedule.device,
    )
    for repeat in scored_repeats:
        repeat = int(repeat)
        if repeat < 1 or repeat > num_repeats:
            raise ValueError(
                f"Scored repeat {repeat} lies outside [1, {num_repeats}]."
            )
        mask[:, repeat - 1] = True
    return mask


def update_soca_eval_counters(
    counters: dict[str, Any],
    trace: SOCABatchTrace,
    *,
    scored_repeats: Sequence[int] | None = None,
) -> None:
    """Accumulate overall and interpretable schedule-relative slices."""
    schedule = trace.direction_schedule
    batch_size, num_repeats = schedule.shape
    score_mask = _scored_repeat_mask(schedule, scored_repeats)

    counters["num_batches"] += 1
    counters["examples"] += batch_size

    _update_metric_group(counters["overall"], trace, score_mask)

    for repeat_index in range(num_repeats):
        selection = torch.zeros_like(score_mask)
        selection[:, repeat_index] = score_mask[:, repeat_index]
        _update_metric_group(
            _group(counters["by_repeat"], f"repeat_{repeat_index + 1}"),
            trace,
            selection,
        )

    for direction, label in (
        (FORWARD, "forward"),
        (REVERSE, "reverse"),
    ):
        selection = score_mask & schedule.eq(direction)
        _update_metric_group(
            _group(counters["by_direction"], label),
            trace,
            selection,
        )

        selected_steps = trace.target_steps_per_repeat[selection]
        for step_count in torch.unique(selected_steps).tolist():
            transition_key = f"{label}_steps_{int(step_count)}"
            step_selection = (
                selection
                & trace.target_steps_per_repeat.eq(int(step_count))
            )
            _update_metric_group(
                _group(counters["by_transition"], transition_key),
                trace,
                step_selection,
            )

    forward_only = schedule.eq(FORWARD).all(dim=1)
    reverse_only = schedule.eq(REVERSE).all(dim=1)
    mixed = ~(forward_only | reverse_only)

    for example_mask, label in (
        (forward_only, "forward_only"),
        (reverse_only, "reverse_only"),
        (mixed, "mixed"),
    ):
        selection = score_mask & example_mask[:, None]
        _update_metric_group(
            _group(counters["by_schedule_class"], label),
            trace,
            selection,
        )

    positions = torch.arange(
        1,
        num_repeats + 1,
        device=schedule.device,
    ).expand_as(schedule)
    no_reversal = torch.full_like(positions, num_repeats + 1)
    first_reverse = torch.where(
        schedule.eq(REVERSE),
        positions,
        no_reversal,
    ).amin(dim=1)

    for value in torch.unique(first_reverse).tolist():
        key = "none" if value == num_repeats + 1 else str(int(value))
        selection = score_mask & first_reverse.eq(value)[:, None]
        _update_metric_group(
            _group(counters["by_reversal_repeat"], key),
            trace,
            selection,
        )

    has_reversal = first_reverse.le(num_repeats)
    reversal_offset = positions - first_reverse[:, None]
    valid_offset_mask = score_mask & has_reversal[:, None]

    for offset in torch.unique(
        reversal_offset[valid_offset_mask]
    ).tolist():
        selection = (
            valid_offset_mask
            & reversal_offset.eq(int(offset))
        )
        _update_metric_group(
            _group(
                counters["by_reversal_offset"],
                f"offset_{int(offset)}",
            ),
            trace,
            selection,
        )


def finalize_soca_eval_counters(
    counters: Mapping[str, Any],
    *,
    copy_loss_weight: float,
) -> dict[str, Any]:
    def finalize_groups(groups):
        return {
            key: _finalize_metric_group(
                group,
                copy_loss_weight=copy_loss_weight,
            )
            for key, group in groups.items()
            if group["transitions"] > 0
        }

    return {
        "num_batches": counters["num_batches"],
        "examples": counters["examples"],
        "overall": _finalize_metric_group(
            counters["overall"],
            copy_loss_weight=copy_loss_weight,
        ),
        "by_repeat": finalize_groups(counters["by_repeat"]),
        "by_direction": finalize_groups(counters["by_direction"]),
        "by_transition": finalize_groups(counters["by_transition"]),
        "by_schedule_class": finalize_groups(
            counters["by_schedule_class"]
        ),
        "by_reversal_repeat": finalize_groups(
            counters["by_reversal_repeat"]
        ),
        "by_reversal_offset": finalize_groups(
            counters["by_reversal_offset"]
        ),
    }


@torch.no_grad()
def evaluate_soca_model(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    copy_loss_weight: float,
    target_steps_per_repeat: int | Sequence[int] | torch.Tensor | None = None,
    scored_repeats: Sequence[int] | None = None,
    max_batches: int | None = None,
    model_forward_kwargs: Mapping[str, Any] | None = None,
    trace_builder: Callable[..., SOCABatchTrace] = build_soca_batch_trace,
    ctx=None,
) -> dict[str, Any]:
    """Evaluate one fixed SoCA condition and restore the prior model mode.

    trace_builder is injectable so future intervention evaluators can use the
    same counters and reporting without modifying this loop.
    """
    model = forward_context.model
    was_training = model.training
    model.eval()
    counters = new_soca_eval_counters()
    ctx = ctx or nullcontext()

    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            trace = trace_builder(
                forward_context,
                batch,
                device,
                target_steps_per_repeat=target_steps_per_repeat,
                model_forward_kwargs=model_forward_kwargs,
                ctx=ctx,
            )
            update_soca_eval_counters(
                counters,
                trace,
                scored_repeats=scored_repeats,
            )
    finally:
        if was_training:
            model.train()

    if counters["num_batches"] == 0:
        raise ValueError("SOCA evaluation consumed no batches.")
    if counters["overall"]["transitions"] == 0:
        raise ValueError("SOCA evaluation scored no model repeats.")

    result = finalize_soca_eval_counters(
        counters,
        copy_loss_weight=copy_loss_weight,
    )
    result["scored_repeats"] = (
        None
        if scored_repeats is None
        else [int(value) for value in scored_repeats]
    )
    result["target_step_semantics"] = (
        "number of exact SOCA transitions represented by each model repeat"
    )
    result["computed_copied_scope"] = (
        "Only repeats with target_steps_per_repeat == 1. "
        "Multi-step jump targets have no direct one-step copied/computed role."
    )
    return result


def evaluate_soca_lengths(
    forward_context: CAForwardContext,
    eval_loaders_by_sequence_length,
    device,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate each total sequence length N independently."""
    results = {}

    for sequence_length, dataloader in (
        eval_loaders_by_sequence_length.items()
    ):
        sequence_length = int(sequence_length)
        if sequence_length % 2 != 0:
            raise ValueError("Every SOCA sequence length must be even.")

        metrics = evaluate_soca_model(
            forward_context,
            dataloader,
            device,
            **kwargs,
        )
        metrics["sequence_length"] = sequence_length
        metrics["component_row_length"] = sequence_length // 2
        results[str(sequence_length)] = metrics

    return results