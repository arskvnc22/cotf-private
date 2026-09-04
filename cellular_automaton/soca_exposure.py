"""Pure bookkeeping helpers for SoCA training exposure."""

from __future__ import annotations

import copy

import torch

from .soca_gen import FORWARD, REVERSE


def _repeat_key(repeat_index):
    return f"repeat_{int(repeat_index)}"


def _new_repeat_exposure(repeat_index, num_repeats):
    return {
        "repeat": int(repeat_index),
        "repeat_loss_weight": 1.0 / int(num_repeats),
        "supervised_states": 0,
        "state_positions": 0,
        "forward_transitions": 0,
        "reverse_transitions": 0,
        "computed_cells": 0,
        "copied_cells": 0,
        "forward_fraction": 0.0,
        "reverse_fraction": 0.0,
    }


def new_soca_step_exposure(num_repeats):
    """Create local exact counters for one optimizer step."""
    num_repeats = int(num_repeats)
    if num_repeats <= 0:
        raise ValueError("num_repeats must be positive.")
    return {
        "num_repeats": num_repeats,
        "sequence_length": None,
        "microbatches": 0,
        "examples": 0,
        "supervised_states": 0,
        "state_positions": 0,
        "computed_cells": 0,
        "copied_cells": 0,
        "forward_only_examples": 0,
        "reverse_only_examples": 0,
        "switch_examples": 0,
        "by_repeat": {
            _repeat_key(repeat_index): _new_repeat_exposure(
                repeat_index, num_repeats
            )
            for repeat_index in range(1, num_repeats + 1)
        },
    }


@torch.no_grad()
def update_soca_step_exposure(exposure, state_sequence, direction_schedule):
    """Add one consumed microbatch to local exposure counters."""
    if state_sequence.ndim != 2:
        raise ValueError("state_sequence must have shape [B, N].")
    if direction_schedule.ndim != 2:
        raise ValueError("direction_schedule must have shape [B, R].")
    if state_sequence.shape[0] != direction_schedule.shape[0]:
        raise ValueError("State and schedule batch sizes do not match.")
    if state_sequence.shape[1] % 2 != 0:
        raise ValueError("SOCA sequence length must be even.")
    if direction_schedule.shape[1] != exposure["num_repeats"]:
        raise ValueError("Schedule depth disagrees with exposure depth.")
    if not bool(
        torch.all(
            direction_schedule.eq(FORWARD)
            | direction_schedule.eq(REVERSE)
        ).item()
    ):
        raise ValueError("Direction schedules must contain only FORWARD or REVERSE.")

    batch_size, sequence_length = state_sequence.shape
    row_length = sequence_length // 2
    stored_length = exposure["sequence_length"]
    if stored_length is None:
        exposure["sequence_length"] = int(sequence_length)
    elif int(stored_length) != int(sequence_length):
        raise ValueError("One exposure record cannot mix sequence lengths.")

    forward_only = direction_schedule.eq(FORWARD).all(dim=1)
    reverse_only = direction_schedule.eq(REVERSE).all(dim=1)
    switch = ~(forward_only | reverse_only)

    exposure["microbatches"] += 1
    exposure["examples"] += int(batch_size)
    exposure["supervised_states"] += int(batch_size * exposure["num_repeats"])
    exposure["state_positions"] += int(
        batch_size * exposure["num_repeats"] * sequence_length
    )
    exposure["computed_cells"] += int(
        batch_size * exposure["num_repeats"] * row_length
    )
    exposure["copied_cells"] += int(
        batch_size * exposure["num_repeats"] * row_length
    )
    exposure["forward_only_examples"] += int(forward_only.sum().item())
    exposure["reverse_only_examples"] += int(reverse_only.sum().item())
    exposure["switch_examples"] += int(switch.sum().item())

    for repeat_offset in range(exposure["num_repeats"]):
        values = exposure["by_repeat"][_repeat_key(repeat_offset + 1)]
        directions = direction_schedule[:, repeat_offset]
        values["supervised_states"] += int(batch_size)
        values["state_positions"] += int(batch_size * sequence_length)
        values["forward_transitions"] += int(
            directions.eq(FORWARD).sum().item()
        )
        values["reverse_transitions"] += int(
            directions.eq(REVERSE).sum().item()
        )
        values["computed_cells"] += int(batch_size * row_length)
        values["copied_cells"] += int(batch_size * row_length)
    return _refresh_exposure(exposure)


def _refresh_exposure(exposure):
    materialized_rows = int(exposure.get("materialized_training_rows", 0))
    examples = int(
        exposure.get("total_examples_seen", exposure.get("examples", 0))
    )
    if "equivalent_dataset_passes" in exposure:
        exposure["equivalent_dataset_passes"] = (
            examples / materialized_rows if materialized_rows else 0.0
        )

    for values in exposure["by_repeat"].values():
        transitions = (
            int(values["forward_transitions"])
            + int(values["reverse_transitions"])
        )
        values["forward_fraction"] = (
            values["forward_transitions"] / transitions if transitions else 0.0
        )
        values["reverse_fraction"] = (
            values["reverse_transitions"] / transitions if transitions else 0.0
        )
    return exposure


def new_soca_training_exposure(
    *,
    materialized_training_rows,
    num_repeats,
    copy_loss_weight,
):
    """Create cumulative exact exposure accounting for one fixed-depth run."""
    step = new_soca_step_exposure(num_repeats)
    return {
        "accounting_exact": True,
        "materialized_training_rows": int(materialized_training_rows),
        "num_repeats": int(num_repeats),
        "copy_loss_weight": float(copy_loss_weight),
        "repeat_loss_aggregation": "equal_mean_across_repeats",
        "sequence_length": None,
        "total_optimizer_steps": 0,
        "total_microbatches": 0,
        "total_examples_seen": 0,
        "total_supervised_states": 0,
        "total_state_positions": 0,
        "total_computed_cells": 0,
        "total_copied_cells": 0,
        "forward_only_examples": 0,
        "reverse_only_examples": 0,
        "switch_examples": 0,
        "equivalent_dataset_passes": 0.0,
        "by_repeat": step["by_repeat"],
    }


def add_soca_training_exposure(cumulative, step_exposure):
    """Add one already-reduced optimizer step to cumulative exposure."""
    if cumulative["num_repeats"] != step_exposure["num_repeats"]:
        raise ValueError("Cumulative and step repeat depths do not match.")
    sequence_length = step_exposure["sequence_length"]
    if sequence_length is None:
        raise ValueError("Cannot add an empty step exposure.")
    if cumulative["sequence_length"] is None:
        cumulative["sequence_length"] = int(sequence_length)
    elif cumulative["sequence_length"] != int(sequence_length):
        raise ValueError("Cumulative exposure cannot mix sequence lengths.")

    cumulative["total_optimizer_steps"] += 1
    for source, target in (
        ("microbatches", "total_microbatches"),
        ("examples", "total_examples_seen"),
        ("supervised_states", "total_supervised_states"),
        ("state_positions", "total_state_positions"),
        ("computed_cells", "total_computed_cells"),
        ("copied_cells", "total_copied_cells"),
        ("forward_only_examples", "forward_only_examples"),
        ("reverse_only_examples", "reverse_only_examples"),
        ("switch_examples", "switch_examples"),
    ):
        cumulative[target] += int(step_exposure[source])

    repeat_names = (
        "supervised_states",
        "state_positions",
        "forward_transitions",
        "reverse_transitions",
        "computed_cells",
        "copied_cells",
    )
    for key, step_repeat in step_exposure["by_repeat"].items():
        total_repeat = cumulative["by_repeat"][key]
        for name in repeat_names:
            total_repeat[name] += int(step_repeat[name])
    return _refresh_exposure(cumulative)


def copy_soca_training_exposure(exposure):
    """Return a refreshed copy safe to retain in logs or summaries."""
    return _refresh_exposure(copy.deepcopy(exposure))
