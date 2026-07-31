"""Pure bookkeeping helpers for cellular-automaton training exposure."""


def _pair_key(ca_steps, num_repeats):
    return f"steps_{int(ca_steps)}_repeats_{int(num_repeats)}"


def new_training_exposure(materialized_training_rows):
    return {
        "accounting_exact": True,
        "legacy_approximated_steps": 0,
        "materialized_training_rows": int(materialized_training_rows),
        "total_optimizer_steps": 0,
        "total_microbatches": 0,
        "total_examples_seen": 0,
        "total_cells_seen": 0,
        "equivalent_dataset_passes": 0.0,
        "by_training_pair": {},
    }


def _refresh_training_exposure(exposure):
    total_examples = int(exposure["total_examples_seen"])
    total_steps = int(exposure["total_optimizer_steps"])
    materialized_rows = int(exposure["materialized_training_rows"])
    exposure["equivalent_dataset_passes"] = (
        total_examples / materialized_rows if materialized_rows else 0.0
    )
    for values in exposure["by_training_pair"].values():
        values["example_fraction"] = (
            values["examples_seen"] / total_examples if total_examples else 0.0
        )
        values["optimizer_step_fraction"] = (
            values["optimizer_steps"] / total_steps if total_steps else 0.0
        )
    return exposure


def add_training_exposure(
    exposure,
    *,
    ca_steps,
    num_repeats,
    optimizer_steps,
    microbatches,
    examples_seen,
    cells_seen,
):
    pair_key = _pair_key(ca_steps, num_repeats)
    pair = exposure["by_training_pair"].setdefault(
        pair_key,
        {
            "ca_steps": int(ca_steps),
            "num_repeats": int(num_repeats),
            "optimizer_steps": 0,
            "microbatches": 0,
            "examples_seen": 0,
            "cells_seen": 0,
            "example_fraction": 0.0,
            "optimizer_step_fraction": 0.0,
        },
    )
    exposure["total_optimizer_steps"] += int(optimizer_steps)
    exposure["total_microbatches"] += int(microbatches)
    exposure["total_examples_seen"] += int(examples_seen)
    exposure["total_cells_seen"] += int(cells_seen)
    pair["optimizer_steps"] += int(optimizer_steps)
    pair["microbatches"] += int(microbatches)
    pair["examples_seen"] += int(examples_seen)
    pair["cells_seen"] += int(cells_seen)
    return _refresh_training_exposure(exposure)


def rebuild_training_exposure(
    train_rows,
    *,
    materialized_training_rows,
    batch_size,
    accumulation_steps,
    world_size,
    num_cells,
    fallback_ca_steps,
    fallback_num_repeats,
):
    """Rebuild cumulative exposure after resume from retained per-step rows.

    New rows contain actual consumed sizes. Older rows are retained using the
    former full-batch estimate and explicitly mark the summary as approximate.
    """
    exposure = new_training_exposure(materialized_training_rows)
    for row in train_rows:
        exact = all(
            field in row
            for field in (
                "microbatches_this_step",
                "examples_this_step",
                "cells_this_step",
            )
        )
        if exact:
            microbatches = int(row["microbatches_this_step"])
            examples_seen = int(row["examples_this_step"])
            cells_seen = int(row["cells_this_step"])
        else:
            microbatches = int(accumulation_steps)
            examples_seen = int(batch_size * accumulation_steps * world_size)
            cells_seen = int(examples_seen * num_cells)
            exposure["accounting_exact"] = False
            exposure["legacy_approximated_steps"] += 1
        add_training_exposure(
            exposure,
            ca_steps=int(row.get("ca_steps", fallback_ca_steps)),
            num_repeats=int(row.get("num_repeats", fallback_num_repeats)),
            optimizer_steps=1,
            microbatches=microbatches,
            examples_seen=examples_seen,
            cells_seen=cells_seen,
        )
    return exposure
