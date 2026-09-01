import pytest

from cellular_automaton.ca_exposure import (
    add_training_exposure,
    new_training_exposure,
    rebuild_training_exposure,
)


def test_actual_partial_batches_are_counted_by_pair():
    exposure = new_training_exposure(materialized_training_rows=10)
    add_training_exposure(
        exposure,
        ca_steps=4,
        num_repeats=4,
        optimizer_steps=1,
        microbatches=1,
        examples_seen=8,
        cells_seen=512,
    )
    add_training_exposure(
        exposure,
        ca_steps=5,
        num_repeats=5,
        optimizer_steps=1,
        microbatches=1,
        examples_seen=2,
        cells_seen=128,
    )

    assert exposure["accounting_exact"] is True
    assert exposure["total_examples_seen"] == 10
    assert exposure["total_cells_seen"] == 640
    assert exposure["equivalent_dataset_passes"] == 1.0
    assert exposure["by_training_pair"]["steps_4_repeats_4"][
        "example_fraction"
    ] == 0.8
    assert exposure["by_training_pair"]["steps_5_repeats_5"][
        "example_fraction"
    ] == 0.2
    assert exposure["by_training_pair"]["steps_4_repeats_4"][
        "optimizer_step_fraction"
    ] == 0.5


def test_resume_rebuild_uses_exact_per_step_counts():
    rows = [
        {
            "step": 1,
            "ca_steps": 4,
            "num_repeats": 4,
            "microbatches_this_step": 2,
            "examples_this_step": 16,
            "cells_this_step": 1024,
        },
        {
            "step": 2,
            "ca_steps": 5,
            "num_repeats": 5,
            "microbatches_this_step": 2,
            "examples_this_step": 10,
            "cells_this_step": 640,
        },
    ]
    exposure = rebuild_training_exposure(
        rows,
        materialized_training_rows=20,
        batch_size=8,
        accumulation_steps=2,
        world_size=1,
        num_cells=64,
        fallback_ca_steps=1,
        fallback_num_repeats=1,
    )

    assert exposure["accounting_exact"] is True
    assert exposure["legacy_approximated_steps"] == 0
    assert exposure["total_optimizer_steps"] == 2
    assert exposure["total_microbatches"] == 4
    assert exposure["total_examples_seen"] == 26
    assert exposure["equivalent_dataset_passes"] == pytest.approx(1.3)


def test_legacy_rows_are_approximated_and_labelled():
    exposure = rebuild_training_exposure(
        [{"step": 1, "ca_steps": 2, "num_repeats": 2}],
        materialized_training_rows=100,
        batch_size=8,
        accumulation_steps=2,
        world_size=2,
        num_cells=64,
        fallback_ca_steps=1,
        fallback_num_repeats=1,
    )

    assert exposure["accounting_exact"] is False
    assert exposure["legacy_approximated_steps"] == 1
    assert exposure["total_examples_seen"] == 32
    assert exposure["total_cells_seen"] == 2048
