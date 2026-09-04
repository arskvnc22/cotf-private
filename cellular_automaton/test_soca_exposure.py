import math

import torch

from cellular_automaton.soca_exposure import (
    add_soca_training_exposure,
    copy_soca_training_exposure,
    new_soca_step_exposure,
    new_soca_training_exposure,
    update_soca_step_exposure,
)


def test_known_schedules_count_every_repeat_direction_and_component():
    states = torch.zeros((4, 8), dtype=torch.long)
    schedules = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ]
    )
    exposure = new_soca_step_exposure(num_repeats=3)

    update_soca_step_exposure(exposure, states, schedules)

    assert exposure["microbatches"] == 1
    assert exposure["examples"] == 4
    assert exposure["supervised_states"] == 12
    assert exposure["state_positions"] == 96
    assert exposure["computed_cells"] == 48
    assert exposure["copied_cells"] == 48
    assert exposure["forward_only_examples"] == 1
    assert exposure["reverse_only_examples"] == 1
    assert exposure["switch_examples"] == 2

    expected_directions = [(3, 1), (2, 2), (1, 3)]
    for repeat_index, (forward, reverse) in enumerate(
        expected_directions, start=1
    ):
        values = exposure["by_repeat"][f"repeat_{repeat_index}"]
        assert values["supervised_states"] == 4
        assert values["state_positions"] == 32
        assert values["computed_cells"] == 16
        assert values["copied_cells"] == 16
        assert values["forward_transitions"] == forward
        assert values["reverse_transitions"] == reverse
        assert values["repeat_loss_weight"] == 1 / 3


def test_cumulative_exposure_tracks_passes_and_realized_direction_balance():
    states = torch.zeros((2, 8), dtype=torch.long)
    schedules = torch.tensor([[0, 0], [1, 1]])
    step = new_soca_step_exposure(num_repeats=2)
    update_soca_step_exposure(step, states, schedules)

    cumulative = new_soca_training_exposure(
        materialized_training_rows=8,
        num_repeats=2,
        copy_loss_weight=1.0,
    )
    add_soca_training_exposure(cumulative, step)
    add_soca_training_exposure(cumulative, step)
    retained = copy_soca_training_exposure(cumulative)

    assert retained["total_optimizer_steps"] == 2
    assert retained["total_microbatches"] == 2
    assert retained["total_examples_seen"] == 4
    assert retained["total_supervised_states"] == 8
    assert retained["total_state_positions"] == 64
    assert retained["forward_only_examples"] == 2
    assert retained["reverse_only_examples"] == 2
    assert retained["switch_examples"] == 0
    assert retained["equivalent_dataset_passes"] == 0.5
    for values in retained["by_repeat"].values():
        assert values["forward_transitions"] == 2
        assert values["reverse_transitions"] == 2
        assert math.isclose(values["forward_fraction"], 0.5)
        assert math.isclose(values["reverse_fraction"], 0.5)


def test_exposure_rejects_mixed_sequence_lengths():
    exposure = new_soca_step_exposure(num_repeats=2)
    update_soca_step_exposure(
        exposure,
        torch.zeros((1, 8), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.long),
    )

    try:
        update_soca_step_exposure(
            exposure,
            torch.zeros((1, 10), dtype=torch.long),
            torch.zeros((1, 2), dtype=torch.long),
        )
    except ValueError as error:
        assert "cannot mix sequence lengths" in str(error)
    else:
        raise AssertionError("Expected mixed sequence lengths to be rejected.")
