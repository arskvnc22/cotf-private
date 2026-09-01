import pytest
import torch

from cellular_automaton.ca_gen import rule30
from cellular_automaton.soca_gen import (
    FORWARD,
    REVERSE,
    MaterializedSOCADataset,
    SOCADataset,
    generate_soca_batch,
    make_single_switch_schedule,
    rollout_soca,
    sample_first_reverse_repeats,
    soca_forward,
    soca_reverse,
    soca_step,
)


def _all_binary_rows(row_length):
    values = torch.arange(2**row_length)
    shifts = torch.arange(row_length - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).long()


@pytest.mark.parametrize("row_length", [1, 2, 3, 4])
def test_forward_and_reverse_are_exact_inverses_exhaustively(row_length):
    rows = _all_binary_rows(row_length)
    previous = rows[:, None, :].expand(-1, rows.shape[0], -1)
    current = rows[None, :, :].expand(rows.shape[0], -1, -1)
    states = torch.cat((previous, current), dim=-1).flatten(0, 1)

    assert states.shape[-1] == 2 * row_length
    assert torch.equal(soca_reverse(soca_forward(states)), states)
    assert torch.equal(soca_forward(soca_reverse(states)), states)


def test_forward_and_reverse_match_the_declared_formulas():
    previous = torch.tensor([[0, 1, 0, 1, 1], [1, 0, 1, 0, 1]])
    current = torch.tensor([[1, 1, 0, 0, 1], [0, 1, 0, 0, 1]])
    states = torch.cat((previous, current), dim=-1)

    expected_forward = torch.cat(
        (current, previous ^ rule30(current)),
        dim=-1,
    )
    expected_reverse = torch.cat(
        (current ^ rule30(previous), previous),
        dim=-1,
    )
    assert torch.equal(soca_forward(states), expected_forward)
    assert torch.equal(soca_reverse(states), expected_reverse)


def test_soca_step_selects_direction_independently_per_example():
    states = torch.tensor(
        [
            [0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    result = soca_step(states, torch.tensor([FORWARD, REVERSE]))

    assert torch.equal(result[0], soca_forward(states[0]))
    assert torch.equal(result[1], soca_reverse(states[1]))


def test_single_switch_schedule_uses_one_based_first_reverse_repeat():
    schedule = make_single_switch_schedule(8, first_reverse_repeat=5)
    assert torch.equal(
        schedule,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.uint8),
    )
    assert torch.equal(
        make_single_switch_schedule(4, first_reverse_repeat=1),
        torch.ones(4, dtype=torch.uint8),
    )
    assert torch.equal(
        make_single_switch_schedule(4, first_reverse_repeat=5),
        torch.zeros(4, dtype=torch.uint8),
    )


def test_batched_single_switch_schedules_can_reverse_at_different_repeats():
    schedule = make_single_switch_schedule(
        4,
        torch.tensor([1, 3, 5]),
        batch_size=3,
    )
    expected = torch.tensor(
        [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
        dtype=torch.uint8,
    )
    assert torch.equal(schedule, expected)


def test_forward_four_reverse_four_returns_to_the_input():
    state = torch.tensor(
        [[0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]],
        dtype=torch.long,
    )
    schedule = make_single_switch_schedule(
        8,
        first_reverse_repeat=5,
        batch_size=1,
    )
    trajectory = rollout_soca(state, schedule)

    four_forward = soca_forward(soca_forward(soca_forward(soca_forward(state))))
    assert trajectory.shape == (1, 8, 12)
    assert torch.equal(trajectory[:, 3], four_forward)
    assert torch.equal(trajectory[:, -1], state)


def test_rollout_records_every_repeat_and_supports_a_common_schedule():
    states = torch.tensor(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
        ],
        dtype=torch.long,
    )
    schedule = torch.tensor([FORWARD, REVERSE, FORWARD])
    trajectory = rollout_soca(states, schedule)

    first = soca_forward(states)
    second = soca_reverse(first)
    third = soca_forward(second)
    assert trajectory.shape == (2, 3, 6)
    assert torch.equal(trajectory[:, 0], first)
    assert torch.equal(trajectory[:, 1], second)
    assert torch.equal(trajectory[:, 2], third)


def test_rollout_accepts_an_existing_unbatched_state_sequence():
    state = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.uint8)
    forward_state = rollout_soca(state, torch.tensor([FORWARD]))[0]
    restored = rollout_soca(forward_state, torch.tensor([REVERSE]))[0]
    assert torch.equal(restored, state)


def test_generate_batch_returns_total_sequence_length_without_token_ids():
    batch = generate_soca_batch(
        5,
        14,
        num_repeats=4,
        first_reverse_repeat=torch.tensor([1, 2, 3, 4, 5]),
    )

    assert set(batch) == {
        "state_sequence",
        "direction_schedule",
        "first_reverse_repeat",
        "example_id",
    }
    assert batch["state_sequence"].shape == (5, 14)
    assert batch["direction_schedule"].shape == (5, 4)
    assert set(batch["state_sequence"].unique().tolist()) <= {0, 1}
    assert torch.equal(
        batch["first_reverse_repeat"], torch.tensor([1, 2, 3, 4, 5])
    )


def test_generate_batch_accepts_an_existing_state_and_explicit_schedule():
    states = torch.tensor(
        [[0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1]],
        dtype=torch.uint8,
    )
    schedule = torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.uint8)
    batch = generate_soca_batch(
        2,
        6,
        num_repeats=3,
        initial_state_sequence=states,
        direction_schedule=schedule,
    )

    assert torch.equal(batch["state_sequence"], states)
    assert torch.equal(batch["direction_schedule"], schedule)
    assert torch.equal(batch["first_reverse_repeat"], torch.tensor([2, 1]))


def test_weighted_reversal_sampling_uses_only_enabled_categories():
    generator = torch.Generator().manual_seed(12)
    samples = sample_first_reverse_repeats(
        100,
        4,
        [0, 0, 3, 0, 2],
        generator=generator,
    )
    assert set(samples.tolist()) <= {3, 5}
    assert {3, 5} <= set(samples.tolist())


def test_indexed_dataset_is_reproducible_and_uses_independent_seeds():
    kwargs = dict(
        num_samples=20,
        sequence_length=24,
        num_repeats=5,
        seed=17,
        schedule_seed=29,
        reversal_weights=[1, 2, 3, 4, 5, 6],
    )
    first = SOCADataset(**kwargs)
    second = SOCADataset(**kwargs)
    different_schedule = SOCADataset(**{**kwargs, "schedule_seed": 31})

    assert torch.equal(first[8]["state_sequence"], second[8]["state_sequence"])
    assert torch.equal(
        first[8]["direction_schedule"], second[8]["direction_schedule"]
    )
    assert torch.equal(
        first[8]["state_sequence"], different_schedule[8]["state_sequence"]
    )
    assert any(
        not torch.equal(
            first[index]["direction_schedule"],
            different_schedule[index]["direction_schedule"],
        )
        for index in range(len(first))
    )


def test_indexed_dataset_retains_bernoulli_extremes():
    all_zero = SOCADataset(
        num_samples=1,
        sequence_length=12,
        num_repeats=2,
        bernoulli_p=0.0,
    )[0]["state_sequence"]
    all_one = SOCADataset(
        num_samples=1,
        sequence_length=12,
        num_repeats=2,
        bernoulli_p=1.0,
    )[0]["state_sequence"]

    assert torch.count_nonzero(all_zero) == 0
    assert torch.count_nonzero(all_one) == 12


def test_materialized_dataset_is_fixed_compact_and_correct():
    kwargs = dict(
        num_samples=12,
        sequence_length=20,
        num_repeats=6,
        seed=31,
        schedule_seed=47,
        reversal_weights=[1, 1, 1, 1, 1, 1, 1],
    )
    first = MaterializedSOCADataset(**kwargs)
    second = MaterializedSOCADataset(**kwargs)

    assert first.state_sequences.dtype == torch.uint8
    assert first.direction_schedules.dtype == torch.uint8
    assert first.storage_bytes == 12 * (20 + 6 + 8 + 8)
    assert torch.equal(first.state_sequences, second.state_sequences)
    assert torch.equal(first.direction_schedules, second.direction_schedules)
    assert torch.equal(first.first_reverse_repeats, second.first_reverse_repeats)
    for index in range(len(first)):
        expected = make_single_switch_schedule(
            6,
            first[index]["first_reverse_repeat"],
        )
        assert torch.equal(first[index]["direction_schedule"], expected)


@pytest.mark.parametrize(
    "state",
    [
        torch.tensor(0),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([0, 2], dtype=torch.long),
        torch.zeros(4, dtype=torch.float32),
    ],
)
def test_state_sequence_validation_rejects_invalid_inputs(state):
    with pytest.raises((TypeError, ValueError)):
        soca_forward(state)


def test_generation_rejects_odd_total_sequence_length():
    with pytest.raises(ValueError, match="must be even"):
        generate_soca_batch(2, 7, num_repeats=3)
    with pytest.raises(ValueError, match="must be even"):
        SOCADataset(num_samples=2, sequence_length=7, num_repeats=3)


def test_schedule_validation_rejects_invalid_values_shapes_and_bounds():
    states = torch.zeros((2, 8), dtype=torch.long)
    with pytest.raises(ValueError, match="binary"):
        rollout_soca(states, torch.tensor([[0, 2], [1, 0]]))
    with pytest.raises(ValueError, match="batch sizes"):
        rollout_soca(states, torch.zeros((3, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="first_reverse_repeat"):
        make_single_switch_schedule(4, first_reverse_repeat=0)
    with pytest.raises(ValueError, match="first_reverse_repeat"):
        make_single_switch_schedule(4, first_reverse_repeat=6)


def test_generation_rejects_conflicting_schedule_policies():
    with pytest.raises(ValueError, match="Provide only one"):
        generate_soca_batch(
            2,
            8,
            num_repeats=3,
            first_reverse_repeat=2,
            reversal_weights=[1, 1, 1, 1],
        )


@pytest.mark.parametrize(
    "weights",
    [
        [1, 1],
        [0, 0, 0, 0],
        [1, -1, 1, 1],
        [1, float("nan"), 1, 1],
    ],
)
def test_reversal_weight_validation(weights):
    with pytest.raises(ValueError):
        sample_first_reverse_repeats(2, 3, weights)
