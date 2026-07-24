import torch

from cellular_automaton.ca_gen import (
    MaterializedRule30Dataset,
    Rule30Dataset,
    apply_rule30,
    rule30,
)


def test_rule30_truth_table():
    # Rows are ordered as neighbourhoods 111, 110, ..., 000.  The centre cell
    # at index 1 therefore sees each of Rule 30's eight possible inputs.
    rows = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    expected_centre = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0])
    assert torch.equal(rule30(rows)[:, 1], expected_centre)


def test_apply_rule30_matches_repeated_single_steps():
    states = torch.tensor([[0, 1, 0, 1, 1, 0]], dtype=torch.long)
    expected = rule30(rule30(rule30(states)))
    assert torch.equal(apply_rule30(states, steps=3), expected)


def test_rule30_dataset_is_reproducible_by_seed_and_index():
    first = Rule30Dataset(num_samples=10, num_cells=64, seed=7)
    second = Rule30Dataset(num_samples=10, num_cells=64, seed=7)
    different_split = Rule30Dataset(num_samples=10, num_cells=64, seed=99)

    assert torch.equal(first[4]["input_id"], second[4]["input_id"])
    assert torch.equal(first[4]["label"], second[4]["label"])
    assert not torch.equal(first[4]["input_id"], different_split[4]["input_id"])
    assert torch.equal(first[4]["label"], rule30(first[4]["input_id"]))


def test_dataset_retains_bernoulli_extremes():
    all_zero = Rule30Dataset(
        num_samples=1,
        num_cells=16,
        bernoulli_p=0.0,
        seed=0,
    )[0]["input_id"]
    all_one = Rule30Dataset(
        num_samples=1,
        num_cells=16,
        bernoulli_p=1.0,
        seed=0,
    )[0]["input_id"]

    assert torch.count_nonzero(all_zero) == 0
    assert torch.count_nonzero(all_one) == 16


def test_materialized_dataset_is_fixed_compact_and_correct():
    first = MaterializedRule30Dataset(
        num_samples=12,
        num_cells=20,
        steps=2,
        seed=31,
    )
    second = MaterializedRule30Dataset(
        num_samples=12,
        num_cells=20,
        steps=2,
        seed=31,
    )

    assert first.inputs.dtype == torch.uint8
    assert first.labels.dtype == torch.uint8
    assert first.storage_bytes == 12 * 20 * 2
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.labels.long(), apply_rule30(first.inputs.long(), steps=2))
