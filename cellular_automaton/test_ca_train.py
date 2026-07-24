import torch
from torch.utils.data import DataLoader, TensorDataset

from cellular_automaton.ca_train import (
    ca_selection_key,
    extrapolation_checkpoint_eligible,
    extrapolation_checkpoint_updates,
    training_pair_for_step,
)
from optim.runner_utils import InfiniteBatchIterator


def test_ca_best_key_uses_cell_accuracy_and_loss_as_tiebreakers():
    first = {
        "exact_sequence_accuracy": 0.5,
        "cell_accuracy": 0.90,
        "loss": 0.2,
    }
    second = {
        "exact_sequence_accuracy": 0.5,
        "cell_accuracy": 0.91,
        "loss": 0.4,
    }
    assert ca_selection_key(second, "exact_sequence_accuracy") > ca_selection_key(
        first, "exact_sequence_accuracy"
    )


def test_strict_extrapolation_gate_uses_worst_training_pair():
    metrics = [
        {"cell_accuracy": 1.0, "exact_sequence_accuracy": 1.0},
        {"cell_accuracy": 0.995, "exact_sequence_accuracy": 0.94},
    ]

    eligible, minimum_cell, minimum_exact = extrapolation_checkpoint_eligible(
        metrics,
        minimum_cell_accuracy=0.99,
        minimum_exact_sequence_accuracy=0.95,
    )

    assert eligible is False
    assert minimum_cell == 0.995
    assert minimum_exact == 0.94


def test_relaxed_strict_extrapolation_gate_can_pass():
    metrics = [
        {"cell_accuracy": 1.0, "exact_sequence_accuracy": 1.0},
        {"cell_accuracy": 0.995, "exact_sequence_accuracy": 0.96},
    ]

    eligible, _, _ = extrapolation_checkpoint_eligible(
        metrics,
        minimum_cell_accuracy=0.99,
        minimum_exact_sequence_accuracy=0.95,
    )

    assert eligible is True


def test_unconstrained_checkpoint_can_update_while_strict_gate_fails():
    update_strict, update_unconstrained = extrapolation_checkpoint_updates(
        (0.8, 0.1, -0.2),
        strict_key=None,
        unconstrained_key=(0.7, 0.2, -0.1),
        strict_eligible=False,
    )

    assert update_strict is False
    assert update_unconstrained is True


def test_strict_and_unconstrained_checkpoints_rank_independently():
    update_strict, update_unconstrained = extrapolation_checkpoint_updates(
        (0.8, 0.1, -0.2),
        strict_key=(0.7, 0.2, -0.1),
        unconstrained_key=(0.9, 0.0, -0.3),
        strict_eligible=True,
    )

    assert update_strict is True
    assert update_unconstrained is False


def _make_shuffled_loader():
    generator = torch.Generator().manual_seed(123)
    return DataLoader(
        TensorDataset(torch.arange(20)),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )


def test_infinite_batch_iterator_restores_batch_order():
    original = InfiniteBatchIterator(_make_shuffled_loader())
    for _ in range(4):
        next(original)
    state = original.state_dict()
    expected = [next(original)[0] for _ in range(5)]

    restored = InfiniteBatchIterator(_make_shuffled_loader(), state=state)
    actual = [next(restored)[0] for _ in range(5)]

    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_variable_training_pairs_are_round_robin_and_resume_stable():
    pairs = [(1, 1), (2, 2), (4, 4)]
    assert [training_pair_for_step(pairs, step) for step in range(7)] == [
        (1, 1),
        (2, 2),
        (4, 4),
        (1, 1),
        (2, 2),
        (4, 4),
        (1, 1),
    ]
    assert training_pair_for_step([], 10) is None
