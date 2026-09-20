import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellular_automaton.ca_train import (
    _set_and_validate_forward_policy,
    ca_selection_key,
    extrapolation_checkpoint_eligible,
    extrapolation_checkpoint_updates,
    should_run_scheduled_evaluation,
    training_pair_for_step,
    _supports_clean_state_intervention,
)
from cellular_automaton.ca_forward import CAForwardPolicy
from optim.runner_utils import InfiniteBatchIterator


class _BasicRecurrentModel:
    def forward(self, inputs, *, num_repeats=None):
        return inputs, num_repeats


class _CleanStateInterventionModel:
    def forward(
        self,
        inputs,
        *,
        num_repeats=None,
        intervention_source_depth=None,
        intervention_input_ids=None,
    ):
        return (
            inputs,
            num_repeats,
            intervention_source_depth,
            intervention_input_ids,
        )


class _PartialInterventionModel:
    def forward(self, inputs, *, intervention_source_depth=None):
        return inputs, intervention_source_depth


def test_clean_state_probe_requires_both_explicit_intervention_arguments():
    assert not _supports_clean_state_intervention(_BasicRecurrentModel())
    assert not _supports_clean_state_intervention(_PartialInterventionModel())
    assert _supports_clean_state_intervention(_CleanStateInterventionModel())

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


def test_scheduled_evaluation_skips_step_zero_and_keeps_positive_intervals():
    assert not should_run_scheduled_evaluation(0, 1_000)
    assert not should_run_scheduled_evaluation(999, 1_000)
    assert should_run_scheduled_evaluation(1_000, 1_000)
    assert should_run_scheduled_evaluation(2_000, 1_000)


@pytest.mark.parametrize("step,eval_freq", [(-1, 1_000), (0, 0)])
def test_scheduled_evaluation_rejects_invalid_inputs(step, eval_freq):
    with pytest.raises(ValueError):
        should_run_scheduled_evaluation(step, eval_freq)


def test_fresh_run_records_its_requested_forward_policy():
    stats = {}
    recent_policy = CAForwardPolicy(repeat_cache_window=4).metadata()

    _set_and_validate_forward_policy(stats, recent_policy, start_step=0)

    assert stats["forward_policy"] == recent_policy


def test_missing_historical_forward_policy_means_full_cache():
    stats = {}
    full_policy = CAForwardPolicy().metadata()

    _set_and_validate_forward_policy(stats, full_policy, start_step=10)

    assert stats["forward_policy"] == full_policy


def test_resume_accepts_the_same_forward_policy():
    recent_policy = CAForwardPolicy(repeat_cache_window=4).metadata()
    stats = {"forward_policy": recent_policy}

    _set_and_validate_forward_policy(stats, recent_policy, start_step=10)

    assert stats["forward_policy"] == recent_policy


def test_resume_rejects_a_policy_change_and_preserves_stored_metadata():
    full_policy = CAForwardPolicy().metadata()
    recent_policy = CAForwardPolicy(repeat_cache_window=4).metadata()
    stats = {"forward_policy": full_policy}

    with pytest.raises(ValueError, match="different forward policy"):
        _set_and_validate_forward_policy(stats, recent_policy, start_step=10)

    assert stats["forward_policy"] == full_policy
