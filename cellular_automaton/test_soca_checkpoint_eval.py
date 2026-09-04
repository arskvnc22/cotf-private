from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cellular_automaton.ca_forward import CAForwardContext
from cellular_automaton.soca_checkpoint_eval import (
    SOCAEvaluationCondition,
    build_soca_evaluation_conditions,
    evaluate_soca_protocol,
    load_soca_checkpoint,
    make_soca_condition_dataset,
)
from cellular_automaton.soca_gen import rollout_soca


class SOCAProtocolOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            input_vocab_size=4,
            output_vocab_size=2,
        )

    def forward(
        self,
        input_ids,
        *,
        direction_schedule,
        num_repeats,
        return_repeat_logits,
        get_logits,
    ):
        row_length = input_ids.shape[1] // 2
        initial_state = input_ids.clone()
        initial_state[:, row_length:] -= 2
        targets = rollout_soca(initial_state, direction_schedule)
        logits = F.one_hot(targets.long(), num_classes=2).float() * 20.0
        return {
            "repeat_logits": logits if return_repeat_logits else None,
            "average_depth": torch.tensor(float(num_repeats)),
        }


def test_condition_matrix_contains_controls_and_every_single_switch():
    conditions = build_soca_evaluation_conditions([4, 8])

    assert len(conditions) == 5 + 9
    horizon_four = [condition for condition in conditions if condition.horizon == 4]
    assert [condition.first_reverse_repeat for condition in horizon_four] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert [condition.schedule_text for condition in horizon_four] == [
        "RRRR",
        "FRRR",
        "FFRR",
        "FFFR",
        "FFFF",
    ]
    assert horizon_four[0].schedule_class == "reverse_only"
    assert horizon_four[-1].schedule_class == "forward_only"
    assert all(
        condition.schedule_class == "forward_then_reverse"
        for condition in horizon_four[1:-1]
    )


def test_condition_datasets_pair_initial_states_across_schedules_and_horizons():
    conditions = (
        SOCAEvaluationCondition(4, 1),
        SOCAEvaluationCondition(4, 3),
        SOCAEvaluationCondition(8, 5),
        SOCAEvaluationCondition(8, 9),
    )
    datasets = [
        make_soca_condition_dataset(
            condition,
            num_samples=16,
            sequence_length=12,
            bernoulli_p=0.5,
            seed=101,
        )
        for condition in conditions
    ]

    for dataset in datasets[1:]:
        assert torch.equal(
            datasets[0].state_sequences,
            dataset.state_sequences,
        )
    for condition, dataset in zip(conditions, datasets):
        expected = torch.tensor(condition.schedule, dtype=torch.uint8)
        assert torch.equal(
            dataset.direction_schedules,
            expected.expand(16, -1),
        )


def test_oracle_is_perfect_for_switches_and_extrapolated_repeats():
    args = SimpleNamespace(
        eval_samples=8,
        sequence_length=12,
        bernoulli_p=0.5,
        eval_seed=211,
        eval_batch_size=4,
        eval_num_workers=0,
        device=torch.device("cpu"),
        copy_loss_weight=1.0,
        eval_max_batches=None,
        dtype=torch.float32,
        trained_repeats=4,
    )
    conditions = build_soca_evaluation_conditions([4, 6])

    full_metrics, summaries = evaluate_soca_protocol(
        CAForwardContext(SOCAProtocolOracle()),
        conditions,
        args,
    )

    assert len(full_metrics) == len(conditions) == 12
    for condition in full_metrics.values():
        overall = condition["metrics"]["overall"]
        assert overall["state"]["cell_accuracy"] == 1.0
        assert overall["state"]["exact_row_accuracy"] == 1.0
        assert overall["computed"]["cell_accuracy"] == 1.0
        assert overall["copied"]["cell_accuracy"] == 1.0

    extrapolated = [
        summary for summary in summaries if summary["horizon"] == 6
    ]
    assert extrapolated
    assert all(
        set(summary["extrapolated_repeats"])
        == {"repeat_5", "repeat_6"}
        for summary in extrapolated
    )
    switched = [
        summary
        for summary in summaries
        if summary["schedule_class"] == "forward_then_reverse"
    ]
    assert switched
    assert all(
        summary["switch_repeat_metrics"]["state"]["exact_row_accuracy"]
        == 1.0
        for summary in switched
    )


def test_checkpoint_loading_is_strict_and_requires_training_metadata(tmp_path):
    source = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "valid.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "itr": 5000,
        },
        checkpoint_path,
    )
    loaded = torch.nn.Linear(3, 2)

    metadata = load_soca_checkpoint(loaded, checkpoint_path, "cpu")

    assert metadata["step"] == 5000
    for expected, actual in zip(source.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
    assert not loaded.training

    malformed_path = tmp_path / "malformed.pt"
    torch.save({"model": source.state_dict()}, malformed_path)
    with pytest.raises(ValueError, match="integer itr"):
        load_soca_checkpoint(
            torch.nn.Linear(3, 2),
            malformed_path,
            "cpu",
        )

    with pytest.raises(RuntimeError):
        load_soca_checkpoint(
            torch.nn.Linear(4, 2),
            checkpoint_path,
            "cpu",
        )
