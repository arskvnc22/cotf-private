from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cellular_automaton.ca_forward import CAForwardContext
from cellular_automaton.soca_eval import (
    SOCABatchTrace,
    evaluate_soca_model,
    finalize_soca_eval_counters,
    new_soca_eval_counters,
    update_soca_eval_counters,
)
from cellular_automaton.soca_gen import SOCADataset, rollout_soca


class SOCAOracle(torch.nn.Module):
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
        raw_state = input_ids.clone()
        raw_state[:, row_length:] -= 2
        targets = rollout_soca(raw_state, direction_schedule)
        logits = F.one_hot(targets.long(), num_classes=2).float() * 20.0
        return {
            "repeat_logits": logits if return_repeat_logits else None,
            "average_depth": torch.tensor(float(num_repeats)),
        }


def test_soca_oracle_is_perfect_in_both_directions_and_restores_mode():
    dataset = SOCADataset(
        num_samples=8,
        sequence_length=12,
        num_repeats=3,
        seed=11,
        reversal_weights=[1, 0, 0, 1],
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = SOCAOracle()
    model.train()

    metrics = evaluate_soca_model(
        CAForwardContext(model),
        loader,
        "cpu",
        copy_loss_weight=1.0,
        target_steps_per_repeat=1,
    )

    assert model.training
    overall = metrics["overall"]
    assert overall["state"]["cell_accuracy"] == 1.0
    assert overall["state"]["exact_row_accuracy"] == 1.0
    assert overall["computed"]["cell_accuracy"] == 1.0
    assert overall["copied"]["cell_accuracy"] == 1.0
    assert overall["joint_pair_cell_accuracy"] == 1.0
    assert set(metrics["by_repeat"]) == {"repeat_1", "repeat_2", "repeat_3"}


def test_one_forward_computed_error_does_not_reduce_copied_accuracy():
    initial = torch.tensor([[0, 1, 1, 0]])
    schedule = torch.tensor([[0]])
    targets = rollout_soca(initial, schedule)
    predictions = targets.clone()
    predictions[0, 0, 2] ^= 1
    logits = F.one_hot(predictions, num_classes=2).float() * 20.0
    trace = SOCABatchTrace(
        initial_state=initial,
        direction_schedule=schedule,
        target_steps_per_repeat=torch.ones_like(schedule),
        targets_by_repeat=targets,
        repeat_logits=logits,
        average_depth=torch.tensor(1.0),
    )
    counters = new_soca_eval_counters()

    update_soca_eval_counters(counters, trace)
    metrics = finalize_soca_eval_counters(
        counters,
        copy_loss_weight=1.0,
    )["overall"]

    assert metrics["state"]["cell_accuracy"] == 3 / 4
    assert metrics["state"]["exact_row_accuracy"] == 0.0
    assert metrics["computed"]["cell_accuracy"] == 1 / 2
    assert metrics["copied"]["cell_accuracy"] == 1.0
    assert metrics["joint_pair_cell_accuracy"] == 1 / 2
