from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cellular_automaton.soca_gen import MaterializedSOCADataset, rollout_soca
from cellular_automaton.soca_main import (
    get_args,
    make_soca_dataloaders,
    resolve_soca_task_config,
)
from cellular_automaton.soca_train import soca_loss, soca_train
from distributed.single import SinlgeNodeBackend


class TrainableSOCAOracle(torch.nn.Module):
    needs_iter = False

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
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
        logits = F.one_hot(targets.long(), num_classes=2).float()
        logits = logits * self.scale
        return {
            "repeat_logits": logits if return_repeat_logits else None,
            "average_depth": torch.tensor(float(num_repeats)),
        }


def _loader_args():
    return SimpleNamespace(
        soca_data_mode="materialized",
        device=torch.device("cpu"),
        soca_train_samples=64,
        sequence_length=8,
        n_repeat=3,
        soca_bernoulli_p=0.5,
        data_seed=17,
        soca_schedule_seed=29,
        soca_schedule_protocol="separate_directions",
        soca_reversal_weights=[1, 0, 0, 1],
        batch_size=8,
        soca_shuffle_seed=31,
        soca_num_workers=0,
        soca_eval_batch_size=8,
        soca_val_samples=16,
        soca_val_seed=41,
    )


def test_dry_run_loaders_use_only_persistent_directions_and_paired_states():
    train_loader, validation = make_soca_dataloaders(_loader_args())

    train_schedules = train_loader.dataset.direction_schedules
    assert bool(
        torch.all(
            train_schedules.eq(0).all(dim=1)
            | train_schedules.eq(1).all(dim=1)
        ).item()
    )
    assert train_schedules.eq(0).all(dim=1).any()
    assert train_schedules.eq(1).all(dim=1).any()

    forward_dataset = validation["forward_only"].dataset
    reverse_dataset = validation["reverse_only"].dataset
    assert torch.equal(
        forward_dataset.state_sequences,
        reverse_dataset.state_sequences,
    )
    assert bool(forward_dataset.direction_schedules.eq(0).all().item())
    assert bool(reverse_dataset.direction_schedules.eq(1).all().item())


def test_schedule_protocols_resolve_without_changing_original_default():
    separate = get_args(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--eval_freq",
            "1",
            "--n_repeat",
            "4",
        ]
    )
    separate.device = torch.device(separate.device)
    resolve_soca_task_config(separate)
    assert separate.soca_schedule_protocol == "separate_directions"
    assert separate.soca_reversal_weights == [1, 0, 0, 0, 1]

    switching = get_args(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--eval_freq",
            "1",
            "--n_repeat",
            "4",
            "--soca_schedule_protocol",
            "uniform_single_switch",
        ]
    )
    switching.device = torch.device(switching.device)
    resolve_soca_task_config(switching)
    assert switching.soca_reversal_weights == [1, 1, 1, 1, 1]

    invalid = get_args(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--eval_freq",
            "1",
            "--n_repeat",
            "4",
            "--soca_schedule_protocol",
            "uniform_single_switch",
            "--soca_reversal_weights",
            "1",
            "1",
            "2",
            "1",
            "1",
        ]
    )
    invalid.device = torch.device(invalid.device)
    with pytest.raises(ValueError, match="equal positive weight"):
        resolve_soca_task_config(invalid)


def test_uniform_switch_loaders_cover_every_schedule_with_paired_validation():
    args = _loader_args()
    args.soca_schedule_protocol = "uniform_single_switch"
    args.soca_reversal_weights = [1, 1, 1, 1]
    args.soca_train_samples = 256

    train_loader, validation = make_soca_dataloaders(args)

    assert set(train_loader.dataset.first_reverse_repeats.tolist()) == {
        1,
        2,
        3,
        4,
    }
    assert set(validation) == {
        "forward_only",
        "reverse_only",
        "switch_at_2",
        "switch_at_3",
    }
    reference_states = validation["forward_only"].dataset.state_sequences
    expected_schedules = {
        "forward_only": [0, 0, 0],
        "reverse_only": [1, 1, 1],
        "switch_at_2": [0, 1, 1],
        "switch_at_3": [0, 0, 1],
    }
    for condition, expected_schedule in expected_schedules.items():
        dataset = validation[condition].dataset
        assert torch.equal(reference_states, dataset.state_sequences)
        expected = torch.tensor(expected_schedule, dtype=torch.uint8)
        assert torch.equal(
            dataset.direction_schedules,
            expected.expand(args.soca_val_samples, -1),
        )


def test_repeat_loss_is_equal_mean_across_depths():
    targets = torch.tensor(
        [
            [
                [0, 0, 1, 1],
                [0, 1, 1, 0],
            ]
        ]
    )
    predictions = targets.clone()
    logits = F.one_hot(predictions, num_classes=2).float() * 5.0
    schedule = torch.tensor([[0, 1]])

    losses = soca_loss(
        logits,
        targets,
        schedule,
        output_vocab_size=2,
        copy_loss_weight=1.0,
    )

    assert losses["computed_loss_by_repeat"].shape == (2,)
    assert losses["copied_loss_by_repeat"].shape == (2,)
    assert torch.allclose(
        losses["computed_loss"],
        losses["computed_loss_by_repeat"].mean(),
    )
    assert torch.allclose(
        losses["copied_loss"],
        losses["copied_loss_by_repeat"].mean(),
    )


def test_one_step_training_smoke_records_exposure_and_validation(tmp_path):
    train_dataset = MaterializedSOCADataset(
        num_samples=8,
        sequence_length=8,
        num_repeats=2,
        seed=51,
        reversal_weights=[1, 0, 1],
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    validation = {
        "forward_only": DataLoader(
            MaterializedSOCADataset(
                num_samples=4,
                sequence_length=8,
                num_repeats=2,
                seed=61,
                first_reverse_repeat=3,
            ),
            batch_size=4,
        ),
        "reverse_only": DataLoader(
            MaterializedSOCADataset(
                num_samples=4,
                sequence_length=8,
                num_repeats=2,
                seed=61,
                first_reverse_repeat=1,
            ),
            batch_size=4,
        ),
    }
    args = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_repeat=2,
        soca_copy_loss_weight=1.0,
        soca_eval_max_batches=None,
        soca_log_every=1,
        soca_save_every=1,
        soca_exact_data_resume=False,
        repeat_cache_window=None,
        eval_freq=1,
        iterations=1,
        acc_steps=1,
        grad_clip=0.0,
        wandb=False,
    )
    model = TrainableSOCAOracle()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    backend = SinlgeNodeBackend(args)

    result = soca_train(
        model,
        optimizer,
        None,
        train_loader,
        validation,
        None,
        args,
        backend,
        tmp_path,
    )

    exposure = result["training_exposure"]
    assert exposure["total_optimizer_steps"] == 1
    assert exposure["total_microbatches"] == 1
    assert exposure["total_examples_seen"] == 4
    assert exposure["total_supervised_states"] == 8
    assert set(result["evaluation_history"]) == {"0", "1"}
    for evaluation in result["evaluation_history"]["1"][
        "conditions"
    ].values():
        assert evaluation["overall"]["computed"]["cell_accuracy"] == 1.0
