import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cellular_automaton.ca_eval import (
    _forward_all_cells,
    evaluate_ca_clean_state_transition_lengths,
    evaluate_ca_clean_state_transitions,
    evaluate_ca_pairs,
    evaluate_ca_model,
    evaluate_loaded_ca_checkpoint,
    evaluate_ca_repeat_horizon_diagnostics,
    finalize_ca_metrics,
    format_ca_repeat_examples,
    new_ca_counters,
    run_final_ca_evaluation,
    update_ca_counters,
)
from cellular_automaton.ca_forward import CAForwardContext, CAForwardPolicy
from cellular_automaton.ca_gen import Rule30Dataset, apply_rule30, rule30


def logits_for_predictions(predictions):
    return F.one_hot(predictions, num_classes=2).float()


def forward_context(model, *, repeat_cache_window=None):
    return CAForwardContext(
        model,
        CAForwardPolicy(repeat_cache_window=repeat_cache_window),
    )


def test_counter_metrics_for_one_known_error():
    labels = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ]
    )
    predictions = labels.clone()
    predictions[0, 0] = 1

    counters = new_ca_counters()
    update_ca_counters(counters, logits_for_predictions(predictions), labels, loss=0.25)
    metrics = finalize_ca_metrics(counters)

    assert metrics["correct_cells"] == 7
    assert metrics["total_cells"] == 8
    assert metrics["cell_accuracy"] == 7 / 8
    assert metrics["exact_sequences"] == 1
    assert metrics["total_sequences"] == 2
    assert metrics["exact_sequence_accuracy"] == 1 / 2
    assert metrics["total_bit_errors"] == 1
    assert metrics["mean_bit_errors_per_sequence"] == 1 / 2
    assert metrics["boundary_accuracy"] == 3 / 4
    assert metrics["interior_accuracy"] == 1.0
    assert metrics["position_accuracy"] == [0.5, 1.0, 1.0, 1.0]
    assert metrics["loss"] == 0.25


class Rule30Oracle(torch.nn.Module):
    def forward(self, inputs, targets=None, get_logits=False):
        predictions = rule30(inputs)
        logits = logits_for_predictions(predictions) * 20.0
        loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1))
        return {"logits": logits if get_logits else None, "loss": loss}


def test_evaluate_ca_model_with_oracle_and_restores_training_mode():
    dataset = Rule30Dataset(num_samples=7, num_cells=16, seed=4)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = Rule30Oracle()
    model.train()

    metrics = evaluate_ca_model(
        forward_context(model), dataloader, device="cpu"
    )

    assert model.training
    assert metrics["num_batches"] == 3
    assert metrics["total_sequences"] == 7
    assert metrics["total_cells"] == 7 * 16
    assert metrics["cell_accuracy"] == 1.0
    assert metrics["exact_sequence_accuracy"] == 1.0
    assert metrics["exact_sequences"] == 7
    assert metrics["total_bit_errors"] == 0
    assert metrics["boundary_accuracy"] == 1.0
    assert metrics["interior_accuracy"] == 1.0
    assert all(value == 1.0 for value in metrics["position_accuracy"])
    assert not math.isnan(metrics["loss"])


class IterativeRule30Oracle(torch.nn.Module):
    def __init__(self, default_repeats=1):
        super().__init__()
        self.default_repeats = default_repeats

    def forward(self, inputs, get_logits=False, num_repeats=None):
        repeats = self.default_repeats if num_repeats is None else num_repeats
        predictions = apply_rule30(inputs, steps=repeats)
        logits = logits_for_predictions(predictions) * 20.0
        return {
            "logits": logits if get_logits else None,
            "average_depth": torch.as_tensor(repeats),
        }


class LanguageModelStyleOracle(torch.nn.Module):
    """Mimic current models that only project the final cell at inference."""

    def forward(self, inputs, targets=None, get_logits=False):
        predictions = rule30(inputs)
        logits = logits_for_predictions(predictions) * 20.0
        if targets is None:
            logits = logits[:, -1:, :]
        return {"logits": logits if get_logits else None, "loss": None}


_ARGUMENT_OMITTED = object()


class WindowRecordingOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_windows = []
        self.interventions = []

    def forward(
        self,
        inputs,
        targets=None,
        get_logits=False,
        return_all_logits=False,
        num_repeats=1,
        return_repeat_states=False,
        repeat_cache_window=_ARGUMENT_OMITTED,
        intervention_source_depth=None,
        intervention_input_ids=None,
    ):
        self.received_windows.append(repeat_cache_window)
        self.interventions.append(
            (intervention_source_depth, intervention_input_ids)
        )
        if intervention_source_depth is None:
            state = apply_rule30(inputs, steps=num_repeats)
        else:
            state = rule30(intervention_input_ids)
        logits = logits_for_predictions(state) * 20.0
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1))
        result = {
            "logits": logits if get_logits else None,
            "loss": loss,
            "average_depth": torch.as_tensor(num_repeats),
        }
        if return_repeat_states:
            result["repeat_states"] = [
                F.one_hot(
                    inputs if repeat == 0 else apply_rule30(inputs, steps=repeat),
                    num_classes=2,
                ).float()
                for repeat in range(num_repeats + 1)
            ]
        return result


class PreservedCacheRecordingOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.repeat_counts = []
        self.source_depths = []
        self.intervention_inputs = []

    def forward(
        self,
        inputs,
        get_logits=False,
        return_all_logits=False,
        num_repeats=1,
        intervention_source_depth=None,
        intervention_input_ids=None,
    ):
        self.inputs.append(inputs.detach().cpu().clone())
        self.repeat_counts.append(num_repeats)
        self.source_depths.append(intervention_source_depth)
        self.intervention_inputs.append(
            intervention_input_ids.detach().cpu().clone()
        )
        logits = logits_for_predictions(rule30(intervention_input_ids)) * 20.0
        return {
            "logits": logits if get_logits else None,
            "average_depth": torch.as_tensor(num_repeats),
        }


class RetryRecordingOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, **kwargs):
        self.calls.append(dict(kwargs))
        if "return_all_logits" in kwargs:
            raise TypeError("unexpected keyword argument 'return_all_logits'")
        predictions = rule30(inputs)
        logits = logits_for_predictions(predictions) * 20.0
        if "targets" not in kwargs:
            logits = logits[:, -1:, :]
        return {
            "logits": logits if kwargs.get("get_logits") else None,
            "loss": None,
        }


def test_forward_policy_is_applied_and_full_cache_argument_is_omitted():
    dataset = Rule30Dataset(num_samples=2, num_cells=8, seed=51)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    full_model = WindowRecordingOracle()
    evaluate_ca_model(forward_context(full_model), dataloader, device="cpu")
    assert full_model.received_windows == [_ARGUMENT_OMITTED]

    recent_model = WindowRecordingOracle()
    evaluate_ca_model(
        forward_context(recent_model, repeat_cache_window=4),
        dataloader,
        device="cpu",
    )
    assert recent_model.received_windows == [4]


def test_all_position_compatibility_retries_keep_the_forward_policy():
    model = RetryRecordingOracle()
    inputs = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]])

    logits, _ = _forward_all_cells(
        forward_context(model, repeat_cache_window=4), inputs
    )

    assert logits.shape == (1, 8, 2)
    assert len(model.calls) == 3
    assert all(call["repeat_cache_window"] == 4 for call in model.calls)


def test_recent_policy_reaches_repeat_diagnostics_and_external_rollouts():
    dataset = Rule30Dataset(num_samples=2, num_cells=8, seed=52)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    eval_loaders = {8: dataloader}
    model = WindowRecordingOracle()
    context = forward_context(model, repeat_cache_window=4)

    evaluate_ca_repeat_horizon_diagnostics(
        context,
        dataloader,
        device="cpu",
        max_repeats=2,
        target_horizons=range(3),
        num_examples=0,
    )
    assert model.received_windows
    assert all(window == 4 for window in model.received_windows)

    model.received_windows.clear()
    run_final_ca_evaluation(
        context,
        eval_loaders,
        device="cpu",
        trained_ca_steps=1,
        trained_num_repeats=1,
        external_ca_steps=[3],
    )
    assert len(model.received_windows) == 4
    assert all(window == 4 for window in model.received_windows)

    model.received_windows.clear()
    evaluate_ca_clean_state_transition_lengths(
        context,
        eval_loaders,
        device="cpu",
        max_transition_depth=2,
    )
    assert len(model.received_windows) == 2
    assert all(window == 4 for window in model.received_windows)


def test_clean_state_probe_preserves_rollout_origin_and_injects_exact_rows():
    input_row = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])
    dataloader = DataLoader(
        [{"input_id": input_row}],
        batch_size=1,
        shuffle=False,
    )
    model = PreservedCacheRecordingOracle()
    model.train()

    diagnostics = evaluate_ca_clean_state_transitions(
        forward_context(model),
        dataloader,
        device="cpu",
        max_transition_depth=3,
    )

    initial_state = input_row.unsqueeze(0)
    expected_states = [initial_state]
    for _ in range(2):
        expected_states.append(rule30(expected_states[-1]))
    assert model.training
    assert model.repeat_counts == [1, 2, 3]
    assert model.source_depths == [0, 1, 2]
    assert all(torch.equal(actual, initial_state) for actual in model.inputs)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.intervention_inputs, expected_states)
    )
    assert diagnostics["max_transition_depth"] == 3
    assert diagnostics["num_batches"] == 1
    final_transition = diagnostics["transitions"]["steps_2_to_3"]
    assert final_transition["source_ca_steps"] == 2
    assert final_transition["target_ca_steps"] == 3
    assert final_transition["num_repeats"] == 3
    assert final_transition["intervention_source_depth"] == 2


def test_clean_state_probe_oracle_respects_batch_limit_and_empty_loader():
    dataset = Rule30Dataset(num_samples=5, num_cells=12, seed=61)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    diagnostics = evaluate_ca_clean_state_transitions(
        forward_context(DiagnosticRule30Oracle()),
        dataloader,
        device="cpu",
        max_transition_depth=4,
        max_batches=2,
    )

    for transition in diagnostics["transitions"].values():
        metrics = transition["metrics"]
        assert metrics["num_batches"] == 2
        assert metrics["total_sequences"] == 4
        assert metrics["cell_accuracy"] == 1.0
        assert metrics["exact_sequence_accuracy"] == 1.0
    for target_depth, transition in enumerate(
        diagnostics["transitions"].values(), start=1
    ):
        metrics = transition["metrics"]
        assert metrics["average_depth"] == float(target_depth)

    empty_loader = DataLoader([], batch_size=1)
    with pytest.raises(ValueError, match="consumed no batches"):
        evaluate_ca_clean_state_transitions(
            forward_context(DiagnosticRule30Oracle()),
            empty_loader,
            device="cpu",
            max_transition_depth=1,
        )


def test_final_evaluation_requests_all_logits_from_language_model_interface():
    dataset = Rule30Dataset(num_samples=3, num_cells=8, seed=17)
    eval_loaders = {8: DataLoader(dataset, batch_size=2, shuffle=False)}

    results = run_final_ca_evaluation(
        forward_context(LanguageModelStyleOracle()),
        eval_loaders,
        device="cpu",
        trained_ca_steps=1,
    )

    metrics = results["in_distribution"]["by_length"]["8"]
    assert metrics["exact_sequence_accuracy"] == 1.0


def test_final_evaluation_internal_and_external_extrapolation():
    dataset = Rule30Dataset(num_samples=5, num_cells=12, seed=8)
    eval_loaders = {12: DataLoader(dataset, batch_size=2, shuffle=False)}
    model = IterativeRule30Oracle(default_repeats=1)

    results = run_final_ca_evaluation(
        forward_context(model),
        eval_loaders,
        device="cpu",
        trained_ca_steps=1,
        trained_num_repeats=1,
        internal_pairs=[(1, 1), (3, 3)],
        external_ca_steps=[1, 3],
    )

    baseline = results["in_distribution"]["by_length"]["12"]
    internal = results["internal_repeat_extrapolation"]["steps_3_repeats_3"][
        "by_length"
    ]["12"]
    external = results["external_rollout"]["steps_3"]["by_length"]["12"]

    assert baseline["exact_sequence_accuracy"] == 1.0
    assert internal["exact_sequence_accuracy"] == 1.0
    assert internal["average_depth"] == 3.0
    assert external["exact_sequence_accuracy"] == 1.0
    assert external["average_depth"] == 3.0
    assert results["external_rollout"]["steps_3"]["model_calls"] == 3


def test_variable_training_pairs_are_evaluated_separately():
    dataset = Rule30Dataset(num_samples=5, num_cells=12, seed=18)
    eval_loaders = {12: DataLoader(dataset, batch_size=2, shuffle=False)}
    model = IterativeRule30Oracle(default_repeats=1)

    metrics = evaluate_ca_pairs(
        forward_context(model),
        eval_loaders,
        device="cpu",
        pairs=[(1, 1), (2, 2)],
    )
    assert metrics["steps_1_repeats_1"]["by_length"]["12"][
        "exact_sequence_accuracy"
    ] == 1.0
    assert metrics["steps_2_repeats_2"]["by_length"]["12"][
        "exact_sequence_accuracy"
    ] == 1.0

    final = run_final_ca_evaluation(
        forward_context(model),
        eval_loaders,
        device="cpu",
        trained_ca_steps=2,
        trained_num_repeats=2,
        trained_pairs=[(1, 1), (2, 2)],
        internal_pairs=[(1, 1), (2, 2), (3, 3)],
    )
    assert set(final["in_distribution"]["by_pair"]) == {
        "steps_1_repeats_1",
        "steps_2_repeats_2",
    }
    assert set(final["internal_repeat_extrapolation"]) == {
        "steps_3_repeats_3"
    }


def test_external_rollout_accounts_for_steps_learned_per_model_call():
    dataset = Rule30Dataset(num_samples=4, num_cells=10, steps=2, seed=12)
    eval_loaders = {10: DataLoader(dataset, batch_size=2, shuffle=False)}
    model = IterativeRule30Oracle(default_repeats=2)

    results = run_final_ca_evaluation(
        forward_context(model),
        eval_loaders,
        device="cpu",
        trained_ca_steps=2,
        trained_num_repeats=2,
        external_ca_steps=[4],
    )

    rollout = results["external_rollout"]["steps_4"]
    assert rollout["model_calls"] == 2
    assert rollout["by_length"]["10"]["exact_sequence_accuracy"] == 1.0
    assert rollout["by_length"]["10"]["average_depth"] == 4.0


class DiagnosticRule30Oracle(torch.nn.Module):
    def forward(
        self,
        inputs,
        get_logits=False,
        return_all_logits=False,
        num_repeats=1,
        return_repeat_states=False,
        intervention_source_depth=None,
        intervention_input_ids=None,
    ):
        state = inputs
        repeat_states = [F.one_hot(state, num_classes=2).float()]
        if intervention_source_depth is None:
            for _ in range(num_repeats):
                state = rule30(state)
                repeat_states.append(F.one_hot(state, num_classes=2).float())
        else:
            state = rule30(intervention_input_ids)
        logits = logits_for_predictions(state) * 20.0
        result = {
            "logits": logits if get_logits else None,
            "average_depth": torch.as_tensor(num_repeats),
        }
        if return_repeat_states:
            result["repeat_states"] = repeat_states
        return result


class IdentityDiagnosticModel(torch.nn.Module):
    def forward(self, inputs, get_logits=False, num_repeats=1):
        logits = logits_for_predictions(inputs) * 20.0
        return {
            "logits": logits if get_logits else None,
            "average_depth": torch.as_tensor(num_repeats),
        }


def test_literal_example_metrics_distinguish_the_three_comparisons():
    input_row = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])
    dataloader = DataLoader(
        [{"input_id": input_row}],
        batch_size=1,
        shuffle=False,
    )

    diagnostics = evaluate_ca_repeat_horizon_diagnostics(
        forward_context(IdentityDiagnosticModel()),
        dataloader,
        device="cpu",
        max_repeats=2,
        target_horizons=range(3),
        num_examples=1,
    )

    horizon_one = rule30(input_row.unsqueeze(0)).squeeze(0)
    horizon_two = rule30(horizon_one.unsqueeze(0)).squeeze(0)
    expected_correct = int(input_row.eq(horizon_two).sum().item())
    transition_correct = int(input_row.eq(horizon_one).sum().item())
    repeat_two = diagnostics["examples"][0]["predictions"]["repeats_2"]

    assert repeat_two["expected_horizon"] == 2
    assert repeat_two["expected_correct_cells"] == expected_correct
    assert repeat_two["best_matching_horizon"] == 0
    assert repeat_two["best_matching_correct_cells"] == input_row.numel()
    assert repeat_two["decoded_transition_from_repeat"] == 1
    assert repeat_two["decoded_transition_correct_cells"] == transition_correct


def test_repeat_horizon_matrix_recurrence_hidden_states_and_literal_rows():
    dataset = Rule30Dataset(num_samples=5, num_cells=12, seed=31)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    diagnostics = evaluate_ca_repeat_horizon_diagnostics(
        forward_context(DiagnosticRule30Oracle()),
        dataloader,
        device="cpu",
        max_repeats=4,
        target_horizons=range(5),
        num_examples=1,
    )

    for repeats in range(1, 5):
        diagonal = diagnostics["repeat_horizon_matrix"][f"repeats_{repeats}"][
            f"steps_{repeats}"
        ]
        assert diagonal["cell_accuracy"] == 1.0
        assert diagonal["exact_sequence_accuracy"] == 1.0
        assert diagonal["matthews_correlation"] == 1.0
        assert diagnostics["best_matching_horizon"][f"repeats_{repeats}"][
            "by_matthews_correlation"
        ] == repeats
        recurrence = diagnostics["decoded_recurrence"][f"repeats_{repeats}"][
            "rule30_from_previous_decoded"
        ]
        assert recurrence["cell_accuracy"] == 1.0

    assert diagnostics["hidden_state_similarity"]["available"]
    assert math.isclose(
        diagnostics["hidden_state_similarity"]["similarity_matrix"][
            "repeats_3"
        ]["repeats_3"]["cosine_similarity"],
        1.0,
        rel_tol=1e-6,
    )
    example = diagnostics["examples"][0]
    for repeats in range(1, 5):
        prediction = example["predictions"][f"repeats_{repeats}"]
        assert prediction["expected_horizon"] == repeats
        assert prediction["expected_correct_cells"] == 12
        assert prediction["expected_total_cells"] == 12
        assert prediction["expected_cell_accuracy"] == 1.0
        assert prediction["expected_correct_mask"] == "|" * 12
        assert prediction["best_matching_horizon"] == repeats
        assert prediction["best_matching_correct_cells"] == 12
        assert prediction["best_matching_total_cells"] == 12
        assert prediction["best_matching_cell_accuracy"] == 1.0
        assert prediction["decoded_transition_from_repeat"] == repeats - 1
        assert prediction["decoded_transition_correct_cells"] == 12
        assert prediction["decoded_transition_total_cells"] == 12
        assert prediction["decoded_transition_cell_accuracy"] == 1.0

    rendered = format_ca_repeat_examples({"12": diagnostics}, step=100)
    assert "[step=100 length=12 example=0 repeat=4]" in rendered
    assert "expected target: Rule30 horizon 4" in rendered
    assert "expected-horizon match:\n12/12 cells = 1.0000" in rendered
    assert "correct mask:\n||||||||||||" in rendered
    assert "best-matching horizon:\nhorizon 4, 12/12 cells = 1.0000" in rendered
    assert (
        "decoded transition:\nprediction repeat 4 vs "
        "Rule30(prediction repeat 3)\n12/12 cells = 1.0000"
    ) in rendered


def test_loaded_checkpoint_evaluation_skips_unsupported_clean_state_probe(capsys):
    dataset = Rule30Dataset(num_samples=4, num_cells=8, seed=71)
    eval_loaders = {8: DataLoader(dataset, batch_size=2, shuffle=False)}
    result = evaluate_loaded_ca_checkpoint(
        forward_context(IterativeRule30Oracle()),
        eval_loaders,
        device="cpu",
        checkpoint_metadata={"step": 10},
        label="best_id",
        split_seed=200,
        samples_per_length=4,
        trained_ca_steps=1,
        trained_num_repeats=1,
        repeat_diagnostic_max_repeats=2,
        repeat_diagnostic_horizons=range(3),
        repeat_diagnostic_examples=0,
        forward_policy_metadata={"repeat_cache_policy": "full"},
    )

    assert result["checkpoint"] == {"step": 10}
    assert result["split"] == {
        "role": "independent_final_test",
        "seed": 200,
        "samples_per_length": 4,
    }
    assert result["repeat_diagnostics"]["8"]["max_repeats"] == 2
    assert result["preserved_cache_clean_state_transitions"] is None
    assert "Skipping preserved-cache clean-state transitions" in capsys.readouterr().out


def test_loaded_checkpoint_evaluation_runs_supported_clean_state_probe():
    dataset = Rule30Dataset(num_samples=4, num_cells=8, seed=72)
    eval_loaders = {8: DataLoader(dataset, batch_size=2, shuffle=False)}
    result = evaluate_loaded_ca_checkpoint(
        forward_context(DiagnosticRule30Oracle()),
        eval_loaders,
        device="cpu",
        checkpoint_metadata={"step": 20},
        label="best_extrapolation_strict",
        split_seed=201,
        samples_per_length=4,
        trained_ca_steps=1,
        trained_num_repeats=1,
        repeat_diagnostic_max_repeats=2,
        repeat_diagnostic_horizons=range(3),
        repeat_diagnostic_examples=0,
    )

    clean = result["preserved_cache_clean_state_transitions"]
    assert clean["8"]["max_transition_depth"] == 2
    assert clean["8"]["transitions"]["steps_1_to_2"]["metrics"][
        "cell_accuracy"
    ] == 1.0


def test_literal_rows_mark_an_unconfigured_expected_horizon():
    dataset = Rule30Dataset(num_samples=2, num_cells=8, seed=41)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    diagnostics = evaluate_ca_repeat_horizon_diagnostics(
        forward_context(DiagnosticRule30Oracle()),
        dataloader,
        device="cpu",
        max_repeats=2,
        target_horizons=(0, 2),
        num_examples=1,
    )

    repeat_one = diagnostics["examples"][0]["predictions"]["repeats_1"]
    assert "expected_target" not in repeat_one
    assert repeat_one["decoded_transition_from_repeat"] == 0
    assert repeat_one["decoded_transition_cell_accuracy"] == 1.0

    rendered = format_ca_repeat_examples({"8": diagnostics}, step=25)
    repeat_one_block = rendered.split(
        "[step=25 length=8 example=0 repeat=1]", 1
    )[1].split("[step=25 length=8 example=0 repeat=2]", 1)[0]
    assert (
        "expected target: Rule30 horizon 1\n"
        "not configured in diagnostic target horizons"
    ) in repeat_one_block
    assert "expected-horizon match:\nnot available" in repeat_one_block
