import argparse
import copy
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from cellular_automaton.ca_attention_diagnostics import (
    _compact_repeat_horizon_metrics,
    _requested_conditions,
    _subtract_metric_matrices,
    _to_cpu_tree,
    _validate_artifact_name,
    _validate_captured_shard,
    _validate_factorial_capture,
    _validate_operation_args,
    capture_factorial_transition_in_memory,
    capture_factorial_transition_batch,
    capture_first_condition_batch,
    complete_artifact,
    evaluate_factorial_full_split,
    initialize_artifact,
    mark_artifact_failed,
    parse_args,
    register_artifact_shard,
    reduce_factorial_diagnostic_values,
    resolve_model_args,
    validate_source_run,
)
from cellular_automaton.ca_gen import rule30
from cellular_automaton.ca_reporting import read_json


def _manifest(model_name="ca_cotf"):
    model = {
        "model": model_name,
        "n_layer_begin": 0,
        "n_layer": 1,
        "n_layer_end": 0,
        "n_embd": 64,
        "n_head": 4,
        "attention_mode": "bidirectional",
    }
    return {
        "run_id": "run_test",
        "status": "completed",
        "provenance": {"dataset": "rule30", "checkpoint_dir": "unused"},
        "model": model,
        "resolved_args": {**model, "config_format": "base", "dtype": "torch.bfloat16"},
    }


@pytest.mark.parametrize("model_name", ("ca_cotf", "ca_cotf_cache_attn"))
def test_validate_source_run_accepts_supported_models(tmp_path, model_name):
    manifest = _manifest(model_name)
    manifest["provenance"]["checkpoint_dir"] = str(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "checkpoint.pt").touch()
    args = argparse.Namespace(
        run_manifest=manifest_path,
        expected_run_id="run_test",
        checkpoint_name="checkpoint.pt",
    )

    with mock.patch(
        "cellular_automaton.ca_attention_diagnostics.validate_manifest",
        return_value=[],
    ):
        validated, checkpoint = validate_source_run(args)

    assert validated["model"]["model"] == model_name
    assert checkpoint == tmp_path / "checkpoint.pt"


def test_validate_source_run_rejects_unrelated_model(tmp_path):
    manifest = _manifest("but_full_depth")
    manifest["provenance"]["checkpoint_dir"] = str(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "checkpoint.pt").touch()
    args = argparse.Namespace(
        run_manifest=manifest_path,
        expected_run_id="run_test",
        checkpoint_name="checkpoint.pt",
    )

    with mock.patch(
        "cellular_automaton.ca_attention_diagnostics.validate_manifest",
        return_value=[],
    ), pytest.raises(ValueError, match="expected source model"):
        validate_source_run(args)


def test_resolve_model_args_preserves_default_and_accepts_override():
    manifest = _manifest()
    default_args = resolve_model_args(manifest)
    manual_args = resolve_model_args(manifest, attention_implementation="manual")

    assert default_args.model == "ca_cotf_cache_attn"
    assert getattr(default_args, "attention_implementation", "sdpa") == "sdpa"
    assert default_args.dtype is torch.bfloat16
    assert default_args.repeat_cache_window is None
    assert manual_args.attention_implementation == "manual"


def test_resolve_model_args_accepts_matching_cache_policy_copies():
    manifest = _manifest()
    manifest["resolved_args"]["repeat_cache_window"] = 8
    manifest["forward_policy"] = {"repeat_cache_window": 8}

    assert resolve_model_args(manifest).repeat_cache_window == 8


@pytest.mark.parametrize(
    ("resolved_window", "policy_window"),
    ((8, 2), (8, "missing"), ("missing", 8)),
)
def test_resolve_model_args_rejects_cache_policy_disagreement(
    resolved_window, policy_window
):
    manifest = _manifest()
    if resolved_window != "missing":
        manifest["resolved_args"]["repeat_cache_window"] = resolved_window
    if policy_window != "missing":
        manifest["forward_policy"] = {"repeat_cache_window": policy_window}

    with pytest.raises(ValueError, match="cache-window disagreement"):
        resolve_model_args(manifest)


def test_compact_metrics_and_manual_minus_sdpa():
    sdpa_result = {
        "repeat_horizon_matrix": {
            "repeats_1": {
                "steps_1": {
                    "cell_accuracy": 0.75,
                    "exact_sequence_accuracy": 0.5,
                    "loss": 0.4,
                    "ignored": 12,
                }
            }
        }
    }
    manual_result = {
        "repeat_horizon_matrix": {
            "repeats_1": {
                "steps_1": {
                    "cell_accuracy": 0.8,
                    "exact_sequence_accuracy": 0.4,
                    "loss": 0.35,
                }
            }
        }
    }

    sdpa = _compact_repeat_horizon_metrics(sdpa_result)
    manual = _compact_repeat_horizon_metrics(manual_result)
    difference = _subtract_metric_matrices(manual, sdpa)

    assert set(sdpa) == {"cell_accuracy", "exact_sequence_accuracy", "loss"}
    assert difference["cell_accuracy"]["repeats_1"]["steps_1"] == pytest.approx(0.05)
    assert difference["exact_sequence_accuracy"]["repeats_1"]["steps_1"] == pytest.approx(-0.1)
    assert difference["loss"]["repeats_1"]["steps_1"] == pytest.approx(-0.05)


@pytest.mark.parametrize("name", ("run84_baseline", "run-84.manual.v1", "A1"))
def test_validate_artifact_name_accepts_safe_basename(name):
    assert _validate_artifact_name(name) == name


@pytest.mark.parametrize("name", ("", ".", "../escape", "with space", "/absolute"))
def test_validate_artifact_name_rejects_unsafe_name(name):
    with pytest.raises(ValueError, match="artifact name"):
        _validate_artifact_name(name)


def test_repeat_cache_window_cli_requires_a_positive_integer():
    required = [
        "--run-manifest",
        "run_manifest.json",
        "--expected-run-id",
        "run_test",
        "--checkpoint-name",
        "checkpoint.pt",
        "--expected-step",
        "20",
    ]
    parsed = parse_args([*required, "--repeat-cache-window", "12"])
    assert parsed.repeat_cache_window == 12
    with pytest.raises(SystemExit):
        parse_args([*required, "--repeat-cache-window", "0"])


def test_factorial_cli_is_distinct_and_requires_its_source_depth():
    required = [
        "--run-manifest", "run_manifest.json", "--expected-run-id", "run_test",
        "--checkpoint-name", "checkpoint.pt", "--expected-step", "20",
    ]
    parsed = parse_args(
        [*required, "--capture-factorial-transition-batch",
         "--evaluate-factorial-full-split",
         "--intervention-source-depth", "12"]
    )
    _validate_operation_args(parsed)
    assert parsed.intervention_source_depth == 12
    assert not parsed.capture_first_condition_batch
    assert parsed.evaluate_factorial_full_split

    raw_only = parse_args(
        [*required, "--capture-factorial-transition-batch",
         "--intervention-source-depth", "12"]
    )
    full_split_only = parse_args(
        [*required, "--evaluate-factorial-full-split",
         "--intervention-source-depth", "12"]
    )
    _validate_operation_args(raw_only)
    _validate_operation_args(full_split_only)

    missing_depth = parse_args([*required, "--capture-factorial-transition-batch"])
    with pytest.raises(ValueError, match="must be supplied together"):
        _validate_operation_args(missing_depth)
    with pytest.raises(SystemExit):
        parse_args([*required, "--intervention-source-depth", "-1"])
    with pytest.raises(SystemExit):
        parse_args(
            [*required, "--capture-first-condition-batch",
             "--capture-factorial-transition-batch"]
        )
    parsed.repeat_cache_window = 2
    with pytest.raises(ValueError, match="requires --capture-first"):
        _validate_operation_args(parsed)


def _artifact_inputs(tmp_path, artifact_name="storage_smoke"):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"fixed checkpoint identity")
    args = argparse.Namespace(
        artifact_name=artifact_name,
        output_root=tmp_path / "artifacts",
        run_manifest=tmp_path / "run_manifest.json",
        expected_step=20,
        device="cpu",
        probe_forward=True,
        capture_first_condition_batch=False,
        capture_factorial_transition_batch=False,
        evaluate_factorial_full_split=False,
        intervention_source_depth=None,
        repeat_cache_window=None,
        verify_full_cache_none=False,
        compare_repeat_horizon_backends=False,
    )
    model_args = argparse.Namespace(
        dtype=torch.bfloat16,
        model="ca_cotf_cache_attn",
        attention_implementation="manual",
    )
    dataset_info = {
        "split": "validation",
        "seed": 1000067,
        "num_samples": 4096,
        "num_cells": 64,
        "batch_size": 128,
        "shuffle": False,
    }
    return args, _manifest(), checkpoint_path, model_args, dataset_info


def test_artifact_initialization_storage_probe_and_completion(tmp_path):
    args, manifest, checkpoint, model_args, dataset_info = _artifact_inputs(tmp_path)
    context = initialize_artifact(
        args, manifest, checkpoint, 20, model_args, dataset_info, 20
    )

    running = read_json(context["manifest_path"])
    probe = torch.load(
        context["directory"] / "storage_probe.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert running["status"] == "running"
    assert running["storage_validation"]["status"] == "passed"
    assert running["source"]["checkpoint"]["expected_step"] == 20
    assert len(running["source"]["checkpoint"]["sha256"]) == 64
    assert running["evaluation"]["dataset"] == dataset_info
    assert running["evaluation"]["conditions"] == [
        {
            "condition_id": "manual_full",
            "attention_implementation": "manual",
            "cache_policy": "full",
            "cache_window": None,
            "repeat_cache_window": None,
            "attention_normalizer": "softmax",
            "normalization_intervention": "none",
            "logit_scaling": "none",
            "probability_transform": "none",
        }
    ]
    assert probe["bf16_values"].dtype is torch.bfloat16
    assert probe["fp32_values"].dtype is torch.float32
    assert probe["populated_mask"].dtype is torch.bool
    assert probe["example_ids"].dtype is torch.int64

    complete_artifact(context, {"result": "ok"})
    completed = read_json(context["manifest_path"])
    assert completed["status"] == "completed"
    assert read_json(context["directory"] / "summary.json") == {"result": "ok"}
    assert len(completed["summary"]["sha256"]) == 64


def test_artifact_collision_is_refused(tmp_path):
    args, manifest, checkpoint, model_args, dataset_info = _artifact_inputs(tmp_path)
    initialize_artifact(args, manifest, checkpoint, 20, model_args, dataset_info, 20)
    with pytest.raises(ValueError, match="already exists"):
        initialize_artifact(
            args, manifest, checkpoint, 20, model_args, dataset_info, 20
        )


def test_artifact_failure_is_recorded(tmp_path):
    args, manifest, checkpoint, model_args, dataset_info = _artifact_inputs(
        tmp_path, artifact_name="failed_probe"
    )
    context = initialize_artifact(
        args, manifest, checkpoint, 20, model_args, dataset_info, 20
    )
    mark_artifact_failed(context, RuntimeError("synthetic failure"))

    failed = read_json(context["manifest_path"])
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "type": "RuntimeError",
        "message": "synthetic failure",
    }


def test_backend_comparison_records_both_conditions():
    args = argparse.Namespace(
        probe_forward=False,
        capture_first_condition_batch=False,
        repeat_cache_window=None,
        compare_repeat_horizon_backends=True,
    )
    model_args = argparse.Namespace(attention_implementation="manual")
    assert [
        condition["condition_id"]
        for condition in _requested_conditions(args, model_args)
    ] == ["sdpa_full", "manual_full"]


class _FakeDiagnosticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_embd=4, n_head=2)
        self.transformer = SimpleNamespace(
            h_mid=[object()],
            h_end=[],
            ln_f=torch.nn.Identity(),
        )
        self.lm_head = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
            )
        self.received_repeat_cache_windows = []
        self.received_calls = []

    def forward(
        self,
        inputs,
        *,
        num_repeats,
        return_repeat_states,
        return_attention_diagnostics,
        **kwargs,
    ):
        batch_size, tokens = inputs.shape
        repeat_cache_window = kwargs.get("repeat_cache_window")
        self.received_calls.append(
            {
                "num_repeats": num_repeats,
                "return_attention_diagnostics": return_attention_diagnostics,
                **kwargs,
            }
        )
        self.received_repeat_cache_windows.append(
            ("repeat_cache_window" in kwargs, repeat_cache_window)
        )
        states = [
            torch.full(
                (batch_size, tokens, self.config.n_embd),
                float(repeat),
                device=inputs.device,
            )
            for repeat in range(num_repeats + 1)
        ]
        intervention_source = kwargs.get("intervention_source_depth")
        if intervention_source is not None:
            for repeat in range(intervention_source + 1, num_repeats + 1):
                states[repeat] = -states[repeat]
        records = []
        for repeat in range(1, num_repeats + 1):
            active_window = repeat_cache_window
            reset_source = kwargs.get("cache_reset_source_depth")
            if reset_source is not None and repeat > reset_source:
                post_reset_repeats = repeat - reset_source
                active_window = (
                    post_reset_repeats
                    if repeat_cache_window is None
                    else min(post_reset_repeats, repeat_cache_window)
                )
            visible_repeats = (
                repeat
                if active_window is None
                else min(repeat, active_window)
            )
            repeat_mass = torch.zeros(
                (batch_size, self.config.n_head, tokens, repeat),
                device=inputs.device,
                dtype=torch.bfloat16,
            )
            repeat_mass[..., -visible_repeats:] = 1.0 / visible_repeats
            repeat_logsumexp_valid = torch.zeros_like(
                repeat_mass, dtype=torch.bool
            )
            repeat_logsumexp_valid[..., -visible_repeats:] = True
            per_query_metric = torch.full(
                (batch_size, self.config.n_head, tokens),
                float(repeat),
                device=inputs.device,
            )
            source_norm = torch.arange(
                1, repeat + 1, device=inputs.device, dtype=torch.float32
            ).view(1, 1, repeat, 1).expand(
                batch_size, self.config.n_head, repeat, tokens
            )
            contribution_norm = source_norm.transpose(2, 3).expand(
                batch_size, self.config.n_head, tokens, repeat
            )
            records.append(
                {
                    "repeat_index": repeat,
                    "middle_layer_index": 0,
                    "cached_repeats": repeat,
                    "repeat_mass": repeat_mass,
                    "visible_repeat_count": torch.full(
                        (batch_size, self.config.n_head, tokens),
                        visible_repeats,
                        device=inputs.device,
                        dtype=torch.long,
                    ),
                    "visible_token_count": torch.full(
                        (batch_size, self.config.n_head, tokens),
                        visible_repeats * tokens,
                        device=inputs.device,
                        dtype=torch.long,
                    ),
                    "repeat_logsumexp": torch.zeros_like(repeat_mass),
                    "repeat_logsumexp_valid": repeat_logsumexp_valid,
                    "query_norm": per_query_metric,
                    "key_norm": source_norm,
                    "value_norm": source_norm * 10,
                    "logit_std": per_query_metric,
                    "logit_spread": per_query_metric * 2,
                    "top_two_logit_gap": per_query_metric / 2,
                    "top_two_logit_gap_valid": torch.ones_like(
                        per_query_metric, dtype=torch.bool
                    ),
                    "normalised_repeat_entropy": per_query_metric * 0,
                    "normalised_attention_entropy": per_query_metric * 0,
                    "effective_support_fraction": per_query_metric * 0 + 1,
                    "maximum_attention_probability": per_query_metric * 0 + 1,
                    "block_contribution_norm": contribution_norm,
                }
            )
        return {
            "logits": self.lm_head(states[-1]),
            "repeat_states": states if return_repeat_states else None,
            "attention_diagnostics": (
                records if return_attention_diagnostics else None
            ),
        }


def test_nested_tensor_transfer_detaches_and_preserves_dtype():
    original = {
        "values": torch.tensor([1.0, 2.0], requires_grad=True),
        "nested": [torch.tensor([3.0], dtype=torch.bfloat16)],
    }
    transferred = _to_cpu_tree(original)

    assert transferred["values"].device.type == "cpu"
    assert not transferred["values"].requires_grad
    assert transferred["nested"][0].dtype is torch.bfloat16


def test_factorial_transition_captures_four_paired_target_conditions_in_memory():
    model = _FakeDiagnosticModel()
    inputs = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])

    result = capture_factorial_transition_in_memory(
        model,
        inputs,
        argparse.Namespace(dtype=torch.bfloat16),
        "cpu",
        source_depth=2,
        max_repeats=5,
        baseline_repeat_cache_window=2,
    )

    assert len(model.received_calls) == 4
    assert all(call["num_repeats"] == 5 for call in model.received_calls)
    assert all(
        call["return_attention_diagnostics"] is True
        for call in model.received_calls
    )
    assert all(call["repeat_cache_window"] == 2 for call in model.received_calls)
    clean_calls = [
        call for call in model.received_calls if "intervention_input_ids" in call
    ]
    expected_clean = rule30(rule30(inputs))
    assert len(clean_calls) == 2
    assert all(
        torch.equal(call["intervention_input_ids"], expected_clean)
        for call in clean_calls
    )
    assert result["source_depth"] == 2
    assert result["target_depth"] == 3
    assert result["max_repeats"] == 5
    assert result["post_intervention_depths"] == [3, 4, 5]
    assert result["experiment_kind"] == "clean_state_cache_reset_factorial"
    assert result["factorial_capture_schema_version"] == 2
    assert result["trained_cache_window"] == 2
    assert model.training
    assert torch.equal(result["clean_source"], expected_clean)
    assert torch.equal(result["clean_target"], rule30(expected_clean))
    expected_targets = torch.stack(
        [rule30(expected_clean), rule30(rule30(expected_clean)),
         rule30(rule30(rule30(expected_clean)))], dim=1
    )
    assert torch.equal(result["clean_targets_by_depth"], expected_targets)
    prefix = result["shared_prefix"]
    expected_prefix_targets = torch.stack(
        [inputs, rule30(inputs), expected_clean], dim=1
    )
    assert prefix["depths"] == [0, 1, 2]
    assert torch.equal(prefix["clean_targets_by_depth"], expected_prefix_targets)
    assert prefix["repeat_states"].shape == (2, 3, 4, 4)
    assert prefix["decoded_logits"].shape == (2, 3, 4, 2)
    expected_prefix_logits = torch.stack(
        [model.lm_head(state) for state in prefix["repeat_states"].unbind(1)],
        dim=1,
    )
    assert torch.equal(prefix["decoded_logits"], expected_prefix_logits)
    assert [
        record["repeat_index"] for record in prefix["attention_records"]
    ] == [1, 2]
    assert [
        int(record["visible_repeat_count"][0, 0, 0])
        for record in prefix["attention_records"]
    ] == [1, 2]
    assert set(result["conditions"]) == {
        "free_baseline", "clean_baseline", "free_reset", "clean_reset"}
    for name, payload in result["conditions"].items():
        expected_windows = [1, 2, 2] if name.endswith("reset") else [2, 2, 2]
        assert payload["condition_id"] == name
        assert payload["clean_state_injected"] == name.startswith("clean")
        assert payload["cache_reset"] == name.endswith("reset")
        assert payload["intervention_source_depth"] == 2
        assert payload["effective_windows"] == expected_windows
        assert len(payload["attention_records"]) == 3
        assert [
            int(record["visible_repeat_count"][0, 0, 0])
            for record in payload["attention_records"]
        ] == expected_windows
        assert payload["repeat_states"].shape == (2, 3, 4, 4)
        assert payload["decoded_logits"].shape == (2, 3, 4, 2)
        assert payload["repeat_states"].device.type == "cpu"
        assert not payload["repeat_states"].requires_grad
        assert "shared_prefix" not in payload

    _validate_factorial_capture(result, model)
    legacy = copy.deepcopy(result)
    legacy["factorial_capture_schema_version"] = 1
    del legacy["shared_prefix"]
    _validate_factorial_capture(legacy, model)
    with pytest.raises(ValueError, match="factorial validator"):
        _validate_captured_shard(result, model, 5)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["conditions"]["clean_reset"].__setitem__(
            "clean_state_injected", False), "metadata"),
        (lambda value: value.__setitem__("trained_cache_window", 0), "cache window"),
        (lambda value: value["conditions"]["free_reset"].__setitem__(
            "effective_windows", [1, 2, 3]), "cache windows"),
        (lambda value: value["shared_prefix"].__setitem__(
            "depths", [0, 1]), "shared-prefix depths"),
        (lambda value: value["shared_prefix"].__setitem__(
            "decoded_logits", value["shared_prefix"]["decoded_logits"][:, :2]
        ), "shared-prefix decoded_logits"),
    ),
)
def test_factorial_validator_rejects_mislabeled_or_inconsistent_capture(
    mutation, message
):
    model = _FakeDiagnosticModel()
    capture = capture_factorial_transition_in_memory(
        model,
        torch.tensor([[0, 1, 0, 1]]),
        argparse.Namespace(dtype=torch.bfloat16),
        "cpu",
        source_depth=2,
        max_repeats=5,
        baseline_repeat_cache_window=2,
    )
    malformed = copy.deepcopy(capture)
    mutation(malformed)

    with pytest.raises(ValueError, match=message):
        _validate_factorial_capture(malformed, model)


@pytest.mark.parametrize("divergence", ("states", "attention"))
def test_factorial_capture_rejects_divergent_pre_intervention_prefix(divergence):
    class DivergentPrefixModel(_FakeDiagnosticModel):
        def forward(self, inputs, **kwargs):
            outputs = super().forward(inputs, **kwargs)
            if "intervention_source_depth" in kwargs:
                if divergence == "states":
                    outputs["repeat_states"][0] += 1
                else:
                    outputs["attention_diagnostics"][0]["query_norm"] += 1
            return outputs

    with pytest.raises(RuntimeError, match="pre_intervention"):
        capture_factorial_transition_in_memory(
            DivergentPrefixModel(),
            torch.tensor([[0, 1, 0, 1]]),
            argparse.Namespace(dtype=torch.bfloat16),
            "cpu",
            source_depth=2,
            max_repeats=4,
            baseline_repeat_cache_window=2,
        )


def test_factorial_shared_prefix_supports_source_depth_zero():
    inputs = torch.tensor([[0, 1, 0, 1]])
    capture = capture_factorial_transition_in_memory(
        _FakeDiagnosticModel(),
        inputs,
        argparse.Namespace(dtype=torch.bfloat16),
        "cpu",
        source_depth=0,
        max_repeats=2,
        baseline_repeat_cache_window=2,
    )

    prefix = capture["shared_prefix"]
    assert prefix["depths"] == [0]
    assert torch.equal(prefix["clean_targets_by_depth"][:, 0], inputs)
    assert prefix["repeat_states"].shape[1] == 1
    assert prefix["decoded_logits"].shape[1] == 1
    assert prefix["attention_records"] == []


def test_factorial_reducer_combines_visible_history_logsumexp():
    capture = capture_factorial_transition_in_memory(
        _FakeDiagnosticModel(), torch.tensor([[0, 1, 0, 1]]),
        argparse.Namespace(dtype=torch.bfloat16), "cpu", 1, 3,
        baseline_repeat_cache_window=2,
    )
    payload = capture["conditions"]["free_reset"]
    last_record = payload["attention_records"][-1]
    scores = last_record["repeat_logsumexp"].clone()
    scores[..., -2] = torch.log(torch.tensor(2.0))
    scores[..., -1] = torch.log(torch.tensor(3.0))
    expected_margin = float((scores[..., -1] - scores[..., -2]).float().mean())
    last_record["repeat_logsumexp"] = scores
    reduced = reduce_factorial_diagnostic_values(payload, [2, 3])

    margin = reduced["current_minus_visible_history_logsumexp"]
    assert reduced["current_key_norm"].tolist() == [[2.0, 3.0]]
    assert torch.isnan(reduced["visible_history_key_norm"][:, 0]).all()
    assert reduced["visible_history_key_norm"][:, 1].item() == 2.0
    assert torch.isnan(margin[:, 0]).all()
    assert margin[:, 1].item() == pytest.approx(expected_margin)


def test_factorial_batch_is_persisted_with_manifest_design(tmp_path):
    args, manifest, checkpoint, model_args, dataset_info = _artifact_inputs(
        tmp_path, artifact_name="factorial"
    )
    args.probe_forward = False
    args.capture_factorial_transition_batch = True
    args.intervention_source_depth = 2
    model_args.repeat_cache_window = 2
    args.artifact_name = None
    with pytest.raises(ValueError, match="requires --artifact-name"):
        initialize_artifact(
            args, manifest, checkpoint, 20, model_args, dataset_info, 5
        )
    args.artifact_name = "factorial"
    context = initialize_artifact(
        args, manifest, checkpoint, 20, model_args, dataset_info, 5
    )
    model = _FakeDiagnosticModel()
    batch = {
        "input_id": torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]]),
        "example_id": torch.tensor([7, 9]),
    }
    info = capture_factorial_transition_batch(
        model, [batch], model_args, "cpu", 2, 5, context["directory"], 2
    )
    register_artifact_shard(context, info)

    shard = torch.load(
        context["directory"] / info["path"], map_location="cpu", weights_only=False
    )
    recorded = read_json(context["manifest_path"])
    assert torch.equal(shard["example_ids"], batch["example_id"])
    assert info["condition_ids"] == [
        "free_baseline", "clean_baseline", "free_reset", "clean_reset"
    ]
    assert recorded["evaluation"]["factorial_design"]["trained_cache_window"] == 2
    assert recorded["shards"] == [info]


def test_factorial_full_split_aggregates_conditions_and_paired_effects():
    model = _FakeDiagnosticModel()
    batches = [
        {"input_id": torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]])},
        {"input_id": torch.tensor([[1, 0, 1, 0]])},
    ]
    result = evaluate_factorial_full_split(
        model,
        batches,
        argparse.Namespace(dtype=torch.bfloat16),
        "cpu",
        source_depth=2,
        max_repeats=4,
        trained_cache_window=2,
    )

    assert result["num_batches"] == 2
    assert result["num_examples"] == 3
    assert result["target_depths"] == [3, 4]
    assert result["pairing_validation"]["batches_exactly_validated"] == 2
    assert result["pairing_validation"]["pre_intervention_attention"] == (
        "not_collected"
    )
    assert result["diagnostic_collection"]["attention_diagnostics"] is False
    assert all(
        call["return_attention_diagnostics"] is False
        for call in model.received_calls
    )
    assert set(result["conditions"]) == {
        "free_baseline", "clean_baseline", "free_reset", "clean_reset"
    }
    for effect in result["paired_effects"].values():
        assert effect["sign"] == "left_minus_right"
        left = result["conditions"][effect["left_condition"]]["by_target_depth"]
        right = result["conditions"][effect["right_condition"]]["by_target_depth"]
        for depth, metrics in effect["by_target_depth"].items():
            for name, value in metrics.items():
                if left[depth][name] is None or right[depth][name] is None:
                    assert value is None
                else:
                    assert value == pytest.approx(
                        left[depth][name] - right[depth][name]
                    )
    assert result["paired_effects"]["clean_state_with_history"][
        "by_target_depth"
    ]["3"]["loss"] != 0
    for depth, metrics in result["state_history_interaction"][
        "by_target_depth"
    ].items():
        after_reset = result["paired_effects"]["clean_state_after_reset"][
            "by_target_depth"
        ][depth]
        with_history = result["paired_effects"]["clean_state_with_history"][
            "by_target_depth"
        ][depth]
        for name, value in metrics.items():
            if after_reset[name] is None or with_history[name] is None:
                assert value is None
            else:
                assert value == pytest.approx(
                    after_reset[name] - with_history[name]
                )
    for condition in result["conditions"].values():
        assert set(condition["by_target_depth"]["3"]) == {
            "cell_accuracy", "exact_sequence_accuracy", "loss"
        }
    assert "state_pair_distances" not in result
    json.dumps(result, allow_nan=False)


def test_factorial_full_split_rejects_divergent_pre_intervention_states():
    class DivergentPrefixModel(_FakeDiagnosticModel):
        def forward(self, inputs, **kwargs):
            outputs = super().forward(inputs, **kwargs)
            if "cache_reset_source_depth" in kwargs:
                outputs["repeat_states"][0] += 1
            return outputs

    with pytest.raises(RuntimeError, match="pre_intervention_states"):
        evaluate_factorial_full_split(
            DivergentPrefixModel(),
            [{"input_id": torch.tensor([[0, 1, 0, 1]])}],
            argparse.Namespace(dtype=torch.bfloat16),
            "cpu",
            source_depth=2,
            max_repeats=4,
            trained_cache_window=2,
        )


@pytest.mark.parametrize(
    ("max_repeats", "expected_error"),
    ((2, ValueError), (True, TypeError)),
)
def test_factorial_transition_rejects_invalid_max_repeats(
    max_repeats, expected_error
):
    with pytest.raises(expected_error, match="max_repeats"):
        capture_factorial_transition_in_memory(
            _FakeDiagnosticModel(),
            torch.zeros((2, 4), dtype=torch.long),
            argparse.Namespace(dtype=torch.bfloat16),
            "cpu",
            source_depth=2,
            max_repeats=max_repeats,
        )


def test_capture_first_full_condition_writes_real_shard_and_verifies_none(tmp_path):
    model = _FakeDiagnosticModel()
    batch = {
        "input_id": torch.tensor(
            [[0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.long
        ),
        "example_id": torch.tensor([0, 1], dtype=torch.long),
    }
    artifact_dir = tmp_path / "artifact"
    info = capture_first_condition_batch(
        model,
        [batch],
        argparse.Namespace(dtype=torch.bfloat16),
        "cpu",
        3,
        artifact_dir,
        repeat_cache_window=None,
        verify_full_cache_none=True,
    )

    shard_path = artifact_dir / info["path"]
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    with pytest.raises(ValueError, match="not a clean-state/cache-reset"):
        _validate_factorial_capture(shard, model)
    assert info["validation"] == "passed"
    assert info["attention_records"] == 3
    assert info["tensor_bytes"] > 0
    assert info["serialized_size_bytes"] == shard_path.stat().st_size
    assert info["full_cache_none_equivalence"] == {
        "status": "passed",
        "comparison": "explicit_none_vs_omitted",
        "exact": True,
    }
    assert model.received_repeat_cache_windows == [(False, None), (True, None)]
    assert shard["condition"]["condition_id"] == "manual_full"
    assert shard["condition"]["repeat_cache_window"] is None
    assert tuple(shard["targets_by_horizon"].shape) == (2, 4, 4)
    assert tuple(shard["decoded_logits_by_repeat"].shape) == (2, 4, 4, 2)
    assert tuple(shard["repeat_states"].shape) == (2, 4, 4, 4)
    assert [
        record["repeat_mass"].shape[-1]
        for record in shard["attention_records"]
    ] == [1, 2, 3]
    assert all(
        tensor.device.type == "cpu" and not tensor.requires_grad
        for tensor in [
            shard["decoded_logits_by_repeat"],
            shard["repeat_states"],
            *[
                record["repeat_mass"]
                for record in shard["attention_records"]
            ],
        ]
    )
    assert shard["attention_records"][0]["repeat_mass"].dtype is torch.bfloat16

    shard["attention_records"].pop()
    with pytest.raises(ValueError, match="record count"):
        _validate_captured_shard(shard, model, 3)


def test_recent_condition_is_requested_and_recorded(tmp_path):
    args = argparse.Namespace(
        probe_forward=False,
        capture_first_condition_batch=True,
        repeat_cache_window=2,
        compare_repeat_horizon_backends=False,
    )
    model_args = argparse.Namespace(attention_implementation="sdpa")
    conditions = _requested_conditions(args, model_args)
    assert conditions == [
        {
            "condition_id": "manual_recent_2",
            "attention_implementation": "manual",
            "cache_policy": "recent",
            "cache_window": 2,
            "repeat_cache_window": 2,
            "attention_normalizer": "softmax",
            "normalization_intervention": "none",
            "logit_scaling": "none",
            "probability_transform": "none",
        }
    ]

    model = _FakeDiagnosticModel()
    batch = {
        "input_id": torch.tensor(
            [[0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.long
        ),
        "example_id": torch.tensor([0, 1], dtype=torch.long),
    }
    info = capture_first_condition_batch(
        model,
        [batch],
        argparse.Namespace(dtype=torch.bfloat16),
        "cpu",
        3,
        tmp_path / "artifact",
        repeat_cache_window=2,
    )
    shard = torch.load(
        tmp_path / "artifact" / info["path"],
        map_location="cpu",
        weights_only=False,
    )
    assert model.received_repeat_cache_windows == [(True, 2)]
    assert shard["condition"] == conditions[0]
    assert torch.equal(
        shard["attention_records"][-1]["visible_repeat_count"],
        torch.full((2, 2, 4), 2, dtype=torch.long),
    )
