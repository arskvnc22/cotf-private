import argparse
import copy
from pathlib import Path

import pytest
import torch

from cellular_automaton import ca_checkpoint_eval
from cellular_automaton.ca_analyze import discover_runs, parse_filter
from cellular_automaton.ca_checkpoint_eval import (
    CHECKPOINT_TYPES,
    _read_checkpoint_metadata,
    build_evaluation_manifest,
    load_checkpoint_weights,
    read_validated_manifest,
    read_validated_training_stats,
    resolve_source_args,
    run_standalone_evaluation,
)
from cellular_automaton.ca_gen import MaterializedRule30Dataset
from cellular_automaton.ca_main import make_ca_fixed_loaders
from cellular_automaton.ca_reporting import (
    read_json,
    validate_manifest,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPOSITORY_ROOT / "iridis" / "ca-rule30" / "runs"
REPRESENTATIVE_MANIFEST = (
    RUNS_ROOT / "run_115__but_full_depth" / "run_manifest.json"
)


def _source_manifest(tmp_path, *, status="failed"):
    manifest = copy.deepcopy(read_json(REPRESENTATIVE_MANIFEST))
    manifest["run_id"] = "run_900__but_full_depth"
    manifest["status"] = status
    manifest["provenance"]["run_dir"] = str(tmp_path / manifest["run_id"])
    manifest["provenance"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    return manifest


def test_every_checked_in_rule30_manifest_has_no_schema_errors():
    manifest_paths = sorted(RUNS_ROOT.glob("run_*/run_manifest.json"))
    assert manifest_paths
    for manifest_path in manifest_paths:
        manifest = read_validated_manifest(manifest_path)
        assert validate_manifest(manifest) == [], manifest_path


def test_every_checked_in_but_manifest_reconstructs_without_errors():
    checked = 0
    for manifest_path in sorted(RUNS_ROOT.glob("run_*/run_manifest.json")):
        manifest = read_validated_manifest(manifest_path)
        if manifest.get("model", {}).get("model") != "but_full_depth":
            continue
        model_args, fallbacks = resolve_source_args(manifest, "cpu")
        assert model_args.model == "but_full_depth", manifest_path
        assert isinstance(fallbacks, list), manifest_path
        checked += 1
    assert checked


def test_historical_optional_manifest_sections_are_compatible(tmp_path):
    manifest = _source_manifest(tmp_path)
    manifest.pop("annotations", None)
    manifest.pop("forward_policy", None)
    manifest.pop("evaluation", None)
    path = tmp_path / "historical_manifest.json"
    write_json(path, manifest)

    loaded = read_validated_manifest(path)
    model_args, fallbacks = resolve_source_args(loaded, "cpu")

    assert model_args.model == "but_full_depth"
    assert model_args.device == torch.device("cpu")
    assert model_args.ca_train_pairs == [[1, 1], [2, 2], [3, 3], [4, 4]]
    assert isinstance(fallbacks, list)


@pytest.mark.parametrize(
    "mutation, expected",
    (
        (lambda manifest: manifest.pop("resolved_args"), "resolved_args"),
        (lambda manifest: manifest.update(schema_version=999), "schema_version"),
    ),
)
def test_manifest_validation_errors_stop_evaluation_before_use(
    tmp_path, mutation, expected
):
    manifest = _source_manifest(tmp_path)
    mutation(manifest)
    path = tmp_path / "invalid_manifest.json"
    write_json(path, manifest)

    with pytest.raises(ValueError, match=expected):
        read_validated_manifest(path)


def test_manifest_reader_rejects_invalid_json(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot read source run manifest"):
        read_validated_manifest(path)


def test_checkpoint_metadata_supports_legacy_id_and_strict_names(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for metadata_name, checkpoint_name, step in (
        ("best.json", "best.pt", 10),
        ("best_extrapolation.json", "best_extrapolation.pt", 20),
        (
            "best_extrapolation_unconstrained.json",
            "best_extrapolation_unconstrained.pt",
            30,
        ),
    ):
        write_json(
            checkpoint_dir / metadata_name,
            {"step": step, "checkpoint": checkpoint_name},
        )
        (checkpoint_dir / checkpoint_name).touch()

    id_metadata, id_path = _read_checkpoint_metadata(
        checkpoint_dir, "best_id"
    )
    strict_metadata, strict_path = _read_checkpoint_metadata(
        checkpoint_dir, "best_extrapolation_strict"
    )
    unconstrained_metadata, unconstrained_path = _read_checkpoint_metadata(
        checkpoint_dir, "best_extrapolation_unconstrained"
    )

    assert (id_metadata["step"], id_path.name) == (10, "best.pt")
    assert (strict_metadata["step"], strict_path.name) == (
        20,
        "best_extrapolation.pt",
    )
    assert (unconstrained_metadata["step"], unconstrained_path.name) == (
        30,
        "best_extrapolation_unconstrained.pt",
    )


def test_all_three_checkpoint_types_are_required(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    with pytest.raises(ValueError, match="best_extrapolation_unconstrained"):
        _read_checkpoint_metadata(
            checkpoint_dir, "best_extrapolation_unconstrained"
        )


def test_checkpoint_step_mismatch_is_rejected(tmp_path):
    model = torch.nn.Linear(2, 2)
    checkpoint_path = tmp_path / "best_id.pt"
    torch.save({"model": model.state_dict(), "itr": 9}, checkpoint_path)

    with pytest.raises(ValueError, match="step mismatch"):
        load_checkpoint_weights(
            model,
            checkpoint_path,
            {"step": 10},
            torch.device("cpu"),
        )


def test_training_statistics_require_normal_validation_history(tmp_path):
    write_json(tmp_path / "training_stats.json", {"train": [], "eval": {}})
    stats = read_validated_training_stats(tmp_path)
    assert stats == {"train": [], "eval": {}}

    write_json(tmp_path / "training_stats.json", {"train": [], "eval": []})
    with pytest.raises(ValueError, match="eval mapping"):
        read_validated_training_stats(tmp_path)


def test_fixed_test_loader_uses_normal_length_offset_seed():
    args = argparse.Namespace(
        device=torch.device("cpu"),
        ca_data_mode="materialized",
        ca_eval_batch_size=2,
        batch_size=4,
        ca_eval_num_cells=[8],
        ca_steps=1,
        ca_bernoulli_p=0.5,
        ca_num_workers=0,
    )
    loaders = make_ca_fixed_loaders(args, base_seed=200, num_samples=3)
    expected = MaterializedRule30Dataset(
        num_samples=3,
        num_cells=8,
        steps=1,
        bernoulli_p=0.5,
        seed=208,
    )

    assert torch.equal(
        loaders[8].dataset[0]["input_id"], expected[0]["input_id"]
    )
    assert loaders[8].batch_size == 2


def test_generated_evaluation_manifest_is_analyzer_compatible(tmp_path):
    source = _source_manifest(tmp_path)
    output_dir = tmp_path / "run_901__but_full_depth_eval"
    artifact_dir = tmp_path / "artifact"
    checkpoint_dir = tmp_path / "checkpoints"
    manifest = build_evaluation_manifest(
        source,
        tmp_path / "source" / "run_manifest.json",
        output_dir,
        artifact_dir,
        checkpoint_dir,
    )

    assert validate_manifest(manifest) == []
    assert manifest["run_id"] == output_dir.name
    assert manifest["model"]["model"] == "but_full_depth"
    assert "standalone_eval" in manifest["annotations"]["tags"]
    assert manifest["provenance"]["evaluation_only"] is True
    assert manifest["standalone_evaluation"]["checkpoint_types"] == list(
        CHECKPOINT_TYPES
    )


class _FakeBut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, inputs, *, num_repeats=None):
        return {"logits": inputs, "num_repeats": num_repeats}


def _fake_checkpoint_analysis(metadata):
    metric = {
        "loss": 0.1,
        "cell_accuracy": 0.9,
        "exact_sequence_accuracy": 0.2,
    }
    return {
        "checkpoint": metadata,
        "forward_policy": {"repeat_cache_policy": "full"},
        "split": {
            "role": "independent_final_test",
            "seed": 2000003,
            "samples_per_length": 4096,
        },
        "task_metrics": {
            "in_distribution": {
                "training_pairs": [[1, 1]],
                "by_pair": {
                    "steps_1_repeats_1": {
                        "ca_steps": 1,
                        "num_repeats": 1,
                        "by_length": {"64": metric},
                    }
                },
            },
            "internal_repeat_extrapolation": {},
            "external_rollout": {},
        },
        "repeat_diagnostics": None,
        "preserved_cache_clean_state_transitions": None,
    }


def test_standalone_evaluation_runs_exactly_three_checkpoints_and_preserves_source(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "run_900__but_full_depth"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "run_901__but_full_depth_eval"
    artifact_dir = tmp_path / "artifacts" / "job_1"
    source_dir.mkdir()
    checkpoint_dir.mkdir()
    manifest = _source_manifest(tmp_path)
    write_json(source_dir / "run_manifest.json", manifest)
    source_before = (source_dir / "run_manifest.json").read_bytes()

    state_dict = _FakeBut().state_dict()
    for index, checkpoint_type in enumerate(CHECKPOINT_TYPES, start=1):
        checkpoint_name = f"{checkpoint_type}.pt"
        write_json(
            checkpoint_dir / f"{checkpoint_type}.json",
            {
                "step": index * 10,
                "checkpoint": checkpoint_name,
                "selection_type": checkpoint_type,
            },
        )
        torch.save(
            {"model": state_dict, "itr": index * 10},
            checkpoint_dir / checkpoint_name,
        )
    validation_metric = {
        "loss": 0.2,
        "cell_accuracy": 0.8,
        "exact_sequence_accuracy": 0.1,
    }
    write_json(
        checkpoint_dir / "training_stats.json",
        {
            "train": [],
            "eval": {
                "10": {
                    "in_distribution": {},
                    "extrapolation_validation": {
                        "steps_7_repeats_7": {
                            "ca_steps": 7,
                            "num_repeats": 7,
                            "by_length": {"64": validation_metric},
                        }
                    },
                    "repeat_diagnostics": None,
                }
            },
        },
    )

    evaluated_labels = []

    def fake_evaluate(*_args, **kwargs):
        evaluated_labels.append(kwargs["label"])
        return _fake_checkpoint_analysis(kwargs["checkpoint_metadata"])

    monkeypatch.setattr(
        ca_checkpoint_eval.models,
        "make_model_from_args",
        lambda _args: _FakeBut(),
    )
    monkeypatch.setattr(
        ca_checkpoint_eval, "make_ca_fixed_loaders", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        ca_checkpoint_eval, "evaluate_loaded_ca_checkpoint", fake_evaluate
    )

    result = run_standalone_evaluation(
        argparse.Namespace(
            source_run_dir=source_dir,
            output_run_dir=output_dir,
            artifact_dir=artifact_dir,
            device="cpu",
        )
    )

    assert evaluated_labels == list(CHECKPOINT_TYPES)
    assert result["checkpoint_types"] == list(CHECKPOINT_TYPES)
    assert (source_dir / "run_manifest.json").read_bytes() == source_before
    output_manifest = read_validated_manifest(output_dir / "run_manifest.json")
    assert output_manifest["status"] == "completed"
    assert output_manifest["provenance"]["source_run_id"] == manifest["run_id"]
    assert (output_dir / "eval_metrics.jsonl").is_file()
    assert (artifact_dir / "summary.json").is_file()
    summary = read_json(artifact_dir / "summary.json")
    assert set(summary["checkpoint_analysis"]) == set(CHECKPOINT_TYPES)
    discovered = discover_runs(
        tmp_path,
        filters=(parse_filter("status=completed"),),
    )
    assert [run.run_id for run in discovered] == [output_dir.name]
    assert {
        record["checkpoint_type"]
        for record in discovered[0].metrics
        if record["data_split"] == "final_test"
    } == set(CHECKPOINT_TYPES)
    assert any(
        record["data_split"] == "validation"
        and record["evaluation_role"] == "extrapolation_validation"
        and record["ca_steps"] == 7
        and record["num_repeats"] == 7
        for record in discovered[0].metrics
    )
