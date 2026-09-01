import argparse
import json

import pytest
import torch

from cellular_automaton.ca_plot_attention_diagnostics import (
    _find_shard,
    _validate_factorial_outcomes,
    main,
    summarize_factorial_state_cosines,
)
from cellular_automaton.ca_attention_diagnostics import (
    capture_factorial_transition_in_memory,
)
from cellular_automaton.test_ca_attention_diagnostics import _FakeDiagnosticModel


CONDITIONS = ("free_baseline", "clean_baseline", "free_reset", "clean_reset")


def _factorial_shard(version=2):
    shard = capture_factorial_transition_in_memory(
        _FakeDiagnosticModel(), torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]]),
        argparse.Namespace(dtype=torch.bfloat16), "cpu", 1, 3,
        baseline_repeat_cache_window=2,
    )
    shard["shared_prefix"]["repeat_states"] = (
        shard["shared_prefix"]["repeat_states"].clone() + 1
    )
    for payload in shard["conditions"].values():
        payload["repeat_states"] = payload["repeat_states"].clone() + 1
    shard.update({"batch_index": 0, "example_ids": torch.tensor([7, 9])})
    if version == 1:
        shard["factorial_capture_schema_version"] = 1
        del shard["shared_prefix"]
    return shard


def _full_split_outcomes():
    metrics = {"cell_accuracy": 0.8, "exact_sequence_accuracy": 0.5, "loss": 0.4}
    by_depth = {str(depth): {**metrics} for depth in (2, 3)}
    return {
        "source_depth": 1, "target_depths": [2, 3],
        "num_examples": 16, "num_batches": 2,
        "conditions": {name: {"by_target_depth": by_depth} for name in CONDITIONS},
        "paired_effects": {
            name: {"by_target_depth": by_depth} for name in (
                "clean_state_with_history", "clean_state_after_reset",
                "cache_reset_on_free_state", "cache_reset_on_clean_state",
            )
        },
        "state_history_interaction": {"by_target_depth": by_depth},
    }


def test_factorial_plot_main_writes_heatmap_and_small_summary(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    shard_path = artifact_dir / "factorial.pt"
    torch.save(_factorial_shard(), shard_path)
    manifest = {
        "source": {"run_id": "run_test"},
        "shards": [{
            "experiment_kind": "clean_state_cache_reset_factorial",
            "source_depth": 1, "batch_index": 0, "path": shard_path.name,
        }],
    }
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps(manifest))
    (artifact_dir / "summary.json").write_text(json.dumps({
        "factorial_full_split_evaluation": _full_split_outcomes()
    }))

    main(["--artifact-dir", str(artifact_dir), "--cosine-example-count", "1"])

    output_dir = artifact_dir / "plots" / "factorial_t00001"
    assert (output_dir / "state_cosine_example_7.png").is_file()
    assert (output_dir / "factorial_outcomes.png").is_file()
    assert (output_dir / "factorial_outcome_effects.png").is_file()
    for group in ("qkv_logits", "attention", "state"):
        assert (output_dir / f"factorial_{group}_conditions.png").is_file()
        assert (output_dir / f"factorial_{group}_effects.png").is_file()
    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved["depths"] == [0, 1, 2, 3]
    assert saved["selected_example_count"] == 1
    assert saved["factorial_diagnostics"]["source_depth"] == 1
    diagnostics = saved["factorial_diagnostics"]
    assert diagnostics["conditions"]["free_reset"][
        "visible_history_key_norm"
    ]["mean"][:2] == [None, None]
    state_effect = diagnostics["effects"]["clean_state_after_reset"][
        "hidden_state_norm"
    ]["mean"]
    clean = diagnostics["conditions"]["clean_reset"]["hidden_state_norm"]["mean"]
    free = diagnostics["conditions"]["free_reset"]["hidden_state_norm"]["mean"]
    assert state_effect[2] == pytest.approx(clean[2] - free[2])
    assert saved["full_split_outcomes"] == {
        "status": "plotted", "num_examples": 16, "num_batches": 2
    }
    assert saved["examples"][0]["example_id"] == 7
    assert set(saved["examples"][0]["conditions"]) == set(CONDITIONS)
    for values in saved["examples"][0]["conditions"].values():
        matrix = torch.tensor(values)
        assert matrix.shape == (4, 4) and torch.allclose(matrix, matrix.T)
        assert torch.allclose(matrix.diag(), torch.ones(4))
    (artifact_dir / "summary.json").unlink()
    raw_only_output = tmp_path / "raw_only"
    main([
        "--artifact-dir", str(artifact_dir),
        "--output-dir", str(raw_only_output),
        "--cosine-example-count", "1",
    ])
    raw_only = json.loads((raw_only_output / "summary.json").read_text())
    assert raw_only["full_split_outcomes"] == "not_available"
    malformed = _full_split_outcomes()
    del malformed["conditions"]["free_baseline"]["by_target_depth"]["3"]
    with pytest.raises(ValueError, match="depth keys"):
        _validate_factorial_outcomes(malformed)
    with pytest.raises(ValueError, match="schema version 2"):
        summarize_factorial_state_cosines(_factorial_shard(version=1))


def test_factorial_full_split_only_plots_outcomes_without_raw_shard(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "artifact_manifest.json").write_text(json.dumps({
        "source": {"run_id": "run_test"},
        "shards": [],
    }))
    (artifact_dir / "summary.json").write_text(json.dumps({
        "factorial_full_split_evaluation": _full_split_outcomes()
    }))

    main([
        "--artifact-dir", str(artifact_dir),
        "--factorial-source-depth", "1",
    ])

    output_dir = artifact_dir / "plots" / "factorial_t00001"
    assert (output_dir / "factorial_outcomes.png").is_file()
    assert (output_dir / "factorial_outcome_effects.png").is_file()
    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved["factorial_batch_capture"] == "not_available"
    assert saved["factorial_diagnostics"] == "not_available"
    assert saved["state_cosines"] == "not_available"
    assert saved["full_split_outcomes"] == {
        "status": "plotted", "num_examples": 16, "num_batches": 2
    }


def test_uninterrupted_shard_lookup_remains_available(tmp_path):
    shard = tmp_path / "baseline.pt"
    shard.touch()
    manifest = {"shards": [{
        "condition_id": "manual_full", "batch_index": 0, "path": shard.name
    }]}
    assert _find_shard(manifest, "manual_full", 0, tmp_path) == shard
