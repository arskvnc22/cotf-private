import copy
import csv
import json

import pytest

from cellular_automaton.ca_intervention_paper_data import (
    EFFECT_CONDITIONS,
    INTERACTION_FORMULA,
    INTERACTION_ID,
    METRICS,
    build_analysis,
    load_run_metadata,
    validate_factorial_summary,
)


CONDITION_VALUES = {
    "free_baseline": (0.50, 0.00, 0.80),
    "clean_baseline": (0.80, 0.40, 0.30),
    "free_reset": (0.55, 0.05, 0.70),
    "clean_reset": (0.95, 0.85, 0.10),
}


def _metrics(values, depth):
    accuracy, exact, loss = values
    offset = depth * 0.001
    return {
        "cell_accuracy": accuracy + offset,
        "exact_sequence_accuracy": exact + offset,
        "loss": loss + offset,
    }


def _subtract(left, right):
    return {metric: left[metric] - right[metric] for metric in METRICS}


def _summary(run_id, source_depth, max_repeats=3, cache_window=2):
    target_depths = list(range(source_depth + 1, max_repeats + 1))
    conditions = {
        condition: {
            "by_target_depth": {
                str(depth): _metrics(values, depth)
                for depth in target_depths
            }
        }
        for condition, values in CONDITION_VALUES.items()
    }
    effects = {}
    for effect, (left, right) in EFFECT_CONDITIONS.items():
        effects[effect] = {
            "by_target_depth": {
                str(depth): _subtract(
                    conditions[left]["by_target_depth"][str(depth)],
                    conditions[right]["by_target_depth"][str(depth)],
                )
                for depth in target_depths
            }
        }
    interaction = {}
    for depth in target_depths:
        clean_after_reset = effects["clean_state_after_reset"][
            "by_target_depth"
        ][str(depth)]
        clean_with_history = effects["clean_state_with_history"][
            "by_target_depth"
        ][str(depth)]
        interaction[str(depth)] = _subtract(
            clean_after_reset, clean_with_history
        )
    evaluation = {
        "source_depth": source_depth,
        "target_depths": target_depths,
        "max_repeats": max_repeats,
        "trained_cache_window": cache_window,
        "conditions": conditions,
        "paired_effects": effects,
        INTERACTION_ID: {
            "formula": INTERACTION_FORMULA,
            "by_target_depth": interaction,
        },
    }
    return {
        "run_id": run_id,
        "checkpoint_step": 12000,
        "dataset": {
            "batch_size": 128,
            "bernoulli_p": 0.5,
            "data_mode": "materialized",
            "num_batches": 32,
            "num_cells": 64,
            "num_samples": 4096,
            "seed": 1000067,
            "shuffle": False,
            "split": "validation",
            "steps": 1,
        },
        "factorial_batch_capture": {"trained_cache_window": cache_window},
        "factorial_full_split_evaluation": evaluation,
    }


def _manifest(run_id, seed=0, cache_policy="recent", cache_window=2):
    manifest = {
        "run_id": run_id,
        "seeds": {"seed": seed, "data_seed": 1},
    }
    if cache_policy is not None:
        manifest["forward_policy"] = {
            "repeat_cache_policy": cache_policy,
            "repeat_cache_window": cache_window,
        }
    return manifest


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run_manifest(root, run_id, **kwargs):
    path = root / run_id / "run_manifest.json"
    _write_json(path, _manifest(run_id, **kwargs))
    return path


def _write_summary(root, run_id, source_depth, **kwargs):
    path = root / run_id / f"t{source_depth}" / "summary.json"
    _write_json(path, _summary(run_id, source_depth, **kwargs))
    return path


def test_build_analysis_writes_validated_tables_and_missing_coverage(tmp_path):
    artifacts = tmp_path / "artifacts"
    manifests = tmp_path / "runs"
    output = tmp_path / "paper-data"
    run_1 = "run_1__ca_cotf_cache_attn"
    run_2 = "run_2__ca_cotf_cache_attn"
    _write_run_manifest(manifests, run_1, seed=7)
    _write_run_manifest(
        manifests, run_2, seed=8, cache_policy=None, cache_window=None
    )
    _write_summary(artifacts, run_1, 1)
    _write_summary(artifacts, run_1, 2)

    coverage = build_analysis(
        artifacts, manifests, output, expected_run_numbers=[1, 2]
    )

    assert coverage["counts"] == {
        "runs_present": 1,
        "summaries": 2,
        "run_source_target_combinations": 3,
        "condition_rows": 12,
        "effect_rows": 15,
    }
    assert coverage["missing_run_ids"] == [run_2]
    missing = next(row for row in coverage["runs"] if row["run_id"] == run_2)
    assert missing == {
        "run_id": run_2,
        "status": "missing_factorial_summaries",
        "training_seed": 8,
        "data_seed": 1,
        "cache_policy": "full",
        "cache_window": None,
        "cache_policy_provenance": (
            "summary.trained_cache_window=null; legacy manifest missing policy"
        ),
        "sources": [],
    }

    with (output / "factorial_conditions.csv").open(newline="") as handle:
        conditions = list(csv.DictReader(handle))
    with (output / "factorial_effects.csv").open(newline="") as handle:
        effects = list(csv.DictReader(handle))
    assert len(conditions) == 12
    assert len(effects) == 15
    clean_reset = next(
        row for row in conditions
        if row["source_depth"] == "1"
        and row["target_depth"] == "2"
        and row["condition_id"] == "clean_reset"
    )
    assert clean_reset["post_intervention_repeat"] == "1"
    assert clean_reset["training_seed"] == "7"
    assert clean_reset["cache_policy"] == "recent"
    assert clean_reset["cache_window"] == "2"
    reset_effect = next(
        row for row in effects
        if row["source_depth"] == "1"
        and row["target_depth"] == "2"
        and row["effect_id"] == "cache_reset_on_clean_state"
    )
    assert float(reset_effect["cell_accuracy"]) == pytest.approx(0.15)
    assert reset_effect["effect_formula"] == "clean_reset - clean_baseline"

    first_manifest = (output / "analysis_manifest.json").read_text()
    build_analysis(artifacts, manifests, output, expected_run_numbers=[1, 2])
    assert (output / "analysis_manifest.json").read_text() == first_manifest


def test_legacy_null_window_is_explicitly_labelled_full(tmp_path):
    run_id = "run_83__ca_cotf_cache_attn"
    manifests = tmp_path / "runs"
    _write_run_manifest(
        manifests, run_id, cache_policy=None, cache_window=None
    )

    metadata = load_run_metadata(manifests, run_id, trained_cache_window=None)

    assert metadata["cache_policy"] == "full"
    assert metadata["cache_window"] is None
    assert "legacy manifest missing policy" in metadata["cache_policy_provenance"]


def test_cache_window_disagreement_is_rejected(tmp_path):
    run_id = "run_1__ca_cotf_cache_attn"
    manifests = tmp_path / "runs"
    _write_run_manifest(manifests, run_id, cache_window=2)

    with pytest.raises(ValueError, match="cache window mismatch"):
        load_run_metadata(manifests, run_id, trained_cache_window=3)


def test_missing_result_can_take_cache_window_from_run_manifest(tmp_path):
    run_id = "run_99__ca_cotf_cache_attn"
    manifests = tmp_path / "runs"
    _write_run_manifest(manifests, run_id, seed=2, cache_window=3)

    metadata = load_run_metadata(manifests, run_id)

    assert metadata["cache_policy"] == "recent"
    assert metadata["cache_window"] == 3
    assert metadata["training_seed"] == 2


def test_summary_source_directory_mismatch_is_rejected(tmp_path):
    run_id = "run_1__ca_cotf_cache_attn"
    manifests = tmp_path / "runs"
    path = tmp_path / "artifacts" / run_id / "t2" / "summary.json"
    _write_json(path, _summary(run_id, 1))
    _write_run_manifest(manifests, run_id)
    metadata = load_run_metadata(manifests, run_id, trained_cache_window=2)

    with pytest.raises(ValueError, match="source depth mismatch"):
        validate_factorial_summary(_summary(run_id, 1), path, metadata)


@pytest.mark.parametrize("malformation", ["condition", "depths", "effect"])
def test_malformed_factorial_summary_is_rejected(tmp_path, malformation):
    run_id = "run_1__ca_cotf_cache_attn"
    manifests = tmp_path / "runs"
    path = tmp_path / "artifacts" / run_id / "t1" / "summary.json"
    _write_run_manifest(manifests, run_id)
    metadata = load_run_metadata(manifests, run_id, trained_cache_window=2)
    summary = _summary(run_id, 1)
    evaluation = summary["factorial_full_split_evaluation"]
    if malformation == "condition":
        del evaluation["conditions"]["free_reset"]
        match = "exactly four conditions"
    elif malformation == "depths":
        evaluation["target_depths"] = [2]
        match = "not consecutive"
    else:
        evaluation["paired_effects"]["clean_state_with_history"][
            "by_target_depth"
        ]["2"]["cell_accuracy"] += 0.1
        match = "stored effect"

    with pytest.raises(ValueError, match=match):
        validate_factorial_summary(summary, path, metadata)


def test_overlapping_free_baseline_disagreement_is_rejected(tmp_path):
    artifacts = tmp_path / "artifacts"
    manifests = tmp_path / "runs"
    output = tmp_path / "output"
    run_id = "run_1__ca_cotf_cache_attn"
    _write_run_manifest(manifests, run_id)
    _write_summary(artifacts, run_id, 1)
    second = _summary(run_id, 2)
    second["factorial_full_split_evaluation"]["conditions"]["free_baseline"][
        "by_target_depth"
    ]["3"]["cell_accuracy"] += 0.01
    # Keep the stored effects internally consistent so the overlap check is
    # the first and only violated invariant.
    for effect, (left, right) in EFFECT_CONDITIONS.items():
        condition_rows = second["factorial_full_split_evaluation"]["conditions"]
        second["factorial_full_split_evaluation"]["paired_effects"][effect][
            "by_target_depth"
        ]["3"] = _subtract(
            condition_rows[left]["by_target_depth"]["3"],
            condition_rows[right]["by_target_depth"]["3"],
        )
    effects = second["factorial_full_split_evaluation"]["paired_effects"]
    second["factorial_full_split_evaluation"][INTERACTION_ID][
        "by_target_depth"
    ]["3"] = _subtract(
        effects["clean_state_after_reset"]["by_target_depth"]["3"],
        effects["clean_state_with_history"]["by_target_depth"]["3"],
    )
    _write_json(artifacts / run_id / "t2" / "summary.json", second)

    with pytest.raises(ValueError, match="overlapping free-baseline"):
        build_analysis(artifacts, manifests, output)


def test_different_evaluation_datasets_are_rejected(tmp_path):
    artifacts = tmp_path / "artifacts"
    manifests = tmp_path / "runs"
    output = tmp_path / "output"
    run_id = "run_1__ca_cotf_cache_attn"
    _write_run_manifest(manifests, run_id)
    _write_summary(artifacts, run_id, 1)
    second = _summary(run_id, 2)
    second["dataset"]["seed"] += 1
    _write_json(artifacts / run_id / "t2" / "summary.json", second)

    with pytest.raises(ValueError, match="one identical evaluation dataset"):
        build_analysis(artifacts, manifests, output)
