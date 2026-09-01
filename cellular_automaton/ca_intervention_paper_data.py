"""Build validated paper-analysis tables from CA factorial interventions.

This module reads only the compact ``summary.json`` files produced by the
clean-state/cache-reset factorial evaluation.  It does not load diagnostic
shards or rerun a model.  The emitted tables preserve every observed source
and target depth; missing intervention depths are reported rather than
imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


CONDITIONS = (
    "free_baseline",
    "clean_baseline",
    "free_reset",
    "clean_reset",
)
METRICS = ("cell_accuracy", "exact_sequence_accuracy", "loss")
EFFECT_CONDITIONS = {
    "clean_state_with_history": ("clean_baseline", "free_baseline"),
    "clean_state_after_reset": ("clean_reset", "free_reset"),
    "cache_reset_on_free_state": ("free_reset", "free_baseline"),
    "cache_reset_on_clean_state": ("clean_reset", "clean_baseline"),
}
INTERACTION_ID = "state_history_interaction"
INTERACTION_FORMULA = "(clean_reset - free_reset) - (clean_baseline - free_baseline)"

_RUN_PATTERN = re.compile(r"^run_(\d+)__ca_cotf_cache_attn$")
_SOURCE_PATTERN = re.compile(r"^t(\d+)$")
_EFFECT_TOLERANCE = 1e-12
_UNSET = object()

_COMMON_FIELDS = (
    "run_id",
    "training_seed",
    "data_seed",
    "evaluation_seed",
    "cache_policy",
    "cache_window",
    "cache_policy_provenance",
    "checkpoint_step",
    "source_depth",
    "target_depth",
    "post_intervention_repeat",
    "summary_path",
)
CONDITION_FIELDS = _COMMON_FIELDS + (
    "condition_id",
    "cell_accuracy",
    "exact_sequence_accuracy",
    "loss",
)
EFFECT_FIELDS = _COMMON_FIELDS + (
    "effect_id",
    "effect_formula",
    "cell_accuracy",
    "exact_sequence_accuracy",
    "loss",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-run-ids",
        type=int,
        nargs="+",
        help="Numeric run IDs whose missing results must be reported.",
    )
    return parser.parse_args(argv)


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON file {path}: {error}") from error


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return value


def _run_number(run_id):
    match = _RUN_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError(f"invalid CA cache-attention run ID: {run_id!r}")
    return int(match.group(1))


def discover_factorial_summaries(artifact_root):
    """Return root factorial summaries ordered by run and source depth."""
    artifact_root = Path(artifact_root)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root does not exist: {artifact_root}")
    records = []
    for path in artifact_root.glob("run_*__ca_cotf_cache_attn/t*/summary.json"):
        run_id = path.parent.parent.name
        source_match = _SOURCE_PATTERN.fullmatch(path.parent.name)
        if source_match is None:
            continue
        records.append((_run_number(run_id), int(source_match.group(1)), path))
    records.sort(key=lambda row: (row[0], row[1], str(row[2])))
    paths = [path for _, _, path in records]
    if not paths:
        raise ValueError(f"no factorial summaries found under {artifact_root}")
    return paths


def load_run_metadata(run_manifest_root, run_id, trained_cache_window=_UNSET):
    """Load seed and cache-policy provenance for one training run."""
    path = Path(run_manifest_root) / run_id / "run_manifest.json"
    manifest = _read_json(path)
    if manifest.get("run_id") != run_id:
        raise ValueError(f"run manifest ID does not match directory: {path}")

    seeds = manifest.get("seeds", {})
    training_seed, data_seed = seeds.get("seed"), seeds.get("data_seed")
    if not isinstance(training_seed, int) or isinstance(training_seed, bool):
        raise ValueError(f"training seed is missing or invalid: {path}")
    if not isinstance(data_seed, int) or isinstance(data_seed, bool):
        raise ValueError(f"data seed is missing or invalid: {path}")

    forward_policy = manifest.get("forward_policy") or {}
    manifest_policy = forward_policy.get("repeat_cache_policy")
    manifest_window = forward_policy.get("repeat_cache_window")
    if manifest_window is not None and (
        isinstance(manifest_window, bool)
        or not isinstance(manifest_window, int)
        or manifest_window <= 0
    ):
        raise ValueError(f"run manifest cache window is invalid: {path}")

    summary_window_available = trained_cache_window is not _UNSET
    if not summary_window_available:
        trained_cache_window = manifest_window
    if trained_cache_window is not None and (
        isinstance(trained_cache_window, bool)
        or not isinstance(trained_cache_window, int)
        or trained_cache_window <= 0
    ):
        raise ValueError(f"factorial trained cache window is invalid for {run_id}")
    if (
        summary_window_available
        and manifest_window is not None
        and manifest_window != trained_cache_window
    ):
        raise ValueError(
            f"cache window mismatch for {run_id}: summary={trained_cache_window}, "
            f"manifest={manifest_window}"
        )

    if trained_cache_window is None:
        if manifest_policy == "recent":
            raise ValueError(
                f"recent cache policy lacks a trained cache window for {run_id}"
            )
        cache_policy = manifest_policy or "full"
        provenance = (
            "run_manifest.forward_policy"
            if manifest_policy is not None
            else "summary.trained_cache_window=null; legacy manifest missing policy"
        )
    else:
        if manifest_policy not in (None, "recent"):
            raise ValueError(
                f"windowed summary conflicts with cache policy {manifest_policy!r} "
                f"for {run_id}"
            )
        cache_policy = "recent"
        provenance = (
            "summary and run_manifest"
            if manifest_policy is not None
            else "summary.trained_cache_window"
        )

    return {
        "run_id": run_id,
        "run_number": _run_number(run_id),
        "training_seed": training_seed,
        "data_seed": data_seed,
        "cache_policy": cache_policy,
        "cache_window": trained_cache_window,
        "cache_policy_provenance": provenance,
        "manifest_path": path,
    }


def _validate_metric_row(row, context):
    if not isinstance(row, dict) or set(row) != set(METRICS):
        raise ValueError(f"{context} must contain exactly {METRICS}")
    return {metric: _numeric(row[metric], f"{context}.{metric}") for metric in METRICS}


def _validate_depth_row(row, target_depths, context):
    if not isinstance(row, dict):
        raise ValueError(f"{context} must be an object")
    by_depth = row.get("by_target_depth")
    expected_keys = {str(depth) for depth in target_depths}
    if not isinstance(by_depth, dict) or set(by_depth) != expected_keys:
        raise ValueError(f"{context} target-depth keys are inconsistent")
    return {
        depth: _validate_metric_row(by_depth[str(depth)], f"{context}[{depth}]")
        for depth in target_depths
    }


def validate_factorial_summary(summary, path, run_metadata):
    """Validate one compact summary and return normalized factorial data."""
    path = Path(path)
    run_id = path.parent.parent.name
    if summary.get("run_id") != run_id or run_metadata["run_id"] != run_id:
        raise ValueError(f"summary run ID does not match its directory: {path}")
    source_match = _SOURCE_PATTERN.fullmatch(path.parent.name)
    if source_match is None:
        raise ValueError(f"summary source directory is invalid: {path.parent}")
    directory_source = int(source_match.group(1))

    evaluation = summary.get("factorial_full_split_evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"factorial full-split evaluation is missing: {path}")
    source_depth = evaluation.get("source_depth")
    if source_depth != directory_source:
        raise ValueError(
            f"source depth mismatch for {path}: directory={directory_source}, "
            f"summary={source_depth}"
        )
    max_repeats = evaluation.get("max_repeats")
    target_depths = evaluation.get("target_depths")
    if not isinstance(max_repeats, int) or max_repeats <= source_depth:
        raise ValueError(f"max_repeats is invalid: {path}")
    expected_depths = list(range(source_depth + 1, max_repeats + 1))
    if target_depths != expected_depths:
        raise ValueError(f"target depths are not consecutive through max_repeats: {path}")

    if "trained_cache_window" not in evaluation:
        raise ValueError(f"trained cache window field is missing: {path}")
    if evaluation["trained_cache_window"] != run_metadata["cache_window"]:
        raise ValueError(f"factorial and run metadata cache windows disagree: {path}")
    capture = summary.get("factorial_batch_capture")
    if isinstance(capture, dict) and capture.get("trained_cache_window") != evaluation[
        "trained_cache_window"
    ]:
        raise ValueError(f"factorial capture and evaluation cache windows disagree: {path}")

    conditions = evaluation.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError(f"factorial summary must contain exactly four conditions: {path}")
    normalized_conditions = {
        condition: _validate_depth_row(
            conditions[condition], target_depths, f"conditions.{condition}"
        )
        for condition in CONDITIONS
    }

    effects = evaluation.get("paired_effects")
    if not isinstance(effects, dict) or set(effects) != set(EFFECT_CONDITIONS):
        raise ValueError(f"factorial paired effects are incomplete: {path}")
    normalized_effects = {
        effect: _validate_depth_row(
            effects[effect], target_depths, f"paired_effects.{effect}"
        )
        for effect in EFFECT_CONDITIONS
    }
    interaction = _validate_depth_row(
        evaluation.get(INTERACTION_ID), target_depths, INTERACTION_ID
    )

    for depth in target_depths:
        for effect, (left, right) in EFFECT_CONDITIONS.items():
            for metric in METRICS:
                expected = (
                    normalized_conditions[left][depth][metric]
                    - normalized_conditions[right][depth][metric]
                )
                observed = normalized_effects[effect][depth][metric]
                if not math.isclose(
                    observed, expected, rel_tol=0.0, abs_tol=_EFFECT_TOLERANCE
                ):
                    raise ValueError(
                        f"stored effect {effect}.{metric} is inconsistent at "
                        f"target depth {depth}: {path}"
                    )
        for metric in METRICS:
            expected = (
                normalized_conditions["clean_reset"][depth][metric]
                - normalized_conditions["free_reset"][depth][metric]
                - normalized_conditions["clean_baseline"][depth][metric]
                + normalized_conditions["free_baseline"][depth][metric]
            )
            if not math.isclose(
                interaction[depth][metric],
                expected,
                rel_tol=0.0,
                abs_tol=_EFFECT_TOLERANCE,
            ):
                raise ValueError(
                    f"stored factorial interaction.{metric} is inconsistent at "
                    f"target depth {depth}: {path}"
                )

    dataset = summary.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"evaluation dataset metadata is missing: {path}")
    evaluation_seed = dataset.get("seed")
    if not isinstance(evaluation_seed, int) or isinstance(evaluation_seed, bool):
        raise ValueError(f"evaluation seed is missing or invalid: {path}")

    checkpoint_step = summary.get("checkpoint_step")
    if not isinstance(checkpoint_step, int) or checkpoint_step < 0:
        raise ValueError(f"checkpoint step is missing or invalid: {path}")
    return {
        "run_id": run_id,
        "source_depth": source_depth,
        "target_depths": target_depths,
        "max_repeats": max_repeats,
        "checkpoint_step": checkpoint_step,
        "evaluation_seed": evaluation_seed,
        "dataset": dataset,
        "conditions": normalized_conditions,
        "effects": normalized_effects,
        "interaction": interaction,
        "summary_path": path,
    }


def _common_row(record, run_metadata, target_depth, artifact_root):
    try:
        summary_path = record["summary_path"].relative_to(artifact_root)
    except ValueError:
        summary_path = record["summary_path"]
    return {
        "run_id": record["run_id"],
        "training_seed": run_metadata["training_seed"],
        "data_seed": run_metadata["data_seed"],
        "evaluation_seed": record["evaluation_seed"],
        "cache_policy": run_metadata["cache_policy"],
        "cache_window": (
            "" if run_metadata["cache_window"] is None
            else run_metadata["cache_window"]
        ),
        "cache_policy_provenance": run_metadata["cache_policy_provenance"],
        "checkpoint_step": record["checkpoint_step"],
        "source_depth": record["source_depth"],
        "target_depth": target_depth,
        "post_intervention_repeat": target_depth - record["source_depth"],
        "summary_path": str(summary_path),
    }


def build_condition_rows(record, run_metadata, artifact_root):
    rows = []
    for target_depth in record["target_depths"]:
        common = _common_row(record, run_metadata, target_depth, artifact_root)
        for condition in CONDITIONS:
            rows.append({
                **common,
                "condition_id": condition,
                **record["conditions"][condition][target_depth],
            })
    return rows


def build_effect_rows(record, run_metadata, artifact_root):
    rows = []
    for target_depth in record["target_depths"]:
        common = _common_row(record, run_metadata, target_depth, artifact_root)
        for effect, (left, right) in EFFECT_CONDITIONS.items():
            rows.append({
                **common,
                "effect_id": effect,
                "effect_formula": f"{left} - {right}",
                **record["effects"][effect][target_depth],
            })
        rows.append({
            **common,
            "effect_id": INTERACTION_ID,
            "effect_formula": INTERACTION_FORMULA,
            **record["interaction"][target_depth],
        })
    return rows


def validate_overlapping_baselines(records):
    """Require identical free-baseline outputs wherever sources overlap."""
    observed = defaultdict(dict)
    for record in records:
        for depth, metrics in record["conditions"]["free_baseline"].items():
            key = (record["run_id"], depth)
            previous = observed[key].get("metrics")
            if previous is not None and previous != metrics:
                raise ValueError(
                    f"overlapping free-baseline trajectories disagree for "
                    f"{record['run_id']} at target depth {depth}"
                )
            observed[key] = {"metrics": metrics, "source": record["source_depth"]}


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_analysis(
    artifact_root, run_manifest_root, output_dir, expected_run_numbers=None
):
    """Validate inputs and write deterministic condition/effect tables."""
    artifact_root = Path(artifact_root)
    run_manifest_root = Path(run_manifest_root)
    output_dir = Path(output_dir)
    summary_paths = discover_factorial_summaries(artifact_root)

    summaries = [_read_json(path) for path in summary_paths]
    windows_by_run = defaultdict(set)
    for path, summary in zip(summary_paths, summaries):
        evaluation = summary.get("factorial_full_split_evaluation") or {}
        if "trained_cache_window" not in evaluation:
            raise ValueError(f"trained cache window field is missing: {path}")
        windows_by_run[path.parent.parent.name].add(evaluation["trained_cache_window"])
    if any(len(windows) != 1 for windows in windows_by_run.values()):
        raise ValueError("factorial summaries disagree on cache window within a run")

    metadata = {
        run_id: load_run_metadata(
            run_manifest_root, run_id, next(iter(windows))
        )
        for run_id, windows in windows_by_run.items()
    }
    records = [
        validate_factorial_summary(summary, path, metadata[path.parent.parent.name])
        for path, summary in zip(summary_paths, summaries)
    ]
    if len({json.dumps(record["dataset"], sort_keys=True) for record in records}) != 1:
        raise ValueError("factorial summaries do not use one identical evaluation dataset")
    validate_overlapping_baselines(records)

    condition_rows, effect_rows = [], []
    for record in records:
        run_metadata = metadata[record["run_id"]]
        condition_rows.extend(build_condition_rows(record, run_metadata, artifact_root))
        effect_rows.extend(build_effect_rows(record, run_metadata, artifact_root))

    present_run_ids = sorted(metadata, key=_run_number)
    expected_run_ids = (
        [f"run_{number}__ca_cotf_cache_attn" for number in expected_run_numbers]
        if expected_run_numbers is not None else present_run_ids
    )
    if len(expected_run_ids) != len(set(expected_run_ids)):
        raise ValueError("expected run IDs contain duplicates")
    missing_run_ids = [run_id for run_id in expected_run_ids if run_id not in metadata]
    unexpected_run_ids = [run_id for run_id in present_run_ids if run_id not in expected_run_ids]

    coverage_runs = []
    for run_id in expected_run_ids + unexpected_run_ids:
        run_records = [record for record in records if record["run_id"] == run_id]
        if run_records:
            run_metadata = metadata[run_id]
        else:
            run_metadata = load_run_metadata(run_manifest_root, run_id)
        coverage_runs.append({
            "run_id": run_id,
            "status": "present" if run_records else "missing_factorial_summaries",
            "training_seed": run_metadata["training_seed"],
            "data_seed": run_metadata["data_seed"],
            "cache_policy": run_metadata["cache_policy"],
            "cache_window": run_metadata["cache_window"],
            "cache_policy_provenance": run_metadata["cache_policy_provenance"],
            "sources": [
                {
                    "source_depth": record["source_depth"],
                    "target_depths": record["target_depths"],
                    "summary_path": str(record["summary_path"].relative_to(artifact_root)),
                }
                for record in run_records
            ],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = output_dir / "factorial_conditions.csv"
    effect_path = output_dir / "factorial_effects.csv"
    coverage_path = output_dir / "coverage.json"
    manifest_path = output_dir / "analysis_manifest.json"
    _write_csv(condition_path, CONDITION_FIELDS, condition_rows)
    _write_csv(effect_path, EFFECT_FIELDS, effect_rows)
    coverage = {
        "schema_version": 1,
        "expected_run_ids": expected_run_ids,
        "present_run_ids": present_run_ids,
        "missing_run_ids": missing_run_ids,
        "unexpected_run_ids": unexpected_run_ids,
        "dataset": records[0]["dataset"],
        "counts": {
            "runs_present": len(present_run_ids),
            "summaries": len(records),
            "run_source_target_combinations": sum(
                len(record["target_depths"]) for record in records
            ),
            "condition_rows": len(condition_rows),
            "effect_rows": len(effect_rows),
        },
        "runs": coverage_runs,
    }
    _write_json(coverage_path, coverage)

    input_paths = summary_paths + sorted(
        {item["manifest_path"] for item in metadata.values()}, key=str
    )
    for run_id in missing_run_ids:
        input_paths.append(run_manifest_root / run_id / "run_manifest.json")
    input_paths = sorted(set(input_paths), key=str)
    analysis_manifest = {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "run_manifest_root": str(run_manifest_root),
        "condition_ids": list(CONDITIONS),
        "metrics": list(METRICS),
        "effect_formulas": {
            **{
                effect: f"{left} - {right}"
                for effect, (left, right) in EFFECT_CONDITIONS.items()
            },
            INTERACTION_ID: INTERACTION_FORMULA,
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in input_paths
        ],
        "outputs": {
            condition_path.name: {
                "rows": len(condition_rows), "sha256": _sha256(condition_path)
            },
            effect_path.name: {
                "rows": len(effect_rows), "sha256": _sha256(effect_path)
            },
            coverage_path.name: {"sha256": _sha256(coverage_path)},
        },
    }
    _write_json(manifest_path, analysis_manifest)
    return coverage


def main(argv=None):
    args = parse_args(argv)
    coverage = build_analysis(
        args.artifact_root,
        args.run_manifest_root,
        args.output_dir,
        args.expected_run_ids,
    )
    counts = coverage["counts"]
    print(
        f"validated {counts['summaries']} summaries from "
        f"{counts['runs_present']} runs; wrote "
        f"{counts['condition_rows']} condition rows and "
        f"{counts['effect_rows']} effect rows"
    )
    if coverage["missing_run_ids"]:
        print("missing factorial results: " + ", ".join(coverage["missing_run_ids"]))


if __name__ == "__main__":
    main()
