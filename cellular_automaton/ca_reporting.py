"""Run metadata and normalized metric export for cellular-automaton experiments.

This module deliberately depends only on the Python standard library. Training
code can therefore write the reporting artifacts without adding a dataframe
dependency, while analysis tools can consume the same schema on a login node.
"""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

try:
    from .ca_forward import CAForwardPolicy
except ImportError:
    from ca_forward import CAForwardPolicy


SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
METRICS_FILENAME = "eval_metrics.jsonl"
NOTES_FILENAME = "notes.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    text = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        default=_json_default,
        allow_nan=False,
    )
    _atomic_write_text(path, text + "\n")


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _argument(args: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = args.get(name, default)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _pairs_from_args(args: Mapping[str, Any]) -> list[dict[str, int]]:
    raw_pairs = _argument(args, "ca_train_pairs")
    if not raw_pairs:
        raw_pairs = [
            [
                _argument(args, "ca_steps", 1),
                _argument(args, "num_repeats", 1),
            ]
        ]

    pairs: list[dict[str, int]] = []
    for pair in raw_pairs:
        if isinstance(pair, Mapping):
            steps = pair.get("ca_steps")
            repeats = pair.get("num_repeats")
        else:
            steps, repeats = pair
        pairs.append({"ca_steps": int(steps), "num_repeats": int(repeats)})
    return pairs


def _selected_arguments(
    args: Mapping[str, Any], names: Iterable[str]
) -> dict[str, Any]:
    return {name: _argument(args, name) for name in names if name in args}


def forward_policy_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized policy metadata, including for historical manifests."""

    stored_policy = manifest.get("forward_policy")
    if (
        isinstance(stored_policy, Mapping)
        and "repeat_cache_window" in stored_policy
    ):
        repeat_cache_window = stored_policy.get("repeat_cache_window")
    else:
        resolved_args = manifest.get("resolved_args", {})
        repeat_cache_window = (
            resolved_args.get("repeat_cache_window")
            if isinstance(resolved_args, Mapping)
            else None
        )
    return CAForwardPolicy(
        repeat_cache_window=repeat_cache_window
    ).metadata()


def build_run_manifest(
    resolved_args: Mapping[str, Any],
    run_dir: Path | str,
    checkpoint_dir: Path | str,
    *,
    status: str = "running",
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one self-contained, forward-compatible run manifest."""

    run_dir = Path(run_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    existing = dict(existing or {})
    timestamps = dict(existing.get("timestamps", {}))
    timestamps.setdefault("started_at", utc_now())
    timestamps["updated_at"] = utc_now()

    model_fields = (
        "model",
        "n_layer",
        "n_head",
        "n_embd",
        "sequence_length",
        "attention_mode",
        "attention_implementation",
        "positional_encoder",
        "n_layer_begin",
        "n_layer_end",
        "n_repeat",
        "min_repeat",
        "depth_random_method",
        "depth_embedding",
        "n_layers",
        "n_heads",
        "block_size",
        "bias",
        "dropout",
        "activation",
        "norm_type",
        "pos_enc",
        "bidirectional_attention",
        "num_repeats",
        "n_repeat",
        "n_repeat_encoder",
        "n_repeat_decoder",
        "repeat_block",
        "vocab_size",
    )
    training_fields = (
        "opt",
        "lr",
        "min_lr",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
        "batch_size",
        "iterations",
        "scheduler",
        "final_div_factor",
        "warmup_percent",
        "lr_decay_iters",
        "warmup_iters",
        "acc_steps",
        "ca_train_pairs",
        "ca_train_samples",
        "ca_train_pair_sampling",
        "ca_train_pair_weights",
        "ca_train_lengths",
        "ca_train_num_cells",
        "ca_boundary",
        "ca_bernoulli_p",
        "ca_data_mode",
        "ca_num_workers",
    )
    evaluation_fields = (
        "eval_freq",
        "ca_eval_num_cells",
        "ca_val_samples",
        "ca_test_samples",
        "ca_extrapolation_val_pairs",
        "ca_extrapolation_best_pair",
        "ca_final_eval_pairs",
        "ca_repeat_diagnostic_max_repeats",
        "ca_repeat_diagnostic_horizons",
    )
    checkpoint_fields = (
        "ca_best_pair",
        "ca_best_length",
        "ca_best_metric",
        "ca_extrapolation_best_metric",
        "ca_extrapolation_min_id_cell_accuracy",
        "ca_extrapolation_min_id_exact_sequence_accuracy",
    )
    seed_fields = (
        "seed",
        "data_seed",
        "ca_val_seed",
        "ca_test_seed",
    )
    forward_policy = CAForwardPolicy(
        repeat_cache_window=_argument(resolved_args, "repeat_cache_window")
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": status,
        "annotations": {
            "tags": list(_argument(resolved_args, "ca_tags", []) or []),
            "note": _argument(resolved_args, "ca_note"),
        },
        "forward_policy": forward_policy.metadata(),
        "model": _selected_arguments(resolved_args, model_fields),
        "training": {
            **_selected_arguments(resolved_args, training_fields),
            "pairs": _pairs_from_args(resolved_args),
        },
        "checkpoint_selection": _selected_arguments(
            resolved_args, checkpoint_fields
        ),
        "evaluation": _selected_arguments(resolved_args, evaluation_fields),
        "seeds": _selected_arguments(resolved_args, seed_fields),
        "provenance": {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "experiment_name": _argument(resolved_args, "exp_name"),
            "dataset": _argument(resolved_args, "dataset"),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "timestamps": timestamps,
        "resolved_args": dict(resolved_args),
    }
    return manifest


def write_run_manifest(run_dir: Path | str, manifest: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / MANIFEST_FILENAME
    write_json(path, manifest)
    return path


def update_run_manifest_status(
    run_dir: Path | str,
    status: str,
    *,
    error: BaseException | None = None,
) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_FILENAME
    manifest: MutableMapping[str, Any] = read_json(path)
    manifest["status"] = status
    manifest.setdefault("timestamps", {})["updated_at"] = utc_now()
    if status in {"completed", "failed", "skipped"}:
        manifest["timestamps"]["completed_at"] = utc_now()
    if error is not None:
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    elif status != "failed":
        manifest.pop("error", None)
    write_json(path, manifest)
    return dict(manifest)


def ensure_notes_file(run_dir: Path | str) -> Path:
    path = Path(run_dir) / NOTES_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Experiment notes\n\n"
            "## Hypothesis\n\n"
            "<!-- What is this run intended to test? -->\n\n"
            "## Observations\n\n"
            "<!-- Add observations without editing run_manifest.json. -->\n",
            encoding="utf-8",
        )
    return path


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = value.item() if hasattr(value, "item") else value
    if isinstance(numeric, float) and not math.isfinite(numeric):
        return None
    return numeric


def _metric_items(metrics: Any) -> Iterable[tuple[str, int | float]]:
    if not isinstance(metrics, Mapping):
        return
    for name, value in metrics.items():
        numeric = _number(value)
        if numeric is not None:
            yield str(name), numeric


def _parse_repeat(value: str | int) -> int:
    text = str(value)
    return int(text.removeprefix("repeats_"))


def _parse_horizon(value: str | int) -> int:
    text = str(value)
    return int(text.removeprefix("steps_"))


def _base_record(
    manifest: Mapping[str, Any],
    *,
    step: int | None,
    data_split: str,
    evaluation_role: str,
    checkpoint_type: str | None = None,
    length: int | None = None,
    ca_steps: int | None = None,
    num_repeats: int | None = None,
    repeat_from: int | None = None,
    repeat_to: int | None = None,
) -> dict[str, Any]:
    policy = forward_policy_metadata(manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "repeat_cache_policy": policy["repeat_cache_policy"],
        "repeat_cache_window": policy["repeat_cache_window"],
        "forward_policy_label": policy["forward_policy_label"],
        "step": step,
        "data_split": data_split,
        "evaluation_role": evaluation_role,
        "checkpoint_type": checkpoint_type,
        "length": length,
        "ca_steps": ca_steps,
        "num_repeats": num_repeats,
        "repeat_from": repeat_from,
        "repeat_to": repeat_to,
    }


def _append_metrics(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    metrics: Any,
    *,
    step: int | None,
    data_split: str,
    evaluation_role: str,
    checkpoint_type: str | None = None,
    length: int | None = None,
    ca_steps: int | None = None,
    num_repeats: int | None = None,
    repeat_from: int | None = None,
    repeat_to: int | None = None,
) -> None:
    base = _base_record(
        manifest,
        step=step,
        data_split=data_split,
        evaluation_role=evaluation_role,
        checkpoint_type=checkpoint_type,
        length=length,
        ca_steps=ca_steps,
        num_repeats=num_repeats,
        repeat_from=repeat_from,
        repeat_to=repeat_to,
    )
    for metric, value in _metric_items(metrics):
        records.append({**base, "metric": metric, "value": value})


def _append_by_length(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    by_length: Mapping[str, Any],
    *,
    step: int | None,
    data_split: str,
    evaluation_role: str,
    checkpoint_type: str | None,
    ca_steps: int | None,
    num_repeats: int | None,
) -> None:
    for length, metrics in by_length.items():
        _append_metrics(
            records,
            manifest,
            metrics,
            step=step,
            data_split=data_split,
            evaluation_role=evaluation_role,
            checkpoint_type=checkpoint_type,
            length=int(length),
            ca_steps=ca_steps,
            num_repeats=num_repeats,
        )


def _append_pair_mapping(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    pairs: Mapping[str, Any],
    *,
    step: int | None,
    data_split: str,
    evaluation_role: str,
    checkpoint_type: str | None = None,
) -> None:
    for pair in pairs.values():
        if not isinstance(pair, Mapping) or "by_length" not in pair:
            continue
        _append_by_length(
            records,
            manifest,
            pair["by_length"],
            step=step,
            data_split=data_split,
            evaluation_role=evaluation_role,
            checkpoint_type=checkpoint_type,
            ca_steps=int(pair["ca_steps"]),
            num_repeats=int(pair["num_repeats"]),
        )


def _default_training_pair(manifest: Mapping[str, Any]) -> tuple[int, int]:
    pairs = manifest.get("training", {}).get("pairs", [])
    if not pairs:
        return 1, 1
    return int(pairs[0]["ca_steps"]), int(pairs[0]["num_repeats"])


def _append_in_distribution(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    value: Any,
    *,
    step: int | None,
    data_split: str,
    checkpoint_type: str | None = None,
) -> None:
    if not isinstance(value, Mapping):
        return
    if "by_pair" in value:
        _append_pair_mapping(
            records,
            manifest,
            value["by_pair"],
            step=step,
            data_split=data_split,
            evaluation_role="in_distribution",
            checkpoint_type=checkpoint_type,
        )
        return
    if "by_length" in value:
        _append_by_length(
            records,
            manifest,
            value["by_length"],
            step=step,
            data_split=data_split,
            evaluation_role="in_distribution",
            checkpoint_type=checkpoint_type,
            ca_steps=int(value.get("ca_steps", _default_training_pair(manifest)[0])),
            num_repeats=int(
                value.get("num_repeats", _default_training_pair(manifest)[1])
            ),
        )
        return

    if value and all(
        isinstance(candidate, Mapping) and "by_length" in candidate
        for candidate in value.values()
    ):
        _append_pair_mapping(
            records,
            manifest,
            value,
            step=step,
            data_split=data_split,
            evaluation_role="in_distribution",
            checkpoint_type=checkpoint_type,
        )
        return

    ca_steps, num_repeats = _default_training_pair(manifest)
    _append_by_length(
        records,
        manifest,
        value,
        step=step,
        data_split=data_split,
        evaluation_role="in_distribution",
        checkpoint_type=checkpoint_type,
        ca_steps=ca_steps,
        num_repeats=num_repeats,
    )


def _append_repeat_diagnostics(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    diagnostics: Any,
    *,
    step: int | None,
    data_split: str,
    checkpoint_type: str | None = None,
) -> None:
    if not isinstance(diagnostics, Mapping):
        return
    for length, length_data in diagnostics.items():
        if not isinstance(length_data, Mapping):
            continue
        length_int = int(length)

        for repeat_key, horizons in length_data.get(
            "repeat_horizon_matrix", {}
        ).items():
            repeat = _parse_repeat(repeat_key)
            for horizon_key, metrics in horizons.items():
                _append_metrics(
                    records,
                    manifest,
                    metrics,
                    step=step,
                    data_split=data_split,
                    evaluation_role="repeat_horizon_diagnostic",
                    checkpoint_type=checkpoint_type,
                    length=length_int,
                    ca_steps=_parse_horizon(horizon_key),
                    num_repeats=repeat,
                )

        for repeat_key, metrics in length_data.get(
            "best_matching_horizon", {}
        ).items():
            repeat = _parse_repeat(repeat_key)
            _append_metrics(
                records,
                manifest,
                metrics,
                step=step,
                data_split=data_split,
                evaluation_role="best_matching_horizon",
                checkpoint_type=checkpoint_type,
                length=length_int,
                num_repeats=repeat,
            )

        for repeat_key, recurrence in length_data.get(
            "decoded_recurrence", {}
        ).items():
            repeat = _parse_repeat(repeat_key)
            if not isinstance(recurrence, Mapping):
                continue
            _append_metrics(
                records,
                manifest,
                recurrence.get("rule30_from_previous_decoded", {}),
                step=step,
                data_split=data_split,
                evaluation_role="decoded_recurrence",
                checkpoint_type=checkpoint_type,
                length=length_int,
                ca_steps=repeat,
                num_repeats=repeat,
                repeat_from=repeat - 1,
                repeat_to=repeat,
            )
            for relation, similarity in (
                ("previous", recurrence.get("versus_previous_decoded")),
                (
                    "two_repeats_earlier",
                    recurrence.get("versus_two_repeats_earlier"),
                ),
            ):
                if not isinstance(similarity, Mapping):
                    continue
                repeat_from = repeat - (1 if relation == "previous" else 2)
                _append_metrics(
                    records,
                    manifest,
                    similarity,
                    step=step,
                    data_split=data_split,
                    evaluation_role="decoded_similarity",
                    checkpoint_type=checkpoint_type,
                    length=length_int,
                    num_repeats=repeat,
                    repeat_from=repeat_from,
                    repeat_to=repeat,
                )

        for left_key, right_values in length_data.get(
            "decoded_similarity", {}
        ).items():
            if not isinstance(right_values, Mapping):
                continue
            for right_key, metrics in right_values.items():
                _append_metrics(
                    records,
                    manifest,
                    metrics,
                    step=step,
                    data_split=data_split,
                    evaluation_role="decoded_state_similarity",
                    checkpoint_type=checkpoint_type,
                    length=length_int,
                    repeat_from=_parse_repeat(left_key),
                    repeat_to=_parse_repeat(right_key),
                )

        hidden = length_data.get("hidden_state_similarity", {})
        if isinstance(hidden, Mapping):
            hidden = hidden.get("similarity_matrix", hidden)
        if isinstance(hidden, Mapping):
            for left_key, right_values in hidden.items():
                if not isinstance(right_values, Mapping):
                    continue
                for right_key, metrics in right_values.items():
                    _append_metrics(
                        records,
                        manifest,
                        metrics,
                        step=step,
                        data_split=data_split,
                        evaluation_role="hidden_state_similarity",
                        checkpoint_type=checkpoint_type,
                        length=length_int,
                        repeat_from=_parse_repeat(left_key),
                        repeat_to=_parse_repeat(right_key),
                    )


def _append_clean_state_transitions(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    diagnostics: Any,
    *,
    step: int | None,
    data_split: str,
    checkpoint_type: str | None = None,
) -> None:
    """Normalize preserved-cache clean-current transition metrics."""
    if not isinstance(diagnostics, Mapping):
        return
    for length, length_data in diagnostics.items():
        if not isinstance(length_data, Mapping):
            continue
        for transition in length_data.get("transitions", {}).values():
            if not isinstance(transition, Mapping):
                continue
            source_depth = int(transition["source_ca_steps"])
            target_depth = int(transition["target_ca_steps"])
            _append_metrics(
                records,
                manifest,
                transition.get("metrics", {}),
                step=step,
                data_split=data_split,
                evaluation_role="preserved_cache_clean_state_transition",
                checkpoint_type=checkpoint_type,
                length=int(length),
                ca_steps=target_depth,
                num_repeats=int(transition["num_repeats"]),
                repeat_from=source_depth,
                repeat_to=target_depth,
            )


def _append_training_exposure(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    exposure: Any,
    *,
    step: int,
) -> None:
    if not isinstance(exposure, Mapping):
        return
    totals = {
        key: value
        for key, value in exposure.items()
        if key != "by_training_pair"
    }
    totals["accounting_exact"] = int(bool(exposure.get("accounting_exact", False)))
    _append_metrics(
        records,
        manifest,
        totals,
        step=step,
        data_split="training",
        evaluation_role="training_exposure",
    )
    for pair in exposure.get("by_training_pair", {}).values():
        if not isinstance(pair, Mapping):
            continue
        metrics = {
            key: value
            for key, value in pair.items()
            if key not in {"ca_steps", "num_repeats"}
        }
        _append_metrics(
            records,
            manifest,
            metrics,
            step=step,
            data_split="training",
            evaluation_role="training_exposure_by_pair",
            ca_steps=int(pair["ca_steps"]),
            num_repeats=int(pair["num_repeats"]),
        )


def _append_final_task_metrics(
    records: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    task_metrics: Any,
    *,
    step: int | None,
    checkpoint_type: str,
) -> None:
    if not isinstance(task_metrics, Mapping):
        return
    _append_in_distribution(
        records,
        manifest,
        task_metrics.get("in_distribution", {}),
        step=step,
        data_split="final_test",
        checkpoint_type=checkpoint_type,
    )
    _append_pair_mapping(
        records,
        manifest,
        task_metrics.get("internal_repeat_extrapolation", {}),
        step=step,
        data_split="final_test",
        evaluation_role="internal_repeat_extrapolation",
        checkpoint_type=checkpoint_type,
    )
    for rollout in task_metrics.get("external_rollout", {}).values():
        if not isinstance(rollout, Mapping) or "by_length" not in rollout:
            continue
        _append_by_length(
            records,
            manifest,
            rollout["by_length"],
            step=step,
            data_split="final_test",
            evaluation_role="external_rollout",
            checkpoint_type=checkpoint_type,
            ca_steps=int(rollout["ca_steps"]),
            num_repeats=(
                int(rollout["num_repeats_per_call"])
                if rollout.get("num_repeats_per_call") is not None
                else None
            ),
        )


def normalize_training_stats(
    stats: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Convert nested training statistics into stable long-form records."""

    records: list[dict[str, Any]] = []
    for step_key, evaluation in stats.get("eval", {}).items():
        if not isinstance(evaluation, Mapping):
            continue
        step = int(step_key)
        _append_in_distribution(
            records,
            manifest,
            evaluation.get("in_distribution", {}),
            step=step,
            data_split="validation",
        )
        _append_pair_mapping(
            records,
            manifest,
            evaluation.get("extrapolation_validation", {}),
            step=step,
            data_split="validation",
            evaluation_role="extrapolation_validation",
        )
        _append_repeat_diagnostics(
            records,
            manifest,
            evaluation.get("repeat_diagnostics", {}),
            step=step,
            data_split="validation",
        )
        _append_training_exposure(
            records,
            manifest,
            evaluation.get("training_exposure", {}),
            step=step,
        )

    canonical_checkpoints = (
        "best_id",
        "best_extrapolation_strict",
        "best_extrapolation_unconstrained",
    )
    analyses = stats.get("checkpoint_analysis", {})
    for checkpoint_type in canonical_checkpoints:
        analysis = analyses.get(checkpoint_type)
        if not isinstance(analysis, Mapping):
            continue
        checkpoint = analysis.get("checkpoint", {})
        step_value = checkpoint.get("step") if isinstance(checkpoint, Mapping) else None
        step = int(step_value) if step_value is not None else None
        _append_final_task_metrics(
            records,
            manifest,
            analysis.get("task_metrics", {}),
            step=step,
            checkpoint_type=checkpoint_type,
        )
        _append_repeat_diagnostics(
            records,
            manifest,
            analysis.get("repeat_diagnostics", {}),
            step=step,
            data_split="final_test",
            checkpoint_type=checkpoint_type,
        )
        _append_clean_state_transitions(
            records,
            manifest,
            analysis.get("preserved_cache_clean_state_transitions", {}),
            step=step,
            data_split="final_test",
            checkpoint_type=checkpoint_type,
        )

    sort_fields = (
        "run_id",
        "forward_policy_label",
        "data_split",
        "checkpoint_type",
        "step",
        "evaluation_role",
        "length",
        "ca_steps",
        "num_repeats",
        "repeat_from",
        "repeat_to",
        "metric",
    )
    records.sort(
        key=lambda record: tuple(
            (record.get(field) is None, str(record.get(field)))
            for field in sort_fields
        )
    )
    return records


def write_eval_metrics(
    run_dir: Path | str,
    stats: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    if manifest is None:
        manifest = read_json(run_dir / MANIFEST_FILENAME)
    records = normalize_training_stats(stats, manifest)
    lines = [
        json.dumps(record, sort_keys=True, allow_nan=False) for record in records
    ]
    text = "\n".join(lines)
    if text:
        text += "\n"
    path = run_dir / METRICS_FILENAME
    _atomic_write_text(path, text)
    return path


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return human-readable validation errors while allowing future fields."""

    errors: list[str] = []
    for key in (
        "schema_version",
        "run_id",
        "status",
        "model",
        "training",
        "seeds",
        "provenance",
        "resolved_args",
    ):
        if key not in manifest:
            errors.append(f"missing required field: {key}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version: {manifest.get('schema_version')!r}"
        )
    pairs = manifest.get("training", {}).get("pairs", [])
    if not isinstance(pairs, Sequence) or not pairs:
        errors.append("training.pairs must contain at least one pair")
    else:
        for index, pair in enumerate(pairs):
            if not isinstance(pair, Mapping):
                errors.append(f"training.pairs[{index}] must be an object")
                continue
            if "ca_steps" not in pair or "num_repeats" not in pair:
                errors.append(
                    f"training.pairs[{index}] requires ca_steps and num_repeats"
                )
    return errors
