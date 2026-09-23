"""Discover, compare, summarize, and plot Rule 30 experiment runs.

Examples
--------
List matching runs::

    python -m cellular_automaton.ca_analyze list \
        --where model.model=but_full_depth

Compare final extrapolation across seeds::

    python -m cellular_automaton.ca_analyze table \
        --metric cell_accuracy --split final_test \
        --role internal_repeat_extrapolation \
        --checkpoint best_extrapolation_unconstrained

Rank individual runs by their best validation step::

    python -m cellular_automaton.ca_analyze leaderboard \
        --where model=ca_cotf \
        --role repeat_horizon_diagnostic --select-pair 7:7 \
        --report-pairs 5:5 6:6 7:7 8:8 9:9

Compare delayed-recall runs using two selected metrics::

    python -m cellular_automaton.ca_analyze delayed-compare \
        --run-id run_141__dca_but_h6_intermediate_persistent \
                 run_142__dca_but_h6_intermediate_subtract \
        --metrics cell_accuracy exact_sequence_accuracy

Compare one-pass and two-pass recall training at one horizon::

    python -m cellular_automaton.ca_analyze recall-repeat-compare \
        --runs iridis/dca-30/runs \
        --where model=dca_cotf_cache --horizon 30

Rank delayed-recall runs (the first metric determines rank)::

    python -m cellular_automaton.ca_analyze delayed-leaderboard \
        --where model~dca \
        --metrics cell_accuracy ground_truth_unique_best_rate

Plot extrapolation validation throughout training::

    python -m cellular_automaton.ca_analyze plot-training \
        --role extrapolation_validation --metric cell_accuracy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .ca_reporting import (
        MANIFEST_FILENAME,
        METRICS_FILENAME,
        forward_policy_metadata,
        read_json,
        validate_manifest,
        write_json,
    )
except ImportError:
    from ca_reporting import (
        MANIFEST_FILENAME,
        METRICS_FILENAME,
        forward_policy_metadata,
        read_json,
        validate_manifest,
        write_json,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = REPO_ROOT / "iridis" / "ca-rule30" / "runs"
DEFAULT_REPORTS_ROOT = REPO_ROOT / "iridis" / "ca-rule30" / "reports"

FIELD_ALIASES = {
    "model": "model.model",
    "seed": "seeds.seed",
    "model_seed": "seeds.seed",
    "data_seed": "seeds.data_seed",
    "val_seed": "seeds.ca_val_seed",
    "test_seed": "seeds.ca_test_seed",
    "optimizer": "training.opt",
    "lr": "training.lr",
    "forget_gate": "model.lstm_forget_gate",
    "control_input": "model.lstm_control_input",
    "weight_decay": "training.weight_decay",
    "grad_clip": "training.grad_clip",
    "training_pairs": "training.pairs",
    "cache_policy": "forward_policy.repeat_cache_policy",
    "cache_window": "forward_policy.repeat_cache_window",
    "repeat_cache_window": "forward_policy.repeat_cache_window",
    "training_cache_policy": "forward_policy.repeat_cache_policy",
    "training_cache_window": "forward_policy.repeat_cache_window",
    "trained_cache_window": "forward_policy.repeat_cache_window",
    "forward_policy_label": "forward_policy.forward_policy_label",
}
FILTER_PATTERN = re.compile(r"^(.+?)(!=|>=|<=|=|>|<|~)(.*)$")


@dataclass(frozen=True)
class Run:
    path: Path
    manifest: Mapping[str, Any]
    metrics: tuple[Mapping[str, Any], ...]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])


@dataclass(frozen=True)
class DelayedMetricSpec:
    """Map one user-facing delayed metric across normalized report roles."""

    label: str
    overall_role: str
    overall_metric: str
    horizon_role: str
    horizon_metric: str
    query_role: str
    query_metric: str
    query_requires_requested_candidate: bool = False
    id_metric: str | None = None
    direction: str = "max"


DELAYED_METRICS = {
    "cell_accuracy": DelayedMetricSpec(
        "cell",
        "delayed_recall_nontrivial_pair_macro",
        "cell_accuracy",
        "delayed_recall_nontrivial_queries_macro",
        "cell_accuracy",
        "delayed_recall_ground_truth_requested",
        "cell_accuracy",
        id_metric="cell_accuracy",
    ),
    "exact_sequence_accuracy": DelayedMetricSpec(
        "exact",
        "delayed_recall_nontrivial_pair_macro",
        "exact_sequence_accuracy",
        "delayed_recall_nontrivial_queries_macro",
        "exact_sequence_accuracy",
        "delayed_recall_ground_truth_requested",
        "exact_sequence_accuracy",
        id_metric="exact_sequence_accuracy",
    ),
    "matthews_correlation": DelayedMetricSpec(
        "MCC",
        "delayed_recall_nontrivial_pair_macro",
        "matthews_correlation",
        "delayed_recall_nontrivial_queries_macro",
        "matthews_correlation",
        "delayed_recall_ground_truth_requested",
        "matthews_correlation",
        id_metric="matthews_correlation",
    ),
    "mean_bit_errors_per_sequence": DelayedMetricSpec(
        "bit errors",
        "delayed_recall_nontrivial_pair_macro",
        "mean_bit_errors_per_sequence",
        "delayed_recall_nontrivial_queries_macro",
        "mean_bit_errors_per_sequence",
        "delayed_recall_ground_truth_requested",
        "mean_bit_errors_per_sequence",
        id_metric="mean_bit_errors_per_sequence",
        direction="min",
    ),
    "loss": DelayedMetricSpec(
        "loss",
        "delayed_recall_nontrivial_pair_macro",
        "loss",
        "delayed_recall_nontrivial_queries_macro",
        "loss",
        "delayed_recall_ground_truth_requested",
        "loss",
        id_metric="loss",
        direction="min",
    ),
    "internal_cell_accuracy": DelayedMetricSpec(
        "internal cell",
        "delayed_recall_nontrivial_internal_pair_macro",
        "decoded_requested_repeat_cell_accuracy",
        "delayed_recall_nontrivial_internal_macro",
        "decoded_requested_repeat_cell_accuracy",
        "delayed_recall_internal_decoded_requested",
        "cell_accuracy",
    ),
    "changed_cell_accuracy": DelayedMetricSpec(
        "changed cell",
        "delayed_recall_nontrivial_pair_macro",
        "cell_accuracy",
        "delayed_recall_nontrivial_queries_macro",
        "cell_accuracy",
        "delayed_recall_ground_truth_requested",
        "cell_accuracy",
    ),
    "internal_changed_cell_accuracy": DelayedMetricSpec(
        "internal changed cell",
        "delayed_recall_nontrivial_internal_pair_macro",
        "decoded_requested_repeat_cell_accuracy",
        "delayed_recall_nontrivial_internal_macro",
        "decoded_requested_repeat_cell_accuracy",
        "delayed_recall_internal_decoded_requested",
        "cell_accuracy",
    ),
    "internal_exact_sequence_accuracy": DelayedMetricSpec(
        "internal exact",
        "delayed_recall_nontrivial_internal_pair_macro",
        "decoded_requested_repeat_exact_sequence_accuracy",
        "delayed_recall_nontrivial_internal_macro",
        "decoded_requested_repeat_exact_sequence_accuracy",
        "delayed_recall_internal_decoded_requested",
        "exact_sequence_accuracy",
    ),
    "ground_truth_rank": DelayedMetricSpec(
        "GT rank",
        "delayed_recall_nontrivial_ground_truth_retrieval_pair_macro",
        "requested_repeat_mean_rank",
        "delayed_recall_nontrivial_ground_truth_retrieval_macro",
        "requested_repeat_mean_rank",
        "delayed_recall_ground_truth_retrieval",
        "requested_repeat_mean_rank",
        direction="min",
    ),
    "ground_truth_unique_best_rate": DelayedMetricSpec(
        "GT unique best",
        "delayed_recall_nontrivial_ground_truth_retrieval_pair_macro",
        "requested_repeat_is_unique_best_rate",
        "delayed_recall_nontrivial_ground_truth_retrieval_macro",
        "requested_repeat_is_unique_best_rate",
        "delayed_recall_ground_truth_retrieval",
        "requested_repeat_is_unique_best_rate",
    ),
    "cosine_similarity": DelayedMetricSpec(
        "cosine",
        "delayed_recall_nontrivial_internal_pair_macro",
        "requested_repeat_logit_cosine_similarity",
        "delayed_recall_nontrivial_internal_macro",
        "requested_repeat_logit_cosine_similarity",
        "delayed_recall_internal_logit_similarity",
        "cosine_similarity",
        query_requires_requested_candidate=True,
    ),
    "cosine_rank": DelayedMetricSpec(
        "cosine rank",
        "delayed_recall_nontrivial_internal_pair_macro",
        "cosine_requested_repeat_mean_rank",
        "delayed_recall_nontrivial_internal_macro",
        "cosine_requested_repeat_mean_rank",
        "delayed_recall_cosine_retrieval",
        "requested_repeat_mean_rank",
        direction="min",
    ),
    "cosine_unique_best_rate": DelayedMetricSpec(
        "cosine unique best",
        "delayed_recall_nontrivial_internal_pair_macro",
        "cosine_requested_repeat_is_unique_best_rate",
        "delayed_recall_nontrivial_internal_macro",
        "cosine_requested_repeat_is_unique_best_rate",
        "delayed_recall_cosine_retrieval",
        "requested_repeat_is_unique_best_rate",
    ),
}

DELAYED_CANDIDATE_ROLES = frozenset(
    {
        "delayed_recall_ground_truth_candidate",
        "delayed_recall_ground_truth_collision",
        "delayed_recall_internal_decoded_candidate",
        "delayed_recall_internal_decoded_collision",
        "delayed_recall_internal_logit_similarity",
    }
)

DELAYED_CANDIDATE_METRICS = frozenset(
    {
        "cell_accuracy",
        "exact_sequence_accuracy",
        "matthews_correlation",
        "mean_bit_errors_per_sequence",
        "cell_agreement",
        "exact_row_collision_rate",
        "cosine_similarity",
        "normalized_mse",
    }
)

DELAYED_CHANGED_CELL_SUPPORT = {
    "changed_cell_accuracy": (
        "delayed_recall_ground_truth_candidate",
        "delayed_recall_ground_truth_collision",
    ),
    "internal_changed_cell_accuracy": (
        "delayed_recall_internal_decoded_candidate",
        "delayed_recall_internal_decoded_collision",
    ),
}

DELAYED_PROTOCOL_FIELDS = (
    "dataset",
    "sequence_length",
    "attention_mode",
    "positional_encoder",
    "lm_cache",
    "ca_train_num_cells",
    "ca_train_pairs",
    "ca_train_samples",
    "ca_boundary",
    "ca_bernoulli_p",
    "ca_data_mode",
    "ca_exact_data_resume",
    "ca_delayed_percentage",
    "query_horizon_policy",
    "ca_query_loss_weight",
    "ca_max_relative_age",
    "batch_size",
    "acc_steps",
    "iterations",
    "opt",
    "lr",
    "weight_decay",
    "grad_clip",
    "scheduler",
    "ca_eval_num_cells",
    "ca_val_samples",
    "ca_test_samples",
    "ca_val_seed",
    "ca_test_seed",
    "ca_delayed_best_metric",
    "eval_freq",
)


def canonical_pairs(pairs: Any) -> str:
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        return str(pairs)
    normalized = []
    for pair in pairs:
        if isinstance(pair, Mapping):
            normalized.append(
                (int(pair["ca_steps"]), int(pair["num_repeats"]))
            )
        else:
            normalized.append((int(pair[0]), int(pair[1])))
    return "|".join(f"{steps}:{repeats}" for steps, repeats in normalized)


def dotted_get(value: Any, path: str, default: Any = None) -> Any:
    path = FIELD_ALIASES.get(path, path)
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if path.startswith("forward_policy.") and isinstance(value, Mapping):
                policy_field = path.split(".", 1)[1]
                return forward_policy_metadata(value).get(policy_field, default)
            return default
        current = current[part]
    if path == "training.pairs":
        return canonical_pairs(current)
    return current


def _parse_literal(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def optional_int(value: str) -> int | None:
    if value.lower() in {"any", "none"}:
        return None
    return int(value)


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == "~":
        return str(right).lower() in str(left).lower()
    if operator in {">", ">=", "<", "<="}:
        try:
            left, right = float(left), float(right)
        except (TypeError, ValueError):
            left, right = str(left), str(right)
    if operator == "=":
        return left == right or str(left) == str(right)
    if operator == "!=":
        return not (left == right or str(left) == str(right))
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    raise ValueError(f"Unsupported filter operator: {operator}")


def parse_filter(expression: str) -> tuple[str, str, Any]:
    match = FILTER_PATTERN.match(expression)
    if match is None:
        raise ValueError(
            f"Invalid filter {expression!r}; use FIELD=VALUE, FIELD>=VALUE, "
            "FIELD!=VALUE, or FIELD~SUBSTRING."
        )
    field, operator, raw_value = match.groups()
    return field.strip(), operator, _parse_literal(raw_value.strip())


def manifest_matches(
    manifest: Mapping[str, Any],
    filters: Sequence[tuple[str, str, Any]],
) -> bool:
    missing = object()
    for field, operator, expected in filters:
        actual = dotted_get(manifest, field, missing)
        if actual is missing or not _compare(actual, operator, expected):
            return False
    return True


def iter_metrics(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error


def read_metrics(path: Path) -> tuple[Mapping[str, Any], ...]:
    return tuple(iter_metrics(path))


def discover_runs(
    root: Path | str,
    *,
    filters: Sequence[tuple[str, str, Any]] = (),
    strict: bool = False,
    load_metrics: bool = True,
) -> list[Run]:
    root = Path(root)
    runs = []
    for manifest_path in sorted(root.rglob(MANIFEST_FILENAME)):
        manifest = read_json(manifest_path)
        errors = validate_manifest(manifest)
        if errors:
            message = f"{manifest_path}: " + "; ".join(errors)
            if strict:
                raise ValueError(message)
            print(f"WARNING: skipping invalid manifest: {message}", file=sys.stderr)
            continue
        if not manifest_matches(manifest, filters):
            continue
        runs.append(
            Run(
                path=manifest_path.parent,
                manifest=manifest,
                metrics=(
                    read_metrics(manifest_path.parent / METRICS_FILENAME)
                    if load_metrics
                    else ()
                ),
            )
        )
    return runs


def _field_list(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def comparison_configuration(
    manifest: Mapping[str, Any],
    *,
    average_over: Sequence[str],
) -> dict[str, Any]:
    """Scientific configuration that must agree before seed aggregation.

    Starting from the complete resolved argument set makes grouping
    conservative and future-proof: a newly introduced model or data argument
    automatically prevents accidental aggregation until it is deliberately
    classified as reporting-only.
    """

    resolved_args = dict(manifest.get("resolved_args", {}))
    for reporting_only in (
        "exp_name",
        "results_base_folder",
        "ca_run_dir",
        "ca_tags",
        "ca_note",
        "wandb",
        "wandb_project",
        "wandb_entity",
        "ca_save_every",
        "ca_log_every",
    ):
        resolved_args.pop(reporting_only, None)
    for coverage_only in (
        "ca_extrapolation_val_pairs",
        "ca_final_eval_pairs",
        "ca_repeat_diagnostic_max_repeats",
        "ca_repeat_diagnostic_horizons",
    ):
        resolved_args.pop(coverage_only, None)

    seed_argument_names = {
        "seed": "seed",
        "model_seed": "seed",
        "data_seed": "data_seed",
        "val_seed": "ca_val_seed",
        "test_seed": "ca_test_seed",
    }
    for field in average_over:
        argument_name = seed_argument_names.get(field, field)
        if "." in argument_name:
            argument_name = argument_name.rsplit(".", 1)[-1]
        resolved_args.pop(argument_name, None)
    policy = forward_policy_metadata(manifest)
    return {
        "resolved_args": resolved_args,
        "training_forward_policy": {
            "repeat_cache_policy": policy["repeat_cache_policy"],
            "repeat_cache_window": policy["repeat_cache_window"],
        },
    }


def configuration_id(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]

def lstm_ut_configuration_fields(
    manifest: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Return LSTM-UT ablation fields, or null fields for other models."""

    model = str(dotted_get(manifest, "model", ""))
    if "lstm_ut" not in model.lower():
        return None, None

    return (
        dotted_get(manifest, "model.lstm_forget_gate", "?"),
        dotted_get(manifest, "model.lstm_control_input", "?"),
    )


def configuration_label(manifest: Mapping[str, Any]) -> str:
    model = dotted_get(manifest, "model", "?")
    architecture = architecture_signature(manifest)
    pairs = dotted_get(manifest, "training_pairs", "?")
    optimizer = dotted_get(manifest, "optimizer", "?")
    lr = dotted_get(manifest, "lr", "?")
    weight_decay = dotted_get(manifest, "weight_decay", "?")
    grad_clip = dotted_get(manifest, "grad_clip", "?")
    cache_policy = forward_policy_signature(manifest)
    forget_gate, control_input = lstm_ut_configuration_fields(manifest)

    lstm_configuration = ""
    if forget_gate is not None:
        initial_cell = dotted_get(
            manifest, "model.lstm_initial_cell", "zero"
        )
        lstm_configuration = (
            f" lstm-forget={forget_gate}"
            f" lstm-control={control_input}"
            f" lstm-initial-cell={initial_cell}"
        )

    return (
        f"{model} {architecture}{lstm_configuration} {cache_policy} "
        f"pairs={pairs} opt={optimizer} lr={lr} "
        f"wd={weight_decay} clip={grad_clip}"
    )

def forward_policy_signature(manifest: Mapping[str, Any]) -> str:
    """Return a concise training-time cache-policy label for configurations."""

    return f"train-cache={training_cache_label(manifest)}"


def training_cache_label(manifest: Mapping[str, Any]) -> str:
    """Return the cache visibility used by the run's training forwards."""

    policy = forward_policy_metadata(manifest)
    window = policy["repeat_cache_window"]
    if window is None:
        return "full"
    return f"recent-{window}"


def architecture_signature(manifest: Mapping[str, Any]) -> str:
    model = manifest.get("model", {})
    total_layers = int(model.get("n_layer", 0) or 0)
    begin_layers = int(model.get("n_layer_begin", 0) or 0)
    end_layers = int(model.get("n_layer_end", 0) or 0)
    middle_layers = max(total_layers - begin_layers - end_layers, 0)
    attention = model.get("attention_mode", "?")
    positional = model.get("positional_encoder", "?")
    embedding = model.get("n_embd", "?")
    heads = model.get("n_head", "?")
    return (
        f"{attention}/{positional} d{embedding}h{heads} "
        f"layers={begin_layers}/{middle_layers}/{end_layers}"
    )


def evaluation_protocol_signature(manifest: Mapping[str, Any]) -> str:
    """Return the evaluation coverage and selection policy for one run."""
    evaluation = manifest.get("evaluation", {})
    training_pairs = dotted_get(manifest, "training_pairs", "?")
    validation_pairs = canonical_pairs(
        evaluation.get("ca_extrapolation_val_pairs", [])
    )
    final_pairs = canonical_pairs(evaluation.get("ca_final_eval_pairs", []))
    selection_pair = evaluation.get("ca_extrapolation_best_pair")
    if (
        isinstance(selection_pair, Sequence)
        and not isinstance(selection_pair, (str, bytes))
        and len(selection_pair) == 2
    ):
        selection = _pair_label(
            (int(selection_pair[0]), int(selection_pair[1]))
        )
    else:
        selection = "?"
    max_repeats = evaluation.get("ca_repeat_diagnostic_max_repeats", "?")
    horizons = evaluation.get("ca_repeat_diagnostic_horizons")
    if isinstance(horizons, Sequence) and not isinstance(horizons, (str, bytes)):
        horizon_label = ",".join(str(int(value)) for value in horizons)
    else:
        horizon_label = "?"
    return (
        f"train={training_pairs or 'none'} "
        f"val={validation_pairs or 'none'} select={selection} "
        f"direct_final={final_pairs or 'none'} "
        f"diagnostic_horizons={horizon_label} repeats=1:{max_repeats}"
    )


def group_runs(
    runs: Sequence[Run],
    *,
    average_over: Sequence[str],
) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        config = comparison_configuration(
            run.manifest, average_over=average_over
        )
        groups[configuration_id(config)].append(run)
    return dict(groups)


def summarize(values: Sequence[float]) -> dict[str, Any]:
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("Cannot summarize an empty sequence.")

    def percentile(fraction: float) -> float:
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "q1": percentile(0.25),
        "q3": percentile(0.75),
        "min": values[0],
        "max": values[-1],
        "values": values,
    }


def _record_matches(record: Mapping[str, Any], args: argparse.Namespace) -> bool:
    comparisons = (
        ("metric", args.metric),
        ("data_split", args.split),
        ("evaluation_role", args.role),
        ("checkpoint_type", args.checkpoint),
        ("length", args.length),
    )
    for field, expected in comparisons:
        if field == "checkpoint_type" and expected == "any":
            continue
        if field == "checkpoint_type" and expected == "none":
            if record.get(field) is not None:
                return False
            continue
        if expected is not None and record.get(field) != expected:
            return False
    return True


def select_records(
    run: Run,
    args: argparse.Namespace,
    *,
    keep_all_steps: bool = False,
) -> list[Mapping[str, Any]]:
    return list(
        iter_selected_records(
            run.metrics,
            args,
            keep_all_steps=keep_all_steps,
        )
    )


def iter_selected_records(
    records: Iterable[Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    keep_all_steps: bool = False,
    requested_pairs: frozenset[tuple[int, int]] | None = None,
) -> Iterable[Mapping[str, Any]]:
    """Yield matches while retaining only latest-step state when necessary."""

    latest = not keep_all_steps and args.step == "latest"
    requested_step = (
        int(args.step)
        if not keep_all_steps
        and args.step not in {None, "all", "final", "latest"}
        else None
    )

    by_measurement: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in records:
        if not _record_matches(record, args):
            continue
        if requested_pairs is not None and _pair(record) not in requested_pairs:
            continue
        if keep_all_steps or args.step in {None, "all"}:
            yield record
            continue
        if args.step == "final":
            if record.get("data_split") == "final_test":
                yield record
            continue
        if requested_step is not None:
            if record.get("step") == requested_step:
                yield record
            continue
        if not latest:
            continue

        key = tuple(
            record.get(field)
            for field in (
                "data_split",
                "evaluation_role",
                "checkpoint_type",
                "length",
                "ca_steps",
                "num_repeats",
                "repeat_from",
                "repeat_to",
                "metric",
            )
        )
        previous = by_measurement.get(key)
        step = record.get("step")
        previous_step = previous.get("step") if previous else None
        if previous is None or (
            step is not None
            and (previous_step is None or int(step) > int(previous_step))
        ):
            by_measurement[key] = record
    if latest:
        yield from by_measurement.values()


def _pair(record: Mapping[str, Any]) -> tuple[int | None, int | None]:
    return record.get("ca_steps"), record.get("num_repeats")


def _pair_label(pair: tuple[int | None, int | None]) -> str:
    steps, repeats = pair
    if steps is None and repeats is None:
        return "n/a"
    return f"{steps}:{repeats}"


def _pair_sort_key(pair: tuple[int | None, int | None]) -> tuple[int, int]:
    return (
        int(pair[0]) if pair[0] is not None else -1,
        int(pair[1]) if pair[1] is not None else -1,
    )


def _category_sort_key(value: Any) -> tuple[Any, ...]:
    match = re.fullmatch(r"(-?\d+):(-?\d+)", str(value))
    if match is not None:
        return 0, int(match.group(1)), int(match.group(2))
    return 1, str(value)


def aggregate_records(
    runs: Sequence[Run],
    args: argparse.Namespace,
    *,
    keep_all_steps: bool = False,
    stream_metrics: bool = False,
) -> tuple[dict[str, list[Run]], list[dict[str, Any]]]:
    average_over = _field_list(args.average_over, ("seed", "data_seed"))
    groups = group_runs(runs, average_over=average_over)
    run_to_group = {
        run.run_id: group_id
        for group_id, members in groups.items()
        for run in members
    }
    buckets: dict[
        tuple[Any, ...], list[tuple[float, Run, Mapping[str, Any]]]
    ] = defaultdict(list)
    requested_pairs = getattr(args, "report_pairs", None)
    requested_pair_set = (
        None if requested_pairs is None else frozenset(requested_pairs)
    )
    for run in runs:
        records = (
            iter_metrics(run.path / METRICS_FILENAME)
            if stream_metrics
            else run.metrics
        )
        for record in iter_selected_records(
            records,
            args,
            keep_all_steps=keep_all_steps,
            requested_pairs=requested_pair_set,
        ):
            key = (
                run_to_group[run.run_id],
                record.get("step") if keep_all_steps else None,
                record.get("length"),
                record.get("ca_steps"),
                record.get("num_repeats"),
                record.get("repeat_from"),
                record.get("repeat_to"),
            )
            buckets[key].append((float(record["value"]), run, record))

    rows = []
    for key, observations in sorted(
        buckets.items(), key=lambda item: str(item[0])
    ):
        (
            group_id,
            step,
            length,
            ca_steps,
            num_repeats,
            repeat_from,
            repeat_to,
        ) = key
        values = [value for value, _run, _record in observations]
        summary = summarize(values)
        representative = groups[group_id][0]
        contributing_runs = sorted(
            {run.run_id: run for _value, run, _record in observations}.values(),
            key=lambda run: run.run_id,
        )
        best_value, best_run, best_record = max(
            observations,
            key=lambda observation: (
                observation[0],
                int(observation[2].get("step") or -1),
                observation[1].run_id,
            ),
        )
        rows.append(
            {
                "configuration_id": group_id,
                "configuration": configuration_label(
                    representative.manifest
                ),
                "training_pairs": dotted_get(
                    representative.manifest, "training_pairs"
                ),
                "step": step,
                "length": length,
                "ca_steps": ca_steps,
                "num_repeats": num_repeats,
                "repeat_from": repeat_from,
                "repeat_to": repeat_to,
                "run_ids": [run.run_id for run in contributing_runs],
                "model_seeds": [
                    dotted_get(run.manifest, "seed")
                    for run in contributing_runs
                ],
                "maximum_run_id": best_run.run_id,
                "maximum_model_seed": dotted_get(best_run.manifest, "seed"),
                "maximum_step": best_record.get("step"),
                "maximum_value": best_value,
                **summary,
            }
        )
    return groups, rows


def _format_summary(summary: Mapping[str, Any]) -> str:
    if int(summary["n"]) == 1:
        return f"{summary['mean']:.6f} (n=1)"
    return (
        f"{summary['mean']:.6f} ± {summary['std']:.6f} "
        f"[min={summary['min']:.6f}, max={summary['max']:.6f}] "
        f"(n={summary['n']})"
    )


def _terminal_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rendered:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _report_dir(args: argparse.Namespace) -> Path:
    path = (
        Path(args.output_dir)
        if args.output_dir is not None
        else DEFAULT_REPORTS_ROOT / f"{_timestamp()}_{args.command}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and key != "values":
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_analysis_manifest(
    report_dir: Path,
    args: argparse.Namespace,
    runs: Sequence[Run],
) -> None:
    arguments = {
        key: value for key, value in vars(args).items() if key != "handler"
    }
    write_json(
        report_dir / "analysis_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "arguments": arguments,
            "selected_runs": [
                {"run_id": run.run_id, "path": str(run.path)} for run in runs
            ],
        },
    )


def command_list(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    rows = []
    for run in runs:
        metric_count = (
            len(run.metrics)
            if run.metrics
            else sum(1 for _ in iter_metrics(run.path / METRICS_FILENAME))
        )
        rows.append(
            (
                run.run_id,
                run.manifest.get("status"),
                dotted_get(run.manifest, "model"),
                dotted_get(run.manifest, "training_pairs"),
                training_cache_label(run.manifest),
                dotted_get(run.manifest, "seed"),
                dotted_get(run.manifest, "data_seed"),
                metric_count,
            )
        )
    if rows:
        print(
            _terminal_table(
                (
                    "run",
                    "status",
                    "model",
                    "training pairs",
                    "training cache",
                    "model seed",
                    "data seed",
                    "metric rows",
                ),
                rows,
            )
        )
    else:
        print("No matching runs.")
    return 0


def _checkpoint_label(value: Any) -> str:
    return "none" if value is None else str(value)


def _available_metric_selections(
    runs: Sequence[Run], args: argparse.Namespace
) -> list[tuple[str, str, str, int]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    same_role_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for run in runs:
        records = (
            run.metrics
            if run.metrics
            else iter_metrics(run.path / METRICS_FILENAME)
        )
        for record in records:
            if args.metric is not None and record.get("metric") != args.metric:
                continue
            if args.length is not None and record.get("length") != args.length:
                continue
            key = (
                str(record.get("data_split")),
                str(record.get("evaluation_role")),
                _checkpoint_label(record.get("checkpoint_type")),
            )
            counts[key] += 1
            if (
                args.role is not None
                and record.get("evaluation_role") == args.role
            ):
                same_role_counts[key] += 1
    selected_counts = same_role_counts or counts
    return [
        (*key, count) for key, count in sorted(selected_counts.items())
    ]


def _no_metric_records_message(
    runs: Sequence[Run],
    args: argparse.Namespace,
    *,
    available: Sequence[tuple[str, str, str, int]] | None = None,
) -> str:
    requested = (
        f"metric={args.metric}, split={args.split}, role={args.role}, "
        f"checkpoint={_checkpoint_label(args.checkpoint)}, "
        f"length={args.length if args.length is not None else 'any'}"
    )
    lines = [
        f"No metric records matched the requested selection ({requested})."
    ]
    if available is None:
        available = _available_metric_selections(runs, args)
    if available:
        lines.append("Available selections for the same role/metric when possible:")
        for split, role, checkpoint, count in available:
            lines.append(
                f"  split={split}, role={role}, checkpoint={checkpoint} "
                f"({count} records)"
            )
    elif runs:
        lines.append("The selected runs contain no records for that metric and length.")
    else:
        lines.append("No run manifests matched the --where filters.")
    return "\n".join(lines)


def command_table(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    if args.report_pairs is not None and len(set(args.report_pairs)) != len(
        args.report_pairs
    ):
        raise ValueError("--report-pairs cannot contain duplicate pairs.")

    groups, rows = aggregate_records(runs, args, stream_metrics=True)
    if not rows:
        raise ValueError(_no_metric_records_message(runs, args))

    pairs = (
        list(args.report_pairs)
        if args.report_pairs is not None
        else sorted(
            {(row["ca_steps"], row["num_repeats"]) for row in rows},
            key=_pair_sort_key,
        )
    )
    by_group_pair = {
        (row["configuration_id"], (row["ca_steps"], row["num_repeats"])): row
        for row in rows
    }
    rendered = []
    export_rows = []
    active_group_ids = {row["configuration_id"] for row in rows}
    for group_id, members in sorted(groups.items()):
        if group_id not in active_group_ids:
            continue
        representative = members[0]
        contributing_run_ids = sorted(
            {
                run_id
                for row in rows
                if row["configuration_id"] == group_id
                for run_id in row["run_ids"]
            }
        )
        contributing_members = [
            run for run in members if run.run_id in contributing_run_ids
        ]
        contributing_seeds = sorted(
            {
                dotted_get(run.manifest, "seed")
                for run in contributing_members
            },
            key=lambda value: (value is None, str(value)),
        )
        trained = {
            (int(pair["ca_steps"]), int(pair["num_repeats"]))
            for pair in representative.manifest["training"]["pairs"]
        }
        training_policy = forward_policy_metadata(representative.manifest)
        forget_gate, control_input = lstm_ut_configuration_fields(
            representative.manifest
        )
        values = []
        export = {
            "configuration_id": group_id,
            "configuration": configuration_label(representative.manifest),
            "architecture": architecture_signature(representative.manifest),
            "lstm_forget_gate": forget_gate,
            "lstm_control_input": control_input,
            "training_cache_policy": training_policy["repeat_cache_policy"],
            "training_cache_window": training_policy["repeat_cache_window"],
            "run_ids": contributing_run_ids,
            "model_seeds": contributing_seeds,
            "training_pairs": canonical_pairs(
                representative.manifest["training"]["pairs"]
            ),
        }
        for pair in pairs:
            row = by_group_pair.get((group_id, pair))
            value = "—" if row is None else _format_summary(row)
            if pair in trained and row is not None:
                value += " [trained]"
            values.append(value)
            if row is not None:
                for field in (
                    "n",
                    "mean",
                    "std",
                    "variance",
                    "median",
                    "q1",
                    "q3",
                    "min",
                    "max",
                ):
                    export[f"{_pair_label(pair)}_{field}"] = row[field]
                export[f"{_pair_label(pair)}_maximum_run_id"] = row[
                    "maximum_run_id"
                ]
                export[f"{_pair_label(pair)}_maximum_model_seed"] = row[
                    "maximum_model_seed"
                ]
                export[f"{_pair_label(pair)}_maximum_step"] = row[
                    "maximum_step"
                ]
        rendered.append(
            (
                configuration_label(representative.manifest),
                ",".join(contributing_run_ids),
                ",".join(map(str, contributing_seeds)),
                *values,
                group_id,
            )
        )
        export_rows.append(export)

    print(
        _terminal_table(
            (
                "configuration",
                "runs",
                "seeds",
                *map(_pair_label, pairs),
                "config id",
            ),
            rendered,
        )
    )
    print("\n[trained] marks pairs present in that configuration's training distribution.")

    if args.output_dir is not None:
        report_dir = _report_dir(args)
        _write_csv(report_dir / "pair_table.csv", export_rows)
        write_json(report_dir / "summary.json", rows)
        _write_analysis_manifest(report_dir, args, runs)
        print(f"\nWrote analysis to {report_dir}")
    return 0


def parse_pair_literal(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+):(\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            f"Invalid pair {value!r}; expected STEPS:REPEATS, for example 6:6."
        )
    return int(match.group(1)), int(match.group(2))


def _manifest_extrapolation_pair(run: Run) -> tuple[int, int] | None:
    pair = run.manifest.get("evaluation", {}).get(
        "ca_extrapolation_best_pair"
    )
    if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
        return None
    if len(pair) != 2:
        return None
    return int(pair[0]), int(pair[1])


def _leaderboard_selection_pair(
    args: argparse.Namespace, runs: Sequence[Run]
) -> tuple[int, int]:
    if args.select_pair is not None:
        return args.select_pair
    discovered = {
        pair
        for run in runs
        if (pair := _manifest_extrapolation_pair(run)) is not None
    }
    if len(discovered) == 1:
        return next(iter(discovered))
    if not discovered:
        raise ValueError(
            "No --select-pair was provided and the selected manifests do not "
            "define evaluation.ca_extrapolation_best_pair."
        )
    choices = ", ".join(_pair_label(pair) for pair in sorted(discovered))
    raise ValueError(
        "Selected runs use different extrapolation selection pairs "
        f"({choices}); pass --select-pair explicitly."
    )


def command_leaderboard(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    selection_pair = _leaderboard_selection_pair(args, runs)
    requested_pairs = (
        None if args.report_pairs is None else set(args.report_pairs)
    )
    rows = []
    missing = []
    available_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    same_role_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for run in runs:
        selected_step = None
        selected_value = None
        values_by_step: dict[
            int, dict[tuple[int | None, int | None], float]
        ] = defaultdict(dict)
        metrics_path = run.path / METRICS_FILENAME
        for record in iter_metrics(metrics_path):
            if (
                (args.metric is None or record.get("metric") == args.metric)
                and (args.length is None or record.get("length") == args.length)
            ):
                availability_key = (
                    str(record.get("data_split")),
                    str(record.get("evaluation_role")),
                    _checkpoint_label(record.get("checkpoint_type")),
                )
                available_counts[availability_key] += 1
                if (
                    args.role is not None
                    and record.get("evaluation_role") == args.role
                ):
                    same_role_counts[availability_key] += 1

            if not _record_matches(record, args):
                continue
            step = record.get("step")
            if step is None:
                continue
            step = int(step)
            pair = _pair(record)
            value = float(record["value"])
            if pair == selection_pair:
                effective_step = step or -1
                candidate_key = (
                    (value, effective_step)
                    if args.direction == "max"
                    else (value, -effective_step)
                )
                selected_key = (
                    None
                    if selected_step is None or selected_value is None
                    else (
                        (selected_value, selected_step or -1)
                        if args.direction == "max"
                        else (selected_value, -(selected_step or -1))
                    )
                )
                if selected_key is None or (
                    candidate_key > selected_key
                    if args.direction == "max"
                    else candidate_key < selected_key
                ):
                    selected_step = step
                    selected_value = value
            if requested_pairs is None or pair in requested_pairs:
                values_by_step[step][pair] = value

        if selected_step is None or selected_value is None:
            missing.append(run.run_id)
            continue
        pair_values = values_by_step.get(selected_step, {})
        manifest = run.manifest
        training_policy = forward_policy_metadata(manifest)
        forget_gate, control_input = lstm_ut_configuration_fields(manifest)
        rows.append(
            {
                "run_id": run.run_id,
                "status": manifest.get("status"),
                "model": dotted_get(manifest, "model"),
                "lstm_forget_gate": forget_gate,
                "lstm_control_input": control_input,
                "model_seed": dotted_get(manifest, "seed"),
                "data_seed": dotted_get(manifest, "data_seed"),
                "architecture": architecture_signature(manifest),
                "training_cache_policy": training_policy[
                    "repeat_cache_policy"
                ],
                "training_cache_window": training_policy[
                    "repeat_cache_window"
                ],
                "training_cache": training_cache_label(manifest),
                "optimizer": dotted_get(manifest, "optimizer"),
                "lr": dotted_get(manifest, "lr"),
                "weight_decay": dotted_get(manifest, "weight_decay"),
                "grad_clip": dotted_get(manifest, "grad_clip"),
                "selection_pair": _pair_label(selection_pair),
                "selection_step": selected_step,
                "selection_value": selected_value,
                "configuration_id": configuration_id(
                    comparison_configuration(manifest, average_over=())
                ),
                "evaluation_protocol": evaluation_protocol_signature(manifest),
                "pair_values": pair_values,
            }
        )

    if not rows:
        counts = same_role_counts or available_counts
        available = [
            (*key, count) for key, count in sorted(counts.items())
        ]
        raise ValueError(
            _no_metric_records_message(runs, args, available=available)
        )

    if args.direction == "max":
        rows.sort(key=lambda row: (-row["selection_value"], row["run_id"]))
    else:
        rows.sort(key=lambda row: (row["selection_value"], row["run_id"]))
    available_pairs = sorted(
        {pair for row in rows for pair in row["pair_values"]},
        key=_pair_sort_key,
    )
    if args.report_pairs is None:
        pairs = available_pairs
    else:
        pairs = list(args.report_pairs)
        if len(set(pairs)) != len(pairs):
            raise ValueError("--report-pairs cannot contain duplicate pairs.")

    protocol_signatures = sorted(
        {row["evaluation_protocol"] for row in rows}
    )
    protocol_ids = {
        signature: f"P{index}"
        for index, signature in enumerate(protocol_signatures, start=1)
    }
    rendered = []
    export_rows = []
    for rank, row in enumerate(rows, start=1):
        protocol_id = protocol_ids[row["evaluation_protocol"]]
        pair_values = [
            (
                "—"
                if pair not in row["pair_values"]
                else f"{row['pair_values'][pair]:.6f}"
            )
            for pair in pairs
        ]
        rendered.append(
            (
                rank,
                row["run_id"],
                row["model"],
                row["lstm_forget_gate"] or "—",
                row["lstm_control_input"] or "—",
                protocol_id,
                row["model_seed"],
                row["data_seed"],
                row["architecture"],
                row["training_cache"],
                row["weight_decay"],
                row["grad_clip"],
                row["selection_step"],
                *pair_values,
                row["configuration_id"],
            )
        )
        export = {
            key: value for key, value in row.items() if key != "pair_values"
        }
        export["rank"] = rank
        export["evaluation_protocol_id"] = protocol_id
        for pair in pairs:
            export[_pair_label(pair)] = row["pair_values"].get(pair)
        export_rows.append(export)

    print(
        _terminal_table(
            (
                "rank",
                "run",
                "model",
                "forget gate",
                "control input",
                "protocol",
                "seed",
                "data",
                "architecture",
                "training cache",
                "wd",
                "clip",
                f"best {_pair_label(selection_pair)} step",
                *map(_pair_label, pairs),
                "config id",
            ),
            rendered,
        )
    )
    print(
        f"\nRanked by {args.direction} {args.split} {args.metric} on "
        f"{_pair_label(selection_pair)}; every pair in a row is reported at "
        "that run's selected step."
    )
    protocol_prefix = "WARNING: mixed evaluation protocols" if len(
        protocol_signatures
    ) > 1 else "Evaluation protocol"
    print(f"{protocol_prefix}:")
    for signature in protocol_signatures:
        print(f"  {protocol_ids[signature]} {signature}")
    if missing:
        print(
            "Skipped selected runs without a matching selection-pair record: "
            + ", ".join(sorted(missing))
        )

    if args.output_dir is not None:
        report_dir = _report_dir(args)
        _write_csv(report_dir / "leaderboard.csv", export_rows)
        write_json(report_dir / "leaderboard.json", export_rows)
        _write_analysis_manifest(report_dir, args, runs)
        print(f"\nWrote leaderboard to {report_dir}")
    return 0


def _selected_delayed_specs(args: argparse.Namespace) -> dict[str, DelayedMetricSpec]:
    metric_names = tuple(args.metrics)
    if not 1 <= len(metric_names) <= 2:
        raise ValueError("--metrics requires one or two delayed metric names.")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("--metrics cannot contain duplicates.")
    return {name: DELAYED_METRICS[name] for name in metric_names}


def _select_delayed_runs(
    runs: Sequence[Run], requested_run_ids: Sequence[str] | None
) -> list[Run]:
    if not requested_run_ids:
        return list(runs)
    if len(set(requested_run_ids)) != len(requested_run_ids):
        raise ValueError("--run-id cannot contain duplicates.")
    by_id = {run.run_id: run for run in runs}
    missing = [run_id for run_id in requested_run_ids if run_id not in by_id]
    if missing:
        raise ValueError(
            "Requested run IDs were not found under --runs: "
            + ", ".join(missing)
        )
    return [by_id[run_id] for run_id in requested_run_ids]


def delayed_protocol_configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that should agree in a controlled DCA comparison."""

    resolved = manifest.get("resolved_args", {})
    protocol = {}
    for field in DELAYED_PROTOCOL_FIELDS:
        value = resolved.get(field)
        if field == "ca_train_pairs":
            value = canonical_pairs(value or ())
        elif isinstance(value, list):
            value = tuple(value)
        protocol[field] = value
    policy = forward_policy_metadata(manifest)
    protocol["repeat_cache_policy"] = policy["repeat_cache_policy"]
    protocol["repeat_cache_window"] = policy["repeat_cache_window"]
    return protocol


def _delayed_comparison_configuration(
    manifest: Mapping[str, Any], *, average_over: Sequence[str]
) -> dict[str, Any]:
    """Group replicate seeds without merging distinct DCA configurations."""

    configuration = comparison_configuration(
        manifest, average_over=average_over
    )
    resolved = dict(configuration["resolved_args"])
    if "data_seed" in average_over:
        # The DCA launcher derives the shuffle seed from the data seed unless
        # explicitly overridden. It belongs to replicate identity here.
        resolved.pop("ca_shuffle_seed", None)
    return {
        **configuration,
        "resolved_args": resolved,
    }


def _delayed_groups(
    runs: Sequence[Run], *, average_over: Sequence[str]
) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        configuration = _delayed_comparison_configuration(
            run.manifest, average_over=average_over
        )
        groups[configuration_id(configuration)].append(run)
    return dict(groups)


def _checkpoint_matches(record: Mapping[str, Any], requested: str) -> bool:
    if requested == "any":
        return True
    if requested == "none":
        return record.get("checkpoint_type") is None
    return record.get("checkpoint_type") == requested


def _put_latest(
    destination: dict[Any, dict[str, Any]],
    key: Any,
    record: Mapping[str, Any],
) -> None:
    step = record.get("step")
    previous = destination.get(key)
    previous_step = previous.get("step") if previous else None
    if previous is None or (
        step is not None
        and (previous_step is None or int(step) > int(previous_step))
    ):
        destination[key] = {
            "step": None if step is None else int(step),
            "value": float(record["value"]),
        }


def _changed_cell_accuracy(
    requested_accuracy: float,
    current_accuracy: float,
    requested_current_agreement: float,
) -> float | None:
    """Recover accuracy where requested and current binary cells differ.

    On a changed cell, a prediction that matches the requested bit cannot
    match the current bit, and vice versa.  This lets the masked accuracy be
    recovered exactly from the three aggregate pairwise agreements already
    present in delayed-recall reports.
    """

    changed_fraction = 1.0 - requested_current_agreement
    if changed_fraction <= 0.0:
        return None
    accuracy = (
        requested_accuracy - current_accuracy + changed_fraction
    ) / (2.0 * changed_fraction)
    # The inputs are ratios accumulated over the same cells.  Clamp only the
    # tiny floating-point spill that can occur at the endpoints.
    if accuracy < -1e-9 or accuracy > 1.0 + 1e-9:
        raise ValueError(
            "Inconsistent delayed-recall candidate metrics produced an "
            f"invalid changed-cell accuracy: {accuracy}."
        )
    return min(1.0, max(0.0, accuracy))


def _derive_delayed_changed_cell_metrics(
    result: dict[str, Any],
    specs: Mapping[str, DelayedMetricSpec],
) -> None:
    """Populate explicitly requested changed-cell recall diagnostics."""

    for metric, (candidate_role, collision_role) in (
        DELAYED_CHANGED_CELL_SUPPORT.items()
    ):
        if metric not in specs:
            continue

        derived_queries = {}
        for key, requested in tuple(result["queries"].items()):
            horizon, query_repeat, selected_metric = key
            if selected_metric != metric:
                continue
            del result["queries"][key]
            if query_repeat == horizon:
                continue

            current = result["candidates"].get(
                (
                    horizon,
                    query_repeat,
                    horizon,
                    candidate_role,
                    "cell_accuracy",
                )
            )
            agreement = result["candidates"].get(
                (
                    horizon,
                    query_repeat,
                    horizon,
                    collision_role,
                    "cell_agreement",
                )
            )
            if current is None or agreement is None:
                continue
            steps = {
                measurement["step"]
                for measurement in (requested, current, agreement)
            }
            if len(steps) != 1:
                continue
            value = _changed_cell_accuracy(
                requested["value"],
                current["value"],
                agreement["value"],
            )
            if value is not None:
                derived_queries[key] = {
                    "step": requested["step"],
                    "value": value,
                }
        result["queries"].update(derived_queries)

        result["overall"].pop(metric, None)
        for key in tuple(result["horizons"]):
            if key[1] == metric:
                del result["horizons"][key]

        by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for (horizon, _query_repeat, selected_metric), measurement in (
            derived_queries.items()
        ):
            if selected_metric == metric:
                by_horizon[horizon].append(measurement)

        for horizon, measurements in by_horizon.items():
            steps = {
                measurement["step"]
                for measurement in measurements
                if measurement["step"] is not None
            }
            result["horizons"][(horizon, metric)] = {
                "step": max(steps) if steps else None,
                "value": sum(item["value"] for item in measurements)
                / len(measurements),
            }

        horizon_measurements = [
            measurement
            for (horizon, selected_metric), measurement in result[
                "horizons"
            ].items()
            if selected_metric == metric
        ]
        if horizon_measurements:
            steps = {
                measurement["step"]
                for measurement in horizon_measurements
                if measurement["step"] is not None
            }
            result["overall"][metric] = {
                "step": max(steps) if steps else None,
                "value": sum(
                    item["value"] for item in horizon_measurements
                )
                / len(horizon_measurements),
            }


def collect_delayed_run(
    run: Run,
    args: argparse.Namespace,
    specs: Mapping[str, DelayedMetricSpec],
) -> dict[str, Any]:
    """Stream one JSONL file and retain only the selected DCA measurements."""

    result: dict[str, Any] = {
        "overall": {},
        "horizons": {},
        "queries": {},
        "in_distribution": {},
        "validation": {},
        "candidates": {},
    }
    overall_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    horizon_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    query_lookup: dict[
        tuple[str, str], list[tuple[str, DelayedMetricSpec]]
    ] = defaultdict(list)
    for name, spec in specs.items():
        overall_lookup[(spec.overall_role, spec.overall_metric)].append(name)
        horizon_lookup[(spec.horizon_role, spec.horizon_metric)].append(name)
        query_lookup[(spec.query_role, spec.query_metric)].append((name, spec))
    id_lookup = {
        spec.id_metric: name
        for name, spec in specs.items()
        if spec.id_metric is not None
    }

    for record in iter_metrics(run.path / METRICS_FILENAME):
        if args.length is not None and record.get("length") != args.length:
            continue
        role = record.get("evaluation_role")
        metric = record.get("metric")
        split = record.get("data_split")

        if split == "validation" and record.get("checkpoint_type") is None:
            selected_metrics = overall_lookup.get((role, metric), ())
            if selected_metrics and record.get("step") is not None:
                validation = result["validation"].setdefault(
                    int(record["step"]), {}
                )
                for selected in selected_metrics:
                    validation[selected] = float(record["value"])

        if split != args.split or not _checkpoint_matches(
            record, args.checkpoint
        ):
            continue

        selected_metrics = overall_lookup.get((role, metric), ())
        if selected_metrics:
            for selected in selected_metrics:
                _put_latest(result["overall"], selected, record)
            continue

        selected_metrics = horizon_lookup.get((role, metric), ())
        if selected_metrics:
            horizon = record.get("num_repeats")
            if horizon is not None:
                for selected in selected_metrics:
                    _put_latest(
                        result["horizons"],
                        (int(horizon), selected),
                        record,
                    )
            continue

        query_matches = query_lookup.get((role, metric), ())
        if query_matches:
            horizon = record.get("num_repeats")
            query_repeat = record.get("repeat_from")
            candidate = record.get("repeat_to")
            if horizon is None or query_repeat is None:
                continue
            for selected, spec in query_matches:
                if (
                    spec.query_requires_requested_candidate
                    and candidate != query_repeat
                ):
                    continue
                _put_latest(
                    result["queries"],
                    (int(horizon), int(query_repeat), selected),
                    record,
                )
            # The requested-candidate cosine record is also useful in the
            # complete candidate export, so it is intentionally not skipped.

        if role == "in_distribution" and metric in id_lookup:
            horizon = record.get("num_repeats")
            if horizon is None:
                horizon = record.get("ca_steps")
            if horizon is not None:
                _put_latest(
                    result["in_distribution"],
                    (int(horizon), id_lookup[metric]),
                    record,
                )

        if role in DELAYED_CANDIDATE_ROLES and metric in DELAYED_CANDIDATE_METRICS:
            horizon = record.get("num_repeats")
            query_repeat = record.get("repeat_from")
            candidate = record.get("repeat_to")
            if None not in (horizon, query_repeat, candidate):
                _put_latest(
                    result["candidates"],
                    (
                        int(horizon),
                        int(query_repeat),
                        int(candidate),
                        str(role),
                        str(metric),
                    ),
                    record,
                )
    _derive_delayed_changed_cell_metrics(result, specs)
    return result


def _delayed_configuration_metadata(
    configuration_id_value: str,
    members: Sequence[Run],
    protocol_ids: Mapping[str, str],
) -> dict[str, Any]:
    representative = members[0]
    manifest = representative.manifest
    protocol = delayed_protocol_configuration(manifest)
    protocol_key = json.dumps(protocol, sort_keys=True, default=str)
    policy = forward_policy_metadata(manifest)
    forget_gate, control_input = lstm_ut_configuration_fields(manifest)
    return {
        "configuration_id": configuration_id_value,
        "configuration": configuration_label(manifest),
        "model": dotted_get(manifest, "model"),
        "lstm_forget_gate": forget_gate,
        "lstm_control_input": control_input,
        "controller_application": manifest.get("model", {}).get(
            "ca_controller_application"
        ),
        "architecture": architecture_signature(manifest),
        "training_cache_policy": policy["repeat_cache_policy"],
        "training_cache_window": policy["repeat_cache_window"],
        "protocol_id": protocol_ids[protocol_key],
        "run_ids": [run.run_id for run in members],
        "model_seeds": [dotted_get(run.manifest, "seed") for run in members],
        "data_seeds": [
            dotted_get(run.manifest, "data_seed") for run in members
        ],
    }


def _delayed_summary_row(
    metadata: Mapping[str, Any],
    metric: str,
    observations: Sequence[tuple[float, Run, int | None]],
    **dimensions: Any,
) -> dict[str, Any]:
    values = [value for value, _run, _step in observations]
    contributing_runs = sorted(
        {run.run_id: run for _value, run, _step in observations}.values(),
        key=lambda run: run.run_id,
    )
    steps = sorted(
        {step for _value, _run, step in observations if step is not None}
    )
    return {
        **metadata,
        **dimensions,
        "metric": metric,
        "metric_label": DELAYED_METRICS[metric].label,
        "contributing_run_ids": [run.run_id for run in contributing_runs],
        "checkpoint_steps": steps,
        **summarize(values),
    }


def aggregate_delayed_runs(
    runs: Sequence[Run],
    run_data: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    specs: Mapping[str, DelayedMetricSpec],
) -> dict[str, Any]:
    average_over = _field_list(args.average_over, ("seed", "data_seed"))
    groups = _delayed_groups(runs, average_over=average_over)
    protocol_values = {
        json.dumps(
            delayed_protocol_configuration(run.manifest),
            sort_keys=True,
            default=str,
        )
        for run in runs
    }
    protocol_ids = {
        value: f"P{index}"
        for index, value in enumerate(sorted(protocol_values), start=1)
    }
    configurations = []
    metadata_by_group = {}
    for configuration_order, (group_id, members) in enumerate(groups.items()):
        metadata = _delayed_configuration_metadata(
            group_id, members, protocol_ids
        )
        metadata["configuration_order"] = configuration_order
        metadata_by_group[group_id] = metadata
        configurations.append(
            {
                **metadata,
                "protocol": delayed_protocol_configuration(
                    members[0].manifest
                ),
            }
        )

    rows_by_scope: dict[str, list[dict[str, Any]]] = {
        "overall": [],
        "horizons": [],
        "queries": [],
        "in_distribution": [],
        "validation": [],
        "candidates": [],
    }
    for group_id, members in groups.items():
        metadata = metadata_by_group[group_id]
        for metric in specs:
            observations = []
            for run in members:
                measurement = run_data[run.run_id]["overall"].get(metric)
                if measurement is not None:
                    observations.append(
                        (measurement["value"], run, measurement["step"])
                    )
            if observations:
                rows_by_scope["overall"].append(
                    _delayed_summary_row(metadata, metric, observations)
                )

        horizon_keys = sorted(
            {
                key
                for run in members
                for key in run_data[run.run_id]["horizons"]
            }
        )
        for horizon, metric in horizon_keys:
            observations = []
            for run in members:
                measurement = run_data[run.run_id]["horizons"].get(
                    (horizon, metric)
                )
                if measurement is not None:
                    observations.append(
                        (measurement["value"], run, measurement["step"])
                    )
            if observations:
                rows_by_scope["horizons"].append(
                    _delayed_summary_row(
                        metadata, metric, observations, horizon=horizon
                    )
                )

        query_keys = sorted(
            {
                key
                for run in members
                for key in run_data[run.run_id]["queries"]
            }
        )
        for horizon, query_repeat, metric in query_keys:
            observations = []
            for run in members:
                measurement = run_data[run.run_id]["queries"].get(
                    (horizon, query_repeat, metric)
                )
                if measurement is not None:
                    observations.append(
                        (measurement["value"], run, measurement["step"])
                    )
            if observations:
                rows_by_scope["queries"].append(
                    _delayed_summary_row(
                        metadata,
                        metric,
                        observations,
                        horizon=horizon,
                        query_repeat=query_repeat,
                        recall_age=horizon - query_repeat,
                        is_no_op=query_repeat == horizon,
                    )
                )

        id_keys = sorted(
            {
                key
                for run in members
                for key in run_data[run.run_id]["in_distribution"]
            }
        )
        for horizon, metric in id_keys:
            observations = []
            for run in members:
                measurement = run_data[run.run_id]["in_distribution"].get(
                    (horizon, metric)
                )
                if measurement is not None:
                    observations.append(
                        (measurement["value"], run, measurement["step"])
                    )
            if observations:
                rows_by_scope["in_distribution"].append(
                    _delayed_summary_row(
                        metadata, metric, observations, horizon=horizon
                    )
                )

        validation_keys = sorted(
            {
                (step, metric)
                for run in members
                for step, values in run_data[run.run_id]["validation"].items()
                for metric in values
            }
        )
        for step, metric in validation_keys:
            observations = [
                (
                    run_data[run.run_id]["validation"][step][metric],
                    run,
                    step,
                )
                for run in members
                if metric
                in run_data[run.run_id]["validation"].get(step, {})
            ]
            rows_by_scope["validation"].append(
                _delayed_summary_row(
                    metadata, metric, observations, training_step=step
                )
            )

        candidate_keys = sorted(
            {
                key
                for run in members
                for key in run_data[run.run_id]["candidates"]
            }
        )
        for key in candidate_keys:
            horizon, query_repeat, candidate_repeat, role, raw_metric = key
            observations = []
            for run in members:
                measurement = run_data[run.run_id]["candidates"].get(key)
                if measurement is not None:
                    observations.append(
                        (measurement["value"], run, measurement["step"])
                    )
            if observations:
                values = [value for value, _run, _step in observations]
                rows_by_scope["candidates"].append(
                    {
                        **metadata,
                        "horizon": horizon,
                        "query_repeat": query_repeat,
                        "recall_age": horizon - query_repeat,
                        "candidate_repeat": candidate_repeat,
                        "evaluation_role": role,
                        "metric": raw_metric,
                        "contributing_run_ids": [
                            run.run_id for _value, run, _step in observations
                        ],
                        **summarize(values),
                    }
                )
    return {
        "configurations": configurations,
        "protocol_count": len(protocol_ids),
        "seed_coverage_count": len(
            {
                tuple(
                    sorted(
                        zip(
                            configuration["model_seeds"],
                            configuration["data_seeds"],
                        ),
                        key=lambda pair: (str(pair[0]), str(pair[1])),
                    )
                )
                for configuration in configurations
            }
        ),
        **rows_by_scope,
    }


def _delayed_table_by_metric(
    rows: Sequence[Mapping[str, Any]],
    specs: Mapping[str, DelayedMetricSpec],
    dimension_fields: Sequence[str],
) -> tuple[tuple[str, ...], list[tuple[Any, ...]]]:
    keyed = {
        (
            row["configuration_id"],
            *(row.get(field) for field in dimension_fields),
            row["metric"],
        ): row
        for row in rows
    }
    configuration_order = {
        row["configuration_id"]: row["configuration_order"] for row in rows
    }
    dimensions = sorted(
        {
            (
                row["configuration_id"],
                *(row.get(field) for field in dimension_fields),
            )
            for row in rows
        },
        key=lambda value: (
            configuration_order[value[0]],
            *(
                (0, int(part))
                if isinstance(part, (bool, int))
                else (1, str(part))
                for part in value[1:]
            ),
        ),
    )
    rendered = []
    for dimension in dimensions:
        values = []
        for metric in specs:
            row = keyed.get((*dimension, metric))
            values.append("—" if row is None else _format_summary(row))
        rendered.append((*dimension, *values))
    headers = (
        "config id",
        *dimension_fields,
        *(spec.label for spec in specs.values()),
    )
    return headers, rendered


def _print_delayed_comparison(
    report: Mapping[str, Any], specs: Mapping[str, DelayedMetricSpec]
) -> None:
    configuration_rows = []
    for config in report["configurations"]:
        configuration_rows.append(
            (
                config["configuration_id"],
                config["model"],
                config["lstm_forget_gate"] or "—",
                config["lstm_control_input"] or "—",
                config["controller_application"],
                config["architecture"],
                config["training_cache_policy"],
                config["protocol_id"],
                ",".join(config["run_ids"]),
                ",".join(map(str, config["model_seeds"])),
                ",".join(map(str, config["data_seeds"])),
            )
        )
    print("Delayed CA configurations")
    print(
        _terminal_table(
            (
                "config id",
                "model",
                "forget gate",
                "control input",
                "controller",
                "architecture",
                "cache",
                "protocol",
                "runs",
                "seeds",
                "data seeds",
            ),
            configuration_rows,
        )
    )
    if report["protocol_count"] > 1:
        print(
            "WARNING: selected configurations use different delayed-CA "
            "protocols; inspect delayed_comparison.json before interpreting "
            "performance differences."
        )
    if report["seed_coverage_count"] > 1:
        print(
            "WARNING: configurations do not have identical model/data seed "
            "coverage; aggregate differences are not fully paired."
        )

    sections = (
        ("Overall nontrivial delayed recall", "overall", ()),
        ("Delayed recall by horizon", "horizons", ("horizon",)),
        (
            "Delayed recall by query",
            "queries",
            ("horizon", "query_repeat", "recall_age", "is_no_op"),
        ),
        (
            "In-distribution CA accuracy from the same checkpoint",
            "in_distribution",
            ("horizon",),
        ),
    )
    for title, key, dimensions in sections:
        rows = report[key]
        if not rows:
            continue
        headers, rendered = _delayed_table_by_metric(rows, specs, dimensions)
        print(f"\n{title}")
        print(_terminal_table(headers, rendered))


def _write_delayed_report(
    report_dir: Path,
    report: Mapping[str, Any],
    args: argparse.Namespace,
    runs: Sequence[Run],
) -> None:
    filenames = {
        "overall": "delayed_summary.csv",
        "queries": "delayed_queries.csv",
        "candidates": "delayed_candidates.csv",
        "validation": "delayed_validation.csv",
    }
    for key, filename in filenames.items():
        _write_csv(report_dir / filename, report[key])
    horizon_rows = [
        {"scope": "delayed_nontrivial", **row}
        for row in report["horizons"]
    ] + [
        {"scope": "in_distribution_same_checkpoint", **row}
        for row in report["in_distribution"]
    ]
    _write_csv(report_dir / "delayed_horizons.csv", horizon_rows)
    write_json(report_dir / "delayed_comparison.json", report)
    _write_analysis_manifest(report_dir, args, runs)


def _build_delayed_report(
    args: argparse.Namespace, runs: Sequence[Run]
) -> tuple[
    list[Run],
    dict[str, DelayedMetricSpec],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    selected_runs = _select_delayed_runs(runs, args.run_id)
    if not selected_runs:
        raise ValueError("No run manifests matched the delayed comparison.")
    specs = _selected_delayed_specs(args)
    if args.length <= 0:
        raise ValueError("--length must be positive.")
    run_data = {
        run.run_id: collect_delayed_run(run, args, specs)
        for run in selected_runs
    }
    if not any(data["overall"] for data in run_data.values()):
        raise ValueError(
            "No overall delayed-recall records matched split="
            f"{args.split}, checkpoint={args.checkpoint}, length={args.length}."
        )
    report = aggregate_delayed_runs(selected_runs, run_data, args, specs)
    if args.strict_match and (
        report["protocol_count"] > 1 or report["seed_coverage_count"] > 1
    ):
        raise ValueError(
            "--strict-match requires one delayed-CA protocol and identical "
            "model/data seed coverage; selected runs produced "
            f"{report['protocol_count']} protocol(s) and "
            f"{report['seed_coverage_count']} seed coverage set(s)."
        )
    return selected_runs, specs, run_data, report


def command_delayed_compare(
    args: argparse.Namespace, runs: Sequence[Run]
) -> int:
    selected_runs, specs, _run_data, report = _build_delayed_report(args, runs)
    _print_delayed_comparison(report, specs)
    if args.output_dir is not None:
        report_dir = _report_dir(args)
        _write_delayed_report(report_dir, report, args, selected_runs)
        print(f"\nWrote delayed comparison to {report_dir}")
    return 0


RECALL_REPEAT_COUNTS = (1, 2)
RECALL_REPEAT_METRICS = ("cell_accuracy", "internal_cell_accuracy")


def _training_recall_repeats(manifest: Mapping[str, Any]) -> int:
    """Return the explicitly recorded delayed-query training recall depth."""

    resolved = manifest.get("resolved_args", {})
    value = (
        resolved.get("ca_recall_repeats")
        if isinstance(resolved, Mapping)
        else None
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "Recall-repeat comparison requires every selected run manifest "
            "to record a positive integer resolved_args.ca_recall_repeats."
        )
    return value


def _recall_repeat_family_configuration(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the controlled configuration with recall depth as intervention."""

    configuration = _delayed_comparison_configuration(
        manifest, average_over=("seed", "data_seed")
    )
    resolved = dict(configuration["resolved_args"])
    resolved.pop("ca_recall_repeats", None)
    return {**configuration, "resolved_args": resolved}


def _paired_recall_runs(
    runs: Sequence[Run], requested_run_ids: Sequence[str] | None
) -> tuple[list[Run], dict[int, dict[tuple[Any, Any], Run]], str]:
    selected = _select_delayed_runs(runs, requested_run_ids)
    if not selected:
        raise ValueError("No run manifests matched the recall-repeat comparison.")

    eligible = []
    for run in selected:
        recall_repeats = _training_recall_repeats(run.manifest)
        if recall_repeats in RECALL_REPEAT_COUNTS:
            eligible.append((run, recall_repeats))
    if not eligible:
        raise ValueError(
            "No selected runs use ca_recall_repeats=1 or ca_recall_repeats=2."
        )

    families: dict[str, list[tuple[Run, int]]] = defaultdict(list)
    for run, recall_repeats in eligible:
        family = _recall_repeat_family_configuration(run.manifest)
        families[configuration_id(family)].append((run, recall_repeats))
    if len(families) != 1:
        details = ", ".join(
            f"{family_id} ({len(members)} runs)"
            for family_id, members in sorted(families.items())
        )
        raise ValueError(
            "Recall-repeat comparison requires exactly one scientific "
            "configuration after removing seeds and ca_recall_repeats; "
            f"found {len(families)}: {details}. Narrow --run-id or --where."
        )

    family_id, family_members = next(iter(families.items()))
    by_depth: dict[int, dict[tuple[Any, Any], Run]] = {
        depth: {} for depth in RECALL_REPEAT_COUNTS
    }
    for run, recall_repeats in family_members:
        seed_key = (
            dotted_get(run.manifest, "seed"),
            dotted_get(run.manifest, "data_seed"),
        )
        if None in seed_key:
            raise ValueError(
                f"Run {run.run_id} lacks a model seed or data seed and cannot "
                "participate in a paired comparison."
            )
        previous = by_depth[recall_repeats].get(seed_key)
        if previous is not None:
            raise ValueError(
                "Recall-repeat comparison found duplicate runs for "
                f"ca_recall_repeats={recall_repeats}, seed={seed_key[0]}, "
                f"data_seed={seed_key[1]}: {previous.run_id}, {run.run_id}."
            )
        by_depth[recall_repeats][seed_key] = run

    missing_depths = [
        depth for depth in RECALL_REPEAT_COUNTS if not by_depth[depth]
    ]
    if missing_depths:
        raise ValueError(
            "Recall-repeat comparison requires both ca_recall_repeats=1 and "
            "ca_recall_repeats=2; missing "
            + ", ".join(map(str, missing_depths))
            + "."
        )
    one_seeds = set(by_depth[1])
    two_seeds = set(by_depth[2])
    if one_seeds != two_seeds:
        only_one = sorted(one_seeds - two_seeds, key=str)
        only_two = sorted(two_seeds - one_seeds, key=str)
        raise ValueError(
            "Recall-repeat comparison requires identical paired (model_seed, "
            "data_seed) coverage. Only one-pass: "
            f"{only_one or 'none'}; only two-pass: {only_two or 'none'}."
        )
    for seed_key in one_seeds:
        one_shuffle_seed = dotted_get(
            by_depth[1][seed_key].manifest, "resolved_args.ca_shuffle_seed"
        )
        two_shuffle_seed = dotted_get(
            by_depth[2][seed_key].manifest, "resolved_args.ca_shuffle_seed"
        )
        if one_shuffle_seed != two_shuffle_seed:
            raise ValueError(
                "Paired recall-repeat runs must use the same CA shuffle seed; "
                f"seed={seed_key[0]}, data_seed={seed_key[1]} has "
                f"one-pass={one_shuffle_seed} and two-pass={two_shuffle_seed}."
            )
    paired_runs = [
        by_depth[depth][seed_key]
        for seed_key in sorted(one_seeds, key=str)
        for depth in RECALL_REPEAT_COUNTS
    ]
    return paired_runs, by_depth, family_id


def _recall_repeat_measurement(
    run_data: Mapping[str, Mapping[str, Any]],
    run: Run,
    *,
    horizon: int,
    query_repeat: int | None,
    metric: str,
) -> Mapping[str, Any]:
    if query_repeat is None:
        measurement = run_data[run.run_id]["horizons"].get((horizon, metric))
        description = f"horizon {horizon} nontrivial overall {metric}"
    else:
        measurement = run_data[run.run_id]["queries"].get(
            (horizon, query_repeat, metric)
        )
        description = (
            f"horizon {horizon}, query repeat {query_repeat}, {metric}"
        )
    if measurement is None:
        raise ValueError(
            f"Run {run.run_id} has no matching {description} record. The "
            "comparison requires complete ground-truth and internal recall "
            "coverage for every requested age."
        )
    return measurement


def _recall_repeat_row(
    run_data: Mapping[str, Mapping[str, Any]],
    by_depth: Mapping[int, Mapping[tuple[Any, Any], Run]],
    *,
    horizon: int,
    recall_age: int | None,
) -> dict[str, Any]:
    query_repeat = None if recall_age is None else horizon - recall_age
    seed_keys = sorted(by_depth[1], key=str)
    row: dict[str, Any] = {
        "scope": "overall_nontrivial" if recall_age is None else "recall_age",
        "horizon": horizon,
        "recall_age": recall_age,
        "query_repeat": query_repeat,
        "is_no_op": recall_age == 0 if recall_age is not None else False,
        "paired_seed_count": len(seed_keys),
    }
    paired_values: list[dict[str, Any]] = []
    for metric in RECALL_REPEAT_METRICS:
        one_values = []
        two_values = []
        deltas = []
        for seed_key in seed_keys:
            one_run = by_depth[1][seed_key]
            two_run = by_depth[2][seed_key]
            one = _recall_repeat_measurement(
                run_data,
                one_run,
                horizon=horizon,
                query_repeat=query_repeat,
                metric=metric,
            )
            two = _recall_repeat_measurement(
                run_data,
                two_run,
                horizon=horizon,
                query_repeat=query_repeat,
                metric=metric,
            )
            one_value = float(one["value"])
            two_value = float(two["value"])
            one_values.append(one_value)
            two_values.append(two_value)
            deltas.append(two_value - one_value)
            paired_values.append(
                {
                    "metric": metric,
                    "model_seed": seed_key[0],
                    "data_seed": seed_key[1],
                    "one_repeat_run_id": one_run.run_id,
                    "two_repeat_run_id": two_run.run_id,
                    "one_repeat_checkpoint_step": one["step"],
                    "two_repeat_checkpoint_step": two["step"],
                    "one_repeat_value": one_value,
                    "two_repeat_value": two_value,
                    "delta_two_minus_one": two_value - one_value,
                }
            )
        prefix = "ground_truth" if metric == "cell_accuracy" else "internal"
        for name, values in (
            ("one_repeat", one_values),
            ("two_repeat", two_values),
            ("delta_two_minus_one", deltas),
        ):
            summary = summarize(values)
            for statistic in ("mean", "std", "min", "max"):
                row[f"{prefix}_{name}_{statistic}"] = summary[statistic]
    row["paired_values"] = paired_values
    return row


def build_recall_repeat_report(
    args: argparse.Namespace, runs: Sequence[Run]
) -> tuple[list[Run], dict[str, Any]]:
    if args.horizon < 2:
        raise ValueError(
            "--horizon must be at least 2 so the nontrivial overall recall "
            "summary contains at least one positive recall age."
        )
    if args.length <= 0:
        raise ValueError("--length must be positive.")
    selected_runs, by_depth, family_id = _paired_recall_runs(runs, args.run_id)
    model_names = {
        dotted_get(run.manifest, "resolved_args.model")
        for run in selected_runs
    }
    if None in model_names or len(model_names) != 1:
        raise ValueError(
            "Recall-repeat comparison requires exactly one model; found "
            f"{sorted(map(str, model_names))}."
        )
    model_name = next(iter(model_names))
    specs = {name: DELAYED_METRICS[name] for name in RECALL_REPEAT_METRICS}
    run_data = {
        run.run_id: collect_delayed_run(run, args, specs)
        for run in selected_runs
    }
    rows = [
        _recall_repeat_row(
            run_data, by_depth, horizon=args.horizon, recall_age=recall_age
        )
        for recall_age in range(args.horizon)
    ]
    rows.append(
        _recall_repeat_row(
            run_data, by_depth, horizon=args.horizon, recall_age=None
        )
    )
    seed_keys = sorted(by_depth[1], key=str)
    one_checkpoint_steps = sorted(
        {
            paired["one_repeat_checkpoint_step"]
            for row in rows
            for paired in row["paired_values"]
            if paired["one_repeat_checkpoint_step"] is not None
        }
    )
    two_checkpoint_steps = sorted(
        {
            paired["two_repeat_checkpoint_step"]
            for row in rows
            for paired in row["paired_values"]
            if paired["two_repeat_checkpoint_step"] is not None
        }
    )
    return selected_runs, {
        "model" : model_name,
        "horizon": args.horizon,
        "length": args.length,
        "data_split": args.split,
        "checkpoint_type": args.checkpoint,
        "configuration_family_id": family_id,
        "recall_repeat_counts": list(RECALL_REPEAT_COUNTS),
        "metrics": {
            "ground_truth": "cell_accuracy against the requested CA state",
            "internal": (
                "cell_accuracy against the state decoded at the requested "
                "evolution repeat"
            ),
        },
        "delta_definition": "two_repeat_value - one_repeat_value",
        "overall_definition": (
            "existing horizon-level nontrivial delayed-recall macro; "
            "recall age zero excluded"
        ),
        "paired_seeds": [
            {"model_seed": key[0], "data_seed": key[1]} for key in seed_keys
        ],
        "one_repeat_run_ids": [by_depth[1][key].run_id for key in seed_keys],
        "two_repeat_run_ids": [by_depth[2][key].run_id for key in seed_keys],
        "one_repeat_checkpoint_steps": one_checkpoint_steps,
        "two_repeat_checkpoint_steps": two_checkpoint_steps,
        "rows": rows,
    }


def _print_recall_repeat_report(report: Mapping[str, Any]) -> None:
    model_name = report["model"]
    model_label = {
        "dca_but": "BUT",
        "dca_cotf_cache": "CoTFormer",
    }.get(model_name, model_name)

    rendered = []
    for row in report["rows"]:
        label = (
            "overall*"
            if row["scope"] == "overall_nontrivial"
            else str(row["recall_age"])
        )
        rendered.append(
            (
                label,
                "—" if row["query_repeat"] is None else row["query_repeat"],
                f"{row['ground_truth_one_repeat_mean']:.6f}",
                f"{row['ground_truth_two_repeat_mean']:.6f}",
                f"{row['ground_truth_delta_two_minus_one_mean']:+.6f}",
                f"{row['internal_one_repeat_mean']:.6f}",
                f"{row['internal_two_repeat_mean']:.6f}",
                f"{row['internal_delta_two_minus_one_mean']:+.6f}",
                row["paired_seed_count"],
            )
        )
    print(
        f"Recall-repeat comparison for {model_label} ({model_name}): "
        f"1 recall pass vs 2 recall passes at horizon {report['horizon']} "
        f"({report['data_split']}/{report['checkpoint_type']})"
    )
    print(
        "Checkpoint steps: "
        f"one-pass={report['one_repeat_checkpoint_steps'] or 'unknown'}, "
        f"two-pass={report['two_repeat_checkpoint_steps'] or 'unknown'}"
    )
    print(
        _terminal_table(
            (
                "recall age",
                "query repeat",
                "GT 1-pass mean",
                "GT 2-pass mean",
                "GT mean delta",
                "internal 1-pass mean",
                "internal 2-pass mean",
                "internal mean delta",
                "paired seeds",
            ),
            rendered,
        )
    )
    print(
        "\nDeltas are two-pass minus one-pass. Internal recall is agreement "
        "with the state decoded at the requested evolution repeat, not "
        "ground-truth accuracy.\n* overall excludes recall age zero."
    )


def command_recall_repeat_compare(
    args: argparse.Namespace, runs: Sequence[Run]
) -> int:
    selected_runs, report = build_recall_repeat_report(args, runs)
    _print_recall_repeat_report(report)
    if args.output_dir is not None:
        report_dir = _report_dir(args)
        csv_rows = [
            {key: value for key, value in row.items() if key != "paired_values"}
            for row in report["rows"]
        ]
        paired_rows = [
            {
                "scope": row["scope"],
                "horizon": row["horizon"],
                "recall_age": row["recall_age"],
                "query_repeat": row["query_repeat"],
                **paired,
            }
            for row in report["rows"]
            for paired in row["paired_values"]
        ]
        _write_csv(report_dir / "recall_repeat_comparison.csv", csv_rows)
        _write_csv(report_dir / "recall_repeat_paired_values.csv", paired_rows)
        write_json(report_dir / "recall_repeat_comparison.json", report)
        _write_analysis_manifest(report_dir, args, selected_runs)
        print(f"\nWrote recall-repeat comparison to {report_dir}")
    return 0


def command_delayed_leaderboard(
    args: argparse.Namespace, runs: Sequence[Run]
) -> int:
    selected_runs, specs, run_data, report = _build_delayed_report(args, runs)
    primary_metric = next(iter(specs))
    default_direction = specs[primary_metric].direction
    direction = (
        default_direction if args.direction == "auto" else args.direction
    )
    overall_by_run: dict[str, dict[str, tuple[float, int | None]]] = defaultdict(dict)
    for run in selected_runs:
        data = run_data[run.run_id]
        for metric, measurement in data["overall"].items():
            overall_by_run[run.run_id][metric] = (
                measurement["value"],
                measurement["step"],
            )
    ranked_runs = [
        run for run in selected_runs if primary_metric in overall_by_run[run.run_id]
    ]
    ranked_runs.sort(
        key=lambda run: (
            (
                -overall_by_run[run.run_id][primary_metric][0]
                if direction == "max"
                else overall_by_run[run.run_id][primary_metric][0]
            ),
            run.run_id,
        )
    )
    rendered = []
    export_rows = []
    for rank, run in enumerate(ranked_runs, start=1):
        manifest = run.manifest
        measurements = overall_by_run[run.run_id]
        forget_gate, control_input = lstm_ut_configuration_fields(manifest)
        row = {
            "rank": rank,
            "run_id": run.run_id,
            "model": dotted_get(manifest, "model"),
            "lstm_forget_gate": forget_gate,
            "lstm_control_input": control_input,
            "controller_application": manifest.get("model", {}).get(
                "ca_controller_application"
            ),
            "architecture": architecture_signature(manifest),
            "training_cache": training_cache_label(manifest),
            "model_seed": dotted_get(manifest, "seed"),
            "data_seed": dotted_get(manifest, "data_seed"),
            "checkpoint_step": measurements[primary_metric][1],
        }
        values = []
        for metric in specs:
            measurement = measurements.get(metric)
            row[metric] = None if measurement is None else measurement[0]
            values.append(
                "—" if measurement is None else f"{measurement[0]:.6f}"
            )
        export_rows.append(row)
        rendered.append(
            (
                rank,
                run.run_id,
                row["model"],
                row["lstm_forget_gate"] or "—",
                row["lstm_control_input"] or "—",
                row["controller_application"],
                row["architecture"],
                row["training_cache"],
                row["model_seed"],
                row["data_seed"],
                row["checkpoint_step"],
                *values,
            )
        )
    print(
        _terminal_table(
            (
                "rank",
                "run",
                "model",
                "forget gate",
                "control input",
                "controller",
                "architecture",
                "cache",
                "seed",
                "data",
                "checkpoint step",
                *(spec.label for spec in specs.values()),
            ),
            rendered,
        )
    )
    print(
        f"\nRanked {direction} by {primary_metric} from "
        f"{args.split}/{args.checkpoint}; additional selected metrics are "
        "descriptive and do not affect rank."
    )
    if args.output_dir is not None:
        report_dir = _report_dir(args)
        _write_csv(report_dir / "delayed_leaderboard.csv", export_rows)
        write_json(report_dir / "delayed_leaderboard.json", export_rows)
        write_json(report_dir / "delayed_comparison.json", report)
        _write_analysis_manifest(report_dir, args, selected_runs)
        print(f"\nWrote delayed leaderboard to {report_dir}")
    return 0


def _load_plotting(report_dir: Path):
    mpl_config = report_dir / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _seed_series(
    runs: Sequence[Run],
    args: argparse.Namespace,
    x_getter,
    line_getter,
    record_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[Any, list[float]]]]:
    average_over = _field_list(args.average_over, ("seed", "data_seed"))
    groups = group_runs(runs, average_over=average_over)
    run_to_group = {
        run.run_id: group_id
        for group_id, members in groups.items()
        for run in members
    }
    values: dict[tuple[str, str], dict[Any, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    raw: dict[tuple[str, str, str], dict[Any, float]] = defaultdict(dict)
    for run in runs:
        records = (
            run.metrics
            if run.metrics
            else iter_metrics(run.path / METRICS_FILENAME)
        )
        for record in iter_selected_records(
            records, args, keep_all_steps=True
        ):
            if record_filter is not None and not record_filter(record):
                continue
            x = x_getter(record)
            line = str(line_getter(record))
            raw[(run_to_group[run.run_id], line, run.run_id)][x] = float(
                record["value"]
            )
    for (group_id, line, _run_id), points in raw.items():
        for x, value in points.items():
            values[(group_id, line)][x].append(value)
    return {"groups": groups, "raw": raw}, values


def _plot_seed_curves(
    plt,
    context: Mapping[str, Any],
    aggregates: Mapping[tuple[str, str], Mapping[Any, Sequence[float]]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    categorical: bool,
):
    figure, axis = plt.subplots(figsize=(12, 6), layout="constrained")
    raw = context["raw"]
    bounded_accuracy = "accuracy" in ylabel or ylabel.endswith("_rate")
    category_positions = None
    if categorical:
        categories = sorted(
            {
                x
                for points in aggregates.values()
                for x in points
            },
            key=_category_sort_key,
        )
        category_positions = {
            category: index for index, category in enumerate(categories)
        }

    def plotted_x(ordered):
        if category_positions is None:
            return ordered
        return [category_positions[value] for value in ordered]

    model_names = {
        group_id: str(dotted_get(members[0].manifest, "model"))
        for group_id, members in context["groups"].items()
    }
    model_labels = {"dca_but": "BUT", "dca_cotf_cache": "CoTFormer"}
    model_colors = {"dca_but": "#0072B2", "dca_cotf_cache": "#D55E00"}
    for index, model in enumerate(sorted(set(model_names.values()))):
        if model not in model_colors:
            model_colors[model] = plt.get_cmap("tab10")(index % 10)
    line_styles = ("--", ":", "-.")
    series_names = {line for _group_id, line in aggregates}
    for (group_id, line), points in aggregates.items():
        ordered = sorted(
            points,
            key=_category_sort_key if categorical else lambda value: value,
        )
        summaries = [summarize(points[x]) for x in ordered]
        means = [summary["mean"] for summary in summaries]
        stds = [summary["std"] for summary in summaries]
        model = model_names[group_id]
        color = model_colors[model]
        label = model_labels.get(model, model)
        if list(model_names.values()).count(model) > 1:
            label += f" [{group_id[:6]}]"
        if len(series_names) > 1:
            label += f" · {line}"
        counts = [summary["n"] for summary in summaries]
        coverage = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
        x_coordinates = plotted_x(ordered)
        axis.plot(
            x_coordinates, means, color=color, marker="o", markersize=5,
            linewidth=2.8, label=f"{label} mean (n={coverage})", zorder=4,
        )
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        # Only the displayed band is bounded; exported means and sample SDs
        # retain their original values, including SDs extending past 0 or 1.
        if bounded_accuracy:
            lower = [max(0.0, value) for value in lower]
            upper = [min(1.0, value) for value in upper]
        axis.fill_between(
            x_coordinates, lower, upper, color=color, alpha=0.12, zorder=1,
        )
        members = {run.run_id: run for run in context["groups"][group_id]}
        seed_curves = sorted(
            (run_id, seed_points)
            for (raw_group, raw_line, run_id), seed_points in raw.items()
            if (raw_group, raw_line) == (group_id, line)
        )
        for index, (run_id, seed_points) in enumerate(seed_curves):
            seed_order = sorted(
                seed_points,
                key=_category_sort_key if categorical else lambda value: value,
            )
            short_id = run_id.split("__", 1)[0].replace("_", " ")
            seed = dotted_get(members[run_id].manifest, "seed")
            axis.plot(
                plotted_x(seed_order), [seed_points[x] for x in seed_order],
                color=color, linestyle=line_styles[index % len(line_styles)],
                alpha=0.65, linewidth=1.2, zorder=3,
                label=f"{label} · {short_id} · seed {seed}",
            )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel.replace("_", " ").capitalize())
    band_note = "Shading: ±1 sample SD across runs"
    if bounded_accuracy:
        band_note += " (display clipped to 0–1)"
        axis.set_ylim(0.0, 1.02)
    if len(series_names) == 1:
        title += f" · {next(iter(series_names)).replace('_', ' ')}"
    axis.set_title(
        f"{title.replace('_', ' ')}\n"
        f"Solid: mean · dashed/dotted: individual runs\n{band_note}",
        fontsize=12,
    )
    axis.grid(alpha=0.25)
    if category_positions is not None:
        axis.set_xticks(
            list(category_positions.values()),
            list(category_positions.keys()),
        )
    if aggregates:
        axis.legend(
            fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1),
            borderaxespad=0, frameon=False, title="Models and runs",
        )
    return figure


def _plot_summary_rows(
    aggregates: Mapping[tuple[str, str], Mapping[Any, Sequence[float]]],
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for (group_id, line), points in aggregates.items():
        for x, values in points.items():
            rows.append(
                {
                    "configuration_id": group_id,
                    "configuration": configuration_label(
                        context["groups"][group_id][0].manifest
                    ),
                    "series": line,
                    "x": x,
                    **summarize(values),
                }
            )
    return rows


def command_plot_horizon(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    report_dir = _report_dir(args)
    plt = _load_plotting(report_dir)
    context, aggregates = _seed_series(
        runs,
        args,
        x_getter=lambda record: _pair_label(_pair(record)),
        line_getter=lambda _record: args.checkpoint or args.role or "metric",
    )
    if not aggregates:
        raise ValueError("No metric records matched the requested horizon plot.")
    figure = _plot_seed_curves(
        plt,
        context,
        aggregates,
        xlabel="Rule 30 steps : model repeats",
        ylabel=args.metric,
        title=f"{args.metric} across evaluated horizons",
        categorical=True,
    )
    figure.savefig(report_dir / "horizon.png", dpi=180)
    plt.close(figure)
    rows = _plot_summary_rows(aggregates, context)
    _write_csv(report_dir / "horizon.csv", rows)
    write_json(report_dir / "horizon.json", rows)
    _write_analysis_manifest(report_dir, args, runs)
    print(f"Wrote horizon plot and data to {report_dir}")
    return 0


def command_plot_training(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    report_dir = _report_dir(args)
    plt = _load_plotting(report_dir)
    context, aggregates = _seed_series(
        runs,
        args,
        x_getter=lambda record: int(record["step"]),
        line_getter=lambda record: _pair_label(_pair(record)),
    )
    if not aggregates:
        raise ValueError("No metric records matched the requested training plot.")
    figure = _plot_seed_curves(
        plt,
        context,
        aggregates,
        xlabel="training step",
        ylabel=args.metric,
        title=f"{args.metric} across training",
        categorical=False,
    )
    figure.savefig(report_dir / "training.png", dpi=180)
    plt.close(figure)
    rows = _plot_summary_rows(aggregates, context)
    _write_csv(report_dir / "training.csv", rows)
    write_json(report_dir / "training.json", rows)
    _write_analysis_manifest(report_dir, args, runs)
    print(f"Wrote training plot and data to {report_dir}")
    return 0


def command_plot_state(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    if len(runs) != 1:
        raise ValueError(
            "plot-state requires exactly one run; select it with "
            "--where run_id=RUN_ID."
        )

    run = runs[0]
    records = list(
        iter_selected_records(
            iter_metrics(run.path / METRICS_FILENAME),
            args,
            keep_all_steps=False,
        )
    )
    records = [
        record
        for record in records
        if record.get("repeat_from") is not None
        and record.get("repeat_to") is not None
    ]
    if not records:
        raise ValueError(
            "No hidden-state records matched the requested state heatmap."
        )

    checkpoint_steps = {int(record["step"]) for record in records}
    if len(checkpoint_steps) != 1:
        raise ValueError(
            "The selected hidden-state records span multiple checkpoint "
            "steps; pass --step STEP to select exactly one."
        )
    checkpoint_step = checkpoint_steps.pop()

    repeats = sorted(
        {
            int(record[field])
            for record in records
            for field in ("repeat_from", "repeat_to")
        }
    )
    repeat_positions = {
        repeat: position for position, repeat in enumerate(repeats)
    }
    matrix = [[math.nan for _ in repeats] for _ in repeats]
    seen_pairs = set()
    for record in records:
        repeat_from = int(record["repeat_from"])
        repeat_to = int(record["repeat_to"])
        pair = repeat_from, repeat_to
        if pair in seen_pairs:
            raise ValueError(
                "Multiple hidden-state values matched repeat pair "
                f"{repeat_from}→{repeat_to} at checkpoint step "
                f"{checkpoint_step}."
            )
        seen_pairs.add(pair)
        matrix[repeat_positions[repeat_from]][repeat_positions[repeat_to]] = (
            float(record["value"])
        )

    report_dir = _report_dir(args)
    plt = _load_plotting(report_dir)
    figure_size = max(7.0, min(10.5, 5.5 + len(repeats) / 12.0))
    figure, axis = plt.subplots(
        figsize=(figure_size + 0.8, figure_size), constrained_layout=True
    )
    metric_label = str(args.metric).replace("_", " ").title()
    image = axis.imshow(
        matrix,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
        vmin=-1.0 if args.metric == "cosine_similarity" else None,
        vmax=1.0 if args.metric == "cosine_similarity" else None,
    )

    tick_stride = max(1, math.ceil(len(repeats) / 11))
    tick_positions = list(range(0, len(repeats), tick_stride))
    if tick_positions[-1] != len(repeats) - 1:
        tick_positions.append(len(repeats) - 1)
    tick_labels = [repeats[position] for position in tick_positions]
    axis.set_xticks(tick_positions, tick_labels)
    axis.set_yticks(tick_positions, tick_labels)
    axis.set_xlabel("Repeat")
    axis.set_ylabel("Repeat")
    axis.set_title(
        f"Hidden-State {metric_label}\n"
        f"{run.run_id} · checkpoint step {checkpoint_step:,}"
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86, pad=0.03)
    colorbar.set_label(metric_label)

    if len(repeats) <= 12:
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if not math.isnan(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < 0.55 else "black",
                    )

    figure.savefig(report_dir / "hidden_state.png", dpi=180)
    plt.close(figure)
    rows = [
        {
            "run_id": run.run_id,
            "checkpoint_step": checkpoint_step,
            "repeat_from": int(record["repeat_from"]),
            "repeat_to": int(record["repeat_to"]),
            "metric": args.metric,
            "value": float(record["value"]),
        }
        for record in sorted(
            records,
            key=lambda record: (
                int(record["repeat_from"]),
                int(record["repeat_to"]),
            ),
        )
    ]
    _write_csv(report_dir / "hidden_state.csv", rows)
    write_json(report_dir / "hidden_state.json", rows)
    _write_analysis_manifest(report_dir, args, runs)
    print(
        "Wrote hidden-state heatmap and data for "
        f"{run.run_id} at checkpoint step {checkpoint_step} to {report_dir}"
    )
    return 0


def _add_discovery_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runs",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Root searched recursively for manifests (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help=(
            "Manifest filter; repeat for AND. Supports =, !=, >, >=, <, <=, "
            "and ~ substring. Aliases include model, seed, data_seed, "
            "optimizer, lr, weight_decay, grad_clip, and training_pairs."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when a malformed manifest is found.",
    )


def _add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metric", default=None)
    parser.add_argument(
        "--split",
        choices=["training", "validation", "final_test"],
        default=None,
    )
    parser.add_argument("--role", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint type. Use 'none' for training-time records without a "
            "checkpoint and 'any' to disable checkpoint filtering."
        ),
    )
    parser.add_argument(
        "--length",
        type=optional_int,
        default=64,
        help="Row length, or 'any' for metrics such as training exposure.",
    )
    parser.add_argument(
        "--step",
        default="latest",
        help="Training step integer, latest, all, or final.",
    )
    parser.add_argument(
        "--average-over",
        default="seed,data_seed",
        help=(
            "Comma-separated manifest seed fields removed from the grouping "
            "signature. Defaults to model and data seeds; validation/test "
            "seeds must still match."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional analysis artifact directory; plots choose a timestamped default.",
    )


def _add_delayed_arguments(parser: argparse.ArgumentParser) -> None:
    _add_discovery_arguments(parser)
    parser.add_argument(
        "--run-id",
        nargs="+",
        default=None,
        metavar="RUN_ID",
        help=(
            "Exact run IDs to compare, in the requested order. Existing "
            "--where filters are applied before this selection."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(DELAYED_METRICS),
        default=["cell_accuracy", "exact_sequence_accuracy"],
        metavar="METRIC",
        help=(
            "One or two delayed metrics. For a leaderboard, the first metric "
            "determines rank and the second is descriptive."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("validation", "final_test"),
        default="final_test",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_delayed_recall",
        help=(
            "Checkpoint type, normally best_delayed_recall. Use 'none' for "
            "validation records or 'any' to disable checkpoint filtering."
        ),
    )
    parser.add_argument(
        "--length",
        type=int,
        default=64,
        help="Evaluated row length.",
    )
    parser.add_argument(
        "--average-over",
        default="seed,data_seed",
        help="Comma-separated replicate fields removed during configuration grouping.",
    )
    parser.add_argument(
        "--strict-match",
        action="store_true",
        help="Reject rather than warn about mixed delayed-CA protocols.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for delayed comparison CSV/JSON artifacts.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze normalized Rule 30 experiment runs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List matching runs.")
    _add_discovery_arguments(list_parser)
    list_parser.set_defaults(handler=command_list)

    table_parser = subparsers.add_parser(
        "table", help="Create a dynamic steps:repeats comparison table."
    )
    _add_discovery_arguments(table_parser)
    _add_metric_arguments(table_parser)
    table_parser.add_argument(
        "--report-pairs",
        type=parse_pair_literal,
        nargs="+",
        default=None,
        metavar="STEPS:REPEATS",
        help=(
            "Only aggregate and display these steps:repeats pairs, in the "
            "requested order. Other normalized metric records are streamed "
            "past without being retained."
        ),
    )
    table_parser.set_defaults(
        handler=command_table,
        metric="cell_accuracy",
        split="final_test",
        role="internal_repeat_extrapolation",
        checkpoint="best_extrapolation_unconstrained",
    )

    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help=(
            "Rank individual runs by their best validation step on one "
            "selection pair."
        ),
    )
    _add_discovery_arguments(leaderboard_parser)
    leaderboard_parser.add_argument("--metric", default="cell_accuracy")
    leaderboard_parser.add_argument(
        "--split",
        choices=["training", "validation", "final_test"],
        default="validation",
    )
    leaderboard_parser.add_argument(
        "--role", default="extrapolation_validation"
    )
    leaderboard_parser.add_argument(
        "--checkpoint",
        default="none",
        help=(
            "Checkpoint type. Validation records normally use 'none'; use "
            "'any' to disable checkpoint filtering."
        ),
    )
    leaderboard_parser.add_argument(
        "--length",
        type=optional_int,
        default=64,
        help="Row length, or 'any'.",
    )
    leaderboard_parser.add_argument(
        "--select-pair",
        type=parse_pair_literal,
        default=None,
        help=(
            "Validation STEPS:REPEATS pair used to select each run's best "
            "step. Defaults to the common manifest extrapolation-best pair."
        ),
    )
    leaderboard_parser.add_argument(
        "--report-pairs",
        type=parse_pair_literal,
        nargs="+",
        default=None,
        metavar="STEPS:REPEATS",
        help=(
            "Pairs displayed at the selected step, in the requested order. "
            "This does not change checkpoint selection. By default every "
            "matching pair is displayed."
        ),
    )
    leaderboard_parser.add_argument(
        "--direction",
        choices=["max", "min"],
        default="max",
        help="Whether larger or smaller metric values rank first.",
    )
    leaderboard_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for leaderboard CSV/JSON and provenance.",
    )
    leaderboard_parser.set_defaults(handler=command_leaderboard)

    delayed_compare_parser = subparsers.add_parser(
        "delayed-compare",
        help=(
            "Compare delayed recall, retrieval, and same-checkpoint CA metrics."
        ),
    )
    _add_delayed_arguments(delayed_compare_parser)
    delayed_compare_parser.set_defaults(handler=command_delayed_compare)

    recall_repeat_parser = subparsers.add_parser(
        "recall-repeat-compare",
        help=(
            "Compare seed-paired one-pass and two-pass ground-truth and "
            "internal delayed-recall accuracy at one horizon."
        ),
    )
    _add_discovery_arguments(recall_repeat_parser)
    recall_repeat_parser.add_argument(
        "--run-id",
        nargs="+",
        default=None,
        metavar="RUN_ID",
        help="Exact one-pass and two-pass run IDs to compare.",
    )
    recall_repeat_parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Evolution horizon whose recall ages should be compared.",
    )
    recall_repeat_parser.add_argument(
        "--split",
        choices=("validation", "final_test"),
        default="final_test",
    )
    recall_repeat_parser.add_argument(
        "--checkpoint",
        default="best_delayed_recall",
        help=(
            "Checkpoint type, normally best_delayed_recall. Use 'none' for "
            "validation records or 'any' to disable checkpoint filtering."
        ),
    )
    recall_repeat_parser.add_argument(
        "--length", type=int, default=64, help="Evaluated row length."
    )
    recall_repeat_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional CSV/JSON comparison artifact directory.",
    )
    recall_repeat_parser.set_defaults(handler=command_recall_repeat_compare)

    delayed_leaderboard_parser = subparsers.add_parser(
        "delayed-leaderboard",
        help="Rank individual delayed-CA runs by a nontrivial recall metric.",
    )
    _add_delayed_arguments(delayed_leaderboard_parser)
    delayed_leaderboard_parser.add_argument(
        "--direction",
        choices=("auto", "max", "min"),
        default="auto",
        help=(
            "Ranking direction. Auto minimizes loss/rank/error metrics and "
            "maximizes accuracy/similarity metrics."
        ),
    )
    delayed_leaderboard_parser.set_defaults(handler=command_delayed_leaderboard)

    horizon_parser = subparsers.add_parser(
        "plot-horizon", help="Plot performance across steps:repeats pairs."
    )
    _add_discovery_arguments(horizon_parser)
    _add_metric_arguments(horizon_parser)
    horizon_parser.set_defaults(
        handler=command_plot_horizon,
        metric="cell_accuracy",
        split="final_test",
        role="internal_repeat_extrapolation",
        checkpoint="best_extrapolation_unconstrained",
        step="all",
    )

    training_parser = subparsers.add_parser(
        "plot-training", help="Plot pair accuracy across training evaluations."
    )
    _add_discovery_arguments(training_parser)
    _add_metric_arguments(training_parser)
    training_parser.set_defaults(
        handler=command_plot_training,
        metric="cell_accuracy",
        split="validation",
        role="extrapolation_validation",
        checkpoint=None,
        step="all",
    )

    state_parser = subparsers.add_parser(
        "plot-state",
        help="Plot a repeat-by-repeat hidden-state similarity heatmap.",
    )
    _add_discovery_arguments(state_parser)
    _add_metric_arguments(state_parser)
    state_parser.set_defaults(
        handler=command_plot_state,
        metric="cosine_similarity",
        split="final_test",
        role="hidden_state_similarity",
        checkpoint="best_extrapolation_unconstrained",
        step="latest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        filters = tuple(parse_filter(value) for value in args.where)
        runs = discover_runs(
            args.runs,
            filters=filters,
            strict=args.strict,
            load_metrics=False,
        )
        return int(args.handler(args, runs))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
