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
from typing import Any, Iterable, Mapping, Sequence

try:
    from .ca_reporting import (
        MANIFEST_FILENAME,
        METRICS_FILENAME,
        read_json,
        validate_manifest,
        write_json,
    )
except ImportError:
    from ca_reporting import (
        MANIFEST_FILENAME,
        METRICS_FILENAME,
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
    "weight_decay": "training.weight_decay",
    "grad_clip": "training.grad_clip",
    "training_pairs": "training.pairs",
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


def read_metrics(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        return ()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error
    return tuple(records)


def discover_runs(
    root: Path | str,
    *,
    filters: Sequence[tuple[str, str, Any]] = (),
    strict: bool = False,
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
                metrics=read_metrics(manifest_path.parent / METRICS_FILENAME),
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
    return {"resolved_args": resolved_args}


def configuration_id(configuration: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def configuration_label(manifest: Mapping[str, Any]) -> str:
    model = dotted_get(manifest, "model", "?")
    pairs = dotted_get(manifest, "training_pairs", "?")
    optimizer = dotted_get(manifest, "optimizer", "?")
    lr = dotted_get(manifest, "lr", "?")
    weight_decay = dotted_get(manifest, "weight_decay", "?")
    grad_clip = dotted_get(manifest, "grad_clip", "?")
    return (
        f"{model} pairs={pairs} opt={optimizer} lr={lr} "
        f"wd={weight_decay} clip={grad_clip}"
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
    records = [record for record in run.metrics if _record_matches(record, args)]
    if keep_all_steps or args.step in {None, "all"}:
        return records
    if args.step == "final":
        return [
            record for record in records if record.get("data_split") == "final_test"
        ]
    if args.step != "latest":
        requested = int(args.step)
        return [record for record in records if record.get("step") == requested]

    by_measurement: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in records:
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
    return list(by_measurement.values())


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
) -> tuple[dict[str, list[Run]], list[dict[str, Any]]]:
    average_over = _field_list(args.average_over, ("seed", "data_seed"))
    groups = group_runs(runs, average_over=average_over)
    run_to_group = {
        run.run_id: group_id
        for group_id, members in groups.items()
        for run in members
    }
    buckets: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for run in runs:
        for record in select_records(run, args, keep_all_steps=keep_all_steps):
            key = (
                run_to_group[run.run_id],
                record.get("step") if keep_all_steps else None,
                record.get("length"),
                record.get("ca_steps"),
                record.get("num_repeats"),
                record.get("repeat_from"),
                record.get("repeat_to"),
            )
            buckets[key].append(float(record["value"]))

    rows = []
    for key, values in sorted(buckets.items(), key=lambda item: str(item[0])):
        (
            group_id,
            step,
            length,
            ca_steps,
            num_repeats,
            repeat_from,
            repeat_to,
        ) = key
        summary = summarize(values)
        representative = groups[group_id][0]
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
                **summary,
            }
        )
    return groups, rows


def _format_summary(summary: Mapping[str, Any]) -> str:
    if int(summary["n"]) == 1:
        return f"{summary['mean']:.6f} (n=1)"
    return (
        f"{summary['mean']:.6f} ± {summary['std']:.6f} "
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
        rows.append(
            (
                run.run_id,
                run.manifest.get("status"),
                dotted_get(run.manifest, "model"),
                dotted_get(run.manifest, "training_pairs"),
                dotted_get(run.manifest, "seed"),
                dotted_get(run.manifest, "data_seed"),
                len(run.metrics),
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


def command_table(args: argparse.Namespace, runs: Sequence[Run]) -> int:
    groups, rows = aggregate_records(runs, args)
    pairs = sorted(
        {(row["ca_steps"], row["num_repeats"]) for row in rows},
        key=_pair_sort_key,
    )
    by_group_pair = {
        (row["configuration_id"], (row["ca_steps"], row["num_repeats"])): row
        for row in rows
    }
    rendered = []
    export_rows = []
    for group_id, members in sorted(groups.items()):
        representative = members[0]
        trained = {
            (int(pair["ca_steps"]), int(pair["num_repeats"]))
            for pair in representative.manifest["training"]["pairs"]
        }
        values = []
        export = {
            "configuration_id": group_id,
            "configuration": configuration_label(representative.manifest),
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
        rendered.append(
            (group_id, configuration_label(representative.manifest), *values)
        )
        export_rows.append(export)

    if rendered:
        print(
            _terminal_table(
                ("config", "configuration", *map(_pair_label, pairs)),
                rendered,
            )
        )
        print("\n[trained] marks pairs present in that configuration's training distribution.")
    else:
        print("No metric records matched the requested selection.")

    if args.output_dir is not None:
        report_dir = _report_dir(args)
        _write_csv(report_dir / "pair_table.csv", export_rows)
        write_json(report_dir / "summary.json", rows)
        _write_analysis_manifest(report_dir, args, runs)
        print(f"\nWrote analysis to {report_dir}")
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
        for record in select_records(run, args, keep_all_steps=True):
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
    figure, axis = plt.subplots(figsize=(11, 6))
    raw = context["raw"]
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

    for (group_id, line, run_id), points in raw.items():
        ordered = sorted(
            points,
            key=_category_sort_key if categorical else lambda value: value,
        )
        axis.plot(
            plotted_x(ordered),
            [points[x] for x in ordered],
            alpha=0.16,
            linewidth=1,
        )
    for (group_id, line), points in aggregates.items():
        ordered = sorted(
            points,
            key=_category_sort_key if categorical else lambda value: value,
        )
        means = [summarize(points[x])["mean"] for x in ordered]
        stds = [summarize(points[x])["std"] for x in ordered]
        representative = context["groups"][group_id][0]
        label = (
            f"{configuration_label(representative.manifest)} | {line}"
        )
        x_coordinates = plotted_x(ordered)
        axis.plot(x_coordinates, means, marker="o", linewidth=2, label=label)
        axis.fill_between(
            x_coordinates,
            [mean - std for mean, std in zip(means, stds)],
            [mean + std for mean, std in zip(means, stds)],
            alpha=0.16,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    if category_positions is not None:
        axis.set_xticks(
            list(category_positions.values()),
            list(category_positions.keys()),
        )
    if aggregates:
        axis.legend(fontsize="small")
    figure.tight_layout()
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
    if args.adjacent:
        filtered_runs = []
        for run in runs:
            records = tuple(
                record
                for record in run.metrics
                if record.get("repeat_from") is not None
                and record.get("repeat_to") == record.get("repeat_from") + 1
            )
            filtered_runs.append(Run(run.path, run.manifest, records))
        runs = filtered_runs
    report_dir = _report_dir(args)
    plt = _load_plotting(report_dir)
    context, aggregates = _seed_series(
        runs,
        args,
        x_getter=lambda record: int(record["step"]),
        line_getter=lambda record: (
            f"repeat {record['repeat_from']}→{record['repeat_to']}"
        ),
    )
    if not aggregates:
        raise ValueError(
            "No hidden-state records matched the requested state plot."
        )
    figure = _plot_seed_curves(
        plt,
        context,
        aggregates,
        xlabel="training step",
        ylabel=args.metric,
        title=f"Hidden-state {args.metric} across training",
        categorical=False,
    )
    figure.savefig(report_dir / "hidden_state.png", dpi=180)
    plt.close(figure)
    rows = _plot_summary_rows(aggregates, context)
    _write_csv(report_dir / "hidden_state.csv", rows)
    write_json(report_dir / "hidden_state.json", rows)
    _write_analysis_manifest(report_dir, args, runs)
    print(f"Wrote hidden-state plot and data to {report_dir}")
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
    table_parser.set_defaults(
        handler=command_table,
        metric="cell_accuracy",
        split="final_test",
        role="internal_repeat_extrapolation",
        checkpoint="best_extrapolation_unconstrained",
    )

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
        "plot-state", help="Plot hidden-state similarity across training."
    )
    _add_discovery_arguments(state_parser)
    _add_metric_arguments(state_parser)
    state_parser.add_argument(
        "--all-pairs",
        dest="adjacent",
        action="store_false",
        help="Plot every measured repeat pair instead of adjacent repeats only.",
    )
    state_parser.set_defaults(
        handler=command_plot_state,
        metric="cosine_similarity",
        split="validation",
        role="hidden_state_similarity",
        checkpoint=None,
        step="all",
        adjacent=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        filters = tuple(parse_filter(value) for value in args.where)
        runs = discover_runs(args.runs, filters=filters, strict=args.strict)
        return int(args.handler(args, runs))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
