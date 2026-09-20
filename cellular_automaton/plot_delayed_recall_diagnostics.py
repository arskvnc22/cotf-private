"""Plot recent delayed-recall behaviour against forward Rule 30 accuracy.

The main figure separates:

1. Forward and delayed ground-truth accuracy.
2. Agreement with the model's own saved internal predictions.
3. Derived no-op gaps.

A second figure classifies the no-op output as best matching:

- the intended current repeat H;
- an earlier repeat;
- the bitwise complement of repeat H.

Run from the repository root with:

    python -m cellular_automaton.plot_delayed_recall_diagnostics \
        --runs iridis/dca-30/runs \
        --run-id \
            run_26__dca_cotf_h30_100pct_delayed_subtract \
            run_27__dca_cotf_h30_100pct_delayed_subtract \
            run_28__dca_cotf_h30_100pct_delayed_subtract \
            run_29__dca_but_h30_100pct_delayed_subtract \
            run_30__dca_but_h30_100pct_delayed_subtract \
            run_31__dca_but_h30_100pct_delayed_subtract \
        --output-dir iridis/dca-30/outputs/recent-recall-diagnostics
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from cellular_automaton.ca_analyze import (
    _delayed_groups,
    _field_list,
    _load_plotting,
    _put_latest,
    _select_delayed_runs,
    _write_analysis_manifest,
    _write_csv,
    configuration_label,
    discover_runs,
    dotted_get,
    iter_metrics,
    parse_filter,
    summarize,
)
from cellular_automaton.ca_reporting import METRICS_FILENAME, write_json


RECENT_AGES = (0, 1, 2)

# field, plot label, colour, line style, first meaningful horizon
GROUND_TRUTH_SERIES = (
    ("id_cell_accuracy", "ID CA", "#222222", "-", 1),
    ("delayed_cell_age_0", "Delayed age 0", "#d62728", "-", 1),
    ("delayed_cell_age_1", "Delayed age 1", "#1f77b4", "--", 2),
    ("delayed_cell_age_2", "Delayed age 2", "#2ca02c", ":", 3),
)

INTERNAL_SERIES = (
    ("internal_cell_age_0", "Internal age 0", "#d62728", "-", 1),
    ("internal_cell_age_1", "Internal age 1", "#1f77b4", "--", 2),
    ("internal_cell_age_2", "Internal age 2", "#2ca02c", ":", 3),
)

GAP_SERIES = (
    (
        "noop_functional_gap",
        "Age-0 delayed − ID",
        "#9467bd",
        "-",
        1,
    ),
    (
        "noop_preservation_loss",
        "1 − internal age 0",
        "#ff7f0e",
        "--",
        1,
    ),
    (
        "age_zero_discontinuity",
        "Internal age 0 − age 1",
        "#17becf",
        ":",
        2,
    ),
)

ALL_SERIES = GROUND_TRUTH_SERIES + INTERNAL_SERIES + GAP_SERIES


def collect_recent_diagnostics(
    run,
    args: argparse.Namespace,
) -> dict[str, dict[Any, dict[str, Any]]]:
    """Stream one metrics file while retaining only records needed here.

    This intentionally does not call collect_delayed_run: the full delayed
    collector retains every candidate metric for every query and can consume
    substantial memory for the H=30 metric files.
    """

    result: dict[str, dict[Any, dict[str, Any]]] = {
        "in_distribution": {},
        "ground_truth": {},
        "internal": {},
        "noop_candidates": {},
    }

    metrics_path = run.path / METRICS_FILENAME
    for record in iter_metrics(metrics_path):
        if record.get("length") != args.length:
            continue
        if record.get("data_split") != args.split:
            continue

        checkpoint_type = record.get("checkpoint_type")
        if args.checkpoint != "any" and checkpoint_type != args.checkpoint:
            continue

        if record.get("metric") != "cell_accuracy":
            continue

        role = record.get("evaluation_role")
        horizon_value = record.get("num_repeats")
        if horizon_value is None:
            horizon_value = record.get("ca_steps")
        if horizon_value is None:
            continue
        horizon = int(horizon_value)

        if role == "in_distribution":
            _put_latest(result["in_distribution"], horizon, record)
            continue

        query_repeat_value = record.get("repeat_from")
        if query_repeat_value is None:
            continue
        query_repeat = int(query_repeat_value)
        recall_age = horizon - query_repeat

        if role == "delayed_recall_ground_truth_requested":
            if recall_age in RECENT_AGES:
                _put_latest(
                    result["ground_truth"],
                    (horizon, recall_age),
                    record,
                )
            continue

        if role == "delayed_recall_internal_decoded_requested":
            if recall_age in RECENT_AGES:
                _put_latest(
                    result["internal"],
                    (horizon, recall_age),
                    record,
                )
            continue

        if (
            role == "delayed_recall_internal_decoded_candidate"
            and query_repeat == horizon
            and record.get("repeat_to") is not None
        ):
            candidate_repeat = int(record["repeat_to"])
            _put_latest(
                result["noop_candidates"],
                (horizon, candidate_repeat),
                record,
            )

    return result


def measurement_value(
    measurements: Mapping[Any, Mapping[str, Any]],
    key: Any,
) -> float | None:
    measurement = measurements.get(key)
    if measurement is None:
        return None
    return float(measurement["value"])


def best_noop_match(
    candidate_scores: Mapping[int, float],
    *,
    horizon: int,
    requested_internal: float | None,
    binary_output: bool,
) -> dict[str, Any]:
    """Classify what the age-zero output most closely resembles.

    Ties are resolved conservatively in favour of the intended current
    repeat. Complement agreement is exact only for binary decoded outputs.
    """

    scores = dict(candidate_scores)
    if horizon not in scores and requested_internal is not None:
        scores[horizon] = requested_internal

    if not scores:
        return {
            "best_match_kind": "missing",
            "best_candidate_repeat": None,
            "best_candidate_age": None,
            "best_match_agreement": None,
            "current_agreement": requested_internal,
            "complement_current_agreement": (
                None
                if requested_internal is None or not binary_output
                else 1.0 - requested_internal
            ),
            "best_margin_over_current": None,
        }

    best_repeat, best_score = max(
        scores.items(),
        key=lambda item: (
            item[1],
            item[0] == horizon,
            item[0],
        ),
    )

    current_score = scores.get(horizon)
    complement_score = (
        None
        if current_score is None or not binary_output
        else 1.0 - current_score
    )

    if complement_score is not None and complement_score > best_score + 1e-12:
        kind = "complement_current"
        selected_repeat = None
        selected_age = None
        selected_score = complement_score
    elif best_repeat == horizon:
        kind = "current"
        selected_repeat = best_repeat
        selected_age = 0
        selected_score = best_score
    else:
        kind = "earlier_repeat"
        selected_repeat = best_repeat
        selected_age = horizon - best_repeat
        selected_score = best_score

    margin = (
        None
        if current_score is None
        else selected_score - current_score
    )

    return {
        "best_match_kind": kind,
        "best_candidate_repeat": selected_repeat,
        "best_candidate_age": selected_age,
        "best_match_agreement": selected_score,
        "current_agreement": current_score,
        "complement_current_agreement": complement_score,
        "best_margin_over_current": margin,
    }


def build_seed_rows(
    groups,
    run_data: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create an explicit run × horizon grid, including missing runs."""

    rows: list[dict[str, Any]] = []

    for configuration_id, members in groups.items():
        representative = members[0]
        configuration = configuration_label(representative.manifest)
        model = dotted_get(representative.manifest, "model")
        expected_run_ids = [run.run_id for run in members]

        horizons = sorted(
            {
                horizon
                for run in members
                for horizon in (
                    set(run_data[run.run_id]["in_distribution"])
                    | {
                        key[0]
                        for key in run_data[run.run_id]["ground_truth"]
                    }
                    | {
                        key[0]
                        for key in run_data[run.run_id]["internal"]
                    }
                    | {
                        key[0]
                        for key in run_data[run.run_id]["noop_candidates"]
                    }
                )
            }
        )

        for run in members:
            data = run_data[run.run_id]
            model_info = run.manifest.get("model", {})
            binary_output = int(model_info.get("vocab_size", 2)) == 2

            for horizon in horizons:
                row: dict[str, Any] = {
                    "configuration_id": configuration_id,
                    "configuration": configuration,
                    "model": model,
                    "run_id": run.run_id,
                    "model_seed": dotted_get(run.manifest, "seed"),
                    "data_seed": dotted_get(run.manifest, "data_seed"),
                    "horizon": horizon,
                    "expected_run_count": len(members),
                    "expected_run_ids": expected_run_ids,
                    "metrics_file_present": (
                        run.path / METRICS_FILENAME
                    ).is_file(),
                }

                row["id_cell_accuracy"] = measurement_value(
                    data["in_distribution"],
                    horizon,
                )

                for age in RECENT_AGES:
                    row[f"delayed_cell_age_{age}"] = measurement_value(
                        data["ground_truth"],
                        (horizon, age),
                    )
                    row[f"internal_cell_age_{age}"] = measurement_value(
                        data["internal"],
                        (horizon, age),
                    )

                delayed_zero = row["delayed_cell_age_0"]
                id_accuracy = row["id_cell_accuracy"]
                internal_zero = row["internal_cell_age_0"]
                internal_one = row["internal_cell_age_1"]

                row["noop_functional_gap"] = (
                    None
                    if delayed_zero is None or id_accuracy is None
                    else delayed_zero - id_accuracy
                )
                row["noop_preservation_loss"] = (
                    None
                    if internal_zero is None
                    else 1.0 - internal_zero
                )
                row["age_zero_discontinuity"] = (
                    None
                    if internal_zero is None or internal_one is None
                    else internal_zero - internal_one
                )

                candidate_scores = {
                    candidate_repeat: float(measurement["value"])
                    for (
                        candidate_horizon,
                        candidate_repeat,
                    ), measurement in data["noop_candidates"].items()
                    if candidate_horizon == horizon
                }
                row.update(
                    best_noop_match(
                        candidate_scores,
                        horizon=horizon,
                        requested_internal=internal_zero,
                        binary_output=binary_output,
                    )
                )

                current_candidate = candidate_scores.get(horizon)
                row["current_candidate_vs_requested_error"] = (
                    None
                    if current_candidate is None or internal_zero is None
                    else current_candidate - internal_zero
                )

                requested_values = [
                    row[field]
                    for field, _label, _colour, _style, minimum_horizon
                    in ALL_SERIES
                    if horizon >= minimum_horizon
                ]
                row["missing_all_requested_metrics"] = all(
                    value is None for value in requested_values
                )
                rows.append(row)

    return rows


def build_summary_rows(
    groups,
    seed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate each series while reporting exact seed coverage."""

    result = []

    for configuration_id, members in groups.items():
        group_rows = [
            row
            for row in seed_rows
            if row["configuration_id"] == configuration_id
        ]
        horizons = sorted({int(row["horizon"]) for row in group_rows})
        expected_ids = [run.run_id for run in members]

        for field, label, _colour, _style, minimum_horizon in ALL_SERIES:
            for horizon in horizons:
                if horizon < minimum_horizon:
                    continue

                horizon_rows = [
                    row for row in group_rows if row["horizon"] == horizon
                ]
                observations = [
                    (row["run_id"], float(row[field]))
                    for row in horizon_rows
                    if row[field] is not None
                ]
                observed_ids = {run_id for run_id, _value in observations}
                values = [value for _run_id, value in observations]
                summary = summarize(values) if values else None

                result.append(
                    {
                        "configuration_id": configuration_id,
                        "configuration": (
                            group_rows[0]["configuration"]
                            if group_rows
                            else None
                        ),
                        "model": (
                            group_rows[0]["model"] if group_rows else None
                        ),
                        "horizon": horizon,
                        "series": field,
                        "series_label": label,
                        "expected_n": len(expected_ids),
                        "n": len(values),
                        "missing_run_ids": [
                            run_id
                            for run_id in expected_ids
                            if run_id not in observed_ids
                        ],
                        "mean": None if summary is None else summary["mean"],
                        "std": None if summary is None else summary["std"],
                        "min": None if summary is None else summary["min"],
                        "max": None if summary is None else summary["max"],
                        "values": values,
                    }
                )

    return result


def plot_panel(
    axis,
    group_rows: Sequence[Mapping[str, Any]],
    members,
    series,
    *,
    ylim: tuple[float, float],
    chance_line: bool,
) -> None:
    horizons = sorted({int(row["horizon"]) for row in group_rows})
    expected_n = len(members)

    rows_by_run = {
        run.run_id: {
            int(row["horizon"]): row
            for row in group_rows
            if row["run_id"] == run.run_id
        }
        for run in members
    }

    for field, label, colour, style, minimum_horizon in series:
        valid_horizons = [
            horizon
            for horizon in horizons
            if horizon >= minimum_horizon
        ]

        # Thin per-seed curves. NaNs deliberately break lines at missing data.
        for run in members:
            run_rows = rows_by_run[run.run_id]
            values = [
                (
                    math.nan
                    if run_rows.get(horizon, {}).get(field) is None
                    else float(run_rows[horizon][field])
                )
                for horizon in valid_horizons
            ]
            axis.plot(
                valid_horizons,
                values,
                color=colour,
                linestyle=style,
                linewidth=0.9,
                alpha=0.22,
            )

        means = []
        counts = []
        for horizon in valid_horizons:
            values = [
                float(row[field])
                for row in group_rows
                if row["horizon"] == horizon and row[field] is not None
            ]
            counts.append(len(values))
            means.append(
                math.nan if not values else statistics.fmean(values)
            )

        nonzero_counts = [count for count in counts if count > 0]
        if nonzero_counts:
            minimum_n = min(nonzero_counts)
            maximum_n = max(nonzero_counts)
            coverage = (
                f"n={minimum_n}/{expected_n}"
                if minimum_n == maximum_n
                else f"n={minimum_n}–{maximum_n}/{expected_n}"
            )
        else:
            coverage = f"n=0/{expected_n}"

        axis.plot(
            valid_horizons,
            means,
            color=colour,
            linestyle=style,
            marker="o",
            markersize=3.5,
            linewidth=2.4,
            label=f"{label} mean ({coverage})",
        )

        # Hollow markers identify horizons where the mean has partial coverage.
        partial_x = [
            horizon
            for horizon, count in zip(valid_horizons, counts)
            if 0 < count < expected_n
        ]
        partial_y = [
            mean
            for mean, count in zip(means, counts)
            if 0 < count < expected_n
        ]
        if partial_x:
            axis.scatter(
                partial_x,
                partial_y,
                s=34,
                facecolors="white",
                edgecolors=colour,
                linewidths=1.2,
                zorder=5,
            )

    if chance_line:
        axis.axhline(
            0.5,
            color="#777777",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
        )
    else:
        axis.axhline(
            0.0,
            color="#777777",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
        )

    axis.set_ylim(*ylim)
    axis.set_xlabel("Forward horizon H")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)


def plot_recent_horizons(
    plt,
    groups,
    seed_rows: Sequence[Mapping[str, Any]],
):
    group_items = list(groups.items())
    figure, axes = plt.subplots(
        nrows=len(group_items),
        ncols=3,
        squeeze=False,
        figsize=(17, 4.5 * len(group_items)),
    )

    for row_index, (configuration_id, members) in enumerate(group_items):
        group_rows = [
            row
            for row in seed_rows
            if row["configuration_id"] == configuration_id
        ]
        model = dotted_get(members[0].manifest, "model")
        missing_files = [
            run.run_id
            for run in members
            if not (run.path / METRICS_FILENAME).is_file()
        ]
        coverage_note = (
            f"{len(members)} selected runs"
            if not missing_files
            else (
                f"{len(members)} selected runs; "
                f"{len(missing_files)} missing metrics file"
            )
        )

        ground_truth_axis, internal_axis, gap_axis = axes[row_index]

        plot_panel(
            ground_truth_axis,
            group_rows,
            members,
            GROUND_TRUTH_SERIES,
            ylim=(0.0, 1.01),
            chance_line=True,
        )
        plot_panel(
            internal_axis,
            group_rows,
            members,
            INTERNAL_SERIES,
            ylim=(0.0, 1.01),
            chance_line=True,
        )
        plot_panel(
            gap_axis,
            group_rows,
            members,
            GAP_SERIES,
            ylim=(-1.01, 1.01),
            chance_line=False,
        )

        ground_truth_axis.set_title(
            f"{model} [{configuration_id}]\n"
            f"Forward vs delayed ground truth\n{coverage_note}"
        )
        internal_axis.set_title(
            "Agreement with saved internal predictions"
        )
        gap_axis.set_title("No-op effects and discontinuity")

        ground_truth_axis.set_ylabel("Cell accuracy")
        internal_axis.set_ylabel("Cell agreement")
        gap_axis.set_ylabel("Signed gap / loss")

    figure.suptitle(
        "Delayed recall near the current repeat\n"
        "thin = individual runs; bold = mean; "
        "hollow mean marker = incomplete seed coverage",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return figure


def plot_best_matches(
    plt,
    groups,
    seed_rows: Sequence[Mapping[str, Any]],
):
    from matplotlib.lines import Line2D

    ordered_runs = [
        (configuration_id, run)
        for configuration_id, members in groups.items()
        for run in members
    ]
    figure_height = max(4.0, 0.75 * len(ordered_runs) + 2.5)
    figure, axis = plt.subplots(figsize=(16, figure_height))

    row_lookup = {
        (row["run_id"], int(row["horizon"])): row
        for row in seed_rows
    }
    all_horizons = sorted(
        {int(row["horizon"]) for row in seed_rows}
    )

    colours = {
        "current": "#2ca02c",
        "earlier_repeat": "#1f77b4",
        "complement_current": "#d62728",
        "missing": "#bdbdbd",
    }
    markers = {
        "current": "o",
        "earlier_repeat": "^",
        "complement_current": "X",
        "missing": "x",
    }

    y_labels = []
    previous_configuration = None

    for y_position, (configuration_id, run) in enumerate(ordered_runs):
        if (
            previous_configuration is not None
            and configuration_id != previous_configuration
        ):
            axis.axhline(
                y_position - 0.5,
                color="#555555",
                linewidth=1.0,
            )
        previous_configuration = configuration_id

        seed = dotted_get(run.manifest, "seed")
        model = dotted_get(run.manifest, "model")
        y_labels.append(
            f"{model} seed={seed} [{configuration_id}]"
        )

        for horizon in all_horizons:
            row = row_lookup.get((run.run_id, horizon))
            if row is None:
                continue

            kind = str(row["best_match_kind"])
            score = row["best_match_agreement"]
            marker_size = (
                20.0
                if score is None
                else 20.0 + 80.0 * max(0.0, min(1.0, float(score)))
            )

            axis.scatter(
                horizon,
                y_position,
                s=marker_size,
                marker=markers[kind],
                color=colours[kind],
                alpha=0.85,
            )

            if kind == "earlier_repeat":
                age = row["best_candidate_age"]
                axis.annotate(
                    f"−{age}",
                    (horizon, y_position),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=colours[kind],
                )

    axis.set_yticks(range(len(y_labels)), y_labels)
    axis.set_xticks(all_horizons)
    axis.set_xlabel("Forward horizon H")
    axis.set_ylabel("Run")
    axis.set_title(
        "Best match for the age-zero/no-op output\n"
        "marker size represents agreement; −k labels an earlier repeat H−k"
    )
    axis.grid(axis="x", alpha=0.15)

    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=colours["current"],
                label="Current repeat H",
            ),
            Line2D(
                [],
                [],
                marker="^",
                linestyle="",
                color=colours["earlier_repeat"],
                label="Earlier repeat",
            ),
            Line2D(
                [],
                [],
                marker="X",
                linestyle="",
                color=colours["complement_current"],
                label="Complement of H",
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                color=colours["missing"],
                label="Missing measurement",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        fontsize=8,
    )

    figure.tight_layout()
    return figure


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot age-zero, age-one, and age-two delayed recall against "
            "same-checkpoint forward CA accuracy."
        )
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Root searched recursively for run manifests.",
    )
    parser.add_argument(
        "--run-id",
        nargs="+",
        default=None,
        help="Exact run IDs to include.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Optional manifest filter; repeat for AND.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "final_test"),
        default="final_test",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_delayed_recall",
        help="Checkpoint type, or 'any'.",
    )
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument(
        "--average-over",
        default="seed,data_seed",
        help="Replicate fields removed from configuration grouping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed run manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.command = "plot-delayed-recall-diagnostics"

    try:
        if args.length <= 0:
            raise ValueError("--length must be positive.")

        filters = tuple(parse_filter(value) for value in args.where)
        discovered_runs = discover_runs(
            args.runs,
            filters=filters,
            strict=args.strict,
            load_metrics=False,
        )
        selected_runs = _select_delayed_runs(
            discovered_runs,
            args.run_id,
        )
        if not selected_runs:
            raise ValueError("No run manifests matched the selection.")

        average_over = _field_list(
            args.average_over,
            ("seed", "data_seed"),
        )
        groups = _delayed_groups(
            selected_runs,
            average_over=average_over,
        )

        run_data = {}
        for run in selected_runs:
            print(f"Reading {run.run_id} ...")
            run_data[run.run_id] = collect_recent_diagnostics(
                run,
                args,
            )

        if not any(
            data["in_distribution"]
            or data["ground_truth"]
            or data["internal"]
            for data in run_data.values()
        ):
            raise ValueError(
                "No matching delayed-recall or in-distribution records found."
            )

        seed_rows = build_seed_rows(groups, run_data)
        summary_rows = build_summary_rows(groups, seed_rows)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt = _load_plotting(args.output_dir)

        recent_figure = plot_recent_horizons(
            plt,
            groups,
            seed_rows,
        )
        recent_figure.savefig(
            args.output_dir / "recent_recall_horizons.png",
            dpi=200,
        )
        plt.close(recent_figure)

        best_match_figure = plot_best_matches(
            plt,
            groups,
            seed_rows,
        )
        best_match_figure.savefig(
            args.output_dir / "noop_best_matches.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(best_match_figure)

        _write_csv(
            args.output_dir / "recent_recall_seed_rows.csv",
            seed_rows,
        )
        _write_csv(
            args.output_dir / "recent_recall_summary.csv",
            summary_rows,
        )
        _write_csv(
            args.output_dir / "noop_best_matches.csv",
            [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    in {
                        "configuration_id",
                        "model",
                        "run_id",
                        "model_seed",
                        "data_seed",
                        "horizon",
                        "metrics_file_present",
                        "best_match_kind",
                        "best_candidate_repeat",
                        "best_candidate_age",
                        "best_match_agreement",
                        "current_agreement",
                        "complement_current_agreement",
                        "best_margin_over_current",
                        "current_candidate_vs_requested_error",
                    }
                }
                for row in seed_rows
            ],
        )

        write_json(
            args.output_dir / "recent_recall_diagnostics.json",
            {
                "recent_ages": list(RECENT_AGES),
                "seed_rows": seed_rows,
                "summary_rows": summary_rows,
            },
        )
        _write_analysis_manifest(
            args.output_dir,
            args,
            selected_runs,
        )

        print(f"Wrote diagnostics to {args.output_dir}")
        return 0

    except (OSError, ValueError) as error:
        parser.error(str(error))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())