"""Plot internal decoded recall agreement as a function of recall age.

Each panel fixes a forward horizon H. Each coloured curve is one model seed.
The bold black curve is the across-seed mean.

The plotted measurement is agreement between the recalled decoded output and
the model's saved decoded prediction at the requested repeat. It therefore
measures internal recall, not agreement with the true Rule 30 trajectory.
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
)
from cellular_automaton.ca_reporting import METRICS_FILENAME, write_json


QUERY_ROLE = "delayed_recall_internal_decoded_requested"
QUERY_METRIC = "cell_accuracy"


def collect_internal_recall(run, args):
    """Stream one JSONL file and retain only requested-repeat agreement."""

    measurements: dict[tuple[int, int], dict[str, Any]] = {}

    for record in iter_metrics(run.path / METRICS_FILENAME):
        if record.get("length") != args.length:
            continue
        if record.get("data_split") != args.split:
            continue
        if record.get("checkpoint_type") != args.checkpoint:
            continue
        if record.get("evaluation_role") != QUERY_ROLE:
            continue
        if record.get("metric") != QUERY_METRIC:
            continue

        horizon = record.get("num_repeats")
        query_repeat = record.get("repeat_from")
        if horizon is None or query_repeat is None:
            continue

        horizon = int(horizon)
        query_repeat = int(query_repeat)
        recall_age = horizon - query_repeat

        _put_latest(
            measurements,
            (horizon, recall_age),
            record,
        )

    return measurements


def selected_horizons(args, run_data):
    available = sorted(
        {
            horizon
            for measurements in run_data.values()
            for horizon, _recall_age in measurements
        }
    )

    if args.horizons == ["all"]:
        return available

    if "all" in args.horizons:
        raise ValueError("'all' cannot be combined with numbered horizons.")

    requested = [int(value) for value in args.horizons]
    missing = [
        horizon for horizon in requested if horizon not in available
    ]
    if missing:
        print(
            "WARNING: no selected run contains horizons: "
            + ", ".join(map(str, missing))
        )
    return requested


def build_rows(groups, run_data, horizons, *, checkpoint):
    """Build an explicit run × horizon × age grid, including missing data."""

    rows = []

    for configuration_id, members in groups.items():
        representative = members[0]
        configuration = configuration_label(representative.manifest)
        model = dotted_get(representative.manifest, "model")

        for run in members:
            measurements = run_data[run.run_id]

            for horizon in horizons:
                for recall_age in range(horizon):
                    measurement = measurements.get(
                        (horizon, recall_age)
                    )
                    rows.append(
                        {
                            "configuration_id": configuration_id,
                            "configuration": configuration,
                            "model": model,
                            "run_id": run.run_id,
                            "model_seed": dotted_get(
                                run.manifest, "seed"
                            ),
                            "data_seed": dotted_get(
                                run.manifest, "data_seed"
                            ),
                            "horizon": horizon,
                            "query_repeat": horizon - recall_age,
                            "recall_age": recall_age,
                            "is_no_op": recall_age == 0,
                            "checkpoint_type": checkpoint,
                            "checkpoint_step": (
                                None
                                if measurement is None
                                else measurement["step"]
                            ),
                            "internal_cell_accuracy": (
                                None
                                if measurement is None
                                else float(measurement["value"])
                            ),
                            "metrics_file_present": (
                                run.path / METRICS_FILENAME
                            ).is_file(),
                        }
                    )

    return rows


def plot_internal_recall(
    plt,
    groups,
    rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    checkpoint: str,
):
    group_items = sorted(
        groups.items(),
        key=lambda item: (
            str(dotted_get(item[1][0].manifest, "model")),
            item[0],
        ),
    )

    figure, axes = plt.subplots(
        nrows=len(horizons),
        ncols=len(group_items),
        squeeze=False,
        figsize=(
            6.8 * len(group_items),
            3.6 * len(horizons),
        ),
    )

    seed_values = sorted(
        {
            dotted_get(run.manifest, "seed")
            for _configuration_id, members in group_items
            for run in members
        },
        key=lambda value: (value is None, str(value)),
    )
    colour_map = plt.get_cmap("tab10")
    seed_colours = {
        seed: colour_map(index % 10)
        for index, seed in enumerate(seed_values)
    }

    for column, (configuration_id, members) in enumerate(group_items):
        representative = members[0]
        model = dotted_get(representative.manifest, "model")

        for row_index, horizon in enumerate(horizons):
            axis = axes[row_index][column]
            panel_rows = [
                row
                for row in rows
                if row["configuration_id"] == configuration_id
                and row["horizon"] == horizon
            ]

            ages = list(range(horizon))
            run_lookup = {
                run.run_id: {
                    int(row["recall_age"]): row
                    for row in panel_rows
                    if row["run_id"] == run.run_id
                }
                for run in members
            }

            for run in members:
                seed = dotted_get(run.manifest, "seed")
                by_age = run_lookup[run.run_id]
                values = [
                    (
                        math.nan
                        if by_age.get(age, {}).get(
                            "internal_cell_accuracy"
                        )
                        is None
                        else float(
                            by_age[age]["internal_cell_accuracy"]
                        )
                    )
                    for age in ages
                ]

                if all(math.isnan(value) for value in values):
                    axis.plot(
                        [],
                        [],
                        color=seed_colours[seed],
                        label=f"seed {seed} (missing)",
                    )
                else:
                    axis.plot(
                        ages,
                        values,
                        color=seed_colours[seed],
                        marker="o",
                        markersize=3.0,
                        linewidth=1.4,
                        alpha=0.85,
                        label=f"seed {seed}",
                    )

            mean_values = []
            coverage = []
            for age in ages:
                values = [
                    float(row["internal_cell_accuracy"])
                    for row in panel_rows
                    if row["recall_age"] == age
                    and row["internal_cell_accuracy"] is not None
                ]
                coverage.append(len(values))
                mean_values.append(
                    math.nan
                    if not values
                    else statistics.fmean(values)
                )

            nonzero_coverage = [
                count for count in coverage if count > 0
            ]
            if nonzero_coverage:
                minimum_n = min(nonzero_coverage)
                maximum_n = max(nonzero_coverage)
                coverage_label = (
                    f"n={minimum_n}"
                    if minimum_n == maximum_n
                    else f"n={minimum_n}–{maximum_n}"
                )
            else:
                coverage_label = "n=0"

            axis.plot(
                ages,
                mean_values,
                color="black",
                linewidth=2.6,
                label=f"mean ({coverage_label})",
                zorder=5,
            )

            partial_ages = [
                age
                for age, count in zip(ages, coverage)
                if 0 < count < len(members)
            ]
            partial_means = [
                mean
                for mean, count in zip(mean_values, coverage)
                if 0 < count < len(members)
            ]
            if partial_ages:
                axis.scatter(
                    partial_ages,
                    partial_means,
                    s=35,
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                )

            axis.axhline(
                0.5,
                color="#777777",
                linestyle=":",
                linewidth=0.9,
            )
            axis.set_xlim(-0.4, max(0.4, horizon - 0.6))
            axis.set_ylim(0.0, 1.01)
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{model} [{configuration_id}] · H={horizon}"
            )
            axis.set_xlabel(
                "Recall age (0 = current; larger = further back)"
            )

            if column == 0:
                axis.set_ylabel("Internal decoded cell agreement")

            if row_index == 0:
                axis.legend(fontsize=8)

    figure.suptitle(
        "Internal recall as a function of recall age\n"
        f"checkpoint={checkpoint}; coloured lines=individual seeds; "
        "black=mean",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot requested-repeat internal decoded agreement against "
            "recall age."
        )
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--run-id", nargs="+", default=None)
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["6", "12", "18", "24", "30"],
        help=(
            "Horizons shown as separate panel rows, or the single value "
            "'all'."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="best_internal_recall",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "final_test"),
        default="final_test",
    )
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument(
        "--average-over",
        default="seed,data_seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    args.command = "plot-internal-recall-by-age"

    try:
        filters = tuple(
            parse_filter(expression)
            for expression in args.where
        )
        discovered = discover_runs(
            args.runs,
            filters=filters,
            strict=args.strict,
            load_metrics=False,
        )
        selected = _select_delayed_runs(
            discovered,
            args.run_id,
        )
        if not selected:
            raise ValueError("No runs matched the selection.")

        average_over = _field_list(
            args.average_over,
            ("seed", "data_seed"),
        )
        groups = _delayed_groups(
            selected,
            average_over=average_over,
        )

        run_data = {}
        for run in selected:
            print(f"Reading {run.run_id} ...")
            run_data[run.run_id] = collect_internal_recall(
                run,
                args,
            )

        horizons = selected_horizons(args, run_data)
        if not horizons:
            raise ValueError("No matching horizons were found.")

        rows = build_rows(
            groups,
            run_data,
            horizons,
            checkpoint=args.checkpoint,
        )
        if not any(
            row["internal_cell_accuracy"] is not None
            for row in rows
        ):
            raise ValueError(
                "No matching internal requested-repeat records found."
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt = _load_plotting(args.output_dir)

        figure = plot_internal_recall(
            plt,
            groups,
            rows,
            horizons,
            checkpoint=args.checkpoint,
        )
        figure.savefig(
            args.output_dir / "internal_recall_by_age.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

        _write_csv(
            args.output_dir / "internal_recall_by_age.csv",
            rows,
        )
        write_json(
            args.output_dir / "internal_recall_by_age.json",
            rows,
        )
        _write_analysis_manifest(
            args.output_dir,
            args,
            selected,
        )

        print(f"Wrote results to {args.output_dir}")
        return 0

    except (OSError, ValueError) as error:
        parser.error(str(error))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())