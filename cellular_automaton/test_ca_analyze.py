from argparse import Namespace
from copy import deepcopy
import json

import pytest

import cellular_automaton.ca_analyze as ca_analyze
from cellular_automaton.ca_analyze import (
    Run,
    aggregate_records,
    architecture_signature,
    comparison_configuration,
    configuration_label,
    discover_runs,
    evaluation_protocol_signature,
    main,
    manifest_matches,
    parse_filter,
    summarize,
)
from cellular_automaton.ca_reporting import (
    build_run_manifest,
    write_eval_metrics,
    write_run_manifest,
)
from cellular_automaton.test_ca_reporting import sample_args, sample_stats


def write_sample_run(root, number, seed, data_seed, accuracy):
    run_dir = root / f"run_{number}__but_full_depth"
    args = sample_args(seed=seed, data_seed=data_seed)
    manifest = build_run_manifest(args, run_dir, root / "checkpoints")
    write_run_manifest(run_dir, manifest)
    stats = sample_stats()
    stats["checkpoint_analysis"]["best_id"]["task_metrics"][
        "internal_repeat_extrapolation"
    ]["steps_3_repeats_3"]["by_length"]["64"]["cell_accuracy"] = accuracy
    write_eval_metrics(run_dir, stats)
    return run_dir


def write_validation_run(
    root,
    number,
    seed,
    evaluations,
    *,
    layers=(0, 1, 0),
    final_pairs=((7, 7),),
):
    run_dir = root / f"run_{number}__ca_cotf"
    args = sample_args(seed=seed, data_seed=11)
    begin, middle, end = layers
    args.update(
        {
            "model": "ca_cotf",
            "n_layer_begin": begin,
            "n_layer": begin + middle + end,
            "n_layer_end": end,
            "ca_extrapolation_val_pairs": [[3, 3], [5, 5]],
            "ca_extrapolation_best_pair": [3, 3],
            "ca_final_eval_pairs": [list(pair) for pair in final_pairs],
            "ca_repeat_diagnostic_max_repeats": 5,
            "ca_repeat_diagnostic_horizons": list(range(6)),
        }
    )
    manifest = build_run_manifest(args, run_dir, root / "checkpoints")
    write_run_manifest(run_dir, manifest)

    stats = sample_stats()
    template = deepcopy(stats["eval"]["50"])
    stats["eval"] = {}
    for step, pair_3_accuracy, pair_5_accuracy in evaluations:
        evaluation = deepcopy(template)
        for pair, accuracy in (
            ((3, 3), pair_3_accuracy),
            ((5, 5), pair_5_accuracy),
        ):
            key = f"steps_{pair[0]}_repeats_{pair[1]}"
            evaluation["extrapolation_validation"][key] = {
                "ca_steps": pair[0],
                "num_repeats": pair[1],
                "by_length": {
                    "64": {
                        "loss": 1.0 - accuracy,
                        "cell_accuracy": accuracy,
                        "exact_sequence_accuracy": 0.0,
                    }
                },
            }
            repeat_key = f"repeats_{pair[1]}"
            horizon_key = f"steps_{pair[0]}"
            matrix = evaluation["repeat_diagnostics"]["64"][
                "repeat_horizon_matrix"
            ]
            matrix.setdefault(repeat_key, {})[horizon_key] = {
                "loss": 1.0 - accuracy,
                "cell_accuracy": accuracy,
                "exact_sequence_accuracy": 0.0,
            }
        stats["eval"][str(step)] = evaluation
    write_eval_metrics(run_dir, stats)
    return run_dir


def metric_args():
    return Namespace(
        metric="cell_accuracy",
        split="final_test",
        role="internal_repeat_extrapolation",
        checkpoint="best_id",
        length=64,
        step="latest",
        average_over="seed,data_seed",
    )


def test_filters_support_aliases_and_arbitrary_pair_strings(tmp_path):
    manifest = build_run_manifest(
        sample_args(), tmp_path / "run_0__but", tmp_path / "checkpoints"
    )
    filters = [
        parse_filter("model=but_full_depth"),
        parse_filter("training_pairs=1:1|2:2|4:4"),
        parse_filter("lr<=0.001"),
        parse_filter("cache_policy=full"),
        parse_filter("forward_policy_label=cache_full"),
    ]
    assert manifest_matches(manifest, filters)
    assert not manifest_matches(manifest, [parse_filter("seed=999")])


def test_list_streams_metric_count_without_eager_loading(
    tmp_path, monkeypatch, capsys
):
    write_sample_run(tmp_path, 0, seed=1, data_seed=11, accuracy=0.7)

    def fail_if_eagerly_loaded(_path):
        raise AssertionError("list must not materialize a complete metric file")

    monkeypatch.setattr(ca_analyze, "read_metrics", fail_if_eagerly_loaded)

    assert main(["list", "--runs", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "run_0__but_full_depth" in output
    assert "metric rows" in output
    assert "training cache" in output
    assert "full" in output


def test_plot_series_streams_filtered_metrics_from_disk(tmp_path):
    write_sample_run(tmp_path, 0, seed=1, data_seed=11, accuracy=0.7)
    run = discover_runs(tmp_path, load_metrics=False)[0]
    args = metric_args()
    args.step = "all"

    _context, aggregates = ca_analyze._seed_series(
        [run],
        args,
        x_getter=lambda record: (
            record.get("ca_steps"),
            record.get("num_repeats"),
        ),
        line_getter=lambda _record: "selected",
    )

    retained_values = [
        value
        for points in aggregates.values()
        for values in points.values()
        for value in values
    ]
    assert retained_values == [pytest.approx(0.7)]


def test_cache_policy_labels_and_grouping_distinguish_intervention(tmp_path):
    full_args = sample_args(seed=1)
    recent_args = sample_args(seed=2)
    recent_args["repeat_cache_window"] = 4
    full = build_run_manifest(
        full_args, tmp_path / "full", tmp_path / "checkpoints"
    )
    recent = build_run_manifest(
        recent_args, tmp_path / "recent", tmp_path / "checkpoints"
    )

    assert "train-cache=full" in configuration_label(full)
    assert "train-cache=recent-4" in configuration_label(recent)
    assert manifest_matches(recent, [parse_filter("cache_window=4")])
    assert manifest_matches(
        recent, [parse_filter("training_cache_window=4")]
    )
    assert comparison_configuration(
        full, average_over=("seed",)
    ) != comparison_configuration(recent, average_over=("seed",))

    legacy_recent = deepcopy(recent)
    legacy_recent["resolved_args"].pop("repeat_cache_window")
    assert comparison_configuration(
        full, average_over=("seed",)
    ) != comparison_configuration(legacy_recent, average_over=("seed",))


def test_historical_manifest_defaults_to_full_cache_label(tmp_path):
    manifest = build_run_manifest(
        sample_args(), tmp_path / "historical", tmp_path / "checkpoints"
    )
    manifest.pop("forward_policy")

    assert "train-cache=full" in configuration_label(manifest)
    assert manifest_matches(manifest, [parse_filter("cache_policy=full")])


def test_seed_aggregation_reports_mean_variance_and_values(tmp_path):
    write_sample_run(tmp_path, 0, seed=1, data_seed=11, accuracy=0.7)
    write_sample_run(tmp_path, 1, seed=2, data_seed=12, accuracy=0.9)
    runs = discover_runs(tmp_path)
    groups, rows = aggregate_records(runs, metric_args())

    assert len(groups) == 1
    row = next(
        row
        for row in rows
        if row["ca_steps"] == 3 and row["num_repeats"] == 3
    )
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(0.8)
    assert row["variance"] == pytest.approx(0.02)
    assert row["values"] == [0.7, 0.9]


def test_non_averaged_data_seed_keeps_runs_separate(tmp_path):
    first = build_run_manifest(
        sample_args(seed=1, data_seed=11),
        tmp_path / "run_0__but",
        tmp_path / "checkpoints",
    )
    second = build_run_manifest(
        sample_args(seed=2, data_seed=12),
        tmp_path / "run_1__but",
        tmp_path / "checkpoints",
    )
    first_config = comparison_configuration(first, average_over=("seed",))
    second_config = comparison_configuration(second, average_over=("seed",))
    assert first_config != second_config


def test_summary_includes_seed_spread_statistics():
    result = summarize([0.5, 0.7, 0.9])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(0.7)
    assert result["median"] == pytest.approx(0.7)
    assert result["min"] == 0.5
    assert result["max"] == 0.9


def test_checkpoint_none_selects_training_exposure_records(tmp_path):
    write_sample_run(tmp_path, 0, seed=1, data_seed=11, accuracy=0.7)
    run = discover_runs(tmp_path)[0]
    args = metric_args()
    args.metric = "examples_seen"
    args.split = "training"
    args.role = "training_exposure_by_pair"
    args.checkpoint = "none"
    args.length = None

    _, rows = aggregate_records([run], args)
    assert any(
        row["ca_steps"] == 1
        and row["num_repeats"] == 1
        and row["mean"] == 3200
        for row in rows
    )


def test_architecture_signature_distinguishes_wrapper_layers(tmp_path):
    middle_only_args = sample_args()
    middle_only_args.update(
        {"n_layer": 1, "n_layer_begin": 0, "n_layer_end": 0}
    )
    wrapped_args = sample_args()
    wrapped_args.update(
        {"n_layer": 3, "n_layer_begin": 1, "n_layer_end": 1}
    )
    middle_only = build_run_manifest(
        middle_only_args, tmp_path / "middle", tmp_path / "checkpoints"
    )
    wrapped = build_run_manifest(
        wrapped_args, tmp_path / "wrapped", tmp_path / "checkpoints"
    )

    assert "layers=0/1/0" in architecture_signature(middle_only)
    assert "layers=1/1/1" in architecture_signature(wrapped)


def test_evaluation_protocol_signature_exposes_coverage_and_selection(
    tmp_path,
):
    run_dir = write_validation_run(
        tmp_path,
        0,
        seed=3,
        evaluations=[(50, 0.8, 0.7)],
        final_pairs=(),
    )
    run = discover_runs(run_dir)[0]

    signature = evaluation_protocol_signature(run.manifest)

    assert "val=3:3|5:5" in signature
    assert "select=3:3" in signature
    assert "direct_final=none" in signature
    assert "diagnostic_horizons=0,1,2,3,4,5 repeats=1:5" in signature


def test_table_reports_contributing_runs_seeds_and_architecture(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root, 0, seed=3, evaluations=[(50, 0.8, 0.7)]
    )

    assert main(
        [
            "table",
            "--runs",
            str(runs_root),
            "--split",
            "validation",
            "--role",
            "extrapolation_validation",
            "--checkpoint",
            "none",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "run_0__ca_cotf" in output
    assert "layers=0/1/0" in output
    assert "3" in output


def test_table_streams_requested_pairs_and_reports_seed_range(
    tmp_path, capsys, monkeypatch
):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root,
        0,
        seed=1,
        evaluations=[(50, 0.7, 0.6), (100, 0.8, 0.65)],
    )
    write_validation_run(
        runs_root,
        1,
        seed=2,
        evaluations=[(50, 0.8, 0.7), (100, 0.9, 0.75)],
    )
    output_dir = tmp_path / "table"

    def reject_eager_loading(path):
        raise AssertionError(f"eagerly loaded metrics from {path}")

    monkeypatch.setattr(ca_analyze, "read_metrics", reject_eager_loading)

    assert main(
        [
            "table",
            "--runs",
            str(runs_root),
            "--split",
            "validation",
            "--role",
            "repeat_horizon_diagnostic",
            "--checkpoint",
            "none",
            "--average-over",
            "seed",
            "--report-pairs",
            "5:5",
            "3:3",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    header = output.splitlines()[0]
    assert header.index("5:5") < header.index("3:3")
    assert "[min=0.650000, max=0.750000]" in output
    assert "[min=0.800000, max=0.900000]" in output

    rows = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert {(row["ca_steps"], row["num_repeats"]) for row in rows} == {
        (3, 3),
        (5, 5),
    }
    by_pair = {(row["ca_steps"], row["num_repeats"]): row for row in rows}
    assert by_pair[(5, 5)]["n"] == 2
    assert by_pair[(5, 5)]["mean"] == pytest.approx(0.7)
    assert by_pair[(5, 5)]["min"] == pytest.approx(0.65)
    assert by_pair[(5, 5)]["max"] == pytest.approx(0.75)
    assert "5:5_min" in (output_dir / "pair_table.csv").read_text(
        encoding="utf-8"
    )
    table_export = (output_dir / "pair_table.csv").read_text(encoding="utf-8")
    assert "training_cache_policy" in table_export
    assert "training_cache_window" in table_export


def test_table_rejects_duplicate_report_pairs(tmp_path, capsys):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root, 0, seed=1, evaluations=[(50, 0.7, 0.6)]
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "table",
                "--runs",
                str(runs_root),
                "--split",
                "validation",
                "--role",
                "repeat_horizon_diagnostic",
                "--checkpoint",
                "none",
                "--report-pairs",
                "3:3",
                "3:3",
            ]
        )
    assert error.value.code == 2
    assert "cannot contain duplicate pairs" in capsys.readouterr().err


def test_table_no_match_explains_available_selection(tmp_path, capsys):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root, 0, seed=3, evaluations=[(50, 0.8, 0.7)]
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "table",
                "--runs",
                str(runs_root),
                "--role",
                "extrapolation_validation",
            ]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "No metric records matched" in message
    assert "split=validation" in message
    assert "checkpoint=none" in message


def test_leaderboard_ranks_runs_and_reports_pairs_at_selected_step(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root,
        0,
        seed=1,
        evaluations=[(50, 0.7, 0.6), (100, 0.9, 0.8)],
    )
    write_validation_run(
        runs_root,
        1,
        seed=2,
        evaluations=[(50, 0.85, 0.75), (100, 0.8, 0.95)],
        layers=(1, 1, 1),
    )
    output_dir = tmp_path / "leaderboard"

    assert main(
        [
            "leaderboard",
            "--runs",
            str(runs_root),
            "--select-pair",
            "3:3",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert output.index("run_0__ca_cotf") < output.index("run_1__ca_cotf")
    assert "layers=0/1/0" in output
    assert "layers=1/1/1" in output
    assert "training cache" in output

    rows = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    by_run = {row["run_id"]: row for row in rows}
    assert by_run["run_0__ca_cotf"]["selection_step"] == 100
    assert by_run["run_0__ca_cotf"]["3:3"] == pytest.approx(0.9)
    assert by_run["run_0__ca_cotf"]["5:5"] == pytest.approx(0.8)
    assert by_run["run_0__ca_cotf"]["training_cache_policy"] == "full"
    assert by_run["run_0__ca_cotf"]["training_cache_window"] is None
    assert by_run["run_1__ca_cotf"]["selection_step"] == 50
    assert by_run["run_1__ca_cotf"]["5:5"] == pytest.approx(0.75)
    assert (output_dir / "leaderboard.csv").is_file()
    assert (output_dir / "analysis_manifest.json").is_file()


def test_leaderboard_can_select_diagnostic_and_limit_reported_pairs(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root,
        0,
        seed=1,
        evaluations=[(50, 0.7, 0.6), (100, 0.9, 0.8)],
    )
    output_dir = tmp_path / "diagnostic_leaderboard"

    assert main(
        [
            "leaderboard",
            "--runs",
            str(runs_root),
            "--role",
            "repeat_horizon_diagnostic",
            "--select-pair",
            "3:3",
            "--report-pairs",
            "3:3",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    header = output.splitlines()[0]
    assert "model" in header
    assert "protocol" in header
    assert "3:3" in header
    assert "5:5" not in header
    assert "ca_cotf" in output

    rows = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert rows[0]["selection_step"] == 100
    assert rows[0]["3:3"] == pytest.approx(0.9)
    assert "5:5" not in rows[0]


def test_leaderboard_streams_metrics_and_preserves_latest_tie_break(
    tmp_path, capsys, monkeypatch
):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root,
        0,
        seed=1,
        evaluations=[(50, 0.9, 0.6), (100, 0.9, 0.8)],
    )
    output_dir = tmp_path / "streaming_leaderboard"

    def reject_eager_loading(path):
        raise AssertionError(f"eagerly loaded metrics from {path}")

    monkeypatch.setattr(ca_analyze, "read_metrics", reject_eager_loading)

    assert main(
        [
            "leaderboard",
            "--runs",
            str(runs_root),
            "--select-pair",
            "3:3",
            "--report-pairs",
            "3:3",
            "5:5",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    capsys.readouterr()
    rows = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert rows[0]["selection_step"] == 100
    assert rows[0]["selection_value"] == pytest.approx(0.9)
    assert rows[0]["3:3"] == pytest.approx(0.9)
    assert rows[0]["5:5"] == pytest.approx(0.8)


def test_leaderboard_warns_about_mixed_evaluation_protocols(tmp_path, capsys):
    runs_root = tmp_path / "runs"
    write_validation_run(
        runs_root,
        0,
        seed=1,
        evaluations=[(50, 0.7, 0.6)],
        final_pairs=((7, 7),),
    )
    write_validation_run(
        runs_root,
        1,
        seed=2,
        evaluations=[(50, 0.8, 0.7)],
        final_pairs=((8, 8),),
    )

    assert main(
        [
            "leaderboard",
            "--runs",
            str(runs_root),
            "--select-pair",
            "3:3",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "WARNING: mixed evaluation protocols" in output
    assert "P1" in output
    assert "P2" in output


def test_cli_writes_tables_and_all_plot_types(tmp_path):
    pytest.importorskip("matplotlib")
    runs_root = tmp_path / "runs"
    write_sample_run(runs_root, 0, seed=1, data_seed=11, accuracy=0.7)
    write_sample_run(runs_root, 1, seed=2, data_seed=12, accuracy=0.9)

    table_dir = tmp_path / "table"
    assert main(
        [
            "table",
            "--runs",
            str(runs_root),
            "--checkpoint",
            "best_id",
            "--output-dir",
            str(table_dir),
        ]
    ) == 0
    assert (table_dir / "pair_table.csv").is_file()
    assert (table_dir / "analysis_manifest.json").is_file()

    horizon_dir = tmp_path / "horizon"
    assert main(
        [
            "plot-horizon",
            "--runs",
            str(runs_root),
            "--checkpoint",
            "best_id",
            "--output-dir",
            str(horizon_dir),
        ]
    ) == 0
    assert (horizon_dir / "horizon.png").is_file()
    assert (horizon_dir / "horizon.csv").is_file()

    training_dir = tmp_path / "training"
    assert main(
        [
            "plot-training",
            "--runs",
            str(runs_root),
            "--output-dir",
            str(training_dir),
        ]
    ) == 0
    assert (training_dir / "training.png").is_file()

    state_dir = tmp_path / "state"
    assert main(
        [
            "plot-state",
            "--runs",
            str(runs_root / "run_0__but_full_depth"),
            "--checkpoint",
            "best_id",
            "--split",
            "final_test",
            "--output-dir",
            str(state_dir),
        ]
    ) == 0
    assert (state_dir / "hidden_state.png").is_file()
    state_rows = json.loads((state_dir / "hidden_state.json").read_text())
    assert {row["checkpoint_step"] for row in state_rows} == {50}
    assert {
        (row["repeat_from"], row["repeat_to"]) for row in state_rows
    } == {(2, 3)}
