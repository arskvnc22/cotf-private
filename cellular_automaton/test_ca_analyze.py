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
    lstm_ut_configuration_fields
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


def write_delayed_run(
    root,
    number,
    *,
    seed,
    data_seed,
    cell_accuracy,
    exact_accuracy,
    loss,
    controller="persistent",
    delayed_percentage=75,
    recall_repeats=1,
    query_accuracies=None,
    internal_query_accuracies=None,
    model="dca_but",
):
    run_dir = root / f"run_{number}__{model}_{controller}"
    args = sample_args(seed=seed, data_seed=data_seed)
    args.update(
        {
            "model": model,
            "ca_controller_application": controller,
            "ca_max_relative_age": 3,
            "ca_delayed_percentage": delayed_percentage,
            "query_horizon_policy": "uniform",
            "ca_query_loss_weight": 2.0,
            "ca_recall_repeats": recall_repeats,
            "ca_delayed_best_metric": "cell_accuracy",
            "ca_boundary": "periodic",
            "ca_bernoulli_p": 0.5,
            "ca_exact_data_resume": True,
            "ca_shuffle_seed": data_seed,
            "scheduler": "none",
            "eval_freq": 50,
            "lm_cache": "none",
            "repeat_cache_window": None,
        }
    )
    manifest = build_run_manifest(args, run_dir, root / "checkpoints")
    write_run_manifest(run_dir, manifest)
    stats = sample_stats()
    replacement = {
        "loss": loss,
        "cell_accuracy": cell_accuracy,
        "exact_sequence_accuracy": exact_accuracy,
        "matthews_correlation": cell_accuracy,
        "mean_bit_errors_per_sequence": 64 * (1.0 - cell_accuracy),
    }
    delayed_locations = [
        stats["eval"]["50"],
        stats["checkpoint_analysis"]["best_delayed_recall"],
    ]
    for location in delayed_locations:
        pair = location["delayed_recall"]["steps_3_repeats_3"]
        pair["nontrivial_queries_macro"] = {
            **pair["nontrivial_queries_macro"],
            **replacement,
        }
        query = pair["queries"]["query_repeat_1"]
        agreement = 0.5
        changed_fraction = 1.0 - agreement
        requested_accuracy = cell_accuracy
        changed_accuracy = cell_accuracy - 0.1
        current_accuracy = (
            requested_accuracy
            + changed_fraction
            - 2.0 * changed_fraction * changed_accuracy
        )
        query["metrics"] = {
            **query["metrics"],
            **replacement,
            "cell_accuracy": requested_accuracy,
        }
        query["ground_truth_comparison_by_repeat"]["repeat_3"] = {
            **query["ground_truth_comparison_by_repeat"]["repeat_3"],
            "cell_accuracy": current_accuracy,
        }
        query["ground_truth_state_collision_by_repeat"]["repeat_3"] = {
            **query["ground_truth_state_collision_by_repeat"]["repeat_3"],
            "cell_agreement": agreement,
        }
        internal = query["internal_consistency"]
        internal["decoded_requested_repeat"] = {
            **internal["decoded_requested_repeat"],
            "cell_accuracy": requested_accuracy,
        }
        internal["decoded_comparison_by_repeat"]["repeat_3"] = {
            **internal["decoded_comparison_by_repeat"]["repeat_3"],
            "cell_accuracy": current_accuracy,
        }
        internal["decoded_state_collision_by_repeat"]["repeat_3"] = {
            **internal["decoded_state_collision_by_repeat"]["repeat_3"],
            "cell_agreement": agreement,
        }
        pair["nontrivial_internal_consistency_macro"] = {
            **pair["nontrivial_internal_consistency_macro"],
            "decoded_requested_repeat_cell_accuracy": cell_accuracy,
        }
        summary = location["delayed_recall_summary"]
        summary["nontrivial_queries_pair_macro"] = {
            **summary["nontrivial_queries_pair_macro"],
            **replacement,
        }
        summary["nontrivial_internal_consistency_pair_macro"] = {
            **summary["nontrivial_internal_consistency_pair_macro"],
            "decoded_requested_repeat_cell_accuracy": cell_accuracy,
        }
        if query_accuracies is not None:
            if set(query_accuracies) != {1, 2, 3}:
                raise ValueError("Test query accuracies must cover repeats 1, 2, 3.")
            internal_accuracies = internal_query_accuracies or query_accuracies
            if set(internal_accuracies) != {1, 2, 3}:
                raise ValueError(
                    "Test internal accuracies must cover repeats 1, 2, 3."
                )
            template = deepcopy(query)
            pair["queries"] = {}
            for query_repeat in (1, 2, 3):
                query_entry = deepcopy(template)
                query_entry["query_repeat"] = query_repeat
                query_entry["recall_age"] = 3 - query_repeat
                query_entry["metrics"]["cell_accuracy"] = query_accuracies[
                    query_repeat
                ]
                query_entry["internal_consistency"][
                    "decoded_requested_repeat"
                ]["cell_accuracy"] = internal_accuracies[query_repeat]
                pair["queries"][f"query_repeat_{query_repeat}"] = query_entry
            nontrivial = [query_accuracies[1], query_accuracies[2]]
            internal_nontrivial = [
                internal_accuracies[1], internal_accuracies[2]
            ]
            pair["nontrivial_queries_macro"]["cell_accuracy"] = sum(
                nontrivial
            ) / len(nontrivial)
            pair["nontrivial_internal_consistency_macro"][
                "decoded_requested_repeat_cell_accuracy"
            ] = sum(internal_nontrivial) / len(internal_nontrivial)
            summary["nontrivial_queries_pair_macro"]["cell_accuracy"] = sum(
                nontrivial
            ) / len(nontrivial)
            summary["nontrivial_internal_consistency_pair_macro"][
                "decoded_requested_repeat_cell_accuracy"
            ] = sum(internal_nontrivial) / len(internal_nontrivial)
    stats["checkpoint_analysis"]["best_delayed_recall"]["task_metrics"] = {
        "in_distribution": {
            "steps_3_repeats_3": {
                "ca_steps": 3,
                "num_repeats": 3,
                "by_length": {
                    "64": {
                        **replacement,
                        "cell_accuracy": min(1.0, cell_accuracy + 0.05),
                    }
                },
            }
        }
    }
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


def test_lstm_ut_ablation_fields_are_labeled_and_grouped(tmp_path):
    variants = (
        ("learned", "previous_and_proposed"),
        ("learned", "proposed"),
        ("none", "previous_and_proposed"),
        ("none", "proposed"),
    )
    manifests = []

    for index, (forget_gate, control_input) in enumerate(variants):
        args = sample_args(seed=index, data_seed=11)
        args.update(
            {
                "model": "lstm_ut_bidir",
                "lstm_forget_gate": forget_gate,
                "lstm_control_input": control_input,
            }
        )
        manifest = build_run_manifest(
            args,
            tmp_path / f"run_{index}__lstm_ut_bidir",
            tmp_path / "checkpoints",
        )
        manifests.append(manifest)

        assert lstm_ut_configuration_fields(manifest) == (
            forget_gate,
            control_input,
        )
        label = configuration_label(manifest)
        assert f"lstm-forget={forget_gate}" in label
        assert f"lstm-control={control_input}" in label

    configurations = {
        json.dumps(
            comparison_configuration(
                manifest,
                average_over=("seed", "data_seed"),
            ),
            sort_keys=True,
        )
        for manifest in manifests
    }
    assert len(configurations) == 4

    assert manifest_matches(
        manifests[3],
        [
            parse_filter('forget_gate="none"'),
            parse_filter("control_input=proposed"),
        ],
    )

    non_lstm = build_run_manifest(
        sample_args(),
        tmp_path / "run_non_lstm",
        tmp_path / "checkpoints",
    )
    assert lstm_ut_configuration_fields(non_lstm) == (None, None)
    assert "lstm-forget=" not in configuration_label(non_lstm)

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


def test_delayed_compare_streams_selected_metrics_and_writes_artifacts(
    tmp_path, capsys, monkeypatch
):
    runs_root = tmp_path / "runs"
    first = write_delayed_run(
        runs_root,
        1,
        seed=1,
        data_seed=11,
        cell_accuracy=0.8,
        exact_accuracy=0.4,
        loss=0.3,
        controller="persistent",
    )
    second = write_delayed_run(
        runs_root,
        2,
        seed=1,
        data_seed=11,
        cell_accuracy=0.9,
        exact_accuracy=0.6,
        loss=0.2,
        controller="subtract",
    )
    output_dir = tmp_path / "delayed"
    original_iter_metrics = ca_analyze.iter_metrics
    calls = []

    def counted_stream(path):
        calls.append(path)
        yield from original_iter_metrics(path)

    def reject_eager_loading(path):
        raise AssertionError(f"eagerly loaded metrics from {path}")

    monkeypatch.setattr(ca_analyze, "iter_metrics", counted_stream)
    monkeypatch.setattr(ca_analyze, "read_metrics", reject_eager_loading)

    assert main(
        [
            "delayed-compare",
            "--runs",
            str(runs_root),
            "--run-id",
            first.name,
            second.name,
            "--metrics",
            "cell_accuracy",
            "exact_sequence_accuracy",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "Overall nontrivial delayed recall" in output
    assert "query_repeat" in output
    assert "recall_age" in output
    assert "is_no_op" in output
    assert "internal cell" not in output
    assert len(calls) == 2
    assert {path.parent.name for path in calls} == {first.name, second.name}

    report = json.loads(
        (output_dir / "delayed_comparison.json").read_text(encoding="utf-8")
    )
    assert {row["metric"] for row in report["overall"]} == {
        "cell_accuracy",
        "exact_sequence_accuracy",
    }
    query = next(
        row
        for row in report["queries"]
        if row["query_repeat"] == 1 and row["metric"] == "cell_accuracy"
    )
    assert query["horizon"] == 3
    assert query["recall_age"] == 2
    assert query["is_no_op"] is False
    assert report["protocol_count"] == 1
    for filename in (
        "delayed_summary.csv",
        "delayed_horizons.csv",
        "delayed_queries.csv",
        "delayed_candidates.csv",
        "delayed_validation.csv",
        "delayed_comparison.json",
        "analysis_manifest.json",
    ):
        assert (output_dir / filename).is_file()
    horizon_export = (output_dir / "delayed_horizons.csv").read_text(
        encoding="utf-8"
    )
    assert "delayed_nontrivial" in horizon_export
    assert "in_distribution_same_checkpoint" in horizon_export
    candidate_rows = (output_dir / "delayed_candidates.csv").read_text(
        encoding="utf-8"
    )
    assert "delayed_recall_ground_truth_candidate" in candidate_rows
    assert "delayed_recall_internal_logit_similarity" in candidate_rows


def test_delayed_compare_accepts_one_metric_and_rejects_more_than_two(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    write_delayed_run(
        runs_root,
        1,
        seed=1,
        data_seed=11,
        cell_accuracy=0.8,
        exact_accuracy=0.4,
        loss=0.3,
    )
    assert main(
        [
            "delayed-compare",
            "--runs",
            str(runs_root),
            "--metrics",
            "cell_accuracy",
        ]
    ) == 0
    output = capsys.readouterr().out
    overall_header = output.split("Overall nontrivial delayed recall", 1)[1]
    assert "cell" in overall_header
    assert "exact" not in overall_header

    with pytest.raises(SystemExit) as error:
        main(
            [
                "delayed-compare",
                "--runs",
                str(runs_root),
                "--metrics",
                "cell_accuracy",
                "exact_sequence_accuracy",
                "loss",
            ]
        )
    assert error.value.code == 2
    assert "one or two" in capsys.readouterr().err


def test_delayed_compare_aggregates_replicate_seeds(tmp_path):
    runs_root = tmp_path / "runs"
    write_delayed_run(
        runs_root,
        1,
        seed=1,
        data_seed=11,
        cell_accuracy=0.7,
        exact_accuracy=0.3,
        loss=0.4,
    )
    write_delayed_run(
        runs_root,
        2,
        seed=2,
        data_seed=12,
        cell_accuracy=0.9,
        exact_accuracy=0.5,
        loss=0.2,
    )
    runs = discover_runs(runs_root, load_metrics=False)
    args = Namespace(
        metrics=["cell_accuracy"],
        run_id=None,
        split="final_test",
        checkpoint="best_delayed_recall",
        length=64,
        average_over="seed,data_seed",
        strict_match=False,
    )
    _selected, _specs, _raw, report = ca_analyze._build_delayed_report(
        args, runs
    )
    assert len(report["configurations"]) == 1
    assert len(report["overall"]) == 1
    assert report["overall"][0]["n"] == 2
    assert report["overall"][0]["mean"] == pytest.approx(0.8)


def test_recall_repeat_compare_reports_paired_age_and_internal_deltas(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "recall-repeat-comparison"
    common = {"model": "dca_cotf_cache", "data_seed": 11}
    write_delayed_run(
        runs_root,
        1,
        seed=1,
        cell_accuracy=0.0,
        exact_accuracy=0.0,
        loss=0.0,
        recall_repeats=1,
        query_accuracies={1: 0.6, 2: 0.7, 3: 0.8},
        internal_query_accuracies={1: 0.5, 2: 0.6, 3: 0.7},
        **common,
    )
    write_delayed_run(
        runs_root,
        2,
        seed=2,
        cell_accuracy=0.0,
        exact_accuracy=0.0,
        loss=0.0,
        recall_repeats=1,
        query_accuracies={1: 0.8, 2: 0.9, 3: 1.0},
        internal_query_accuracies={1: 0.7, 2: 0.8, 3: 0.9},
        **common,
    )
    write_delayed_run(
        runs_root,
        3,
        seed=1,
        cell_accuracy=0.0,
        exact_accuracy=0.0,
        loss=0.0,
        recall_repeats=2,
        query_accuracies={1: 0.7, 2: 0.9, 3: 0.9},
        internal_query_accuracies={1: 0.6, 2: 0.8, 3: 0.8},
        **common,
    )
    write_delayed_run(
        runs_root,
        4,
        seed=2,
        cell_accuracy=0.0,
        exact_accuracy=0.0,
        loss=0.0,
        recall_repeats=2,
        query_accuracies={1: 0.9, 2: 1.0, 3: 1.0},
        internal_query_accuracies={1: 0.8, 2: 0.9, 3: 0.95},
        **common,
    )

    assert main(
        [
            "recall-repeat-compare",
            "--runs",
            str(runs_root),
            "--horizon",
            "3",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "GT mean delta" in output
    assert "internal mean delta" in output
    assert "Checkpoint steps: one-pass=[50], two-pass=[50]" in output
    assert "Deltas are two-pass minus one-pass" in output
    assert "overall excludes recall age zero" in output

    report = json.loads(
        (output_dir / "recall_repeat_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["recall_repeat_counts"] == [1, 2]
    assert report["one_repeat_checkpoint_steps"] == [50]
    assert report["two_repeat_checkpoint_steps"] == [50]
    assert report["paired_seeds"] == [
        {"model_seed": 1, "data_seed": 11},
        {"model_seed": 2, "data_seed": 11},
    ]
    by_age = {
        row["recall_age"]: row
        for row in report["rows"]
        if row["scope"] == "recall_age"
    }
    assert by_age[0]["query_repeat"] == 3
    assert by_age[2]["query_repeat"] == 1
    assert by_age[1]["ground_truth_one_repeat_mean"] == pytest.approx(0.8)
    assert by_age[1]["ground_truth_two_repeat_mean"] == pytest.approx(0.95)
    assert by_age[1][
        "ground_truth_delta_two_minus_one_mean"
    ] == pytest.approx(0.15)
    assert by_age[0]["internal_one_repeat_mean"] == pytest.approx(0.8)
    assert by_age[0]["internal_two_repeat_mean"] == pytest.approx(0.875)
    assert by_age[0][
        "internal_delta_two_minus_one_mean"
    ] == pytest.approx(0.075)

    overall = next(
        row for row in report["rows"] if row["scope"] == "overall_nontrivial"
    )
    assert overall["ground_truth_one_repeat_mean"] == pytest.approx(0.75)
    assert overall["ground_truth_two_repeat_mean"] == pytest.approx(0.875)
    assert overall[
        "ground_truth_delta_two_minus_one_mean"
    ] == pytest.approx(0.125)
    assert overall["internal_one_repeat_mean"] == pytest.approx(0.65)
    assert overall["internal_two_repeat_mean"] == pytest.approx(0.775)
    assert overall["internal_delta_two_minus_one_mean"] == pytest.approx(0.125)
    assert len(overall["paired_values"]) == 4
    assert (output_dir / "recall_repeat_comparison.csv").is_file()
    assert (output_dir / "recall_repeat_paired_values.csv").is_file()
    assert (output_dir / "analysis_manifest.json").is_file()


def test_recall_repeat_compare_requires_paired_seed_coverage(tmp_path, capsys):
    runs_root = tmp_path / "runs"
    common = {
        "data_seed": 11,
        "cell_accuracy": 0.7,
        "exact_accuracy": 0.4,
        "loss": 0.3,
        "model": "dca_cotf_cache",
        "query_accuracies": {1: 0.6, 2: 0.7, 3: 0.8},
        "internal_query_accuracies": {1: 0.5, 2: 0.6, 3: 0.7},
    }
    write_delayed_run(
        runs_root, 1, seed=1, recall_repeats=1, **common
    )
    write_delayed_run(
        runs_root, 2, seed=2, recall_repeats=1, **common
    )
    write_delayed_run(
        runs_root, 3, seed=1, recall_repeats=2, **common
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "recall-repeat-compare",
                "--runs",
                str(runs_root),
                "--horizon",
                "3",
            ]
        )
    assert error.value.code == 2
    assert "identical paired" in capsys.readouterr().err


def test_recall_repeat_compare_rejects_other_configuration_differences(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    common = {
        "seed": 1,
        "data_seed": 11,
        "cell_accuracy": 0.7,
        "exact_accuracy": 0.4,
        "loss": 0.3,
        "model": "dca_cotf_cache",
        "query_accuracies": {1: 0.6, 2: 0.7, 3: 0.8},
        "internal_query_accuracies": {1: 0.5, 2: 0.6, 3: 0.7},
    }
    write_delayed_run(
        runs_root,
        1,
        recall_repeats=1,
        controller="persistent",
        **common,
    )
    write_delayed_run(
        runs_root,
        2,
        recall_repeats=2,
        controller="subtract",
        **common,
    )

    with pytest.raises(SystemExit) as error:
        main(
            [
                "recall-repeat-compare",
                "--runs",
                str(runs_root),
                "--horizon",
                "3",
            ]
        )
    assert error.value.code == 2
    assert "exactly one scientific configuration" in capsys.readouterr().err


def test_changed_cell_accuracy_excludes_cells_matching_current_state():
    assert ca_analyze._changed_cell_accuracy(0.8, 0.6, 0.5) == pytest.approx(
        0.7
    )
    assert ca_analyze._changed_cell_accuracy(1.0, 1.0, 1.0) is None


@pytest.mark.parametrize(
    ("ordinary_metric", "changed_metric"),
    (
        ("cell_accuracy", "changed_cell_accuracy"),
        (
            "internal_cell_accuracy",
            "internal_changed_cell_accuracy",
        ),
    ),
)
def test_changed_cell_metrics_do_not_replace_ordinary_accuracy(
    tmp_path, ordinary_metric, changed_metric
):
    runs_root = tmp_path / "runs"
    write_delayed_run(
        runs_root,
        1,
        seed=1,
        data_seed=11,
        cell_accuracy=0.8,
        exact_accuracy=0.4,
        loss=0.3,
    )
    runs = discover_runs(runs_root, load_metrics=False)
    args = Namespace(
        metrics=[ordinary_metric, changed_metric],
        run_id=None,
        split="final_test",
        checkpoint="best_delayed_recall",
        length=64,
        average_over="seed,data_seed",
        strict_match=False,
    )

    _selected, _specs, _raw, report = ca_analyze._build_delayed_report(
        args, runs
    )

    overall = {row["metric"]: row["mean"] for row in report["overall"]}
    assert overall[ordinary_metric] == pytest.approx(0.8)
    assert overall[changed_metric] == pytest.approx(0.7)


def test_delayed_compare_strict_protocol_match_rejects_mismatch(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    write_delayed_run(
        runs_root,
        1,
        seed=1,
        data_seed=11,
        cell_accuracy=0.8,
        exact_accuracy=0.4,
        loss=0.3,
        delayed_percentage=50,
    )
    write_delayed_run(
        runs_root,
        2,
        seed=1,
        data_seed=11,
        cell_accuracy=0.9,
        exact_accuracy=0.6,
        loss=0.2,
        controller="subtract",
        delayed_percentage=75,
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "delayed-compare",
                "--runs",
                str(runs_root),
                "--strict-match",
            ]
        )
    assert error.value.code == 2
    assert "requires one delayed-CA protocol" in capsys.readouterr().err


def test_delayed_leaderboard_uses_first_metric_and_auto_direction(
    tmp_path, capsys
):
    runs_root = tmp_path / "runs"
    first = write_delayed_run(
        runs_root,
        1,
        seed=1,
        data_seed=11,
        cell_accuracy=0.8,
        exact_accuracy=0.4,
        loss=0.3,
    )
    second = write_delayed_run(
        runs_root,
        2,
        seed=2,
        data_seed=12,
        cell_accuracy=0.7,
        exact_accuracy=0.3,
        loss=0.1,
    )
    output_dir = tmp_path / "leaderboard"
    assert main(
        [
            "delayed-leaderboard",
            "--runs",
            str(runs_root),
            "--metrics",
            "loss",
            "cell_accuracy",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert output.index(second.name) < output.index(first.name)
    assert "Ranked min by loss" in output
    rows = json.loads(
        (output_dir / "delayed_leaderboard.json").read_text(encoding="utf-8")
    )
    assert rows[0]["run_id"] == second.name
    assert rows[0]["loss"] == pytest.approx(0.1)
    assert rows[0]["cell_accuracy"] == pytest.approx(0.7)


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
