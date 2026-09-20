import json

import pytest

from cellular_automaton.plot_dca_recall_cache_diagnostics import (
    _read_summary,
    main,
    select_plot_rows,
)


def _attention_row(mass=None):
    return {
        "recall_index": 1,
        "middle_layer_index_zero_based": 0,
        "cache_blocks": [
            {
                "cache_block_index_one_based": index,
                "phase": "evolution",
                "evolution_repeat": index,
            }
            for index in range(1, 4)
        ]
        + [
            {
                "cache_block_index_one_based": 4,
                "phase": "recall",
                "recall_index": 1,
            }
        ],
        "mean_mass_by_head_and_cache_block": mass
        or [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        "mean_value_norm_by_head_and_cache_block": [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ],
        "mean_block_contribution_norm_by_head_and_cache_block": [
            [0.01, 0.02, 0.03, 0.04],
            [0.04, 0.03, 0.02, 0.01],
        ],
    }


def _summary():
    results = []
    for depth, cell_accuracy, internal_accuracy in (
        (1, 0.55, 0.60),
        (3, 0.85, 0.95),
    ):
        results.append(
            {
                "horizon": 3,
                "intervention_depth": depth,
                "recall_age": 3 - depth,
                "is_latest_repeat": depth == 3,
                "conditions": {
                    "baseline": {
                        "metrics": {"cell_accuracy": cell_accuracy},
                        "internal_consistency": {
                            "decoded_requested_repeat": {
                                "cell_accuracy": internal_accuracy
                            }
                        },
                        "attention": [_attention_row()],
                    }
                },
            }
        )
    return {
        "experiment": "dca_model_level_recall_cache_interventions",
        "results": results,
    }


def test_plotter_writes_comparable_per_head_figures_and_metadata(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    output_dir = tmp_path / "plots"

    main(
        [
            "--summary",
            str(summary_path),
            "--depths",
            "1",
            "3",
            "--condition",
            "baseline",
            "--output-dir",
            str(output_dir),
        ]
    )

    first = output_dir / "recall_cache_h3_d1_r1_l0_baseline.png"
    latest = output_dir / "recall_cache_h3_d3_r1_l0_baseline.png"
    assert first.is_file() and first.stat().st_size > 0
    assert latest.is_file() and latest.stat().st_size > 0

    metadata = json.loads(
        (output_dir / "plot_summary.json").read_text(encoding="utf-8")
    )
    assert metadata["selected_depths"] == [1, 3]
    assert metadata["shared_scales"]["attention_mass"]["vmax"] == 1.0
    assert metadata["figures"][0]["requested_block_label"] == "E1"
    assert metadata["figures"][0][
        "candidate_state_input_block_label"
    ] == "E2"
    assert metadata["figures"][1]["requested_block_label"] == "E3"
    assert metadata["figures"][1][
        "candidate_state_input_block_label"
    ] is None
    assert metadata["figures"][1]["is_latest_repeat"] is True
    assert metadata["figures"][1]["recall_block_labels"] == ["R1"]


def test_select_plot_rows_rejects_misaligned_value_diagnostics():
    summary = _summary()
    summary["results"][0]["conditions"]["baseline"]["attention"][0][
        "mean_value_norm_by_head_and_cache_block"
    ] = [[1.0, 2.0]]

    with pytest.raises(ValueError, match="does not align"):
        select_plot_rows(
            summary,
            horizon=3,
            depths=[1],
            condition_id="baseline",
            recall_index=1,
            layer_index=0,
        )


def test_old_summary_without_value_diagnostics_requests_rerun():
    summary = _summary()
    del summary["results"][0]["conditions"]["baseline"]["attention"][0][
        "mean_value_norm_by_head_and_cache_block"
    ]

    with pytest.raises(ValueError, match="rerun the evaluator"):
        select_plot_rows(
            summary,
            horizon=3,
            depths=[1],
            condition_id="baseline",
            recall_index=1,
            layer_index=0,
        )


def test_summary_loader_rejects_unrelated_json(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"experiment": "something_else"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not a DCA"):
        _read_summary(path)
