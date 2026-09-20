from types import SimpleNamespace

import pytest
import torch

from cellular_automaton import dca_intervene_eval as evaluator
from models.dca_cotf_att_intervene import CausalSelfAttention


def test_parse_args_accepts_horizons_and_intervention_depths():
    args = evaluator.parse_args(
        [
            "--run-manifest",
            "run_manifest.json",
            "--expected-run-id",
            "run_1",
            "--checkpoint-name",
            "checkpoint.pt",
            "--expected-step",
            "12",
            "--horizons",
            "6",
            "12",
            "--intervention-depths",
            "3",
            "6",
            "--output-dir",
            "results",
        ]
    )

    assert args.horizons == [6, 12]
    assert args.intervention_depths == [3, 6]
    assert args.num_recall_repeats is None
    assert args.collect_attention is True
    assert args.interventions == list(evaluator.INTERVENTIONS)


def test_resolve_evaluation_pairs_defaults_to_every_depth_per_horizon():
    pairs, skipped = evaluator.resolve_evaluation_pairs([2, 4], None, 12)

    assert pairs == [
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert skipped == []


def test_resolve_evaluation_pairs_records_impossible_cartesian_pairs():
    pairs, skipped = evaluator.resolve_evaluation_pairs([3, 6], [2, 5], 12)

    assert pairs == [(3, 2), (6, 2), (6, 5)]
    assert skipped == [
        {
            "horizon": 3,
            "intervention_depth": 5,
            "reason": "intervention_depth_exceeds_horizon",
        }
    ]


def test_resolve_evaluation_pairs_rejects_unusable_depth_and_horizon():
    with pytest.raises(ValueError, match="invalid for every"):
        evaluator.resolve_evaluation_pairs([3, 4], [5], 12)

    with pytest.raises(ValueError, match="ca_max_relative_age"):
        evaluator.resolve_evaluation_pairs([13], None, 12)


def test_recall_repeats_default_to_manifest_and_allow_explicit_override():
    model_args = SimpleNamespace(ca_recall_repeats=3)

    assert evaluator.resolve_num_recall_repeats(None, model_args) == 3
    assert evaluator.resolve_num_recall_repeats(2, model_args) == 2


def test_recall_repeats_require_a_valid_manifest_or_explicit_value():
    with pytest.raises(ValueError, match="resolved_args.ca_recall_repeats"):
        evaluator.resolve_num_recall_repeats(None, SimpleNamespace())


def test_attention_masks_target_block_at_the_repeat_block_boundary():
    logits = torch.zeros(1, 1, 2, 8)

    only = CausalSelfAttention.apply_recall_cache_intervention_mask(
        logits,
        intervention="target-repeat-only",
        target_repeat=2,
        evolution_repeats=3,
        recall_index=1,
        tokens_per_repeat=2,
    )
    masked = CausalSelfAttention.apply_recall_cache_intervention_mask(
        logits,
        intervention="target-repeat-masked",
        target_repeat=2,
        evolution_repeats=3,
        recall_index=1,
        tokens_per_repeat=2,
    )

    assert torch.isfinite(only[0, 0, 0]).tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert torch.isfinite(masked[0, 0, 0]).tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_value_corruption_permutates_only_the_target_repeat_values():
    attention = object.__new__(CausalSelfAttention)
    torch.nn.Module.__init__(attention)
    full = torch.arange(2 * 1 * 6 * 1).reshape(2, 1, 6, 1).float()
    before = full.clone()
    attention.all_values = (full, full[:, :, :6, :])

    metadata = attention.permute_cached_target_values(
        target_repeat=2,
        tokens_per_repeat=2,
        permutation_offset=1,
    )

    assert torch.equal(full[:, :, :2], before[:, :, :2])
    assert torch.equal(full[:, :, 4:], before[:, :, 4:])
    assert torch.equal(full[0, :, 2:4], before[1, :, 2:4])
    assert torch.equal(full[1, :, 2:4], before[0, :, 2:4])
    assert metadata["target_cache_block_zero_based"] == 1
    assert metadata["target_token_start"] == 2
    assert metadata["target_token_end_exclusive"] == 4


def test_requested_repeat_attention_preserves_heads_and_ranks_repeats():
      accumulator = evaluator.RequestedRepeatAttention(
          horizon=3,
          intervention_depth=2,
      )

      first_mass = torch.tensor(
          [
              [0.1, 0.2, 0.3, 0.4],
              [0.2, 0.3, 0.1, 0.4],
          ]
      ).reshape(1, 2, 1, 4).expand(3, -1, -1, -1).clone()

      second_mass = torch.tensor(
          [
              [0.2, 0.4, 0.1, 0.3],
              [0.1, 0.5, 0.2, 0.2],
          ]
      ).reshape(1, 2, 1, 4)
      first_value_norm = torch.tensor(
          [
              [1.0, 2.0, 3.0, 4.0],
              [2.0, 4.0, 6.0, 8.0],
          ]
      ).reshape(1, 2, 4, 1).expand(3, -1, -1, -1).clone()
      second_value_norm = torch.tensor(
          [
              [5.0, 6.0, 7.0, 8.0],
              [10.0, 12.0, 14.0, 16.0],
          ]
      ).reshape(1, 2, 4, 1)
      first_contribution = torch.tensor(
          [
              [0.1, 0.2, 0.3, 0.4],
              [0.2, 0.4, 0.6, 0.8],
          ]
      ).reshape(1, 2, 1, 4).expand(3, -1, -1, -1).clone()
      second_contribution = torch.tensor(
          [
              [0.5, 0.6, 0.7, 0.8],
              [1.0, 1.2, 1.4, 1.6],
          ]
      ).reshape(1, 2, 1, 4)

      accumulator.add(
          [
              {
                  "repeat_index": 4,
                  "middle_layer_index": 0,
                  "repeat_mass": first_mass,
                  "value_norm": first_value_norm,
                  "block_contribution_norm": first_contribution,
              }
          ],
          batch_size=3,
      )
      accumulator.add(
          [
              {
                  "repeat_index": 4,
                  "middle_layer_index": 0,
                  "repeat_mass": second_mass,
                  "value_norm": second_value_norm,
                  "block_contribution_norm": second_contribution,
              }
          ],
          batch_size=1,
      )

      row = accumulator.finalize()[0]

      assert row["recall_index"] == 1
      assert row["examples"] == 4
      assert row["positions_per_example"] == 1
      assert row["attention_observations_per_head"] == 4
      assert row["value_observations_per_head_and_cache_block"] == 4
      assert row["contribution_observations_per_head_and_cache_block"] == 4

      assert row["requested_repeat_mass_by_head"] == pytest.approx(
          [0.25, 0.35]
      )
      assert row["requested_repeat_mass_mean"] == pytest.approx(0.30)
      assert row["other_evolution_mass_by_head"] == pytest.approx(
          [0.375, 0.30]
      )
      assert row["recall_block_mass_by_head"] == pytest.approx(
          [0.375, 0.35]
      )

      assert row[
          "requested_mean_rank_among_evolution_by_head"
      ] == pytest.approx([1.75, 1.0])
      assert row[
          "requested_is_best_among_evolution_rate_by_head"
      ] == pytest.approx([0.25, 1.0])

      assert row[
          "requested_mean_rank_among_all_cache_blocks_by_head"
      ] == pytest.approx([2.5, 1.75])
      assert row[
          "requested_is_best_among_all_cache_blocks_rate_by_head"
      ] == pytest.approx([0.25, 0.25])
      assert torch.allclose(
          torch.tensor(row["mean_value_norm_by_head_and_cache_block"]),
          torch.tensor([[2.0, 3.0, 4.0, 5.0], [4.0, 6.0, 8.0, 10.0]]),
      )
      assert row["mean_value_norm_by_cache_block"] == pytest.approx(
          [3.0, 4.5, 6.0, 7.5]
      )
      assert torch.allclose(
          torch.tensor(
              row[
                  "mean_block_contribution_norm_by_head_and_cache_block"
              ]
          ),
          torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.4, 0.6, 0.8, 1.0]]),
      )
      assert row[
          "mean_block_contribution_norm_by_cache_block"
      ] == pytest.approx([0.3, 0.45, 0.6, 0.75])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "value_norm",
            torch.full((1, 1, 2, 1), float("nan")),
            "value_norm must be finite and nonnegative",
        ),
        (
            "block_contribution_norm",
            torch.tensor([[[[-1.0, 1.0]]]]),
            "block_contribution_norm must be finite and nonnegative",
        ),
    ),
)
def test_requested_repeat_attention_rejects_invalid_value_diagnostics(
    field, value, message
):
    diagnostic = {
        "repeat_index": 2,
        "middle_layer_index": 0,
        "repeat_mass": torch.tensor([[[[0.5, 0.5]]]]),
        "value_norm": torch.ones(1, 1, 2, 1),
        "block_contribution_norm": torch.ones(1, 1, 1, 2),
    }
    diagnostic[field] = value

    accumulator = evaluator.RequestedRepeatAttention(
        horizon=1,
        intervention_depth=1,
    )
    with pytest.raises(RuntimeError, match=message):
        accumulator.add([diagnostic], batch_size=1)


def _metadata(condition, *, horizon=6, depth=4, recall_repeats=2, layers=2):
    return {
        "condition": condition,
        "target_repeat": depth,
        "recall_age": horizon - depth,
        "num_evolution_repeats": horizon,
        "num_recall_repeats": recall_repeats,
        "value_permutations": layers if condition == "target-value-corruption" else 0,
        "masked_attention_calls": (
            recall_repeats * layers
            if condition in ("target-repeat-only", "target-repeat-masked")
            else 0
        ),
    }


@pytest.mark.parametrize(
    "condition",
    ["baseline", *evaluator.INTERVENTIONS],
)
def test_validate_intervention_metadata_accepts_expected_model_report(condition):
    evaluator.validate_intervention_metadata(
        [_metadata(condition)],
        condition=condition,
        intervention_depth=4,
        horizon=6,
        num_recall_repeats=2,
        num_middle_layers=2,
    )


def test_validate_intervention_metadata_rejects_wrong_target():
    record = _metadata("target-repeat-masked")
    record["target_repeat"] = 3

    with pytest.raises(RuntimeError, match="target-repeat metadata mismatch"):
        evaluator.validate_intervention_metadata(
            [record],
            condition="target-repeat-masked",
            intervention_depth=4,
            horizon=6,
            num_recall_repeats=2,
            num_middle_layers=2,
        )


def test_compare_logit_captures_reports_logit_and_prediction_deltas():
    sdpa = evaluator.LogitCapture()
    manual = evaluator.LogitCapture()
    sdpa.add(torch.tensor([[[2.0, 1.0], [0.0, 1.0]]]))
    manual.add(torch.tensor([[[2.0, 1.0], [0.0, 1.1]]]))

    comparison = evaluator.compare_logit_captures(sdpa, manual)

    assert comparison["exact_logit_equality"] is False
    assert comparison["logits_within_tolerance"] is False
    assert comparison["max_absolute_logit_delta"] == pytest.approx(0.1)
    assert comparison["mean_absolute_logit_delta"] == pytest.approx(0.025)
    assert comparison["prediction_agreement"] == 1.0


def test_attention_backend_changes_all_stacks_and_restores_them():
    def block(implementation):
        return SimpleNamespace(
            attn=SimpleNamespace(attention_implementation=implementation)
        )

    model = SimpleNamespace(
        transformer=SimpleNamespace(
            h_begin=[block("sdpa")],
            h_mid=[block("manual")],
            h_end=[block("sdpa")],
        )
    )

    with evaluator.attention_backend(model, "manual"):
        assert [
            item.attn.attention_implementation
            for stack in (
                model.transformer.h_begin,
                model.transformer.h_mid,
                model.transformer.h_end,
            )
            for item in stack
        ] == ["manual", "manual", "manual"]

    assert model.transformer.h_begin[0].attn.attention_implementation == "sdpa"
    assert model.transformer.h_mid[0].attn.attention_implementation == "manual"
    assert model.transformer.h_end[0].attn.attention_implementation == "sdpa"


def test_evaluate_interventions_orders_backends_and_pairs_every_condition(
    monkeypatch,
):
    calls = []

    def fake_run_condition(**kwargs):
        condition = kwargs["condition"]
        backend = kwargs["backend"]
        calls.append((backend, condition))
        value = 0.8 if backend == "sdpa" else 0.75
        if condition != "baseline":
            value -= 0.1
        result = {
            "metrics": {field: value for field in evaluator.METRIC_FIELDS},
            "input_trace": {
                "sha256": "same",
                "num_batches": 1,
                "num_examples": 2,
                "batch_shapes": [[2, 4]],
            },
        }
        logits = evaluator.LogitCapture()
        logits.add(torch.zeros(2, 4, 2))
        return result, logits

    monkeypatch.setattr(evaluator, "run_condition", fake_run_condition)
    attention = SimpleNamespace(attention_mode="bidirectional")
    model = SimpleNamespace(
        transformer=SimpleNamespace(
            h_mid=[SimpleNamespace(attn=attention)]
        )
    )

    results = evaluator.evaluate_interventions(
        model=model,
        dataloader=object(),
        model_args=object(),
        device="cpu",
        evaluation_pairs=[(6, 4)],
        num_recall_repeats=2,
        interventions=("target-repeat-masked", "target-repeat-only"),
        permutation_offset=1,
        max_batches=1,
        num_examples=0,
        collect_attention=True,
    )

    assert calls == [
        ("sdpa", "baseline"),
        ("manual", "baseline"),
        ("manual", "target-repeat-masked"),
        ("manual", "target-repeat-only"),
    ]
    result = results[0]
    assert result["horizon"] == 6
    assert result["intervention_depth"] == 4
    assert result["recall_age"] == 2
    assert result["backend_check"]["same_input_trace"] is True
    assert result["backend_check"]["manual_minus_sdpa_metrics"][
        "cell_accuracy"
    ] == pytest.approx(-0.05)
    assert result["effects"]["target-repeat-masked"][
        "intervention_minus_manual_baseline"
    ]["predicted_one_rate"] == pytest.approx(-0.1)


def test_compact_console_report_contains_main_intervention_results(capsys):
      metrics = {
          "cell_accuracy": 0.6,
          "exact_sequence_accuracy": 0.1,
          "predicted_one_rate": 0.4,
          "matthews_correlation": 0.2,
      }
      summary = {
          "protocol": {"attention_mode": "bidirectional"},
          "results": [
              {
                  "horizon": 6,
                  "intervention_depth": 4,
                  "backend_check": {
                      "logits": {
                          "max_absolute_logit_delta": 1e-6,
                          "prediction_agreement": 1.0,
                      }
                  },
                  "conditions": {
                      "baseline": {
                          "metrics": {
                              "cell_accuracy": 0.7,
                          },
                          "internal_consistency": {
                              "decoded_requested_repeat": {
                                  "cell_accuracy": 0.8,
                                  "exact_sequence_accuracy": 0.2,
                              }
                          },
                          "attention": [
                              {
                                  "recall_index": 1,
                                  "middle_layer_index_zero_based": 0,
                                  "requested_repeat_mass_mean": 0.25,
                                  "requested_repeat_mass_by_head": [0.3, 0.2],
                                  "requested_mean_rank_among_evolution_by_head": [
                                      1.5,
                                      2.0,
                                  ],
                                  "requested_is_best_among_evolution_rate_by_head": [
                                      0.7,
                                      0.5,
                                  ],
                                  "requested_is_unique_best_among_evolution_rate_by_head": [
                                      0.6,
                                      0.4,
                                  ],
                                  "requested_mean_rank_among_all_cache_blocks_by_head": [
                                      2.0,
                                      2.5,
                                  ],
                                  "requested_is_best_among_all_cache_blocks_rate_by_head": [
                                      0.5,
                                      0.3,
                                  ],
                                  "requested_is_unique_best_among_all_cache_blocks_rate_by_head": [
                                      0.4,
                                      0.2,
                                  ],
                              }
                          ],
                      },
                      "target-repeat-masked": {
                          "metrics": metrics,
                          "internal_consistency": {
                              "decoded_requested_repeat": {
                                  "cell_accuracy": 0.65,
                                  "exact_sequence_accuracy": 0.1,
                              }
                          },
                          "attention": [
                              {"requested_repeat_mass_mean": 0.0},
                              {"requested_repeat_mass_mean": 0.0},
                          ],
                      },
                  },
                  "effects": {
                      "target-repeat-masked": {
                          "intervention_minus_manual_baseline": {
                              "cell_accuracy": -0.1,
                              "predicted_one_rate": 0.05,
                          }
                      }
                  },
              }
          ],
      }

      evaluator.print_compact_results(summary, "results/summary.json")
      output = capsys.readouterr().out

      assert "Intervention results" in output
      assert "cell_acc" in output
      assert "int_cell" in output
      assert "d_int" in output
      assert "pred_1" in output
      assert "MCC" in output
      assert "target_attn" in output
      assert "target-repeat-masked" in output

      assert "Normal-forward recall attention" in output
      assert "req_mass" in output
      assert "evol_rank" in output
      assert "evol_best" in output
      assert "evol_unique" in output
      assert "all_rank" in output
      assert "all_best" in output
      assert "all_unique" in output

      assert "results/summary.json" in output
