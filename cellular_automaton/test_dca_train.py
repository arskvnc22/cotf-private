from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cellular_automaton.ca_delayed_train import (
    count_delayed_percentage,
    delayed_recall_selection_key,
    delayed_schedule_state_dict,
    format_checkpoint_selections,
    format_delayed_recall_lines,
    make_delayed_pair_counters,
    restore_delayed_pair_counters,
    training_pair_for_step,
)
from cellular_automaton.ca_forward import CAForwardContext
from cellular_automaton.ca_gen import rule30
from cellular_automaton.dca_eval import (
    evaluate_delayed_recall_query,
    summarize_delayed_recall_pairs,
)


class _SameRunOracle(torch.nn.Module):
    """Expose exact per-repeat logits and recall one of those same-run states."""

    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(
            ln_f=torch.nn.Identity(),
            h_end=torch.nn.ModuleList(),
        )
        self.lm_head = torch.nn.Identity()

    def forward(
        self,
        inputs,
        *,
        num_repeats,
        delayed_recall,
        recall_age,
        return_repeat_states,
        get_logits,
        return_all_logits,
    ):
        del delayed_recall, get_logits, return_all_logits
        state = inputs
        initial_logits = F.one_hot(state, num_classes=2).float() * 20.0
        repeat_states = [initial_logits]
        for _ in range(num_repeats):
            state = rule30(state)
            repeat_states.append(
                F.one_hot(state, num_classes=2).float() * 20.0
            )
        query_repeat = num_repeats - recall_age
        return {
            "logits": repeat_states[query_repeat],
            "repeat_states": repeat_states if return_repeat_states else None,
            "average_depth": torch.tensor(float(num_repeats + 1)),
        }


def _metrics(cell_accuracy, exact_accuracy=None, loss=0.5):
    if exact_accuracy is None:
        exact_accuracy = cell_accuracy
    return {
        "loss": loss,
        "cell_accuracy": cell_accuracy,
        "exact_sequence_accuracy": exact_accuracy,
        "mean_bit_errors_per_sequence": 1.0 - cell_accuracy,
        "matthews_correlation": cell_accuracy,
        "average_depth": 1.0,
    }


def _query(num_repeats, query_repeat, accuracy):
    return {
        "num_repeats": num_repeats,
        "query_repeat": query_repeat,
        "recall_age": num_repeats - query_repeat,
        "is_no_op": query_repeat == num_repeats,
        "metrics": _metrics(accuracy),
    }


def _pair(num_repeats, nontrivial_accuracy, internal_accuracy=0.8):
    queries = {
        "query_repeat_1": _query(num_repeats, 1, nontrivial_accuracy),
        f"query_repeat_{num_repeats}": _query(num_repeats, num_repeats, 1.0),
    }
    return {
        "queries": queries,
        "nontrivial_queries_macro": _metrics(nontrivial_accuracy),
        "nontrivial_internal_consistency_macro": {
            "decoded_requested_repeat_cell_accuracy": internal_accuracy,
            "decoded_requested_repeat_exact_sequence_accuracy": internal_accuracy,
            "requested_repeat_logit_cosine_similarity": internal_accuracy,
            "requested_repeat_logit_normalized_mse": 1.0 - internal_accuracy,
            "requested_repeat_is_best_cosine_rate": internal_accuracy,
            "requested_repeat_is_unique_best_cosine_rate": internal_accuracy,
        },
        "nontrivial_ground_truth_retrieval_macro": {
            "requested_repeat_is_best_rate": nontrivial_accuracy,
            "requested_repeat_is_unique_best_rate": nontrivial_accuracy,
            "requested_repeat_mean_rank": 1.0,
            "requested_repeat_mean_reciprocal_rank": nontrivial_accuracy,
            "requested_repeat_mean_margin_over_closest_wrong": 0.1,
            "requested_repeat_positive_margin_rate": nontrivial_accuracy,
        },
    }


def test_delayed_summary_is_pair_balanced_and_excludes_no_op_queries():
    summary = summarize_delayed_recall_pairs(
        {
            "steps_2_repeats_2": _pair(2, 0.5),
            "steps_5_repeats_5": _pair(5, 0.9),
        }
    )

    assert summary["eligible_pairs"] == 2
    assert summary["nontrivial_queries"] == 2
    assert summary["nontrivial_queries_pair_macro"]["cell_accuracy"] == pytest.approx(
        0.7
    )
    assert summary["worst_nontrivial_query"]["metrics"]["cell_accuracy"] == 0.5


def test_delayed_selection_uses_recall_before_normal_accuracy():
    weaker_recall = {
        "nontrivial_queries_pair_macro": _metrics(0.7),
        "worst_nontrivial_query": {"metrics": _metrics(0.7)},
    }
    stronger_recall = {
        "nontrivial_queries_pair_macro": _metrics(0.8),
        "worst_nontrivial_query": {"metrics": _metrics(0.8)},
    }
    perfect_normal = [_metrics(1.0)]
    weak_normal = [_metrics(0.4)]

    assert delayed_recall_selection_key(
        stronger_recall, weak_normal, "cell_accuracy"
    ) > delayed_recall_selection_key(
        weaker_recall, perfect_normal, "cell_accuracy"
    )


def test_same_run_internal_consistency_metrics_compare_requested_repeat():
    inputs = torch.tensor(
        [
            [1, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 0],
        ]
    )
    model = _SameRunOracle()
    model.train()

    result = evaluate_delayed_recall_query(
        CAForwardContext(model),
        [{"input_id": inputs}],
        "cpu",
        num_repeats=3,
        query_repeat=2,
        num_examples=1,
    )

    assert model.training
    assert result["metrics"]["cell_accuracy"] == 1.0
    consistency = result["internal_consistency"]
    assert consistency["decoded_requested_repeat"]["cell_accuracy"] == 1.0
    assert consistency["requested_repeat_logit_similarity"][
        "cosine_similarity"
    ] == pytest.approx(1.0)
    assert consistency["cosine_retrieval"][
        "requested_repeat_is_best_rate"
    ] == 1.0
    assert consistency["cosine_retrieval"][
        "requested_repeat_mean_rank"
    ] == 1.0
    assert consistency["cosine_retrieval"][
        "requested_repeat_mean_reciprocal_rank"
    ] == 1.0
    assert consistency["cosine_retrieval"][
        "requested_repeat_mean_margin_over_closest_wrong"
    ] > 0.0
    assert set(result["ground_truth_comparison_by_repeat"]) == {
        "repeat_1",
        "repeat_2",
        "repeat_3",
    }
    assert result["ground_truth_comparison_by_repeat"]["repeat_2"][
        "matthews_correlation"
    ] == 1.0
    assert set(consistency["decoded_comparison_by_repeat"]) == {
        "repeat_1",
        "repeat_2",
        "repeat_3",
    }
    assert consistency["decoded_comparison_by_repeat"]["repeat_2"][
        "matthews_correlation"
    ] == 1.0
    assert result["ground_truth_state_collision_by_repeat"]["repeat_2"][
        "exact_row_collision_rate"
    ] == 1.0
    assert consistency["decoded_state_collision_by_repeat"]["repeat_2"][
        "exact_row_collision_rate"
    ] == 1.0
    assert result["ground_truth_retrieval"][
        "requested_repeat_is_unique_best_rate"
    ] == 1.0
    assert consistency["decoded_retrieval"][
        "requested_repeat_is_unique_best_rate"
    ] == 1.0
    assert result["examples"][0]["model_requested_repeat_prediction"] == (
        result["examples"][0]["prediction"]
    )
    rendered = format_delayed_recall_lines(
        {
            "steps_3_repeats_3": {
                "queries": {"query_repeat_2": result},
            }
        },
        split="validation",
        step=25,
    )
    assert "DCA recall [validation step=25]" in rendered
    assert "horizon=3 query_repeat=2 recall_age=1 no_op=false" in rendered
    assert "ground_truth cell=1.0000 exact=1.0000 mcc=1.0000" in rendered
    assert "rank=1.0000 mrr=1.0000" in rendered
    assert "internal decoded_cell=1.0000 decoded_rank=1.0000" in rendered
    assert "cosine_rank=1.0000" in rendered


def test_checkpoint_selection_formatter_omits_nested_metadata():
    selection = {
        "step": 250,
        "metric": "cell_accuracy",
        "value": 0.875,
        "checkpoint": "best_delayed_recall.pt",
        "internal_consistency": {"large": {"nested": "payload"}},
    }

    rendered = format_checkpoint_selections(
        ("best_delayed_recall", selection),
        ("best_extrapolation", None),
    )

    assert rendered == (
        "Selected checkpoints | "
        "best_delayed_recall(step=250, metric=cell_accuracy, "
        "value=0.8750, file=best_delayed_recall.pt) | "
        "best_extrapolation=none"
    )
    assert "internal_consistency" not in rendered


def _schedule(args, pairs, counters, start, stop):
    rows = []
    for step in range(start, stop):
        pair = training_pair_for_step(pairs, step)
        rows.append((pair, *count_delayed_percentage(args, pair, counters)))
    return rows


def test_delayed_scheduler_resume_matches_uninterrupted_schedule():
    args = SimpleNamespace(
        ca_delayed_percentage=75,
        query_horizon_policy="uniform",
    )
    pairs = ((1, 1), (3, 3), (5, 5))
    split_step = 11
    final_step = 37

    uninterrupted = make_delayed_pair_counters(pairs)
    expected = _schedule(args, pairs, uninterrupted, 0, final_step)

    interrupted = make_delayed_pair_counters(pairs)
    prefix = _schedule(args, pairs, interrupted, 0, split_step)
    saved = delayed_schedule_state_dict(args, interrupted)
    resumed = restore_delayed_pair_counters(
        args,
        pairs,
        saved,
        start_step=split_step,
    )
    suffix = _schedule(args, pairs, resumed, split_step, final_step)

    assert prefix + suffix == expected
    assert delayed_schedule_state_dict(args, resumed) == (
        delayed_schedule_state_dict(args, uninterrupted)
    )


def test_delayed_scheduler_legacy_replay_and_configuration_validation():
    args = SimpleNamespace(
        ca_delayed_percentage=50,
        query_horizon_policy="uniform",
    )
    pairs = ((2, 2), (4, 4))
    expected = make_delayed_pair_counters(pairs)
    _schedule(args, pairs, expected, 0, 13)

    replayed = restore_delayed_pair_counters(
        args,
        pairs,
        None,
        start_step=13,
    )
    assert replayed == expected

    state = delayed_schedule_state_dict(args, expected)
    changed_args = SimpleNamespace(
        ca_delayed_percentage=75,
        query_horizon_policy="uniform",
    )
    with pytest.raises(ValueError, match="different ca_delayed_percentage"):
        restore_delayed_pair_counters(
            changed_args,
            pairs,
            state,
            start_step=13,
        )


@pytest.mark.parametrize("delayed_percentage", [0, 1, 33, 50, 75, 100])
def test_legacy_scheduler_reconstruction_matches_round_robin(delayed_percentage):
    args = SimpleNamespace(
        ca_delayed_percentage=delayed_percentage,
        query_horizon_policy="uniform",
    )
    pairs = ((1, 1), (3, 3), (5, 5))
    expected = make_delayed_pair_counters(pairs)
    _schedule(args, pairs, expected, 0, 137)

    assert restore_delayed_pair_counters(
        args,
        pairs,
        None,
        start_step=137,
    ) == expected
