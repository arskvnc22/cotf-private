"""Evaluation metrics for cellular-automaton row prediction."""

import math
import inspect
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .ca_forward import CAForwardContext
    from .ca_gen import apply_rule30, rule30
except ImportError:
    from ca_forward import CAForwardContext
    from ca_gen import apply_rule30, rule30


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return math.nan
    return numerator / denominator


def _matthews_correlation(true_negatives, false_positives, false_negatives, true_positives):
    """Return binary MCC, using zero for a degenerate constant-class table."""
    denominator = math.sqrt(
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )
    if denominator == 0:
        return 0.0
    return (
        true_positives * true_negatives - false_positives * false_negatives
    ) / denominator


def new_ca_counters():
    """Create integer counters used to aggregate an entire evaluation split."""
    return {
        "num_batches": 0,
        "sequences": 0,
        "exact_sequences": 0,
        "cells": 0,
        "correct_cells": 0,
        "bit_errors": 0,
        "boundary_cells": 0,
        "correct_boundary_cells": 0,
        "interior_cells": 0,
        "correct_interior_cells": 0,
        "target_zeros": 0,
        "correct_target_zeros": 0,
        "target_ones": 0,
        "correct_target_ones": 0,
        "predicted_ones": 0,
        "position_correct": None,
        "position_total": None,
        "weighted_loss": 0.0,
        "loss_weight": 0,
        "average_depth_sum": 0.0,
        "average_depth_count": 0,
    }


def update_ca_counters(counters, logits, labels, *, loss=None, average_depth=None):
    """Accumulate metrics from one batch of aligned per-cell predictions."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, N, C], got {tuple(logits.shape)}.")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape [B, N], got {tuple(labels.shape)}.")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "Logit and label positions do not align: "
            f"{tuple(logits.shape[:2])} versus {tuple(labels.shape)}."
        )
    if logits.shape[-1] != 2:
        raise ValueError(f"Rule 30 evaluation expects 2 output classes, got {logits.shape[-1]}.")
    if labels.numel() == 0:
        raise ValueError("Cannot evaluate an empty batch.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Rule 30 labels must contain only 0 and 1.")

    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(labels)
    batch_size, num_cells = labels.shape

    counters["num_batches"] += 1
    counters["sequences"] += batch_size
    counters["exact_sequences"] += int(correct.all(dim=-1).sum().item())
    counters["cells"] += labels.numel()
    batch_correct = int(correct.sum().item())
    counters["correct_cells"] += batch_correct
    counters["bit_errors"] += labels.numel() - batch_correct

    boundary_positions = [0] if num_cells == 1 else [0, num_cells - 1]
    boundary_correct = correct[:, boundary_positions]
    counters["boundary_cells"] += boundary_correct.numel()
    counters["correct_boundary_cells"] += int(boundary_correct.sum().item())

    if num_cells > 2:
        interior_correct = correct[:, 1:-1]
        counters["interior_cells"] += interior_correct.numel()
        counters["correct_interior_cells"] += int(interior_correct.sum().item())

    target_zeros = labels.eq(0)
    target_ones = labels.eq(1)
    counters["target_zeros"] += int(target_zeros.sum().item())
    counters["correct_target_zeros"] += int((correct & target_zeros).sum().item())
    counters["target_ones"] += int(target_ones.sum().item())
    counters["correct_target_ones"] += int((correct & target_ones).sum().item())
    counters["predicted_ones"] += int(predictions.eq(1).sum().item())

    position_correct = correct.sum(dim=0).detach().cpu().to(torch.long)
    if counters["position_correct"] is None:
        counters["position_correct"] = torch.zeros(num_cells, dtype=torch.long)
        counters["position_total"] = torch.zeros(num_cells, dtype=torch.long)
    elif counters["position_correct"].numel() != num_cells:
        raise ValueError("One evaluation split cannot mix different row lengths.")
    counters["position_correct"] += position_correct
    counters["position_total"] += batch_size

    if loss is not None:
        loss_value = float(torch.as_tensor(loss).detach().cpu().float().item())
        counters["weighted_loss"] += loss_value * labels.numel()
        counters["loss_weight"] += labels.numel()

    if average_depth is not None:
        depth_value = float(torch.as_tensor(average_depth).detach().cpu().float().item())
        counters["average_depth_sum"] += depth_value * batch_size
        counters["average_depth_count"] += batch_size


def finalize_ca_metrics(counters):
    """Convert accumulated counts into a JSON-friendly metric dictionary."""
    sequences = counters["sequences"]
    cells = counters["cells"]
    position_correct = counters["position_correct"]
    position_total = counters["position_total"]

    position_accuracy = []
    if position_correct is not None:
        position_accuracy = [
            _safe_ratio(int(correct), int(total))
            for correct, total in zip(position_correct, position_total)
        ]

    true_negatives = counters["correct_target_zeros"]
    true_positives = counters["correct_target_ones"]
    false_positives = counters["target_zeros"] - true_negatives
    false_negatives = counters["target_ones"] - true_positives

    return {
        "loss": _safe_ratio(counters["weighted_loss"], counters["loss_weight"]),
        "cell_accuracy": _safe_ratio(counters["correct_cells"], cells),
        "correct_cells": counters["correct_cells"],
        "total_cells": cells,
        "exact_sequence_accuracy": _safe_ratio(counters["exact_sequences"], sequences),
        "exact_sequences": counters["exact_sequences"],
        "total_sequences": sequences,
        "mean_bit_errors_per_sequence": _safe_ratio(counters["bit_errors"], sequences),
        "total_bit_errors": counters["bit_errors"],
        "boundary_accuracy": _safe_ratio(
            counters["correct_boundary_cells"], counters["boundary_cells"]
        ),
        "interior_accuracy": _safe_ratio(
            counters["correct_interior_cells"], counters["interior_cells"]
        ),
        "zero_accuracy": _safe_ratio(
            counters["correct_target_zeros"], counters["target_zeros"]
        ),
        "one_accuracy": _safe_ratio(
            counters["correct_target_ones"], counters["target_ones"]
        ),
        "target_one_rate": _safe_ratio(counters["target_ones"], cells),
        "predicted_one_rate": _safe_ratio(counters["predicted_ones"], cells),
        "matthews_correlation": _matthews_correlation(
            true_negatives,
            false_positives,
            false_negatives,
            true_positives,
        ),
        "position_accuracy": position_accuracy,
        "average_depth": _safe_ratio(
            counters["average_depth_sum"], counters["average_depth_count"]
        ),
        "num_batches": counters["num_batches"],
    }


@torch.no_grad()
def evaluate_ca_model(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    max_batches=None,
    ctx=None,
):
    """Evaluate one fixed-length CA split and restore the model's prior mode."""
    model = forward_context.model
    was_training = model.training
    model.eval()
    counters = new_ca_counters()
    ctx = ctx or nullcontext()

    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["input_id"].to(
                device, dtype=torch.long, non_blocking=True
            )
            labels = batch["label"].to(
                device, dtype=torch.long, non_blocking=True
            )
            with ctx:
                outputs = forward_context.call(
                    inputs, targets=labels, get_logits=True
                )

            if not isinstance(outputs, dict):
                raise TypeError("CA models must return a dictionary of outputs.")
            logits = outputs.get("logits")
            if logits is None:
                raise KeyError("Model output is missing logits; use get_logits=True.")
            loss = outputs.get("cross_entropy_loss", outputs.get("loss"))
            update_ca_counters(
                counters,
                logits,
                labels,
                loss=loss,
                average_depth=outputs.get("average_depth"),
            )
    finally:
        if was_training:
            model.train()

    if counters["num_batches"] == 0:
        raise ValueError("Evaluation consumed no batches.")
    return finalize_ca_metrics(counters)


def evaluate_ca_lengths(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    *,
    max_batches=None,
    ctx=None,
):
    """Evaluate every configured row length independently."""
    return {
        num_cells: evaluate_ca_model(
            forward_context,
            dataloader,
            device,
            max_batches=max_batches,
            ctx=ctx,
        )
        for num_cells, dataloader in eval_loaders.items()
    }
def _bit_string(row):
    """Convert one binary tensor row into a readable string."""
    return "".join(str(int(value)) for value in row.tolist())


def _new_similarity_accumulator():
    return {
        "sequences": 0,
        "cells": 0,
        "equal_cells": 0,
        "cosine_sum": 0.0,
        "normalized_mse_sum": 0.0,
    }


def _update_similarity(accumulator, left, right, *, decoded=False):
    """Aggregate per-sequence cosine/NMSE and optional decoded cell agreement."""
    if left.shape != right.shape:
        raise ValueError(
            f"Cannot compare states with shapes {tuple(left.shape)} and {tuple(right.shape)}."
        )
    batch_size = left.shape[0]
    left_float = left.detach().float().reshape(batch_size, -1)
    right_float = right.detach().float().reshape(batch_size, -1)
    if decoded:
        # Bipolar binary rows avoid the undefined cosine of an all-zero row.
        left_float = left_float.mul(2.0).sub(1.0)
        right_float = right_float.mul(2.0).sub(1.0)
        accumulator["cells"] += left.numel()
        accumulator["equal_cells"] += int(left.eq(right).sum().item())

    cosine = F.cosine_similarity(left_float, right_float, dim=-1, eps=1e-8)
    squared_error = (left_float - right_float).pow(2).mean(dim=-1)
    reference_energy = 0.5 * (
        left_float.pow(2).mean(dim=-1) + right_float.pow(2).mean(dim=-1)
    )
    normalized_mse = squared_error / reference_energy.clamp_min(1e-8)

    accumulator["sequences"] += batch_size
    accumulator["cosine_sum"] += float(cosine.sum().item())
    accumulator["normalized_mse_sum"] += float(normalized_mse.sum().item())


def _finalize_similarity(accumulator, *, decoded=False):
    sequences = accumulator["sequences"]
    result = {
        "cosine_similarity": _safe_ratio(accumulator["cosine_sum"], sequences),
        "normalized_mse": _safe_ratio(
            accumulator["normalized_mse_sum"], sequences
        ),
        "total_sequences": sequences,
    }
    if decoded:
        result["cell_agreement"] = _safe_ratio(
            accumulator["equal_cells"], accumulator["cells"]
        )
    return result


def _decode_dca_repeat_states(model, repeat_states, num_repeats):
    """Decode states captured during one DCA run with its shared output head."""
    if repeat_states is None or len(repeat_states) != num_repeats + 1:
        raise ValueError(
            "Delayed-recall consistency requires the post-begin state followed "
            "by one state per evolution repeat."
        )
    transformer = getattr(model, "transformer", None)
    lm_head = getattr(model, "lm_head", None)
    if transformer is None or lm_head is None:
        raise TypeError(
            "Delayed-recall consistency requires transformer.ln_f and lm_head."
        )
    if len(transformer.h_end) != 0:
        raise ValueError(
            "Same-run per-repeat decoding currently requires n_layer_end == 0."
        )
    return {
        repeat: lm_head(transformer.ln_f(repeat_states[repeat]))
        for repeat in range(1, num_repeats + 1)
    }

def _probe_lstm_memory_snapshot(model, cell, reference):
    """
    Decode one LSTM cell snapshot with a fixed, parameter-free probe.

    The LSTM block exposes tanh(cell) to the hidden stream, so the closest
    existing decoder is lm_head(ln_f(tanh(cell))). This head was not trained
    directly on isolated cell states; its output is therefore a diagnostic
    probe rather than a guaranteed semantic decoding.
    """
    if cell.ndim != 3:
        raise ValueError(
            "LSTM memory diagnostics expect [batch, cells, embedding] cells."
        )
    if reference.shape != cell.shape[:2]:
        raise ValueError(
            "The memory reference must align with the batch and cell axes."
        )

    transformer = getattr(model, "transformer", None)
    lm_head = getattr(model, "lm_head", None)
    if transformer is None or lm_head is None:
        raise TypeError(
            "LSTM memory decoding requires transformer.ln_f and lm_head."
        )

    cell_fp32 = cell.detach().float()
    exposed_memory = torch.tanh(cell_fp32)

    # Model parameters remain FP32 in the current mixed-precision setup.
    probe_input = exposed_memory.to(dtype=lm_head.weight.dtype)
    decoded_logits = lm_head(transformer.ln_f(probe_input)).float()
    decoded_predictions = decoded_logits.argmax(dim=-1)
    decoded_probabilities = decoded_logits.softmax(dim=-1)

    correct = decoded_predictions.eq(reference)
    batch_size = reference.shape[0]

    summary = {
        "cell_mean": float(cell_fp32.mean().item()),
        "cell_std": float(
            cell_fp32.std(unbiased=False).item()
        ),
        "cell_rms": float(
            cell_fp32.square().mean().sqrt().item()
        ),
        "cell_abs_max": float(cell_fp32.abs().max().item()),
        "exposed_saturation_fraction": float(
            exposed_memory.abs().gt(0.99).float().mean().item()
        ),
        "decoded_cell_accuracy": float(
            correct.float().mean().item()
        ),
        "decoded_exact_sequence_accuracy": float(
            correct.all(dim=1).float().mean().item()
        ),
        "decoded_mean_confidence": float(
            decoded_probabilities.max(dim=-1).values.mean().item()
        ),
        "examples": [
            {
                "decoded": _bit_string(decoded_predictions[index]),
                "reference": _bit_string(reference[index]),
            }
            for index in range(batch_size)
        ],
    }

    artifact = {
        "cell": cell_fp32.cpu(),
        "exposed_memory": exposed_memory.cpu(),
        "decoded_logits": decoded_logits.detach().cpu(),
        "decoded_predictions": decoded_predictions.detach().cpu(),
        "reference": reference.detach().cpu(),
    }

    return summary, artifact


@torch.no_grad()
def save_lstm_memory_diagnostics(
    forward_context,
    dataloader,
    device,
    *,
    num_repeats,
    num_recall_repeats,
    evolution_repeats,
    recall_steps,
    num_examples,
    artifact_path,
    query_repeats=None,
    ctx=None,
):
    """
    Save raw shared-cell snapshots and return a JSON-compatible summary.

    Only the first fixed validation batch and the requested number of examples
    are retained. Evolution snapshots are stored once because they are
    independent of the subsequently requested recall age.
    """
    evolution_repeats = tuple(evolution_repeats)
    recall_steps = tuple(recall_steps)

    if not evolution_repeats and not recall_steps:
        raise ValueError(
            "At least one evolution repeat or recall step is required."
        )
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")

    if query_repeats is None:
        query_repeats = tuple(range(1, num_repeats + 1))
    else:
        query_repeats = tuple(query_repeats)

    model = forward_context.model
    was_training = model.training
    model.eval()
    ctx = ctx or nullcontext()

    try:
        try:
            batch = next(iter(dataloader))
        except StopIteration as error:
            raise ValueError(
                "LSTM memory diagnostics received an empty dataloader."
            ) from error

        inputs = batch["input_id"][:num_examples].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        true_states_by_repeat = {}
        state = inputs
        for repeat in range(1, num_repeats + 1):
            state = rule30(state)
            true_states_by_repeat[repeat] = state

        artifact = {
            "schema_version": 1,
            "probe": "lm_head(ln_f(tanh(cell)))",
            "num_repeats": int(num_repeats),
            "num_recall_repeats": int(num_recall_repeats),
            "input_ids": inputs.detach().cpu(),
            "true_states_by_repeat": {
                f"repeat_{repeat}": state.detach().cpu()
                for repeat, state in true_states_by_repeat.items()
            },
            "evolution": {},
            "recall_by_query": {},
        }
        summary = {
            "probe": "lm_head(ln_f(tanh(cell)))",
            "num_examples": int(inputs.shape[0]),
            "num_repeats": int(num_repeats),
            "num_recall_repeats": int(num_recall_repeats),
            "evolution": {},
            "recall_by_query": {},
        }

        for query_index, query_repeat in enumerate(query_repeats):
            recall_age = num_repeats - query_repeat

            with ctx:
                _, _, outputs = _forward_all_cells(
                    forward_context,
                    inputs,
                    num_repeats=num_repeats,
                    delayed_recall=True,
                    recall_age=recall_age,
                    num_recall_repeats=num_recall_repeats,
                    return_outputs=True,
                    return_memory_states=True,
                    memory_evolution_repeats=evolution_repeats,
                    memory_recall_steps=recall_steps,
                )

            memory_states = outputs.get("memory_states")
            if not isinstance(memory_states, dict):
                raise KeyError(
                    "DCA LSTM output is missing memory_states."
                )

            # Evolution is deterministic and independent of the later query.
            # Store it once rather than duplicating it for every recall age.
            if query_index == 0:
                for repeat in evolution_repeats:
                    cell = memory_states["evolution"].get(repeat)
                    if cell is None:
                        raise KeyError(
                            f"Missing evolution memory snapshot {repeat}."
                        )

                    snapshot_summary, snapshot_artifact = (
                        _probe_lstm_memory_snapshot(
                            model,
                            cell,
                            true_states_by_repeat[repeat],
                        )
                    )
                    key = f"repeat_{repeat}"
                    summary["evolution"][key] = snapshot_summary
                    artifact["evolution"][key] = snapshot_artifact

            query_key = f"query_repeat_{query_repeat}"
            target = true_states_by_repeat[query_repeat]

            summary_query = {
                "query_repeat": int(query_repeat),
                "recall_age": int(recall_age),
                "steps": {},
            }
            artifact_query = {
                "query_repeat": int(query_repeat),
                "recall_age": int(recall_age),
                "target": target.detach().cpu(),
                "steps": {},
            }

            for recall_step in recall_steps:
                cell = memory_states["recall"].get(recall_step)
                if cell is None:
                    raise KeyError(
                        f"Missing recall memory snapshot {recall_step}."
                    )

                snapshot_summary, snapshot_artifact = (
                    _probe_lstm_memory_snapshot(
                        model,
                        cell,
                        target,
                    )
                )
                step_key = f"recall_step_{recall_step}"
                summary_query["steps"][step_key] = snapshot_summary
                artifact_query["steps"][step_key] = snapshot_artifact

            summary["recall_by_query"][query_key] = summary_query
            artifact["recall_by_query"][query_key] = artifact_query

    finally:
        if was_training:
            model.train()

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)

    summary["artifact_path"] = str(artifact_path)
    return summary

def _new_ranking_accumulator():
    return {
        "sequences": 0,
        "requested_is_best": 0,
        "requested_is_unique_best": 0,
        "rank_sum": 0.0,
        "reciprocal_rank_sum": 0.0,
        "margin_sum": 0.0,
        "margin_count": 0,
        "positive_margin": 0,
    }


def _update_ranking(accumulator, scores, requested_index):
    """Accumulate tie-aware ranks for a [batch, candidate] score matrix."""
    if scores.ndim != 2:
        raise ValueError("Ranking scores must have shape [batch, candidates].")
    if not 0 <= requested_index < scores.shape[1]:
        raise ValueError("Requested ranking index is outside the candidates.")
    requested = scores[:, requested_index]
    tied_with_requested = torch.isclose(
        scores,
        requested.unsqueeze(1),
        rtol=1e-5,
        atol=1e-7,
    )
    strictly_better = (scores > requested.unsqueeze(1)) & ~tied_with_requested
    ranks = 1 + strictly_better.sum(dim=1)
    maxima = scores.max(dim=1, keepdim=True).values
    best_mask = torch.isclose(scores, maxima, rtol=1e-5, atol=1e-7)
    requested_is_best = best_mask[:, requested_index]
    requested_is_unique_best = requested_is_best & (
        best_mask.sum(dim=1) == 1
    )

    batch_size = scores.shape[0]
    accumulator["sequences"] += batch_size
    accumulator["requested_is_best"] += int(requested_is_best.sum().item())
    accumulator["requested_is_unique_best"] += int(
        requested_is_unique_best.sum().item()
    )
    accumulator["rank_sum"] += float(ranks.float().sum().item())
    accumulator["reciprocal_rank_sum"] += float(
        ranks.float().reciprocal().sum().item()
    )

    if scores.shape[1] > 1:
        wrong_mask = torch.ones_like(scores, dtype=torch.bool)
        wrong_mask[:, requested_index] = False
        closest_wrong = scores.masked_fill(~wrong_mask, -torch.inf).max(dim=1).values
        margins = requested - closest_wrong
        accumulator["margin_sum"] += float(margins.sum().item())
        accumulator["margin_count"] += batch_size
        accumulator["positive_margin"] += int((margins > 1e-7).sum().item())


def _finalize_ranking(accumulator):
    sequences = accumulator["sequences"]
    margin_count = accumulator["margin_count"]
    return {
        "requested_repeat_is_best_rate": _safe_ratio(
            accumulator["requested_is_best"], sequences
        ),
        "requested_repeat_is_unique_best_rate": _safe_ratio(
            accumulator["requested_is_unique_best"], sequences
        ),
        "requested_repeat_mean_rank": _safe_ratio(
            accumulator["rank_sum"], sequences
        ),
        "requested_repeat_mean_reciprocal_rank": _safe_ratio(
            accumulator["reciprocal_rank_sum"], sequences
        ),
        "requested_repeat_mean_margin_over_closest_wrong": (
            accumulator["margin_sum"] / margin_count
            if margin_count
            else None
        ),
        "requested_repeat_positive_margin_rate": (
            accumulator["positive_margin"] / margin_count
            if margin_count
            else None
        ),
        "requested_repeat_is_best_sequences": accumulator[
            "requested_is_best"
        ],
        "requested_repeat_is_unique_best_sequences": accumulator[
            "requested_is_unique_best"
        ],
        "total_sequences": sequences,
    }


def _ranking_example(scores, requested_repeat):
    values = [float(value) for value in scores.detach().cpu().tolist()]
    requested = values[requested_repeat - 1]
    best = max(values)
    best_repeats = [
        repeat
        for repeat, value in enumerate(values, start=1)
        if math.isclose(value, best, rel_tol=1e-5, abs_tol=1e-7)
    ]
    rank = 1 + sum(
        value > requested
        and not math.isclose(value, requested, rel_tol=1e-5, abs_tol=1e-7)
        for value in values
    )
    wrong = [
        value
        for repeat, value in enumerate(values, start=1)
        if repeat != requested_repeat
    ]
    return {
        "scores_by_repeat": {
            f"repeat_{repeat}": value
            for repeat, value in enumerate(values, start=1)
        },
        "best_repeats": best_repeats,
        "requested_repeat_rank": rank,
        "requested_repeat_reciprocal_rank": 1.0 / rank,
        "requested_repeat_margin_over_closest_wrong": (
            requested - max(wrong) if wrong else None
        ),
    }


def _new_decoded_agreement_accumulator():
    return {
        "sequences": 0,
        "exact_sequences": 0,
        "cells": 0,
        "equal_cells": 0,
    }


def _update_decoded_agreement(accumulator, left, right):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Decoded agreement expects aligned [batch, cells] rows.")
    equal = left.eq(right)
    accumulator["sequences"] += left.shape[0]
    accumulator["exact_sequences"] += int(equal.all(dim=1).sum().item())
    accumulator["cells"] += left.numel()
    accumulator["equal_cells"] += int(equal.sum().item())


def _finalize_decoded_agreement(accumulator):
    return {
        "cell_agreement": _safe_ratio(
            accumulator["equal_cells"], accumulator["cells"]
        ),
        "exact_row_collision_rate": _safe_ratio(
            accumulator["exact_sequences"], accumulator["sequences"]
        ),
        "exact_row_collisions": accumulator["exact_sequences"],
        "total_sequences": accumulator["sequences"],
    }

def _forward_all_cells(
    forward_context: CAForwardContext,
    inputs,
    *,
    num_repeats=None,
    delayed_recall=False,
    recall_age=None,
    return_repeat_states=False,
    return_outputs=False,
    intervention_source_depth=None,
    intervention_input_ids=None,
    num_recall_repeats=1,
    return_memory_states=False,
    memory_evolution_repeats=None,
    memory_recall_steps=None,

):
    forward_kwargs = {
        "get_logits": True,
        "return_all_logits": True,
    }

    if num_repeats is not None:
        forward_kwargs["num_repeats"] = num_repeats

    if delayed_recall:
        if recall_age is None:
            raise ValueError(
                "recall_age is required for delayed recall evaluation."
            )
        if num_recall_repeats <= 0:
            raise ValueError("num_recall_repeats must be positive.")

        forward_kwargs["delayed_recall"] = True
        forward_kwargs["recall_age"] = recall_age
        forward_kwargs["num_recall_repeats"] = num_recall_repeats

    elif recall_age is not None:
        raise ValueError(
            "recall_age must be None when delayed_recall=False."
        )

    if return_repeat_states:
        forward_kwargs["return_repeat_states"] = True
    if return_memory_states:
        forward_kwargs["return_memory_states"] = True
        forward_kwargs["memory_evolution_repeats"] = (
            memory_evolution_repeats
        )
        forward_kwargs["memory_recall_steps"] = memory_recall_steps
    elif (
        memory_evolution_repeats is not None
        or memory_recall_steps is not None
    ):
        raise ValueError(
            "Memory snapshot selections require return_memory_states=True."
        )

    if intervention_source_depth is not None or intervention_input_ids is not None:
        forward_kwargs["intervention_source_depth"] = intervention_source_depth
        forward_kwargs["intervention_input_ids"] = intervention_input_ids

    try:
        outputs = forward_context.call(
            inputs,
            **forward_kwargs,
        )
    except TypeError as error:
        if "return_all_logits" in str(error):
            forward_kwargs.pop("return_all_logits")
            outputs = forward_context.call(
                inputs,
                **forward_kwargs,
            )
        elif num_repeats is not None and "num_repeats" in str(error):
            raise TypeError(
                "Internal-repeat evaluation requires the model "
                "to accept num_repeats."
            ) from error
        else:
            raise

    if not isinstance(outputs, dict):
        raise TypeError(
            "CA models must return a dictionary of outputs."
        )

    logits = outputs.get("logits")

    if logits is None:
        raise KeyError(
            "Model output is missing logits; use get_logits=True."
        )

    expected_shape = (*inputs.shape, 2)

    if tuple(logits.shape) != expected_shape:
        retry_kwargs = dict(forward_kwargs)
        retry_kwargs["targets"] = inputs

        outputs = forward_context.call(
            inputs,
            **retry_kwargs,
        )
        logits = outputs.get("logits")

    if logits is None or tuple(logits.shape) != expected_shape:
        raise ValueError(
            "CA inference requires logits for every cell: "
            f"expected {expected_shape}, got "
            f"{None if logits is None else tuple(logits.shape)}."
        )

    result = (
        logits,
        outputs.get("average_depth"),
    )

    if return_outputs:
        return (*result, outputs)

    return result
@torch.no_grad()
def evaluate_delayed_recall_query(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    num_repeats,
    query_repeat,
    num_recall_repeats=1,

    max_batches=None,
    num_examples=2,
    ctx=None,
):
    """
    Evaluate one requested recall target.

    Example:
        num_repeats=5
        query_repeat=3
        recall_age=2
        target=Rule30(input, steps=3)
    """
    if num_repeats <= 0:
        raise ValueError("num_repeats must be positive.")

    if not 1 <= query_repeat <= num_repeats:
        raise ValueError(
            "query_repeat must be between 1 and num_repeats."
        )

    if num_examples < 0:
        raise ValueError("num_examples cannot be negative.")

    recall_age = num_repeats - query_repeat

    model = forward_context.model
    was_training = model.training
    model.eval()

    counters = new_ca_counters()
    internal_decoded_counters = new_ca_counters()
    ground_truth_comparison_counters = {
        repeat: new_ca_counters()
        for repeat in range(1, num_repeats + 1)
    }
    internal_decoded_comparison_counters = {
        repeat: new_ca_counters()
        for repeat in range(1, num_repeats + 1)
    }
    ground_truth_collision = {
        repeat: _new_decoded_agreement_accumulator()
        for repeat in range(1, num_repeats + 1)
    }
    internal_decoded_collision = {
        repeat: _new_decoded_agreement_accumulator()
        for repeat in range(1, num_repeats + 1)
    }
    logit_similarity = {
        repeat: _new_similarity_accumulator()
        for repeat in range(1, num_repeats + 1)
    }
    ground_truth_ranking = _new_ranking_accumulator()
    internal_decoded_ranking = _new_ranking_accumulator()
    cosine_ranking = _new_ranking_accumulator()
    examples = []
    ctx = ctx or nullcontext()

    try:
        for batch_index, batch in enumerate(dataloader):
            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            inputs = batch["input_id"].to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )

            # Generate every true CA state so we can determine whether an
            # incorrect prediction resembles a different repeat.
            states_by_repeat = {}
            state = inputs

            for repeat in range(1, num_repeats + 1):
                state = rule30(state)
                states_by_repeat[repeat] = state

            targets = states_by_repeat[query_repeat]

            with ctx:
                logits, average_depth, outputs = _forward_all_cells(
                    forward_context,
                    inputs,
                    num_repeats=num_repeats,
                    delayed_recall=True,
                    recall_age=recall_age,
                    num_recall_repeats=num_recall_repeats,
                    return_repeat_states=True,
                    return_outputs=True,
                )

            repeat_logits = _decode_dca_repeat_states(
                model,
                outputs.get("repeat_states"),
                num_repeats,
            )
            requested_repeat_logits = repeat_logits[query_repeat]
            internal_predictions = {
                repeat: candidate_logits.argmax(dim=-1)
                for repeat, candidate_logits in repeat_logits.items()
            }
            requested_repeat_predictions = internal_predictions[query_repeat]

            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                targets.reshape(-1),
            )

            update_ca_counters(
                counters,
                logits,
                targets,
                loss=loss,
                average_depth=average_depth,
            )

            for repeat, candidate_target in states_by_repeat.items():
                candidate_loss = F.cross_entropy(
                    logits.reshape(-1, 2),
                    candidate_target.reshape(-1),
                )
                update_ca_counters(
                    ground_truth_comparison_counters[repeat],
                    logits,
                    candidate_target,
                    loss=candidate_loss,
                    average_depth=average_depth,
                )
                _update_decoded_agreement(
                    ground_truth_collision[repeat],
                    targets,
                    candidate_target,
                )

            internal_loss = F.cross_entropy(
                logits.reshape(-1, 2),
                requested_repeat_predictions.reshape(-1),
            )
            update_ca_counters(
                internal_decoded_counters,
                logits,
                requested_repeat_predictions,
                loss=internal_loss,
                average_depth=average_depth,
            )

            for repeat, candidate_prediction in internal_predictions.items():
                candidate_loss = F.cross_entropy(
                    logits.reshape(-1, 2),
                    candidate_prediction.reshape(-1),
                )
                update_ca_counters(
                    internal_decoded_comparison_counters[repeat],
                    logits,
                    candidate_prediction,
                    loss=candidate_loss,
                    average_depth=average_depth,
                )
                _update_decoded_agreement(
                    internal_decoded_collision[repeat],
                    requested_repeat_predictions,
                    candidate_prediction,
                )

            predictions = logits.argmax(dim=-1)
            ground_truth_scores = torch.stack(
                [
                    predictions.eq(states_by_repeat[repeat]).float().mean(dim=1)
                    for repeat in range(1, num_repeats + 1)
                ],
                dim=1,
            )
            internal_decoded_scores = torch.stack(
                [
                    predictions
                    .eq(internal_predictions[repeat])
                    .float()
                    .mean(dim=1)
                    for repeat in range(1, num_repeats + 1)
                ],
                dim=1,
            )
            _update_ranking(
                ground_truth_ranking,
                ground_truth_scores,
                query_repeat - 1,
            )
            _update_ranking(
                internal_decoded_ranking,
                internal_decoded_scores,
                query_repeat - 1,
            )

            per_repeat_cosine = []
            for repeat in range(1, num_repeats + 1):
                _update_similarity(
                    logit_similarity[repeat],
                    logits,
                    repeat_logits[repeat],
                )
                per_repeat_cosine.append(
                    F.cosine_similarity(
                        logits.detach().float().reshape(inputs.shape[0], -1),
                        repeat_logits[repeat]
                        .detach()
                        .float()
                        .reshape(inputs.shape[0], -1),
                        dim=-1,
                        eps=1e-8,
                    )
                )
            cosine_matrix = torch.stack(per_repeat_cosine, dim=1)
            _update_ranking(
                cosine_ranking,
                cosine_matrix,
                query_repeat - 1,
            )
            probabilities = logits.softmax(dim=-1)

            remaining_examples = num_examples - len(examples)
            batch_examples = min(
                remaining_examples,
                inputs.shape[0],
            )

            for example_index in range(batch_examples):
                prediction = predictions[example_index]
                target = targets[example_index]
                internal_target = requested_repeat_predictions[example_index]

                correct_mask = prediction.eq(target)
                correct_cells = int(correct_mask.sum().item())
                total_cells = int(target.numel())
                internal_correct_mask = prediction.eq(internal_target)
                internal_correct_cells = int(internal_correct_mask.sum().item())

                target_probabilities = probabilities[
                    example_index
                ].gather(
                    dim=-1,
                    index=target.unsqueeze(-1),
                ).squeeze(-1)

                matches_by_repeat = {}

                for repeat, true_state in states_by_repeat.items():
                    repeat_target = true_state[example_index]
                    repeat_correct = int(
                        prediction.eq(repeat_target).sum().item()
                    )

                    matches_by_repeat[f"repeat_{repeat}"] = {
                        "correct_cells": repeat_correct,
                        "total_cells": total_cells,
                        "cell_accuracy": (
                            repeat_correct / total_cells
                        ),
                        "state": _bit_string(repeat_target),
                    }

                best_accuracy = max(
                    match["cell_accuracy"]
                    for match in matches_by_repeat.values()
                )

                best_matching_repeats = [
                    repeat
                    for repeat in range(1, num_repeats + 1)
                    if matches_by_repeat[
                        f"repeat_{repeat}"
                    ]["cell_accuracy"] == best_accuracy
                ]

                internal_matches_by_repeat = {}
                for repeat, candidate_prediction in internal_predictions.items():
                    candidate_row = candidate_prediction[example_index]
                    repeat_correct = int(
                        prediction.eq(candidate_row).sum().item()
                    )
                    internal_matches_by_repeat[f"repeat_{repeat}"] = {
                        "correct_cells": repeat_correct,
                        "total_cells": total_cells,
                        "cell_accuracy": repeat_correct / total_cells,
                        "state": _bit_string(candidate_row),
                    }

                example_cosines = {
                    f"repeat_{repeat}": float(
                        cosine_matrix[example_index, repeat - 1].item()
                    )
                    for repeat in range(1, num_repeats + 1)
                }
                best_internal_cosine = max(example_cosines.values())
                best_internal_repeats = [
                    repeat
                    for repeat in range(1, num_repeats + 1)
                    if math.isclose(
                        example_cosines[f"repeat_{repeat}"],
                        best_internal_cosine,
                        rel_tol=1e-5,
                        abs_tol=1e-7,
                    )
                ]
                ground_truth_rank = _ranking_example(
                    ground_truth_scores[example_index],
                    query_repeat,
                )
                internal_decoded_rank = _ranking_example(
                    internal_decoded_scores[example_index],
                    query_repeat,
                )
                cosine_rank = _ranking_example(
                    cosine_matrix[example_index],
                    query_repeat,
                )

                examples.append(
                    {
                        "input": _bit_string(
                            inputs[example_index]
                        ),
                        "prediction": _bit_string(prediction),
                        "target": _bit_string(target),
                        "correct_mask": _bit_string(
                            correct_mask.long()
                        ),
                        "correct_cells": correct_cells,
                        "total_cells": total_cells,
                        "cell_accuracy": (
                            correct_cells / total_cells
                        ),
                        "hamming_distance": (
                            total_cells - correct_cells
                        ),
                        "exact_match": (
                            correct_cells == total_cells
                        ),
                        "mean_target_probability": float(
                            target_probabilities.mean().item()
                        ),
                        "best_matching_repeats": (
                            best_matching_repeats
                        ),
                        "best_matching_accuracy": best_accuracy,
                        "matches_by_repeat": matches_by_repeat,
                        "model_requested_repeat_prediction": _bit_string(
                            internal_target
                        ),
                        "internal_correct_mask": _bit_string(
                            internal_correct_mask.long()
                        ),
                        "internal_correct_cells": internal_correct_cells,
                        "internal_hamming_distance": (
                            total_cells - internal_correct_cells
                        ),
                        "internal_exact_match": (
                            internal_correct_cells == total_cells
                        ),
                        "internal_matches_by_repeat": (
                            internal_matches_by_repeat
                        ),
                        "ground_truth_retrieval": ground_truth_rank,
                        "internal_decoded_retrieval": internal_decoded_rank,
                        "query_logit_cosine_by_repeat": example_cosines,
                        "best_internal_cosine_repeats": best_internal_repeats,
                        "best_internal_cosine": best_internal_cosine,
                        "cosine_retrieval": cosine_rank,
                    }
                )
    finally:
        if was_training:
            model.train()

    if counters["num_batches"] == 0:
        raise ValueError(
            "Delayed-recall evaluation consumed no batches."
        )

    metrics = finalize_ca_metrics(counters)
    internal_decoded_metrics = finalize_ca_metrics(
        internal_decoded_counters
    )
    logit_similarity_by_repeat = {
        f"repeat_{repeat}": _finalize_similarity(accumulator)
        for repeat, accumulator in logit_similarity.items()
    }
    ground_truth_comparison_by_repeat = {
        f"repeat_{repeat}": finalize_ca_metrics(candidate_counters)
        for repeat, candidate_counters in ground_truth_comparison_counters.items()
    }
    internal_decoded_comparison_by_repeat = {
        f"repeat_{repeat}": finalize_ca_metrics(candidate_counters)
        for repeat, candidate_counters in internal_decoded_comparison_counters.items()
    }
    ground_truth_collision_by_repeat = {
        f"repeat_{repeat}": _finalize_decoded_agreement(accumulator)
        for repeat, accumulator in ground_truth_collision.items()
    }
    internal_decoded_collision_by_repeat = {
        f"repeat_{repeat}": _finalize_decoded_agreement(accumulator)
        for repeat, accumulator in internal_decoded_collision.items()
    }

    return {
        "num_repeats": num_repeats,
        "query_repeat": query_repeat,
        "num_recall_repeats": num_recall_repeats,
        "recall_age": recall_age,
        "is_no_op": query_repeat == num_repeats,
        "metrics": metrics,
        "ground_truth_comparison_by_repeat": (
            ground_truth_comparison_by_repeat
        ),
        "ground_truth_state_collision_by_repeat": (
            ground_truth_collision_by_repeat
        ),
        "ground_truth_retrieval": _finalize_ranking(
            ground_truth_ranking
        ),
        "internal_consistency": {
            "decoded_requested_repeat": internal_decoded_metrics,
            "decoded_comparison_by_repeat": (
                internal_decoded_comparison_by_repeat
            ),
            "decoded_state_collision_by_repeat": (
                internal_decoded_collision_by_repeat
            ),
            "decoded_retrieval": _finalize_ranking(
                internal_decoded_ranking
            ),
            "requested_repeat_logit_similarity": (
                logit_similarity_by_repeat[f"repeat_{query_repeat}"]
            ),
            "logit_similarity_by_repeat": logit_similarity_by_repeat,
            "cosine_retrieval": _finalize_ranking(cosine_ranking),
        },
        "examples": examples,
    }

def _macro_delayed_metrics(query_results):
    if not query_results:
        return None

    metric_names = (
        "loss",
        "cell_accuracy",
        "exact_sequence_accuracy",
        "mean_bit_errors_per_sequence",
        "matthews_correlation",
        "average_depth",
    )

    return {
        metric_name: sum(
            result["metrics"][metric_name]
            for result in query_results
        ) / len(query_results)
        for metric_name in metric_names
    }


def _macro_internal_consistency(query_results):
    if not query_results:
        return None
    decoded_metrics = tuple(
        result["internal_consistency"]["decoded_requested_repeat"]
        for result in query_results
    )
    requested_similarity = tuple(
        result["internal_consistency"]["requested_repeat_logit_similarity"]
        for result in query_results
    )
    cosine_retrieval = tuple(
        result["internal_consistency"]["cosine_retrieval"]
        for result in query_results
    )
    decoded_retrieval = tuple(
        result["internal_consistency"]["decoded_retrieval"]
        for result in query_results
    )
    ranking_names = (
        "requested_repeat_is_best_rate",
        "requested_repeat_is_unique_best_rate",
        "requested_repeat_mean_rank",
        "requested_repeat_mean_reciprocal_rank",
        "requested_repeat_mean_margin_over_closest_wrong",
        "requested_repeat_positive_margin_rate",
    )
    return {
        "decoded_requested_repeat_cell_accuracy": sum(
            metrics["cell_accuracy"] for metrics in decoded_metrics
        )
        / len(decoded_metrics),
        "decoded_requested_repeat_exact_sequence_accuracy": sum(
            metrics["exact_sequence_accuracy"] for metrics in decoded_metrics
        )
        / len(decoded_metrics),
        "requested_repeat_logit_cosine_similarity": sum(
            metrics["cosine_similarity"] for metrics in requested_similarity
        )
        / len(requested_similarity),
        "requested_repeat_logit_normalized_mse": sum(
            metrics["normalized_mse"] for metrics in requested_similarity
        )
        / len(requested_similarity),
        "requested_repeat_is_best_cosine_rate": sum(
            metrics["requested_repeat_is_best_rate"]
            for metrics in cosine_retrieval
        )
        / len(cosine_retrieval),
        "requested_repeat_is_unique_best_cosine_rate": sum(
            metrics["requested_repeat_is_unique_best_rate"]
            for metrics in cosine_retrieval
        )
        / len(cosine_retrieval),
        **{
            f"cosine_{name}": sum(metrics[name] for metrics in cosine_retrieval)
            / len(cosine_retrieval)
            for name in ranking_names
            if all(metrics[name] is not None for metrics in cosine_retrieval)
        },
        **{
            f"decoded_{name}": sum(metrics[name] for metrics in decoded_retrieval)
            / len(decoded_retrieval)
            for name in ranking_names
            if all(metrics[name] is not None for metrics in decoded_retrieval)
        },
    }


def _macro_ground_truth_retrieval(query_results):
    if not query_results:
        return None
    retrieval = tuple(result["ground_truth_retrieval"] for result in query_results)
    metric_names = (
        "requested_repeat_is_best_rate",
        "requested_repeat_is_unique_best_rate",
        "requested_repeat_mean_rank",
        "requested_repeat_mean_reciprocal_rank",
        "requested_repeat_mean_margin_over_closest_wrong",
        "requested_repeat_positive_margin_rate",
    )
    return {
        name: sum(metrics[name] for metrics in retrieval) / len(retrieval)
        for name in metric_names
        if all(metrics[name] is not None for metrics in retrieval)
    }


def _pair_balanced_macro(pair_results, field):
    """Average one per-pair metric mapping with equal weight per pair."""
    metric_mappings = [
        result[field]
        for result in pair_results
        if result.get(field) is not None
    ]
    if not metric_mappings:
        return None

    # Single-repeat pairs do not define every retrieval margin/rank field.
    # Retain only metrics defined for every included pair so the all-query
    # summary can include the 1:1 pair without inventing missing values.
    metric_names = tuple(
        name
        for name in metric_mappings[0]
        if all(name in metrics for metrics in metric_mappings)
    )
    return {
        name: sum(metrics[name] for metrics in metric_mappings)
        / len(metric_mappings)
        for name in metric_names
    }


def summarize_delayed_recall_pairs(pair_results):
    """Pair-balanced summary used for delayed-recall reporting and selection."""
    all_pairs = list(pair_results.values())
    if not all_pairs:
        raise ValueError("Delayed-recall selection requires at least one pair.")

    eligible_pairs = [
        result
        for result in all_pairs
        if result["nontrivial_queries_macro"] is not None
    ]

    all_queries_pair_macro = _pair_balanced_macro(
        all_pairs, "all_queries_macro"
    )
    all_internal_pair_macro = _pair_balanced_macro(
        all_pairs, "all_queries_internal_consistency_macro"
    )
    all_ground_truth_pair_macro = _pair_balanced_macro(
        all_pairs, "all_queries_ground_truth_retrieval_macro"
    )
    pair_macro = _pair_balanced_macro(
        eligible_pairs, "nontrivial_queries_macro"
    )
    internal_pair_macro = _pair_balanced_macro(
        eligible_pairs, "nontrivial_internal_consistency_macro"
    )
    ground_truth_pair_macro = _pair_balanced_macro(
        eligible_pairs, "nontrivial_ground_truth_retrieval_macro"
    )
    all_queries = [
        query
        for result in all_pairs
        for query in result["queries"].values()
    ]
    nontrivial_queries = [
        query
        for result in eligible_pairs
        for query in result["queries"].values()
        if not query["is_no_op"]
    ]
    worst_query = min(
        all_queries,
        key=lambda result: result["metrics"]["cell_accuracy"],
    )
    worst_nontrivial_query = (
        min(
            nontrivial_queries,
            key=lambda result: result["metrics"]["cell_accuracy"],
        )
        if nontrivial_queries
        else None
    )
    return {
        "pairs": len(all_pairs),
        "all_queries": len(all_queries),
        "eligible_pairs": len(eligible_pairs),
        "nontrivial_queries": len(nontrivial_queries),
        "all_queries_pair_macro": all_queries_pair_macro,
        "all_internal_consistency_pair_macro": all_internal_pair_macro,
        "all_ground_truth_retrieval_pair_macro": (
            all_ground_truth_pair_macro
        ),
        "nontrivial_queries_pair_macro": pair_macro,
        "nontrivial_internal_consistency_pair_macro": internal_pair_macro,
        "nontrivial_ground_truth_retrieval_pair_macro": (
            ground_truth_pair_macro
        ),
        "worst_query": {
            "num_repeats": worst_query["num_repeats"],
            "query_repeat": worst_query["query_repeat"],
            "recall_age": worst_query["recall_age"],
            "metrics": worst_query["metrics"],
        },
        "worst_nontrivial_query": (
            {
                "num_repeats": worst_nontrivial_query["num_repeats"],
                "query_repeat": worst_nontrivial_query["query_repeat"],
                "recall_age": worst_nontrivial_query["recall_age"],
                "metrics": worst_nontrivial_query["metrics"],
            }
            if worst_nontrivial_query is not None
            else None
        ),
    }

def evaluate_delayed_recall_pairs(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    pairs,
    max_batches=None,
    num_recall_repeats=1,
    num_examples=2,
    ctx=None,
):
    return {
        ca_pair_key(ca_steps, num_repeats):
            evaluate_delayed_recall_pair(
                forward_context,
                dataloader,
                device,
                ca_steps=ca_steps,
                num_repeats=num_repeats,
                max_batches=max_batches,
                num_recall_repeats=num_recall_repeats,
                num_examples=num_examples,
                ctx=ctx,
            )
        for ca_steps, num_repeats in pairs
    }
def evaluate_delayed_recall_pair(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    ca_steps,
    num_repeats,
    max_batches=None,
    num_recall_repeats=1,
    num_examples=2,
    ctx=None,
):
    if ca_steps != num_repeats:
        raise ValueError(
            "Delayed recall currently requires diagonal pairs "
            "such as 5:5."
        )

    queries = {}

    for query_repeat in range(1, num_repeats + 1):
        query_result = evaluate_delayed_recall_query(
            forward_context,
            dataloader,
            device,
            num_repeats=num_repeats,
            query_repeat=query_repeat,
            num_recall_repeats=num_recall_repeats,
            max_batches=max_batches,
            num_examples=num_examples,
            ctx=ctx,
        )

        queries[f"query_repeat_{query_repeat}"] = (
            query_result
        )

    all_queries = list(queries.values())

    nontrivial_queries = [
        result
        for result in all_queries
        if not result["is_no_op"]
    ]

    if nontrivial_queries:
        worst_query = min(
            nontrivial_queries,
            key=lambda result: result["metrics"][
                "cell_accuracy"
            ],
        )

        worst_nontrivial_query = {
            "query_repeat": worst_query["query_repeat"],
            "recall_age": worst_query["recall_age"],
            "metrics": worst_query["metrics"],
        }
    else:
        worst_nontrivial_query = None

    no_op = queries[f"query_repeat_{num_repeats}"]

    return {
        "ca_steps": ca_steps,
        "num_repeats": num_repeats,
        "queries": queries,
        "all_queries_macro": _macro_delayed_metrics(
            all_queries
        ),
        "nontrivial_queries_macro": (
            _macro_delayed_metrics(nontrivial_queries)
        ),
        "all_queries_internal_consistency_macro": (
            _macro_internal_consistency(all_queries)
        ),
        "nontrivial_internal_consistency_macro": (
            _macro_internal_consistency(nontrivial_queries)
        ),
        "all_queries_ground_truth_retrieval_macro": (
            _macro_ground_truth_retrieval(all_queries)
        ),
        "nontrivial_ground_truth_retrieval_macro": (
            _macro_ground_truth_retrieval(nontrivial_queries)
        ),
        "worst_nontrivial_query": worst_nontrivial_query,
        "no_op": {
            "query_repeat": no_op["query_repeat"],
            "recall_age": no_op["recall_age"],
            "metrics": no_op["metrics"],
        },
    }


@torch.no_grad()
def evaluate_ca_clean_state_transitions(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    max_transition_depth,
    max_batches=None,
    ctx=None,
):
    """Evaluate clean-current transitions while preserving free-rollout K/V.

    For source depth ``t``, independently run the model from the original row
    through ``t`` free repeats, replace only its current recurrent
    representation with the canonical representation of exact Rule 30 row
    ``x_t``, and run repeat ``t + 1`` with the historical middle-block cache
    intact.  Compare that output with exact row ``x_(t+1)``.
    """
    if max_transition_depth <= 0:
        raise ValueError("max_transition_depth must be positive.")

    model = forward_context.model
    was_training = model.training
    model.eval()
    counters = {
        target_depth: new_ca_counters()
        for target_depth in range(1, max_transition_depth + 1)
    }
    ctx = ctx or nullcontext()
    consumed_batches = 0

    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            consumed_batches += 1
            initial_state = batch["input_id"].to(
                device, dtype=torch.long, non_blocking=True
            )
            clean_state = initial_state

            for source_depth in range(max_transition_depth):
                target_depth = source_depth + 1
                clean_target = rule30(clean_state)
                with ctx:
                    logits, average_depth = _forward_all_cells(
                        forward_context,
                        initial_state,
                        num_repeats=target_depth,
                        intervention_source_depth=source_depth,
                        intervention_input_ids=clean_state,
                    )
                loss = F.cross_entropy(
                    logits.reshape(-1, 2), clean_target.reshape(-1)
                )
                update_ca_counters(
                    counters[target_depth],
                    logits,
                    clean_target,
                    loss=loss,
                    average_depth=average_depth,
                )
                clean_state = clean_target
    finally:
        if was_training:
            model.train()

    if consumed_batches == 0:
        raise ValueError("Clean-state transition evaluation consumed no batches.")

    return {
        "max_transition_depth": max_transition_depth,
        "transitions": {
            f"steps_{target_depth - 1}_to_{target_depth}": {
                "source_ca_steps": target_depth - 1,
                "target_ca_steps": target_depth,
                "num_repeats": target_depth,
                "intervention_source_depth": target_depth - 1,
                "metrics": finalize_ca_metrics(counters[target_depth]),
            }
            for target_depth in range(1, max_transition_depth + 1)
        },
        "num_batches": consumed_batches,
    }


def evaluate_ca_clean_state_transition_lengths(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    **kwargs,
):
    """Run the preserved-cache clean-state probe at every row length."""
    return {
        str(num_cells): evaluate_ca_clean_state_transitions(
            forward_context,
            dataloader,
            device,
            **kwargs,
        )
        for num_cells, dataloader in eval_loaders.items()
    }


def _binary_row_string(row):
    return "".join(str(int(value)) for value in row.detach().cpu().tolist())


@torch.no_grad()
def evaluate_ca_repeat_horizon_diagnostics(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    max_repeats,
    target_horizons=None,
    max_batches=None,
    num_examples=1,
    collect_hidden_states=True,
    ctx=None,
):
    """Diagnose what each internal repeat decodes to on the same fixed rows.

    The result includes a repeat-versus-true-horizon metric matrix, decoded
    recurrence consistency, decoded-state similarities, optional hidden-state
    similarities, and a few literal binary rows for human inspection.
    """
    if max_repeats <= 0:
        raise ValueError("max_repeats must be positive.")
    if num_examples < 0:
        raise ValueError("num_examples cannot be negative.")
    if target_horizons is None:
        target_horizons = tuple(range(max_repeats + 1))
    else:
        target_horizons = tuple(sorted(set(int(value) for value in target_horizons)))
    if not target_horizons or target_horizons[0] < 0:
        raise ValueError("target_horizons must contain non-negative values.")

    model = forward_context.model
    repeat_counts = tuple(range(1, max_repeats + 1))
    matrix_counters = {
        (repeats, horizon): new_ca_counters()
        for repeats in repeat_counts
        for horizon in target_horizons
    }
    recurrence_counters = {
        repeats: new_ca_counters() for repeats in repeat_counts
    }
    decoded_similarity = {
        (left, right): _new_similarity_accumulator()
        for left in range(max_repeats + 1)
        for right in range(max_repeats + 1)
    }
    hidden_similarity = None
    hidden_supported = (
        collect_hidden_states
        and "return_repeat_states" in inspect.signature(model.forward).parameters
    )
    if hidden_supported:
        hidden_similarity = {
            (left, right): _new_similarity_accumulator()
            for left in range(max_repeats + 1)
            for right in range(max_repeats + 1)
        }

    examples = []
    was_training = model.training
    model.eval()
    ctx = ctx or nullcontext()
    consumed_batches = 0

    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            consumed_batches += 1
            inputs = batch["input_id"].to(
                device, dtype=torch.long, non_blocking=True
            )

            max_horizon = max(target_horizons)
            all_targets = {0: inputs}
            current_target = inputs
            for horizon in range(1, max_horizon + 1):
                current_target = rule30(current_target)
                all_targets[horizon] = current_target

            predictions = {0: inputs}
            recurrence_targets = {}
            repeat_states = None
            for repeats in repeat_counts:
                with ctx:
                    logits, average_depth, outputs = _forward_all_cells(
                        forward_context,
                        inputs,
                        num_repeats=repeats,
                        return_repeat_states=(
                            hidden_supported and repeats == max_repeats
                        ),
                        return_outputs=True,
                    )
                prediction = logits.argmax(dim=-1)
                predictions[repeats] = prediction
                if hidden_supported and repeats == max_repeats:
                    repeat_states = outputs.get("repeat_states")

                for horizon in target_horizons:
                    target = all_targets[horizon]
                    loss = F.cross_entropy(
                        logits.reshape(-1, 2), target.reshape(-1)
                    )
                    update_ca_counters(
                        matrix_counters[(repeats, horizon)],
                        logits,
                        target,
                        loss=loss,
                        average_depth=average_depth,
                    )

                recurrence_target = rule30(predictions[repeats - 1])
                recurrence_targets[repeats] = recurrence_target
                recurrence_loss = F.cross_entropy(
                    logits.reshape(-1, 2), recurrence_target.reshape(-1)
                )
                update_ca_counters(
                    recurrence_counters[repeats],
                    logits,
                    recurrence_target,
                    loss=recurrence_loss,
                    average_depth=average_depth,
                )

            for left in range(max_repeats + 1):
                for right in range(max_repeats + 1):
                    _update_similarity(
                        decoded_similarity[(left, right)],
                        predictions[left],
                        predictions[right],
                        decoded=True,
                    )

            if hidden_supported:
                if repeat_states is None or len(repeat_states) != max_repeats + 1:
                    raise ValueError(
                        "return_repeat_states must contain the post-begin state "
                        "followed by one state per requested repeat."
                    )
                for left in range(max_repeats + 1):
                    for right in range(max_repeats + 1):
                        _update_similarity(
                            hidden_similarity[(left, right)],
                            repeat_states[left],
                            repeat_states[right],
                        )

            remaining_examples = num_examples - len(examples)
            for example_index in range(min(remaining_examples, inputs.shape[0])):
                target_rows = {
                    f"steps_{horizon}": _binary_row_string(
                        all_targets[horizon][example_index]
                    )
                    for horizon in target_horizons
                }
                prediction_rows = {}
                for repeats in repeat_counts:
                    prediction = predictions[repeats][example_index]
                    total_cells = int(prediction.numel())
                    horizon_correct_cells = {
                        horizon: int(
                            prediction.eq(
                                all_targets[horizon][example_index]
                            ).sum().item()
                        )
                        for horizon in target_horizons
                    }
                    best_horizon = max(
                        target_horizons,
                        key=lambda horizon: (
                            horizon_correct_cells[horizon],
                            -horizon,
                        ),
                    )
                    best_correct_cells = horizon_correct_cells[best_horizon]
                    row = {
                        "decoded": _binary_row_string(prediction),
                        "best_matching_horizon": int(best_horizon),
                        "best_matching_correct_cells": best_correct_cells,
                        "best_matching_total_cells": total_cells,
                        "best_matching_cell_accuracy": _safe_ratio(
                            best_correct_cells, total_cells
                        ),
                    }
                    # Retain the former key for compatibility with existing
                    # diagnostic readers.
                    row["best_cell_accuracy"] = row[
                        "best_matching_cell_accuracy"
                    ]

                    if repeats in target_horizons:
                        expected_target = all_targets[repeats][example_index]
                        expected_correct = prediction.eq(expected_target)
                        expected_correct_cells = int(
                            expected_correct.sum().item()
                        )
                        expected_target_string = _binary_row_string(
                            expected_target
                        )
                        expected_correct_mask = "".join(
                            "|" if bool(value) else "x"
                            for value in expected_correct.cpu().tolist()
                        )
                        row.update(
                            {
                                "expected_horizon": int(repeats),
                                "expected_target": expected_target_string,
                                "expected_correct_cells": (
                                    expected_correct_cells
                                ),
                                "expected_total_cells": total_cells,
                                "expected_cell_accuracy": _safe_ratio(
                                    expected_correct_cells, total_cells
                                ),
                                "expected_correct_mask": (
                                    expected_correct_mask
                                ),
                                # Compatibility aliases for the previous
                                # diagonal-target example schema.
                                "matching_target": expected_target_string,
                                "correct_mask": expected_correct_mask,
                            }
                        )

                    decoded_transition_target = recurrence_targets[repeats][
                        example_index
                    ]
                    decoded_transition_correct_cells = int(
                        prediction.eq(decoded_transition_target).sum().item()
                    )
                    row.update(
                        {
                            "decoded_transition_from_repeat": repeats - 1,
                            "decoded_transition_target": _binary_row_string(
                                decoded_transition_target
                            ),
                            "decoded_transition_correct_cells": (
                                decoded_transition_correct_cells
                            ),
                            "decoded_transition_total_cells": total_cells,
                            "decoded_transition_cell_accuracy": _safe_ratio(
                                decoded_transition_correct_cells, total_cells
                            ),
                        }
                    )
                    prediction_rows[f"repeats_{repeats}"] = row
                examples.append(
                    {
                        "input": _binary_row_string(inputs[example_index]),
                        "targets": target_rows,
                        "predictions": prediction_rows,
                    }
                )
    finally:
        if was_training:
            model.train()

    if consumed_batches == 0:
        raise ValueError("Repeat diagnostics consumed no batches.")

    matrix = {}
    best_matches = {}
    for repeats in repeat_counts:
        row = {
            f"steps_{horizon}": finalize_ca_metrics(
                matrix_counters[(repeats, horizon)]
            )
            for horizon in target_horizons
        }
        matrix[f"repeats_{repeats}"] = row
        best_cell_horizon = max(
            target_horizons,
            key=lambda horizon: (
                row[f"steps_{horizon}"]["cell_accuracy"],
                -horizon,
            ),
        )
        best_mcc_horizon = max(
            target_horizons,
            key=lambda horizon: (
                row[f"steps_{horizon}"]["matthews_correlation"],
                -horizon,
            ),
        )
        best_matches[f"repeats_{repeats}"] = {
            "by_cell_accuracy": int(best_cell_horizon),
            "cell_accuracy": row[f"steps_{best_cell_horizon}"]["cell_accuracy"],
            "by_matthews_correlation": int(best_mcc_horizon),
            "matthews_correlation": row[f"steps_{best_mcc_horizon}"][
                "matthews_correlation"
            ],
        }

    recurrence = {}
    for repeats in repeat_counts:
        row = {
            "rule30_from_previous_decoded": finalize_ca_metrics(
                recurrence_counters[repeats]
            ),
            "versus_previous_decoded": _finalize_similarity(
                decoded_similarity[(repeats, repeats - 1)], decoded=True
            ),
        }
        if repeats >= 2:
            row["versus_two_repeats_earlier"] = _finalize_similarity(
                decoded_similarity[(repeats, repeats - 2)], decoded=True
            )
        recurrence[f"repeats_{repeats}"] = row

    decoded_matrix = {
        f"repeats_{left}": {
            f"repeats_{right}": _finalize_similarity(
                decoded_similarity[(left, right)], decoded=True
            )
            for right in range(max_repeats + 1)
        }
        for left in range(max_repeats + 1)
    }
    if hidden_supported:
        hidden_result = {
            "available": True,
            "similarity_matrix": {
                f"repeats_{left}": {
                    f"repeats_{right}": _finalize_similarity(
                        hidden_similarity[(left, right)]
                    )
                    for right in range(max_repeats + 1)
                }
                for left in range(max_repeats + 1)
            },
        }
    else:
        hidden_result = {
            "available": False,
            "reason": "model forward does not expose return_repeat_states",
        }

    return {
        "max_repeats": max_repeats,
        "target_horizons": list(target_horizons),
        "repeat_horizon_matrix": matrix,
        "best_matching_horizon": best_matches,
        "decoded_recurrence": recurrence,
        "decoded_similarity": decoded_matrix,
        "hidden_state_similarity": hidden_result,
        "examples": examples,
        "num_batches": consumed_batches,
    }


def evaluate_ca_repeat_horizon_lengths(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    **kwargs,
):
    """Run repeat diagnostics independently at every configured row length."""
    return {
        str(num_cells): evaluate_ca_repeat_horizon_diagnostics(
            forward_context,
            dataloader,
            device,
            **kwargs,
        )
        for num_cells, dataloader in eval_loaders.items()
    }


def format_ca_repeat_examples(diagnostics_by_length, *, step=None):
    """Render literal decoded/target rows for terminal and Slurm logs."""
    lines = []
    for length, diagnostics in diagnostics_by_length.items():
        for example_index, example in enumerate(diagnostics.get("examples", [])):
            example_fields = []
            if step is not None:
                example_fields.append(f"step={step}")
            example_fields.extend(
                (f"length={length}", f"example={example_index}")
            )
            lines.append(f"[{' '.join(example_fields)}]")
            lines.append("input:")
            lines.append(example["input"])

            repeat_rows = sorted(
                example["predictions"].items(),
                key=lambda item: int(item[0].rsplit("_", 1)[1]),
            )
            for repeat_key, prediction in repeat_rows:
                repeats = int(repeat_key.rsplit("_", 1)[1])
                repeat_fields = list(example_fields)
                repeat_fields.append(f"repeat={repeats}")
                lines.append("")
                lines.append(f"[{' '.join(repeat_fields)}]")
                lines.append("")
                lines.append("prediction:")
                lines.append(prediction["decoded"])
                lines.append("")

                expected_horizon = prediction.get(
                    "expected_horizon", repeats
                )
                expected_target = prediction.get(
                    "expected_target", prediction.get("matching_target")
                )
                lines.append(
                    f"expected target: Rule30 horizon {expected_horizon}"
                )
                if expected_target is None:
                    lines.append(
                        "not configured in diagnostic target horizons"
                    )
                    lines.append("")
                    lines.append("expected-horizon match:")
                    lines.append("not available")
                else:
                    expected_mask = prediction.get(
                        "expected_correct_mask",
                        prediction.get("correct_mask"),
                    )
                    expected_total = prediction.get(
                        "expected_total_cells", len(prediction["decoded"])
                    )
                    expected_correct = prediction.get(
                        "expected_correct_cells",
                        (
                            expected_mask.count("|")
                            if expected_mask is not None
                            else None
                        ),
                    )
                    expected_accuracy = prediction.get(
                        "expected_cell_accuracy",
                        (
                            _safe_ratio(expected_correct, expected_total)
                            if expected_correct is not None
                            else math.nan
                        ),
                    )
                    lines.append(expected_target)
                    lines.append("")
                    lines.append("expected-horizon match:")
                    lines.append(
                        f"{expected_correct}/{expected_total} cells = "
                        f"{expected_accuracy:.4f}"
                    )
                    lines.append("correct mask:")
                    lines.append(expected_mask)

                best_accuracy = prediction.get(
                    "best_matching_cell_accuracy"
                )
                if best_accuracy is None:
                    best_accuracy = prediction["best_cell_accuracy"]
                best_total = prediction.get("best_matching_total_cells")
                best_correct = prediction.get("best_matching_correct_cells")
                lines.append("")
                lines.append("best-matching horizon:")
                if best_correct is None or best_total is None:
                    lines.append(
                        f"horizon {prediction['best_matching_horizon']}, "
                        f"{best_accuracy:.4f} cell accuracy"
                    )
                else:
                    lines.append(
                        f"horizon {prediction['best_matching_horizon']}, "
                        f"{best_correct}/{best_total} cells = "
                        f"{best_accuracy:.4f}"
                    )

                lines.append("")
                lines.append("decoded transition:")
                transition_accuracy = prediction.get(
                    "decoded_transition_cell_accuracy"
                )
                if transition_accuracy is None:
                    lines.append("not available in diagnostic payload")
                    continue
                transition_from_repeat = prediction[
                    "decoded_transition_from_repeat"
                ]
                transition_correct = prediction[
                    "decoded_transition_correct_cells"
                ]
                transition_total = prediction[
                    "decoded_transition_total_cells"
                ]
                lines.append(
                    f"prediction repeat {repeats} vs "
                    f"Rule30(prediction repeat {transition_from_repeat})"
                )
                lines.append(
                    f"{transition_correct}/{transition_total} cells = "
                    f"{transition_accuracy:.4f}"
                )
    return "\n".join(lines).rstrip()


def _sum_depths(total_depth, new_depth):
    if new_depth is None:
        return total_depth
    value = float(torch.as_tensor(new_depth).detach().cpu().float().item())
    return value if total_depth is None else total_depth + value


@torch.no_grad()
def evaluate_ca_target(
    forward_context: CAForwardContext,
    dataloader,
    device,
    *,
    ca_steps,
    num_repeats=None,
    external_model_calls=1,
    max_batches=None,
    ctx=None,
):
    """Evaluate a generated CA horizon using internal repeats or full rollouts.

    ``ca_steps`` determines the exact Rule 30 target.  ``num_repeats`` controls
    the repeated middle block within each model call.  ``external_model_calls``
    controls how many complete model predictions are fed back as new inputs.
    """
    if ca_steps <= 0:
        raise ValueError("ca_steps must be positive.")
    if num_repeats is not None and num_repeats <= 0:
        raise ValueError("num_repeats must be positive when provided.")
    if external_model_calls <= 0:
        raise ValueError("external_model_calls must be positive.")

    model = forward_context.model
    was_training = model.training
    model.eval()
    counters = new_ca_counters()
    ctx = ctx or nullcontext()

    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["input_id"].to(
                device, dtype=torch.long, non_blocking=True
            )
            targets = apply_rule30(inputs, steps=ca_steps)
            current_state = inputs
            total_depth = None
            logits = None

            for _ in range(external_model_calls):
                with ctx:
                    logits, average_depth = _forward_all_cells(
                        forward_context,
                        current_state,
                        num_repeats=num_repeats,
                    )
                total_depth = _sum_depths(total_depth, average_depth)
                current_state = logits.argmax(dim=-1)

            loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
            update_ca_counters(
                counters,
                logits,
                targets,
                loss=loss,
                average_depth=total_depth,
            )
    finally:
        if was_training:
            model.train()

    if counters["num_batches"] == 0:
        raise ValueError("Evaluation consumed no batches.")
    return finalize_ca_metrics(counters)


def evaluate_ca_target_lengths(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    *,
    ca_steps,
    num_repeats=None,
    external_model_calls=1,
    max_batches=None,
    ctx=None,
):
    """Evaluate one horizon/repeat setting independently at every row length."""
    return {
        str(num_cells): evaluate_ca_target(
            forward_context,
            dataloader,
            device,
            ca_steps=ca_steps,
            num_repeats=num_repeats,
            external_model_calls=external_model_calls,
            max_batches=max_batches,
            ctx=ctx,
        )
        for num_cells, dataloader in eval_loaders.items()
    }


def ca_pair_key(ca_steps, num_repeats):
    """Return the stable logging/checkpoint key for one horizon-depth pair."""
    return f"steps_{ca_steps}_repeats_{num_repeats}"


def evaluate_ca_pairs(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    *,
    pairs,
    max_batches=None,
    ctx=None,
):
    """Evaluate every configured CA-step/model-repeat pair at every length."""
    return {
        ca_pair_key(ca_steps, num_repeats): {
            "ca_steps": ca_steps,
            "num_repeats": num_repeats,
            "by_length": evaluate_ca_target_lengths(
                forward_context,
                eval_loaders,
                device,
                ca_steps=ca_steps,
                num_repeats=num_repeats,
                max_batches=max_batches,
                ctx=ctx,
            ),
        }
        for ca_steps, num_repeats in pairs
    }


def run_final_ca_evaluation(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    *,
    trained_ca_steps,
    trained_num_repeats=None,
    trained_pairs=(),
    internal_pairs=(),
    external_ca_steps=(),
    max_batches=None,
    ctx=None,
):
    """Run the complete post-training CA generalization evaluation.

    The caller is responsible for loading the selected best checkpoint before
    invoking this function.  No optimizer state is changed here.
    """
    trained_pairs = tuple(trained_pairs)
    if trained_pairs:
        in_distribution = {
            "training_pairs": [list(pair) for pair in trained_pairs],
            "by_pair": evaluate_ca_pairs(
                forward_context,
                eval_loaders,
                device,
                pairs=trained_pairs,
                max_batches=max_batches,
                ctx=ctx,
            ),
        }
        baseline_pairs = set(trained_pairs)
    else:
        in_distribution = {
            "ca_steps": trained_ca_steps,
            "num_repeats": trained_num_repeats,
            "by_length": evaluate_ca_target_lengths(
                forward_context,
                eval_loaders,
                device,
                ca_steps=trained_ca_steps,
                num_repeats=trained_num_repeats,
                max_batches=max_batches,
                ctx=ctx,
            ),
        }
        baseline_pairs = {(trained_ca_steps, trained_num_repeats)}

    results = {
        "in_distribution": in_distribution,
        "internal_repeat_extrapolation": {},
        "external_rollout": {},
    }

    for ca_steps, num_repeats in internal_pairs:
        if (ca_steps, num_repeats) in baseline_pairs:
            continue
        key = ca_pair_key(ca_steps, num_repeats)
        results["internal_repeat_extrapolation"][key] = {
            "ca_steps": ca_steps,
            "num_repeats": num_repeats,
            "by_length": evaluate_ca_target_lengths(
                forward_context,
                eval_loaders,
                device,
                ca_steps=ca_steps,
                num_repeats=num_repeats,
                max_batches=max_batches,
                ctx=ctx,
            ),
        }

    for target_ca_steps in external_ca_steps:
        if target_ca_steps % trained_ca_steps != 0:
            raise ValueError(
                "External target CA steps must be divisible by the number of "
                f"CA steps learned per model call ({trained_ca_steps}); got "
                f"{target_ca_steps}."
            )
        model_calls = target_ca_steps // trained_ca_steps
        key = f"steps_{target_ca_steps}"
        results["external_rollout"][key] = {
            "ca_steps": target_ca_steps,
            "model_calls": model_calls,
            "num_repeats_per_call": trained_num_repeats,
            "by_length": evaluate_ca_target_lengths(
                forward_context,
                eval_loaders,
                device,
                ca_steps=target_ca_steps,
                num_repeats=trained_num_repeats,
                external_model_calls=model_calls,
                max_batches=max_batches,
                ctx=ctx,
            ),
        }

    return results


def supports_clean_state_intervention(model):
    """Return whether ``model.forward`` explicitly supports the cache probe."""
    parameters = inspect.signature(model.forward).parameters
    return {
        "intervention_source_depth",
        "intervention_input_ids",
    }.issubset(parameters)


def evaluate_loaded_ca_checkpoint(
    forward_context: CAForwardContext,
    eval_loaders,
    device,
    *,
    checkpoint_metadata,
    label,
    split_seed,
    samples_per_length,
    trained_ca_steps,
    trained_num_repeats=None,
    trained_pairs=(),
    internal_pairs=(),
    external_ca_steps=(),
    final_eval_max_batches=None,
    repeat_diagnostic_max_repeats=None,
    repeat_diagnostic_horizons=None,
    repeat_diagnostic_max_batches=None,
    eval_max_batches=None,
    repeat_diagnostic_examples=1,
    forward_policy_metadata=None,
    ctx=None,
):
    """Run the canonical final evaluation for one already-loaded checkpoint.

    Both training and standalone checkpoint evaluation call this function so
    task metrics, repeat diagnostics, intervention capability checks, and
    returned reporting structure cannot drift between the two entry points.
    The caller remains responsible for loading and validating checkpoint
    weights before invoking it.
    """
    task_metrics = run_final_ca_evaluation(
        forward_context,
        eval_loaders,
        device,
        trained_ca_steps=trained_ca_steps,
        trained_num_repeats=trained_num_repeats,
        trained_pairs=trained_pairs,
        internal_pairs=internal_pairs,
        external_ca_steps=external_ca_steps,
        max_batches=final_eval_max_batches,
        ctx=ctx,
    )
    repeat_diagnostics = None
    preserved_cache_clean_state_transitions = None
    if repeat_diagnostic_max_repeats is not None:
        diagnostic_max_batches = (
            final_eval_max_batches
            or repeat_diagnostic_max_batches
            or eval_max_batches
        )
        repeat_diagnostics = evaluate_ca_repeat_horizon_lengths(
            forward_context,
            eval_loaders,
            device,
            max_repeats=repeat_diagnostic_max_repeats,
            target_horizons=repeat_diagnostic_horizons,
            max_batches=diagnostic_max_batches,
            num_examples=repeat_diagnostic_examples,
            collect_hidden_states=True,
            ctx=ctx,
        )
        if supports_clean_state_intervention(forward_context.model):
            preserved_cache_clean_state_transitions = (
                evaluate_ca_clean_state_transition_lengths(
                    forward_context,
                    eval_loaders,
                    device,
                    max_transition_depth=repeat_diagnostic_max_repeats,
                    max_batches=diagnostic_max_batches,
                    ctx=ctx,
                )
            )
        else:
            print(
                "Skipping preserved-cache clean-state transitions: "
                "model.forward does not support the intervention arguments.",
                flush=True,
            )
        rendered_examples = format_ca_repeat_examples(
            repeat_diagnostics, step=f"final_{label}"
        )
        if rendered_examples:
            print(rendered_examples, flush=True)

    return {
        "checkpoint": checkpoint_metadata,
        "forward_policy": dict(forward_policy_metadata or {}),
        "split": {
            "role": "independent_final_test",
            "seed": int(split_seed),
            "samples_per_length": int(samples_per_length),
        },
        "task_metrics": task_metrics,
        "repeat_diagnostics": repeat_diagnostics,
        "preserved_cache_clean_state_transitions": (
            preserved_cache_clean_state_transitions
        ),
    }
