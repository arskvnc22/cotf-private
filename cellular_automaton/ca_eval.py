"""Evaluation metrics for cellular-automaton row prediction."""

import math
import inspect
from contextlib import nullcontext

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


def _forward_all_cells(
    forward_context: CAForwardContext,
    inputs,
    *,
    num_repeats=None,
    return_repeat_states=False,
    return_outputs=False,
    intervention_source_depth=None,
    intervention_input_ids=None,
):
    """Call once and require one binary logit vector per input cell.

    Internal recurrence remains entirely inside the model. This helper only
    ensures that inference returns logits for every spatial CA cell.
    """
    forward_kwargs = {"get_logits": True, "return_all_logits": True}
    if num_repeats is not None:
        forward_kwargs["num_repeats"] = num_repeats
    if return_repeat_states:
        forward_kwargs["return_repeat_states"] = True
    if intervention_source_depth is not None or intervention_input_ids is not None:
        forward_kwargs["intervention_source_depth"] = intervention_source_depth
        forward_kwargs["intervention_input_ids"] = intervention_input_ids

    try:
        outputs = forward_context.call(inputs, **forward_kwargs)
    except TypeError as error:
        if "return_all_logits" in str(error):
            # Models not yet converted to the explicit all-position interface
            # retain the existing target-provided compatibility path below.
            forward_kwargs.pop("return_all_logits")
            outputs = forward_context.call(inputs, **forward_kwargs)
        elif num_repeats is not None and "num_repeats" in str(error):
            raise TypeError(
                "Internal-repeat evaluation requires the model forward method "
                "to accept num_repeats."
            ) from error
        else:
            raise

    if not isinstance(outputs, dict):
        raise TypeError("CA models must return a dictionary of outputs.")
    logits = outputs.get("logits")
    if logits is None:
        raise KeyError("Model output is missing logits; use get_logits=True.")
    expected_shape = (*inputs.shape, 2)
    if tuple(logits.shape) != expected_shape:
        # The current language-model implementations only project the final
        # position when targets are omitted.  Supplying shape-compatible dummy
        # targets requests all logits; targets do not otherwise affect the
        # forward activations.  Future CA-native encoders can return all-cell
        # logits directly and will not take this compatibility branch.
        retry_kwargs = dict(forward_kwargs)
        retry_kwargs["targets"] = inputs
        outputs = forward_context.call(inputs, **retry_kwargs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else None
    if logits is None or tuple(logits.shape) != expected_shape:
        raise ValueError(
            "CA inference requires logits for every cell: expected "
            f"{expected_shape}, got "
            f"{None if logits is None else tuple(logits.shape)}."
        )
    result = (logits, outputs.get("average_depth"))
    if return_outputs:
        return (*result, outputs)
    return result


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
