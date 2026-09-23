"""Training loop dedicated to cellular-automaton row prediction."""

import copy
import inspect
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from optim.runner_utils import (
    InfiniteBatchIterator,
    add_scalar_metrics,
    infinite_batches,
    sanitize_for_json,
    save_model_checkpoint,
    save_training_checkpoint,
)

try:
    from .ca_eval import (
        ca_pair_key,
        evaluate_loaded_ca_checkpoint,
        evaluate_ca_lengths,
        evaluate_ca_pairs,
        evaluate_ca_repeat_horizon_lengths,
        format_ca_repeat_examples,
        supports_clean_state_intervention,
    )
    from .ca_forward import (
        FULL_CA_FORWARD_POLICY,
        CAForwardContext,
        CAForwardPolicy,
    )

    from .ca_exposure import (
        add_training_exposure,
        rebuild_training_exposure,
    )
    from .ca_gen import apply_rule30
    from .ca_reporting import write_eval_metrics
except ImportError:
    from ca_eval import (
        ca_pair_key,
        evaluate_loaded_ca_checkpoint,
        evaluate_ca_lengths,
        evaluate_ca_pairs,
        evaluate_ca_repeat_horizon_lengths,
        format_ca_repeat_examples,
        supports_clean_state_intervention,
    )
    from ca_forward import (
        FULL_CA_FORWARD_POLICY,
        CAForwardContext,
        CAForwardPolicy,
    )

    from ca_exposure import add_training_exposure, rebuild_training_exposure
    from ca_gen import apply_rule30
    from ca_reporting import write_eval_metrics


def _autocast_context(args):
    if args.device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=args.dtype)


def _loss_from_outputs(outputs):
    if not isinstance(outputs, dict):
        raise TypeError("CA models must return a dictionary of outputs.")
    loss = outputs.get("cross_entropy_loss", outputs.get("loss"))
    if loss is None:
        raise KeyError("CA model output is missing loss/cross_entropy_loss.")
    if loss.numel() != 1:
        raise ValueError("CA training loss must be a scalar.")
    return loss


def _finite_or(value, fallback):
    value = float(value)
    return value if math.isfinite(value) else fallback


def _supports_clean_state_intervention(model):
    """Return whether ``model.forward`` explicitly supports the cache probe."""
    return supports_clean_state_intervention(model)


def ca_selection_key(metrics, primary_metric):
    """Return a deterministic best-checkpoint key with useful tie-breakers."""
    exact = _finite_or(metrics["exact_sequence_accuracy"], -math.inf)
    cell = _finite_or(metrics["cell_accuracy"], -math.inf)
    negative_loss = -_finite_or(metrics["loss"], math.inf)
    if primary_metric == "exact_sequence_accuracy":
        return exact, cell, negative_loss
    if primary_metric == "cell_accuracy":
        return cell, exact, negative_loss
    if primary_metric == "loss":
        return negative_loss, exact, cell
    raise ValueError(f"Unsupported CA best metric: {primary_metric}.")


def average_ca_selection_metrics(metrics):
    """Average checkpoint-selection metrics over a fixed set of CA pairs."""
    if not metrics:
        raise ValueError("Average extrapolation selection requires at least one pair.")
    fields = ("cell_accuracy", "exact_sequence_accuracy", "loss")
    return {
        field: math.fsum(float(values[field]) for values in metrics) / len(metrics)
        for field in fields
    }


def extrapolation_checkpoint_eligible(
    id_metrics,
    *,
    minimum_cell_accuracy,
    minimum_exact_sequence_accuracy,
):
    """Return the strict selector's eligibility and worst ID accuracies."""
    if not id_metrics:
        raise ValueError("Strict extrapolation selection requires ID metrics.")
    minimum_id_cell_accuracy = min(
        metrics["cell_accuracy"] for metrics in id_metrics
    )
    minimum_id_exact_sequence_accuracy = min(
        metrics["exact_sequence_accuracy"] for metrics in id_metrics
    )
    eligible = (
        minimum_id_cell_accuracy >= minimum_cell_accuracy
        and minimum_id_exact_sequence_accuracy
        >= minimum_exact_sequence_accuracy
    )
    return (
        eligible,
        minimum_id_cell_accuracy,
        minimum_id_exact_sequence_accuracy,
    )


def extrapolation_checkpoint_updates(
    candidate_key,
    *,
    strict_key,
    unconstrained_key,
    strict_eligible,
):
    """Return independent update decisions for both extrapolation selectors."""
    update_unconstrained = (
        unconstrained_key is None or candidate_key > unconstrained_key
    )
    update_strict = strict_eligible and (
        strict_key is None or candidate_key > strict_key
    )
    return update_strict, update_unconstrained


def training_pair_for_step(train_pairs, step):
    """Select one homogeneous horizon/depth pair by deterministic round robin."""
    if not train_pairs:
        return None
    if step < 0:
        raise ValueError("Training step cannot be negative.")
    return tuple(train_pairs[step % len(train_pairs)])


def should_run_scheduled_evaluation(step, eval_freq):
    """Evaluate trained states at positive multiples of the evaluation period."""
    if step < 0:
        raise ValueError("Training step cannot be negative.")
    if eval_freq <= 0:
        raise ValueError("Evaluation frequency must be positive.")
    return step > 0 and step % eval_freq == 0


def _write_json(path, value):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(value), handle, indent=2)
    temporary_path.replace(path)


def _load_json(path, default):
    path = Path(path)
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _set_and_validate_forward_policy(stats, current_policy, *, start_step):
    """Record one run-wide policy and reject incompatible resume attempts."""
    if start_step > 0:
        stored_policy = stats.get(
            "forward_policy", FULL_CA_FORWARD_POLICY.metadata()
        )
        if stored_policy != current_policy:
            raise ValueError(
                "Cannot resume CA training with a different forward policy: "
                f"stored={stored_policy}, requested={current_policy}."
            )
    stats["forward_policy"] = current_policy


def _add_fixed_cot_diagnostics(logs, model, outputs=None):
    """Add optional fixed-CoT metrics without assuming a particular model."""
    if outputs:
        for output_key, log_key in (
            ("sim_of_xs", "diag/boundary_sim"),
            ("var_into", "diag/var_into"),
            ("var_outof", "diag/var_outof"),
        ):
            value = outputs.get(output_key)
            if value is not None:
                scalar = torch.as_tensor(value).detach().float().mean().item()
                logs[log_key] = float(scalar)

        diagnostics = outputs.get("diag_metrics") or {}
        add_scalar_metrics(logs, diagnostics, prefix="diag")
        macro_budget = diagnostics.get("macro_budget")
        if macro_budget is not None:
            for repeat_index, value in enumerate(macro_budget):
                loop = repeat_index + 1
                logs[f"diag_macro/budget_loop_{loop}"] = float(value)
                for metric_key, log_name in (
                    ("macro_in_entropy", "within_entropy"),
                    ("macro_same_pos", "same_pos_budget"),
                ):
                    values = diagnostics.get(metric_key)
                    if values is not None and repeat_index < len(values):
                        logs[f"diag_macro/{log_name}_loop_{loop}"] = float(
                            values[repeat_index]
                        )

        head_entropy = diagnostics.get("head_rep_entropy")
        head_budget = diagnostics.get("head_budget")
        if head_entropy is not None and head_budget is not None:
            head_budget = torch.as_tensor(head_budget)
            if head_budget.ndim == 2:
                for head_index in (0, 5, 11):
                    if head_index >= len(head_entropy) or head_index >= head_budget.shape[0]:
                        continue
                    logs[f"diag_head_{head_index}/repeat_entropy"] = float(
                        head_entropy[head_index]
                    )
                    for repeat_index in range(head_budget.shape[1]):
                        logs[
                            f"diag_head_{head_index}/budget_loop_{repeat_index + 1}"
                        ] = float(head_budget[head_index, repeat_index])

    for attribute, prefix in (
        ("forward_metrics", "diag_step"),
        ("backward_metrics", "diag_grad"),
    ):
        metrics = getattr(model, attribute, None)
        if isinstance(metrics, dict):
            add_scalar_metrics(logs, metrics, prefix=prefix)
            metrics.clear()


def _collect_diagnostics(forward_context, dataloader, args):
    batch = next(iter(dataloader))
    inputs = batch["input_id"].to(
        args.device, dtype=torch.long, non_blocking=args.device.type == "cuda"
    )
    labels = batch["label"].to(
        args.device, dtype=torch.long, non_blocking=args.device.type == "cuda"
    )
    model = forward_context.model
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_context(args):
            return forward_context.call(inputs, targets=labels, get_logits=False)
    finally:
        if was_training:
            model.train()


def _wandb_log(args, values, step):
    if not getattr(args, "wandb", False):
        return
    import wandb

    wandb.log(values, step=step)


def _distributed_mean(value, device, world_size):
    if world_size == 1:
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float((tensor / world_size).item())


def train_ca(
    model,
    optimizer,
    scheduler,
    train_loader,
    eval_loaders,
    test_loaders,
    args,
    distributed_backend,
    checkpoint_dir,
    *,
    start_step=0,
    data_state=None,
):
    """Train a model to predict all cells in a Rule 30 target row."""
    checkpoint_dir = Path(checkpoint_dir)
    raw_model = distributed_backend.get_raw_model(model)
    forward_policy = CAForwardPolicy.from_args(args)
    training_forward = CAForwardContext(model, forward_policy)
    evaluation_forward = CAForwardContext(raw_model, forward_policy)
    forward_policy_metadata = forward_policy.metadata()
    pin_memory = args.device.type == "cuda"
    train_pairs = tuple(args.ca_train_pairs or ())
    extrapolation_pairs = tuple(args.ca_extrapolation_val_pairs or ())
    if args.ca_exact_data_resume:
        batch_iterator = InfiniteBatchIterator(train_loader, state=data_state)
    else:
        batch_iterator = infinite_batches(train_loader)

    def current_data_state():
        if args.ca_exact_data_resume:
            return batch_iterator.state_dict()
        return {}

    log_every = args.ca_log_every or max(1, min(100, args.eval_freq))
    save_every = args.ca_save_every or args.eval_freq
    best_length = int(args.ca_best_length)
    best_pair = tuple(args.ca_best_pair) if args.ca_best_pair is not None else None
    extrapolation_best_pair = (
        tuple(args.ca_extrapolation_best_pair)
        if args.ca_extrapolation_best_pair is not None
        else None
    )
    diagnostic_length = int(args.ca_diagnostic_length)
    accepts_log_metrics = "log_metrics" in inspect.signature(raw_model.forward).parameters
    needs_iter = bool(getattr(raw_model, "needs_iter", False))

    stats_path = checkpoint_dir / "training_stats.json"
    best_id_path = checkpoint_dir / "best_id.json"
    best_extrapolation_strict_path = (
        checkpoint_dir / "best_extrapolation_strict.json"
    )
    best_extrapolation_unconstrained_path = (
        checkpoint_dir / "best_extrapolation_unconstrained.json"
    )
    best_average_extrap_path = checkpoint_dir / "best_average_extrap.json"
    legacy_best_path = checkpoint_dir / "best.json"
    legacy_best_extrapolation_path = checkpoint_dir / "best_extrapolation.json"
    stats = _load_json(
        stats_path,
        {
            "train": [],
            "eval": {},
            "timing": [],
            "best_id": None,
            "best_extrapolation_strict": None,
            "best_extrapolation_unconstrained": None,
            "best_average_extrap": None,
        },
    )
    _set_and_validate_forward_policy(
        stats, forward_policy_metadata, start_step=start_step
    )
    stats.setdefault("best_id", stats.get("best"))
    stats.setdefault(
        "best_extrapolation_strict", stats.get("best_extrapolation")
    )
    stats.setdefault("best_extrapolation_unconstrained", None)
    stats.setdefault("best_average_extrap", None)
    best_id_info = _load_json(
        best_id_path, _load_json(legacy_best_path, None)
    )
    best_extrapolation_strict_info = _load_json(
        best_extrapolation_strict_path,
        _load_json(legacy_best_extrapolation_path, None),
    )
    best_extrapolation_unconstrained_info = _load_json(
        best_extrapolation_unconstrained_path, None
    )
    best_average_extrap_info = _load_json(best_average_extrap_path, None)
    stats["train"] = [row for row in stats["train"] if row["step"] <= start_step]
    stats["timing"] = [row for row in stats["timing"] if row["step"] <= start_step]
    stats["eval"] = {
        step: values
        for step, values in stats["eval"].items()
        if int(step) <= start_step
    }
    training_world_size = distributed_backend.get_world_size()
    training_exposure = rebuild_training_exposure(
        stats["train"],
        materialized_training_rows=len(train_loader.dataset),
        batch_size=args.batch_size,
        accumulation_steps=args.acc_steps,
        world_size=training_world_size,
        num_cells=args.ca_train_num_cells,
        fallback_ca_steps=args.ca_steps,
        fallback_num_repeats=args.n_repeat,
    )
    stats["training_exposure"] = training_exposure

    def selection_is_usable(info):
        return (
            info is not None
            and int(info["step"]) <= start_step
            and (checkpoint_dir / info["checkpoint"]).is_file()
        )

    if not selection_is_usable(best_id_info):
        best_id_info = None
        stats["best_id"] = None
        stats["best"] = None
    if not selection_is_usable(best_extrapolation_strict_info):
        best_extrapolation_strict_info = None
        stats["best_extrapolation_strict"] = None
        stats["best_extrapolation"] = None
    if not selection_is_usable(best_extrapolation_unconstrained_info):
        best_extrapolation_unconstrained_info = None
        stats["best_extrapolation_unconstrained"] = None
    if not selection_is_usable(best_average_extrap_info):
        best_average_extrap_info = None
        stats["best_average_extrap"] = None

    best_id_key = (
        tuple(best_id_info["selection_key"])
        if best_id_info is not None
        else None
    )
    best_extrapolation_strict_key = (
        tuple(best_extrapolation_strict_info["selection_key"])
        if best_extrapolation_strict_info is not None
        else None
    )
    best_extrapolation_unconstrained_key = (
        tuple(best_extrapolation_unconstrained_info["selection_key"])
        if best_extrapolation_unconstrained_info is not None
        else None
    )
    best_average_extrap_key = (
        tuple(best_average_extrap_info["selection_key"])
        if best_average_extrap_info is not None
        else None
    )
    best_id_checkpoint_path = checkpoint_dir / "best_id.pt"
    best_extrapolation_strict_checkpoint_path = (
        checkpoint_dir / "best_extrapolation_strict.pt"
    )
    best_extrapolation_unconstrained_checkpoint_path = (
        checkpoint_dir / "best_extrapolation_unconstrained.pt"
    )
    best_average_extrap_checkpoint_path = (
        checkpoint_dir / "best_average_extrap.pt"
    )
    legacy_best_checkpoint_path = checkpoint_dir / "best.pt"
    legacy_best_extrapolation_checkpoint_path = (
        checkpoint_dir / "best_extrapolation.pt"
    )

    timing_start = time.perf_counter()
    interval_steps = 0
    interval_examples = 0
    interval_cells = 0
    interval_data_wait = 0.0
    interval_pair_loss_sums = {ca_pair_key(*pair): 0.0 for pair in train_pairs}
    interval_pair_loss_counts = {ca_pair_key(*pair): 0 for pair in train_pairs}

    def evaluate_and_maybe_select(step):
        nonlocal best_id_info, best_id_key
        nonlocal best_extrapolation_strict_info
        nonlocal best_extrapolation_strict_key
        nonlocal best_extrapolation_unconstrained_info
        nonlocal best_extrapolation_unconstrained_key
        nonlocal best_average_extrap_info, best_average_extrap_key, timing_start
        distributed_backend.sync()
        if distributed_backend.is_master_process():
            excluded_start = time.perf_counter()
            if train_pairs:
                in_distribution_eval = evaluate_ca_pairs(
                    evaluation_forward,
                    eval_loaders,
                    args.device,
                    pairs=train_pairs,
                    max_batches=args.ca_eval_max_batches,
                    ctx=_autocast_context(args),
                )
                selected_pair_key = ca_pair_key(*best_pair)
                selected_metrics = in_distribution_eval[selected_pair_key][
                    "by_length"
                ][str(best_length)]
                id_selection_metrics = [
                    values["by_length"][str(best_length)]
                    for values in in_distribution_eval.values()
                ]
            else:
                in_distribution_eval = evaluate_ca_lengths(
                    evaluation_forward,
                    eval_loaders,
                    args.device,
                    max_batches=args.ca_eval_max_batches,
                    ctx=_autocast_context(args),
                )
                in_distribution_eval = {
                    str(length): values
                    for length, values in in_distribution_eval.items()
                }
                selected_metrics = in_distribution_eval[str(best_length)]
                id_selection_metrics = [selected_metrics]

            repeat_diagnostics = None
            if args.ca_repeat_diagnostic_max_repeats is not None:
                repeat_diagnostics = evaluate_ca_repeat_horizon_lengths(
                    evaluation_forward,
                    eval_loaders,
                    args.device,
                    max_repeats=args.ca_repeat_diagnostic_max_repeats,
                    target_horizons=args.ca_repeat_diagnostic_horizons,
                    max_batches=(
                        args.ca_repeat_diagnostic_max_batches
                        or args.ca_eval_max_batches
                    ),
                    num_examples=args.ca_repeat_diagnostic_examples,
                    collect_hidden_states=True,
                    ctx=_autocast_context(args),
                )
                rendered_examples = format_ca_repeat_examples(
                    repeat_diagnostics, step=step
                )
                if rendered_examples:
                    print(rendered_examples, flush=True)

            extrapolation_eval = {}
            if extrapolation_pairs:
                if repeat_diagnostics is None:
                    raise RuntimeError(
                        "Extrapolation selection requires repeat diagnostics."
                    )
                extrapolation_eval = {
                    ca_pair_key(ca_steps, num_repeats): {
                        "ca_steps": ca_steps,
                        "num_repeats": num_repeats,
                        "by_length": {
                            str(length): repeat_diagnostics[str(length)][
                                "repeat_horizon_matrix"
                            ][f"repeats_{num_repeats}"][f"steps_{ca_steps}"]
                            for length in eval_loaders
                        },
                    }
                    for ca_steps, num_repeats in extrapolation_pairs
                }

            eval_record = {
                "in_distribution": in_distribution_eval,
                "extrapolation_validation": extrapolation_eval,
                "repeat_diagnostics": repeat_diagnostics,
                "training_exposure": copy.deepcopy(training_exposure),
            }
            stats["eval"][str(step)] = eval_record
            candidate_key = ca_selection_key(selected_metrics, args.ca_best_metric)
            if best_id_key is None or candidate_key > best_id_key:
                best_id_key = candidate_key
                best_id_info = {
                    "step": int(step),
                    "length": best_length,
                    "metric": args.ca_best_metric,
                    "value": float(selected_metrics[args.ca_best_metric]),
                    "selection_key": list(candidate_key),
                    "checkpoint": best_id_checkpoint_path.name,
                    "selection_type": "in_distribution",
                    "forward_policy": forward_policy_metadata,
                }
                if best_pair is not None:
                    best_id_info["ca_steps"] = best_pair[0]
                    best_id_info["num_repeats"] = best_pair[1]
                save_model_checkpoint(
                    best_id_checkpoint_path,
                    model=raw_model,
                    step=step,
                    metadata=best_id_info,
                )
                # Preserve the historical filenames for existing analysis tools.
                save_model_checkpoint(
                    legacy_best_checkpoint_path,
                    model=raw_model,
                    step=step,
                    metadata=best_id_info,
                )
                stats["best_id"] = best_id_info
                stats["best"] = best_id_info
                _write_json(best_id_path, best_id_info)
                _write_json(legacy_best_path, best_id_info)

            (
                extrapolation_strict_eligible,
                min_id_cell_accuracy,
                min_id_exact_accuracy,
            ) = extrapolation_checkpoint_eligible(
                id_selection_metrics,
                minimum_cell_accuracy=(
                    args.ca_extrapolation_min_id_cell_accuracy
                ),
                minimum_exact_sequence_accuracy=(
                    args.ca_extrapolation_min_id_exact_sequence_accuracy
                ),
            )
            extrapolation_strict_eligible = (
                bool(extrapolation_pairs) and extrapolation_strict_eligible
            )
            if extrapolation_pairs:
                selected_extrapolation_key = ca_pair_key(
                    *extrapolation_best_pair
                )
                selected_extrapolation_metrics = extrapolation_eval[
                    selected_extrapolation_key
                ]["by_length"][str(best_length)]
                candidate_extrapolation_key = ca_selection_key(
                    selected_extrapolation_metrics,
                    args.ca_extrapolation_best_metric,
                )
                (
                    update_strict_extrapolation,
                    update_unconstrained_extrapolation,
                ) = extrapolation_checkpoint_updates(
                    candidate_extrapolation_key,
                    strict_key=best_extrapolation_strict_key,
                    unconstrained_key=best_extrapolation_unconstrained_key,
                    strict_eligible=extrapolation_strict_eligible,
                )

                def extrapolation_metadata(checkpoint_path, selection_type):
                    return {
                        "step": int(step),
                        "length": best_length,
                        "metric": args.ca_extrapolation_best_metric,
                        "value": float(
                            selected_extrapolation_metrics[
                                args.ca_extrapolation_best_metric
                            ]
                        ),
                        "selection_key": list(candidate_extrapolation_key),
                        "checkpoint": checkpoint_path.name,
                        "selection_type": selection_type,
                        "ca_steps": extrapolation_best_pair[0],
                        "num_repeats": extrapolation_best_pair[1],
                        "minimum_id_cell_accuracy": min_id_cell_accuracy,
                        "minimum_id_exact_sequence_accuracy": min_id_exact_accuracy,
                        "required_minimum_id_cell_accuracy": (
                            args.ca_extrapolation_min_id_cell_accuracy
                        ),
                        "required_minimum_id_exact_sequence_accuracy": (
                            args.ca_extrapolation_min_id_exact_sequence_accuracy
                        ),
                        "strict_id_gate_passed": extrapolation_strict_eligible,
                        "forward_policy": forward_policy_metadata,
                    }

                if update_unconstrained_extrapolation:
                    best_extrapolation_unconstrained_key = (
                        candidate_extrapolation_key
                    )
                    best_extrapolation_unconstrained_info = (
                        extrapolation_metadata(
                            best_extrapolation_unconstrained_checkpoint_path,
                            "extrapolation_unconstrained",
                        )
                    )
                    save_model_checkpoint(
                        best_extrapolation_unconstrained_checkpoint_path,
                        model=raw_model,
                        step=step,
                        metadata=best_extrapolation_unconstrained_info,
                    )
                    stats["best_extrapolation_unconstrained"] = (
                        best_extrapolation_unconstrained_info
                    )
                    _write_json(
                        best_extrapolation_unconstrained_path,
                        best_extrapolation_unconstrained_info,
                    )

                if update_strict_extrapolation:
                    best_extrapolation_strict_key = candidate_extrapolation_key
                    best_extrapolation_strict_info = extrapolation_metadata(
                        best_extrapolation_strict_checkpoint_path,
                        "extrapolation_strict",
                    )
                    save_model_checkpoint(
                        best_extrapolation_strict_checkpoint_path,
                        model=raw_model,
                        step=step,
                        metadata=best_extrapolation_strict_info,
                    )
                    # Preserve the historical strict-selector filenames.
                    save_model_checkpoint(
                        legacy_best_extrapolation_checkpoint_path,
                        model=raw_model,
                        step=step,
                        metadata=best_extrapolation_strict_info,
                    )
                    stats["best_extrapolation_strict"] = (
                        best_extrapolation_strict_info
                    )
                    stats["best_extrapolation"] = (
                        best_extrapolation_strict_info
                    )
                    _write_json(
                        best_extrapolation_strict_path,
                        best_extrapolation_strict_info,
                    )
                    _write_json(
                        legacy_best_extrapolation_path,
                        best_extrapolation_strict_info,
                    )

                average_source_metrics = [
                    extrapolation_eval[ca_pair_key(*pair)]["by_length"][
                        str(best_length)
                    ]
                    for pair in extrapolation_pairs
                ]
                average_extrapolation_metrics = average_ca_selection_metrics(
                    average_source_metrics
                )
                candidate_average_key = ca_selection_key(
                    average_extrapolation_metrics,
                    args.ca_extrapolation_best_metric,
                )
                if (
                    best_average_extrap_key is None
                    or candidate_average_key > best_average_extrap_key
                ):
                    best_average_extrap_key = candidate_average_key
                    best_average_extrap_info = {
                        "step": int(step),
                        "length": best_length,
                        "metric": f"mean_{args.ca_extrapolation_best_metric}",
                        "value": float(
                            average_extrapolation_metrics[
                                args.ca_extrapolation_best_metric
                            ]
                        ),
                        "selection_key": list(candidate_average_key),
                        "checkpoint": best_average_extrap_checkpoint_path.name,
                        "selection_type": "average_extrapolation",
                        "selection_aggregation": "arithmetic_mean",
                        "selection_pairs": [list(pair) for pair in extrapolation_pairs],
                        "per_pair_values": {
                            ca_pair_key(*pair): float(
                                metrics[args.ca_extrapolation_best_metric]
                            )
                            for pair, metrics in zip(
                                extrapolation_pairs, average_source_metrics
                            )
                        },
                        "minimum_id_cell_accuracy": min_id_cell_accuracy,
                        "minimum_id_exact_sequence_accuracy": min_id_exact_accuracy,
                        "strict_id_gate_passed": extrapolation_strict_eligible,
                        "forward_policy": forward_policy_metadata,
                    }
                    save_model_checkpoint(
                        best_average_extrap_checkpoint_path,
                        model=raw_model,
                        step=step,
                        metadata=best_average_extrap_info,
                    )
                    stats["best_average_extrap"] = best_average_extrap_info
                    _write_json(
                        best_average_extrap_path,
                        best_average_extrap_info,
                    )

            repeat_summary = None
            if repeat_diagnostics is not None:
                repeat_summary = {
                    length: {
                        "best_matching_horizon": values[
                            "best_matching_horizon"
                        ],
                        "decoded_recurrence_cell_accuracy": {
                            repeat: recurrence[
                                "rule30_from_previous_decoded"
                            ]["cell_accuracy"]
                            for repeat, recurrence in values[
                                "decoded_recurrence"
                            ].items()
                        },
                    }
                    for length, values in repeat_diagnostics.items()
                }
            print(
                json.dumps(
                    {
                        "step": step,
                        "ca_eval": in_distribution_eval,
                        "extrapolation_eval": extrapolation_eval,
                        "repeat_summary": repeat_summary,
                        "extrapolation_strict_checkpoint_eligible": (
                            extrapolation_strict_eligible
                        ),
                        "minimum_id_cell_accuracy": min_id_cell_accuracy,
                        "minimum_id_exact_sequence_accuracy": min_id_exact_accuracy,
                        "best_id": best_id_info,
                        "best_extrapolation_strict": (
                            best_extrapolation_strict_info
                        ),
                        "best_extrapolation_unconstrained": (
                            best_extrapolation_unconstrained_info
                        ),
                        "best_average_extrap": best_average_extrap_info,
                    },
                    indent=2,
                )
            )
            logs = {"iter": step}
            if train_pairs:
                add_scalar_metrics(logs, in_distribution_eval, prefix="eval")
            else:
                for length, metrics in in_distribution_eval.items():
                    add_scalar_metrics(logs, metrics, prefix=f"eval/length_{length}")
            if extrapolation_eval:
                add_scalar_metrics(
                    logs,
                    extrapolation_eval,
                    prefix="extrapolation_validation",
                )
            if repeat_diagnostics is not None:
                for length, values in repeat_diagnostics.items():
                    for repeat, match in values[
                        "best_matching_horizon"
                    ].items():
                        logs[
                            f"repeat_diag/length_{length}/{repeat}/best_horizon_mcc"
                        ] = match["matthews_correlation"]
                    for repeat, recurrence in values[
                        "decoded_recurrence"
                    ].items():
                        logs[
                            f"repeat_diag/length_{length}/{repeat}/rule30_consistency"
                        ] = recurrence["rule30_from_previous_decoded"][
                            "cell_accuracy"
                        ]
            if args.ca_log_fixed_cot_diagnostics:
                diagnostic_outputs = _collect_diagnostics(
                    evaluation_forward, eval_loaders[diagnostic_length], args
                )
                _add_fixed_cot_diagnostics(logs, raw_model, diagnostic_outputs)
            _wandb_log(args, logs, step)
            _write_json(stats_path, stats)
            if args.ca_run_dir is not None:
                write_eval_metrics(args.ca_run_dir, stats)
            timing_start += time.perf_counter() - excluded_start
        distributed_backend.sync()

    for step in range(start_step, args.iterations):
        if should_run_scheduled_evaluation(step, args.eval_freq):
            evaluate_and_maybe_select(step)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        examples_this_step = 0
        cells_this_step = 0
        microbatches_this_step = 0
        active_pair = training_pair_for_step(train_pairs, step)
        if active_pair is not None:
            active_ca_steps, active_num_repeats = active_pair
            active_pair_key = ca_pair_key(*active_pair)
        else:
            active_ca_steps = None
            active_num_repeats = None
            active_pair_key = None
        for microstep_index in range(args.acc_steps):
            data_wait_start = time.perf_counter()
            batch = next(batch_iterator)
            interval_data_wait += time.perf_counter() - data_wait_start
            inputs = batch["input_id"].to(
                args.device, dtype=torch.long, non_blocking=pin_memory
            )
            examples_this_step += int(inputs.shape[0]) * training_world_size
            cells_this_step += int(inputs.numel()) * training_world_size
            microbatches_this_step += 1
            if active_pair is None:
                labels = batch["label"].to(
                    args.device, dtype=torch.long, non_blocking=pin_memory
                )
            else:
                # Initial rows remain materialized and fixed. The selected CA
                # horizon is generated once for the whole GPU batch, ensuring
                # that every sample in this optimizer step has the same target.
                labels = apply_rule30(inputs, steps=active_ca_steps)
       
            forward_kwargs = {"targets": labels, "get_logits": False}
            if active_pair is not None:
                forward_kwargs["num_repeats"] = active_num_repeats
            if needs_iter:
                forward_kwargs["iter"] = step
            if (
                args.ca_log_fixed_cot_diagnostics
                and accepts_log_metrics
                and (step + 1) % args.eval_freq == 0
                and microstep_index == args.acc_steps - 1
            ):
                forward_kwargs["log_metrics"] = True
            with _autocast_context(args):
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_index,
                    gradient_accumulation_steps=args.acc_steps,
                ):
                    outputs = training_forward.call(inputs, **forward_kwargs)
                    loss = _loss_from_outputs(outputs)
            (loss / args.acc_steps).backward()
            accumulated_loss += float(loss.detach().float().item())

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf)
        grad_norm = float(torch.as_tensor(grad_norm).detach().cpu().item())
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        completed_step = step + 1
        mean_loss = _distributed_mean(
            accumulated_loss / args.acc_steps,
            args.device,
            distributed_backend.get_world_size(),
        )
        train_row = {
            "step": completed_step,
            "loss": mean_loss,
            "grad_norm": grad_norm,
            "ca_steps": (
                active_ca_steps if active_pair is not None else args.ca_steps
            ),
            "num_repeats": (
                active_num_repeats if active_pair is not None else args.n_repeat
            ),
            "microbatches_this_step": microbatches_this_step,
            "examples_this_step": examples_this_step,
            "cells_this_step": cells_this_step,
        }
        if active_pair is not None:
            interval_pair_loss_sums[active_pair_key] += mean_loss
            interval_pair_loss_counts[active_pair_key] += 1
        stats["train"].append(train_row)
        add_training_exposure(
            training_exposure,
            ca_steps=train_row["ca_steps"],
            num_repeats=train_row["num_repeats"],
            optimizer_steps=1,
            microbatches=microbatches_this_step,
            examples_seen=examples_this_step,
            cells_seen=cells_this_step,
        )
        stats["training_exposure"] = training_exposure
        interval_steps += 1
        interval_examples += examples_this_step
        interval_cells += cells_this_step

        if completed_step % log_every == 0 or completed_step == args.iterations:
            if distributed_backend.is_master_process():
                elapsed = time.perf_counter() - timing_start
                step_seconds = elapsed / interval_steps
                timing = {
                    "step": completed_step,
                    "step_ms": step_seconds * 1000.0,
                    "data_wait_ms": interval_data_wait * 1000.0 / interval_steps,
                    "examples_per_second": interval_examples / elapsed,
                    "cells_per_second": interval_cells / elapsed,
                    "eta_hours": (
                        (args.iterations - completed_step) * step_seconds / 3600.0
                    ),
                }
                stats["timing"].append(timing)
                pair_losses = {
                    key: interval_pair_loss_sums[key] / interval_pair_loss_counts[key]
                    for key in interval_pair_loss_sums
                    if interval_pair_loss_counts[key]
                }
                print(
                    json.dumps(
                        {
                            "step": completed_step,
                            "train_loss": mean_loss,
                            "grad_norm": grad_norm,
                            **({"pair_losses": pair_losses} if pair_losses else {}),
                            "training_exposure": {
                                "materialized_training_rows": training_exposure[
                                    "materialized_training_rows"
                                ],
                                "total_examples_seen": training_exposure[
                                    "total_examples_seen"
                                ],
                                "total_cells_seen": training_exposure[
                                    "total_cells_seen"
                                ],
                                "equivalent_dataset_passes": training_exposure[
                                    "equivalent_dataset_passes"
                                ],
                                "examples_seen_by_pair": {
                                    key: values["examples_seen"]
                                    for key, values in training_exposure[
                                        "by_training_pair"
                                    ].items()
                                },
                                "example_fraction_by_pair": {
                                    key: values["example_fraction"]
                                    for key, values in training_exposure[
                                        "by_training_pair"
                                    ].items()
                                },
                            },
                            **timing,
                        }
                    ),
                    flush=True,
                )
                current_lr = (
                    scheduler.get_last_lr()[0] if scheduler is not None else args.lr
                )
                logs = {
                    "iter": completed_step,
                    "train/loss": mean_loss,
                    "train/grad_norm": grad_norm,
                    "train/step_ms": timing["step_ms"],
                    "train/examples_per_second": timing["examples_per_second"],
                    "train/cells_per_second": timing["cells_per_second"],
                    "train/eta_hours": timing["eta_hours"],
                    "train/total_examples_seen": training_exposure[
                        "total_examples_seen"
                    ],
                    "train/total_cells_seen": training_exposure[
                        "total_cells_seen"
                    ],
                    "train/equivalent_dataset_passes": training_exposure[
                        "equivalent_dataset_passes"
                    ],
                    "lr": current_lr,
                }
                for key, value in pair_losses.items():
                    logs[f"train_pair/{key}/loss"] = value
                for key, values in training_exposure[
                    "by_training_pair"
                ].items():
                    logs[f"train_pair/{key}/examples_seen"] = values[
                        "examples_seen"
                    ]
                    logs[f"train_pair/{key}/example_fraction"] = values[
                        "example_fraction"
                    ]
                if args.ca_log_fixed_cot_diagnostics:
                    _add_fixed_cot_diagnostics(logs, raw_model)
                _wandb_log(args, logs, completed_step)
            timing_start = time.perf_counter()
            interval_steps = 0
            interval_examples = 0
            interval_cells = 0
            interval_data_wait = 0.0
            for key in interval_pair_loss_sums:
                interval_pair_loss_sums[key] = 0.0
                interval_pair_loss_counts[key] = 0

        if completed_step % save_every == 0:
            save_training_checkpoint(
                checkpoint_dir / f"ckpt_{completed_step}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=completed_step,
                distributed_backend=distributed_backend,
                data_state=current_data_state(),
            )
            if distributed_backend.is_master_process():
                _write_json(stats_path, stats)

    evaluate_and_maybe_select(args.iterations)
    save_training_checkpoint(
        checkpoint_dir / f"ckpt_{args.iterations}.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=args.iterations,
        distributed_backend=distributed_backend,
        data_state=current_data_state(),
    )

    if distributed_backend.is_master_process():
        if best_id_info is None:
            raise RuntimeError(
                "CA training completed without selecting a best ID checkpoint."
            )

        supports_repeat_override = (
            "num_repeats" in inspect.signature(raw_model.forward).parameters
        )
        if train_pairs:
            trained_ca_steps, trained_repeats = best_pair
        else:
            trained_ca_steps = args.ca_steps
            trained_repeats = args.n_repeat if supports_repeat_override else None
        if args.ca_final_repeat_diagnostic_max_repeats is None:
            args.ca_final_repeat_diagnostic_max_repeats = (
                args.ca_repeat_diagnostic_max_repeats
            )

        if args.ca_final_repeat_diagnostic_max_repeats is not None:
            if args.ca_final_repeat_diagnostic_max_repeats <= 0:
                raise ValueError(
                    "--ca_final_repeat_diagnostic_max_repeats must be positive."
                )
            if (
                args.ca_repeat_diagnostic_max_repeats is not None
                and args.ca_final_repeat_diagnostic_max_repeats
                < args.ca_repeat_diagnostic_max_repeats
            ):
                raise ValueError(
                    "Final diagnostic depth cannot be smaller than scheduled "
                    "diagnostic depth."
                )
        final_repeat_diagnostic_horizons = None
        if args.ca_final_repeat_diagnostic_max_repeats is not None:
            final_repeat_diagnostic_horizons = sorted(
                set(args.ca_repeat_diagnostic_horizons or [])
                | set(
                    range(
                        args.ca_final_repeat_diagnostic_max_repeats + 1
                    )
                )
            )
        def evaluate_checkpoint(checkpoint_path, metadata, label):
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            raw_model.load_state_dict(checkpoint["model"], strict=True)
            return evaluate_loaded_ca_checkpoint(
                evaluation_forward,
                test_loaders,
                args.device,
                checkpoint_metadata=metadata,
                label=label,
                split_seed=args.ca_test_seed,
                samples_per_length=args.ca_test_samples,
                trained_ca_steps=trained_ca_steps,
                trained_num_repeats=trained_repeats,
                trained_pairs=train_pairs,
                internal_pairs=args.ca_final_eval_pairs,
                external_ca_steps=args.ca_final_external_steps,
                final_eval_max_batches=args.ca_final_eval_max_batches,
                repeat_diagnostic_max_repeats=(
                    args.ca_final_repeat_diagnostic_max_repeats
                ),
                repeat_diagnostic_horizons=final_repeat_diagnostic_horizons,
                repeat_diagnostic_max_batches=(
                    args.ca_repeat_diagnostic_max_batches
                ),
                eval_max_batches=args.ca_eval_max_batches,
                repeat_diagnostic_examples=args.ca_repeat_diagnostic_examples,
                forward_policy_metadata=forward_policy_metadata,
                ctx=_autocast_context(args),
            )

        best_id_checkpoint_eval = evaluate_checkpoint(
            checkpoint_dir / best_id_info["checkpoint"],
            best_id_info,
            "best_id",
        )
        final_eval = best_id_checkpoint_eval["task_metrics"]
        best_extrapolation_strict_checkpoint_eval = None
        if best_extrapolation_strict_info is not None:
            best_extrapolation_strict_checkpoint_eval = evaluate_checkpoint(
                checkpoint_dir / best_extrapolation_strict_info["checkpoint"],
                best_extrapolation_strict_info,
                "best_extrapolation_strict",
            )
        best_extrapolation_unconstrained_checkpoint_eval = None
        if best_extrapolation_unconstrained_info is not None:
            best_extrapolation_unconstrained_checkpoint_eval = evaluate_checkpoint(
                checkpoint_dir
                / best_extrapolation_unconstrained_info["checkpoint"],
                best_extrapolation_unconstrained_info,
                "best_extrapolation_unconstrained",
            )
        best_average_extrap_checkpoint_eval = None
        if best_average_extrap_info is not None:
            best_average_extrap_checkpoint_eval = evaluate_checkpoint(
                checkpoint_dir / best_average_extrap_info["checkpoint"],
                best_average_extrap_info,
                "best_average_extrap",
            )

        stats["best_id"] = best_id_info
        stats["best_extrapolation_strict"] = best_extrapolation_strict_info
        stats["best_extrapolation_unconstrained"] = (
            best_extrapolation_unconstrained_info
        )
        stats["best_average_extrap"] = best_average_extrap_info
        # Backward-compatible summary fields retain their historical meaning.
        stats["best"] = best_id_info
        stats["best_extrapolation"] = best_extrapolation_strict_info
        stats["final_eval"] = final_eval
        stats["checkpoint_analysis"] = {
            "best_id": best_id_checkpoint_eval,
            "best_extrapolation_strict": (
                best_extrapolation_strict_checkpoint_eval
            ),
            "best_extrapolation_unconstrained": (
                best_extrapolation_unconstrained_checkpoint_eval
            ),
            "best_average_extrap": best_average_extrap_checkpoint_eval,
            # Compatibility aliases for existing analysis notebooks.
            "best_in_distribution": best_id_checkpoint_eval,
            "best_extrapolation": best_extrapolation_strict_checkpoint_eval,
        }
        stats["args"] = sanitize_for_json(vars(args))
        _write_json(checkpoint_dir / "best_id_eval.json", final_eval)
        _write_json(checkpoint_dir / "best_eval.json", final_eval)
        _write_json(
            checkpoint_dir / "best_id_repeat_diagnostics.json",
            best_id_checkpoint_eval["repeat_diagnostics"],
        )
        _write_json(
            checkpoint_dir / "best_repeat_diagnostics.json",
            best_id_checkpoint_eval["repeat_diagnostics"],
        )
        if best_extrapolation_strict_checkpoint_eval is not None:
            _write_json(
                checkpoint_dir / "best_extrapolation_strict_eval.json",
                best_extrapolation_strict_checkpoint_eval["task_metrics"],
            )
            _write_json(
                checkpoint_dir
                / "best_extrapolation_strict_repeat_diagnostics.json",
                best_extrapolation_strict_checkpoint_eval[
                    "repeat_diagnostics"
                ],
            )
            _write_json(
                checkpoint_dir / "best_extrapolation_eval.json",
                best_extrapolation_strict_checkpoint_eval["task_metrics"],
            )
            _write_json(
                checkpoint_dir / "best_extrapolation_repeat_diagnostics.json",
                best_extrapolation_strict_checkpoint_eval[
                    "repeat_diagnostics"
                ],
            )
        if best_extrapolation_unconstrained_checkpoint_eval is not None:
            _write_json(
                checkpoint_dir / "best_extrapolation_unconstrained_eval.json",
                best_extrapolation_unconstrained_checkpoint_eval[
                    "task_metrics"
                ],
            )
            _write_json(
                checkpoint_dir
                / "best_extrapolation_unconstrained_repeat_diagnostics.json",
                best_extrapolation_unconstrained_checkpoint_eval[
                    "repeat_diagnostics"
                ],
            )
        if best_average_extrap_checkpoint_eval is not None:
            _write_json(
                checkpoint_dir / "best_average_extrap_eval.json",
                best_average_extrap_checkpoint_eval["task_metrics"],
            )
            _write_json(
                checkpoint_dir / "best_average_extrap_repeat_diagnostics.json",
                best_average_extrap_checkpoint_eval["repeat_diagnostics"],
            )
        _write_json(checkpoint_dir / "summary.json", stats)
        if args.ca_run_dir is not None:
            write_eval_metrics(args.ca_run_dir, stats)
        final_logs = {
            "iter": args.iterations,
            "best_id/step": best_id_info["step"],
            "best/step": best_id_info["step"],
        }
        add_scalar_metrics(final_logs, final_eval, prefix="final_best_id")
        add_scalar_metrics(final_logs, final_eval, prefix="final")
        if best_extrapolation_strict_checkpoint_eval is not None:
            final_logs["best_extrapolation_strict/step"] = (
                best_extrapolation_strict_info["step"]
            )
            final_logs["best_extrapolation/step"] = (
                best_extrapolation_strict_info["step"]
            )
            add_scalar_metrics(
                final_logs,
                best_extrapolation_strict_checkpoint_eval["task_metrics"],
                prefix="final_best_extrapolation_strict",
            )
            add_scalar_metrics(
                final_logs,
                best_extrapolation_strict_checkpoint_eval["task_metrics"],
                prefix="final_best_extrapolation",
            )
        if best_extrapolation_unconstrained_checkpoint_eval is not None:
            final_logs["best_extrapolation_unconstrained/step"] = (
                best_extrapolation_unconstrained_info["step"]
            )
            add_scalar_metrics(
                final_logs,
                best_extrapolation_unconstrained_checkpoint_eval[
                    "task_metrics"
                ],
                prefix="final_best_extrapolation_unconstrained",
            )
        if best_average_extrap_checkpoint_eval is not None:
            final_logs["best_average_extrap/step"] = (
                best_average_extrap_info["step"]
            )
            add_scalar_metrics(
                final_logs,
                best_average_extrap_checkpoint_eval["task_metrics"],
                prefix="final_best_average_extrap",
            )
        _wandb_log(args, final_logs, args.iterations)
        print(
            json.dumps(
                {
                    "best_id": best_id_info,
                    "best_extrapolation_strict": (
                        best_extrapolation_strict_info
                    ),
                    "best_extrapolation_unconstrained": (
                        best_extrapolation_unconstrained_info
                    ),
                    "best_average_extrap": best_average_extrap_info,
                    "best_id_final_eval": final_eval,
                    "best_extrapolation_strict_final_eval": (
                        best_extrapolation_strict_checkpoint_eval["task_metrics"]
                        if best_extrapolation_strict_checkpoint_eval is not None
                        else None
                    ),
                    "best_extrapolation_unconstrained_final_eval": (
                        best_extrapolation_unconstrained_checkpoint_eval[
                            "task_metrics"
                        ]
                        if best_extrapolation_unconstrained_checkpoint_eval
                        is not None
                        else None
                    ),
                    "best_average_extrap_final_eval": (
                        best_average_extrap_checkpoint_eval["task_metrics"]
                        if best_average_extrap_checkpoint_eval is not None
                        else None
                    ),
                    # Compatibility aliases for existing log parsers.
                    "best": best_id_info,
                    "best_extrapolation": best_extrapolation_strict_info,
                    "final_eval": final_eval,
                    "best_extrapolation_final_eval": (
                        best_extrapolation_strict_checkpoint_eval["task_metrics"]
                        if best_extrapolation_strict_checkpoint_eval is not None
                        else None
                    ),
                },
                indent=2,
            )
        )

    distributed_backend.sync()
    return stats if distributed_backend.is_master_process() else None
