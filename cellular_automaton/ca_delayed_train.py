"""Training loop dedicated to cellular-automaton row prediction."""

import copy
import inspect
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
import torch
debug=False
def prindeb(statement):
    if debug ==True:
        print(statement)
from optim.runner_utils import (
    InfiniteBatchIterator,
    add_scalar_metrics,
    infinite_batches,
    sanitize_for_json,
    save_model_checkpoint,
    save_training_checkpoint,
)

try:
    from .dca_eval import (
        ca_pair_key,
        evaluate_loaded_ca_checkpoint,
        evaluate_ca_lengths,
        evaluate_ca_pairs,
        evaluate_ca_repeat_horizon_lengths,
        format_ca_repeat_examples,
        supports_clean_state_intervention,
        evaluate_delayed_recall_pair,
        evaluate_delayed_recall_pairs,
        summarize_delayed_recall_pairs,
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
    from .ca_gen import apply_rule30, rollout_rule30
    from .ca_reporting import write_eval_metrics
except ImportError:
    from dca_eval import (
        ca_pair_key,
        evaluate_loaded_ca_checkpoint,
        evaluate_ca_lengths,
        evaluate_ca_pairs,
        evaluate_ca_repeat_horizon_lengths,
        format_ca_repeat_examples,
        supports_clean_state_intervention,
        evaluate_delayed_recall_pairs,
        summarize_delayed_recall_pairs,
    )
    from ca_forward import (
        FULL_CA_FORWARD_POLICY,
        CAForwardContext,
        CAForwardPolicy,
    )

    from ca_exposure import add_training_exposure, rebuild_training_exposure
    from ca_gen import apply_rule30, rollout_rule30
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

def delayed_ca_trajectory_loss(
    repeat_logits,
    targets_by_repeat,
    *,
    query_loss=None,
    query_loss_weight=1.0,
):
    """
    repeat_logits:      [B, R, N, 2]
    targets_by_repeat:  [B, R, N]
    query_loss:         scalar CE for the extra query repeat, or None
    """
    if repeat_logits.ndim != 4:
        raise ValueError(
            "repeat_logits must have shape [B, R, N, C]."
        )

    batch_size, num_repeats, num_cells, num_classes = (
        repeat_logits.shape
    )

    expected_targets = (batch_size, num_repeats, num_cells)

    if tuple(targets_by_repeat.shape) != expected_targets:
        raise ValueError(
            "targets_by_repeat does not match repeat_logits: "
            f"expected {expected_targets}, got "
            f"{tuple(targets_by_repeat.shape)}."
        )

    if query_loss_weight < 0:
        raise ValueError("query_loss_weight cannot be negative.")

    cell_losses = torch.nn.functional.cross_entropy(
        repeat_logits.reshape(-1, num_classes),
        targets_by_repeat.reshape(-1),
        reduction="none",
    ).reshape(batch_size, num_repeats, num_cells)

    # Each repeat contributes equally, regardless of sequence length.
    evolution_loss_by_repeat = cell_losses.mean(dim=(0, 2))
    evolution_loss = evolution_loss_by_repeat.mean()

    if query_loss is None:
        total_loss = evolution_loss
    else:
        if query_loss.numel() != 1:
            raise ValueError("query_loss must be scalar.")

        total_loss = (
            evolution_loss
            + query_loss_weight * query_loss
        ) / (1.0 + query_loss_weight)

    return {
        "loss": total_loss,
        "evolution_loss": evolution_loss,
        "evolution_loss_by_repeat": evolution_loss_by_repeat,
        "query_loss": query_loss,
    }
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


def delayed_recall_selection_key(summary, normal_metrics, primary_metric):
    """Rank validation checkpoints on pair-balanced nontrivial recall."""
    delayed = summary["nontrivial_queries_pair_macro"]
    if primary_metric == "loss":
        primary = -_finite_or(delayed["loss"], math.inf)
    else:
        primary = _finite_or(delayed[primary_metric], -math.inf)
    worst_cell = _finite_or(
        summary["worst_nontrivial_query"]["metrics"]["cell_accuracy"],
        -math.inf,
    )
    normal_cell = sum(
        _finite_or(metrics["cell_accuracy"], -math.inf)
        for metrics in normal_metrics
    ) / len(normal_metrics)
    return (
        primary,
        worst_cell,
        _finite_or(delayed["exact_sequence_accuracy"], -math.inf),
        normal_cell,
        -_finite_or(delayed["loss"], math.inf),
    )


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


def _trainer_print(args, *values, **kwargs):
    """Allow the CPU playground to hide routine trainer reports."""
    if not getattr(args, "quiet_trainer_output", False):
        print(*values, **kwargs)


def _format_metric(value):
    if value is None:
        return "n/a"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def format_ca_pair_lines(
    pair_results,
    *,
    length,
    split,
    step=None,
):
    """Return one readable line for each ordinary CA pair."""
    if not pair_results:
        return ""
    if "by_pair" in pair_results:
        pair_results = pair_results["by_pair"]
    step_text = f" step={step}" if step is not None else ""
    lines = []
    for pair in pair_results.values():
        if not isinstance(pair, dict) or "by_length" not in pair:
            continue
        metrics = pair["by_length"][str(length)]
        lines.append(
            f"CA normal [{split}{step_text}] "
            f"steps={pair['ca_steps']} repeats={pair['num_repeats']} "
            f"length={length} | "
            f"cell={_format_metric(metrics['cell_accuracy'])} "
            f"exact={_format_metric(metrics['exact_sequence_accuracy'])} "
            f"mcc={_format_metric(metrics['matthews_correlation'])} "
            f"loss={_format_metric(metrics['loss'])}"
        )
    return "\n".join(lines)


def format_delayed_recall_lines(delayed_recall, *, split, step=None):
    """Return one compact aggregate line for each delayed-recall query."""
    if not delayed_recall:
        return ""
    step_text = f" step={step}" if step is not None else ""
    lines = []
    for pair in delayed_recall.values():
        for query in pair["queries"].values():
            ground_truth = query["metrics"]
            ground_retrieval = query["ground_truth_retrieval"]
            internal = query["internal_consistency"]
            decoded = internal["decoded_requested_repeat"]
            decoded_retrieval = internal["decoded_retrieval"]
            requested_similarity = internal[
                "requested_repeat_logit_similarity"
            ]
            cosine_retrieval = internal["cosine_retrieval"]
            lines.append(
                f"DCA recall [{split}{step_text}] "
                f"horizon={query['num_repeats']} "
                f"query_repeat={query['query_repeat']} "
                f"recall_age={query['recall_age']} "
                f"no_op={str(bool(query['is_no_op'])).lower()} | "
                f"ground_truth cell="
                f"{_format_metric(ground_truth['cell_accuracy'])} "
                f"exact={_format_metric(ground_truth['exact_sequence_accuracy'])} "
                f"mcc={_format_metric(ground_truth['matthews_correlation'])} "
                f"rank={_format_metric(ground_retrieval['requested_repeat_mean_rank'])} "
                f"mrr={_format_metric(ground_retrieval['requested_repeat_mean_reciprocal_rank'])} "
                f"margin={_format_metric(ground_retrieval['requested_repeat_mean_margin_over_closest_wrong'])} "
                f"unique_best={_format_metric(ground_retrieval['requested_repeat_is_unique_best_rate'])} | "
                f"internal decoded_cell="
                f"{_format_metric(decoded['cell_accuracy'])} "
                f"decoded_rank="
                f"{_format_metric(decoded_retrieval['requested_repeat_mean_rank'])} "
                f"cosine="
                f"{_format_metric(requested_similarity['cosine_similarity'])} "
                f"cosine_rank="
                f"{_format_metric(cosine_retrieval['requested_repeat_mean_rank'])} "
                f"cosine_margin="
                f"{_format_metric(cosine_retrieval['requested_repeat_mean_margin_over_closest_wrong'])}"
            )
    return "\n".join(lines)


def format_delayed_recall_summary(summary, *, split, step=None):
    """Return the pair-balanced nontrivial delayed-recall headline."""
    if not summary:
        return ""
    step_text = f" step={step}" if step is not None else ""
    ground_truth = summary["nontrivial_queries_pair_macro"]
    retrieval = summary["nontrivial_ground_truth_retrieval_pair_macro"]
    return (
        f"DCA summary [{split}{step_text}] nontrivial pair macro | "
        f"cell={_format_metric(ground_truth['cell_accuracy'])} "
        f"exact={_format_metric(ground_truth['exact_sequence_accuracy'])} "
        f"mcc={_format_metric(ground_truth['matthews_correlation'])} "
        f"rank={_format_metric(retrieval['requested_repeat_mean_rank'])} "
        f"mrr={_format_metric(retrieval['requested_repeat_mean_reciprocal_rank'])} "
        f"margin={_format_metric(retrieval['requested_repeat_mean_margin_over_closest_wrong'])} "
        f"unique_best={_format_metric(retrieval['requested_repeat_is_unique_best_rate'])}"
    )


def format_checkpoint_selections(*selections):
    """Describe selected checkpoints without embedding evaluation payloads."""
    parts = []
    for name, selection in selections:
        if selection is None:
            parts.append(f"{name}=none")
            continue
        parts.append(
            f"{name}(step={selection['step']}, "
            f"metric={selection['metric']}, "
            f"value={_format_metric(selection['value'])}, "
            f"file={selection['checkpoint']})"
        )
    return "Selected checkpoints | " + " | ".join(parts)


def _distributed_mean(value, device, world_size):
    if world_size == 1:
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float((tensor / world_size).item())
def make_delayed_pair_counters(train_pairs):
    counters = {}

    for pair in train_pairs:
        ca_steps, num_repeats = pair

        counters[tuple(pair)] = {
            "total": 0,
            "normal": 0,
            "delayed": 0,
            "delayed_credit": 0,
            "next_query_repeat": 1,
            "query_counts": {
                repeat: 0
                for repeat in range(1, num_repeats + 1)
            },
        }

    return counters


def count_delayed_percentage(args, pair, pair_counters):
    """
    Choose the mode and target for one optimizer step.

    Returns:
        is_delayed
        query_repeat
        target_steps
    """
    ca_steps, num_repeats = pair
    pair = tuple(pair)

    delayed_percentage = args.ca_delayed_percentage

    if not 0 <= delayed_percentage <= 100:
        raise ValueError(
            "ca_delayed_percentage must be between 0 and 100."
        )

    if ca_steps != num_repeats:
        raise ValueError(
            "Delayed recall currently requires pairs like 5:5, "
            "where one model repeat corresponds to one CA update."
        )

    counter = pair_counters[pair]
    counter["total"] += 1

    # Weighted round robin between normal and delayed batches.
    counter["delayed_credit"] += delayed_percentage

    if counter["delayed_credit"] < 100:
        counter["normal"] += 1
        return False, None, ca_steps

    counter["delayed_credit"] -= 100
    counter["delayed"] += 1

    if args.query_horizon_policy != "uniform":
        raise ValueError(
            f"Unknown query policy: {args.query_horizon_policy}"
        )

    # Round robin through all valid recall targets.
    query_repeat = counter["next_query_repeat"]

    counter["query_counts"][query_repeat] += 1
    counter["next_query_repeat"] += 1

    if counter["next_query_repeat"] > num_repeats:
        counter["next_query_repeat"] = 1

    target_steps = query_repeat

    return True, query_repeat, target_steps


def delayed_schedule_state_dict(args, pair_counters):
    """Return a stable, validated representation of the delayed scheduler."""
    return {
        "version": 1,
        "delayed_percentage": int(args.ca_delayed_percentage),
        "query_horizon_policy": args.query_horizon_policy,
        "pairs": [
            {
                "ca_steps": pair[0],
                "num_repeats": pair[1],
                **copy.deepcopy(counter),
            }
            for pair, counter in pair_counters.items()
        ],
    }


def _pair_visits_before_step(train_pairs, start_step):
    if not train_pairs:
        raise ValueError("Delayed scheduling requires at least one training pair.")
    full_cycles, remainder = divmod(int(start_step), len(train_pairs))
    return {
        tuple(pair): full_cycles + int(index < remainder)
        for index, pair in enumerate(train_pairs)
    }


def _restore_counter_from_schedule(args, pair, visits):
    num_repeats = pair[1]
    delayed, delayed_credit = divmod(
        int(visits) * int(args.ca_delayed_percentage),
        100,
    )
    query_cycles, query_remainder = divmod(delayed, num_repeats)
    return {
        "total": int(visits),
        "normal": int(visits) - delayed,
        "delayed": delayed,
        "delayed_credit": delayed_credit,
        "next_query_repeat": query_remainder + 1,
        "query_counts": {
            repeat: query_cycles + int(repeat <= query_remainder)
            for repeat in range(1, num_repeats + 1)
        },
    }


def restore_delayed_pair_counters(args, train_pairs, state, *, start_step):
    """Restore scheduler state, replaying old checkpoints that lack it."""
    counters = make_delayed_pair_counters(train_pairs)
    visits_by_pair = _pair_visits_before_step(train_pairs, start_step)
    if state is None:
        return {
            pair: _restore_counter_from_schedule(
                args,
                pair,
                visits_by_pair[pair],
            )
            for pair in counters
        }

    if int(state.get("version", -1)) != 1:
        raise ValueError("Unsupported delayed scheduler checkpoint version.")
    if int(state.get("delayed_percentage", -1)) != int(
        args.ca_delayed_percentage
    ):
        raise ValueError(
            "Cannot resume with a different ca_delayed_percentage."
        )
    if state.get("query_horizon_policy") != args.query_horizon_policy:
        raise ValueError(
            "Cannot resume with a different query_horizon_policy."
        )

    restored_pairs = {
        (int(row["ca_steps"]), int(row["num_repeats"])): row
        for row in state.get("pairs", ())
    }
    if set(restored_pairs) != set(counters):
        raise ValueError(
            "Cannot resume delayed scheduling with different training pairs."
        )

    for pair, counter in counters.items():
        row = restored_pairs[pair]
        restored_query_counts = {
            int(repeat): int(count)
            for repeat, count in row["query_counts"].items()
        }
        if set(restored_query_counts) != set(counter["query_counts"]):
            raise ValueError(
                f"Invalid delayed query counters for training pair {pair}."
            )
        restored = {
            "total": int(row["total"]),
            "normal": int(row["normal"]),
            "delayed": int(row["delayed"]),
            "delayed_credit": int(row["delayed_credit"]),
            "next_query_repeat": int(row["next_query_repeat"]),
            "query_counts": restored_query_counts,
        }
        if restored["total"] != restored["normal"] + restored["delayed"]:
            raise ValueError(f"Invalid normal/delayed totals for pair {pair}.")
        if sum(restored_query_counts.values()) != restored["delayed"]:
            raise ValueError(f"Invalid delayed query total for pair {pair}.")
        if not 0 <= restored["delayed_credit"] < 100:
            raise ValueError(f"Invalid delayed credit for pair {pair}.")
        if not 1 <= restored["next_query_repeat"] <= pair[1]:
            raise ValueError(f"Invalid next query repeat for pair {pair}.")
        expected_total = visits_by_pair[pair]
        if restored["total"] != expected_total:
            raise ValueError(
                f"Delayed scheduler total for pair {pair} is "
                f"{restored['total']}, expected {expected_total} at "
                f"training step {start_step}."
            )
        expected_counter = _restore_counter_from_schedule(
            args,
            pair,
            expected_total,
        )
        if restored != expected_counter:
            raise ValueError(
                f"Delayed scheduler contents for pair {pair} do not match "
                f"the deterministic schedule at training step {start_step}."
            )
        counters[pair] = restored
    return counters


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
    resume_data_state = data_state or {}
    delayed_pair_counters = restore_delayed_pair_counters(
        args,
        train_pairs,
        resume_data_state.get("delayed_schedule"),
        start_step=start_step,
    )
    extrapolation_pairs = tuple(args.ca_extrapolation_val_pairs or ())
    if args.ca_exact_data_resume:
        if "batch_iterator" in resume_data_state:
            batch_iterator_state = resume_data_state["batch_iterator"]
        else:
            # Backward compatibility for checkpoints that stored the iterator
            # fields directly in data_state.
            batch_iterator_state = resume_data_state
        batch_iterator = InfiniteBatchIterator(
            train_loader,
            state=batch_iterator_state,
        )
    else:
        batch_iterator = infinite_batches(train_loader)

    def current_data_state():
        state = {
            "delayed_schedule": delayed_schedule_state_dict(
                args, delayed_pair_counters
            )
        }
        if args.ca_exact_data_resume:
            state["batch_iterator"] = batch_iterator.state_dict()
        return state

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
    best_delayed_recall_path = checkpoint_dir / "best_delayed_recall.json"
    best_extrapolation_strict_path = (
        checkpoint_dir / "best_extrapolation_strict.json"
    )
    best_extrapolation_unconstrained_path = (
        checkpoint_dir / "best_extrapolation_unconstrained.json"
    )
    legacy_best_path = checkpoint_dir / "best.json"
    legacy_best_extrapolation_path = checkpoint_dir / "best_extrapolation.json"
    stats = _load_json(
        stats_path,
        {
            "train": [],
            "eval": {},
            "timing": [],
            "best_id": None,
            "best_delayed_recall": None,
            "best_extrapolation_strict": None,
            "best_extrapolation_unconstrained": None,
        },
    )
    _set_and_validate_forward_policy(
        stats, forward_policy_metadata, start_step=start_step
    )
    stats.setdefault("best_id", stats.get("best"))
    stats.setdefault("best_delayed_recall", None)
    stats.setdefault(
        "best_extrapolation_strict", stats.get("best_extrapolation")
    )
    stats.setdefault("best_extrapolation_unconstrained", None)
    best_id_info = _load_json(
        best_id_path, _load_json(legacy_best_path, None)
    )
    best_delayed_recall_info = _load_json(best_delayed_recall_path, None)
    best_extrapolation_strict_info = _load_json(
        best_extrapolation_strict_path,
        _load_json(legacy_best_extrapolation_path, None),
    )
    best_extrapolation_unconstrained_info = _load_json(
        best_extrapolation_unconstrained_path, None
    )
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
    if not selection_is_usable(best_delayed_recall_info):
        best_delayed_recall_info = None
        stats["best_delayed_recall"] = None
    if not selection_is_usable(best_extrapolation_strict_info):
        best_extrapolation_strict_info = None
        stats["best_extrapolation_strict"] = None
        stats["best_extrapolation"] = None
    if not selection_is_usable(best_extrapolation_unconstrained_info):
        best_extrapolation_unconstrained_info = None
        stats["best_extrapolation_unconstrained"] = None

    best_id_key = (
        tuple(best_id_info["selection_key"])
        if best_id_info is not None
        else None
    )
    best_delayed_recall_key = (
        tuple(best_delayed_recall_info["selection_key"])
        if best_delayed_recall_info is not None
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
    best_id_checkpoint_path = checkpoint_dir / "best_id.pt"
    best_delayed_recall_checkpoint_path = (
        checkpoint_dir / "best_delayed_recall.pt"
    )
    best_extrapolation_strict_checkpoint_path = (
        checkpoint_dir / "best_extrapolation_strict.pt"
    )
    best_extrapolation_unconstrained_checkpoint_path = (
        checkpoint_dir / "best_extrapolation_unconstrained.pt"
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
        nonlocal best_delayed_recall_info, best_delayed_recall_key
        nonlocal best_extrapolation_strict_info
        nonlocal best_extrapolation_strict_key
        nonlocal best_extrapolation_unconstrained_info
        nonlocal best_extrapolation_unconstrained_key, timing_start
        distributed_backend.sync()
        if not distributed_backend.is_master_process():
            distributed_backend.sync()
            return
        excluded_start = time.perf_counter()
        delayed_recall_eval = None

        if train_pairs:
            # Existing ordinary CA evaluation.
            in_distribution_eval = evaluate_ca_pairs(
                evaluation_forward,
                eval_loaders,
                args.device,
                pairs=train_pairs,
                max_batches=args.ca_eval_max_batches,
                ctx=_autocast_context(args),
            )

            # Use only the configured checkpoint-selection length for DCA.
            eval_loader = eval_loaders[best_length]

            delayed_recall_eval = evaluate_delayed_recall_pairs(
                evaluation_forward,
                eval_loader,
                args.device,
                pairs=train_pairs,
                max_batches=args.ca_eval_max_batches,
                num_examples=2,
                ctx=_autocast_context(args),
            )
            selected_pair_key = ca_pair_key(*best_pair)

            selected_metrics = in_distribution_eval[
                selected_pair_key
            ]["by_length"][str(best_length)]

            id_selection_metrics = [
                values["by_length"][str(best_length)]
                for values in in_distribution_eval.values()
            ]

            delayed_recall_summary = summarize_delayed_recall_pairs(
                delayed_recall_eval
            )
            eval_record = {
                "in_distribution": in_distribution_eval,
                "delayed_recall": delayed_recall_eval,
                "delayed_recall_summary": delayed_recall_summary,
                "extrapolation_validation": {},
                "repeat_diagnostics": None,
                "delayed_schedule": delayed_schedule_state_dict(
                    args, delayed_pair_counters
                ),
                "training_exposure": copy.deepcopy(training_exposure),
            }
            stats["eval"][str(step)] = eval_record

            candidate_key = ca_selection_key(
                selected_metrics, args.ca_best_metric
            )
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
                    "ca_steps": best_pair[0],
                    "num_repeats": best_pair[1],
                    "forward_policy": forward_policy_metadata,
                }
                save_model_checkpoint(
                    best_id_checkpoint_path,
                    model=raw_model,
                    step=step,
                    metadata=best_id_info,
                )
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

            delayed_candidate_key = delayed_recall_selection_key(
                delayed_recall_summary,
                id_selection_metrics,
                args.ca_delayed_best_metric,
            )
            if (
                best_delayed_recall_key is None
                or delayed_candidate_key > best_delayed_recall_key
            ):
                best_delayed_recall_key = delayed_candidate_key
                delayed_metrics = delayed_recall_summary[
                    "nontrivial_queries_pair_macro"
                ]
                best_delayed_recall_info = {
                    "step": int(step),
                    "length": best_length,
                    "metric": args.ca_delayed_best_metric,
                    "value": float(
                        delayed_metrics[args.ca_delayed_best_metric]
                    ),
                    "selection_key": list(delayed_candidate_key),
                    "checkpoint": best_delayed_recall_checkpoint_path.name,
                    "selection_type": "delayed_recall_nontrivial_pair_macro",
                    "eligible_pairs": delayed_recall_summary[
                        "eligible_pairs"
                    ],
                    "nontrivial_queries": delayed_recall_summary[
                        "nontrivial_queries"
                    ],
                    "internal_consistency": delayed_recall_summary[
                        "nontrivial_internal_consistency_pair_macro"
                    ],
                    "ground_truth_retrieval": delayed_recall_summary[
                        "nontrivial_ground_truth_retrieval_pair_macro"
                    ],
                    "forward_policy": forward_policy_metadata,
                }
                save_model_checkpoint(
                    best_delayed_recall_checkpoint_path,
                    model=raw_model,
                    step=step,
                    metadata=best_delayed_recall_info,
                )
                stats["best_delayed_recall"] = best_delayed_recall_info
                _write_json(
                    best_delayed_recall_path,
                    best_delayed_recall_info,
                )

            _trainer_print(
                args,
                format_ca_pair_lines(
                    in_distribution_eval,
                    length=best_length,
                    split="validation",
                    step=step,
                ),
            )
            _trainer_print(
                args,
                format_delayed_recall_lines(
                    delayed_recall_eval,
                    split="validation",
                    step=step,
                ),
            )
            _trainer_print(
                args,
                format_delayed_recall_summary(
                    delayed_recall_summary,
                    split="validation",
                    step=step,
                ),
            )
            _trainer_print(
                args,
                format_checkpoint_selections(
                    ("best_id", best_id_info),
                    ("best_delayed_recall", best_delayed_recall_info),
                ),
            )
            logs = {"iter": step}
            add_scalar_metrics(logs, in_distribution_eval, prefix="eval")
            add_scalar_metrics(
                logs,
                delayed_recall_eval,
                prefix="delayed_recall",
            )
            add_scalar_metrics(
                logs,
                delayed_recall_summary,
                prefix="delayed_recall_summary",
            )
            _wandb_log(args, logs, step)
            _write_json(stats_path, stats)
            if args.ca_run_dir is not None:
                write_eval_metrics(args.ca_run_dir, stats)
            timing_start += time.perf_counter() - excluded_start

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

            selected_metrics = in_distribution_eval[
                str(best_length)
            ]

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
                    _trainer_print(args, rendered_examples, flush=True)

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
                "delayed_recall": delayed_recall_eval,
                "extrapolation_validation": extrapolation_eval,
                "repeat_diagnostics": repeat_diagnostics,
                "training_exposure": copy.deepcopy(
                    training_exposure
                ),
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
            _trainer_print(
                args,
                json.dumps(
                    {
                        "step": step,
                        "ca_eval": in_distribution_eval,
                        "delayed_recall_eval": delayed_recall_eval,
                        "extrapolation_eval": extrapolation_eval,
                        # Existing fields continue here.
                    },
                    indent=2,
                ),
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
        if step % args.eval_freq == 0:
            prindeb("==========RUNNNING EVALUATION FORWARDS======")
            prindeb("==========RUNNNING EVALUATION FORWARDS======")
            prindeb("==========RUNNNING EVALUATION FORWARDS======")
            evaluate_and_maybe_select(step)
            prindeb("==========DONE==========")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_evolution_loss = 0.0
        accumulated_query_loss = 0.0
        accumulated_evolution_loss_by_repeat = None
        examples_this_step = 0
        cells_this_step = 0
        microbatches_this_step = 0
        active_pair = training_pair_for_step(train_pairs, step)
        is_delayed, query_repeat, target_steps = (count_delayed_percentage(args,active_pair,delayed_pair_counters,))
        
        
        
        if active_pair is not None:
            active_ca_steps, evolution_repeats = active_pair
            active_pair_key = ca_pair_key(*active_pair)
            if is_delayed:
                recall_age = evolution_repeats-query_repeat
            else: 
                recall_age = None
            prindeb(f"acive pair {active_pair} || is delayed = {is_delayed}   || query repeat = {query_repeat}   ||  target steps = {target_steps}  ||active_num_repeats = {evolution_repeats}")



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
                prindeb(f"labels{labels}")
            else:
                # Initial rows remain materialized and fixed. The selected CA
                # horizon is generated once for the whole GPU batch, ensuring
                # that every sample in this optimizer step has the same target.
                with torch.no_grad():
                    targets_by_repeat = rollout_rule30(
                        inputs,
                        evolution_repeats,
                    )

                    if is_delayed:
                        # query_repeat is one-indexed.
                        labels = targets_by_repeat[:, query_repeat - 1] # if is delayed model calculates loss on the query
                    else:
                        labels = targets_by_repeat[:, -1]

       
            forward_kwargs = {
                "targets": None, # for delayed ca loss is calculated explicitly in the trainer due there being 2 objectivs
                "get_logits": is_delayed,
                "return_all_logits":is_delayed,
                "num_repeats": evolution_repeats,
                "delayed_recall": is_delayed,
                "recall_age": recall_age,
                "return_repeat_logits": True,
            }

            if active_pair is not None:
                forward_kwargs["num_repeats"] = evolution_repeats
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

                    repeat_logits = outputs.get("repeat_logits")
                    if repeat_logits is None:
                        raise KeyError(
                            "DCA model did not return repeat_logits."
                        )

                    query_loss = None

                    if is_delayed:
                        query_logits = outputs.get("logits")

                        if query_logits is None:
                            raise KeyError(
                                "Delayed DCA forward did not return query logits."
                            )

                        expected_shape = (*labels.shape, 2)
                        if tuple(query_logits.shape) != expected_shape:
                            raise ValueError(
                                "Query logits have the wrong shape: "
                                f"expected {expected_shape}, got "
                                f"{tuple(query_logits.shape)}."
                            )

                        query_loss = torch.nn.functional.cross_entropy(
                            query_logits.reshape(-1, query_logits.size(-1)),
                            labels.reshape(-1),
                        )

                    losses = delayed_ca_trajectory_loss(
                        repeat_logits,
                        targets_by_repeat,
                        query_loss=query_loss,
                        query_loss_weight=args.ca_query_loss_weight,
                    )

                    loss = losses["loss"]
                    accumulated_evolution_loss += float(
                        losses["evolution_loss"].detach().float().item()
                    )

                    if losses["query_loss"] is not None:
                        accumulated_query_loss += float(
                            losses["query_loss"].detach().float().item()
                        )

                    repeat_losses = losses[
                        "evolution_loss_by_repeat"
                    ].detach().float()

                    if accumulated_evolution_loss_by_repeat is None:
                        accumulated_evolution_loss_by_repeat = repeat_losses.clone()
                    else:
                        accumulated_evolution_loss_by_repeat += repeat_losses
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
        mean_evolution_loss = _distributed_mean(
            accumulated_evolution_loss / args.acc_steps,
            args.device,
            distributed_backend.get_world_size(),
        )

        mean_query_loss = None
        if is_delayed:
            mean_query_loss = _distributed_mean(
                accumulated_query_loss / args.acc_steps,
                args.device,
                distributed_backend.get_world_size(),
            )

        mean_evolution_loss_by_repeat = [
            _distributed_mean(
                value.item() / args.acc_steps,
                args.device,
                distributed_backend.get_world_size(),
            )
            for value in accumulated_evolution_loss_by_repeat
        ]
        train_row = {
            "step": completed_step,
            "loss": mean_loss,
            "grad_norm": grad_norm,
            "ca_steps": (
                active_ca_steps if active_pair is not None else args.ca_steps
            ),
            "num_repeats": (
                evolution_repeats if active_pair is not None else args.n_repeat
            ),
            "is_delayed": bool(is_delayed),
            "query_repeat": query_repeat,
            "recall_age": recall_age,
            "target_steps": target_steps,
            "executed_repeats": evolution_repeats + int(is_delayed),
            "microbatches_this_step": microbatches_this_step,
            "examples_this_step": examples_this_step,
            "cells_this_step": cells_this_step,
            "evolution_loss": mean_evolution_loss,
            "query_loss": mean_query_loss,
            "evolution_loss_by_repeat": mean_evolution_loss_by_repeat,
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
            is_delayed=is_delayed,
            query_repeat=query_repeat,
            recall_age=recall_age,
            target_steps=target_steps,
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
                _trainer_print(
                    args,
                    json.dumps(
                        {
                            "step": completed_step,
                            "train_loss": mean_loss,
                            "grad_norm": grad_norm,
                            "evolution_loss": mean_evolution_loss,
                            "query_loss": mean_query_loss,
                            "active_pair": active_pair_key,
                            "mode": "delayed" if is_delayed else "normal",
                            "query_repeat": query_repeat,
                            "recall_age": recall_age,
                            "query_loss_weight": args.ca_query_loss_weight,
                            "evolution_loss_by_repeat": {
                                f"repeat_{repeat}": value
                                for repeat, value in enumerate(
                                    mean_evolution_loss_by_repeat,
                                    start=1,
                                )
                            },
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
                                "optimizer_steps_by_mode": {
                                    mode: values["optimizer_steps"]
                                    for mode, values in training_exposure[
                                        "by_mode"
                                    ].items()
                                },
                                "delayed_queries_by_pair": {
                                    pair: {
                                        query: values["optimizer_steps"]
                                        for query, values in pair_values[
                                            "delayed_by_query_repeat"
                                        ].items()
                                    }
                                    for pair, pair_values in training_exposure[
                                        "by_training_pair"
                                    ].items()
                                    if pair_values[
                                        "delayed_by_query_repeat"
                                    ]
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
                    "train/evolution_loss": mean_evolution_loss,
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
                if mean_query_loss is not None:
                    logs["train/query_loss"] = mean_query_loss

                logs[
                    f"train_pair/{active_pair_key}/"
                    f"{'delayed' if is_delayed else 'normal'}/evolution_loss"
                ] = mean_evolution_loss

                for repeat, value in enumerate(
                    mean_evolution_loss_by_repeat,
                    start=1,
                ):
                    logs[
                        f"train_pair/{active_pair_key}/"
                        f"{'delayed' if is_delayed else 'normal'}/"
                        f"evolution_repeat_{repeat}_loss"
                    ] = value

                if is_delayed:
                    logs[
                        f"train_pair/{active_pair_key}/delayed/"
                        f"query_repeat_{query_repeat}/query_loss"
                    ] = mean_query_loss

                    logs[
                        f"train_pair/{active_pair_key}/delayed/"
                        f"query_repeat_{query_repeat}/total_loss"
                    ] = mean_loss
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
                    for mode, mode_values in values["by_mode"].items():
                        logs[
                            f"train_pair/{key}/{mode}/examples_seen"
                        ] = mode_values["examples_seen"]
                    for query, query_values in values[
                        "delayed_by_query_repeat"
                    ].items():
                        logs[
                            f"train_pair/{key}/delayed/{query}/examples_seen"
                        ] = query_values["examples_seen"]
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
        if best_delayed_recall_info is None:
            raise RuntimeError(
                "CA training completed without selecting a delayed-recall checkpoint."
            )

        supports_repeat_override = (
            "num_repeats" in inspect.signature(raw_model.forward).parameters
        )
        if train_pairs:
            trained_ca_steps, trained_repeats = best_pair
        else:
            trained_ca_steps = args.ca_steps
            trained_repeats = args.n_repeat if supports_repeat_override else None

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
                    args.ca_repeat_diagnostic_max_repeats
                ),
                repeat_diagnostic_horizons=args.ca_repeat_diagnostic_horizons,
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
        best_delayed_recall_checkpoint_eval = evaluate_checkpoint(
            checkpoint_dir / best_delayed_recall_info["checkpoint"],
            best_delayed_recall_info,
            "best_delayed_recall",
        )
        delayed_test_eval = evaluate_delayed_recall_pairs(
            evaluation_forward,
            test_loaders[best_length],
            args.device,
            pairs=train_pairs,
            max_batches=(
                args.ca_final_eval_max_batches
                or args.ca_eval_max_batches
            ),
            num_examples=2,
            ctx=_autocast_context(args),
        )
        delayed_test_summary = summarize_delayed_recall_pairs(
            delayed_test_eval
        )
        best_delayed_recall_checkpoint_eval["delayed_recall"] = (
            delayed_test_eval
        )
        best_delayed_recall_checkpoint_eval["delayed_recall_summary"] = (
            delayed_test_summary
        )
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

        stats["best_id"] = best_id_info
        stats["best_delayed_recall"] = best_delayed_recall_info
        stats["best_extrapolation_strict"] = best_extrapolation_strict_info
        stats["best_extrapolation_unconstrained"] = (
            best_extrapolation_unconstrained_info
        )
        # Backward-compatible summary fields retain their historical meaning.
        stats["best"] = best_id_info
        stats["best_extrapolation"] = best_extrapolation_strict_info
        stats["final_eval"] = final_eval
        stats["checkpoint_analysis"] = {
            "best_id": best_id_checkpoint_eval,
            "best_delayed_recall": best_delayed_recall_checkpoint_eval,
            "best_extrapolation_strict": (
                best_extrapolation_strict_checkpoint_eval
            ),
            "best_extrapolation_unconstrained": (
                best_extrapolation_unconstrained_checkpoint_eval
            ),
            # Compatibility aliases for existing analysis notebooks.
            "best_in_distribution": best_id_checkpoint_eval,
            "best_extrapolation": best_extrapolation_strict_checkpoint_eval,
        }
        stats["args"] = sanitize_for_json(vars(args))
        _write_json(checkpoint_dir / "best_id_eval.json", final_eval)
        _write_json(checkpoint_dir / "best_eval.json", final_eval)
        _write_json(
            checkpoint_dir / "best_delayed_recall_eval.json",
            best_delayed_recall_checkpoint_eval,
        )
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
        _write_json(checkpoint_dir / "summary.json", stats)
        if args.ca_run_dir is not None:
            write_eval_metrics(args.ca_run_dir, stats)
        final_logs = {
            "iter": args.iterations,
            "best_id/step": best_id_info["step"],
            "best_delayed_recall/step": best_delayed_recall_info["step"],
            "best/step": best_id_info["step"],
        }
        add_scalar_metrics(final_logs, final_eval, prefix="final_best_id")
        add_scalar_metrics(final_logs, final_eval, prefix="final")
        add_scalar_metrics(
            final_logs,
            delayed_test_summary,
            prefix="final_best_delayed_recall",
        )
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
        _wandb_log(args, final_logs, args.iterations)
        _trainer_print(
            args,
            format_ca_pair_lines(
                final_eval["in_distribution"],
                length=best_length,
                split="final_test_best_id",
            ),
        )
        _trainer_print(
            args,
            format_delayed_recall_lines(
                delayed_test_eval,
                split="final_test_best_delayed_recall",
            ),
        )
        _trainer_print(
            args,
            format_delayed_recall_summary(
                delayed_test_summary,
                split="final_test_best_delayed_recall",
            ),
        )
        _trainer_print(
            args,
            format_checkpoint_selections(
                ("best_id", best_id_info),
                ("best_delayed_recall", best_delayed_recall_info),
                ("best_extrapolation_strict", best_extrapolation_strict_info),
                (
                    "best_extrapolation_unconstrained",
                    best_extrapolation_unconstrained_info,
                ),
            ),
        )

    distributed_backend.sync()
    return stats if distributed_backend.is_master_process() else None
