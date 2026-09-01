from optim.runner_utils import (
    InfiniteBatchIterator,
    add_scalar_metrics,
    infinite_batches,
    sanitize_for_json,
    save_model_checkpoint,
    save_training_checkpoint,
)

from optim.runner_utils import (
    load_training_checkpoint,
    make_optimizer,
    make_scheduler,
    resolve_resume_checkpoint,
)


from .ca_forward import (
    CAForwardContext,
    CAForwardPolicy,
)


from .soca_gen import rollout_soca

from .ca_reporting import read_json, write_json

import copy
import inspect
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch


def _autocast_context(args):
    if args.device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=args.dtype)

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

def _finite_or(value, fallback):
    value = float(value)
    return value if math.isfinite(value) else fallback

def encode_soca_inputs(state_sequence,input_vocab_size):
    if state_sequence.ndim != 2:
        raise ValueError("state_sequence must have shape [B, N].")

    batch_size, sequence_length = state_sequence.shape
    if sequence_length % 2 != 0:
        raise ValueError("SOCA sequence length must be even.")

    if not torch.all((state_sequence == 0) | (state_sequence == 1)):
        raise ValueError("SOCA states must be binary.")

    row_length = sequence_length // 2
    input_ids = state_sequence.long().clone()

    if input_vocab_size == 2:
        return input_ids
    # Previous half remains 0/1; current half becomes 2/3.
    if input_vocab_size == 4:
        input_ids[:, row_length:] += 2
        return input_ids
    raise ValueError("SOCA input_vocab_size must currently be 2 or 4.")


def encode_soca_targets(targets_by_repeat, output_vocab_size):
    """Encode exact raw binary targets for the selected decoder."""
    if targets_by_repeat.ndim != 3:
        raise ValueError("targets_by_repeat must have shape [B, R, N].")

    targets = targets_by_repeat.long()

    if output_vocab_size == 2:
        return targets

    if output_vocab_size == 4:
        sequence_length = targets.shape[-1]
        row_length = sequence_length // 2

        targets = targets.clone()
        targets[:, :, row_length:] += 2
        return targets

    raise ValueError(
        "SOCA output_vocab_size must currently be 2 or 4."
    )

def soca_component_masks(direction_schedule, sequence_length):
    """Return boolean [B, R, N] computed and copied masks."""
    if direction_schedule.ndim != 2:
        raise ValueError("direction_schedule must have shape [B, R].")
    if sequence_length % 2 != 0:
        raise ValueError("SOCA sequence length must be even.")

    device = direction_schedule.device
    row_length = sequence_length // 2

    second_half = (
        torch.arange(sequence_length, device=device) >= row_length
    )  # [N]

    reverse = direction_schedule.bool().unsqueeze(-1)  # [B, R, 1]

    # Forward: second half computed.
    # Reverse: first half computed.
    computed_mask = torch.where(
        reverse,
        ~second_half,
        second_half,
    )

    copied_mask = ~computed_mask
    return computed_mask, copied_mask


def new_soca_counters(num_repeats, device):
    """Create exact numerator/denominator counters for one training step."""
    scalar_names = (
        "total_correct",
        "total_positions",
        "computed_correct",
        "computed_positions",
        "copied_correct",
        "copied_positions",
        "exact_states",
        "states",
        "forward_transitions",
        "reverse_transitions",
        "switch_examples",
        "examples",
    )
    counters = {
        name: torch.zeros((), dtype=torch.long, device=device)
        for name in scalar_names
    }
    for role in ("computed", "copied"):
        counters[f"{role}_correct_by_repeat"] = torch.zeros(
            num_repeats, dtype=torch.long, device=device
        )
        counters[f"{role}_positions_by_repeat"] = torch.zeros(
            num_repeats, dtype=torch.long, device=device
        )
    return counters


@torch.no_grad()
def update_soca_counters(
    counters,
    repeat_logits,
    raw_targets_by_repeat,
    direction_schedule,
    *,
    output_vocab_size,
):
    """Accumulate exact class accuracies for one local microbatch."""
    predictions = repeat_logits.argmax(dim=-1)
    targets = encode_soca_targets(
        raw_targets_by_repeat,
        output_vocab_size,
    )
    if tuple(predictions.shape) != tuple(targets.shape):
        raise ValueError("SOCA prediction and target shapes do not align.")

    correct = predictions.eq(targets)
    computed_mask, copied_mask = soca_component_masks(
        direction_schedule,
        repeat_logits.shape[2],
    )

    counters["total_correct"] += correct.sum()
    counters["total_positions"] += correct.numel()
    counters["computed_correct"] += (correct & computed_mask).sum()
    counters["computed_positions"] += computed_mask.sum()
    counters["copied_correct"] += (correct & copied_mask).sum()
    counters["copied_positions"] += copied_mask.sum()

    exact_states = correct.all(dim=-1)
    counters["exact_states"] += exact_states.sum()
    counters["states"] += exact_states.numel()
    counters["forward_transitions"] += direction_schedule.eq(0).sum()
    counters["reverse_transitions"] += direction_schedule.eq(1).sum()
    has_forward = direction_schedule.eq(0).any(dim=1)
    has_reverse = direction_schedule.eq(1).any(dim=1)
    counters["switch_examples"] += (has_forward & has_reverse).sum()
    counters["examples"] += direction_schedule.shape[0]

    counters["computed_correct_by_repeat"] += (
        correct & computed_mask
    ).sum(dim=(0, 2))
    counters["computed_positions_by_repeat"] += computed_mask.sum(dim=(0, 2))
    counters["copied_correct_by_repeat"] += (
        correct & copied_mask
    ).sum(dim=(0, 2))
    counters["copied_positions_by_repeat"] += copied_mask.sum(dim=(0, 2))


def reduce_soca_counters(counters):
    """Sum counters across ranks before calculating any ratios."""
    reduced = {name: value.clone() for name, value in counters.items()}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for value in reduced.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return reduced


def _safe_counter_ratio(numerator, denominator):
    denominator = int(denominator.item())
    if denominator == 0:
        return 0.0
    return float(numerator.item()) / denominator


def finalize_soca_counters(counters):
    """Convert reduced counters into JSON- and logging-friendly metrics."""
    metrics = {
        "accuracy": _safe_counter_ratio(
            counters["total_correct"], counters["total_positions"]
        ),
        "computed_accuracy": _safe_counter_ratio(
            counters["computed_correct"], counters["computed_positions"]
        ),
        "copied_accuracy": _safe_counter_ratio(
            counters["copied_correct"], counters["copied_positions"]
        ),
        "exact_state_accuracy": _safe_counter_ratio(
            counters["exact_states"], counters["states"]
        ),
        "forward_transitions": int(counters["forward_transitions"].item()),
        "reverse_transitions": int(counters["reverse_transitions"].item()),
        "switch_examples": int(counters["switch_examples"].item()),
        "examples": int(counters["examples"].item()),
    }
    for role in ("computed", "copied"):
        metrics[f"{role}_accuracy_by_repeat"] = [
            _safe_counter_ratio(correct, total)
            for correct, total in zip(
                counters[f"{role}_correct_by_repeat"],
                counters[f"{role}_positions_by_repeat"],
            )
        ]
    return metrics


def soca_loss(
    repeat_logits,
    raw_targets_by_repeat,
    direction_schedule,
    *,
    output_vocab_size,
    copy_loss_weight,
):
    """
    repeat_logits:         [B, R, N, C]
    raw_targets_by_repeat: [B, R, N]
    direction_schedule:    [B, R]
    """
    if repeat_logits.ndim != 4:
        raise ValueError("repeat_logits must have shape [B, R, N, C].")

    batch_size, repeats, sequence_length, classes = repeat_logits.shape

    if classes != output_vocab_size:
        raise ValueError("Logit class dimension disagrees with output vocabulary.")

    expected_target_shape = (batch_size, repeats, sequence_length)
    if tuple(raw_targets_by_repeat.shape) != expected_target_shape:
        raise ValueError("Target and logit shapes do not align.")

    if tuple(direction_schedule.shape) != (batch_size, repeats):
        raise ValueError("Direction schedule and repeat logits do not align.")

    targets = encode_soca_targets(
        raw_targets_by_repeat,
        output_vocab_size,
    )

    element_loss = torch.nn.functional.cross_entropy(
        repeat_logits.reshape(-1, classes),
        targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, repeats, sequence_length)

    computed_mask, copied_mask = soca_component_masks(
        direction_schedule,
        sequence_length,
    )

    # [R]: equal normalization within every repeat.
    computed_loss_by_repeat = (
        (element_loss * computed_mask).sum(dim=(0, 2))
        / computed_mask.sum(dim=(0, 2))
    )
    copied_loss_by_repeat = (
        (element_loss * copied_mask).sum(dim=(0, 2))
        / copied_mask.sum(dim=(0, 2))
    )

    computed_loss = computed_loss_by_repeat.mean()
    copied_loss = copied_loss_by_repeat.mean()

    total_loss = computed_loss + copy_loss_weight * copied_loss

    return {
        "loss": total_loss,
        "computed_loss": computed_loss,
        "copied_loss": copied_loss,
        "computed_loss_by_repeat": computed_loss_by_repeat,
        "copied_loss_by_repeat": copied_loss_by_repeat,
    }


def soca_train(
    model,
    optimizer,
    scheduler,
    train_loader,
    eval_loader,
    test_loader,
    args,
    distributed_backend,
    checkpoint_dir,
    *,
    start_step=0,
    data_state=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    raw_model = distributed_backend.get_raw_model(model)
    needs_iter = bool(getattr(raw_model, "needs_iter", False))
    forward_policy = CAForwardPolicy.from_args(args)
    training_forward = CAForwardContext(model, forward_policy)
    evaluation_forward = CAForwardContext(raw_model, forward_policy)
    pin_memory = args.device.type == "cuda"
    log_every = getattr(args, "soca_log_every", None) or max(
        1, min(100, args.eval_freq)
    )
    save_every = getattr(args, "soca_save_every", None) or args.eval_freq

    if args.soca_exact_data_resume:
        batch_iterator = InfiniteBatchIterator(
            train_loader,
            state=data_state,
        )
    else:
        batch_iterator = infinite_batches(train_loader)

    def current_data_state():
        if args.soca_exact_data_resume:
            return batch_iterator.state_dict()
        return {}

    for step in range(start_step, args.iterations):
        if step % args.eval_freq == 0:
            # TODO Call SOCA evaluator.
            # TODO Select/save best checkpoint on validation metrics.
            pass

        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        accumulated_computed_loss = 0.0
        accumulated_copied_loss = 0.0
        step_counters = None

        for microstep_index in range(args.acc_steps):
            batch = next(batch_iterator)
            state_sequence = batch["state_sequence"].to(
                args.device,
                dtype=torch.long,
                non_blocking=pin_memory,
            )
            direction_schedule = batch["direction_schedule"].to(
                args.device,
                dtype=torch.long,
                non_blocking=pin_memory,
            )

            with torch.no_grad():
                targets_by_repeat = rollout_soca(
                    state_sequence,
                    direction_schedule,
                )
                input_ids = encode_soca_inputs(
                    state_sequence,
                    raw_model.config.input_vocab_size,
                )

            forward_kwargs = {
                "direction_schedule": direction_schedule,
                "num_repeats": direction_schedule.shape[1],
                "return_repeat_logits": True,
                "get_logits": False,
            }
            if needs_iter:
                forward_kwargs["iter"] = step

            with _autocast_context(args):
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_index,
                    gradient_accumulation_steps=args.acc_steps,
                ):
                    outputs = training_forward.call(input_ids, **forward_kwargs)
                    repeat_logits = outputs.get("repeat_logits")
                    if repeat_logits is None:
                        raise KeyError("SOCA model did not return repeat_logits.")
                    losses = soca_loss(
                        repeat_logits,
                        targets_by_repeat,
                        direction_schedule,
                        output_vocab_size=raw_model.config.output_vocab_size,
                        copy_loss_weight=args.soca_copy_loss_weight,
                    )
                    loss = losses["loss"]

            (loss / args.acc_steps).backward()
            accumulated_loss += float(loss.detach().float().item())
            accumulated_computed_loss += float(
                losses["computed_loss"].detach().float().item()
            )
            accumulated_copied_loss += float(
                losses["copied_loss"].detach().float().item()
            )
            if step_counters is None:
                step_counters = new_soca_counters(
                    repeat_logits.shape[1], repeat_logits.device
                )
            update_soca_counters(
                step_counters,
                repeat_logits,
                targets_by_repeat,
                direction_schedule,
                output_vocab_size=raw_model.config.output_vocab_size,
            )

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                math.inf,
            )

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
        mean_computed_loss = _distributed_mean(
            accumulated_computed_loss / args.acc_steps,
            args.device,
            distributed_backend.get_world_size(),
        )
        mean_copied_loss = _distributed_mean(
            accumulated_copied_loss / args.acc_steps,
            args.device,
            distributed_backend.get_world_size(),
        )
        if step_counters is None:
            raise RuntimeError("SOCA training step consumed no microbatches.")
        train_metrics = finalize_soca_counters(
            reduce_soca_counters(step_counters)
        )

        if completed_step % log_every == 0 or completed_step == args.iterations:
            logs = {
                "step": completed_step,
                "train/loss": mean_loss,
                "train/computed_loss": mean_computed_loss,
                "train/copied_loss": mean_copied_loss,
                "train/grad_norm": grad_norm,
                "train/accuracy": train_metrics["accuracy"],
                "train/computed_accuracy": train_metrics["computed_accuracy"],
                "train/copied_accuracy": train_metrics["copied_accuracy"],
                "train/exact_state_accuracy": train_metrics[
                    "exact_state_accuracy"
                ],
                "train/forward_transitions": train_metrics[
                    "forward_transitions"
                ],
                "train/reverse_transitions": train_metrics[
                    "reverse_transitions"
                ],
                "train/switch_examples": train_metrics["switch_examples"],
            }
            for role in ("computed", "copied"):
                for repeat_index, accuracy in enumerate(
                    train_metrics[f"{role}_accuracy_by_repeat"], start=1
                ):
                    logs[
                        f"train/{role}_accuracy_repeat_{repeat_index}"
                    ] = accuracy
            if distributed_backend.is_master_process():
                print(json.dumps(logs), flush=True)
                _wandb_log(args, logs, completed_step)

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

    save_training_checkpoint(
        checkpoint_dir / f"ckpt_{args.iterations}.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=args.iterations,
        distributed_backend=distributed_backend,
        data_state=current_data_state(),
    )
