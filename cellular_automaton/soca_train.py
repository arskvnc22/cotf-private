from optim.runner_utils import (
    InfiniteBatchIterator,
    infinite_batches,
    save_training_checkpoint,
)
from .ca_forward import (
    CAForwardContext,
    CAForwardPolicy,
)
from .soca_gen import rollout_soca
from .soca_exposure import (
    add_soca_training_exposure,
    copy_soca_training_exposure,
    new_soca_step_exposure,
    new_soca_training_exposure,
    update_soca_step_exposure,
)

import copy
import json
import math
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


def encode_soca_inputs(state_sequence, input_vocab_size):
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

    raise ValueError("SOCA output_vocab_size must currently be 2 or 4.")


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
        "forward_correct",
        "forward_positions",
        "reverse_correct",
        "reverse_positions",
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
    forward_position_mask = (
        direction_schedule.eq(0).unsqueeze(-1).expand_as(correct)
    )
    reverse_position_mask = (
        direction_schedule.eq(1)
        .unsqueeze(-1)
        .expand_as(correct)
    )

    counters["total_correct"] += correct.sum()
    counters["total_positions"] += correct.numel()
    counters["computed_correct"] += (correct & computed_mask).sum()
    counters["computed_positions"] += computed_mask.sum()
    counters["copied_correct"] += (correct & copied_mask).sum()
    counters["copied_positions"] += copied_mask.sum()
    counters["forward_correct"] += (correct & forward_position_mask).sum()
    counters["forward_positions"] += forward_position_mask.sum()

    counters["reverse_correct"] += (correct & reverse_position_mask).sum()
    counters["reverse_positions"] += reverse_position_mask.sum()

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
        "forward_accuracy": _safe_counter_ratio(
            counters["forward_correct"],
            counters["forward_positions"],
        ),
        "reverse_accuracy": _safe_counter_ratio(
            counters["reverse_correct"],
            counters["reverse_positions"],
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


@torch.no_grad()
def evaluate_soca_validation(evaluation_forward, eval_loaders, args):
    """Evaluate the fixed named validation conditions used by the dry run."""
    # Keep this import local because soca_eval imports the task encoding helpers
    # from this module.
    from .soca_eval import evaluate_soca_model

    results = {}
    for condition, dataloader in eval_loaders.items():
        results[str(condition)] = evaluate_soca_model(
            evaluation_forward,
            dataloader,
            args.device,
            copy_loss_weight=args.soca_copy_loss_weight,
            target_steps_per_repeat=1,
            max_batches=args.soca_eval_max_batches,
            ctx=_autocast_context(args),
        )
    if not results:
        raise ValueError("SOCA validation requires at least one condition loader.")
    return results


def summarize_soca_validation(evaluations):
    """Return the small scalar subset intended for terminal and W&B logs."""
    summary = {}
    for condition, result in evaluations.items():
        overall = result["overall"]
        summary[condition] = {
            "loss": overall["configured_loss"],
            "state_accuracy": overall["state"]["cell_accuracy"],
            "exact_state_accuracy": overall["state"]["exact_row_accuracy"],
            "computed_accuracy": overall["computed"]["cell_accuracy"],
            "copied_accuracy": overall["copied"]["cell_accuracy"],
            "joint_pair_cell_accuracy": overall[
                "joint_pair_cell_accuracy"
            ],
            "by_repeat": {
                repeat: {
                    "state_accuracy": values["state"]["cell_accuracy"],
                    "computed_accuracy": values["computed"]["cell_accuracy"],
                    "copied_accuracy": values["copied"]["cell_accuracy"],
                    "exact_state_accuracy": values["state"][
                        "exact_row_accuracy"
                    ],
                }
                for repeat, values in result["by_repeat"].items()
            },
        }
    return summary


def soca_train(
    model,
    optimizer,
    scheduler,
    train_loader,
    eval_loaders,
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
    save_every = getattr(args, "soca_save_every", None) or args.iterations

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

    training_exposure = new_soca_training_exposure(
        materialized_training_rows=len(train_loader.dataset),
        num_repeats=args.n_repeat,
        copy_loss_weight=args.soca_copy_loss_weight,
    )
    evaluation_history = {}
    last_checkpoint_step = None

    def evaluate(completed_step):
        evaluations = evaluate_soca_validation(
            evaluation_forward,
            eval_loaders,
            args,
        )
        summary = summarize_soca_validation(evaluations)
        evaluation_history[str(completed_step)] = {
            "conditions": evaluations,
            "training_exposure": copy_soca_training_exposure(
                training_exposure
            ),
        }
        if distributed_backend.is_master_process():
            print(
                json.dumps(
                    {
                        "step": int(completed_step),
                        "validation": summary,
                        "training_exposure": copy_soca_training_exposure(
                            training_exposure
                        ),
                    }
                ),
                flush=True,
            )
            logs = {"step": int(completed_step)}
            for condition, condition_summary in summary.items():
                for name in (
                    "loss",
                    "state_accuracy",
                    "exact_state_accuracy",
                    "computed_accuracy",
                    "copied_accuracy",
                    "joint_pair_cell_accuracy",
                ):
                    value = condition_summary[name]
                    if value is not None:
                        logs[f"val/{condition}/{name}"] = float(value)
                for repeat, repeat_summary in condition_summary[
                    "by_repeat"
                ].items():
                    for name, value in repeat_summary.items():
                        if value is not None:
                            logs[
                                f"val/{condition}/{repeat}/{name}"
                            ] = float(value)
            _wandb_log(args, logs, completed_step)

    evaluate(start_step)

    for step in range(start_step, args.iterations):

        model.train()
        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        accumulated_computed_loss = 0.0
        accumulated_copied_loss = 0.0
        step_counters = None
        step_exposure = new_soca_step_exposure(args.n_repeat)

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
            update_soca_step_exposure(
                step_exposure,
                state_sequence,
                direction_schedule,
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
        add_soca_training_exposure(training_exposure, step_exposure)
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
                "train/forward_accuracy": train_metrics["forward_accuracy"],
                "train/reverse_accuracy": train_metrics["reverse_accuracy"],
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
            exposure_snapshot = copy_soca_training_exposure(
                training_exposure
            )
            logs["exposure/examples"] = exposure_snapshot[
                "total_examples_seen"
            ]
            logs["exposure/supervised_states"] = exposure_snapshot[
                "total_supervised_states"
            ]
            logs["exposure/equivalent_dataset_passes"] = exposure_snapshot[
                "equivalent_dataset_passes"
            ]
            for repeat, values in exposure_snapshot["by_repeat"].items():
                logs[f"exposure/{repeat}/forward_fraction"] = values[
                    "forward_fraction"
                ]
                logs[f"exposure/{repeat}/reverse_fraction"] = values[
                    "reverse_fraction"
                ]
            if distributed_backend.is_master_process():
                print(
                    json.dumps(
                        {
                            **logs,
                            "training_exposure": exposure_snapshot,
                        }
                    ),
                    flush=True,
                )
                _wandb_log(args, logs, completed_step)

        if (
            completed_step % args.eval_freq == 0
            or completed_step == args.iterations
        ):
            evaluate(completed_step)

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
            last_checkpoint_step = completed_step

    if last_checkpoint_step != args.iterations:
        save_training_checkpoint(
            checkpoint_dir / f"ckpt_{args.iterations}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=args.iterations,
            distributed_backend=distributed_backend,
            data_state=current_data_state(),
        )
    return {
        "evaluation_history": evaluation_history,
        "training_exposure": copy_soca_training_exposure(training_exposure),
    }
