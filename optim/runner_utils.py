"""Task-neutral utilities shared by dedicated experiment runners."""

import inspect
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch


def print_master(distributed_backend, message):
    if distributed_backend.is_master_process():
        print(message)


def infinite_batches(dataloader, *, start_epoch=0, start_batch=0):
    """Yield batches forever while advancing DistributedSampler epochs."""
    epoch = start_epoch
    first_epoch = True
    while True:
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(dataloader):
            if first_epoch and batch_index < start_batch:
                continue
            yield batch
        first_epoch = False
        start_batch = 0
        epoch += 1


class InfiniteBatchIterator:
    """Cycle a DataLoader while exposing enough state for ordered resume.

    The generator state is captured immediately before an epoch iterator is
    created.  Restoring that state and skipping ``batch_offset`` batches
    reproduces the same RandomSampler order in single-process runs.  Under DDP,
    ``DistributedSampler.set_epoch`` provides the corresponding deterministic
    ordering.
    """

    def __init__(self, dataloader, state=None):
        if len(dataloader) == 0:
            raise ValueError("Cannot cycle an empty DataLoader.")
        state = state or {}
        self.dataloader = dataloader
        self.epoch = int(state.get("epoch", 0))
        target_batch_offset = int(state.get("batch_offset", 0))
        if not 0 <= target_batch_offset <= len(dataloader):
            raise ValueError(
                "Checkpoint batch_offset is outside the DataLoader epoch: "
                f"{target_batch_offset} versus {len(dataloader)} batches."
            )

        generator_state = state.get("epoch_generator_state")
        generator = getattr(dataloader, "generator", None)
        if generator_state is not None and generator is not None:
            generator.set_state(generator_state.cpu())

        self.batch_offset = 0
        self.epoch_generator_state = None
        self._open_epoch()
        for _ in range(target_batch_offset):
            try:
                next(self._iterator)
            except StopIteration as error:
                raise ValueError(
                    "Checkpoint batch_offset exceeds the restored epoch."
                ) from error
            self.batch_offset += 1

    def _open_epoch(self):
        sampler = getattr(self.dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)
        generator = getattr(self.dataloader, "generator", None)
        if generator is not None:
            self.epoch_generator_state = generator.get_state().clone()
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batch_offset = 0
            self._open_epoch()
            batch = next(self._iterator)
        self.batch_offset += 1
        return batch

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "batch_offset": self.batch_offset,
            "epoch_generator_state": self.epoch_generator_state,
        }


def make_optimizer(args, model, distributed_backend, device_type):
    """Build an optimizer from the repository's parameter-group contract."""
    raw_model = distributed_backend.get_raw_model(model)
    group_specs = raw_model.get_parameter_group_specs()
    parameter_map = dict(model.named_parameters())
    optimized_parameters = 0

    for group in group_specs:
        parameters = []
        for parameter_name in group["params"]:
            translated_names = distributed_backend.translate_model_parameter_name_for_node(
                parameter_name
            )
            parameters.extend(parameter_map[name] for name in translated_names)
        group["params"] = parameters
        optimized_parameters += sum(parameter.numel() for parameter in parameters)

    print_master(
        distributed_backend,
        f"number of optimized parameters: {optimized_parameters / 1e6:.2f}M",
    )
    if args.opt == "adamw":
        use_fused = device_type == "cuda" and "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters
        print_master(distributed_backend, f"using fused AdamW: {use_fused}")
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            **extra_args,
        )
    if args.opt == "adafactor":
        from optim.adafactor import Adafactor

        return Adafactor(group_specs, lr=args.lr)
    return torch.optim.SGD(
        group_specs,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )


def make_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler not in {"cos", "linear"}:
        raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.iterations,
        pct_start=args.warmup_percent,
        anneal_strategy=args.scheduler,
        cycle_momentum=False,
        div_factor=1e2,
        final_div_factor=args.final_div_factor,
    )


def sanitize_for_json(value):
    """Recursively convert common experiment objects to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def to_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().float().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_scalar_metrics(logs, metrics, *, prefix):
    """Flatten nested scalar dictionaries into slash-delimited log keys."""
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            add_scalar_metrics(logs, value, prefix=full_key)
            continue
        scalar = to_scalar(value)
        if scalar is not None:
            logs[full_key] = scalar


def latest_checkpoint_name(checkpoint_dir):
    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith("ckpt_") or not filename.endswith(".pt"):
            continue
        step_text = filename[len("ckpt_") : -len(".pt")]
        if step_text.isdigit():
            candidates.append((int(step_text), filename))
    return max(candidates)[1] if candidates else None


def _local_rng_state():
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def _gather_rng_states(distributed_backend):
    local_state = _local_rng_state()
    if distributed_backend.get_world_size() == 1:
        return [local_state]

    # Avoid all_gather_object under NCCL: it can allocate unexpectedly large
    # buffers in affected PyTorch versions.  RNG payloads are tiny, so serialize
    # them into explicitly sized byte tensors instead.
    payload = pickle.dumps(local_state, protocol=pickle.HIGHEST_PROTOCOL)
    collective_device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.distributed.get_backend() == "nccl"
        else torch.device("cpu")
    )
    payload_tensor = torch.tensor(
        list(payload), dtype=torch.uint8, device=collective_device
    )
    payload_size = torch.tensor(
        [payload_tensor.numel()], dtype=torch.long, device=collective_device
    )
    gathered_sizes = [torch.zeros_like(payload_size) for _ in range(
        distributed_backend.get_world_size()
    )]
    torch.distributed.all_gather(gathered_sizes, payload_size)
    maximum_size = max(int(size.item()) for size in gathered_sizes)
    padded_payload = torch.zeros(
        maximum_size, dtype=torch.uint8, device=collective_device
    )
    padded_payload[: payload_tensor.numel()] = payload_tensor
    gathered_payloads = [torch.empty_like(padded_payload) for _ in range(
        distributed_backend.get_world_size()
    )]
    torch.distributed.all_gather(gathered_payloads, padded_payload)

    if not distributed_backend.is_master_process():
        return None
    return [
        pickle.loads(bytes(tensor[: int(size.item())].cpu().tolist()))
        for tensor, size in zip(gathered_payloads, gathered_sizes)
    ]


def _restore_rng_state(checkpoint, distributed_backend):
    states = checkpoint.get("rng_states_by_rank")
    if not states:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    state = states[rank] if rank < len(states) else states[0]
    torch.set_rng_state(state["torch_cpu"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"].cpu())


def save_training_checkpoint(
    checkpoint_path,
    *,
    model,
    optimizer,
    scheduler,
    step,
    distributed_backend,
    data_state=None,
):
    """Save resumable state; every rank must call this function."""
    rng_states = _gather_rng_states(distributed_backend)
    if distributed_backend.is_master_process():
        checkpoint = {
            "model": distributed_backend.get_raw_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "itr": int(step),
            "rng_states_by_rank": rng_states,
            "data_state": data_state or {},
        }
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)
    distributed_backend.sync()


def save_model_checkpoint(
    checkpoint_path,
    *,
    model,
    step,
    metadata=None,
):
    """Save a lightweight rank-zero checkpoint used for best-model evaluation."""
    torch.save(
        {
            "model": model.state_dict(),
            "itr": int(step),
            "metadata": metadata or {},
        },
        checkpoint_path,
    )


def resolve_resume_checkpoint(args, checkpoint_dir):
    requested = getattr(args, "use_pretrained", None)
    if requested in {None, False, "None", "none"}:
        return None
    if requested == "auto":
        requested = latest_checkpoint_name(checkpoint_dir)
        if requested is None:
            return None
    path = Path(requested)
    return path if path.is_absolute() else Path(checkpoint_dir) / path


def load_training_checkpoint(
    checkpoint_path,
    *,
    model,
    optimizer,
    scheduler,
    distributed_backend,
    device,
):
    """Load a resumable checkpoint on every rank."""
    if checkpoint_path is None:
        return 0, {}
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_rng_state(checkpoint, distributed_backend)
    return int(checkpoint["itr"]), checkpoint.get("data_state", {})
