"""Evaluate frozen DCA checkpoints, predictions, attention, and cache interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from numbers import Integral
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config
import models

try:
    from .ca_eval import (
        _forward_all_cells,
        evaluate_ca_repeat_horizon_diagnostics,
    )
    from .ca_forward import CAForwardContext, CAForwardPolicy
    from .ca_gen import MaterializedRule30Dataset, Rule30Dataset, rule30
    from .ca_reporting import read_json, utc_now, validate_manifest, write_json
    from .dca_eval import evaluate_delayed_recall_query
except ImportError:
    from ca_eval import _forward_all_cells, evaluate_ca_repeat_horizon_diagnostics
    from ca_forward import CAForwardContext, CAForwardPolicy
    from ca_gen import MaterializedRule30Dataset, Rule30Dataset, rule30
    from ca_reporting import read_json, utc_now, validate_manifest, write_json
    from dca_eval import evaluate_delayed_recall_query


ARTIFACT_SCHEMA_VERSION = 1
BATCH_SHARD_SCHEMA_VERSION = 1
FACTORIAL_CAPTURE_SCHEMA_VERSION = 2
DEFAULT_OUTPUT_ROOT = Path("/scratch/ab3u21/dca-attention-evaluations")

_FACTORIAL_CONDITION_FACTORS = {
    "free_baseline": (False, False),
    "clean_baseline": (True, False),
    "free_reset": (False, True),
    "clean_reset": (True, True),
}

FACTORIAL_EFFECT_CONDITIONS = {
    "clean_state_with_history": ("clean_baseline", "free_baseline"),
    "clean_state_after_reset": ("clean_reset", "free_reset"),
    "cache_reset_on_free_state": ("free_reset", "free_baseline"),
    "cache_reset_on_clean_state": ("clean_reset", "clean_baseline"),
}

DCA_EVALUATION_MODELS = {"dca_but", "dca_cotf_cache"}
DCA_CACHE_INTERVENTION_MODEL = "dca_cotf_cache"
DCA_CONDITIONS = (
    "baseline",
    "target-value-corruption",
    "target-repeat-only",
    "target-repeat-masked",
)


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--artifact-name",
        help=(
            "Unique basename for a persistent diagnostic artifact. Existing "
            "artifact directories are never overwritten."
        ),
    )
    parser.add_argument(
        "--diagnostic-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--diagnostic-data-mode",
        choices=("manifest", "indexed", "materialized"),
        default="manifest",
    )
    parser.add_argument("--probe-forward", action="store_true")
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--capture-first-condition-batch",
        "--capture-first-baseline-batch",
        dest="capture_first_condition_batch",
        action="store_true",
        help=(
            "Run and atomically store one maximum-depth manual-attention "
            "diagnostic batch for the requested cache condition."
        ),
    )
    capture_group.add_argument(
        "--capture-factorial-transition-batch",
        action="store_true",
        help="Capture one paired clean-state/cache-reset factorial batch.",
    )
    parser.add_argument(
        "--evaluate-factorial-full-split",
        action="store_true",
        help="Evaluate paired factorial outcomes over the complete fixed split.",
    )
    parser.add_argument(
        "--intervention-source-depth",
        type=_nonnegative_int,
        help="Source depth t for interventions applied when computing repeat t+1.",
    )
    parser.add_argument(
        "--repeat-cache-window",
        type=_positive_int,
        help=(
            "Expose only this many recent repeat blocks. Omit for the exact "
            "full-cache path."
        ),
    )
    parser.add_argument(
        "--verify-full-cache-none",
        action="store_true",
        help=(
            "On a full-cache condition, compare an omitted cache-window "
            "argument with explicit repeat_cache_window=None."
        ),
    )
    parser.add_argument(
        "--compare-repeat-horizon-backends",
        action="store_true",
        help=(
            "Evaluate the full fixed split with standalone SDPA and manual "
            "attention forwards and report compact repeat-versus-horizon matrices."
        ),
    )
    parser.add_argument("--max-repeats", type=int)
    parser.add_argument(
        "--equivalence-repeats", type=int, nargs="+", default=(1, 10, 17, 20)
    )
    parser.add_argument(
        "--evaluate-delayed-recall",
        action="store_true",
        help=(
            "Evaluate and decode DCA delayed-recall predictions from the "
            "selected frozen checkpoint."
        ),
    )
    parser.add_argument(
        "--num-repeats",
        type=_positive_int,
        help="Evolution depth used by the standalone delayed-recall evaluation.",
    )
    parser.add_argument(
        "--query-repeats",
        type=_positive_int,
        nargs="+",
        help="Requested evolution repeats to inspect; defaults to every repeat.",
    )
    parser.add_argument(
        "--num-recall-repeats",
        type=_positive_int,
        default=2,
        help="Number of recall iterations; the final iteration is decoded.",
    )
    parser.add_argument(
        "--conditions",
        choices=DCA_CONDITIONS,
        nargs="+",
        default=("baseline",),
        help=(
            "Paired inference conditions. Intervention requests automatically "
            "include a baseline on the same attention backend."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=_positive_int,
        help="Optional cap on fixed-split batches per query and condition.",
    )
    parser.add_argument(
        "--num-examples",
        type=_nonnegative_int,
        default=4,
        help="Decoded example predictions retained per query and condition.",
    )
    parser.add_argument(
        "--collect-recall-attention",
        action="store_true",
        help=(
            "Aggregate manual-attention diagnostics for the recall iterations. "
            "Supported by dca_cotf_cache only."
        ),
    )
    parser.add_argument(
        "--intervention-permutation-offset",
        type=_positive_int,
        default=1,
        help=(
            "Cyclic batch-permutation offset for target-repeat value "
            "corruption; reduced modulo each batch size."
        ),
    )
    return parser.parse_args(argv)


def _require_mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(path, value): # prevents a partially written final file from appearingvalid if process crashes during torch save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) # ensure exists parent dir
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_artifact_name(name):
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", name
    ):
        raise ValueError(
            "artifact name must be a basename containing only letters, "
            "numbers, '.', '_' and '-'"
        )
    return name


def _condition_spec(backend, repeat_cache_window):
    if repeat_cache_window is None:
        condition_id = f"{backend}_full"
        cache_policy = "full"
    else:
        condition_id = f"{backend}_recent_{repeat_cache_window}"
        cache_policy = "recent"
    return {
        "condition_id": condition_id,
        "attention_implementation": backend,
        "cache_policy": cache_policy,
        "cache_window": repeat_cache_window,
        "repeat_cache_window": repeat_cache_window,
        "attention_normalizer": "softmax",
        "normalization_intervention": "none",
        "logit_scaling": "none",
        "probability_transform": "none",
    }


def _requested_conditions(args, model_args):
    requested = set()
    requested_window = getattr(args, "repeat_cache_window", None)
    configured_backend = getattr(model_args, "attention_implementation", "sdpa")
    if args.probe_forward:
        requested.update((("manual", None), (configured_backend, None)))
    if getattr(args, "capture_first_condition_batch", False):
        requested.add(("manual", requested_window))
    if args.compare_repeat_horizon_backends:
        requested.update((("sdpa", None), ("manual", None)))
    return [
        _condition_spec(backend, repeat_cache_window)
        for backend in ("sdpa", "manual")
        for repeat_cache_window in (
            (None,) if requested_window is None else (None, requested_window)
        )
        if (backend, repeat_cache_window) in requested
    ]


def _write_and_validate_storage_probe(artifact_dir):
    probe = {
        "bf16_values": torch.tensor(
            [[-1.5, 0.0, 2.25], [3.0, -4.5, 8.0]], dtype=torch.bfloat16
        ),
        "fp32_values": torch.tensor([0.125, -2.5, 16.0], dtype=torch.float32),
        "populated_mask": torch.tensor(
            [[True, False, False], [True, True, False]], dtype=torch.bool
        ),
        "visible_mask": torch.tensor(
            [[True, False, False], [True, False, False]], dtype=torch.bool
        ),
        "example_ids": torch.tensor([0, 127], dtype=torch.int64),
    }
    probe_path = Path(artifact_dir) / "storage_probe.pt"
    _atomic_torch_save(probe_path, probe)
    loaded = torch.load(probe_path, map_location="cpu", weights_only=False)
    if loaded.keys() != probe.keys() or any(
        loaded[name].dtype != expected.dtype
        or not torch.equal(loaded[name], expected)
        for name, expected in probe.items()
    ):
        raise RuntimeError("storage probe did not survive an exact save/load cycle")
    return {
        "status": "passed",
        "path": probe_path.name,
        "size_bytes": probe_path.stat().st_size,
        "sha256": _sha256_file(probe_path),
        "tensor_dtypes": {name: str(value.dtype) for name, value in probe.items()},
    }


def _to_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, Mapping):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _iter_tensors(value):  # recursively visit nested item and forward every tensor it produces
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _tensor_bytes(value):
    return sum(tensor.numel() * tensor.element_size() for tensor in _iter_tensors(value)) # calculate storage


def _assert_exact_tree(left, right, path="root"): # left = CPU shart immediately before saving right = shard reloaded from the pt.file
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise RuntimeError(f"{path} changed from a tensor during save/load")
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise RuntimeError(f"{path} changed during save/load")
        return
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            raise RuntimeError(f"{path} mapping keys changed during save/load")
        for key in left:
            _assert_exact_tree(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise RuntimeError(f"{path} sequence changed during save/load")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_exact_tree(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise RuntimeError(f"{path} changed during save/load")


def _validate_captured_shard(shard, model, max_repeats):
    if shard.get("experiment_kind") == "clean_state_cache_reset_factorial":
        raise ValueError("factorial captures require the factorial validator")
    tensors = list(_iter_tensors(shard))
    if not tensors:
        raise ValueError("captured shard contains no tensors")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("captured shard contains a non-CPU tensor")
    if any(tensor.requires_grad for tensor in tensors):
        raise ValueError("captured shard unexpectedly retains gradients")
    if any(
        not torch.isfinite(tensor).all()
        for tensor in tensors
        if tensor.is_floating_point()
    ):
        raise ValueError("captured shard contains non-finite floating-point values")

    inputs = shard["inputs"]
    batch_size, tokens_per_repeat = inputs.shape
    if tuple(shard["example_ids"].shape) != (batch_size,):
        raise ValueError("captured example IDs do not align with inputs")
    expected_targets = (batch_size, max_repeats + 1, tokens_per_repeat)
    if tuple(shard["targets_by_horizon"].shape) != expected_targets:
        raise ValueError(f"captured targets must have shape {expected_targets}")
    expected_states = (
        batch_size,
        max_repeats + 1,
        tokens_per_repeat,
        model.config.n_embd,
    )
    if tuple(shard["repeat_states"].shape) != expected_states:
        raise ValueError(f"captured repeat states must have shape {expected_states}")
    expected_logits = (batch_size, max_repeats + 1, tokens_per_repeat, 2)
    if tuple(shard["decoded_logits_by_repeat"].shape) != expected_logits:
        raise ValueError(f"captured logits must have shape {expected_logits}")

    records = shard["attention_records"]
    repeat_cache_window = shard["condition"].get("repeat_cache_window")
    middle_layers = len(model.transformer.h_mid)
    if len(records) != max_repeats * middle_layers:
        raise ValueError("captured shard has an invalid attention record count")
    expected_pairs = {
        (repeat_index, layer_index)
        for repeat_index in range(1, max_repeats + 1)
        for layer_index in range(middle_layers)
    }
    observed_pairs = set()
    for record in records:
        pair = (record.get("repeat_index"), record.get("middle_layer_index"))
        observed_pairs.add(pair)
        repeat_index = pair[0]
        repeat_mass = record.get("repeat_mass")
        expected_mass_shape = (
            batch_size,
            model.config.n_head,
            tokens_per_repeat,
            repeat_index,
        )
        if repeat_mass is None or tuple(repeat_mass.shape) != expected_mass_shape:
            raise ValueError(f"captured repeat mass has an invalid shape for {pair}")
        if not torch.allclose(
            repeat_mass.sum(dim=-1),
            torch.ones_like(repeat_mass[..., 0]),
            rtol=1e-2,
            atol=1e-2,
        ):
            raise ValueError(f"captured repeat mass does not sum to one for {pair}")
        expected_visible_repeats = (
            repeat_index
            if repeat_cache_window is None
            else min(repeat_index, repeat_cache_window)
        )
        visible_repeat_count = record.get("visible_repeat_count")
        visible_token_count = record.get("visible_token_count")
        expected_count_shape = (
            batch_size,
            model.config.n_head,
            tokens_per_repeat,
        )
        if (
            visible_repeat_count is None
            or tuple(visible_repeat_count.shape) != expected_count_shape
            or not torch.equal(
                visible_repeat_count,
                torch.full_like(
                    visible_repeat_count, expected_visible_repeats
                ),
            )
        ):
            raise ValueError(f"captured visible repeat count is invalid for {pair}")
        if (
            visible_token_count is None
            or tuple(visible_token_count.shape) != expected_count_shape
            or not torch.equal(
                visible_token_count,
                torch.full_like(
                    visible_token_count,
                    expected_visible_repeats * tokens_per_repeat,
                ),
            )
        ):
            raise ValueError(f"captured visible token count is invalid for {pair}")
        repeat_logsumexp = record.get("repeat_logsumexp")
        repeat_logsumexp_valid = record.get("repeat_logsumexp_valid")
        if (
            repeat_logsumexp is None
            or tuple(repeat_logsumexp.shape) != expected_mass_shape
            or repeat_logsumexp_valid is None
            or tuple(repeat_logsumexp_valid.shape) != expected_mass_shape
        ):
            raise ValueError(f"captured repeat log-sum-exp is invalid for {pair}")
        expected_valid = torch.zeros_like(repeat_logsumexp_valid)
        expected_valid[..., -expected_visible_repeats:] = True
        if not torch.equal(repeat_logsumexp_valid, expected_valid):
            raise ValueError(
                f"captured repeat log-sum-exp validity is invalid for {pair}"
            )
        if not torch.equal(
            repeat_logsumexp.masked_select(~repeat_logsumexp_valid),
            torch.zeros_like(
                repeat_logsumexp.masked_select(~repeat_logsumexp_valid)
            ),
        ):
            raise ValueError(
                f"captured invalid repeat log-sum-exp entries are not zero for {pair}"
            )
    if observed_pairs != expected_pairs:
        raise ValueError("captured shard has missing or unexpected attention records")


def _factorial_condition_spec(condition_id, source_depth):
    try:
        clean_state_injected, cache_reset = _FACTORIAL_CONDITION_FACTORS[
            condition_id
        ]
    except KeyError as error:
        raise ValueError(f"unknown factorial condition: {condition_id}") from error
    return {
        "condition_id": condition_id,
        "clean_state_injected": clean_state_injected,
        "cache_reset": cache_reset,
        "intervention_source_depth": source_depth,
    }


def _validate_factorial_capture(capture, model):
    if capture.get("experiment_kind") != "clean_state_cache_reset_factorial":
        raise ValueError("value is not a clean-state/cache-reset factorial capture")
    schema_version = capture.get("factorial_capture_schema_version")
    if schema_version not in (1, FACTORIAL_CAPTURE_SCHEMA_VERSION):
        raise ValueError("unsupported factorial capture schema version")

    source_depth = capture.get("source_depth")
    max_repeats = capture.get("max_repeats")
    if (
        isinstance(source_depth, bool)
        or not isinstance(source_depth, Integral)
        or source_depth < 0
        or isinstance(max_repeats, bool)
        or not isinstance(max_repeats, Integral)
        or max_repeats <= source_depth
    ):
        raise ValueError("factorial source depth or maximum repeat is invalid")
    target_depth = source_depth + 1
    post_depths = list(range(target_depth, max_repeats + 1))
    if capture.get("target_depth") != target_depth:
        raise ValueError("factorial target depth is inconsistent with source depth")
    if capture.get("post_intervention_depths") != post_depths:
        raise ValueError("factorial post-intervention depths are invalid")

    trained_window = capture.get("trained_cache_window")
    if trained_window is not None and (
        isinstance(trained_window, bool)
        or not isinstance(trained_window, Integral)
        or trained_window <= 0
    ):
        raise ValueError("factorial trained cache window must be positive or null")

    tensors = list(_iter_tensors(capture))
    if not tensors or any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("factorial capture must contain only CPU tensors")
    if any(tensor.requires_grad for tensor in tensors):
        raise ValueError("factorial capture unexpectedly retains gradients")
    if any(
        not torch.isfinite(tensor).all()
        for tensor in tensors
        if tensor.is_floating_point()
    ):
        raise ValueError("factorial capture contains non-finite values")

    inputs = capture.get("inputs")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
        raise ValueError("factorial inputs must have shape [batch, cells]")
    batch_size, tokens_per_repeat = inputs.shape
    post_count = len(post_depths)
    if tuple(capture["clean_source"].shape) != (batch_size, tokens_per_repeat):
        raise ValueError("factorial clean source does not align with inputs")
    if tuple(capture["clean_target"].shape) != (batch_size, tokens_per_repeat):
        raise ValueError("factorial clean target does not align with inputs")
    if tuple(capture["clean_targets_by_depth"].shape) != (
        batch_size,
        post_count,
        tokens_per_repeat,
    ):
        raise ValueError("factorial clean targets have an invalid shape")

    middle_layers = len(model.transformer.h_mid)
    if schema_version == FACTORIAL_CAPTURE_SCHEMA_VERSION:
        prefix = capture.get("shared_prefix")
        if not isinstance(prefix, Mapping):
            raise ValueError("factorial shared prefix is missing")
        prefix_depths = list(range(target_depth))
        if prefix.get("depths") != prefix_depths:
            raise ValueError("factorial shared-prefix depths are invalid")
        expected_prefix_shapes = {
            "clean_targets_by_depth": (
                batch_size, target_depth, tokens_per_repeat
            ),
            "decoded_logits": (
                batch_size, target_depth, tokens_per_repeat, 2
            ),
            "repeat_states": (
                batch_size, target_depth, tokens_per_repeat, model.config.n_embd
            ),
        }
        for name, shape in expected_prefix_shapes.items():
            value = prefix.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"factorial shared-prefix {name} is invalid")
        expected_targets = [inputs]
        for _ in range(source_depth):
            expected_targets.append(rule30(expected_targets[-1]))
        if not torch.equal(
            prefix["clean_targets_by_depth"],
            torch.stack(expected_targets, dim=1),
        ):
            raise ValueError("factorial shared-prefix targets are invalid")
        prefix_records = prefix.get("attention_records")
        expected_prefix_pairs = {
            (repeat_index, layer_index)
            for repeat_index in range(1, target_depth)
            for layer_index in range(middle_layers)
        }
        observed_prefix_pairs = {
            (record.get("repeat_index"), record.get("middle_layer_index"))
            for record in prefix_records
        } if isinstance(prefix_records, list) else set()
        if (
            not isinstance(prefix_records, list)
            or len(prefix_records) != len(expected_prefix_pairs)
            or observed_prefix_pairs != expected_prefix_pairs
        ):
            raise ValueError("factorial shared-prefix attention records are invalid")
        for record in prefix_records:
            repeat_index = record["repeat_index"]
            expected_visible = (
                repeat_index if trained_window is None
                else min(repeat_index, trained_window)
            )
            visible_count = record.get("visible_repeat_count")
            if (
                record.get("cached_repeats") != repeat_index
                or visible_count is None
                or not torch.equal(
                    visible_count,
                    torch.full_like(visible_count, expected_visible),
                )
            ):
                raise ValueError("factorial shared-prefix visibility is invalid")

    conditions = capture.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        _FACTORIAL_CONDITION_FACTORS
    ):
        raise ValueError("factorial capture must contain exactly four conditions")
    expected_record_pairs = {
        (repeat_index, layer_index)
        for repeat_index in post_depths
        for layer_index in range(middle_layers)
    }
    for condition_id, payload in conditions.items():
        expected_spec = _factorial_condition_spec(condition_id, source_depth)
        if any(payload.get(key) != value for key, value in expected_spec.items()):
            raise ValueError(f"factorial metadata is invalid for {condition_id}")
        expected_windows = [
            min(
                repeat_index - source_depth if payload["cache_reset"] else repeat_index,
                trained_window,
            )
            if trained_window is not None
            else repeat_index - source_depth
            if payload["cache_reset"]
            else repeat_index
            for repeat_index in post_depths
        ]
        if payload.get("effective_windows") != expected_windows:
            raise ValueError(f"factorial cache windows are invalid for {condition_id}")
        if tuple(payload["decoded_logits"].shape) != (
            batch_size,
            post_count,
            tokens_per_repeat,
            2,
        ) or tuple(payload["repeat_states"].shape) != (
            batch_size,
            post_count,
            tokens_per_repeat,
            model.config.n_embd,
        ):
            raise ValueError(
                f"factorial tensors have invalid shapes for {condition_id}"
            )
        records = payload.get("attention_records")
        observed_pairs = {
            (record.get("repeat_index"), record.get("middle_layer_index"))
            for record in records
        }
        if (
            len(records) != len(expected_record_pairs)
            or observed_pairs != expected_record_pairs
        ):
            raise ValueError(
                f"factorial attention records are invalid for {condition_id}"
            )
        for record in records:
            repeat_index = record["repeat_index"]
            expected_visible = expected_windows[repeat_index - target_depth]
            visible_count = record.get("visible_repeat_count")
            if (
                record.get("cached_repeats") != repeat_index
                or visible_count is None
                or not torch.equal(
                    visible_count,
                    torch.full_like(visible_count, expected_visible),
                )
            ):
                raise ValueError(f"factorial visibility is invalid for {condition_id}")


def _capture_forward_payload(
    model,
    inputs,
    model_args,
    device,
    max_repeats,
    *,
    pass_repeat_cache_window,
    repeat_cache_window,
    forward_overrides=None,
):
    forward_kwargs = {
        "get_logits": True,
        "return_all_logits": True,
        "num_repeats": max_repeats,
        "return_repeat_states": True,
        "return_attention_diagnostics": True,
    }
    if pass_repeat_cache_window:
        forward_kwargs["repeat_cache_window"] = repeat_cache_window
    if forward_overrides:
        duplicate_keys = forward_kwargs.keys() & forward_overrides.keys()
        if duplicate_keys:
            raise ValueError(
                "factorial forward overrides duplicate reserved arguments: "
                + ", ".join(sorted(duplicate_keys))
            )
        forward_kwargs.update(forward_overrides)
    with _autocast_context(model_args, device):
        outputs = model(inputs, **forward_kwargs)
    repeat_states, records = _validate_maximum_forward(
        outputs, inputs, max_repeats, model
    )
    with _autocast_context(model_args, device):
        decoded_logits = torch.stack(
            [
                model.lm_head(model.transformer.ln_f(state))
                for state in repeat_states
            ],
            dim=1,
        )
    payload = _to_cpu_tree(
        {
            "decoded_logits_by_repeat": decoded_logits,
            "repeat_states": torch.stack(repeat_states, dim=1),
            "attention_records": records,
        }
    )
    del outputs, repeat_states, records, decoded_logits
    return payload


def capture_factorial_transition_in_memory(
    model, inputs, model_args, device, source_depth, max_repeats,
    baseline_repeat_cache_window=None,
):
    """Capture one paired state-by-cache trajectory entirely in memory."""
    if isinstance(source_depth, bool) or not isinstance(source_depth, int):
        raise TypeError("source_depth must be a non-negative integer")
    if source_depth < 0:
        raise ValueError("source_depth must be non-negative")
    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int):
        raise TypeError("max_repeats must be an integer")
    if max_repeats <= source_depth:
        raise ValueError("max_repeats must be greater than source_depth")
    if inputs.ndim != 2:
        raise ValueError("factorial inputs must have shape [batch, cells]")
    if len(model.transformer.h_end) != 0:
        raise ValueError("intermediate-state decoding currently requires n_layer_end=0")

    inputs = inputs.to(device, dtype=torch.long, non_blocking=True)
    clean_prefix_targets = [inputs]
    clean_source = inputs
    for _ in range(source_depth):
        clean_source = rule30(clean_source)
        clean_prefix_targets.append(clean_source)
    target_depth = source_depth + 1
    post_intervention_depths = list(range(target_depth, max_repeats + 1))
    clean_targets = []
    clean_target = clean_source
    for _ in post_intervention_depths:
        clean_target = rule30(clean_target)
        clean_targets.append(clean_target)
    clean_targets = torch.stack(clean_targets, dim=1)
    pass_baseline_window = baseline_repeat_cache_window is not None
    state_args = {
        "intervention_source_depth": source_depth,
        "intervention_input_ids": clean_source,
    }
    cache_args = {"cache_reset_source_depth": source_depth}
    conditions = (("free_baseline", {}), ("clean_baseline", state_args),
                  ("free_reset", cache_args),
                  ("clean_reset", {**state_args, **cache_args}))
    baseline_payload = None
    shared_prefix = None
    condition_payloads = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for condition_id, overrides in conditions:
                payload = _capture_forward_payload(
                    model,
                    inputs,
                    model_args,
                    device,
                    max_repeats,
                    pass_repeat_cache_window=pass_baseline_window,
                    repeat_cache_window=baseline_repeat_cache_window,
                    forward_overrides=overrides,
                )
                if baseline_payload is None:
                    baseline_payload = payload
                else:
                    _assert_exact_tree(
                        baseline_payload["repeat_states"][:, :target_depth],
                        payload["repeat_states"][:, :target_depth],
                        f"{condition_id}.pre_intervention_states",
                    )
                    baseline_prefix = baseline_payload["attention_records"][
                        : source_depth * len(model.transformer.h_mid)
                    ]
                    condition_prefix = payload["attention_records"][
                        : source_depth * len(model.transformer.h_mid)
                    ]
                    _assert_exact_tree(
                        baseline_prefix,
                        condition_prefix,
                        f"{condition_id}.pre_intervention_attention",
                    )

                post_records = [
                    record
                    for record in payload["attention_records"]
                    if record["repeat_index"] >= target_depth
                ]
                expected_record_count = (
                    len(post_intervention_depths) * len(model.transformer.h_mid)
                )
                if len(post_records) != expected_record_count:
                    raise ValueError("post-intervention attention record count is invalid")
                is_reset = condition_id.endswith("reset")
                effective_windows = []
                for repeat_index in post_intervention_depths:
                    available = (
                        repeat_index - source_depth if is_reset else repeat_index
                    )
                    effective_windows.append(
                        available
                        if baseline_repeat_cache_window is None
                        else min(available, baseline_repeat_cache_window)
                    )
                for record in post_records:
                    repeat_index = record["repeat_index"]
                    expected_visible = effective_windows[repeat_index - target_depth]
                    if record["cached_repeats"] != repeat_index:
                        raise ValueError("record has the wrong physical cache depth")
                    if not torch.equal(
                        record["visible_repeat_count"],
                        torch.full_like(
                            record["visible_repeat_count"], expected_visible
                        ),
                    ):
                        raise ValueError("record has the wrong cache visibility")
                condition_payloads[condition_id] = {
                    **_factorial_condition_spec(condition_id, source_depth),
                    "decoded_logits": payload["decoded_logits_by_repeat"][
                        :, target_depth : max_repeats + 1
                    ].clone(),
                    "repeat_states": payload["repeat_states"][
                        :, target_depth : max_repeats + 1
                    ].clone(),
                    "attention_records": post_records,
                    "effective_windows": effective_windows,
                }

            pairs = (("free_baseline", "free_reset"),
                     ("clean_baseline", "clean_reset"))
            for left, right in pairs:
                left_target = condition_payloads[left]["attention_records"][
                    : len(model.transformer.h_mid)
                ]
                right_target = condition_payloads[right]["attention_records"][
                    : len(model.transformer.h_mid)
                ]
                for left_record, right_record in zip(left_target, right_target):
                    for metric in ("query_norm", "key_norm", "value_norm"):
                        _assert_exact_tree(
                            left_record[metric],
                            right_record[metric],
                            f"{left}_vs_{right}.{metric}",
                        )
            shared_prefix = {
                "depths": list(range(target_depth)),
                "clean_targets_by_depth": torch.stack(
                    clean_prefix_targets, dim=1
                ),
                "decoded_logits": baseline_payload["decoded_logits_by_repeat"][
                    :, :target_depth
                ].clone(),
                "repeat_states": baseline_payload["repeat_states"][
                    :, :target_depth
                ].clone(),
                "attention_records": [
                    record
                    for record in baseline_payload["attention_records"]
                    if record["repeat_index"] <= source_depth
                ],
            }
    finally:
        if was_training:
            model.train()

    capture = _to_cpu_tree(
        {
            "experiment_kind": "clean_state_cache_reset_factorial",
            "factorial_capture_schema_version": FACTORIAL_CAPTURE_SCHEMA_VERSION,
            "source_depth": source_depth,
            "target_depth": target_depth,
            "max_repeats": max_repeats,
            "post_intervention_depths": post_intervention_depths,
            "trained_cache_window": baseline_repeat_cache_window,
            "inputs": inputs,
            "clean_source": clean_source,
            "clean_target": clean_targets[:, 0],
            "clean_targets_by_depth": clean_targets,
            "shared_prefix": shared_prefix,
            "conditions": condition_payloads,
        }
    )
    _validate_factorial_capture(capture, model)
    return capture


def capture_factorial_transition_batch(
    model,
    dataloader,
    model_args,
    device,
    source_depth,
    max_repeats,
    artifact_dir,
    trained_cache_window,
):
    """Capture, atomically store, reload, and validate one paired batch."""
    batch = next(iter(dataloader))
    inputs = batch["input_id"].to(dtype=torch.long)
    example_ids = batch["example_id"].to(dtype=torch.long)
    if tuple(example_ids.shape) != (inputs.shape[0],):
        raise ValueError("factorial example IDs do not align with inputs")
    capture = capture_factorial_transition_in_memory(
        model,
        inputs,
        model_args,
        device,
        source_depth,
        max_repeats,
        baseline_repeat_cache_window=trained_cache_window,
    )
    capture["batch_index"] = 0
    capture["example_ids"] = example_ids
    _validate_factorial_capture(capture, model)
    shard_path = (
        Path(artifact_dir)
        / "factorials"
        / f"source_depth_{source_depth:05d}"
        / "batch_00000.pt"
    )
    if shard_path.exists():
        raise ValueError(f"factorial shard already exists: {shard_path}")
    _atomic_torch_save(shard_path, capture)
    loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
    _assert_exact_tree(capture, loaded)
    _validate_factorial_capture(loaded, model)
    return {
        "experiment_kind": capture["experiment_kind"],
        "batch_index": 0,
        "source_depth": source_depth,
        "target_depth": source_depth + 1,
        "trained_cache_window": trained_cache_window,
        "condition_ids": list(capture["conditions"]),
        "path": str(shard_path.relative_to(artifact_dir)),
        "examples": int(inputs.shape[0]),
        "tensor_bytes": _tensor_bytes(capture),
        "serialized_size_bytes": shard_path.stat().st_size,
        "sha256": _sha256_file(shard_path),
        "validation": "passed",
    }


def _capture_factorial_outcomes_in_memory(
    model, inputs, model_args, device, source_depth, max_repeats,
    baseline_repeat_cache_window=None,
):
    """Run four paired branches while retaining only targets and logits."""
    if isinstance(source_depth, bool) or not isinstance(source_depth, int):
        raise TypeError("source_depth must be a non-negative integer")
    if source_depth < 0:
        raise ValueError("source_depth must be non-negative")
    if isinstance(max_repeats, bool) or not isinstance(max_repeats, int):
        raise TypeError("max_repeats must be an integer")
    if max_repeats <= source_depth:
        raise ValueError("max_repeats must be greater than source_depth")
    if inputs.ndim != 2:
        raise ValueError("factorial inputs must have shape [batch, cells]")
    if len(model.transformer.h_end) != 0:
        raise ValueError("intermediate-state decoding currently requires n_layer_end=0")
    inputs = inputs.to(device, dtype=torch.long, non_blocking=True)
    clean_source = inputs
    for _ in range(source_depth):
        clean_source = rule30(clean_source)
    target_depth = source_depth + 1
    clean_targets = []
    clean_target = clean_source
    for _ in range(target_depth, max_repeats + 1):
        clean_target = rule30(clean_target)
        clean_targets.append(clean_target)

    state_args = {
        "intervention_source_depth": source_depth,
        "intervention_input_ids": clean_source,
    }
    cache_args = {"cache_reset_source_depth": source_depth}
    conditions = (("free_baseline", {}), ("clean_baseline", state_args),
                  ("free_reset", cache_args),
                  ("clean_reset", {**state_args, **cache_args}))
    baseline_prefix = None
    condition_logits = {}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for condition_id, overrides in conditions:
                forward_kwargs = {
                    "get_logits": True,
                    "return_all_logits": True,
                    "num_repeats": max_repeats,
                    "return_repeat_states": True,
                    "return_attention_diagnostics": False,
                    **overrides,
                }
                if baseline_repeat_cache_window is not None:
                    forward_kwargs["repeat_cache_window"] = (
                        baseline_repeat_cache_window
                    )
                with _autocast_context(model_args, device):
                    outputs = model(inputs, **forward_kwargs)
                repeat_states = _validate_repeat_state_sequence(
                    outputs, inputs, max_repeats, model
                )
                prefix = torch.stack(repeat_states[:target_depth], dim=1).cpu()
                if baseline_prefix is None:
                    baseline_prefix = prefix
                else:
                    _assert_exact_tree(
                        baseline_prefix,
                        prefix,
                        f"{condition_id}.pre_intervention_states",
                    )
                with _autocast_context(model_args, device):
                    logits = torch.stack(
                        [
                            model.lm_head(model.transformer.ln_f(state))
                            for state in repeat_states[target_depth:]
                        ],
                        dim=1,
                    )
                condition_logits[condition_id] = logits.cpu()
                del outputs, repeat_states, prefix, logits
    finally:
        if was_training:
            model.train()
    return {
        "clean_targets_by_depth": torch.stack(clean_targets, dim=1).cpu(),
        "condition_logits": condition_logits,
    }


_FACTORIAL_DIAGNOSTIC_FIELDS = {
    "query_norm": "query_norm",
    "logit_std": "logit_std",
    "logit_spread": "logit_spread",
    "top_two_logit_gap": "top_two_logit_gap",
    "normalised_repeat_entropy": "normalised_repeat_entropy",
    "normalised_attention_entropy": "normalised_attention_entropy",
    "effective_support_fraction": "effective_support_fraction",
    "maximum_attention_probability": "maximum_attention_probability",
}


def reduce_factorial_diagnostic_values(payload, depths):
    """Reduce existing attention/state tensors to per-example depth curves."""
    records_by_depth = {depth: [] for depth in depths}
    for record in payload["attention_records"]:
        records_by_depth[record["repeat_index"]].append(record)
    reduced = {name: [] for name in _FACTORIAL_DIAGNOSTIC_FIELDS}
    for name in (
        "current_key_norm", "visible_history_key_norm",
        "current_value_norm", "visible_history_value_norm",
        "current_repeat_attention_mass", "visible_history_attention_mass",
        "current_block_contribution_norm",
        "visible_history_block_contribution_norm_sum",
        "current_minus_visible_history_logsumexp",
    ):
        reduced[name] = []

    for depth in depths:
        layer_values = []
        for record in records_by_depth[depth]:
            valid = record["repeat_logsumexp_valid"].bool()
            repeat_visible = valid.any(dim=2)
            if not torch.equal(
                valid, repeat_visible.unsqueeze(2).expand_as(valid)
            ):
                raise ValueError("repeat visibility differs across query cells")
            history_visible = repeat_visible.clone()
            history_visible[..., -1] = False
            has_history = history_visible.any(dim=(1, 2))
            values = {
                output_name: record[field_name].float().mean(dim=(1, 2))
                for output_name, field_name in _FACTORIAL_DIAGNOSTIC_FIELDS.items()
            }
            if "top_two_logit_gap_valid" in record:
                gap_valid = record["top_two_logit_gap_valid"].bool()
                gap_count = gap_valid.sum(dim=(1, 2))
                values["top_two_logit_gap"] = torch.where(
                    gap_count > 0,
                    record["top_two_logit_gap"].float().sum(dim=(1, 2))
                    / gap_count.clamp_min(1),
                    torch.full_like(gap_count.float(), float("nan")),
                )
            for vector_name in ("key", "value"):
                norms = record[f"{vector_name}_norm"].float()
                values[f"current_{vector_name}_norm"] = norms[..., -1, :].mean(
                    dim=(1, 2)
                )
                mask = history_visible.unsqueeze(-1).expand_as(norms)
                count = mask.sum(dim=(1, 2, 3))
                values[f"visible_history_{vector_name}_norm"] = torch.where(
                    count > 0,
                    norms.masked_fill(~mask, 0).sum(dim=(1, 2, 3))
                    / count.clamp_min(1),
                    torch.full_like(count.float(), float("nan")),
                )
            repeat_mass = record["repeat_mass"].float()
            contribution = record["block_contribution_norm"].float()
            values["current_repeat_attention_mass"] = repeat_mass[..., -1].mean(
                dim=(1, 2)
            )
            values["current_block_contribution_norm"] = contribution[
                ..., -1
            ].mean(dim=(1, 2))
            history_mask = valid.clone()
            history_mask[..., -1] = False
            for name, tensor in (
                ("visible_history_attention_mass", repeat_mass),
                ("visible_history_block_contribution_norm_sum", contribution),
            ):
                history_sum = tensor.masked_fill(~history_mask, 0).sum(dim=-1)
                values[name] = torch.where(
                    has_history,
                    history_sum.mean(dim=(1, 2)),
                    torch.full_like(has_history.float(), float("nan")),
                )
            history_logsumexp = torch.logsumexp(
                record["repeat_logsumexp"].float().masked_fill(
                    ~history_mask, float("-inf")
                ),
                dim=-1,
            )
            margin = (
                record["repeat_logsumexp"].float()[..., -1]
                - history_logsumexp
            ).mean(dim=(1, 2))
            values["current_minus_visible_history_logsumexp"] = torch.where(
                has_history,
                margin,
                torch.full_like(margin, float("nan")),
            )
            layer_values.append(values)
        if not layer_values:
            raise ValueError(f"no attention records found for target depth {depth}")
        for name in reduced:
            reduced[name].append(
                torch.stack([values[name] for values in layer_values]).mean(dim=0)
            )
    reduced["hidden_state_norm"] = [
        values for values in torch.linalg.vector_norm(
            payload["repeat_states"].float(), dim=-1
        ).mean(dim=-1).unbind(dim=1)
    ]
    return {name: torch.stack(values, dim=1) for name, values in reduced.items()}


def evaluate_factorial_full_split(
    model,
    dataloader,
    model_args,
    device,
    source_depth,
    max_repeats,
    trained_cache_window,
):
    """Aggregate paired factorial outcomes over one deterministic split."""
    depths = list(range(source_depth + 1, max_repeats + 1))
    metric_names = ("cell_accuracy", "exact_sequence_accuracy", "loss")
    condition_sums = {
        condition_id: {name: torch.zeros(len(depths), dtype=torch.float64)
                       for name in metric_names}
        for condition_id in _FACTORIAL_CONDITION_FACTORS
    }
    num_examples = 0
    num_batches = 0
    for batch in dataloader:
        outcomes = _capture_factorial_outcomes_in_memory(
            model,
            batch["input_id"],
            model_args,
            device,
            source_depth,
            max_repeats,
            baseline_repeat_cache_window=trained_cache_window,
        )
        targets = outcomes["clean_targets_by_depth"].long()
        for condition_id, condition_values in outcomes["condition_logits"].items():
            logits = condition_values.float()
            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            losses = F.cross_entropy(
                logits.reshape(-1, 2), targets.reshape(-1), reduction="none"
            ).reshape_as(targets)
            metrics = {
                "cell_accuracy": correct.float().mean(dim=-1),
                "exact_sequence_accuracy": correct.all(dim=-1).float(),
                "loss": losses.mean(dim=-1),
            }
            for name, values in metrics.items():
                condition_sums[condition_id][name] += values.double().sum(dim=0)
        num_examples += targets.shape[0]
        num_batches += 1
        del outcomes, targets
    if num_examples == 0:
        raise ValueError("factorial full-split evaluation received no examples")

    condition_means = {
        condition_id: {
            name: values / num_examples for name, values in metric_sums.items()
        }
        for condition_id, metric_sums in condition_sums.items()
    }
    def by_depth(metrics):
        return {
            str(depth): {
                name: (
                    float(values[index])
                    if torch.isfinite(values[index])
                    else None
                )
                for name, values in metrics.items()
            }
            for index, depth in enumerate(depths)
        }

    effect_means = {
        effect_id: {
            name: condition_means[left][name] - condition_means[right][name]
            for name in metric_names
        }
        for effect_id, (left, right) in FACTORIAL_EFFECT_CONDITIONS.items()
    }
    interaction_means = {
        name: effect_means["clean_state_after_reset"][name]
        - effect_means["clean_state_with_history"][name]
        for name in metric_names
    }

    return {
        "source_depth": source_depth,
        "target_depths": depths,
        "max_repeats": max_repeats,
        "trained_cache_window": trained_cache_window,
        "num_batches": num_batches,
        "num_examples": num_examples,
        "pairing_validation": {
            "status": "passed",
            "batches_exactly_validated": num_batches,
            "pre_intervention_states": "exact",
            "pre_intervention_attention": "not_collected",
        },
        "diagnostic_collection": {
            "attention_diagnostics": False,
            "repeat_states": "transient_for_pairing_and_decoding_only",
            "retained_batch_payload": "post_intervention_logits_only",
        },
        "conditions": {
            condition_id: {"by_target_depth": by_depth(metric_sums)}
            for condition_id, metric_sums in condition_means.items()
        },
        "paired_effects": {
            effect_id: {
                "left_condition": pair[0],
                "right_condition": pair[1],
                "sign": "left_minus_right",
                "by_target_depth": by_depth(effect_means[effect_id]),
            }
            for effect_id, pair in FACTORIAL_EFFECT_CONDITIONS.items()
        },
        "state_history_interaction": {
            "formula": (
                "(clean_reset - free_reset) - "
                "(clean_baseline - free_baseline)"
            ),
            "by_target_depth": by_depth(interaction_means),
        },
        "metric_signs": {
            "cell_accuracy": "positive is better",
            "exact_sequence_accuracy": "positive is better",
            "loss": "positive is worse",
        },
    }


def capture_first_condition_batch(
    model,
    dataloader,
    model_args,
    device,
    max_repeats,
    artifact_dir,
    repeat_cache_window=None,
    verify_full_cache_none=False,
):
    if max_repeats <= 0:
        raise ValueError("max repeats must be positive for condition capture")
    if len(model.transformer.h_end) != 0:
        raise ValueError("intermediate-state decoding currently requires n_layer_end=0")
    if verify_full_cache_none and repeat_cache_window is not None:
        raise ValueError(
            "full-cache None verification requires repeat_cache_window=None"
        )

    batch = next(iter(dataloader))
    cpu_inputs = batch["input_id"].to(dtype=torch.long) # cpu inputs shape = 128,64 contains the acutal cellualr states
    example_ids = batch["example_id"].to(dtype=torch.long) # example ids shape = 128 examples are each row
    inputs = cpu_inputs.to(device, dtype=torch.long, non_blocking=True)
    was_training = model.training
    model.eval() # cancels dropout etc
    try:
        with torch.inference_mode(): # no autograd etc
            omitted_payload = None
            if verify_full_cache_none:
                omitted_payload = _capture_forward_payload(
                    model,
                    inputs,
                    model_args,
                    device,
                    max_repeats,
                    pass_repeat_cache_window=False,
                    repeat_cache_window=None,
                )
            captured_payload = _capture_forward_payload(
                model,
                inputs,
                model_args,
                device,
                max_repeats,
                pass_repeat_cache_window=True,
                repeat_cache_window=repeat_cache_window,
            )
            if omitted_payload is not None:
                _assert_exact_tree(
                    omitted_payload,
                    captured_payload,
                    "full_cache_none_equivalence",
                )

        targets = [cpu_inputs]
        for _ in range(max_repeats):
            targets.append(rule30(targets[-1]))
        shard = _to_cpu_tree(
            {
                "batch_shard_schema_version": BATCH_SHARD_SCHEMA_VERSION,
                "condition": _condition_spec("manual", repeat_cache_window),
                "batch_index": 0,
                "example_ids": example_ids,
                "inputs": cpu_inputs,
                "targets_by_horizon": torch.stack(targets, dim=1),
                **captured_payload,
            }
        )
        del captured_payload, omitted_payload, inputs
    finally:
        if was_training:
            model.train()

    _validate_captured_shard(shard, model, max_repeats)
    condition_id = shard["condition"]["condition_id"]
    shard_path = (
        Path(artifact_dir) / "conditions" / condition_id / "batch_00000.pt"
    )
    if shard_path.exists():
        raise ValueError(f"batch shard already exists: {shard_path}")
    _atomic_torch_save(shard_path, shard)
    loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
    _assert_exact_tree(shard, loaded)
    _validate_captured_shard(loaded, model, max_repeats)
    info = {
        "condition_id": condition_id,
        "batch_index": 0,
        "path": str(shard_path.relative_to(artifact_dir)),
        "examples": int(shard["inputs"].shape[0]),
        "attention_records": len(shard["attention_records"]),
        "tensor_bytes": _tensor_bytes(shard),
        "tensor_bytes_by_field": {
            key: _tensor_bytes(value)
            for key, value in shard.items()
            if any(True for _ in _iter_tensors(value))
        },
        "serialized_size_bytes": shard_path.stat().st_size,
        "sha256": _sha256_file(shard_path),
        "validation": "passed",
    }
    if verify_full_cache_none:
        info["full_cache_none_equivalence"] = {
            "status": "passed",
            "comparison": "explicit_none_vs_omitted",
            "exact": True,
        }
    return info


# Backwards-compatible import for callers that only capture the full baseline.
capture_first_baseline_batch = capture_first_condition_batch


def initialize_artifact(
    args,
    source_manifest,
    checkpoint_path,
    checkpoint_step,
    model_args,
    dataset_info,
    max_repeats,
):
    if args.artifact_name is None:
        if (
            getattr(args, "capture_first_condition_batch", False)
            or getattr(args, "capture_factorial_transition_batch", False)
            or getattr(args, "evaluate_factorial_full_split", False)
        ):
            raise ValueError(
                "batch capture requires --artifact-name"
            )
        return None
    artifact_name = _validate_artifact_name(args.artifact_name)
    artifact_dir = Path(args.output_root) / source_manifest["run_id"] / artifact_name
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact_dir.mkdir()
    except FileExistsError as error:
        raise ValueError(f"artifact directory already exists: {artifact_dir}") from error

    now = utc_now()
    operations = ["storage_validation"]
    if args.probe_forward:
        operations.append("forward_equivalence_probe")
    if getattr(args, "capture_first_condition_batch", False):
        operations.append("capture_first_condition_batch")
    if getattr(args, "capture_factorial_transition_batch", False):
        operations.append("capture_factorial_transition_batch")
    if getattr(args, "evaluate_factorial_full_split", False):
        operations.append("evaluate_factorial_full_split")
    if getattr(args, "verify_full_cache_none", False):
        operations.append("full_cache_none_equivalence")
    if args.compare_repeat_horizon_backends:
        operations.append("repeat_horizon_backend_comparison")
    if args.evaluate_delayed_recall:
        operations.append("standalone_delayed_recall_evaluation")
        if args.collect_recall_attention:
            operations.append("recall_attention_diagnostics")
        operations.extend(
            f"cache_intervention:{condition}"
            for condition in _normalise_dca_conditions(args.conditions)
            if condition != "baseline"
        )
    artifact_manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_name": artifact_name,
        "status": "running",
        "timestamps": {"created_at": now, "updated_at": now},
        "source": {
            "run_id": source_manifest["run_id"],
            "run_manifest": str(Path(args.run_manifest).resolve()),
            "checkpoint": {
                "path": str(Path(checkpoint_path).resolve()),
                "filename": Path(checkpoint_path).name,
                "expected_step": int(args.expected_step),
                "observed_step": int(checkpoint_step),
                "size_bytes": Path(checkpoint_path).stat().st_size,
                "sha256": _sha256_file(checkpoint_path),
            },
        },
        "evaluation": {
            "requested_operations": operations,
            "max_evaluation_repeats": max_repeats,
            "device": str(args.device),
            "dtype": str(model_args.dtype),
            "diagnostic_model": model_args.model,
            "dataset": dataset_info,
            "conditions": _requested_conditions(args, model_args),
        },
        "storage_validation": {"status": "pending"},
        "shards": [],
    }
    if (
        getattr(args, "capture_factorial_transition_batch", False)
        or getattr(args, "evaluate_factorial_full_split", False)
    ):
        source_depth = args.intervention_source_depth
        artifact_manifest["evaluation"]["factorial_design"] = {
            "experiment_kind": "clean_state_cache_reset_factorial",
            "source_depth": source_depth,
            "target_depth": source_depth + 1,
            "trained_cache_window": getattr(
                model_args, "repeat_cache_window", None
            ),
            "conditions": [
                _factorial_condition_spec(condition_id, source_depth)
                for condition_id in _FACTORIAL_CONDITION_FACTORS
            ],
        }
    if args.evaluate_delayed_recall:
        artifact_manifest["evaluation"]["dca_delayed_recall"] = {
            "num_repeats": int(args.num_repeats),
            "query_repeats": (
                list(args.query_repeats)
                if args.query_repeats is not None
                else list(range(1, args.num_repeats + 1))
            ),
            "num_recall_repeats": int(args.num_recall_repeats),
            "conditions": _normalise_dca_conditions(args.conditions),
            "max_batches": args.max_batches,
            "num_examples": int(args.num_examples),
            "collect_recall_attention": bool(
                args.collect_recall_attention
            ),
            "intervention_permutation_offset": int(
                args.intervention_permutation_offset
            ),
            "attention_implementation": getattr(
                model_args, "attention_implementation", None
            ),
            "source_forward_policy": CAForwardPolicy.from_args(
                model_args
            ).metadata(),
        }
    context = {
        "directory": artifact_dir,
        "manifest_path": artifact_dir / "artifact_manifest.json",
        "manifest": artifact_manifest,
    }
    write_json(context["manifest_path"], artifact_manifest)
    try:
        artifact_manifest["storage_validation"] = (
            _write_and_validate_storage_probe(artifact_dir)
        )
        artifact_manifest["timestamps"]["updated_at"] = utc_now()
        write_json(context["manifest_path"], artifact_manifest)
    except Exception as error:
        mark_artifact_failed(context, error)
        raise
    return context


def register_artifact_shard(context, shard_info):
    context["manifest"]["shards"].append(shard_info)
    context["manifest"]["timestamps"]["updated_at"] = utc_now()
    write_json(context["manifest_path"], context["manifest"])


def mark_artifact_failed(context, error):
    if context is None:
        return
    manifest = context["manifest"]
    manifest["status"] = "failed"
    manifest["error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    manifest["timestamps"]["updated_at"] = utc_now()
    manifest["timestamps"]["failed_at"] = utc_now()
    write_json(context["manifest_path"], manifest)


def complete_artifact(context, summary):
    summary_path = context["directory"] / "summary.json"
    write_json(summary_path, summary)
    manifest = context["manifest"]
    manifest["status"] = "completed"
    manifest["summary"] = {
        "path": summary_path.name,
        "size_bytes": summary_path.stat().st_size,
        "sha256": _sha256_file(summary_path),
    }
    manifest["timestamps"]["updated_at"] = utc_now()
    manifest["timestamps"]["completed_at"] = utc_now()
    write_json(context["manifest_path"], manifest)


def resolve_model_args(manifest, attention_implementation=None):
    source_values = dict(_require_mapping(manifest["resolved_args"], "resolved_args"))
    missing = object()
    resolved_window = source_values.get("repeat_cache_window", missing)
    forward_policy_value = manifest.get("forward_policy", missing)
    if forward_policy_value is missing:
        policy_window = missing
    else:
        forward_policy = _require_mapping(
            forward_policy_value, "forward_policy"
        )
        policy_window = forward_policy.get("repeat_cache_window", missing)
    if (resolved_window is missing) != (policy_window is missing) or (
        resolved_window is not missing and resolved_window != policy_window
    ):
        resolved_label = (
            "<missing>" if resolved_window is missing else repr(resolved_window)
        )
        policy_label = (
            "<missing>" if policy_window is missing else repr(policy_window)
        )
        raise ValueError(
            "manifest cache-window disagreement: "
            f"resolved_args={resolved_label}, forward_policy={policy_label}"
        )
    source_values["repeat_cache_window"] = (
        None if resolved_window is missing else resolved_window
    )
    config_format = source_values.get("config_format")
    if config_format not in config.registered_formats():
        raise ValueError(f"unsupported config format: {config_format!r}")

    parsed_args = config.parse_args_with_format(
        format=config_format,
        base_parser=argparse.ArgumentParser(add_help=False, allow_abbrev=False),
        args=[],
        namespace=argparse.Namespace(**source_values),
    )
    if not isinstance(parsed_args.dtype, torch.dtype):
        raise ValueError(f"unsupported dtype: {parsed_args.dtype!r}")
    model_args = argparse.Namespace(**source_values)
    model_args.dtype = parsed_args.dtype
    model_args.source_attention_implementation = source_values.get(
        "attention_implementation", "sdpa"
    )
    source_model = source_values.get("model")
    model_args.model = (
        source_model
        if source_model in DCA_EVALUATION_MODELS
        else "ca_cotf_cache_attn"
    )
    if attention_implementation is not None:
        model_args.attention_implementation = attention_implementation
    return model_args


def load_frozen_model(model_args, checkpoint_path, expected_step, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = _require_mapping(checkpoint, "checkpoint")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint.model must be a non-empty state dictionary")
    checkpoint_step = checkpoint.get("itr")
    if hasattr(checkpoint_step, "item"):
        checkpoint_step = checkpoint_step.item()
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, Integral):
        raise ValueError("checkpoint.itr must be an integer")
    if checkpoint_step != expected_step:
        raise ValueError(f"expected checkpoint step {expected_step}, found {checkpoint_step}")

    model = models.make_model_from_args(model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, checkpoint_step, len(state_dict)


class ExampleIndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample["example_id"] = int(index)
        return sample


def build_fixed_diagnostic_dataset(model_args, split, device, data_mode="manifest"):
    num_cells = int(model_args.ca_diagnostic_length)
    if num_cells not in model_args.ca_eval_num_cells:
        raise ValueError("diagnostic length is not present in ca_eval_num_cells")
    if split == "validation":
        num_samples, base_seed = model_args.ca_val_samples, model_args.ca_val_seed
    elif split == "test":
        num_samples, base_seed = model_args.ca_test_samples, model_args.ca_test_seed
    else:
        raise ValueError(f"unsupported diagnostic split: {split!r}")

    selected_mode = model_args.ca_data_mode if data_mode == "manifest" else data_mode
    dataset_class = {
        "indexed": Rule30Dataset,
        "materialized": MaterializedRule30Dataset,
    }.get(selected_mode)
    if dataset_class is None:
        raise ValueError(f"unsupported diagnostic data mode: {selected_mode!r}")
    split_seed = int(base_seed) + num_cells
    dataset = dataset_class(
        num_samples=int(num_samples),
        num_cells=num_cells,
        steps=int(model_args.ca_steps),
        bernoulli_p=float(model_args.ca_bernoulli_p),
        seed=split_seed,
    )
    indexed_dataset = ExampleIndexedDataset(dataset)
    batch_size = int(model_args.ca_eval_batch_size or model_args.batch_size)
    loader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(model_args.ca_num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )
    info = {
        "split": split,
        "data_mode": selected_mode,
        "num_samples": len(indexed_dataset),
        "num_cells": num_cells,
        "steps": int(model_args.ca_steps),
        "bernoulli_p": float(model_args.ca_bernoulli_p),
        "seed": split_seed,
        "batch_size": batch_size,
        "num_batches": len(loader),
        "shuffle": False,
    }
    return loader, info


def _autocast_context(model_args, device):
    if torch.device(device).type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=model_args.dtype)


def _normalise_dca_conditions(requested_conditions):
    conditions = list(dict.fromkeys(requested_conditions))
    if any(condition != "baseline" for condition in conditions):
        conditions = ["baseline"] + [
            condition for condition in conditions if condition != "baseline"
        ]
    return conditions


def _dca_attention_modules(model):
    transformer = getattr(model, "transformer", None)
    middle_blocks = getattr(transformer, "h_mid", None)
    if middle_blocks is None or not middle_blocks:
        raise ValueError(
            "DCA cache interventions require at least one middle block."
        )
    attention_modules = []
    for layer_index, block in enumerate(middle_blocks):
        attention = getattr(block, "attn", None)
        if attention is None or not hasattr(attention, "all_values"):
            raise TypeError(
                "The selected model does not expose the CoTFormer repeat cache."
            )
        attention_modules.append((layer_index, attention))
    return attention_modules


@contextmanager
def _temporary_attention_backend(model, backend):
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise TypeError("Model does not expose transformer attention blocks.")
    attention_modules = []
    for stack_name in ("h_begin", "h_mid", "h_end"):
        for block in getattr(transformer, stack_name, ()):
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(
                attention, "attention_implementation"
            ):
                attention_modules.append(attention)
    if not attention_modules:
        raise TypeError("Model does not expose selectable attention backends.")
    previous = [
        attention.attention_implementation
        for attention in attention_modules
    ]
    try:
        for attention in attention_modules:
            attention.attention_implementation = backend
        yield
    finally:
        for attention, prior_backend in zip(
            attention_modules, previous, strict=True
        ):
            attention.attention_implementation = prior_backend


def _batch_permute_target_values(
    attention,
    *,
    query_repeat,
    tokens_per_repeat,
    permutation_offset,
):
    cache_state = attention.all_values
    if not isinstance(cache_state, tuple) or len(cache_state) != 2:
        raise RuntimeError("Unexpected CoTFormer value-cache representation.")
    full_values, populated_values = cache_state
    if populated_values is None:
        raise RuntimeError("Value cache is empty at the first recall iteration.")

    batch_size = int(full_values.shape[0])
    if batch_size < 2:
        raise ValueError(
            "Target-value corruption requires evaluation batches containing "
            "at least two examples."
        )
    offset = int(permutation_offset) % batch_size
    if offset == 0:
        offset = 1
    permutation = torch.roll(
        torch.arange(batch_size, device=full_values.device),
        shifts=offset,
    )
    block_start = (query_repeat - 1) * tokens_per_repeat
    block_end = block_start + tokens_per_repeat
    if populated_values.shape[2] < block_end:
        raise RuntimeError(
            "Requested evolution block has not been populated before recall."
        )
    replacement = full_values[
        :, :, block_start:block_end, :
    ].index_select(0, permutation).clone()
    full_values[:, :, block_start:block_end, :] = replacement
    return {
        "batch_size": batch_size,
        "permutation_offset": offset,
        "target_cache_block_zero_based": query_repeat - 1,
        "target_token_start": block_start,
        "target_token_end_exclusive": block_end,
    }


@contextmanager
def _dca_cache_intervention(
    model,
    *,
    condition,
    num_repeats,
    query_repeat,
    permutation_offset,
):
    metadata = {
        "condition": condition,
        "target_repeat": query_repeat,
        "recall_age": num_repeats - query_repeat,
        "value_permutations": 0,
        "masked_attention_calls": 0,
        "batch_permutations": [],
    }
    if condition == "baseline":
        yield metadata
        return
    if condition not in DCA_CONDITIONS:
        raise ValueError(f"Unsupported DCA condition: {condition!r}.")

    attention_modules = _dca_attention_modules(model)
    original_forwards = []

    def install_layer(layer_index, attention):
        original_forward = attention.forward
        state = {"call_index": 0, "corrupted": False}

        def intervened_forward(_attention, *call_args, **call_kwargs):
            if not call_args:
                raise RuntimeError("Attention intervention received no input state.")
            x = call_args[0]
            tokens_per_repeat = int(x.shape[1])
            if _attention.all_values is None:
                state["call_index"] = 0
                state["corrupted"] = False
            state["call_index"] += 1
            recall_index = state["call_index"] - num_repeats

            if recall_index <= 0:
                return original_forward(*call_args, **call_kwargs)

            if condition == "target-value-corruption" and not state["corrupted"]:
                permutation = _batch_permute_target_values(
                    _attention,
                    query_repeat=query_repeat,
                    tokens_per_repeat=tokens_per_repeat,
                    permutation_offset=permutation_offset,
                )
                state["corrupted"] = True
                metadata["value_permutations"] += 1
                if layer_index == 0:
                    metadata["batch_permutations"].append(permutation)
                return original_forward(*call_args, **call_kwargs)

            # if condition != "target-repeat-only":
            #     return original_forward(*call_args, **call_kwargs)

            # original_softmax = F.softmax

            # def target_only_softmax(attention_logits, *softmax_args, **softmax_kwargs):
            #     if attention_logits.ndim != 4:
            #         raise RuntimeError(
            #             "Target-repeat masking expected [B, H, Q, K] logits."
            #         )
            #     key_tokens = int(attention_logits.shape[-1])
            #     if key_tokens % tokens_per_repeat:
            #         raise RuntimeError(
            #             "Attention keys do not contain whole repeat blocks."
            #         )
            #     cached_blocks = key_tokens // tokens_per_repeat
            #     if cached_blocks < num_repeats + recall_index:
            #         raise RuntimeError(
            #             "Recall cache contains fewer blocks than expected."
            #         )

            #     allowed = torch.zeros(
            #         key_tokens,
            #         dtype=torch.bool,
            #         device=attention_logits.device,
            #     )
            #     target_start = (query_repeat - 1) * tokens_per_repeat
            #     allowed[target_start : target_start + tokens_per_repeat] = True
            #     # Preserve the growing recall trajectory: this includes every
            #     # recall-phase block accumulated so far and the current/self
            #     # block appended by this attention call.
            #     allowed[num_repeats * tokens_per_repeat :] = True
            #     attention_logits.masked_fill_(
            #         ~allowed.view(1, 1, 1, -1),
            #         float("-inf"),
            #     )
            #     return original_softmax(
            #         attention_logits, *softmax_args, **softmax_kwargs
            #     )

            # F.softmax = target_only_softmax
            # try:
            #     output = original_forward(*call_args, **call_kwargs)
            # finally:
            #     F.softmax = original_softmax
            # metadata["masked_attention_calls"] += 1
            # return output
            masking_conditions = {
                "target-repeat-only",
                "target-repeat-masked",
            }
            if condition not in masking_conditions:
                return original_forward(*call_args, **call_kwargs)

            original_softmax = F.softmax

            def intervention_softmax(
                attention_logits,
                *softmax_args,
                **softmax_kwargs,
            ):
                if attention_logits.ndim != 4:
                    raise RuntimeError(
                        "Recall masking expected [B, H, Q, K] logits."
                    )

                key_tokens = int(attention_logits.shape[-1])
                if key_tokens % tokens_per_repeat:
                    raise RuntimeError(
                        "Attention keys do not contain whole repeat blocks."
                    )

                cached_blocks = key_tokens // tokens_per_repeat
                if cached_blocks < num_repeats + recall_index:
                    raise RuntimeError(
                        "Recall cache contains fewer blocks than expected."
                    )

                target_start = (
                    (query_repeat - 1) * tokens_per_repeat
                )
                target_end = target_start + tokens_per_repeat

                if condition == "target-repeat-only":
                    # Mask every evolution block except the requested one.
                    # All recall-phase blocks remain visible.
                    masked = torch.zeros(
                        key_tokens,
                        dtype=torch.bool,
                        device=attention_logits.device,
                    )
                    masked[: num_repeats * tokens_per_repeat] = True
                    masked[target_start:target_end] = False

                elif condition == "target-repeat-masked":
                    # Mask only the requested evolution block. All other
                    # evolution and recall-phase blocks remain visible.
                    masked = torch.zeros(
                        key_tokens,
                        dtype=torch.bool,
                        device=attention_logits.device,
                    )
                    masked[target_start:target_end] = True

                else:
                    raise RuntimeError(
                        f"Unexpected masking condition: {condition!r}."
                    )

                attention_logits.masked_fill_(
                    masked.view(1, 1, 1, -1),
                    float("-inf"),
                )

                return original_softmax(
                    attention_logits,
                    *softmax_args,
                    **softmax_kwargs,
                )

            F.softmax = intervention_softmax
            try:
                output = original_forward(*call_args, **call_kwargs)
            finally:
                F.softmax = original_softmax

            metadata["masked_attention_calls"] += 1
            return output


        attention.forward = MethodType(intervened_forward, attention)
        original_forwards.append((attention, original_forward))

    try:
        for layer_index, attention in attention_modules:
            install_layer(layer_index, attention)
        yield metadata
    finally:
        for attention, original_forward in reversed(original_forwards):
            attention.forward = original_forward


class _RecallAttentionAccumulator:
    _VECTOR_FIELDS = (
        "macro_repeat_mass",
        "macro_same_position_mass",
        "macro_block_contribution_norm",
        "macro_within_repeat_entropy",
        "head_repeat_mass",
    )
    _SCALAR_FIELDS = (
        "repeat_entropy",
        "normalised_repeat_entropy",
        "attention_entropy",
        "normalised_attention_entropy",
        "maximum_attention_probability",
        "effective_support",
        "effective_support_fraction",
    )

    def __init__(self, num_repeats, query_repeat):
        self.num_repeats = int(num_repeats)
        self.query_repeat = int(query_repeat)
        self.records = {}

    def add(self, attention_records, batch_size):
        for diagnostic in attention_records:
            repeat_index = int(diagnostic["repeat_index"])
            if repeat_index <= self.num_repeats:
                continue
            layer_index = int(diagnostic["middle_layer_index"])
            recall_index = repeat_index - self.num_repeats
            key = (recall_index, layer_index)
            cached_repeats = int(diagnostic["cached_repeats"])
            entry = self.records.setdefault(
                key,
                {
                    "examples": 0,
                    "batches": 0,
                    "cached_repeats": cached_repeats,
                    "sums": {},
                },
            )
            if entry["cached_repeats"] != cached_repeats:
                raise RuntimeError(
                    "Attention cache depth changed across equivalent batches."
                )
            entry["examples"] += int(batch_size)
            entry["batches"] += 1

            values = {}
            for field in self._VECTOR_FIELDS:
                value = diagnostic.get(field)
                if value is not None:
                    values[field] = value.detach().float().cpu()
            for field in self._SCALAR_FIELDS:
                value = diagnostic.get(field)
                if value is not None:
                    values[f"mean_{field}"] = (
                        value.detach().float().mean().cpu()
                    )
            for name, value in values.items():
                weighted = value * int(batch_size)
                if name not in entry["sums"]:
                    entry["sums"][name] = weighted
                else:
                    if entry["sums"][name].shape != weighted.shape:
                        raise RuntimeError(
                            f"Attention diagnostic shape changed for {name}."
                        )
                    entry["sums"][name] += weighted

    def finalize(self):
        finalized = {}
        for (recall_index, layer_index), entry in sorted(self.records.items()):
            cached_repeats = entry["cached_repeats"]
            labels = [
                f"evolution_repeat_{repeat}"
                for repeat in range(1, self.num_repeats + 1)
            ] + [
                f"recall_repeat_{repeat}"
                for repeat in range(1, cached_repeats - self.num_repeats + 1)
            ]
            metrics = {
                name: (value / entry["examples"]).tolist()
                for name, value in entry["sums"].items()
            }
            target_mass = metrics.get("macro_repeat_mass")
            finalized[
                f"recall_{recall_index}_middle_layer_{layer_index + 1}"
            ] = {
                "recall_index": recall_index,
                "middle_layer_index_zero_based": layer_index,
                "middle_layer_number": layer_index + 1,
                "examples": entry["examples"],
                "batches": entry["batches"],
                "cached_repeats": cached_repeats,
                "cache_block_labels": labels,
                "requested_evolution_block_zero_based": self.query_repeat - 1,
                "requested_repeat_attention_mass": (
                    target_mass[self.query_repeat - 1]
                    if target_mass is not None
                    else None
                ),
                "metrics": metrics,
            }
        return finalized


@contextmanager
def _capture_recall_attention(model, accumulator):
    original_forward = model.forward

    def diagnostic_forward(_model, inputs, *call_args, **call_kwargs):
        call_kwargs["return_attention_diagnostics"] = True
        outputs = original_forward(inputs, *call_args, **call_kwargs)
        diagnostics = outputs.get("attention_diagnostics")
        if diagnostics is None:
            raise RuntimeError(
                "Model did not return requested attention diagnostics."
            )
        accumulator.add(diagnostics, int(inputs.shape[0]))
        return outputs

    model.forward = MethodType(diagnostic_forward, model)
    try:
        yield
    finally:
        model.forward = original_forward


def _numeric_delta(intervention_value, baseline_value):
    if intervention_value is None or baseline_value is None:
        return None
    return float(intervention_value) - float(baseline_value)


def _condition_effect(baseline, intervention):
    baseline_internal = baseline["internal_consistency"]
    intervention_internal = intervention["internal_consistency"]
    baseline_decoded = baseline_internal["decoded_requested_repeat"]
    intervention_decoded = intervention_internal["decoded_requested_repeat"]
    baseline_similarity = baseline_internal["requested_repeat_logit_similarity"]
    intervention_similarity = intervention_internal[
        "requested_repeat_logit_similarity"
    ]
    baseline_cosine_rank = baseline_internal["cosine_retrieval"]
    intervention_cosine_rank = intervention_internal["cosine_retrieval"]
    return {
        "intervention_minus_baseline": {
            "cell_accuracy": _numeric_delta(
                intervention["metrics"]["cell_accuracy"],
                baseline["metrics"]["cell_accuracy"],
            ),
            "exact_sequence_accuracy": _numeric_delta(
                intervention["metrics"]["exact_sequence_accuracy"],
                baseline["metrics"]["exact_sequence_accuracy"],
            ),
            "internal_decoded_cell_accuracy": _numeric_delta(
                intervention_decoded["cell_accuracy"],
                baseline_decoded["cell_accuracy"],
            ),
            "internal_decoded_exact_sequence_accuracy": _numeric_delta(
                intervention_decoded["exact_sequence_accuracy"],
                baseline_decoded["exact_sequence_accuracy"],
            ),
            "requested_repeat_logit_cosine_similarity": _numeric_delta(
                intervention_similarity["cosine_similarity"],
                baseline_similarity["cosine_similarity"],
            ),
            "requested_repeat_mean_cosine_rank": _numeric_delta(
                intervention_cosine_rank["requested_repeat_mean_rank"],
                baseline_cosine_rank["requested_repeat_mean_rank"],
            ),
        },
        "example_prediction_changes": _example_prediction_changes(
            baseline.get("examples", []), intervention.get("examples", [])
        ),
    }


def _example_prediction_changes(baseline_examples, intervention_examples):
    if len(baseline_examples) != len(intervention_examples):
        raise RuntimeError("Paired conditions retained different example counts.")
    changes = []
    for baseline, intervention in zip(
        baseline_examples, intervention_examples, strict=True
    ):
        if baseline["input"] != intervention["input"]:
            raise RuntimeError("Paired conditions evaluated different examples.")
        changed_cells = sum(
            left != right
            for left, right in zip(
                baseline["prediction"], intervention["prediction"], strict=True
            )
        )
        changes.append(
            {
                "input": baseline["input"],
                "target": baseline["target"],
                "baseline_prediction": baseline["prediction"],
                "intervention_prediction": intervention["prediction"],
                "changed_prediction_cells": changed_cells,
                "baseline_cell_accuracy": baseline["cell_accuracy"],
                "intervention_cell_accuracy": intervention["cell_accuracy"],
                "cell_accuracy_delta": (
                    intervention["cell_accuracy"] - baseline["cell_accuracy"]
                ),
            }
        )
    return changes


def evaluate_dca_checkpoint(
    model,
    dataloader,
    model_args,
    args,
):
    num_repeats = int(args.num_repeats)
    max_relative_age = int(model_args.ca_max_relative_age)
    if num_repeats > max_relative_age:
        raise ValueError(
            f"num_repeats={num_repeats} exceeds checkpoint configuration "
            f"ca_max_relative_age={max_relative_age}."
        )
    query_repeats = (
        list(args.query_repeats)
        if args.query_repeats is not None
        else list(range(1, num_repeats + 1))
    )
    conditions = _normalise_dca_conditions(args.conditions)
    model_name = str(model_args.model)
    if any(condition != "baseline" for condition in conditions):
        if model_name != DCA_CACHE_INTERVENTION_MODEL:
            raise ValueError(
                "Cache interventions are supported only for dca_cotf_cache; "
                f"selected model is {model_name!r}."
            )
    if args.collect_recall_attention and model_name != DCA_CACHE_INTERVENTION_MODEL:
        raise ValueError(
            "Recall-attention diagnostics are supported only for "
            "dca_cotf_cache."
        )
    forward_policy = CAForwardPolicy.from_args(model_args)
    masking_conditions = {
    "target-repeat-only",
    "target-repeat-masked",
    }
    if (
        masking_conditions.intersection(conditions)
        and forward_policy.repeat_cache_window is not None
    ):
        raise ValueError(
            "Recall-block masking requires a full-cache source policy; "
            "a recent-cache source would compound two different masks."
        )
    forward_context = CAForwardContext(model, forward_policy)
    evaluation_backend = getattr(
        model_args, "attention_implementation", None
    )
    source_backend = getattr(
        model_args,
        "source_attention_implementation",
        evaluation_backend,
    )
    compare_attention_backends = (
        model_name == DCA_CACHE_INTERVENTION_MODEL
        and source_backend != evaluation_backend
    )
    queries = {}
    for query_repeat in query_repeats:
        condition_results = {}
        source_backend_baseline = None
        if compare_attention_backends:
            with _temporary_attention_backend(model, source_backend):
                source_backend_baseline = evaluate_delayed_recall_query(
                    forward_context,
                    dataloader,
                    args.device,
                    num_repeats=num_repeats,
                    query_repeat=query_repeat,
                    num_recall_repeats=args.num_recall_repeats,
                    max_batches=args.max_batches,
                    num_examples=args.num_examples,
                    ctx=_autocast_context(model_args, args.device),
                )
            source_backend_baseline["condition"] = (
                "source-attention-backend-baseline"
            )
            source_backend_baseline["attention_implementation"] = (
                source_backend
            )
        for condition in conditions:
            attention_accumulator = (
                _RecallAttentionAccumulator(num_repeats, query_repeat)
                if args.collect_recall_attention
                else None
            )
            with ExitStack() as stack:
                intervention = stack.enter_context(
                    _dca_cache_intervention(
                        model,
                        condition=condition,
                        num_repeats=num_repeats,
                        query_repeat=query_repeat,
                        permutation_offset=(
                            args.intervention_permutation_offset
                        ),
                    )
                )
                if attention_accumulator is not None:
                    stack.enter_context(
                        _capture_recall_attention(
                            model, attention_accumulator
                        )
                    )
                result = evaluate_delayed_recall_query(
                    forward_context,
                    dataloader,
                    args.device,
                    num_repeats=num_repeats,
                    query_repeat=query_repeat,
                    num_recall_repeats=args.num_recall_repeats,
                    max_batches=args.max_batches,
                    num_examples=args.num_examples,
                    ctx=_autocast_context(model_args, args.device),
                )
            result["condition"] = condition
            result["attention_implementation"] = evaluation_backend
            result["intervention"] = intervention
            if attention_accumulator is not None:
                result["recall_attention"] = attention_accumulator.finalize()
            condition_results[condition] = result

        baseline = condition_results["baseline"]
        effects = {
            condition: _condition_effect(baseline, result)
            for condition, result in condition_results.items()
            if condition != "baseline"
        }
        backend_comparison = None
        if source_backend_baseline is not None:
            backend_effect = _condition_effect(
                source_backend_baseline, baseline
            )
            backend_comparison = {
                "source_attention_implementation": source_backend,
                "evaluation_attention_implementation": evaluation_backend,
                "source_backend_baseline": source_backend_baseline,
                "evaluation_minus_source": backend_effect[
                    "intervention_minus_baseline"
                ],
                "example_prediction_changes": backend_effect[
                    "example_prediction_changes"
                ],
            }
        queries[f"query_repeat_{query_repeat}"] = {
            "num_repeats": num_repeats,
            "query_repeat": query_repeat,
            "recall_age": num_repeats - query_repeat,
            "conditions": condition_results,
            "effects": effects,
            "attention_backend_comparison": backend_comparison,
        }

    return {
        "model": model_name,
        "source_attention_implementation": source_backend,
        "evaluation_attention_implementation": evaluation_backend,
        "forward_policy": forward_policy.metadata(),
        "num_repeats": num_repeats,
        "num_recall_repeats": int(args.num_recall_repeats),
        "query_repeats": query_repeats,
        "conditions": conditions,
        "max_batches": args.max_batches,
        "num_examples_per_condition": int(args.num_examples),
        "metric_semantics": {
            "decoded_requested_repeat_cell_accuracy": (
                "Cellwise equality between the final recall argmax and the "
                "same-run argmax decoded after the requested evolution repeat."
            ),
            "requested_repeat_logit_cosine_similarity": (
                "Cosine similarity between flattened final recall logits and "
                "flattened logits decoded after the requested evolution repeat."
            ),
            "requested_repeat_cosine_rank": (
                "One plus the number of evolution repeats whose flattened-logit "
                "cosine is strictly greater than the requested repeat cosine, "
                "with rtol=1e-5 and atol=1e-7 ties. Rank 1 need not be unique."
            ),
        },
        "intervention_semantics": {
            "target-value-corruption": (
                "Immediately before the first recall iteration, cyclically "
                "batch-permute only the requested evolution block's cached "
                "values independently in every middle layer; keys and all "
                "other value blocks remain unchanged."
            ),
            "target-repeat-only": (
                "During every recall attention call, expose the requested "
                "evolution block plus all recall-phase blocks accumulated so "
                "far, including the current/self block; mask every other "
                "evolution block without deleting cache storage."
            ),
            "target-repeat-masked": (
                "During every recall attention call, mask the requested "
                "evolution block while leaving every other evolution and "
                "recall-phase block visible. Cache storage remains intact, "
                "but the requested block receives zero attention probability."
            ),
        },
        "queries": queries,
    }


def _compare_logits(left, right):
    if left.shape != right.shape:
        raise ValueError(f"logit shapes differ: {tuple(left.shape)} and {tuple(right.shape)}")
    if torch.bfloat16 in (left.dtype, right.dtype):
        relative_tolerance, absolute_tolerance = 1e-2, 1e-3
    elif torch.float16 in (left.dtype, right.dtype):
        relative_tolerance, absolute_tolerance = 5e-3, 5e-4
    else:
        relative_tolerance, absolute_tolerance = 1e-5, 1e-6
    difference = left.detach().float().sub(right.detach().float()).abs()
    left_predictions = left.argmax(dim=-1)
    right_predictions = right.argmax(dim=-1)
    prediction_equal = left_predictions.eq(right_predictions)
    return {
        "exact_logit_equality": bool(torch.equal(left, right)),
        "logits_within_tolerance": bool(
            torch.allclose(
                left, right, rtol=relative_tolerance, atol=absolute_tolerance
            )
        ),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "maximum_absolute_logit_difference": float(difference.max().cpu()),
        "mean_absolute_logit_difference": float(difference.mean().cpu()),
        "differing_predicted_cells": int((~prediction_equal).sum().cpu()),
        "predicted_cells": prediction_equal.numel(),
        "prediction_agreement": float(prediction_equal.float().mean().cpu()),
        "exact_sequence_agreement": float(
            prediction_equal.all(dim=-1).float().mean().cpu()
        ),
    }


def _validate_repeat_state_sequence(outputs, inputs, max_repeats, model):
    repeat_states = outputs.get("repeat_states")
    if repeat_states is None or len(repeat_states) != max_repeats + 1:
        raise ValueError("maximum forward returned an invalid repeat-state sequence")
    expected_state_shape = (*inputs.shape, model.config.n_embd)
    if any(tuple(state.shape) != expected_state_shape for state in repeat_states):
        raise ValueError(f"repeat states must have shape {expected_state_shape}")
    if any(state.requires_grad for state in repeat_states):
        raise ValueError("repeat states unexpectedly retain gradients")
    return repeat_states


def _validate_maximum_forward(outputs, inputs, max_repeats, model):
    repeat_states = _validate_repeat_state_sequence(
        outputs, inputs, max_repeats, model
    )
    records = outputs.get("attention_diagnostics")

    middle_layers = len(model.transformer.h_mid)
    expected_pairs = {
        (repeat_index, layer_index)
        for repeat_index in range(1, max_repeats + 1)
        for layer_index in range(middle_layers)
    }
    if records is None or len(records) != len(expected_pairs):
        raise ValueError("maximum forward returned an invalid number of attention records")
    observed_pairs = set()
    for record in records:
        pair = (record.get("repeat_index"), record.get("middle_layer_index"))
        if pair in observed_pairs:
            raise ValueError(f"duplicate attention record: {pair}")
        observed_pairs.add(pair)
        repeat_index = pair[0]
        if record.get("cached_repeats") != repeat_index:
            raise ValueError(f"cached repeat count does not match record {pair}")
        repeat_mass = record.get("repeat_mass")
        expected_mass_shape = (
            inputs.shape[0], model.config.n_head, inputs.shape[1], repeat_index
        )
        if repeat_mass is None or tuple(repeat_mass.shape) != expected_mass_shape:
            raise ValueError(f"repeat mass has an invalid shape for record {pair}")
        if not torch.allclose(
            repeat_mass.sum(dim=-1),
            torch.ones_like(repeat_mass[..., 0]),
            rtol=1e-2,
            atol=1e-2,
        ):
            raise ValueError(f"repeat mass does not sum to one for record {pair}")
    if observed_pairs != expected_pairs:
        raise ValueError("maximum forward has missing or unexpected attention records")
    return repeat_states, records


def run_forward_equivalence_probe(
    model, dataloader, model_args, device, max_repeats, equivalence_repeats
):
    if max_repeats <= 0:
        raise ValueError("max repeats must be positive")
    depths = sorted(set(int(value) for value in equivalence_repeats))
    if not depths or depths[0] <= 0 or depths[-1] > max_repeats:
        raise ValueError("equivalence repeats must lie within 1..max_repeats")
    if len(model.transformer.h_end) != 0:
        raise ValueError("intermediate-state decoding currently requires n_layer_end=0")

    configured_backend = getattr(model_args, "attention_implementation", "sdpa")
    configured_forward = CAForwardContext(model)
    batch = next(iter(dataloader))
    inputs = batch["input_id"].to(device, dtype=torch.long, non_blocking=True)
    with torch.inference_mode():
        with _autocast_context(model_args, device):
            maximum_outputs = model(
                inputs,
                get_logits=True,
                return_all_logits=True,
                num_repeats=max_repeats,
                return_repeat_states=True,
                return_attention_diagnostics=True,
            )
        repeat_states, records = _validate_maximum_forward(
            maximum_outputs, inputs, max_repeats, model
        )
        comparisons = {}
        for depth in depths:
            with _autocast_context(model_args, device):
                decoded_logits = model.lm_head(
                    model.transformer.ln_f(repeat_states[depth])
                )
                standalone_manual = model(
                    inputs,
                    get_logits=True,
                    return_all_logits=True,
                    num_repeats=depth,
                    return_attention_diagnostics=True,
                )["logits"]
                standalone_existing, _ = _forward_all_cells(
                    configured_forward, inputs, num_repeats=depth
                )
            comparisons[str(depth)] = {
                "maximum_manual_vs_standalone_manual": _compare_logits(
                    decoded_logits, standalone_manual
                ),
                "standalone_manual_vs_configured_backend": _compare_logits(
                    standalone_manual, standalone_existing
                ),
            }

    one_forward_equivalent = all(
        values["maximum_manual_vs_standalone_manual"]["differing_predicted_cells"]
        == 0
        for values in comparisons.values()
    )
    return {
        "batch_examples": inputs.shape[0],
        "max_repeats": max_repeats,
        "checked_repeats": depths,
        "repeat_states": len(repeat_states),
        "attention_records": len(records),
        "configured_attention_implementation": configured_backend,
        "one_forward_prediction_equivalent": one_forward_equivalent,
        "comparisons": comparisons,
    }


_BACKEND_COMPARISON_METRICS = (
    "cell_accuracy",
    "exact_sequence_accuracy",
    "loss",
)


def _compact_repeat_horizon_metrics(result):
    matrix = result["repeat_horizon_matrix"]
    return {
        metric: {
            repeat_key: {
                step_key: values[metric]
                for step_key, values in step_row.items()
            }
            for repeat_key, step_row in matrix.items()
        }
        for metric in _BACKEND_COMPARISON_METRICS
    }


def _subtract_metric_matrices(left, right):
    return {
        metric: {
            repeat_key: {
                step_key: value - right[metric][repeat_key][step_key]
                for step_key, value in step_row.items()
            }
            for repeat_key, step_row in repeat_rows.items()
        }
        for metric, repeat_rows in left.items()
    }


def run_repeat_horizon_backend_comparison(
    manifest,
    checkpoint_path,
    expected_step,
    dataloader,
    device,
    max_repeats,
):
    backend_results = {}
    num_batches = None
    target_horizons = tuple(range(max_repeats + 1))
    for backend in ("sdpa", "manual"):
        backend_args = resolve_model_args(
            manifest, attention_implementation=backend
        )
        backend_model, _, _ = load_frozen_model(
            backend_args, checkpoint_path, expected_step, device
        )
        backend_forward = CAForwardContext(backend_model)
        try:
            result = evaluate_ca_repeat_horizon_diagnostics(
                backend_forward,
                dataloader,
                device,
                max_repeats=max_repeats,
                target_horizons=target_horizons,
                num_examples=0,
                collect_hidden_states=False,
                ctx=_autocast_context(backend_args, device),
            )
        finally:
            del backend_forward
            del backend_model
        backend_results[backend] = _compact_repeat_horizon_metrics(result)
        if num_batches is None:
            num_batches = result["num_batches"]
        elif result["num_batches"] != num_batches:
            raise ValueError("backend comparisons consumed different batch counts")

    return {
        "max_repeats": max_repeats,
        "target_horizons": list(target_horizons),
        "num_batches": num_batches,
        "backends": backend_results,
        "manual_minus_sdpa": _subtract_metric_matrices(
            backend_results["manual"], backend_results["sdpa"]
        ),
    }


def validate_source_run(args):
    if not args.run_manifest.is_file():
        raise ValueError(f"run manifest does not exist: {args.run_manifest}")

    manifest = _require_mapping(read_json(args.run_manifest), "run manifest")
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid run manifest: " + "; ".join(errors))
    if manifest["run_id"] != args.expected_run_id:
        raise ValueError(f"expected run {args.expected_run_id!r}, found {manifest['run_id']!r}")
    if manifest["status"] != "completed":
        raise ValueError(f"run status is {manifest['status']!r}, not 'completed'")

    provenance = _require_mapping(manifest["provenance"], "provenance")
    model_config = _require_mapping(manifest["model"], "model")
    resolved_args = _require_mapping(manifest["resolved_args"], "resolved_args")
    if provenance.get("dataset") != "rule30":
        raise ValueError(f"expected Rule 30, found {provenance.get('dataset')!r}")
    supported_models = (
        DCA_EVALUATION_MODELS
        if args.evaluate_delayed_recall
        else {"ca_cotf", "ca_cotf_cache_attn"}
    )
    if model_config.get("model") not in supported_models:
        raise ValueError(
            f"expected one of {sorted(supported_models)}, "
            f"found {model_config.get('model')!r}"
        )

    config_keys = ("model", "n_layer_begin", "n_layer", "n_layer_end",
                   "n_embd", "n_head", "attention_mode")
    for key in config_keys:
        if model_config.get(key) != resolved_args.get(key):
            raise ValueError(f"manifest model.{key} does not match resolved_args.{key}")

    checkpoint_name = Path(args.checkpoint_name)
    if checkpoint_name.name != args.checkpoint_name or checkpoint_name.suffix != ".pt":
        raise ValueError("checkpoint name must be a .pt basename")
    checkpoint_dir_value = provenance.get("checkpoint_dir")
    if not isinstance(checkpoint_dir_value, str) or not checkpoint_dir_value:
        raise ValueError("provenance.checkpoint_dir must be a non-empty path")
    checkpoint_path = Path(checkpoint_dir_value) / checkpoint_name
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    return manifest, checkpoint_path


def _validate_operation_args(args):
    factorial = (
        args.capture_factorial_transition_batch
        or args.evaluate_factorial_full_split
    )
    source_depth_supplied = args.intervention_source_depth is not None
    if factorial != source_depth_supplied:
        raise ValueError(
            "a factorial operation and --intervention-source-depth "
            "must be supplied together"
        )
    if args.repeat_cache_window is not None and not args.capture_first_condition_batch:
        raise ValueError(
            "--repeat-cache-window requires --capture-first-condition-batch"
        )
    if args.verify_full_cache_none and not args.capture_first_condition_batch:
        raise ValueError(
            "--verify-full-cache-none requires --capture-first-condition-batch"
        )
    if args.verify_full_cache_none and args.repeat_cache_window is not None:
        raise ValueError(
            "--verify-full-cache-none requires the full-cache condition"
        )
    if args.evaluate_delayed_recall:
        legacy_operation = (
            args.probe_forward
            or args.capture_first_condition_batch
            or args.capture_factorial_transition_batch
            or args.evaluate_factorial_full_split
            or args.repeat_cache_window is not None
            or args.verify_full_cache_none
            or args.compare_repeat_horizon_backends
        )
        if legacy_operation:
            raise ValueError(
                "Standalone DCA delayed-recall evaluation cannot be combined "
                "with the legacy CA-attention operations in one invocation."
            )
        if args.num_repeats is None:
            raise ValueError(
                "--evaluate-delayed-recall requires --num-repeats"
            )
        query_repeats = (
            args.query_repeats
            if args.query_repeats is not None
            else range(1, args.num_repeats + 1)
        )
        invalid_queries = [
            query_repeat
            for query_repeat in query_repeats
            if query_repeat > args.num_repeats
        ]
        if invalid_queries:
            raise ValueError(
                "query repeats must be between 1 and --num-repeats; got "
                + ", ".join(str(value) for value in invalid_queries)
            )
    else:
        dca_only_requested = (
            args.num_repeats is not None
            or args.query_repeats is not None
            or tuple(args.conditions) != ("baseline",)
            or args.max_batches is not None
            or args.num_examples != 4
            or args.collect_recall_attention
            or args.intervention_permutation_offset != 1
        )
        if dca_only_requested:
            raise ValueError(
                "standalone DCA evaluation arguments require "
                "--evaluate-delayed-recall"
            )


def main(argv=None):
    args = parse_args(argv)
    artifact_context = None
    try:
        _validate_operation_args(args)
        manifest, checkpoint_path = validate_source_run(args)
        attention_override = None
        if args.evaluate_delayed_recall:
            source_model = manifest["model"]["model"]
            conditions = _normalise_dca_conditions(args.conditions)
            manual_attention_conditions = {
                "target-repeat-only",
                "target-repeat-masked",
            }
            if source_model == DCA_CACHE_INTERVENTION_MODEL and (
                args.collect_recall_attention
                or manual_attention_conditions.intersection(conditions)
            ):
                attention_override = "manual"
        model_args = resolve_model_args(
            manifest, attention_implementation=attention_override
        )
        model, checkpoint_step, state_entries = load_frozen_model(
            model_args, checkpoint_path, args.expected_step, args.device
        )
        diagnostic_loader, dataset_info = build_fixed_diagnostic_dataset(
            model_args,
            args.diagnostic_split,
            args.device,
            args.diagnostic_data_mode,
        )
        forward_probe = None
        backend_comparison = None
        condition_batch_capture = None
        factorial_batch_capture = None
        factorial_full_split_evaluation = None
        dca_delayed_recall_evaluation = None
        max_repeats = args.max_repeats
        if args.evaluate_delayed_recall:
            max_repeats = args.num_repeats
        if max_repeats is None:
            max_repeats = model_args.ca_repeat_diagnostic_max_repeats
        artifact_context = initialize_artifact(
            args,
            manifest,
            checkpoint_path,
            checkpoint_step,
            model_args,
            dataset_info,
            max_repeats,
        )
        if args.capture_first_condition_batch:
            if max_repeats is None:
                raise ValueError(
                    "max repeats must be supplied for condition batch capture"
                )
            condition_batch_capture = capture_first_condition_batch(
                model,
                diagnostic_loader,
                model_args,
                args.device,
                int(max_repeats),
                artifact_context["directory"],
                repeat_cache_window=args.repeat_cache_window,
                verify_full_cache_none=args.verify_full_cache_none,
            )
            register_artifact_shard(
                artifact_context, condition_batch_capture
            )
        if args.capture_factorial_transition_batch:
            factorial_batch_capture = capture_factorial_transition_batch(
                model,
                diagnostic_loader,
                model_args,
                args.device,
                args.intervention_source_depth,
                int(max_repeats),
                artifact_context["directory"],
                trained_cache_window=getattr(
                    model_args, "repeat_cache_window", None
                ),
            )
            register_artifact_shard(artifact_context, factorial_batch_capture)
        if args.evaluate_factorial_full_split:
            factorial_full_split_evaluation = evaluate_factorial_full_split(
                model,
                diagnostic_loader,
                model_args,
                args.device,
                args.intervention_source_depth,
                int(max_repeats),
                trained_cache_window=getattr(
                    model_args, "repeat_cache_window", None
                ),
            )
        if args.probe_forward:
            if max_repeats is None:
                raise ValueError("max repeats must be supplied for the forward probe")
            forward_probe = run_forward_equivalence_probe(
                model,
                diagnostic_loader,
                model_args,
                args.device,
                int(max_repeats),
                args.equivalence_repeats,
            )
        if args.compare_repeat_horizon_backends:
            if max_repeats is None:
                raise ValueError(
                    "max repeats must be supplied for the backend comparison"
                )
            backend_comparison = run_repeat_horizon_backend_comparison(
                manifest,
                checkpoint_path,
                args.expected_step,
                diagnostic_loader,
                args.device,
                int(max_repeats),
            )
        if args.evaluate_delayed_recall:
            dca_delayed_recall_evaluation = evaluate_dca_checkpoint(
                model,
                diagnostic_loader,
                model_args,
                args,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        mark_artifact_failed(artifact_context, error)
        raise SystemExit(f"error: {error}") from error
    summary = {
        "run_id": manifest["run_id"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "device": str(next(model.parameters()).device),
        "diagnostic_dtype": str(model_args.dtype),
        "diagnostic_model": model_args.model,
        "eval_mode": not model.training,
        "dataset": dataset_info,
        "state_entries": state_entries,
        "model": manifest["model"],
    }
    if forward_probe is not None:
        summary["forward_probe"] = forward_probe
    if backend_comparison is not None:
        summary["repeat_horizon_backend_comparison"] = backend_comparison
    if condition_batch_capture is not None:
        summary["condition_batch_capture"] = condition_batch_capture
    if factorial_batch_capture is not None:
        summary["factorial_batch_capture"] = factorial_batch_capture
    if factorial_full_split_evaluation is not None:
        summary["factorial_full_split_evaluation"] = (
            factorial_full_split_evaluation
        )
    if dca_delayed_recall_evaluation is not None:
        summary["dca_delayed_recall_evaluation"] = (
            dca_delayed_recall_evaluation
        )
    try:
        if artifact_context is not None:
            summary["artifact_directory"] = str(artifact_context["directory"])
            complete_artifact(artifact_context, summary)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        mark_artifact_failed(artifact_context, error)
        raise SystemExit(f"error: {error}") from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
