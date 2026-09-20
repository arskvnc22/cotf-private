"""Evaluate model-level delayed-recall cache interventions.

For every requested ``(horizon, intervention depth)`` pair this evaluator:

1. runs an unintervened PyTorch SDPA baseline;
2. runs an unintervened manual-attention baseline on the same inputs;
3. checks and reports their logit and metric deltas; and
4. runs each requested model-level intervention with manual attention.

The evolution always runs for ``horizon`` repeats.  ``intervention depth`` is
the stored repeat requested by the backwards controller, so recall age is
``horizon - intervention_depth``.  Interventions affect only recall.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from numbers import Integral
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
import models

try:
    from .dca_eval import evaluate_delayed_recall_query
    from .ca_forward import CAForwardContext, CAForwardPolicy
    from .ca_gen import MaterializedRule30Dataset, Rule30Dataset
    from .ca_reporting import read_json, utc_now, validate_manifest, write_json
except ImportError:
    from dca_eval import evaluate_delayed_recall_query
    from ca_forward import CAForwardContext, CAForwardPolicy
    from ca_gen import MaterializedRule30Dataset, Rule30Dataset
    from ca_reporting import read_json, utc_now, validate_manifest, write_json


SCHEMA_VERSION = 2
SOURCE_MODEL = "dca_cotf_cache"
RUNTIME_MODEL = "dca_cotf_att_intervene"
INTERVENTIONS = (
    "target-value-corruption",
    "target-repeat-only",
    "target-repeat-masked",
)
METRIC_FIELDS = (
    "loss",
    "cell_accuracy",
    "exact_sequence_accuracy",
    "mean_bit_errors_per_sequence",
    "zero_accuracy",
    "one_accuracy",
    "target_one_rate",
    "predicted_one_rate",
    "matthews_correlation",
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--expected-step", type=nonnegative_int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--horizons",
        type=positive_int,
        nargs="+",
        required=True,
        help="Total evolution repeats before recall.",
    )
    parser.add_argument(
        "--intervention-depths",
        type=positive_int,
        nargs="+",
        help=(
            "Stored repeat depths to query/intervene on. If omitted, evaluate "
            "every valid depth 1..H independently for each horizon H."
        ),
    )
    parser.add_argument(
        "--num-recall-repeats",
        type=positive_int,
        help=(
            "Recall passes after evolution. Defaults to ca_recall_repeats "
            "from the source run manifest."
        ),
    )
    parser.add_argument(
    "--eval-batch-size",
    type=positive_int,
    help=(
        "Evaluation batch-size override. Defaults to ca_eval_batch_size "
        "from the source manifest."
    ),
    )
    parser.add_argument(
        "--backend-check-policy",
        choices=("all", "once-per-horizon", "none"),
        default="once-per-horizon",
        help=(
            "Run SDPA/manual equivalence at every requested depth, only at "
            "the greatest requested depth of each horizon, or not at all."
        ),
    )

    parser.add_argument(
        "--interventions",
        choices=INTERVENTIONS,
        nargs="+",
        default=list(INTERVENTIONS),
    )
    parser.add_argument(
        "--value-permutation-offset",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--diagnostic-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--diagnostic-data-mode",
        choices=("manifest", "indexed", "materialized"),
        default="manifest",
    )
    parser.add_argument("--max-batches", type=positive_int)
    parser.add_argument(
        "--collect-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Report requested-repeat probability mass by recall pass, layer, "
            "and head (enabled by default)."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=nonnegative_int,
        default=0,
        help="Example rows retained per condition; zero keeps output compact.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_evaluation_pairs(horizons, intervention_depths, max_relative_age):
    """Return valid pairs and explicit records of impossible cartesian pairs."""
    horizons = sorted(set(int(value) for value in horizons))
    if not horizons or horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers")
    if max(horizons) > int(max_relative_age):
        raise ValueError(
            f"largest horizon {max(horizons)} exceeds ca_max_relative_age "
            f"{max_relative_age}"
        )

    if intervention_depths is None:
        return (
            [
                (horizon, depth)
                for horizon in horizons
                for depth in range(1, horizon + 1)
            ],
            [],
        )

    depths = sorted(set(int(value) for value in intervention_depths))
    if not depths or depths[0] <= 0:
        raise ValueError("intervention depths must contain positive integers")

    pairs = []
    skipped = []
    used_depths = set()
    for horizon in horizons:
        for depth in depths:
            if depth <= horizon:
                pairs.append((horizon, depth))
                used_depths.add(depth)
            else:
                skipped.append(
                    {
                        "horizon": horizon,
                        "intervention_depth": depth,
                        "reason": "intervention_depth_exceeds_horizon",
                    }
                )

    unused = [depth for depth in depths if depth not in used_depths]
    if unused:
        raise ValueError(
            "intervention depths are invalid for every requested horizon: "
            f"{unused}"
        )
    return pairs, skipped


def resolve_num_recall_repeats(requested, model_args):
    value = (
        getattr(model_args, "ca_recall_repeats", None)
        if requested is None
        else requested
    )
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(
            "num recall repeats must be a positive integer, either explicitly "
            "or in resolved_args.ca_recall_repeats"
        )
    return int(value)


def _require_mapping(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_source_run(args):
    if not args.run_manifest.is_file():
        raise ValueError(f"run manifest does not exist: {args.run_manifest}")
    manifest = _require_mapping(read_json(args.run_manifest), "run manifest")
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid run manifest: " + "; ".join(errors))
    if manifest["run_id"] != args.expected_run_id:
        raise ValueError(
            f"expected run {args.expected_run_id!r}, found {manifest['run_id']!r}"
        )
    if manifest["status"] != "completed":
        raise ValueError(
            f"run status is {manifest['status']!r}, not 'completed'"
        )

    provenance = _require_mapping(manifest["provenance"], "provenance")
    model_config = _require_mapping(manifest["model"], "model")
    resolved_args = _require_mapping(manifest["resolved_args"], "resolved_args")
    if provenance.get("dataset") != "rule30":
        raise ValueError(
            f"expected Rule 30, found {provenance.get('dataset')!r}"
        )
    if model_config.get("model") != SOURCE_MODEL:
        raise ValueError(
            f"expected source model {SOURCE_MODEL!r}, found "
            f"{model_config.get('model')!r}"
        )
    if resolved_args.get("model") != SOURCE_MODEL:
        raise ValueError("manifest model and resolved_args.model disagree")

    resolved_window = resolved_args.get("repeat_cache_window")
    policy = _require_mapping(manifest.get("forward_policy"), "forward_policy")
    policy_window = policy.get("repeat_cache_window")
    if resolved_window is not None or policy_window is not None:
        raise ValueError(
            "recall-cache interventions require a checkpoint trained with the "
            "full-cache policy (repeat_cache_window=None)"
        )

    checkpoint_name = Path(args.checkpoint_name)
    if (
        checkpoint_name.name != args.checkpoint_name
        or checkpoint_name.suffix != ".pt"
    ):
        raise ValueError("checkpoint name must be a .pt basename")
    checkpoint_dir = provenance.get("checkpoint_dir")
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("provenance.checkpoint_dir must be a non-empty path")
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    return manifest, checkpoint_path


def resolve_model_args(manifest):
    values = dict(_require_mapping(manifest["resolved_args"], "resolved_args"))
    config_format = values.get("config_format")
    if config_format not in config.registered_formats():
        raise ValueError(f"unsupported config format: {config_format!r}")
    parsed = config.parse_args_with_format(
        format=config_format,
        base_parser=argparse.ArgumentParser(add_help=False, allow_abbrev=False),
        args=[],
        namespace=argparse.Namespace(**values),
    )
    if not isinstance(parsed.dtype, torch.dtype):
        raise ValueError(f"unsupported dtype: {parsed.dtype!r}")
    model_args = argparse.Namespace(**values)
    model_args.dtype = parsed.dtype
    model_args.model = RUNTIME_MODEL
    model_args.repeat_cache_window = None
    return model_args


def load_frozen_model(model_args, checkpoint_path, expected_step, device):
    checkpoint = _require_mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        "checkpoint",
    )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint.model must be a non-empty state dictionary")
    checkpoint_step = checkpoint.get("itr")
    if hasattr(checkpoint_step, "item"):
        checkpoint_step = checkpoint_step.item()
    if isinstance(checkpoint_step, bool) or not isinstance(
        checkpoint_step, Integral
    ):
        raise ValueError("checkpoint.itr must be an integer")
    if checkpoint_step != expected_step:
        raise ValueError(
            f"expected checkpoint step {expected_step}, found {checkpoint_step}"
        )

    model = models.make_model_from_args(model_args)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    if not hasattr(model, "transformer") or not model.transformer.h_mid:
        raise ValueError("interventions require at least one middle layer")
    return model, int(checkpoint_step), len(state_dict)


def build_fixed_diagnostic_dataset(
    model_args,
    split,
    device,
    data_mode,
    eval_batch_size=None,
    ):

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
    batch_size = int(
            eval_batch_size
            or model_args.ca_eval_batch_size
            or model_args.batch_size
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(model_args.ca_num_workers),
        pin_memory=torch.device(device).type == "cuda",
    )
    return loader, {
        "split": split,
        "data_mode": selected_mode,
        "num_samples": len(dataset),
        "num_cells": num_cells,
        "steps": int(model_args.ca_steps),
        "bernoulli_p": float(model_args.ca_bernoulli_p),
        "seed": split_seed,
        "batch_size": batch_size,
        "num_batches": len(loader),
        "shuffle": False,
    }


def autocast_context(model_args, device):
    if torch.device(device).type != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=model_args.dtype)


@contextmanager
def attention_backend(model, backend):
    """Temporarily change implementation only; never alter attention_mode."""
    if backend not in ("manual", "sdpa"):
        raise ValueError(f"unsupported attention backend: {backend!r}")
    blocks = (
        list(model.transformer.h_begin)
        + list(model.transformer.h_mid)
        + list(model.transformer.h_end)
    )
    attention_modules = [block.attn for block in blocks]
    original = [module.attention_implementation for module in attention_modules]
    try:
        for module in attention_modules:
            module.attention_implementation = backend
        yield
    finally:
        for module, implementation in zip(attention_modules, original):
            module.attention_implementation = implementation


class BatchTrace:
    """Fingerprint the ordered model inputs to prove paired data identity."""

    def __init__(self):
        self._digest = hashlib.sha256()
        self.batch_shapes = []
        self.num_examples = 0

    def add(self, inputs):
        tensor = inputs.detach().to(device="cpu").contiguous()
        self._digest.update(str(tensor.dtype).encode("ascii"))
        self._digest.update(str(tuple(tensor.shape)).encode("ascii"))
        self._digest.update(tensor.numpy().tobytes())
        self.batch_shapes.append(list(tensor.shape))
        self.num_examples += int(tensor.shape[0])

    def finalize(self):
        return {
            "sha256": self._digest.hexdigest(),
            "num_batches": len(self.batch_shapes),
            "num_examples": self.num_examples,
            "batch_shapes": self.batch_shapes,
        }


class LogitCapture:
    def __init__(self):
        self.batches = []

    def add(self, logits):
        self.batches.append(logits.detach().to(device="cpu"))


def compare_logit_captures(sdpa_capture, manual_capture):
    sdpa_batches = sdpa_capture.batches
    manual_batches = manual_capture.batches
    if len(sdpa_batches) != len(manual_batches):
        raise RuntimeError("attention backends produced different batch counts")
    if not sdpa_batches:
        raise RuntimeError("attention backend comparison captured no logits")

    dtypes = {batch.dtype for batch in (*sdpa_batches, *manual_batches)}
    if torch.bfloat16 in dtypes:
        rtol, atol = 1e-2, 1e-3
    elif torch.float16 in dtypes:
        rtol, atol = 5e-3, 5e-4
    else:
        rtol, atol = 1e-5, 1e-6

    exact = True
    within_tolerance = True
    max_absolute_delta = 0.0
    sum_absolute_delta = 0.0
    num_logits = 0
    matching_predictions = 0
    num_predictions = 0
    for sdpa, manual in zip(sdpa_batches, manual_batches):
        if sdpa.shape != manual.shape:
            raise RuntimeError(
                f"attention backend logit shapes differ: {tuple(sdpa.shape)} "
                f"and {tuple(manual.shape)}"
            )
        exact = exact and torch.equal(sdpa, manual)
        within_tolerance = within_tolerance and torch.allclose(
            sdpa, manual, rtol=rtol, atol=atol
        )
        delta = sdpa.float().sub(manual.float()).abs()
        max_absolute_delta = max(max_absolute_delta, float(delta.max().item()))
        sum_absolute_delta += float(delta.sum().item())
        num_logits += delta.numel()
        predictions_equal = sdpa.argmax(dim=-1).eq(manual.argmax(dim=-1))
        matching_predictions += int(predictions_equal.sum().item())
        num_predictions += predictions_equal.numel()

    return {
        "exact_logit_equality": exact,
        "logits_within_tolerance": within_tolerance,
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "max_absolute_logit_delta": max_absolute_delta,
        "mean_absolute_logit_delta": sum_absolute_delta / num_logits,
        "prediction_agreement": matching_predictions / num_predictions,
        "num_logits": num_logits,
    }


class RequestedRepeatAttention:
      """Aggregate repeat-block attention at recall steps only.

      Results remain separate by recall pass and middle layer. Masses are
      averaged over evaluated examples and query positions, but retained
      separately for every attention head and cached repeat block.

      Ranking rates are calculated over individual example-position
      observations, separately for each head.
      """

      def __init__(self, *, horizon, intervention_depth):
          self.horizon = int(horizon)
          self.intervention_depth = int(intervention_depth)
          self.records = {}

      @staticmethod
      def _ranking_sums(repeat_mass, requested_index):
          """Return per-head ranking sums for one batch.

          repeat_mass has shape [batch, heads, query_positions, cache_blocks].
          Ties within tolerance count as jointly best. Mean rank uses competition
          ranking: one plus the number of blocks strictly above the requested one.
          """
          requested = repeat_mass[..., requested_index]
          requested_expanded = requested.unsqueeze(-1)

          tied_with_requested = torch.isclose(
              repeat_mass,
              requested_expanded,
              rtol=1e-5,
              atol=1e-7,
          )
          strictly_better = (
              repeat_mass > requested_expanded
          ) & ~tied_with_requested
          rank = 1 + strictly_better.sum(dim=-1)

          maximum = repeat_mass.max(dim=-1).values
          requested_is_best = torch.isclose(
              requested,
              maximum,
              rtol=1e-5,
              atol=1e-7,
          )
          tied_for_best = torch.isclose(
              repeat_mass,
              maximum.unsqueeze(-1),
              rtol=1e-5,
              atol=1e-7,
          )
          requested_is_unique_best = requested_is_best & (
              tied_for_best.sum(dim=-1) == 1
          )

          return {
              "rank_sum": rank.float().sum(dim=(0, 2)).cpu(),
              "best_sum": requested_is_best.sum(dim=(0, 2)).cpu(),
              "unique_best_sum": requested_is_unique_best.sum(
                  dim=(0, 2)
              ).cpu(),
          }

      def add(self, diagnostics, *, batch_size):
          requested_index = self.intervention_depth - 1

          for diagnostic in diagnostics:
              absolute_repeat = int(diagnostic["repeat_index"])

              # Evolution-step diagnostics are intentionally excluded. Only
              # attention calls made during recall are aggregated.
              if absolute_repeat <= self.horizon:
                  continue

              recall_index = absolute_repeat - self.horizon
              layer_index = int(diagnostic["middle_layer_index"])
              key = (recall_index, layer_index)

              repeat_mass = (
                  diagnostic["repeat_mass"]
                  .detach()
                  .float()
              )
              if repeat_mass.ndim != 4:
                  raise RuntimeError(
                      "repeat_mass must have shape "
                      "[batch, heads, query_positions, cache_blocks]"
                  )

              observed_batch, _, num_positions, cached_repeats = (
                  repeat_mass.shape
              )
              if observed_batch != int(batch_size):
                  raise RuntimeError(
                      "attention diagnostic batch size does not match inputs"
                  )

              expected_cached_repeats = self.horizon + recall_index
              if cached_repeats != expected_cached_repeats:
                  raise RuntimeError(
                      "attention cache depth does not match recall index"
                  )

              if not 0 <= requested_index < self.horizon:
                  raise RuntimeError(
                      "requested repeat is outside the evolution cache"
                  )

              if not torch.allclose(
                  repeat_mass.sum(dim=-1),
                  torch.ones_like(repeat_mass[..., 0]),
                  rtol=1e-4,
                  atol=1e-5,
              ):
                  raise RuntimeError(
                      "per-position repeat attention does not sum to one"
                  )

              value_norm = diagnostic["value_norm"].detach().float()
              expected_value_shape = (
                  observed_batch,
                  repeat_mass.shape[1],
                  cached_repeats,
                  num_positions,
              )
              if tuple(value_norm.shape) != expected_value_shape:
                  raise RuntimeError(
                      "value_norm must have shape "
                      "[batch, heads, cache_blocks, token_positions]"
                  )

              block_contribution_norm = (
                  diagnostic["block_contribution_norm"].detach().float()
              )
              if tuple(block_contribution_norm.shape) != tuple(
                  repeat_mass.shape
              ):
                  raise RuntimeError(
                      "block_contribution_norm must match repeat_mass shape"
                  )

              for name, values in (
                  ("value_norm", value_norm),
                  ("block_contribution_norm", block_contribution_norm),
              ):
                  if not torch.isfinite(values).all() or (values < 0).any():
                      raise RuntimeError(
                          f"{name} must be finite and nonnegative"
                      )

              # Sum over examples and query positions, preserving heads and
              # cache blocks.
              mass_sum = repeat_mass.sum(dim=(0, 2)).cpu()
              value_norm_sum = value_norm.sum(dim=(0, 3)).cpu()
              contribution_norm_sum = block_contribution_norm.sum(
                  dim=(0, 2)
              ).cpu()
              evolution_ranking = self._ranking_sums(
                  repeat_mass[..., : self.horizon],
                  requested_index,
              )
              all_cache_ranking = self._ranking_sums(
                  repeat_mass,
                  requested_index,
              )

              entry = self.records.setdefault(
                  key,
                  {
                      "examples": 0,
                      "positions_per_example": num_positions,
                      "observations_per_head": 0,
                      "value_observations_per_head_and_block": 0,
                      "contribution_observations_per_head_and_block": 0,
                      "mass_sum": torch.zeros_like(mass_sum),
                      "value_norm_sum": torch.zeros_like(value_norm_sum),
                      "contribution_norm_sum": torch.zeros_like(
                          contribution_norm_sum
                      ),
                      "evolution_rank_sum": torch.zeros_like(
                          evolution_ranking["rank_sum"]
                      ),
                      "evolution_best_sum": torch.zeros_like(
                          evolution_ranking["best_sum"]
                      ),
                      "evolution_unique_best_sum": torch.zeros_like(
                          evolution_ranking["unique_best_sum"]
                      ),
                      "all_cache_rank_sum": torch.zeros_like(
                          all_cache_ranking["rank_sum"]
                      ),
                      "all_cache_best_sum": torch.zeros_like(
                          all_cache_ranking["best_sum"]
                      ),
                      "all_cache_unique_best_sum": torch.zeros_like(
                          all_cache_ranking["unique_best_sum"]
                      ),
                  },
              )

              if entry["mass_sum"].shape != mass_sum.shape:
                  raise RuntimeError(
                      "attention head count or cache depth changed between batches"
                  )
              if (
                  entry["value_norm_sum"].shape != value_norm_sum.shape
                  or entry["contribution_norm_sum"].shape
                  != contribution_norm_sum.shape
              ):
                  raise RuntimeError(
                      "value diagnostic head count or cache depth changed "
                      "between batches"
                  )
              if entry["positions_per_example"] != num_positions:
                  raise RuntimeError(
                      "query-position count changed between batches"
                  )

              observations = observed_batch * num_positions
              entry["examples"] += observed_batch
              entry["observations_per_head"] += observations
              entry["value_observations_per_head_and_block"] += (
                  observed_batch * num_positions
              )
              entry["contribution_observations_per_head_and_block"] += (
                  observed_batch * num_positions
              )
              entry["mass_sum"] += mass_sum
              entry["value_norm_sum"] += value_norm_sum
              entry["contribution_norm_sum"] += contribution_norm_sum
              entry["evolution_rank_sum"] += evolution_ranking["rank_sum"]
              entry["evolution_best_sum"] += evolution_ranking["best_sum"]
              entry["evolution_unique_best_sum"] += evolution_ranking[
                  "unique_best_sum"
              ]
              entry["all_cache_rank_sum"] += all_cache_ranking["rank_sum"]
              entry["all_cache_best_sum"] += all_cache_ranking["best_sum"]
              entry["all_cache_unique_best_sum"] += all_cache_ranking[
                  "unique_best_sum"
              ]

      def finalize(self):
          rows = []
          requested_index = self.intervention_depth - 1

          for (recall_index, layer_index), entry in sorted(
              self.records.items()
          ):
              observations = entry["observations_per_head"]
              mean_mass = entry["mass_sum"] / observations
              mean_value_norm = entry["value_norm_sum"] / entry[
                  "value_observations_per_head_and_block"
              ]
              mean_contribution_norm = entry["contribution_norm_sum"] / entry[
                  "contribution_observations_per_head_and_block"
              ]

              requested = mean_mass[:, requested_index]
              evolution = mean_mass[:, : self.horizon].sum(dim=1)
              recall = mean_mass[:, self.horizon :].sum(dim=1)
              other_evolution = evolution - requested

              evolution_rank = entry["evolution_rank_sum"] / observations
              evolution_best = entry["evolution_best_sum"] / observations
              evolution_unique_best = (
                  entry["evolution_unique_best_sum"] / observations
              )
              all_cache_rank = entry["all_cache_rank_sum"] / observations
              all_cache_best = entry["all_cache_best_sum"] / observations
              all_cache_unique_best = (
                  entry["all_cache_unique_best_sum"] / observations
              )

              cache_blocks = []
              for block_index in range(mean_mass.shape[1]):
                  if block_index < self.horizon:
                      cache_blocks.append(
                          {
                              "cache_block_index_one_based": block_index + 1,
                              "phase": "evolution",
                              "evolution_repeat": block_index + 1,
                          }
                      )
                  else:
                      cache_blocks.append(
                          {
                              "cache_block_index_one_based": block_index + 1,
                              "phase": "recall",
                              "recall_index": (
                                  block_index - self.horizon + 1
                              ),
                          }
                      )

              rows.append(
                  {
                      "recall_index": recall_index,
                      "middle_layer_index_zero_based": layer_index,
                      "requested_evolution_repeat": self.intervention_depth,
                      "cache_blocks": cache_blocks,

                      # Complete mean distribution. The outer list is heads;
                      # the inner list follows cache_blocks.
                      "mean_mass_by_head_and_cache_block": (
                          mean_mass.tolist()
                      ),
                      "mean_mass_by_cache_block": (
                          mean_mass.mean(dim=0).tolist()
                      ),
                      "mean_value_norm_by_head_and_cache_block": (
                          mean_value_norm.tolist()
                      ),
                      "mean_value_norm_by_cache_block": (
                          mean_value_norm.mean(dim=0).tolist()
                      ),
                      "mean_block_contribution_norm_by_head_and_cache_block": (
                          mean_contribution_norm.tolist()
                      ),
                      "mean_block_contribution_norm_by_cache_block": (
                          mean_contribution_norm.mean(dim=0).tolist()
                      ),

                      # Existing aggregate fields retained for compatibility.
                      "requested_repeat_mass_mean": float(requested.mean()),
                      "requested_repeat_mass_by_head": requested.tolist(),
                      "other_evolution_mass_mean": float(
                          other_evolution.mean()
                      ),
                      "other_evolution_mass_by_head": (
                          other_evolution.tolist()
                      ),
                      "recall_block_mass_mean": float(recall.mean()),
                      "recall_block_mass_by_head": recall.tolist(),

                      # Requested-repeat ranking among evolution blocks only.
                      "requested_mean_rank_among_evolution_by_head": (
                          evolution_rank.tolist()
                      ),
                      "requested_is_best_among_evolution_rate_by_head": (
                          evolution_best.tolist()
                      ),
                      "requested_is_unique_best_among_evolution_rate_by_head": (
                          evolution_unique_best.tolist()
                      ),

                      # Ranking against every cached block, including recall
                      # blocks present at this recall pass.
                      "requested_mean_rank_among_all_cache_blocks_by_head": (
                          all_cache_rank.tolist()
                      ),
                      "requested_is_best_among_all_cache_blocks_rate_by_head": (
                          all_cache_best.tolist()
                      ),
                      "requested_is_unique_best_among_all_cache_blocks_rate_by_head": (
                          all_cache_unique_best.tolist()
                      ),

                      "examples": entry["examples"],
                      "positions_per_example": (
                          entry["positions_per_example"]
                      ),
                      "attention_observations_per_head": observations,
                      "value_observations_per_head_and_cache_block": entry[
                          "value_observations_per_head_and_block"
                      ],
                      "contribution_observations_per_head_and_cache_block": (
                          entry[
                              "contribution_observations_per_head_and_block"
                          ]
                      ),
                  }
              )

          return rows



def validate_attention_intervention(condition, attention_rows):
    tolerance = 1e-6
    if condition == "target-repeat-masked" and any(
        row["requested_repeat_mass_mean"] > tolerance for row in attention_rows
    ):
        raise RuntimeError(
            "target-repeat-masked left nonzero requested-repeat mass"
        )
    if condition == "target-repeat-only" and any(
        row["other_evolution_mass_mean"] > tolerance for row in attention_rows
    ):
        raise RuntimeError(
            "target-repeat-only left other evolution blocks visible"
        )


def validate_intervention_metadata(
    records,
    *,
    condition,
    intervention_depth,
    horizon,
    num_recall_repeats,
    num_middle_layers,
):
    if not records:
        raise RuntimeError("model returned no recall intervention metadata")
    for record in records:
        if record["condition"] != condition:
            raise RuntimeError("condition metadata mismatch")
        if record["target_repeat"] != intervention_depth:
            raise RuntimeError("target-repeat metadata mismatch")
        if record["recall_age"] != horizon - intervention_depth:
            raise RuntimeError("recall-age metadata mismatch")
        if record["num_evolution_repeats"] != horizon:
            raise RuntimeError("evolution-horizon metadata mismatch")
        if record["num_recall_repeats"] != num_recall_repeats:
            raise RuntimeError("recall-repeat metadata mismatch")

        if condition == "target-value-corruption":
            if record["value_permutations"] != num_middle_layers:
                raise RuntimeError(
                    "not every middle-layer value cache was corrupted"
                )
            if record["masked_attention_calls"] != 0:
                raise RuntimeError("value corruption unexpectedly applied a mask")
        elif condition in ("target-repeat-only", "target-repeat-masked"):
            expected_calls = num_recall_repeats * num_middle_layers
            if record["masked_attention_calls"] != expected_calls:
                raise RuntimeError("unexpected number of masked attention calls")
            if record["value_permutations"] != 0:
                raise RuntimeError("masking unexpectedly modified cached values")
        elif condition == "baseline" and (
            record["value_permutations"] != 0
            or record["masked_attention_calls"] != 0
        ):
            raise RuntimeError("baseline metadata reports an intervention")


class InterventionForwardContext:
    """Inject one model-level condition and capture returned evidence."""

    def __init__(
        self,
        base_context,
        *,
        condition,
        permutation_offset,
        attention_accumulator=None,
    ):
        self.base_context = base_context
        self.model = base_context.model
        self.condition = condition
        self.permutation_offset = int(permutation_offset)
        self.attention_accumulator = attention_accumulator
        self.intervention_metadata = []
        self.batch_trace = BatchTrace()
        self.logits = LogitCapture()

    def call(self, inputs, **call_kwargs):
        self.batch_trace.add(inputs)
        if self.condition != "baseline":
            call_kwargs["recall_cache_intervention"] = self.condition
            call_kwargs["recall_value_permutation_offset"] = self.permutation_offset
        if self.attention_accumulator is not None:
            call_kwargs["return_attention_diagnostics"] = True

        outputs = self.base_context.call(inputs, **call_kwargs)
        metadata = outputs.get("recall_cache_intervention")
        if metadata is None:
            raise RuntimeError("intervention model did not return recall metadata")
        if metadata["condition"] != self.condition:
            raise RuntimeError("returned intervention condition does not match request")
        self.intervention_metadata.append(metadata)

        logits = outputs.get("logits")
        if logits is not None:
            self.logits.add(logits)
        if self.attention_accumulator is not None:
            diagnostics = outputs.get("attention_diagnostics")
            if diagnostics is None:
                raise RuntimeError("model did not return requested attention diagnostics")
            self.attention_accumulator.add(
                diagnostics, batch_size=int(inputs.shape[0])
            )
        return outputs


def select_fields(mapping, fields):
    return {field: mapping.get(field) for field in fields}


def compact_query_result(result):
    internal = result["internal_consistency"]
    return {
        "metrics": select_fields(result["metrics"], METRIC_FIELDS),
        "ground_truth_retrieval": result["ground_truth_retrieval"],
        "internal_consistency": {
            "decoded_requested_repeat": select_fields(
                internal["decoded_requested_repeat"],
                ("cell_accuracy", "exact_sequence_accuracy"),
            ),
            "requested_repeat_logit_similarity": internal[
                "requested_repeat_logit_similarity"
            ],
            "cosine_retrieval": internal["cosine_retrieval"],
        },
        "examples": result["examples"],
    }


def metric_deltas(reference, comparison):
    deltas = {}
    for name, reference_value in reference["metrics"].items():
        comparison_value = comparison["metrics"].get(name)
        if reference_value is None or comparison_value is None:
            deltas[name] = None
        else:
            deltas[name] = float(comparison_value) - float(reference_value)
    return deltas


def run_condition(
    *,
    model,
    dataloader,
    model_args,
    device,
    horizon,
    intervention_depth,
    num_recall_repeats,
    condition,
    backend,
    permutation_offset,
    max_batches,
    num_examples,
    collect_attention,
):
    attention = (
        RequestedRepeatAttention(
            horizon=horizon, intervention_depth=intervention_depth
        )
        if collect_attention
        else None
    )
    context = InterventionForwardContext(
        CAForwardContext(model, CAForwardPolicy.from_args(model_args)),
        condition=condition,
        permutation_offset=permutation_offset,
        attention_accumulator=attention,
    )
    with attention_backend(model, backend):
        full_result = evaluate_delayed_recall_query(
            context,
            dataloader,
            device,
            num_repeats=horizon,
            query_repeat=intervention_depth,
            num_recall_repeats=num_recall_repeats,
            max_batches=max_batches,
            num_examples=num_examples,
            ctx=autocast_context(model_args, device),
        )

    validate_intervention_metadata(
        context.intervention_metadata,
        condition=condition,
        intervention_depth=intervention_depth,
        horizon=horizon,
        num_recall_repeats=num_recall_repeats,
        num_middle_layers=len(model.transformer.h_mid),
    )
    compact = compact_query_result(full_result)
    compact["backend"] = backend
    compact["input_trace"] = context.batch_trace.finalize()
    compact["intervention_metadata"] = context.intervention_metadata
    if attention is not None:
        rows = attention.finalize()
        expected_rows = num_recall_repeats * len(model.transformer.h_mid)
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} recall attention rows, found {len(rows)}"
            )
        validate_attention_intervention(condition, rows)
        compact["attention"] = rows
    return compact, context.logits


# def evaluate_interventions(
#     *,
#     model,
#     dataloader,
#     model_args,
#     device,
#     evaluation_pairs,
#     num_recall_repeats,
#     interventions,
#     permutation_offset,
#     max_batches,
#     num_examples,
#     collect_attention,
# ):
#     results = []
#     for horizon, intervention_depth in evaluation_pairs:
#         common = {
#             "model": model,
#             "dataloader": dataloader,
#             "model_args": model_args,
#             "device": device,
#             "horizon": horizon,
#             "intervention_depth": intervention_depth,
#             "num_recall_repeats": num_recall_repeats,
#             "permutation_offset": permutation_offset,
#             "max_batches": max_batches,
#             "num_examples": num_examples,
#         }
#         sdpa, sdpa_logits = run_condition(
#             **common,
#             condition="baseline",
#             backend="sdpa",
#             collect_attention=False,
#         )
#         manual, manual_logits = run_condition(
#             **common,
#             condition="baseline",
#             backend="manual",
#             collect_attention=collect_attention,
#         )
#         if sdpa["input_trace"] != manual["input_trace"]:
#             raise RuntimeError("manual and SDPA baselines did not consume identical data")

#         conditions = {"baseline": manual}
#         for condition in interventions:
#             condition_result, _ = run_condition(
#                 **common,
#                 condition=condition,
#                 backend="manual",
#                 collect_attention=collect_attention,
#             )
#             if condition_result["input_trace"] != manual["input_trace"]:
#                 raise RuntimeError(
#                     f"{condition} did not consume the baseline's exact input order"
#                 )
#             conditions[condition] = condition_result

#         results.append(
#             {
#                 "horizon": horizon,
#                 "intervention_depth": intervention_depth,
#                 "recall_age": horizon - intervention_depth,
#                 "is_latest_repeat": intervention_depth == horizon,
#                 "backend_check": {
#                     "attention_mode_held_fixed": model.transformer.h_mid[
#                         0
#                     ].attn.attention_mode,
#                     "reference": "sdpa",
#                     "comparison": "manual",
#                     "same_input_trace": True,
#                     "logits": compare_logit_captures(
#                         sdpa_logits, manual_logits
#                     ),
#                     "manual_minus_sdpa_metrics": metric_deltas(sdpa, manual),
#                     "sdpa_baseline": sdpa,
#                 },
#                 "conditions": conditions,
#                 "effects": {
#                     condition: {
#                         "intervention_minus_manual_baseline": metric_deltas(
#                             manual, conditions[condition]
#                         )
#                     }
#                     for condition in interventions
#                 },
#             }
#         )
#     return results



def evaluate_interventions(
      *,
      model,
      dataloader,
      model_args,
      device,
      evaluation_pairs,
      num_recall_repeats,
      interventions,
      permutation_offset,
      max_batches,
      num_examples,
      collect_attention,
      backend_check_policy="once-per-horizon",
  ):
      if backend_check_policy not in (
          "all",
          "once-per-horizon",
          "none",
      ):
          raise ValueError(
              f"unsupported backend-check policy: {backend_check_policy!r}"
          )

      # For the once-per-horizon policy, use the greatest evaluated depth.
      # This is explicit in every result record and tends to exercise the
      # deepest cache path.
      backend_check_depth_by_horizon = {}
      if backend_check_policy == "once-per-horizon":
          for horizon, intervention_depth in evaluation_pairs:
              previous = backend_check_depth_by_horizon.get(horizon)
              if previous is None or intervention_depth > previous:
                  backend_check_depth_by_horizon[horizon] = intervention_depth

      results = []
      for horizon, intervention_depth in evaluation_pairs:
          common = {
              "model": model,
              "dataloader": dataloader,
              "model_args": model_args,
              "device": device,
              "horizon": horizon,
              "intervention_depth": intervention_depth,
              "num_recall_repeats": num_recall_repeats,
              "permutation_offset": permutation_offset,
              "max_batches": max_batches,
              "num_examples": num_examples,
          }

          should_check_backend = (
              backend_check_policy == "all"
              or (
                  backend_check_policy == "once-per-horizon"
                  and intervention_depth
                  == backend_check_depth_by_horizon[horizon]
              )
          )

          sdpa = None
          sdpa_logits = None
          if should_check_backend:
              sdpa, sdpa_logits = run_condition(
                  **common,
                  condition="baseline",
                  backend="sdpa",
                  collect_attention=False,
                )

          manual, manual_logits = run_condition(
              **common,
              condition="baseline",
              backend="manual",
              collect_attention=collect_attention,
            )

          if should_check_backend:
              if sdpa["input_trace"] != manual["input_trace"]:
                  raise RuntimeError(
                      "manual and SDPA baselines did not consume identical data"
                    ) 

              backend_check = {
                  "performed": True,
                  "policy": backend_check_policy,
                  "checked_intervention_depth": intervention_depth,
                  "attention_mode_held_fixed": model.transformer.h_mid[
                      0
                  ].attn.attention_mode,
                  "reference": "sdpa",
                  "comparison": "manual",
                  "same_input_trace": True,
                  "logits": compare_logit_captures(
                      sdpa_logits,
                      manual_logits,
                  ),
                  "manual_minus_sdpa_metrics": metric_deltas(
                      sdpa,
                      manual,
                  ),
                  "sdpa_baseline": sdpa,
              }
          else:
              backend_check = {
                  "performed": False,
                  "policy": backend_check_policy,
                  "checked_intervention_depth": (
                      backend_check_depth_by_horizon.get(horizon)
                  ),
              }

          conditions = {"baseline": manual}
          for condition in interventions:
              condition_result, _ = run_condition(
                  **common,
                  condition=condition,
                  backend="manual",
                  collect_attention=collect_attention,
              )
              if condition_result["input_trace"] != manual["input_trace"]:
                  raise RuntimeError(
                      f"{condition} did not consume the baseline's exact "
                      "input order"
                  )
              conditions[condition] = condition_result

          results.append(
              {
                  "horizon": horizon,
                  "intervention_depth": intervention_depth,
                  "recall_age": horizon - intervention_depth,
                  "is_latest_repeat": intervention_depth == horizon,
                  "backend_check": backend_check,
                  "conditions": conditions,
                  "effects": {
                      condition: {
                          "intervention_minus_manual_baseline": metric_deltas(
                              manual,
                              conditions[condition],
                          )
                      }
                      for condition in interventions
                  },
              }
          )

      return results


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _training_repeat_horizons(manifest):
    pairs = manifest.get("training", {}).get("pairs", [])
    horizons = []
    for pair in pairs:
        if isinstance(pair, Mapping):
            horizons.append(int(pair["num_repeats"]))
        else:
            horizons.append(int(pair[1]))
    return sorted(set(horizons))


def build_summary(
    *,
    args,
    manifest,
    checkpoint_path,
    checkpoint_step,
    state_entries,
    model,
    dataset_info,
    evaluation_pairs,
    skipped_pairs,
    results,
):
    return _json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": "dca_model_level_recall_cache_interventions",
            "created_at": utc_now(),
            "source": {
                "run_id": manifest["run_id"],
                "run_manifest": str(args.run_manifest.resolve()),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_step": checkpoint_step,
                "checkpoint_state_entries": state_entries,
                "trained_model": SOURCE_MODEL,
                "runtime_model": RUNTIME_MODEL,
                "training_repeat_horizons": _training_repeat_horizons(manifest),
            },
            "protocol": {
                "requested_horizons": sorted(set(args.horizons)),
                "requested_intervention_depths": (
                    None
                    if args.intervention_depths is None
                    else sorted(set(args.intervention_depths))
                ),
                "evaluation_pairs": [
                    {"horizon": horizon, "intervention_depth": depth}
                    for horizon, depth in evaluation_pairs
                ],
                "skipped_pairs": skipped_pairs,
                "num_recall_repeats": args.num_recall_repeats,
                "interventions": list(args.interventions),
                "value_permutation_offset": args.value_permutation_offset,
                "collect_attention": args.collect_attention,
                "max_batches": args.max_batches,
                "eval_batch_size_override": args.eval_batch_size,
                "backend_check_policy": args.backend_check_policy,
                "attention_mode": model.transformer.h_mid[0].attn.attention_mode,
                "condition_order": [
                    "optional_sdpa_baseline",
                    "manual_baseline",
                    "interventions",
                ],
                "backend_check_changes_attention_mode": False,
            },
            "dataset": dataset_info,
            "results": results,
        }
    )


def print_compact_results(summary, output_path):
    def display(value):
        return "       -" if value is None else f"{value:>8.5f}"

    def requested_attention_mean(condition_result):
        rows = condition_result.get("attention", [])
        if not rows:
            return None
        return sum(
            row["requested_repeat_mass_mean"] for row in rows
        ) / len(rows)
    def internal_cell_accuracy(condition_result):
        return condition_result[
            "internal_consistency"
        ]["decoded_requested_repeat"]["cell_accuracy"]


    mode = summary["protocol"]["attention_mode"]
    print(
        f"Attention implementation check "
        f"(attention_mode={mode!r} held fixed):"
    )
    print(
        "  horizon depth checked  max|logit delta|  "
        "prediction agreement  manual cell acc  "
        "manual internal cell acc"
    )

    for result in summary["results"]:
        check = result["backend_check"]
        baseline = result["conditions"]["baseline"]
        performed = check.get("performed", True)

        if performed:
            logits = check["logits"]
            checked = "yes"
            max_delta = f"{logits['max_absolute_logit_delta']:>17.6g}"
            agreement = f"{logits['prediction_agreement']:>20.6f}"
        else:
            checked = "no"
            max_delta = f"{'-':>17}"
            agreement = f"{'-':>20}"

        print(
            f"  {result['horizon']:>7} "
            f"{result['intervention_depth']:>5} "
            f"{checked:>7}  "
            f"{max_delta}  "
            f"{agreement}  "
            f"{baseline['metrics']['cell_accuracy']:>15.6f}  "
            f"{internal_cell_accuracy(baseline):>24.6f}"
        )

    print("\nIntervention results (manual backend; deltas from manual baseline):")
    print(
        "  horizon depth  condition                 cell_acc   d_cell "
        " int_cell    d_int   exact    pred_1  d_pred_1      MCC target_attn"

    )
    for result in summary["results"]:
        baseline = result["conditions"]["baseline"]
        baseline_internal_cell = internal_cell_accuracy(baseline)

        for condition, effect in result["effects"].items():
            delta = effect["intervention_minus_manual_baseline"]
            condition_result = result["conditions"][condition]
            metrics = condition_result["metrics"]
            internal_cell = internal_cell_accuracy(condition_result)
            internal_delta = internal_cell - baseline_internal_cell

            print(
                f"  {result['horizon']:>7} {result['intervention_depth']:>5}  "
                f"{condition:<25} "
                f"{display(metrics['cell_accuracy'])} "
                f"{display(delta['cell_accuracy'])} "
                f"{display(internal_cell)} "
                f"{display(internal_delta)} "
                f"{display(metrics['exact_sequence_accuracy'])} "
                f"{display(metrics['predicted_one_rate'])} "
                f"{display(delta['predicted_one_rate'])} "
                f"{display(metrics['matthews_correlation'])} "
                f"{display(requested_attention_mean(condition_result))}"
            )

    print(
        "  int_cell compares recall output with the model-decoded state of the "
        "requested cached repeat; d_int is its change from the manual baseline."
    )
    print(
        "  target_attn is mean probability mass on the requested cached repeat "
        "across recall passes, middle layers, heads, examples, and positions."
    )
    print(
        "\nNormal-forward recall attention "
        "(manual no-intervention baseline only):"
    )
    print(
        "  horizon depth recall layer0 head0  req_mass "
        "evol_rank evol_best evol_unique "
        "all_rank all_best all_unique"
    )

    attention_rows_printed = 0
    for result in summary["results"]:
        baseline = result["conditions"]["baseline"]

        for row in baseline.get("attention", []):
            requested_by_head = row[
                "requested_repeat_mass_by_head"
            ]
            evolution_rank = row[
                "requested_mean_rank_among_evolution_by_head"
            ]
            evolution_best = row[
                "requested_is_best_among_evolution_rate_by_head"
            ]
            evolution_unique = row[
                "requested_is_unique_best_among_evolution_rate_by_head"
            ]
            all_cache_rank = row[
                "requested_mean_rank_among_all_cache_blocks_by_head"
            ]
            all_cache_best = row[
                "requested_is_best_among_all_cache_blocks_rate_by_head"
            ]
            all_cache_unique = row[
                "requested_is_unique_best_among_all_cache_blocks_rate_by_head"
            ]

            for head_index, requested_mass in enumerate(
                requested_by_head
            ):
                print(
                    f"  {result['horizon']:>7} "
                    f"{result['intervention_depth']:>5} "
                    f"{row['recall_index']:>6} "
                    f"{row['middle_layer_index_zero_based']:>6} "
                    f"{head_index:>5} "
                    f"{requested_mass:>9.5f} "
                    f"{evolution_rank[head_index]:>9.4f} "
                    f"{evolution_best[head_index]:>9.5f} "
                    f"{evolution_unique[head_index]:>11.5f} "
                    f"{all_cache_rank[head_index]:>8.4f} "
                    f"{all_cache_best[head_index]:>8.5f} "
                    f"{all_cache_unique[head_index]:>10.5f}"
                )
                attention_rows_printed += 1

    if attention_rows_printed == 0:
        print("  No baseline attention was collected.")

    print(
        "  *_best is the fraction of evaluated example-position "
        "observations where the requested repeat tied for highest mass."
    )
    print(
        "  *_unique is the fraction where it alone had the highest mass. "
        "evol_* compares only evolution repeats; all_* also includes "
        "recall-phase cache blocks."
    )

    print(f"\nFull results: {output_path}")



def main(argv=None):
    args = parse_args(argv)
    args.interventions = list(dict.fromkeys(args.interventions))
    if args.output_dir.exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    manifest, checkpoint_path = validate_source_run(args)
    model_args = resolve_model_args(manifest)
    args.num_recall_repeats = resolve_num_recall_repeats(
        args.num_recall_repeats, model_args
    )
    evaluation_pairs, skipped_pairs = resolve_evaluation_pairs(
        args.horizons,
        args.intervention_depths,
        model_args.ca_max_relative_age,
    )
    model, checkpoint_step, state_entries = load_frozen_model(
        model_args, checkpoint_path, args.expected_step, args.device
    )
    dataloader, dataset_info = build_fixed_diagnostic_dataset(
          model_args,
          args.diagnostic_split,
          args.device,
          args.diagnostic_data_mode,
          args.eval_batch_size,
      )
    results = evaluate_interventions(
          model=model,
          dataloader=dataloader,
          model_args=model_args,
          device=args.device,
          evaluation_pairs=evaluation_pairs,
          num_recall_repeats=args.num_recall_repeats,
          interventions=tuple(args.interventions),
          permutation_offset=args.value_permutation_offset,
          max_batches=args.max_batches,
          num_examples=args.num_examples,
          collect_attention=args.collect_attention,
          backend_check_policy=args.backend_check_policy,
        )
    summary = build_summary(
        args=args,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        state_entries=state_entries,
        model=model,
        dataset_info=dataset_info,
        evaluation_pairs=evaluation_pairs,
        skipped_pairs=skipped_pairs,
        
        results=results,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_path = args.output_dir / "summary.json"
    write_json(output_path, summary)
    print_compact_results(summary, output_path)
    return summary


if __name__ == "__main__":
    main()
