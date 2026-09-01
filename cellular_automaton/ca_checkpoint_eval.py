"""Evaluate the three selected BUT checkpoints without resuming training."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from collections.abc import Mapping
from contextlib import nullcontext
from numbers import Integral
from pathlib import Path

import torch

import config
import models

from .ca_eval import evaluate_loaded_ca_checkpoint
from .ca_forward import CAForwardContext, CAForwardPolicy
from .ca_main import make_ca_fixed_loaders
from .ca_reporting import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    build_run_manifest,
    ensure_notes_file,
    forward_policy_metadata,
    read_json,
    update_run_manifest_status,
    utc_now,
    validate_manifest,
    write_eval_metrics,
    write_json,
    write_run_manifest,
)


CHECKPOINT_TYPES = (
    "best_id",
    "best_extrapolation_strict",
    "best_extrapolation_unconstrained",
)
CHECKPOINT_METADATA_CANDIDATES = {
    "best_id": ("best_id.json", "best.json"),
    "best_extrapolation_strict": (
        "best_extrapolation_strict.json",
        "best_extrapolation.json",
    ),
    "best_extrapolation_unconstrained": (
        "best_extrapolation_unconstrained.json",
    ),
}
CHECKPOINT_FILE_DEFAULTS = {
    "best_id.json": "best_id.pt",
    "best.json": "best.pt",
    "best_extrapolation_strict.json": "best_extrapolation_strict.pt",
    "best_extrapolation.json": "best_extrapolation.pt",
    "best_extrapolation_unconstrained.json": (
        "best_extrapolation_unconstrained.pt"
    ),
}
TRAINING_STATS_FILENAME = "training_stats.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _require_mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def read_validated_manifest(path):
    """Read a source manifest and reject every schema validation error."""
    path = Path(path)
    try:
        manifest = _require_mapping(read_json(path), "source run manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read source run manifest {path}: {error}") from error
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(
            f"invalid source run manifest {path}: " + "; ".join(errors)
        )
    return dict(manifest)


def _merge_manifest_sections(manifest):
    """Recover optional historical fields without overriding resolved values."""
    resolved = dict(
        _require_mapping(manifest["resolved_args"], "manifest.resolved_args")
    )
    for section_name in (
        "model",
        "training",
        "evaluation",
        "checkpoint_selection",
        "seeds",
    ):
        section = manifest.get(section_name, {})
        if isinstance(section, Mapping):
            for name, value in section.items():
                if name != "pairs":
                    resolved.setdefault(name, value)

    if not resolved.get("ca_train_pairs"):
        pairs = manifest.get("training", {}).get("pairs", [])
        resolved["ca_train_pairs"] = [
            [int(pair["ca_steps"]), int(pair["num_repeats"])]
            for pair in pairs
        ]
    return resolved


def resolve_source_args(manifest, device):
    """Reconstruct model/evaluation arguments with explicit compatibility fallbacks."""
    source_values = _merge_manifest_sections(manifest)
    fallbacks = []

    def fallback(name, value, reason):
        if source_values.get(name) is None:
            source_values[name] = value
            fallbacks.append({"field": name, "value": value, "reason": reason})

    fallback(
        "ca_test_samples",
        source_values.get("ca_val_samples"),
        "historical default: final-test sample count equals validation count",
    )
    fallback(
        "ca_eval_batch_size",
        source_values.get("batch_size"),
        "historical default: evaluation batch size equals training batch size",
    )
    fallback("ca_final_eval_pairs", [], "historical default: no direct final pairs")
    fallback(
        "ca_final_external_steps",
        [],
        "historical default: no external full-model rollouts",
    )
    fallback(
        "ca_repeat_diagnostic_examples",
        1,
        "historical default: print one diagnostic example",
    )
    if source_values.get("ca_repeat_diagnostic_max_repeats") is not None:
        fallback(
            "ca_repeat_diagnostic_horizons",
            list(
                range(
                    int(source_values["ca_repeat_diagnostic_max_repeats"]) + 1
                )
            ),
            "historical default: diagnostic horizons span zero through max repeats",
        )
    if source_values.get("ca_best_pair") is None:
        pairs = [tuple(pair) for pair in source_values.get("ca_train_pairs", [])]
        if pairs:
            fallback(
                "ca_best_pair",
                list(max(pairs)),
                "historical default: largest training pair selects the ID checkpoint",
            )

    required = (
        "config_format",
        "model",
        "dataset",
        "batch_size",
        "ca_train_pairs",
        "ca_eval_num_cells",
        "ca_steps",
        "ca_bernoulli_p",
        "ca_data_mode",
        "ca_num_workers",
        "ca_test_samples",
        "ca_test_seed",
    )
    missing = [name for name in required if source_values.get(name) is None]
    if missing:
        raise ValueError(
            "source manifest cannot reproduce the normal final evaluation; "
            "missing resolved fields: " + ", ".join(missing)
        )

    config_format = source_values["config_format"]
    if config_format not in config.registered_formats():
        raise ValueError(f"unsupported config format: {config_format!r}")
    parsed = config.parse_args_with_format(
        format=config_format,
        base_parser=argparse.ArgumentParser(add_help=False, allow_abbrev=False),
        args=[],
        namespace=argparse.Namespace(**source_values),
    )
    if not isinstance(parsed.dtype, torch.dtype):
        raise ValueError(f"unsupported dtype: {parsed.dtype!r}")

    model_args = argparse.Namespace(**source_values)
    model_args.dtype = parsed.dtype
    model_args.device = torch.device(device)
    return model_args, fallbacks


def validate_source_run(source_run_dir):
    source_run_dir = Path(source_run_dir).resolve()
    manifest_path = source_run_dir / MANIFEST_FILENAME
    manifest = read_validated_manifest(manifest_path)
    model = _require_mapping(manifest["model"], "manifest.model")
    resolved = _require_mapping(
        manifest["resolved_args"], "manifest.resolved_args"
    )
    model_name = model.get("model", resolved.get("model"))
    if model_name != "but_full_depth":
        raise ValueError(
            "standalone BUT evaluation requires model='but_full_depth'; "
            f"found {model_name!r}"
        )
    provenance = _require_mapping(manifest["provenance"], "manifest.provenance")
    checkpoint_dir_value = provenance.get("checkpoint_dir")
    if not isinstance(checkpoint_dir_value, str) or not checkpoint_dir_value:
        raise ValueError("manifest.provenance.checkpoint_dir must be a path")
    checkpoint_dir = Path(checkpoint_dir_value)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    return manifest, manifest_path, checkpoint_dir


def _read_checkpoint_metadata(checkpoint_dir, checkpoint_type):
    candidates = CHECKPOINT_METADATA_CANDIDATES[checkpoint_type]
    metadata_path = next(
        (checkpoint_dir / name for name in candidates if (checkpoint_dir / name).is_file()),
        None,
    )
    if metadata_path is None:
        raise ValueError(
            f"missing {checkpoint_type} metadata; tried: "
            + ", ".join(str(checkpoint_dir / name) for name in candidates)
        )
    metadata = _require_mapping(
        read_json(metadata_path), f"{checkpoint_type} metadata"
    )
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, Integral):
        raise ValueError(f"{metadata_path} must contain an integer step")
    checkpoint_name = metadata.get(
        "checkpoint", CHECKPOINT_FILE_DEFAULTS[metadata_path.name]
    )
    checkpoint_basename = Path(str(checkpoint_name))
    if (
        checkpoint_basename.name != str(checkpoint_name)
        or checkpoint_basename.suffix != ".pt"
    ):
        raise ValueError(
            f"{metadata_path} checkpoint must be a .pt basename, got "
            f"{checkpoint_name!r}"
        )
    checkpoint_path = checkpoint_dir / checkpoint_basename
    if not checkpoint_path.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
    return dict(metadata), checkpoint_path


def read_validated_training_stats(checkpoint_dir):
    """Read the source validation history needed for normal analyzer parity."""
    path = Path(checkpoint_dir) / TRAINING_STATS_FILENAME
    try:
        stats = _require_mapping(read_json(path), "source training statistics")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read source training statistics {path}: {error}") from error
    if not isinstance(stats.get("eval"), Mapping):
        raise ValueError(f"{path} must contain an eval mapping")
    if not isinstance(stats.get("train"), list):
        raise ValueError(f"{path} must contain a train list")
    return copy.deepcopy(dict(stats))


def load_checkpoint_weights(model, checkpoint_path, metadata, device):
    checkpoint = _require_mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        f"checkpoint {checkpoint_path}",
    )
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{checkpoint_path} has no model state dictionary")
    checkpoint_step = checkpoint.get("itr")
    if hasattr(checkpoint_step, "item"):
        checkpoint_step = checkpoint_step.item()
    if isinstance(checkpoint_step, bool) or not isinstance(
        checkpoint_step, Integral
    ):
        raise ValueError(f"{checkpoint_path} must contain an integer itr")
    if int(checkpoint_step) != int(metadata["step"]):
        raise ValueError(
            f"checkpoint step mismatch for {checkpoint_path}: metadata selects "
            f"{metadata['step']}, checkpoint contains {checkpoint_step}"
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()


def _autocast_context(model_args):
    if model_args.device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(
        device_type=model_args.device.type, dtype=model_args.dtype
    )


def build_evaluation_manifest(
    source_manifest,
    source_manifest_path,
    output_run_dir,
    artifact_dir,
    checkpoint_dir,
    compatibility_fallbacks=(),
):
    resolved_args = copy.deepcopy(_merge_manifest_sections(source_manifest))
    for fallback in compatibility_fallbacks:
        resolved_args[fallback["field"]] = copy.deepcopy(fallback["value"])
    resolved_args["repeat_cache_window"] = forward_policy_metadata(
        source_manifest
    )["repeat_cache_window"]
    resolved_args["ca_run_dir"] = str(Path(output_run_dir).resolve())
    source_annotations = source_manifest.get("annotations", {})
    source_tags = list(source_annotations.get("tags", []) or [])
    resolved_args["ca_tags"] = list(dict.fromkeys([*source_tags, "standalone_eval"]))
    resolved_args["ca_note"] = (
        "Standalone final-test evaluation of the source run's ID, strict "
        "extrapolation, and unconstrained extrapolation checkpoints."
    )
    manifest = build_run_manifest(
        resolved_args,
        output_run_dir,
        checkpoint_dir,
        status="running",
    )
    manifest["provenance"].update(
        {
            "evaluation_only": True,
            "source_run_id": source_manifest["run_id"],
            "source_run_status": source_manifest["status"],
            "source_manifest": str(Path(source_manifest_path).resolve()),
            "evaluation_artifact_dir": str(Path(artifact_dir).resolve()),
        }
    )
    manifest["standalone_evaluation"] = {
        "checkpoint_types": list(CHECKPOINT_TYPES),
        "attention_diagnostics": False,
        "source_run_id": source_manifest["run_id"],
    }
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError(
            "generated evaluation manifest is invalid: " + "; ".join(errors)
        )
    return manifest


def run_standalone_evaluation(args):
    manifest, manifest_path, checkpoint_dir = validate_source_run(
        args.source_run_dir
    )
    model_args, compatibility_fallbacks = resolve_source_args(
        manifest, args.device
    )
    stats = read_validated_training_stats(checkpoint_dir)
    checkpoint_specs = {
        checkpoint_type: _read_checkpoint_metadata(
            checkpoint_dir, checkpoint_type
        )
        for checkpoint_type in CHECKPOINT_TYPES
    }
    output_run_dir = args.output_run_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    if (output_run_dir / MANIFEST_FILENAME).exists() or (
        output_run_dir / METRICS_FILENAME
    ).exists():
        raise ValueError(
            f"output run directory already contains evaluation records: {output_run_dir}"
        )
    if artifact_dir.exists():
        raise ValueError(f"evaluation artifact directory already exists: {artifact_dir}")
    output_run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=False)

    output_manifest = build_evaluation_manifest(
        manifest,
        manifest_path,
        output_run_dir,
        artifact_dir,
        checkpoint_dir,
        compatibility_fallbacks,
    )
    write_run_manifest(output_run_dir, output_manifest)
    ensure_notes_file(output_run_dir)

    try:
        model = models.make_model_from_args(model_args)
        trained_pairs = tuple(
            tuple(int(value) for value in pair)
            for pair in (model_args.ca_train_pairs or ())
        )
        supports_repeat_override = "num_repeats" in inspect.signature(
            model.forward
        ).parameters
        if trained_pairs:
            trained_ca_steps, trained_repeats = tuple(model_args.ca_best_pair)
        else:
            trained_ca_steps = int(model_args.ca_steps)
            trained_repeats = (
                int(model_args.n_repeat) if supports_repeat_override else None
            )

        test_loaders = make_ca_fixed_loaders(
            model_args,
            base_seed=model_args.ca_test_seed,
            num_samples=model_args.ca_test_samples,
        )
        policy_metadata = forward_policy_metadata(manifest)
        forward_context = CAForwardContext(
            model,
            CAForwardPolicy(
                repeat_cache_window=policy_metadata["repeat_cache_window"]
            ),
        )
        analyses = {}
        checkpoint_sources = {}
        for checkpoint_type in CHECKPOINT_TYPES:
            metadata, checkpoint_path = checkpoint_specs[checkpoint_type]
            load_checkpoint_weights(
                model, checkpoint_path, metadata, model_args.device
            )
            analyses[checkpoint_type] = evaluate_loaded_ca_checkpoint(
                forward_context,
                test_loaders,
                model_args.device,
                checkpoint_metadata=metadata,
                label=checkpoint_type,
                split_seed=model_args.ca_test_seed,
                samples_per_length=model_args.ca_test_samples,
                trained_ca_steps=trained_ca_steps,
                trained_num_repeats=trained_repeats,
                trained_pairs=trained_pairs,
                internal_pairs=model_args.ca_final_eval_pairs,
                external_ca_steps=model_args.ca_final_external_steps,
                final_eval_max_batches=getattr(
                    model_args, "ca_final_eval_max_batches", None
                ),
                repeat_diagnostic_max_repeats=getattr(
                    model_args, "ca_repeat_diagnostic_max_repeats", None
                ),
                repeat_diagnostic_horizons=getattr(
                    model_args, "ca_repeat_diagnostic_horizons", None
                ),
                repeat_diagnostic_max_batches=getattr(
                    model_args, "ca_repeat_diagnostic_max_batches", None
                ),
                eval_max_batches=getattr(model_args, "ca_eval_max_batches", None),
                repeat_diagnostic_examples=model_args.ca_repeat_diagnostic_examples,
                forward_policy_metadata=policy_metadata,
                ctx=_autocast_context(model_args),
            )
            checkpoint_sources[checkpoint_type] = str(checkpoint_path)

        stats["best_id"] = analyses["best_id"]["checkpoint"]
        stats["best_extrapolation_strict"] = analyses[
            "best_extrapolation_strict"
        ]["checkpoint"]
        stats["best_extrapolation_unconstrained"] = analyses[
            "best_extrapolation_unconstrained"
        ]["checkpoint"]
        stats["best"] = stats["best_id"]
        stats["best_extrapolation"] = stats["best_extrapolation_strict"]
        stats["final_eval"] = analyses["best_id"]["task_metrics"]
        stats["checkpoint_analysis"] = analyses
        stats["standalone_evaluation"] = {
            "source_run_id": manifest["run_id"],
            "source_manifest": str(manifest_path),
            "checkpoint_sources": checkpoint_sources,
            "compatibility_fallbacks": compatibility_fallbacks,
        }
        summary_path = artifact_dir / "summary.json"
        write_json(summary_path, stats)
        write_eval_metrics(output_run_dir, stats, manifest=output_manifest)
        completed_manifest = read_json(output_run_dir / MANIFEST_FILENAME)
        completed_manifest["provenance"]["evaluation_summary"] = str(summary_path)
        completed_manifest["timestamps"]["updated_at"] = utc_now()
        write_run_manifest(output_run_dir, completed_manifest)
        update_run_manifest_status(output_run_dir, "completed")
        return {
            "source_run_id": manifest["run_id"],
            "output_run_dir": str(output_run_dir),
            "artifact_dir": str(artifact_dir),
            "checkpoint_types": list(analyses),
            "compatibility_fallbacks": compatibility_fallbacks,
        }
    except BaseException as error:
        update_run_manifest_status(output_run_dir, "failed", error=error)
        raise


def main(argv=None):
    result = run_standalone_evaluation(parse_args(argv))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
