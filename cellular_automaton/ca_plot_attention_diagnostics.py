"""Plot a captured cellular-automaton attention diagnostic shard."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import torch

from cellular_automaton.ca_attention_diagnostics import (
    FACTORIAL_EFFECT_CONDITIONS,
    reduce_factorial_diagnostic_values,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--condition", default="manual_full")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overlap-artifact",
        type=Path,
        help="Older artifact whose shared repeat prefix must be compared.",
    )
    parser.add_argument("--relative-age-bins", type=int, default=20)
    parser.add_argument(
        "--selected-depths",
        type=int,
        nargs="+",
        help="Repeat depths used for CDF and log-sum-exp profile lines.",
    )
    parser.add_argument(
        "--factorial-source-depth",
        type=int,
        help="Factorial intervention source depth; inferred when unique.",
    )
    parser.add_argument(
        "--cosine-example-count",
        type=int,
        default=2,
        help="Number of leading captured examples used for factorial cosine plots.",
    )
    return parser.parse_args(argv)


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _find_shard(manifest, condition, batch_index, artifact_dir):
    matches = [
        shard
        for shard in manifest.get("shards", ())
        if shard.get("condition_id") == condition
        and shard.get("batch_index") == batch_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one shard for condition={condition!r}, "
            f"batch_index={batch_index}; found {len(matches)}"
        )
    path = artifact_dir / matches[0]["path"]
    if not path.is_file():
        raise ValueError(f"diagnostic shard does not exist: {path}")
    return path


_FACTORIAL_CONDITIONS = (
    "free_baseline", "clean_baseline", "free_reset", "clean_reset"
)

_FACTORIAL_DIAGNOSTIC_GROUPS = {
    "qkv_logits": (
        "query_norm", "current_key_norm", "visible_history_key_norm",
        "current_value_norm", "visible_history_value_norm", "logit_std",
        "logit_spread", "top_two_logit_gap",
    ),
    "attention": (
        "normalised_repeat_entropy", "normalised_attention_entropy",
        "effective_support_fraction", "maximum_attention_probability",
        "current_repeat_attention_mass", "visible_history_attention_mass",
        "current_block_contribution_norm",
        "visible_history_block_contribution_norm_sum",
        "current_minus_visible_history_logsumexp",
    ),
}


def _curve_statistics(values):
    values = values.float()
    result = {}
    for name, curve in (
        ("mean", torch.nanmean(values, dim=0)),
        ("p10", torch.nanquantile(values, 0.1, dim=0)),
        ("p90", torch.nanquantile(values, 0.9, dim=0)),
    ):
        result[name] = [float(value) if torch.isfinite(value) else None for value in curve]
    return result


def summarize_factorial_diagnostics(shard):
    source, maximum = int(shard["source_depth"]), int(shard["max_repeats"])
    prefix, conditions = shard["shared_prefix"], shard["conditions"]
    prefix_depths, post_depths = list(range(1, source + 1)), list(
        range(source + 1, maximum + 1)
    )
    prefix_values = reduce_factorial_diagnostic_values(
        {
            "attention_records": prefix["attention_records"],
            "repeat_states": prefix["repeat_states"][:, 1:],
        },
        prefix_depths,
    ) if prefix_depths else None
    raw = {}
    for condition_id in _FACTORIAL_CONDITIONS:
        post = reduce_factorial_diagnostic_values(
            conditions[condition_id], post_depths
        )
        raw[condition_id] = {
            name: torch.cat((prefix_values[name], values), dim=1)
            if prefix_values is not None else values
            for name, values in post.items()
        }
        initial_norm = torch.linalg.vector_norm(
            prefix["repeat_states"][:, 0].float(), dim=-1
        ).mean(dim=-1, keepdim=True)
        raw[condition_id]["hidden_state_norm"] = torch.cat(
            (initial_norm, raw[condition_id]["hidden_state_norm"]), dim=1
        )
    effects = {
        effect_id: {
            name: raw[left][name] - raw[right][name] for name in raw[left]
        }
        for effect_id, (left, right) in FACTORIAL_EFFECT_CONDITIONS.items()
    }
    effects["state_history_interaction"] = {
        name: effects["clean_state_after_reset"][name]
        - effects["clean_state_with_history"][name]
        for name in effects["clean_state_with_history"]
    }
    for values in effects.values():
        for name, curve in values.items():
            prefix_count = source + 1 if name == "hidden_state_norm" else source
            finite_prefix = curve[:, :prefix_count][torch.isfinite(curve[:, :prefix_count])]
            if finite_prefix.numel() and not torch.equal(
                finite_prefix, torch.zeros_like(finite_prefix)
            ):
                raise ValueError("factorial diagnostic prefix effect is not zero")
    distances = {}
    for effect_id, (left, right) in FACTORIAL_EFFECT_CONDITIONS.items():
        left_states = torch.cat((prefix["repeat_states"], conditions[left]["repeat_states"]), 1)
        right_states = torch.cat((prefix["repeat_states"], conditions[right]["repeat_states"]), 1)
        distances[effect_id] = torch.linalg.vector_norm(
            left_states.float() - right_states.float(), dim=-1
        ).mean(dim=-1)
    statistics = lambda rows: {
        row_id: {name: _curve_statistics(values) for name, values in metrics.items()}
        for row_id, metrics in rows.items()
    }
    return {
        "source_depth": source,
        "diagnostic_depths": list(range(1, maximum + 1)),
        "state_depths": list(range(maximum + 1)),
        "conditions": statistics(raw),
        "effects": statistics(effects),
        "state_distances": {
            effect_id: _curve_statistics(values) for effect_id, values in distances.items()
        },
    }


def summarize_factorial_state_cosines(shard, example_count=2):
    if shard.get("factorial_capture_schema_version") != 2:
        raise ValueError(
            "full-trajectory cosine plots require factorial schema version 2"
        )
    if example_count <= 0:
        raise ValueError("cosine example count must be positive")
    prefix, conditions = shard.get("shared_prefix"), shard.get("conditions")
    if not isinstance(prefix, dict) or not isinstance(conditions, dict):
        raise ValueError("factorial shard is missing prefix or conditions")
    if set(conditions) != set(_FACTORIAL_CONDITIONS):
        raise ValueError("factorial shard does not contain exactly four conditions")
    source_depth, max_repeats = int(shard["source_depth"]), int(shard["max_repeats"])
    prefix_states, example_ids = prefix["repeat_states"], shard["example_ids"]
    if prefix.get("depths") != list(range(source_depth + 1)):
        raise ValueError("factorial shared-prefix depths are invalid")
    expected_prefix = (prefix_states.shape[0], source_depth + 1)
    if prefix_states.ndim != 4 or prefix_states.shape[:2] != expected_prefix:
        raise ValueError("factorial shared-prefix states are invalid")

    examples = []
    for example_index in range(min(example_count, prefix_states.shape[0])):
        matrices = {}
        for condition_id in _FACTORIAL_CONDITIONS:
            suffix = conditions[condition_id]["repeat_states"]
            expected_suffix = (
                prefix_states.shape[0], max_repeats - source_depth,
                *prefix_states.shape[2:]
            )
            if tuple(suffix.shape) != expected_suffix:
                raise ValueError(f"factorial states are invalid for {condition_id}")
            trajectory = torch.cat(
                (prefix_states[example_index], suffix[example_index]), dim=0
            ).float().reshape(max_repeats + 1, -1)
            norms = torch.linalg.vector_norm(trajectory, dim=1)
            if (norms <= 1e-12).any():
                raise ValueError(f"cosine trajectory is invalid for {condition_id}")
            normalized = trajectory / norms[:, None]
            cosine = normalized @ normalized.T
            if (
                not torch.isfinite(cosine).all()
                or not torch.allclose(cosine, cosine.T, atol=1e-5)
                or not torch.allclose(cosine.diag(), torch.ones_like(norms), atol=1e-5)
            ):
                raise ValueError(f"cosine matrix is invalid for {condition_id}")
            matrices[condition_id] = cosine.tolist()
            del trajectory, normalized, cosine
        examples.append({
            "example_index": example_index,
            "example_id": int(example_ids[example_index]),
            "conditions": matrices,
        })
    return {
        "source_depth": source_depth,
        "max_repeats": max_repeats,
        "depths": list(range(max_repeats + 1)),
        "requested_example_count": example_count,
        "selected_example_count": len(examples),
        "examples": examples,
    }


def _plot_factorial_state_cosines(summary, example, output_path, title):
    figure, axes = plt.subplots(
        1, 4, figsize=(19, 4.8), sharex=True, sharey=True,
        constrained_layout=True,
    )
    depths = summary["depths"]
    step = max(1, len(depths) // 8)
    ticks = list(range(0, len(depths), step))
    image = None
    for axis, condition_id in zip(axes, _FACTORIAL_CONDITIONS):
        matrix = torch.tensor(example["conditions"][condition_id])
        image = axis.imshow(
            matrix.numpy(), origin="lower", vmin=-1, vmax=1,
            aspect="equal", cmap="coolwarm",
        )
        boundary = summary["source_depth"] + 0.5
        axis.axvline(boundary, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(boundary, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(ticks, [depths[index] for index in ticks])
        axis.set_yticks(ticks, [depths[index] for index in ticks])
        axis.set(title=condition_id, xlabel="State depth")
    axes[0].set_ylabel("State depth")
    figure.colorbar(image, ax=axes, label="Whole-row state cosine similarity")
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


_FACTORIAL_OUTCOME_METRICS = (
    ("cell_accuracy", "Cell accuracy"),
    ("exact_sequence_accuracy", "Exact-sequence accuracy"),
    ("loss", "Cross-entropy loss"),
)


def _validate_factorial_outcomes(evaluation):
    depths = evaluation.get("target_depths")
    if not isinstance(depths, list) or not depths:
        raise ValueError("factorial outcome target depths are missing")
    expected_depths = {str(depth) for depth in depths}
    rows = list(evaluation.get("conditions", {}).values())
    rows += list(evaluation.get("paired_effects", {}).values())
    rows.append(evaluation.get("state_history_interaction", {}))
    for row in rows:
        by_depth = row.get("by_target_depth", {})
        if set(by_depth) != expected_depths:
            raise ValueError("factorial outcome depth keys are inconsistent")
        if any(
            set(by_depth[str(depth)])
            != {metric for metric, _ in _FACTORIAL_OUTCOME_METRICS}
            for depth in depths
        ):
            raise ValueError("factorial outcome metrics are inconsistent")
    return depths


def _plot_factorial_outcomes(evaluation, output_dir, title):
    depths = _validate_factorial_outcomes(evaluation)
    specifications = (
        (evaluation["conditions"], "Full-split outcomes", "outcomes"),
        ({
            **evaluation["paired_effects"],
            "state_history_interaction": evaluation["state_history_interaction"],
        }, "Paired effects (left minus right)", "outcome_effects"),
    )
    for rows, subtitle, filename in specifications:
        figure, axes = plt.subplots(
            1, 3, figsize=(15, 4.5), constrained_layout=True
        )
        for axis, (metric, label) in zip(axes, _FACTORIAL_OUTCOME_METRICS):
            for row_id, row in rows.items():
                axis.plot(
                    depths,
                    [row["by_target_depth"][str(depth)][metric] for depth in depths],
                    marker="o",
                    label=row_id,
                )
            axis.set(xlabel="Target depth", ylabel=label, title=label)
            if filename == "outcome_effects":
                axis.axhline(0, color="black", linewidth=0.8)
        axes[0].legend(fontsize=8)
        figure.suptitle(f"{title} · {subtitle}")
        figure.savefig(output_dir / f"factorial_{filename}.png", dpi=160)
        plt.close(figure)


def _plot_factorial_diagnostic_group(
    summary, rows, metrics, output_path, title
):
    columns = 3
    figure, axes = plt.subplots(
        math.ceil(len(metrics) / columns), columns,
        figsize=(15, 4 * math.ceil(len(metrics) / columns)),
        constrained_layout=True,
    )
    axes = list(axes.flat)
    for axis, metric in zip(axes, metrics):
        depths = (
            summary["state_depths"]
            if metric in ("hidden_state_norm", "hidden_state_distance")
            else summary["diagnostic_depths"]
        )
        for row_id, row in rows.items():
            if metric not in row:
                continue
            values = row[metric]
            curves = {
                name: [float("nan") if value is None else value for value in curve]
                for name, curve in values.items()
            }
            axis.fill_between(depths, curves["p10"], curves["p90"], alpha=0.08)
            axis.plot(depths, curves["mean"], label=row_id)
        axis.set(xlabel="State/repeat depth", title=metric.replace("_", " "))
        axis.axvline(
            summary["source_depth"] + 0.5,
            color="black", linestyle="--", linewidth=0.7,
        )
    for axis in axes[len(metrics):]:
        axis.set_visible(False)
    axes[0].legend(fontsize=7)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_factorial_diagnostics(summary, output_dir, title):
    for group, metrics in _FACTORIAL_DIAGNOSTIC_GROUPS.items():
        for row_kind in ("conditions", "effects"):
            _plot_factorial_diagnostic_group(
                summary, summary[row_kind], metrics,
                output_dir / f"factorial_{group}_{row_kind}.png",
                f"{title} · {group} · {row_kind}",
            )
    state_effects = {
        effect_id: {
            "hidden_state_norm": summary["effects"][effect_id]["hidden_state_norm"],
            **({"hidden_state_distance": summary["state_distances"][effect_id]}
               if effect_id in summary["state_distances"] else {}),
        }
        for effect_id in summary["effects"]
    }
    _plot_factorial_diagnostic_group(
        summary, summary["conditions"], ("hidden_state_norm",),
        output_dir / "factorial_state_conditions.png", f"{title} · state conditions",
    )
    _plot_factorial_diagnostic_group(
        summary, state_effects, ("hidden_state_norm", "hidden_state_distance"),
        output_dir / "factorial_state_effects.png", f"{title} · state effects",
    )


def _validate_unit_interval(name, value, tolerance=1e-5):
    value = torch.as_tensor(value).float()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if value.min() < -tolerance or value.max() > 1.0 + tolerance:
        raise ValueError(f"{name} lies outside [0, 1]")


def summarize_shard(shard, relative_age_bins=20, selected_depths=None):
    if relative_age_bins <= 0:
        raise ValueError("relative age bins must be positive")
    logits = shard["decoded_logits_by_repeat"]
    targets = shard["targets_by_horizon"]
    if logits.ndim != 4 or targets.ndim != 3:
        raise ValueError("logits or targets have an unexpected rank")
    if logits.shape[:3] != targets.shape:
        raise ValueError("logits and targets do not align")

    predictions = logits.argmax(dim=-1)
    correct = predictions[:, :, None, :].eq(targets[:, None, :, :])
    cell_accuracy = correct.float().mean(dim=(0, 3))
    sequence_accuracy = correct.all(dim=-1).float().mean(dim=0)
    _validate_unit_interval("cell accuracy", cell_accuracy)
    _validate_unit_interval("sequence accuracy", sequence_accuracy)

    records = sorted(
        shard["attention_records"], key=lambda record: record["repeat_index"]
    )
    repeats = [int(record["repeat_index"]) for record in records]
    if repeats != list(range(1, len(records) + 1)):
        raise ValueError("attention records must cover consecutive repeats from one")
    if logits.shape[1] != len(records) + 1:
        raise ValueError("repeat-state logits do not align with attention records")
    if selected_depths is None:
        selected_depths = sorted(set(
            max(2, round(value))
            for value in torch.linspace(2, len(records), min(7, len(records))).tolist()
        ))
    else:
        selected_depths = sorted(set(int(depth) for depth in selected_depths))
    if not selected_depths or selected_depths[0] < 2 or selected_depths[-1] > len(records):
        raise ValueError("selected depths must lie within 2..maximum repeats")

    curves = []
    repeat_age_mass = []
    source_repeat_mass = []
    relative_age_mass = []
    relative_age_populated = []
    block_contribution_norm_by_age = []
    attention_mass_cdfs = {}
    attention_window_widths = []
    fixed_attention_windows = []
    repeat_logsumexp_margin_by_age = []
    repeat_logsumexp_profiles = {}
    for record in records:
        repeat_entropy = record["normalised_repeat_entropy"].float()
        token_entropy = record["normalised_attention_entropy"].float()
        maximum_probability = record["maximum_attention_probability"].float()
        support_fraction = record["effective_support_fraction"].float()
        repeat_mass = record["repeat_mass"].float()
        repeat_logsumexp = record["repeat_logsumexp"].float()
        repeat_logsumexp_valid = record.get("repeat_logsumexp_valid")
        if repeat_logsumexp_valid is None:
            repeat_logsumexp_valid = torch.ones_like(
                repeat_logsumexp, dtype=torch.bool
            )
        else:
            repeat_logsumexp_valid = repeat_logsumexp_valid.bool()
        if repeat_logsumexp_valid.shape != repeat_logsumexp.shape:
            raise ValueError(
                "repeat log-sum-exp values and validity mask do not align"
            )
        block_contribution_norm = record["block_contribution_norm"].float()
        for name, value in (
            ("normalised repeat entropy", repeat_entropy),
            ("normalised token entropy", token_entropy),
            ("maximum attention probability", maximum_probability),
            ("effective support fraction", support_fraction),
            ("repeat mass", repeat_mass),
        ):
            _validate_unit_interval(name, value)
        if not torch.allclose(
            repeat_mass.sum(dim=-1),
            torch.ones_like(repeat_mass[..., 0]),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError("repeat mass does not sum to one")
        if (
            not torch.isfinite(block_contribution_norm).all()
            or block_contribution_norm.min() < 0
        ):
            raise ValueError("block contribution norm must be finite and nonnegative")
        if not torch.isfinite(repeat_logsumexp).all():
            raise ValueError("repeat log-sum-exp contains non-finite values")
        masked_repeat_logsumexp = repeat_logsumexp.masked_fill(
            ~repeat_logsumexp_valid, float("-inf")
        )
        if not repeat_logsumexp_valid.any(dim=-1).all():
            raise ValueError("a query has no valid repeat log-sum-exp block")
        reconstructed_mass = torch.softmax(masked_repeat_logsumexp, dim=-1)
        if not torch.allclose(reconstructed_mass, repeat_mass, rtol=1e-4, atol=1e-5):
            raise ValueError("block log-sum-exp does not reconstruct repeat mass")

        repeat_example_mean = repeat_entropy.mean(dim=(1, 2))
        token_example_mean = token_entropy.mean(dim=(1, 2))
        age_mass = repeat_mass.mean(dim=(0, 1, 2)).flip(0)
        repeat_mass_by_age = repeat_mass.flip(-1)
        repeat_logsumexp_by_age = masked_repeat_logsumexp.flip(-1)
        repeat_logsumexp_valid_by_age = repeat_logsumexp_valid.flip(-1)
        logsumexp_margin_valid = (
            repeat_logsumexp_valid_by_age
            & repeat_logsumexp_valid_by_age[..., :1]
        )
        raw_logsumexp_margin = (
            repeat_logsumexp_by_age - repeat_logsumexp_by_age[..., :1]
        )
        logsumexp_margin = torch.where(
            logsumexp_margin_valid,
            raw_logsumexp_margin,
            torch.zeros_like(raw_logsumexp_margin),
        )
        if not torch.allclose(
            logsumexp_margin[..., 0],
            torch.zeros_like(logsumexp_margin[..., 0]),
        ):
            raise ValueError("current-block log-sum-exp margin is not zero")
        margin_count = logsumexp_margin_valid.sum(dim=(0, 1, 2))
        mean_logsumexp_margin = logsumexp_margin.sum(
            dim=(0, 1, 2)
        ) / margin_count.clamp_min(1)
        mean_logsumexp_margin_values = [
            float(value) if int(count) > 0 else None
            for value, count in zip(mean_logsumexp_margin, margin_count)
        ]
        repeat_logsumexp_margin_by_age.append(
            mean_logsumexp_margin_values
        )
        repeat_age_mass.append(age_mass.tolist())
        source_repeat_mass.append(age_mass.flip(0).tolist())
        if age_mass.numel() == 1:
            relative_age_mass.append(None)
            relative_age_populated.append(None)
        else:
            relative_ages = torch.arange(age_mass.numel()) / (age_mass.numel() - 1)
            bin_indices = torch.floor(relative_ages * relative_age_bins).long()
            bin_indices.clamp_max_(relative_age_bins - 1)
            binned_mass = torch.zeros(relative_age_bins)
            binned_mass.scatter_add_(0, bin_indices, age_mass)
            populated = torch.zeros(relative_age_bins, dtype=torch.bool)
            populated[bin_indices] = True
            if not torch.allclose(binned_mass.sum(), torch.tensor(1.0), atol=1e-5):
                raise ValueError("relative-age mass does not sum to one")
            relative_age_mass.append(binned_mass.tolist())
            relative_age_populated.append(populated.tolist())
        age_contribution = block_contribution_norm.mean(dim=(0, 1, 2)).flip(0)
        block_contribution_norm_by_age.append(age_contribution.tolist())
        depth = int(record["repeat_index"])
        cumulative_mass = repeat_mass_by_age.cumsum(dim=-1)
        width_summary = {"repeat": depth, "thresholds": {}}
        for threshold in (0.5, 0.8, 0.9):
            widths = (cumulative_mass < threshold).sum(dim=-1).add(1).float()
            widths.clamp_max_(depth)
            example_width = widths.mean(dim=(1, 2))
            normalized_example_width = example_width / depth
            width_summary["thresholds"][str(threshold)] = {
                "mean_blocks": float(widths.mean()),
                "example_p10_blocks": float(example_width.quantile(0.10)),
                "example_p90_blocks": float(example_width.quantile(0.90)),
                "mean_fraction": float(widths.mean() / depth),
                "example_p10_fraction": float(
                    normalized_example_width.quantile(0.10)
                ),
                "example_p90_fraction": float(
                    normalized_example_width.quantile(0.90)
                ),
            }
        attention_window_widths.append(width_summary)

        absolute_recent = {
            str(window): float(
                repeat_mass_by_age[..., : min(window, depth)].sum(dim=-1).mean()
            )
            for window in (1, 3, 6, 12)
        }
        fractional_recent = {}
        for fraction in (0.2, 0.4, 0.6):
            window = math.ceil(fraction * depth)
            fractional_recent[str(fraction)] = {
                "blocks": window,
                "mass": float(
                    repeat_mass_by_age[..., :window].sum(dim=-1).mean()
                ),
            }
        fixed_attention_windows.append({
            "repeat": depth,
            "absolute_recent_mass": absolute_recent,
            "fractional_recent_mass": fractional_recent,
            "oldest_one_mass": float(repeat_mass_by_age[..., -1].mean()),
            "oldest_three_mass": float(
                repeat_mass_by_age[..., -min(3, depth):].sum(dim=-1).mean()
            ),
        })

        if depth in selected_depths:
            relative_ages = torch.arange(depth).float() / (depth - 1)
            attention_mass_cdfs[str(depth)] = {
                "relative_ages": relative_ages.tolist(),
                "cumulative_mass": age_mass.cumsum(dim=0).tolist(),
            }
            repeat_logsumexp_profiles[str(depth)] = {
                "relative_ages": relative_ages.tolist(),
                "mean_margin_to_current": mean_logsumexp_margin_values,
            }
        curves.append(
            {
                "repeat": int(record["repeat_index"]),
                "matching_cell_accuracy": float(
                    cell_accuracy[record["repeat_index"], record["repeat_index"]]
                ),
                "matching_sequence_accuracy": float(
                    sequence_accuracy[
                        record["repeat_index"], record["repeat_index"]
                    ]
                ),
                "normalised_repeat_entropy_mean": float(repeat_entropy.mean()),
                "normalised_repeat_entropy_head_means": repeat_entropy.mean(
                    dim=(0, 2)
                ).tolist(),
                "normalised_repeat_entropy_example_p10": float(
                    repeat_example_mean.quantile(0.10)
                ),
                "normalised_repeat_entropy_example_p90": float(
                    repeat_example_mean.quantile(0.90)
                ),
                "normalised_token_entropy_mean": float(token_entropy.mean()),
                "normalised_token_entropy_head_means": token_entropy.mean(
                    dim=(0, 2)
                ).tolist(),
                "normalised_token_entropy_example_p10": float(
                    token_example_mean.quantile(0.10)
                ),
                "normalised_token_entropy_example_p90": float(
                    token_example_mean.quantile(0.90)
                ),
                "maximum_attention_probability_mean": float(
                    maximum_probability.mean()
                ),
                "effective_support_fraction_mean": float(support_fraction.mean()),
                "current_repeat_mass_mean": float(repeat_mass[..., -1].mean()),
            }
        )

    return {
        "batch_index": int(shard["batch_index"]),
        "examples": int(logits.shape[0]),
        "cells": int(logits.shape[2]),
        "repeats": repeats,
        "cell_accuracy": cell_accuracy.tolist(),
        "sequence_accuracy": sequence_accuracy.tolist(),
        "curves": curves,
        "repeat_age_mass": repeat_age_mass,
        "source_repeat_mass": source_repeat_mass,
        "relative_age_bin_edges": torch.linspace(
            0, 1, relative_age_bins + 1
        ).tolist(),
        "relative_age_mass": relative_age_mass,
        "relative_age_populated": relative_age_populated,
        "block_contribution_norm_by_age": block_contribution_norm_by_age,
        "selected_depths": selected_depths,
        "attention_mass_cdfs": attention_mass_cdfs,
        "attention_window_widths": attention_window_widths,
        "fixed_attention_windows": fixed_attention_windows,
        "repeat_logsumexp_margin_by_age": repeat_logsumexp_margin_by_age,
        "repeat_logsumexp_profiles": repeat_logsumexp_profiles,
    }


def _plot_accuracy(summary, output_path, title):
    repeats = summary["repeats"]
    curves = summary["curves"]
    cell_matrix = torch.tensor(summary["cell_accuracy"])[1:, 1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(repeats, [row["matching_cell_accuracy"] for row in curves],
                 marker="o", label="Cell accuracy")
    axes[0].plot(repeats, [row["matching_sequence_accuracy"] for row in curves],
                 marker="o", label="Exact sequence accuracy")
    axes[0].set(xlabel="Model repeats = target horizon", ylabel="Accuracy",
                ylim=(-0.02, 1.02))
    axes[0].legend()
    image = axes[1].imshow(cell_matrix, origin="lower", vmin=0, vmax=1,
                           aspect="auto", cmap="viridis")
    axes[1].set(xlabel="Target Rule 30 horizon", ylabel="Model repeats")
    ticks = list(range(0, len(repeats), max(1, len(repeats) // 10)))
    axes[1].set_xticks(ticks, [repeats[index] for index in ticks])
    axes[1].set_yticks(ticks, [repeats[index] for index in ticks])
    fig.colorbar(image, ax=axes[1], label="Cell accuracy")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_entropy(summary, output_path, title):
    repeats = summary["repeats"]
    curves = summary["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True,
                             constrained_layout=True)
    specifications = (
        ("repeat", "normalised_repeat_entropy", "Entropy across repeat blocks"),
        ("token", "normalised_token_entropy", "Entropy across cached tokens"),
    )
    for axis, (_, key, label) in zip(axes, specifications):
        head_values = torch.tensor([row[f"{key}_head_means"] for row in curves])
        for head in range(head_values.shape[1]):
            axis.plot(repeats, head_values[:, head], linewidth=0.9, alpha=0.55,
                      label=f"Head {head}")
        mean = [row[f"{key}_mean"] for row in curves]
        lower = [row[f"{key}_example_p10"] for row in curves]
        upper = [row[f"{key}_example_p90"] for row in curves]
        axis.fill_between(repeats, lower, upper, color="black", alpha=0.12,
                          label="Example p10–p90")
        axis.plot(repeats, mean, color="black", linewidth=2.2, label="Mean")
        axis.set(xlabel="Model repeat", ylabel="Normalized entropy",
                 title=label, ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_concentration(summary, output_path, title):
    repeats = summary["repeats"]
    curves = summary["curves"]
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for key, label in (
        ("maximum_attention_probability_mean", "Maximum token probability"),
        ("effective_support_fraction_mean", "Effective support fraction"),
        ("current_repeat_mass_mean", "Current-repeat mass"),
    ):
        axis.plot(repeats, [row[key] for row in curves], marker="o", label=label)
    axis.set(xlabel="Model repeat", ylabel="Mean value", ylim=(-0.02, 1.02),
             title=title)
    axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_repeat_age_mass(summary, output_path, title, *, log_scale=False):
    repeats = summary["repeats"]
    matrix = torch.full((len(repeats), len(repeats)), float("nan"))
    for row, values in enumerate(summary["repeat_age_mass"]):
        matrix[row, : len(values)] = torch.tensor(values)
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    colormap = matplotlib.colormaps["magma"].copy()
    colormap.set_bad("#e6e6e6")
    image_options = (
        {"norm": LogNorm(vmin=1e-3, vmax=1, clip=True)}
        if log_scale
        else {"vmin": 0, "vmax": 1}
    )
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=colormap,
                        **image_options)
    ticks = list(range(0, len(repeats), max(1, len(repeats) // 10)))
    axis.set_xticks(ticks, ticks)
    axis.set_yticks(ticks, [repeats[index] for index in ticks])
    axis.set(xlabel="Source age (0 = current repeat)", ylabel="Model repeat",
             title=title)
    colorbar_label = (
        "Mean attention mass (log scale; values ≤0.1% share lowest color)"
        if log_scale
        else "Mean attention mass"
    )
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_age_mass_vs_contribution(summary, output_path, title):
    repeats = summary["repeats"]
    mass_matrix = torch.full((len(repeats), len(repeats)), float("nan"))
    contribution_matrix = torch.full_like(mass_matrix, float("nan"))
    for row, (mass, contribution) in enumerate(zip(
        summary["repeat_age_mass"],
        summary["block_contribution_norm_by_age"],
    )):
        if len(mass) != len(contribution):
            raise ValueError("mass and contribution ages do not align")
        mass_matrix[row, : len(mass)] = torch.tensor(mass)
        contribution_matrix[row, : len(contribution)] = torch.tensor(contribution)

    positive_contributions = contribution_matrix[
        torch.isfinite(contribution_matrix) & (contribution_matrix > 0)
    ]
    if positive_contributions.numel() == 0:
        raise ValueError("block contribution matrix has no positive values")
    contribution_max = float(positive_contributions.max())
    contribution_min = min(1e-3, contribution_max / 1000)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colormap = matplotlib.colormaps["magma"].copy()
    colormap.set_bad("#e6e6e6")
    images = (
        axes[0].imshow(
            mass_matrix,
            origin="lower",
            aspect="auto",
            cmap=colormap,
            norm=LogNorm(vmin=1e-3, vmax=1, clip=True),
        ),
        axes[1].imshow(
            contribution_matrix,
            origin="lower",
            aspect="auto",
            cmap=colormap,
            norm=LogNorm(
                vmin=contribution_min,
                vmax=contribution_max,
                clip=True,
            ),
        ),
    )
    ticks = list(range(0, len(repeats), max(1, len(repeats) // 10)))
    for axis in axes:
        axis.set_xticks(ticks, ticks)
        axis.set_yticks(ticks, [repeats[index] for index in ticks])
        axis.set(xlabel="Source age (0 = current repeat)", ylabel="Model repeat")
    axes[0].set_title("Attention mass")
    axes[1].set_title("Weighted-value block contribution norm")
    figure.colorbar(images[0], ax=axes[0], label="Mean attention mass (log scale)")
    figure.colorbar(
        images[1],
        ax=axes[1],
        label="Mean block-contribution L2 norm (log scale)",
    )
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_repeat_mass_coordinates(summary, output_path, title):
    repeats = summary["repeats"]
    size = len(repeats)
    age_matrix = torch.full((size, size), float("nan"))
    source_matrix = torch.full_like(age_matrix, float("nan"))
    relative_matrix = torch.full(
        (size, len(summary["relative_age_bin_edges"]) - 1), float("nan")
    )
    for row, (age_mass, source_mass, relative_mass, relative_populated) in enumerate(zip(
        summary["repeat_age_mass"],
        summary["source_repeat_mass"],
        summary["relative_age_mass"],
        summary["relative_age_populated"],
    )):
        age_matrix[row, : len(age_mass)] = torch.tensor(age_mass)
        source_matrix[row, : len(source_mass)] = torch.tensor(source_mass)
        if relative_mass is not None:
            relative_matrix[row] = torch.tensor(relative_mass)
            relative_matrix[row, ~torch.tensor(relative_populated)] = float("nan")

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    colormap = matplotlib.colormaps["magma"].copy()
    colormap.set_bad("#e6e6e6")
    norm = LogNorm(vmin=1e-3, vmax=1, clip=True)
    images = (
        axes[0].imshow(age_matrix, origin="lower", aspect="auto",
                       cmap=colormap, norm=norm),
        axes[1].imshow(source_matrix, origin="lower", aspect="auto",
                       cmap=colormap, norm=norm),
        axes[2].imshow(
            relative_matrix,
            origin="lower",
            aspect="auto",
            extent=(0, 1, 0.5, size + 0.5),
            cmap=colormap,
            norm=norm,
        ),
    )
    ticks = list(range(0, size, max(1, size // 10)))
    axes[0].set_xticks(ticks, ticks)
    axes[1].set_xticks(ticks, [index + 1 for index in ticks])
    for axis in axes[:2]:
        axis.set_yticks(ticks, [repeats[index] for index in ticks])
    axes[2].set_yticks(
        [repeats[index] for index in ticks],
        [repeats[index] for index in ticks],
    )
    axes[0].set(
        xlabel="Absolute source age (0 = current)",
        ylabel="Model repeat",
        title="Aligned by recent history",
    )
    axes[1].set(
        xlabel="Absolute source-repeat index (1 = earliest)",
        ylabel="Model repeat",
        title="Aligned by initial history",
    )
    axes[2].set(
        xlabel="Relative source age (0 = current, 1 = oldest)",
        ylabel="Model repeat",
        title="Mass in equal-width relative-age bins",
    )
    figure.colorbar(
        images[2], ax=axes, label="Mean attention mass (log scale)"
    )
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_attention_mass_cdf(summary, output_path, title):
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for depth in summary["selected_depths"]:
        values = summary["attention_mass_cdfs"][str(depth)]
        step_x = [0.0]
        step_y = [0.0]
        previous = 0.0
        for relative_age, cumulative_mass in zip(
            values["relative_ages"], values["cumulative_mass"]
        ):
            step_x.extend((relative_age, relative_age))
            step_y.extend((previous, cumulative_mass))
            previous = cumulative_mass
        axis.plot(step_x, step_y, label=f"Repeat {depth}")
    axis.set(
        xlabel="Relative source age (0 = current, 1 = oldest)",
        ylabel="Cumulative attention mass",
        xlim=(-0.01, 1.01),
        ylim=(0, 1.02),
        title=title,
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_attention_window_widths(summary, output_path, title):
    repeats = summary["repeats"]
    rows = summary["attention_window_widths"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for color, threshold in zip(("C0", "C1", "C2"), (0.5, 0.8, 0.9)):
        key = str(threshold)
        absolute_mean = [row["thresholds"][key]["mean_blocks"] for row in rows]
        absolute_low = [
            row["thresholds"][key]["example_p10_blocks"] for row in rows
        ]
        absolute_high = [
            row["thresholds"][key]["example_p90_blocks"] for row in rows
        ]
        fraction_mean = [row["thresholds"][key]["mean_fraction"] for row in rows]
        fraction_low = [
            row["thresholds"][key]["example_p10_fraction"] for row in rows
        ]
        fraction_high = [
            row["thresholds"][key]["example_p90_fraction"] for row in rows
        ]
        label = f"{int(threshold * 100)}% mass"
        axes[0].fill_between(
            repeats, absolute_low, absolute_high, color=color, alpha=0.12
        )
        axes[0].plot(repeats, absolute_mean, color=color, label=label)
        axes[1].fill_between(
            repeats, fraction_low, fraction_high, color=color, alpha=0.12
        )
        axes[1].plot(repeats, fraction_mean, color=color, label=label)
    axes[0].set(
        xlabel="Model repeat",
        ylabel="Newest repeat blocks required",
        title="Absolute attention-window width",
    )
    axes[1].set(
        xlabel="Model repeat",
        ylabel="Fraction of available repeats",
        ylim=(0, 1.02),
        title="Fractional attention-window width",
    )
    axes[0].legend(title="Mean; shading = example p10–p90")
    axes[1].legend(title="Mean; shading = example p10–p90")
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_fixed_attention_windows(summary, output_path, title):
    repeats = summary["repeats"]
    rows = summary["fixed_attention_windows"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                                constrained_layout=True)
    for window in (1, 3, 6, 12):
        axes[0].plot(
            repeats,
            [row["absolute_recent_mass"][str(window)] for row in rows],
            label=f"Newest {window}",
        )
    axes[0].plot(
        repeats,
        [row["oldest_one_mass"] for row in rows],
        linestyle="--",
        label="Oldest 1",
    )
    axes[0].plot(
        repeats,
        [row["oldest_three_mass"] for row in rows],
        linestyle="--",
        label="Oldest 3",
    )
    for fraction in (0.2, 0.4, 0.6):
        axes[1].plot(
            repeats,
            [row["fractional_recent_mass"][str(fraction)]["mass"] for row in rows],
            label=f"Newest {int(fraction * 100)}%",
        )
    axes[0].set(
        xlabel="Model repeat",
        ylabel="Mean attention mass",
        ylim=(-0.02, 1.02),
        title="Fixed absolute windows",
    )
    axes[1].set(
        xlabel="Model repeat",
        ylim=(-0.02, 1.02),
        title="Fixed fractional windows",
    )
    axes[0].legend(ncol=2)
    axes[1].legend()
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_repeat_logsumexp_margins(summary, output_path, title):
    repeats = summary["repeats"]
    matrix = torch.full((len(repeats), len(repeats)), float("nan"))
    for row, values in enumerate(summary["repeat_logsumexp_margin_by_age"]):
        matrix[row, : len(values)] = torch.tensor([
            float("nan") if value is None else value for value in values
        ])
    finite = matrix[torch.isfinite(matrix)]
    limit = float(finite.abs().max())
    if limit == 0:
        limit = 1.0

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    image = axes[0].imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    ticks = list(range(0, len(repeats), max(1, len(repeats) // 10)))
    axes[0].set_xticks(ticks, ticks)
    axes[0].set_yticks(ticks, [repeats[index] for index in ticks])
    axes[0].set(
        xlabel="Absolute source age (0 = current)",
        ylabel="Model repeat",
        title="Mean block log-sum-exp margin to current",
    )
    figure.colorbar(image, ax=axes[0], label="Log-score margin")

    for depth in summary["selected_depths"]:
        values = summary["repeat_logsumexp_profiles"][str(depth)]
        profile = [
            float("nan") if value is None else value
            for value in values["mean_margin_to_current"]
        ]
        axes[1].plot(
            values["relative_ages"],
            profile,
            label=f"Repeat {depth}",
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Relative source age (0 = current, 1 = oldest)",
        ylabel="Mean log-sum-exp margin to current",
        xlim=(0, 1),
        title="Selected-depth margin profiles",
    )
    axes[1].legend(ncol=2)
    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _compare_values(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            return {"kind": "type_mismatch", "exact": False}
        result = {
            "kind": "tensor",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "exact": bool(torch.equal(left, right)),
        }
        if left.shape == right.shape and left.is_floating_point() and right.is_floating_point():
            difference = left.float().sub(right.float()).abs()
            result["maximum_absolute_difference"] = float(difference.max())
            result["mean_absolute_difference"] = float(difference.mean())
        return result
    exact = type(left) is type(right) and left == right
    return {"kind": "metadata", "exact": bool(exact), "left": left, "right": right}


def compare_artifact_overlap(
    candidate_manifest,
    candidate_shard,
    candidate_path,
    reference_manifest,
    reference_shard,
    reference_path,
):
    candidate_records = {
        (record["repeat_index"], record["middle_layer_index"]): record
        for record in candidate_shard["attention_records"]
    }
    reference_records = {
        (record["repeat_index"], record["middle_layer_index"]): record
        for record in reference_shard["attention_records"]
    }
    candidate_max = max(pair[0] for pair in candidate_records)
    reference_max = max(pair[0] for pair in reference_records)
    common_repeats = min(candidate_max, reference_max)

    candidate_condition = candidate_shard["condition"]
    reference_condition = reference_shard["condition"]
    metadata = {
        "run_id": _compare_values(
            candidate_manifest["source"]["run_id"],
            reference_manifest["source"]["run_id"],
        ),
        "checkpoint_sha256": _compare_values(
            candidate_manifest["source"]["checkpoint"]["sha256"],
            reference_manifest["source"]["checkpoint"]["sha256"],
        ),
        "dataset": _compare_values(
            candidate_manifest["evaluation"]["dataset"],
            reference_manifest["evaluation"]["dataset"],
        ),
        "condition": _compare_values(candidate_condition, reference_condition),
        "batch_index": _compare_values(
            candidate_shard["batch_index"], reference_shard["batch_index"]
        ),
    }
    trajectory = {
        "example_ids": _compare_values(
            candidate_shard["example_ids"], reference_shard["example_ids"]
        ),
        "inputs": _compare_values(candidate_shard["inputs"], reference_shard["inputs"]),
        "targets_by_horizon": _compare_values(
            candidate_shard["targets_by_horizon"][:, : common_repeats + 1],
            reference_shard["targets_by_horizon"][:, : common_repeats + 1],
        ),
        "decoded_logits_by_repeat": _compare_values(
            candidate_shard["decoded_logits_by_repeat"][:, : common_repeats + 1],
            reference_shard["decoded_logits_by_repeat"][:, : common_repeats + 1],
        ),
        "repeat_states": _compare_values(
            candidate_shard["repeat_states"][:, : common_repeats + 1],
            reference_shard["repeat_states"][:, : common_repeats + 1],
        ),
    }
    candidate_predictions = candidate_shard["decoded_logits_by_repeat"][
        :, : common_repeats + 1
    ].argmax(dim=-1)
    reference_predictions = reference_shard["decoded_logits_by_repeat"][
        :, : common_repeats + 1
    ].argmax(dim=-1)
    prediction_equal = candidate_predictions.eq(reference_predictions)

    expected_pairs = sorted(
        pair for pair in reference_records if pair[0] <= common_repeats
    )
    candidate_pairs = sorted(
        pair for pair in candidate_records if pair[0] <= common_repeats
    )
    record_results = []
    for pair in sorted(set(expected_pairs) | set(candidate_pairs)):
        if pair not in candidate_records or pair not in reference_records:
            record_results.append({
                "repeat_index": pair[0],
                "middle_layer_index": pair[1],
                "exact": False,
                "missing_from": (
                    "candidate" if pair not in candidate_records else "reference"
                ),
            })
            continue
        candidate_record = candidate_records[pair]
        reference_record = reference_records[pair]
        candidate_keys = set(candidate_record)
        reference_keys = set(reference_record)
        fields = {
            key: _compare_values(candidate_record[key], reference_record[key])
            for key in sorted(candidate_keys & reference_keys)
        }
        record_results.append({
            "repeat_index": pair[0],
            "middle_layer_index": pair[1],
            "exact": (
                candidate_keys == reference_keys
                and all(result["exact"] for result in fields.values())
            ),
            "candidate_only_fields": sorted(candidate_keys - reference_keys),
            "reference_only_fields": sorted(reference_keys - candidate_keys),
            "fields": fields,
        })

    exact = (
        all(result["exact"] for result in metadata.values())
        and all(result["exact"] for result in trajectory.values())
        and expected_pairs == candidate_pairs
        and all(result["exact"] for result in record_results)
    )
    return {
        "comparison_schema_version": 1,
        "status": "passed" if exact else "mismatch",
        "candidate_artifact": str(candidate_path),
        "reference_artifact": str(reference_path),
        "candidate_max_repeats": candidate_max,
        "reference_max_repeats": reference_max,
        "common_repeats": common_repeats,
        "metadata": metadata,
        "trajectory": trajectory,
        "prediction_agreement": float(prediction_equal.float().mean()),
        "differing_predictions": int((~prediction_equal).sum()),
        "attention_records": record_results,
    }


def main(argv=None):
    args = parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    manifest_path = artifact_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"artifact manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    artifact_summary_path = artifact_dir / "summary.json"
    artifact_summary = (
        _read_json(artifact_summary_path)
        if artifact_summary_path.is_file()
        else {}
    )
    full_split = artifact_summary.get("factorial_full_split_evaluation")
    factorial_shards = [
        shard for shard in manifest.get("shards", ())
        if shard.get("experiment_kind") == "clean_state_cache_reset_factorial"
        and shard.get("batch_index") == args.batch_index
    ]
    if factorial_shards:
        if args.overlap_artifact is not None:
            raise ValueError("factorial cosine plotting does not support overlap")
        matches = [
            shard for shard in factorial_shards
            if args.factorial_source_depth is None
            or shard.get("source_depth") == args.factorial_source_depth
        ]
        if len(matches) != 1:
            raise ValueError("factorial source depth is ambiguous or unavailable")
        shard_path = artifact_dir / matches[0]["path"]
        if not shard_path.is_file():
            raise ValueError(f"factorial shard does not exist: {shard_path}")
        # shard = torch.load(
        #     shard_path, map_location="cpu", weights_only=False, mmap=True
        # )
        shard = torch.load(str(shard_path), map_location="cpu", weights_only=False, mmap=True)
        summary = summarize_factorial_state_cosines(
            shard, args.cosine_example_count
        )
        output_dir = (
            args.output_dir.resolve() if args.output_dir is not None
            else artifact_dir / "plots" /
            f"factorial_t{summary['source_depth']:05d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for example in summary["examples"]:
            title = (
                f"{manifest['source']['run_id']} · source depth "
                f"{summary['source_depth']} · example {example['example_id']}"
            )
            _plot_factorial_state_cosines(
                summary,
                example,
                output_dir / f"state_cosine_example_{example['example_id']}.png",
                title,
            )
        summary["shard"] = str(shard_path)
        summary["run_id"] = manifest["source"]["run_id"]
        summary["definition"] = (
            "cosine between flattened [cell, hidden] recurrent states"
        )
        if full_split is None:
            summary["full_split_outcomes"] = "not_available"
        else:
            if full_split.get("source_depth") != summary["source_depth"]:
                raise ValueError(
                    "factorial batch and full-split source depths disagree"
                )
            _plot_factorial_outcomes(
                full_split,
                output_dir,
                f"{manifest['source']['run_id']} · source depth "
                f"{summary['source_depth']}",
            )
            summary["full_split_outcomes"] = {
                "status": "plotted",
                "num_examples": full_split["num_examples"],
                "num_batches": full_split["num_batches"],
            }
        diagnostics = summarize_factorial_diagnostics(shard)
        _plot_factorial_diagnostics(
            diagnostics,
            output_dir,
            f"{manifest['source']['run_id']} · source depth "
            f"{summary['source_depth']}",
        )
        summary["factorial_diagnostics"] = diagnostics
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output_directory": str(output_dir), "files": sorted(
            path.name for path in output_dir.iterdir()
        )}, indent=2))
        return
    if full_split is not None:
        source_depth = full_split.get("source_depth")
        if isinstance(source_depth, bool) or not isinstance(source_depth, int):
            raise ValueError("factorial full-split source depth is invalid")
        if (
            args.factorial_source_depth is not None
            and args.factorial_source_depth != source_depth
        ):
            raise ValueError(
                "factorial source depth does not match full-split evaluation"
            )
        output_dir = (
            args.output_dir.resolve() if args.output_dir is not None
            else artifact_dir / "plots" / f"factorial_t{source_depth:05d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _plot_factorial_outcomes(
            full_split,
            output_dir,
            f"{manifest['source']['run_id']} · source depth {source_depth}",
        )
        summary = {
            "run_id": manifest["source"]["run_id"],
            "source_depth": source_depth,
            "factorial_batch_capture": "not_available",
            "factorial_diagnostics": "not_available",
            "state_cosines": "not_available",
            "full_split_outcomes": {
                "status": "plotted",
                "num_examples": full_split["num_examples"],
                "num_batches": full_split["num_batches"],
            },
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output_directory": str(output_dir), "files": sorted(
            path.name for path in output_dir.iterdir()
        )}, indent=2))
        return
    shard_path = _find_shard(
        manifest, args.condition, args.batch_index, artifact_dir
    )
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    if shard.get("condition", {}).get("condition_id") != args.condition:
        raise ValueError("shard condition does not match requested condition")
    summary = summarize_shard(
        shard,
        args.relative_age_bins,
        args.selected_depths,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifact_dir / "plots" / "initial"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    title = (
        f"{manifest['source']['run_id']} · {args.condition} · "
        f"batch {args.batch_index} (n={summary['examples']})"
    )
    _plot_accuracy(summary, output_dir / "accuracy.png", title)
    _plot_entropy(summary, output_dir / "entropy.png", title)
    _plot_concentration(summary, output_dir / "concentration.png", title)
    _plot_repeat_age_mass(summary, output_dir / "repeat_age_mass.png", title)
    _plot_repeat_age_mass(
        summary,
        output_dir / "repeat_age_mass_log.png",
        title,
        log_scale=True,
    )
    _plot_age_mass_vs_contribution(
        summary,
        output_dir / "repeat_age_mass_vs_contribution.png",
        title,
    )
    _plot_repeat_mass_coordinates(
        summary,
        output_dir / "repeat_mass_coordinate_views.png",
        title,
    )
    _plot_attention_mass_cdf(
        summary,
        output_dir / "attention_mass_relative_cdf.png",
        title,
    )
    _plot_attention_window_widths(
        summary,
        output_dir / "attention_window_widths.png",
        title,
    )
    _plot_fixed_attention_windows(
        summary,
        output_dir / "fixed_attention_windows.png",
        title,
    )
    _plot_repeat_logsumexp_margins(
        summary,
        output_dir / "repeat_logsumexp_margins.png",
        title,
    )

    overlap_report = None
    if args.overlap_artifact is not None:
        reference_dir = args.overlap_artifact.resolve()
        reference_manifest_path = reference_dir / "artifact_manifest.json"
        if not reference_manifest_path.is_file():
            raise ValueError(
                f"overlap artifact manifest does not exist: {reference_manifest_path}"
            )
        reference_manifest = _read_json(reference_manifest_path)
        reference_shard_path = _find_shard(
            reference_manifest, args.condition, args.batch_index, reference_dir
        )
        reference_shard = torch.load(
            reference_shard_path, map_location="cpu", weights_only=False
        )
        overlap_report = compare_artifact_overlap(
            manifest,
            shard,
            artifact_dir,
            reference_manifest,
            reference_shard,
            reference_dir,
        )
        with (output_dir / "overlap_comparison.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(overlap_report, handle, indent=2, allow_nan=False)
            handle.write("\n")

    output_summary = {
        "analysis_schema_version": 1,
        "source": {
            "artifact_manifest": str(manifest_path),
            "artifact_schema_version": manifest["artifact_schema_version"],
            "run_id": manifest["source"]["run_id"],
            "condition_id": args.condition,
            "batch_index": args.batch_index,
            "shard": str(shard_path),
        },
        "scope": "single captured batch; descriptive statistics only",
        "definitions": {
            "matching_cell_accuracy": (
                "mean correctness across examples and cells when model repeat "
                "equals target Rule 30 horizon"
            ),
            "matching_sequence_accuracy": (
                "fraction of examples with every cell correct when model repeat "
                "equals target Rule 30 horizon"
            ),
            "entropy_example_band": (
                "10th to 90th percentile of per-example means; not a confidence "
                "interval"
            ),
            "source_age": "zero is the current repeat; larger values are older",
            "source_repeat_index": (
                "one is the earliest repeat; the current repeat lies on the diagonal"
            ),
            "relative_source_age": (
                "source age divided by repeat minus one; masses are summed into "
                f"{args.relative_age_bins} equal-width bins"
            ),
            "attention_window_width": (
                "minimum number of newest repeat blocks needed to reach the "
                "specified mass threshold, computed per example, head and query"
            ),
            "repeat_logsumexp_margin": (
                "pre-softmax repeat-block log-sum-exp minus the current block's "
                "log-sum-exp, computed per example, head and query before averaging"
            ),
        },
        "metrics": summary,
    }
    if overlap_report is not None:
        output_summary["overlap_comparison"] = {
            "path": "overlap_comparison.json",
            "status": overlap_report["status"],
            "common_repeats": overlap_report["common_repeats"],
        }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(output_summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_directory": str(output_dir), "files": sorted(
        path.name for path in output_dir.iterdir()
    )}, indent=2))
    if overlap_report is not None and overlap_report["status"] != "passed":
        raise RuntimeError("shared artifact trajectory does not match exactly")


if __name__ == "__main__":
    main()
