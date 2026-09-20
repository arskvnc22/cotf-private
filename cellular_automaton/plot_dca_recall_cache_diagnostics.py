"""Plot per-head DCA recall attention and associated value diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch, Rectangle


ATTENTION_FIELD = "mean_mass_by_head_and_cache_block"
VALUE_FIELD = "mean_value_norm_by_head_and_cache_block"
CONTRIBUTION_FIELD = (
    "mean_block_contribution_norm_by_head_and_cache_block"
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
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--depths",
        type=positive_int,
        nargs="+",
        required=True,
        help="Predetermined intervention depths to plot.",
    )
    parser.add_argument(
        "--horizon",
        type=positive_int,
        help="Required only when the summary contains multiple horizons.",
    )
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--recall-index", type=positive_int, default=1)
    parser.add_argument("--layer-index", type=nonnegative_int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _read_summary(path):
    with Path(path).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, Mapping):
        raise ValueError("summary must contain a JSON mapping")
    if summary.get("experiment") != "dca_model_level_recall_cache_interventions":
        raise ValueError("summary is not a DCA recall-cache intervention report")
    if not isinstance(summary.get("results"), list) or not summary["results"]:
        raise ValueError("summary contains no intervention results")
    return summary


def _resolve_horizon(summary, requested):
    horizons = sorted({int(result["horizon"]) for result in summary["results"]})
    if requested is None:
        if len(horizons) != 1:
            raise ValueError(
                "summary contains multiple horizons; select one with --horizon"
            )
        return horizons[0]
    if requested not in horizons:
        raise ValueError(f"horizon {requested} is not present in the summary")
    return requested


def _matrix(row, field, expected_shape=None):
    if field not in row:
        raise ValueError(
            f"attention row is missing {field!r}; rerun the evaluator with "
            "value-diagnostic aggregation enabled"
        )
    values = torch.tensor(row[field], dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError(f"{field} must have shape [heads, cache_blocks]")
    if expected_shape is not None and tuple(values.shape) != expected_shape:
        raise ValueError(f"{field} does not align with the attention matrix")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{field} must be finite and nonnegative")
    return values


def _cache_labels(cache_blocks, horizon):
    labels = []
    for expected_index, block in enumerate(cache_blocks, start=1):
        if block.get("cache_block_index_one_based") != expected_index:
            raise ValueError("cache block indices must be consecutive and one-based")
        if block.get("phase") == "evolution":
            repeat = int(block["evolution_repeat"])
            if repeat != expected_index or repeat > horizon:
                raise ValueError("evolution cache block metadata is inconsistent")
            labels.append(f"E{repeat}")
        elif block.get("phase") == "recall":
            labels.append(f"R{int(block['recall_index'])}")
        else:
            raise ValueError("cache block has an unsupported phase")
    return labels


def _internal_cell_accuracy(condition):
    try:
        value = condition["internal_consistency"]["decoded_requested_repeat"][
            "cell_accuracy"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("condition is missing internal cell accuracy") from error
    return float(value)


def select_plot_rows(
    summary,
    *,
    horizon,
    depths,
    condition_id,
    recall_index,
    layer_index,
):
    selected = []
    for depth in sorted(set(depths)):
        matches = [
            result
            for result in summary["results"]
            if int(result["horizon"]) == horizon
            and int(result["intervention_depth"]) == depth
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one result for horizon={horizon}, depth={depth}; "
                f"found {len(matches)}"
            )
        result = matches[0]
        conditions = result.get("conditions", {})
        if condition_id not in conditions:
            raise ValueError(
                f"condition {condition_id!r} is unavailable at depth {depth}"
            )
        condition = conditions[condition_id]
        attention_matches = [
            row
            for row in condition.get("attention", ())
            if int(row["recall_index"]) == recall_index
            and int(row["middle_layer_index_zero_based"]) == layer_index
        ]
        if len(attention_matches) != 1:
            raise ValueError(
                "expected one attention row for "
                f"depth={depth}, recall={recall_index}, layer={layer_index}; "
                f"found {len(attention_matches)}"
            )
        attention_row = attention_matches[0]
        mass = _matrix(attention_row, ATTENTION_FIELD)
        if not torch.allclose(
            mass.sum(dim=1),
            torch.ones(mass.shape[0]),
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("mean attention mass does not sum to one per head")
        value_norm = _matrix(attention_row, VALUE_FIELD, tuple(mass.shape))
        contribution = _matrix(
            attention_row, CONTRIBUTION_FIELD, tuple(mass.shape)
        )
        cache_blocks = attention_row.get("cache_blocks")
        if not isinstance(cache_blocks, list) or len(cache_blocks) != mass.shape[1]:
            raise ValueError("cache block metadata does not align with matrices")
        labels = _cache_labels(cache_blocks, horizon)
        metrics = condition.get("metrics", {})
        if "cell_accuracy" not in metrics:
            raise ValueError("condition is missing cell accuracy")
        selected.append(
            {
                "horizon": horizon,
                "depth": depth,
                "recall_age": int(result["recall_age"]),
                "is_latest_repeat": bool(result["is_latest_repeat"]),
                "cell_accuracy": float(metrics["cell_accuracy"]),
                "internal_cell_accuracy": _internal_cell_accuracy(condition),
                "condition": condition_id,
                "labels": labels,
                "mass": mass,
                "value_norm": value_norm,
                "contribution": contribution,
            }
        )
    return selected


def _positive_log_norm(matrices, *, fixed_max=None):
    positive = torch.cat([matrix.reshape(-1) for matrix in matrices])
    positive = positive[positive > 0]
    if positive.numel() == 0:
        raise ValueError("log-scaled diagnostic contains no positive values")
    maximum = float(positive.max()) if fixed_max is None else float(fixed_max)
    minimum = max(float(positive.min()), maximum * 1e-4)
    if minimum >= maximum:
        minimum = maximum * 1e-4
    return LogNorm(vmin=minimum, vmax=maximum, clip=True)


def _safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _plot_row(row, output_path, norms):
    matrices = (row["mass"], row["value_norm"], row["contribution"])
    titles = (
        "Attention probability mass",
        r"Mean $\|V\|_2$",
        "Weighted-V contribution norm\n(before head mixing and c_proj)",
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, max(4.6, 2.7 + 0.55 * matrices[0].shape[0])),
        constrained_layout=True,
    )
    images = []
    for axis, matrix, title, norm in zip(axes, matrices, titles, norms):
        masked = torch.where(matrix > 0, matrix, torch.nan)
        plotted = matrix if isinstance(norm, Normalize) and not isinstance(
            norm, LogNorm
        ) else masked
        image = axis.imshow(
            plotted.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            norm=norm,
        )
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("Cache block")
        axis.set_xticks(
            range(len(row["labels"])), row["labels"], rotation=90
        )
        axis.set_yticks(
            range(matrix.shape[0]),
            [f"head {index}" for index in range(matrix.shape[0])],
        )
        axis.axvspan(
            row["horizon"] - 0.5,
            len(row["labels"]) - 0.5,
            color="#bdbdbd",
            alpha=0.15,
        )
        axis.add_patch(
            Rectangle(
                (row["depth"] - 1.5, -0.5),
                1,
                matrix.shape[0],
                fill=False,
                edgecolor="#00e5ff",
                linewidth=1.8,
            )
        )
        if row["depth"] < row["horizon"]:
            axis.add_patch(
                Rectangle(
                    (row["depth"] - 0.5, -0.5),
                    1,
                    matrix.shape[0],
                    fill=False,
                    edgecolor="#7cff6b",
                    linestyle="--",
                    linewidth=1.8,
                )
            )
        figure.colorbar(image, ax=axis, shrink=0.84)

    legend = [
        Patch(facecolor="none", edgecolor="#00e5ff", label="requested block d"),
        Patch(
            facecolor="none",
            edgecolor="#7cff6b",
            linestyle="--",
            label="candidate state-input block d+1",
        ),
        Patch(
            facecolor="#bdbdbd",
            edgecolor="none",
            alpha=0.35,
            label="recall-phase cache",
        ),
    ]
    latest = " · latest-state/no-op control" if row["is_latest_repeat"] else ""
    figure.suptitle(
        f"DCA recall cache · H={row['horizon']} · d={row['depth']} · "
        f"recall age={row['recall_age']} · {row['condition']}\n"
        f"cell accuracy={row['cell_accuracy']:.4f} · internal cell "
        f"accuracy={row['internal_cell_accuracy']:.4f}{latest}"
    )
    figure.legend(handles=legend, loc="outside lower center", ncol=3)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    summary_path = args.summary.resolve()
    summary = _read_summary(summary_path)
    horizon = _resolve_horizon(summary, args.horizon)
    rows = select_plot_rows(
        summary,
        horizon=horizon,
        depths=args.depths,
        condition_id=args.condition,
        recall_index=args.recall_index,
        layer_index=args.layer_index,
    )
    attention_norm = _positive_log_norm(
        [row["mass"] for row in rows], fixed_max=1.0
    )
    value_max = max(float(row["value_norm"].max()) for row in rows)
    value_norm = Normalize(vmin=0.0, vmax=value_max if value_max > 0 else 1.0)
    contribution_norm = _positive_log_norm(
        [row["contribution"] for row in rows]
    )
    norms = (attention_norm, value_norm, contribution_norm)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else summary_path.parent / "plots" / "recall-cache"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    condition_component = _safe_component(args.condition)
    for row in rows:
        filename = (
            f"recall_cache_h{horizon}_d{row['depth']}_r{args.recall_index}_"
            f"l{args.layer_index}_{condition_component}.png"
        )
        _plot_row(row, output_dir / filename, norms)
        figures.append(
            {
                "depth": row["depth"],
                "recall_age": row["recall_age"],
                "cell_accuracy": row["cell_accuracy"],
                "internal_cell_accuracy": row["internal_cell_accuracy"],
                "is_latest_repeat": row["is_latest_repeat"],
                "requested_block_label": f"E{row['depth']}",
                "candidate_state_input_block_label": (
                    f"E{row['depth'] + 1}"
                    if row["depth"] < row["horizon"]
                    else None
                ),
                "recall_block_labels": row["labels"][row["horizon"] :],
                "path": filename,
            }
        )

    plot_summary = {
        "source_summary": str(summary_path),
        "horizon": horizon,
        "selected_depths": [row["depth"] for row in rows],
        "condition": args.condition,
        "recall_index": args.recall_index,
        "middle_layer_index_zero_based": args.layer_index,
        "scope": "aggregated descriptive recall diagnostics",
        "definitions": {
            "attention_mass": "mean over evaluated examples and query positions",
            "value_norm": (
                "mean L2 norm of cached V over examples and cache-block token "
                "positions, retained separately by head"
            ),
            "weighted_value_contribution_norm": (
                "mean L2 norm of each cache block's attention-weighted V sum "
                "before head concatenation and c_proj"
            ),
            "candidate_state_input_block": (
                "d+1 is marked as a timing hypothesis, not asserted as the "
                "correct learned address"
            ),
        },
        "shared_scales": {
            "attention_mass": {
                "kind": "log",
                "vmin": float(attention_norm.vmin),
                "vmax": float(attention_norm.vmax),
            },
            "value_norm": {
                "kind": "linear",
                "vmin": float(value_norm.vmin),
                "vmax": float(value_norm.vmax),
            },
            "weighted_value_contribution_norm": {
                "kind": "log",
                "vmin": float(contribution_norm.vmin),
                "vmax": float(contribution_norm.vmax),
            },
        },
        "figures": figures,
    }
    metadata_path = output_dir / "plot_summary.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(plot_summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output_directory": str(output_dir), "figures": figures}))


if __name__ == "__main__":
    main()
