"""Plot the initial frozen CoTFormer cache-window diagnostics.

The figure compares diagonal Rule 30 cell accuracy and normalized token-level
attention entropy for the full-cache, recent-12, and recent-4 inference
conditions. It reads the descriptive single-batch summaries produced by
``ca_plot_attention_diagnostics.py``; it does not recompute model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    (
        "Full cache",
        "#1f77b4",
        REPO_ROOT
        / "ca-attention-diagnostics/run84_manual_full_batch0_job_1388372/plots/initial/summary.json",
    ),
    (
        "Recent 12",
        "#e07a1f",
        REPO_ROOT
        / "ca-attention-diagnostics/run84_manual_recent12_r40/plots/initial/summary.json",
    ),
    (
        "Recent 4",
        "#2a9d55",
        REPO_ROOT
        / "ca-attention-diagnostics/run84_manual_recent4_r40/plots/initial/summary.json",
    ),
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "dis-draft/dis-images/cotformer-cache-window-accuracy-entropy.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def load_summary(path: Path) -> tuple[list[dict[str, float]], int]:
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("scope") != "single captured batch; descriptive statistics only":
        raise ValueError(f"Unexpected diagnostic scope in {path}")
    metrics = summary.get("metrics", {})
    curves = metrics.get("curves")
    if not curves:
        raise ValueError(f"No diagnostic curves found in {path}")
    examples = metrics.get("examples")
    if not isinstance(examples, int) or examples < 1:
        raise ValueError(f"Invalid example count in {path}")
    return curves, examples


def main() -> None:
    args = parse_args()
    loaded = [
        (label, colour, *load_summary(path))
        for label, colour, path in CONDITIONS
    ]
    expected_repeats = [row["repeat"] for row in loaded[0][2]]
    expected_examples = loaded[0][3]
    for label, _, curves, examples in loaded[1:]:
        repeats = [row["repeat"] for row in curves]
        if repeats != expected_repeats:
            raise ValueError(f"Repeat grid for {label} does not match full cache")
        if examples != expected_examples:
            raise ValueError(f"Example count for {label} does not match full cache")

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharex=True)
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.22, top=0.77, wspace=0.18
    )
    for axis in axes:
        axis.axvspan(1, 12, color="#d9d9d9", alpha=0.35, zorder=0)
        axis.axvline(12, color="#666666", linewidth=0.9, linestyle="--")
        axis.set_xlim(1, 40)
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.set_xlabel("Model repeat = Rule 30 horizon")

    axes[0].axhline(
        0.5, color="#555555", linewidth=0.9, linestyle=":", label="Chance"
    )
    for label, colour, curves, _ in loaded:
        repeats = [row["repeat"] for row in curves]
        axes[0].plot(
            repeats,
            [row["matching_cell_accuracy"] for row in curves],
            color=colour,
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            repeats,
            [row["normalised_token_entropy_mean"] for row in curves],
            color=colour,
            linewidth=2.0,
            label=label,
        )

    axes[0].set_title("(a) Depth-matched cell accuracy")
    axes[0].set_ylabel("Cell accuracy")
    axes[0].set_ylim(0.45, 1.02)
    axes[1].set_title("(b) Token-level attention entropy")
    axes[1].set_ylabel("Mean normalized entropy")
    axes[1].set_ylim(0.0, 1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in ("Full cache", "Recent 12", "Recent 4", "Chance")]
    figure.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    figure.text(
        0.5,
        0.035,
        "Shaded region: supervised repeats 1--12; curves summarize one "
        f"frozen-checkpoint validation batch (n={expected_examples}).",
        ha="center",
        fontsize=8.5,
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
