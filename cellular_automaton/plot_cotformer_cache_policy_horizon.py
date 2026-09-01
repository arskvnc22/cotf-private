"""Plot seed-averaged CoTFormer accuracy by trained cache policy.

This is a focused view of the same normalized records selected by the
``ca_analyze table`` command used for the cache-policy comparison.  It retains
only depth-matched evaluations (Rule 30 steps == model repeats), averages runs
within the analyzer's conservative scientific-configuration groups, and shows
one sample-standard-deviation band around each mean curve.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from cellular_automaton.ca_analyze import (
    METRICS_FILENAME,
    REPO_ROOT,
    discover_runs,
    dotted_get,
    group_runs,
    iter_metrics,
    iter_selected_records,
    parse_filter,
    summarize,
    training_cache_label,
)


TRAINING_PAIRS = "|".join(f"{depth}:{depth}" for depth in range(1, 13))
DEFAULT_RUNS = REPO_ROOT / "iridis/ca-rule30/runs"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "dis-draft/dis-images/cotformer-trained-cache-policy-horizon.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        default="best_extrapolation_unconstrained",
    )
    parser.add_argument("--maximum-depth", type=int, default=20)
    return parser.parse_args()


def policy_sort_key(label: str) -> tuple[int, int]:
    if label == "full":
        return (0, 0)
    if label.startswith("recent-"):
        return (1, int(label.removeprefix("recent-")))
    return (2, 0)


def display_policy(label: str) -> str:
    if label == "full":
        return "Full cache"
    if label.startswith("recent-"):
        return f"Recent {label.removeprefix('recent-')}"
    return label


def main() -> None:
    args = parse_args()
    if args.maximum_depth < 1:
        raise ValueError("--maximum-depth must be positive")

    filters = tuple(
        parse_filter(value)
        for value in (
            "model=ca_cotf_cache_attn",
            "status=completed",
            f"training_pairs={TRAINING_PAIRS}",
        )
    )
    runs = discover_runs(args.runs, filters=filters, load_metrics=False)
    if not runs:
        raise ValueError("No completed cache-policy CoTFormer runs matched")

    selection = SimpleNamespace(
        metric="cell_accuracy",
        split="final_test",
        role="repeat_horizon_diagnostic",
        checkpoint=args.checkpoint,
        length=64,
        step="latest",
    )
    groups = group_runs(runs, average_over=("seed",))
    series = []
    for group_id, members in groups.items():
        values_by_depth: dict[int, list[float]] = defaultdict(list)
        for run in members:
            seen_depths = set()
            records = iter_metrics(run.path / METRICS_FILENAME)
            for record in iter_selected_records(records, selection):
                steps = record.get("ca_steps")
                repeats = record.get("num_repeats")
                if steps != repeats or steps is None:
                    continue
                depth = int(steps)
                if depth > args.maximum_depth:
                    continue
                if depth in seen_depths:
                    raise ValueError(
                        f"Duplicate depth-matched record for {run.run_id} at {depth}"
                    )
                seen_depths.add(depth)
                values_by_depth[depth].append(float(record["value"]))
        if not values_by_depth:
            continue
        representative = members[0].manifest
        series.append(
            {
                "group_id": group_id,
                "policy": training_cache_label(representative),
                "eval_freq": dotted_get(representative, "resolved_args.eval_freq"),
                "runs": members,
                "values": dict(values_by_depth),
            }
        )
    if not series:
        raise ValueError("No depth-matched final-test records matched")

    policy_counts = Counter(item["policy"] for item in series)
    series.sort(
        key=lambda item: (
            policy_sort_key(item["policy"]),
            item["eval_freq"] if item["eval_freq"] is not None else -1,
        )
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = Path(tempfile.gettempdir()) / "rs-cot-matplotlib"
    mpl_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.6, 5.8))
    axis.axvspan(1, 12, color="#d9d9d9", alpha=0.30, zorder=0)
    axis.axvline(12, color="#666666", linestyle="--", linewidth=0.9)
    axis.axhline(0.5, color="#555555", linestyle=":", linewidth=0.9)

    policies = sorted({item["policy"] for item in series}, key=policy_sort_key)
    colours = dict(zip(policies, plt.get_cmap("tab10").colors, strict=False))
    policy_occurrence: Counter[str] = Counter()
    csv_rows = []
    for item in series:
        policy = item["policy"]
        occurrence = policy_occurrence[policy]
        policy_occurrence[policy] += 1
        linestyle = "-" if occurrence == 0 else "--"
        depths = sorted(item["values"])
        summaries = [summarize(item["values"][depth]) for depth in depths]
        means = [summary["mean"] for summary in summaries]
        stds = [summary["std"] for summary in summaries]
        seed_count = len(
            {
                dotted_get(run.manifest, "seed")
                for run in item["runs"]
            }
        )
        label = f"{display_policy(policy)} (n={seed_count})"
        if policy_counts[policy] > 1:
            label = (
                f"{display_policy(policy)}, eval/{item['eval_freq']} "
                f"(n={seed_count})"
            )
        colour = colours[policy]
        axis.plot(
            depths,
            means,
            color=colour,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axis.fill_between(
            depths,
            [max(0.0, mean - std) for mean, std in zip(means, stds)],
            [min(1.0, mean + std) for mean, std in zip(means, stds)],
            color=colour,
            alpha=0.12,
            linewidth=0,
        )
        run_ids = ",".join(run.run_id for run in item["runs"])
        seeds = ",".join(
            str(seed)
            for seed in sorted(
                dotted_get(run.manifest, "seed") for run in item["runs"]
            )
        )
        for depth, summary in zip(depths, summaries):
            csv_rows.append(
                {
                    "configuration_id": item["group_id"],
                    "training_cache": policy,
                    "eval_freq": item["eval_freq"],
                    "depth": depth,
                    "n": summary["n"],
                    "mean": summary["mean"],
                    "sample_std": summary["std"],
                    "sample_variance": summary["variance"],
                    "runs": run_ids,
                    "seeds": seeds,
                }
            )

    axis.set_xlim(1, args.maximum_depth)
    axis.set_ylim(0.45, 1.01)
    axis.set_xticks(range(1, args.maximum_depth + 1))
    axis.set_xlabel("Evaluated depth (Rule 30 steps = model repeats)")
    axis.set_ylabel("Cell accuracy")
    axis.set_title("CoTFormer extrapolation by training cache policy")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.015,
        "Curves are seed means; bands show ±1 sample SD. Shading marks supervised depths 1--12.",
        ha="center",
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.08, right=0.76, bottom=0.17, top=0.90)
    figure.savefig(output, dpi=300)
    plt.close(figure)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
