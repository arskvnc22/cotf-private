"""Measure fixed CA materialization, loading, and fresh-batch generation."""

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

try:
    from .ca_gen import MaterializedRule30Dataset, generate_rule30_batch
except ImportError:
    from ca_gen import MaterializedRule30Dataset, generate_rule30_batch


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--train_samples", type=int, default=100_000)
    parser.add_argument("--val_samples", type=int, default=10_000)
    parser.add_argument("--train_num_cells", type=int, default=64)
    parser.add_argument("--eval_num_cells", type=int, nargs="+", default=[64])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--bernoulli_p", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_seed", type=int, default=1_000_003)
    parser.add_argument(
        "--online_batches",
        type=int,
        default=None,
        help="Fresh batches to generate; default matches one stored training pass.",
    )
    parser.add_argument(
        "--skip_cuda",
        action="store_true",
        help="Skip GPU materialization and streaming measurements.",
    )
    return parser.parse_args()


def timed_materialized_dataset(**kwargs):
    start = time.perf_counter()
    dataset = MaterializedRule30Dataset(**kwargs)
    return dataset, time.perf_counter() - start


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def batch_sizes(args):
    if args.online_batches is not None:
        return [args.batch_size] * args.online_batches
    full_batches, remainder = divmod(args.train_samples, args.batch_size)
    sizes = [args.batch_size] * full_batches
    if remainder:
        sizes.append(remainder)
    return sizes


def benchmark_fresh_generation(args, *, device, copy_to_cpu):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sizes = batch_sizes(args)
    copied_bytes = 0

    if device.type == "cuda":
        warmup_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        generate_rule30_batch(
            batch_size=min(args.batch_size, 32),
            num_cells=args.train_num_cells,
            device=device,
            steps=args.steps,
            bernoulli_p=args.bernoulli_p,
            generator=warmup_generator,
        )
        synchronize(device)

    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for current_batch_size in sizes:
            batch = generate_rule30_batch(
                batch_size=current_batch_size,
                num_cells=args.train_num_cells,
                device=device,
                steps=args.steps,
                bernoulli_p=args.bernoulli_p,
                generator=generator,
            )
            if copy_to_cpu:
                cpu_inputs = batch["input_id"].to(torch.uint8).cpu()
                cpu_labels = batch["label"].to(torch.uint8).cpu()
                copied_bytes += (
                    cpu_inputs.numel() * cpu_inputs.element_size()
                    + cpu_labels.numel() * cpu_labels.element_size()
                )
    synchronize(device)
    seconds = time.perf_counter() - start
    rows = sum(sizes)
    return {
        "device": str(device),
        "copy_to_cpu": copy_to_cpu,
        "seconds": seconds,
        "batches": len(sizes),
        "rows": rows,
        "rows_per_second": rows / seconds,
        "copied_mib": copied_bytes / (1024 ** 2),
    }


def benchmark_cuda_materialized_split(args, *, num_samples, num_cells, seed):
    """Time bulk GPU generation separately from compression and CPU transfer."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    synchronize(device)
    generation_start = time.perf_counter()
    with torch.no_grad():
        batch = generate_rule30_batch(
            batch_size=num_samples,
            num_cells=num_cells,
            device=device,
            steps=args.steps,
            bernoulli_p=args.bernoulli_p,
            generator=generator,
        )
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_start

    transfer_start = time.perf_counter()
    cpu_inputs = batch["input_id"].to(torch.uint8).cpu()
    cpu_labels = batch["label"].to(torch.uint8).cpu()
    synchronize(device)
    compression_and_copy_seconds = time.perf_counter() - transfer_start
    storage_bytes = (
        cpu_inputs.numel() * cpu_inputs.element_size()
        + cpu_labels.numel() * cpu_labels.element_size()
    )
    return {
        "rows": num_samples,
        "num_cells": num_cells,
        "generation_seconds": generation_seconds,
        "compression_and_copy_to_cpu_seconds": compression_and_copy_seconds,
        "total_seconds": generation_seconds + compression_and_copy_seconds,
        "generation_rows_per_second": num_samples / generation_seconds,
        "total_rows_per_second": num_samples
        / (generation_seconds + compression_and_copy_seconds),
        "stored_mib": storage_bytes / (1024 ** 2),
    }


def main(args):
    if args.online_batches is not None and args.online_batches <= 0:
        raise ValueError("--online_batches must be positive when provided.")

    train_dataset, train_generation_seconds = timed_materialized_dataset(
        num_samples=args.train_samples,
        num_cells=args.train_num_cells,
        steps=args.steps,
        bernoulli_p=args.bernoulli_p,
        seed=args.seed,
    )

    eval_datasets = {}
    eval_generation_seconds = {}
    for num_cells in args.eval_num_cells:
        dataset, seconds = timed_materialized_dataset(
            num_samples=args.val_samples,
            num_cells=num_cells,
            steps=args.steps,
            bernoulli_p=args.bernoulli_p,
            seed=args.val_seed + num_cells,
        )
        eval_datasets[num_cells] = dataset
        eval_generation_seconds[str(num_cells)] = seconds

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    loaded_rows = 0
    loader_checksum = 0
    loader_start = time.perf_counter()
    for batch in train_loader:
        inputs = batch["input_id"].long()
        labels = batch["label"].long()
        loaded_rows += inputs.shape[0]
        loader_checksum += int(inputs.sum()) + int(labels.sum())
    loader_seconds = time.perf_counter() - loader_start

    fresh_generation = {
        "cpu_no_copy": benchmark_fresh_generation(
            args, device="cpu", copy_to_cpu=False
        )
    }

    cuda_results = None
    if not args.skip_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA benchmarking was requested but torch.cuda.is_available() is false. "
                "Use --skip_cuda for a CPU-only run."
            )

        warmup_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        generate_rule30_batch(
            batch_size=min(args.batch_size, 32),
            num_cells=args.train_num_cells,
            device="cuda",
            steps=args.steps,
            bernoulli_p=args.bernoulli_p,
            generator=warmup_generator,
        )
        torch.cuda.synchronize()

        cuda_splits = {
            "train": benchmark_cuda_materialized_split(
                args,
                num_samples=args.train_samples,
                num_cells=args.train_num_cells,
                seed=args.seed,
            ),
            "eval_by_length": {},
        }
        for num_cells in args.eval_num_cells:
            cuda_splits["eval_by_length"][str(num_cells)] = (
                benchmark_cuda_materialized_split(
                    args,
                    num_samples=args.val_samples,
                    num_cells=num_cells,
                    seed=args.val_seed + num_cells,
                )
            )
        split_results = [cuda_splits["train"], *cuda_splits["eval_by_length"].values()]
        cuda_splits["all_splits_total_seconds"] = sum(
            result["total_seconds"] for result in split_results
        )
        cuda_splits["all_splits_generation_seconds"] = sum(
            result["generation_seconds"] for result in split_results
        )
        cuda_splits["all_splits_compression_and_copy_to_cpu_seconds"] = sum(
            result["compression_and_copy_to_cpu_seconds"]
            for result in split_results
        )
        cuda_results = cuda_splits

        fresh_generation["cuda_no_copy"] = benchmark_fresh_generation(
            args, device="cuda", copy_to_cpu=False
        )
        fresh_generation["cuda_with_uint8_copy_to_cpu"] = benchmark_fresh_generation(
            args, device="cuda", copy_to_cpu=True
        )

    total_storage_bytes = train_dataset.storage_bytes + sum(
        dataset.storage_bytes for dataset in eval_datasets.values()
    )
    total_materialized_rows = args.train_samples + (
        args.val_samples * len(eval_datasets)
    )
    total_generation_seconds = train_generation_seconds + sum(
        eval_generation_seconds.values()
    )
    results = {
        "configuration": vars(args),
        "materialization": {
            "cpu": {
                "train_seconds": train_generation_seconds,
                "eval_seconds_by_length": eval_generation_seconds,
                "total_seconds": total_generation_seconds,
                "total_rows": total_materialized_rows,
                "rows_per_second": total_materialized_rows / total_generation_seconds,
                "storage_mib": total_storage_bytes / (1024 ** 2),
            },
            "cuda_generate_then_store_on_cpu": cuda_results,
        },
        "stored_dataloader_pass": {
            "seconds": loader_seconds,
            "rows": loaded_rows,
            "rows_per_second": loaded_rows / loader_seconds,
            "checksum": loader_checksum,
        },
        "fresh_online_generation": fresh_generation,
    }
    if cuda_results is not None:
        cpu_online = fresh_generation["cpu_no_copy"]
        cuda_online = fresh_generation["cuda_no_copy"]
        cuda_online_copy = fresh_generation["cuda_with_uint8_copy_to_cpu"]
        gpu_total = cuda_results["all_splits_total_seconds"]
        gpu_copy = cuda_results[
            "all_splits_compression_and_copy_to_cpu_seconds"
        ]
        results["comparisons"] = {
            "bulk_gpu_generate_and_copy_speedup_vs_cpu_materialization": (
                total_generation_seconds / gpu_total
            ),
            "bulk_gpu_copy_fraction": gpu_copy / gpu_total,
            "online_gpu_no_copy_speedup_vs_cpu": (
                cuda_online["rows_per_second"] / cpu_online["rows_per_second"]
            ),
            "online_gpu_with_copy_speedup_vs_cpu": (
                cuda_online_copy["rows_per_second"]
                / cpu_online["rows_per_second"]
            ),
            "online_copy_penalty_relative_to_gpu_no_copy": (
                cuda_online["rows_per_second"]
                / cuda_online_copy["rows_per_second"]
            ),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(get_args())
