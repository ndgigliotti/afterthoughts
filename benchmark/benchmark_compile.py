#!/usr/bin/env python
# Copyright 2024-2026 Nicholas Gigliotti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark different torch.compile configurations for the Encoder.

This script compares:
- No compilation (baseline)
- mode="default" with dynamic=False
- mode="default" with dynamic=True
- mode="reduce-overhead" with dynamic=False
- mode="reduce-overhead" with dynamic=True

Results include warmup time, steady-state throughput, and total time.
"""

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from afterthoughts.chunk import tokenize_with_sentence_boundaries
from afterthoughts.tokenize import (
    DynamicTokenSampler,
    TokenizedDataset,
    dynamic_pad_collate,
)


@dataclass
class BenchmarkResult:
    name: str
    warmup_time: float
    steady_state_time: float
    total_time: float
    num_warmup_batches: int
    num_steady_batches: int
    docs_per_second: float
    tokens_per_second: float


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def load_sample_docs(num_docs: int = 1000, min_length: int = 200) -> list[str]:
    """Load sample documents from wikitext dataset."""
    print(f"Loading {num_docs} sample documents from wikitext...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    docs = []
    for example in dataset:
        text = example["text"].strip()
        if len(text) >= min_length:
            docs.append(text)
            if len(docs) >= num_docs:
                break
    print(f"Loaded {len(docs)} documents")
    return docs


def prepare_data(
    docs: list[str],
    tokenizer,
    max_length: int,
    batch_tokens: int,
) -> tuple[TokenizedDataset, DataLoader, int]:
    """Tokenize documents and create DataLoader."""
    inputs = tokenize_with_sentence_boundaries(
        docs,
        tokenizer,
        max_length=max_length,
        prechunk=True,
        prechunk_overlap=0.5,
        batch_size=10,
        n_jobs=-1,
        return_tokenized_dataset=True,
    )
    loader = DataLoader(
        inputs,
        shuffle=False,
        pin_memory=True,
        batch_sampler=DynamicTokenSampler(inputs, max_tokens=batch_tokens),
        collate_fn=dynamic_pad_collate,
    )
    total_tokens = sum(len(seq) for seq in inputs.data["input_ids"])
    return inputs, loader, total_tokens


def create_model(
    model_name: str,
    compile_mode: str | None,
    dynamic: bool,
    device: str = "cuda",
) -> torch.nn.Module:
    """Create and optionally compile a model."""
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": device},
    ).eval()

    if compile_mode is not None:
        model = torch.compile(model, mode=compile_mode, dynamic=dynamic)

    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_warmup_batches: int = 5,
) -> tuple[float, float, int, int]:
    """Run inference and measure warmup vs steady-state time.

    Returns:
        warmup_time: Time for first num_warmup_batches
        steady_time: Time for remaining batches
        num_warmup: Number of warmup batches actually run
        num_steady: Number of steady-state batches
    """
    torch.cuda.synchronize()

    warmup_time = 0.0
    steady_time = 0.0
    num_warmup = 0
    num_steady = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if batch_idx < num_warmup_batches:
            warmup_time += elapsed
            num_warmup += 1
        else:
            steady_time += elapsed
            num_steady += 1

    return warmup_time, steady_time, num_warmup, num_steady


def benchmark_config(
    name: str,
    model_name: str,
    compile_mode: str | None,
    dynamic: bool,
    docs: list[str],
    tokenizer,
    max_length: int,
    batch_tokens: int,
    num_warmup_batches: int,
    device: str = "cuda",
) -> BenchmarkResult:
    """Benchmark a single configuration."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {name}")
    print(f"  compile_mode={compile_mode}, dynamic={dynamic}")
    print(f"{'=' * 60}")

    clear_cuda_cache()

    # Create model
    print("Creating model...")
    model = create_model(model_name, compile_mode, dynamic, device)

    # Prepare data
    print("Preparing data...")
    inputs, loader, total_tokens = prepare_data(docs, tokenizer, max_length, batch_tokens)
    num_docs = len(docs)

    # Run inference
    print(f"Running inference ({len(loader)} batches, {num_warmup_batches} warmup)...")
    warmup_time, steady_time, num_warmup, num_steady = run_inference(
        model, loader, device, num_warmup_batches
    )

    total_time = warmup_time + steady_time

    # Calculate throughput (using steady-state only for fair comparison)
    if num_steady > 0:
        # Estimate docs/tokens processed in steady state
        steady_fraction = num_steady / (num_warmup + num_steady)
        steady_docs = num_docs * steady_fraction
        steady_tokens = total_tokens * steady_fraction
        docs_per_second = steady_docs / steady_time if steady_time > 0 else 0
        tokens_per_second = steady_tokens / steady_time if steady_time > 0 else 0
    else:
        docs_per_second = num_docs / total_time if total_time > 0 else 0
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    result = BenchmarkResult(
        name=name,
        warmup_time=warmup_time,
        steady_state_time=steady_time,
        total_time=total_time,
        num_warmup_batches=num_warmup,
        num_steady_batches=num_steady,
        docs_per_second=docs_per_second,
        tokens_per_second=tokens_per_second,
    )

    # Cleanup
    del model
    clear_cuda_cache()

    return result


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Configuration':<35} {'Warmup':>10} {'Steady':>10} {'Total':>10} {'Tok/s':>12}")
    print(f"{'':35} {'(sec)':>10} {'(sec)':>10} {'(sec)':>10} {'(steady)':>12}")
    print("-" * 80)

    # Find baseline for comparison
    baseline = next((r for r in results if r.name == "No compilation"), results[0])

    for r in results:
        print(
            f"{r.name:<35} "
            f"{r.warmup_time:>10.2f} "
            f"{r.steady_state_time:>10.2f} "
            f"{r.total_time:>10.2f} "
            f"{r.tokens_per_second:>12,.0f}"
        )

    print("-" * 80)
    print("\nSpeedups vs baseline (total time):")
    for r in results:
        if r != baseline:
            speedup = baseline.total_time / r.total_time if r.total_time > 0 else 0
            print(f"  {r.name}: {speedup:.2f}x")

    print("\nSteady-state throughput comparison:")
    for r in results:
        if r != baseline:
            speedup = (
                r.tokens_per_second / baseline.tokens_per_second
                if baseline.tokens_per_second > 0
                else 0
            )
            print(f"  {r.name}: {speedup:.2f}x tokens/sec")


def plot_results(results: list[BenchmarkResult], output_dir: Path):
    """Generate and save benchmark visualization figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results]
    warmup_times = [r.warmup_time for r in results]
    steady_times = [r.steady_state_time for r in results]
    throughputs = [r.tokens_per_second / 1000 for r in results]  # Convert to k tokens/s

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.tab10.colors

    # Figure 1: Stacked bar chart of warmup vs steady-state time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.6

    ax1.bar(x, warmup_times, width, label="Warmup", color=colors[0])
    ax1.bar(x, steady_times, width, bottom=warmup_times, label="Steady-state", color=colors[1])

    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Compilation Mode: Warmup vs Steady-State Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.legend()

    # Add total time labels on top of bars
    for i, (w, s) in enumerate(zip(warmup_times, steady_times, strict=False)):
        ax1.annotate(
            f"{w + s:.2f}s",
            xy=(i, w + s),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig1.savefig(output_dir / "compile_time_breakdown.png", dpi=150)
    print(f"Saved: {output_dir / 'compile_time_breakdown.png'}")

    # Figure 2: Throughput comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    bars = ax2.bar(x, throughputs, width, color=colors[2])

    # Highlight the best throughput
    max_idx = np.argmax(throughputs)
    bars[max_idx].set_color(colors[3])

    ax2.set_ylabel("Throughput (k tokens/sec)")
    ax2.set_title("Compilation Mode: Steady-State Throughput")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")

    # Add value labels on bars
    for i, v in enumerate(throughputs):
        ax2.annotate(
            f"{v:.1f}k",
            xy=(i, v),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    fig2.savefig(output_dir / "compile_throughput.png", dpi=150)
    print(f"Saved: {output_dir / 'compile_throughput.png'}")

    # Figure 3: Speedup vs baseline (total time)
    baseline = next((r for r in results if r.name == "No compilation"), results[0])
    speedups = [baseline.total_time / r.total_time for r in results]

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    bar_colors = [colors[3] if s > 1.0 else colors[0] for s in speedups]
    bars = ax3.bar(x, speedups, width, color=bar_colors)

    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1.0x)")
    ax3.set_ylabel("Speedup (vs no compilation)")
    ax3.set_title("Compilation Mode: Total Time Speedup")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=30, ha="right")

    # Add value labels
    for i, v in enumerate(speedups):
        ax3.annotate(
            f"{v:.2f}x",
            xy=(i, v),
            ha="center",
            va="bottom" if v > 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    fig3.savefig(output_dir / "compile_speedup.png", dpi=150)
    print(f"Saved: {output_dir / 'compile_speedup.png'}")

    # Figure 4: Combined summary figure
    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Time breakdown
    axes[0].bar(x, warmup_times, width, label="Warmup", color=colors[0])
    axes[0].bar(x, steady_times, width, bottom=warmup_times, label="Steady", color=colors[1])
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Time Breakdown")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)

    # Panel 2: Throughput
    bars2 = axes[1].bar(x, throughputs, width, color=colors[2])
    bars2[max_idx].set_color(colors[3])
    axes[1].set_ylabel("k tokens/sec")
    axes[1].set_title("Steady-State Throughput")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    # Panel 3: Speedup
    bar_colors = [colors[3] if s > 1.0 else colors[0] for s in speedups]
    axes[2].bar(x, speedups, width, color=bar_colors)
    axes[2].axhline(y=1.0, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Speedup")
    axes[2].set_title("Total Time Speedup")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    fig4.savefig(output_dir / "compile_summary.png", dpi=150)
    print(f"Saved: {output_dir / 'compile_summary.png'}")

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile configurations")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to benchmark",
    )
    parser.add_argument("--num-docs", type=int, default=500, help="Number of documents to process")
    parser.add_argument("--batch-tokens", type=int, default=16384, help="Max tokens per batch")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Number of warmup batches")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "none",
            "default",
            "default-dynamic",
            "reduce-overhead",
            "reduce-overhead-dynamic",
        ],
        help="Configurations to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save figures",
    )
    args = parser.parse_args()

    # Configuration mapping
    config_map = {
        "none": ("No compilation", None, False),
        "default": ("default (static)", "default", False),
        "default-dynamic": ("default (dynamic=True)", "default", True),
        "reduce-overhead": ("reduce-overhead (static)", "reduce-overhead", False),
        "reduce-overhead-dynamic": ("reduce-overhead (dynamic=True)", "reduce-overhead", True),
        "max-autotune": ("max-autotune (static)", "max-autotune", False),
        "max-autotune-dynamic": ("max-autotune (dynamic=True)", "max-autotune", True),
    }

    print(f"Model: {args.model}")
    print(f"Documents: {args.num_docs}")
    print(f"Batch tokens: {args.batch_tokens}")
    print(f"Warmup batches: {args.warmup_batches}")
    print(f"Configurations: {args.configs}")

    # Load tokenizer and documents once
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_length = tokenizer.model_max_length
    if max_length > 10000:
        max_length = 512

    docs = load_sample_docs(args.num_docs)

    # Run benchmarks
    results = []
    for config_key in args.configs:
        if config_key not in config_map:
            print(f"Unknown config: {config_key}, skipping")
            continue

        name, compile_mode, dynamic = config_map[config_key]
        result = benchmark_config(
            name=name,
            model_name=args.model,
            compile_mode=compile_mode,
            dynamic=dynamic,
            docs=docs,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_tokens=args.batch_tokens,
            num_warmup_batches=args.warmup_batches,
        )
        results.append(result)

    print_results(results)
    plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
