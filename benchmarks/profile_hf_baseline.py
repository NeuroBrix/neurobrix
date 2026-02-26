#!/usr/bin/env python3
"""
HuggingFace Diffusers Baseline Benchmark.

Runs the same model through HuggingFace's native pipeline to establish
the performance baseline we need to match or beat.

Usage:
    python benchmarks/profile_hf_baseline.py
    python benchmarks/profile_hf_baseline.py --steps 20 --full

Compares: NeuroBrix compiled mode vs HuggingFace diffusers (native pipeline)
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.cuda


def run_sana_hf(steps: int, prompt: str, warmup: bool = True):
    """Run Sana through HuggingFace diffusers pipeline."""
    print(f"\n{'='*70}")
    print(f"  HuggingFace Diffusers Baseline — Sana 1600M")
    print(f"{'='*70}")
    print(f"  Steps:  {steps}")
    print(f"  Prompt: {prompt[:50]}...")
    print(f"{'='*70}\n")

    from diffusers import SanaPipeline

    # Use local snapshot if available, otherwise try Hub
    local_path = Path("/home/mlops/hf_snapshots/Sana_1600M_1024px_MultiLing")
    model_id = str(local_path) if local_path.exists() else "Efficient-Large-Model/Sana_1600M_1024px_MultiLing_BF16"

    print(f"[HF] Loading Sana pipeline from: {model_id}")
    t0 = time.perf_counter()
    pipe = SanaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    # V100-32G can't fit all components at once — use CPU offload like NeuroBrix does
    pipe.enable_model_cpu_offload()
    t_load = time.perf_counter() - t0
    print(f"[HF] Pipeline loaded: {t_load:.2f}s")

    # Warmup run
    if warmup:
        print("\n[HF WARMUP] First run...")
        torch.manual_seed(42)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        torch.cuda.synchronize()
        t_warmup = time.perf_counter() - t0
        print(f"[HF WARMUP] {t_warmup:.2f}s")
        image.save("/home/mlops/NeuroBrix_System/benchmark_hf_warmup.png")
    else:
        t_warmup = 0.0

    # Hot run
    print("\n[HF HOT] Second run...")
    torch.manual_seed(42)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    torch.cuda.synchronize()
    t_hot = time.perf_counter() - t0
    print(f"[HF HOT] {t_hot:.2f}s")
    image.save("/home/mlops/NeuroBrix_System/benchmark_hf_hot.png")

    # Profiled run
    print("\n[HF PROFILER] Capturing CUDA trace...")
    torch.manual_seed(42)

    profile_dir = Path("/home/mlops/NeuroBrix_System/benchmarks/traces")
    profile_dir.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

    torch.cuda.synchronize()

    trace_path = profile_dir / f"trace_hf_sana_{steps}steps.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"[HF PROFILER] Trace saved: {trace_path}")

    print("\n[HF PROFILER] Top 30 CUDA kernels by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    all_events = prof.key_averages()
    total_cuda_calls = sum(e.count for e in all_events if e.self_cuda_time_total > 0)
    total_cuda_time = sum(e.self_cuda_time_total for e in all_events)

    print(f"\n[HF PROFILER] Total CUDA kernel launches: {total_cuda_calls}")
    print(f"[HF PROFILER] Total CUDA time: {total_cuda_time / 1000:.1f}ms")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    results = {
        "engine": "huggingface_diffusers",
        "model": "Sana_1600M_1024px_MultiLing",
        "steps": steps,
        "load_s": t_load,
        "warmup_s": t_warmup,
        "hot_s": t_hot,
        "total_cuda_calls": total_cuda_calls,
        "total_cuda_time_ms": total_cuda_time / 1000,
    }

    print(f"\n{'='*70}")
    print(f"  HF BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  Load time:          {t_load:.2f}s")
    print(f"  Warmup time:        {t_warmup:.2f}s")
    print(f"  Hot run time:       {t_hot:.2f}s")
    print(f"  CUDA kernel count:  {total_cuda_calls}")
    print(f"  CUDA total time:    {total_cuda_time / 1000:.1f}ms")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HuggingFace baseline benchmark")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--full", action="store_true", help="20 steps")
    parser.add_argument("--prompt", default="A cat sitting on a windowsill at sunset")
    args = parser.parse_args()

    if args.full:
        args.steps = 20

    results = run_sana_hf(args.steps, args.prompt)

    results_path = Path("/home/mlops/NeuroBrix_System/benchmarks/results")
    results_path.mkdir(parents=True, exist_ok=True)
    out_file = results_path / f"hf_baseline_sana_{args.steps}steps.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
