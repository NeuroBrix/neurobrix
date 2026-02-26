#!/usr/bin/env python3
"""
DtypeEngine Overhead Profiler for NeuroBrix.

Runs a diffusion model (default: Sana 1600M) and measures:
1. Baseline timing (current DtypeEngine with all protections)
2. Per-wrapper overhead breakdown (clamp, contiguous, cast counts)
3. CUDA kernel launch count via torch.profiler

Usage:
    PYTHONPATH=src python benchmarks/profile_dtype_overhead.py
    PYTHONPATH=src python benchmarks/profile_dtype_overhead.py --model Sana_1600M_1024px_MultiLing --steps 5
    PYTHONPATH=src python benchmarks/profile_dtype_overhead.py --full  # 20 steps for real benchmark

Output: Detailed breakdown of DtypeEngine overhead per wrapper type.
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.cuda

# ============================================================================
# INSTRUMENTATION — Monkey-patch DtypeEngine to count extra operations
# ============================================================================

class DtypeOverheadCounter:
    """Counts extra CUDA operations injected by DtypeEngine wrappers."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.fp32_cast_count = 0        # .float() calls
        self.fp32_contiguous_count = 0   # .contiguous() calls in fp32 wrapper
        self.fp32_contiguous_noop = 0    # .contiguous() that were no-ops
        self.fp16_downcast_count = 0     # _safe_downcast calls
        self.fp16_clamp_pre_count = 0    # .clamp() before .to() in _safe_downcast
        self.fp16_clamp_post_count = 0   # .clamp() after matmul result
        self.overflow_clamp_count = 0    # .clamp() in overflow protect (add/sub)
        self.overflow_clamp_noop = 0     # overflow clamp on non-fp16 (no-op)
        self.total_op_calls = 0          # total compiled op executions

    def summary(self):
        total_extra = (
            self.fp32_cast_count +
            self.fp32_contiguous_count - self.fp32_contiguous_noop +
            self.fp16_clamp_pre_count +
            self.fp16_clamp_post_count +
            self.overflow_clamp_count - self.overflow_clamp_noop
        )
        return {
            "total_compiled_ops": self.total_op_calls,
            "extra_cuda_kernels": total_extra,
            "overhead_pct": f"{total_extra / max(self.total_op_calls, 1) * 100:.1f}%",
            "fp32_wrapper": {
                "float_casts": self.fp32_cast_count,
                "contiguous_calls": self.fp32_contiguous_count,
                "contiguous_noops": self.fp32_contiguous_noop,
                "effective_contiguous": self.fp32_contiguous_count - self.fp32_contiguous_noop,
            },
            "fp16_wrapper": {
                "downcast_calls": self.fp16_downcast_count,
                "pre_clamp": self.fp16_clamp_pre_count,
                "post_clamp": self.fp16_clamp_post_count,
            },
            "overflow_protect": {
                "clamp_calls": self.overflow_clamp_count,
                "noop_calls": self.overflow_clamp_noop,
                "effective_clamps": self.overflow_clamp_count - self.overflow_clamp_noop,
            },
        }

# Global counter
COUNTER = DtypeOverheadCounter()


def patch_dtype_engine():
    """
    Monkey-patch DtypeEngine to instrument all wrappers.

    This MUST be called before any DtypeEngine is instantiated.
    """
    from neurobrix.core.dtype.engine import DtypeEngine, _safe_downcast, _DTYPE_MAX

    original_make_fp32 = DtypeEngine._make_fp32_wrapper
    original_make_lower = DtypeEngine._make_lower_precision_wrapper
    original_make_overflow = DtypeEngine._make_overflow_protect_wrapper

    def instrumented_fp32_wrapper(self, func):
        original_wrapper = original_make_fp32(self, func)

        def counted_fp32(*args, **kwargs):
            for a in args:
                if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != torch.float32:
                    COUNTER.fp32_cast_count += 1
                if isinstance(a, torch.Tensor):
                    COUNTER.fp32_contiguous_count += 1
                    if a.is_contiguous():
                        COUNTER.fp32_contiguous_noop += 1
            return original_wrapper(*args, **kwargs)
        return counted_fp32

    def instrumented_lower_wrapper(self, func):
        original_wrapper = original_make_lower(self, func)
        compute = self.compute_dtype
        needs_inf_fix = (compute == torch.float16)

        def counted_lower(*args, **kwargs):
            for a in args:
                if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != compute:
                    COUNTER.fp16_downcast_count += 1
                    # _safe_downcast does clamp if fp16 target and different dtype
                    max_val = _DTYPE_MAX.get(compute)
                    if max_val is not None:
                        COUNTER.fp16_clamp_pre_count += 1
            result = original_wrapper(*args, **kwargs)
            if needs_inf_fix and isinstance(result, torch.Tensor) and result.dtype == torch.float16:
                COUNTER.fp16_clamp_post_count += 1
            return result
        return counted_lower

    def instrumented_overflow_wrapper(self, func):
        original_wrapper = original_make_overflow(self, func)

        def counted_overflow(*args, **kwargs):
            result = original_wrapper(*args, **kwargs)
            COUNTER.overflow_clamp_count += 1
            if not (isinstance(result, torch.Tensor) and result.dtype == torch.float16):
                COUNTER.overflow_clamp_noop += 1
            return result
        return counted_overflow

    DtypeEngine._make_fp32_wrapper = instrumented_fp32_wrapper
    DtypeEngine._make_lower_precision_wrapper = instrumented_lower_wrapper
    DtypeEngine._make_overflow_protect_wrapper = instrumented_overflow_wrapper
    print("[PROFILER] DtypeEngine instrumented for overhead counting")


def patch_compiled_sequence_counter():
    """Patch CompiledSequence.run to count total ops per execution."""
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence

    original_run_inner = CompiledSequence._run_inner

    def counted_run_inner(self, arena, debug=False):
        COUNTER.total_op_calls += len(self._ops)
        return original_run_inner(self, arena, debug)

    CompiledSequence._run_inner = counted_run_inner
    print("[PROFILER] CompiledSequence instrumented for op counting")


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(model_name: str, hardware: str, steps: int, prompt: str):
    """Run a full inference benchmark with profiling."""
    from neurobrix.nbx import NBXContainer
    from neurobrix.core.prism import PrismSolver, load_profile, InputConfig
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.executor import RuntimeExecutor
    from neurobrix.cli.commands.run import find_model

    print(f"\n{'='*70}")
    print(f"  NeuroBrix DtypeEngine Overhead Profiler")
    print(f"{'='*70}")
    print(f"  Model:    {model_name}")
    print(f"  Hardware: {hardware}")
    print(f"  Steps:    {steps}")
    print(f"  Prompt:   {prompt[:50]}...")
    print(f"{'='*70}\n")

    # Load model
    nbx_path = find_model(model_name)
    container = NBXContainer.load(str(nbx_path))
    manifest = container.get_manifest() or {}

    # Prism allocation
    hw_profile = load_profile(hardware)
    cache_path = container._cache_path
    defaults_path = cache_path / "runtime" / "defaults.json"
    cached_defaults = {}
    if defaults_path.exists():
        with open(defaults_path) as f:
            cached_defaults = json.load(f)

    height = cached_defaults.get("height", 1024)
    width = cached_defaults.get("width", 1024)
    vae_scale = cached_defaults.get("vae_scale_factor", 8)

    input_config = InputConfig(
        batch_size=2, height=height, width=width,
        dtype="float16", vae_scale=vae_scale,
    )
    solver = PrismSolver()
    execution_plan = solver.solve_smart(container, hw_profile, input_config)
    print(f"[PRISM] Strategy: {execution_plan.strategy}")

    # Load runtime
    loader = NBXRuntimeLoader()
    pkg = loader.load(str(nbx_path))

    inputs = {
        "global.prompt": prompt,
        "global.num_inference_steps": steps,
        "global.seed": 42,
    }
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    executor = RuntimeExecutor(pkg, execution_plan, mode="compiled")

    # ====================================================================
    # BENCHMARK 1: Warm-up (first run compiles the sequence)
    # ====================================================================
    print("\n[WARMUP] First run (includes compilation)...")
    COUNTER.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = executor.execute(inputs)
    torch.cuda.synchronize()
    t_warmup = time.perf_counter() - t0
    warmup_counts = COUNTER.summary()
    print(f"[WARMUP] {t_warmup:.2f}s (includes compilation)")
    _print_counter_summary("WARMUP", warmup_counts)

    # Save output if it's an image
    _save_output(outputs, "benchmark_warmup.png")

    # ====================================================================
    # BENCHMARK 2: Hot run (compiled sequence already cached)
    # ====================================================================
    print("\n[HOT RUN] Second run (pre-compiled)...")
    COUNTER.reset()

    # Reset seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = executor.execute(inputs)
    torch.cuda.synchronize()
    t_hot = time.perf_counter() - t0
    hot_counts = COUNTER.summary()
    print(f"[HOT RUN] {t_hot:.2f}s")
    _print_counter_summary("HOT RUN", hot_counts)

    # ====================================================================
    # BENCHMARK 3: CUDA profiler trace (optional)
    # ====================================================================
    print("\n[PROFILER] Capturing CUDA trace...")
    COUNTER.reset()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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
        outputs = executor.execute(inputs)

    torch.cuda.synchronize()

    # Export trace
    trace_path = profile_dir / f"trace_{model_name}_{steps}steps.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"[PROFILER] Trace saved: {trace_path}")

    # Print CUDA kernel summary
    print("\n[PROFILER] Top 30 CUDA kernels by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Count total CUDA kernel launches
    all_events = prof.key_averages()
    total_cuda_calls = sum(e.count for e in all_events if e.self_cuda_time_total > 0)
    total_cuda_time = sum(e.self_cuda_time_total for e in all_events)
    print(f"\n[PROFILER] Total CUDA kernel launches: {total_cuda_calls}")
    print(f"[PROFILER] Total CUDA time: {total_cuda_time / 1000:.1f}ms")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"  OVERHEAD SUMMARY")
    print(f"{'='*70}")
    print(f"  Warmup time:        {t_warmup:.2f}s")
    print(f"  Hot run time:       {t_hot:.2f}s")
    print(f"  Compiled ops/run:   {hot_counts['total_compiled_ops']}")
    print(f"  Extra CUDA kernels: {hot_counts['extra_cuda_kernels']}")
    print(f"  Overhead ratio:     {hot_counts['overhead_pct']}")
    print(f"")
    print(f"  Total CUDA calls:   {total_cuda_calls}")
    print(f"  Total CUDA time:    {total_cuda_time / 1000:.1f}ms")
    print(f"")
    print(f"  Breakdown:")
    print(f"    FP32 .float() casts:          {hot_counts['fp32_wrapper']['float_casts']}")
    print(f"    FP32 .contiguous() effective:  {hot_counts['fp32_wrapper']['effective_contiguous']}")
    print(f"    FP16 pre-matmul .clamp():      {hot_counts['fp16_wrapper']['pre_clamp']}")
    print(f"    FP16 post-matmul .clamp():     {hot_counts['fp16_wrapper']['post_clamp']}")
    print(f"    Overflow protect .clamp():     {hot_counts['overflow_protect']['effective_clamps']}")
    print(f"{'='*70}")

    return {
        "model": model_name,
        "steps": steps,
        "warmup_s": t_warmup,
        "hot_s": t_hot,
        "counts": hot_counts,
        "total_cuda_calls": total_cuda_calls,
        "total_cuda_time_ms": total_cuda_time / 1000,
    }


def _print_counter_summary(label, counts):
    print(f"  [{label}] Compiled ops: {counts['total_compiled_ops']}, "
          f"Extra CUDA: {counts['extra_cuda_kernels']} ({counts['overhead_pct']})")
    print(f"    FP32 casts={counts['fp32_wrapper']['float_casts']}, "
          f"contiguous(eff)={counts['fp32_wrapper']['effective_contiguous']}")
    print(f"    FP16 pre-clamp={counts['fp16_wrapper']['pre_clamp']}, "
          f"post-clamp={counts['fp16_wrapper']['post_clamp']}")
    print(f"    Overflow clamp(eff)={counts['overflow_protect']['effective_clamps']}")


def _save_output(outputs, filename):
    """Save output image if applicable."""
    try:
        from neurobrix.core.config import get_output_processing
        output_tensor = None
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor) and val.ndim == 4:
                output_tensor = val
                break
        if output_tensor is not None:
            from torchvision.utils import save_image
            img = output_tensor.float().clamp(0, 1)
            path = Path("/home/mlops/NeuroBrix_System") / filename
            save_image(img, str(path))
            print(f"  Output saved: {path}")
    except Exception:
        pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Profile DtypeEngine overhead")
    parser.add_argument("--model", default="Sana_1600M_1024px_MultiLing",
                        help="Model name in cache")
    parser.add_argument("--hardware", default="v100-32g",
                        help="Hardware profile")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of diffusion steps (5 for quick test, 20 for real benchmark)")
    parser.add_argument("--full", action="store_true",
                        help="Full 20-step benchmark")
    parser.add_argument("--prompt", default="A cat sitting on a windowsill at sunset",
                        help="Generation prompt")
    args = parser.parse_args()

    if args.full:
        args.steps = 20

    # Instrument BEFORE any imports that create DtypeEngine instances
    patch_dtype_engine()
    patch_compiled_sequence_counter()

    results = run_benchmark(args.model, args.hardware, args.steps, args.prompt)

    # Save results as JSON
    results_path = Path("/home/mlops/NeuroBrix_System/benchmarks/results")
    results_path.mkdir(parents=True, exist_ok=True)
    out_file = results_path / f"dtype_overhead_{args.model}_{args.steps}steps.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
