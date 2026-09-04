#!/usr/bin/env python3
"""What plan replay removes from a kernel launch — the number that judges tensor parallelism.

The collective-latency measurement (2026-09-03) put a cross-GPU all-reduce at
**124 us** with `device_us == host_issue_us` in every cell: the cost is host-side
API issue, not the wire. That verdict named its own condition for reopening —
"a decode loop whose kernel launches are REPLAYED rather than issued" — because
replay swaps Triton's Python dispatch band for direct C-launcher calls.

So the question this answers is narrow and decisive: **how much of the 16.4 us
per-launch dispatch does replay actually remove?** The collective's cost is
N launches plus events; if the launch term collapses, the collective collapses
with it, and tensor parallelism reopens with its direct-peer reduction already
written. If it does not, the wall is confirmed with figures and nobody returns
to it.

This measures the two paths on the SAME kernel, same grid, same arguments:

  JIT path     kernel[grid](args)      what the collective probe measured
  direct path  compiled.run(...)       what replay issues

    python3 tools/launch_band_probe.py [--iters 20000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import triton                                                    # noqa: E402
import triton.language as tl                                     # noqa: E402

from neurobrix.kernels.nbx_tensor import (                       # noqa: E402
    DeviceAllocator, NBXDtype, NBXTensor,
)

BLOCK = 1024


@triton.jit
def _bump(out_ptr, p0, p1, p2, p3, n, N_SRC: tl.constexpr, BLOCK: tl.constexpr):
    """The COLLECTIVE'S OWN kernel signature — four source pointers plus an
    output, summed in fixed order.

    Arity matters: the dispatch band scales with the number of arguments it has
    to bind and check, so timing a two-argument kernel and applying the result
    to a five-pointer collective would understate what replay removes. This is
    `_sum_fixed_order` from tools/collective_latency_probe.py, verbatim in
    shape, so the recomputation below uses like for like."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.load(p0 + offs, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(p1 + offs, mask=mask, other=0.0).to(tl.float32)
    if N_SRC > 2:
        acc += tl.load(p2 + offs, mask=mask, other=0.0).to(tl.float32)
    if N_SRC > 3:
        acc += tl.load(p3 + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(tl.float16), mask=mask)


def _capture_launch(fn):
    """Capture the CompiledKernel and its final launch tuple.

    Intercepted at `CompiledKernel.run`, which is exactly where replay records
    — so what is timed below is what replay actually issues, not an
    approximation of it.
    """
    from triton.compiler.compiler import CompiledKernel

    grabbed = {}
    original = CompiledKernel.run

    def _prop(self):
        raw = original.__get__(self) if isinstance(original, property) else original

        def _run(g0, g1, g2, stream, function, packed_metadata,
                 launch_md, enter_hook, exit_hook, *vals):
            grabbed.setdefault("k", (self, g0, g1, g2, stream, function,
                                     packed_metadata, tuple(vals)))
            return raw(g0, g1, g2, stream, function, packed_metadata,
                       launch_md, enter_hook, exit_hook, *vals)
        return _run

    CompiledKernel.run = property(_prop)
    try:
        fn()
    finally:
        CompiledKernel.run = original
    return grabbed.get("k")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    DeviceAllocator.set_device(args.device)
    DeviceAllocator.ensure_triton_device(args.device)

    n = 2048                                   # the decode all-reduce payload
    t = NBXTensor.zeros((n,), dtype=NBXDtype.float16,
                        device=f"cuda:{args.device}")
    grid = ((n + BLOCK - 1) // BLOCK,)

    a = NBXTensor.zeros((n,), dtype=NBXDtype.float16, device=f"cuda:{args.device}")
    b = NBXTensor.zeros((n,), dtype=NBXDtype.float16, device=f"cuda:{args.device}")
    c = NBXTensor.zeros((n,), dtype=NBXDtype.float16, device=f"cuda:{args.device}")
    call = lambda: _bump[grid](t, a, b, c, a, n, N_SRC=3, BLOCK=BLOCK)
    call()                                     # warm: compile + autotune
    DeviceAllocator.sync_device()

    grabbed = _capture_launch(call)
    if grabbed is None:
        print("could not intercept a launch", file=sys.stderr)
        return 2
    ck, g0, g1, g2, stream, function, packed_metadata, vals = grabbed
    raw = type(ck).__dict__["run"].__get__(ck) if isinstance(
        type(ck).__dict__.get("run"), property) else ck.run

    for _ in range(200):
        call()
    DeviceAllocator.sync_device()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        call()
    DeviceAllocator.sync_device()
    jit_us = (time.perf_counter() - t0) * 1e6 / args.iters

    flat = tuple(int(v.data_ptr()) if hasattr(v, "data_ptr") else v for v in vals)
    for _ in range(200):
        raw(g0, g1, g2, stream, function, packed_metadata, None, None, None, *flat)
    DeviceAllocator.sync_device()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        raw(g0, g1, g2, stream, function, packed_metadata, None, None, None, *flat)
    DeviceAllocator.sync_device()
    direct_us = (time.perf_counter() - t0) * 1e6 / args.iters

    print("=" * 74)
    print("LAUNCH BAND — what plan replay removes")
    print("=" * 74)
    print(f"  JIT dispatch   (issued)  : {jit_us:8.2f} us/launch")
    print(f"  direct C call  (replayed): {direct_us:8.2f} us/launch")
    print(f"  removed                  : {jit_us - direct_us:8.2f} us"
          f"   ({jit_us / max(direct_us, 1e-9):.1f}x)")
    print()

    # The collective, recomputed. Measured 2026-09-03 on 3 cards:
    #   direct_peer_reduce = 124 us, of which 3 Triton launches at 16.4 us.
    n_launch = 3
    measured = 124.0
    non_launch = measured - n_launch * jit_us
    replayed = non_launch + n_launch * direct_us
    print(f"  collective, measured 2026-09-03 (3 cards) : {measured:7.1f} us")
    print(f"    of which {n_launch} launches at {jit_us:.1f} us       : "
          f"{n_launch * jit_us:7.1f} us")
    print(f"    everything else (events, switches, copies): {non_launch:7.1f} us")
    print(f"  same collective with launches REPLAYED     : {replayed:7.1f} us")
    print()
    per_token = replayed * 96 / 1000.0
    print(f"  96 collectives per token                   : {per_token:7.2f} ms")
    print(f"  share of the measured 25.1 ms token        : {per_token / 25.1 * 100:7.1f} %")
    print()
    print("  Study scenarios were 3.8 / 5.7 / 11.5 %. The wall was 47 %.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
