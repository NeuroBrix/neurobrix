#!/usr/bin/env python3
"""Metal first light: one NeuroBrix kernel, compared against its CUDA reference.

The milestone of the Metal adoption plan is deliberately small — *one kernel
of the family, unmodified, producing on an Apple GPU what it produces on
CUDA*. It is the right first step because it needs no engine change at all:
R33 kept `torch` out of the triton tree, so our 424 `@triton.jit` kernels are
portable **as source**, and the only question is whether the backend compiles
and runs them correctly.

`rms_norm` is the kernel chosen: elementwise work plus a reduction, both
fully covered by the Metal backend's published surface, and used by nearly
every model in the catalogue.

This has two halves because the two machines are different machines:

    # on the CUDA box, today
    python3 tools/metal_first_light.py record --out metal_reference.npz

    # on the Apple machine, at first light
    python3 tools/metal_first_light.py compare --ref metal_reference.npz

`record` runs the kernel and writes inputs AND outputs. `compare` re-runs it
on whatever device it finds and holds the result to the same bar:

* **fp32 was compared BIT-EXACTLY** until 2026-09-05, on the reasoning that
  RMSNorm in fp32 is a deterministic sequence of IEEE operations that a
  correct port reproduces exactly. **That premise was wrong**, and the first
  Apple GPU showed why: it holds only for a FIXED summation order, and
  `tl.sum` does not promise one across targets. Metal reduces with
  `simd_sum` per simdgroup and then a threadgroup pass; CUDA's tree groups
  differently; floating-point addition is not associative. Measured: three of
  four shapes diverged by 4.8e-07 to 9.5e-07, while BOTH sides sat within
  3 ULP of an exact fp64 oracle — and at one shape Metal was CLOSER to the
  truth than the CUDA reference. Bit-identity was asking for a coincidence of
  reduction trees, not for correctness.

  The bar since is two things a correct port must satisfy and a merely-close
  one cannot fake:

  1. **Determinism**, run to run: five executions per shape must produce one
     distinct byte pattern. A differing tree is a fixed property of the
     target; a race is not, and this separates them.
  2. **No worse than the reference**, in ULP against an fp64 oracle. The
     bound is READ from the reference — whatever distance CUDA achieved for
     that shape is the bar — never written into this file. A port that lands
     closer to the truth than CUDA passes, which is the correct answer and
     the old bar called it a failure.

* **fp16 is drift-gated**, unchanged, because the accumulation happens in
  fp32 and the final narrowing can legitimately differ by an ulp.

A failure here is information, not a defeat: it names which shape and which
dtype diverged, which is the first line of the port's bug report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Shapes chosen to exercise the kernel's regimes rather than one happy case:
# a feature dim below one block, one above it, a long-batch case, and a
# head-dim-shaped case.
SHAPES = [
    (4, 128),
    (2, 4096),
    (1024, 64),
    (8, 1536),
]
DTYPES = ["float32", "float16"]
SEED = 20260903


def _make_inputs(shape, dtype):
    rng = np.random.RandomState(SEED + shape[0] * 31 + shape[1])
    x = rng.randn(*shape).astype(dtype)
    weight = rng.randn(shape[-1]).astype(dtype)
    return x, weight


def _run_kernel(x_np, w_np, eps=1e-6):
    """Run the engine's own rms_norm through the NBXTensor path.

    NBXTensor deliberately, not torch: mode 2 is what ports to Metal, and it
    is the path whose kernels are portable as source. Comparing the torch
    path would measure PyTorch's backend, not ours.
    """
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels.wrappers import rms_norm

    x = NBXTensor.from_numpy(np.ascontiguousarray(x_np))
    w = NBXTensor.from_numpy(np.ascontiguousarray(w_np))
    out = rms_norm(x, w, eps=eps)
    return np.asarray(out.numpy()), _device_label()


def _device_label() -> str:
    """What ran it, for the record — never for a decision."""
    try:
        import triton
        return str(triton.runtime.driver.active.get_current_target().backend)
    except Exception:
        return "unknown"


def cmd_record(args) -> int:
    payload = {}
    for dtype in DTYPES:
        for shape in SHAPES:
            key = f"{dtype}_{shape[0]}x{shape[1]}"
            x, w = _make_inputs(shape, dtype)
            out, device = _run_kernel(x, w)
            payload[f"{key}__x"] = x
            payload[f"{key}__w"] = w
            payload[f"{key}__out"] = out
            print(f"  recorded {key:24} on {device}  "
                  f"out[0,:3]={out.reshape(-1)[:3]}")
    payload["__device"] = np.str_(_device_label())
    out_path = Path(args.out)
    with open(out_path, "wb") as fh:
        np.savez(fh, **payload)
    print(f"\nreference written: {out_path} ({out_path.stat().st_size:,} bytes)")
    print("Copy it to the Apple machine and run:")
    print(f"  python3 tools/metal_first_light.py compare --ref {out_path.name}")
    return 0


# Five, because the property being tested is "one distinct result", and a
# single repeat cannot distinguish a stable answer from a lucky one.
_DETERMINISM_RUNS = 5

# The fp16 bar, unchanged.
_FP16_DRIFT_BAR = 1e-3


def _oracle(x, w, eps=1e-6):
    """RMSNorm computed in float64 — the truth both sides are measured against.

    Not a third implementation to be compared with: an oracle. It is the same
    arithmetic the kernel performs, carried out with enough precision that its
    own rounding is far below an fp32 ulp.
    """
    x64 = x.astype(np.float64)
    mean_square = (x64 * x64).sum(axis=1) / x.shape[1]
    return x64 * (1.0 / np.sqrt(mean_square + eps))[:, None] * w.astype(np.float64)


def _ulp_distance(a, b):
    """Distance in fp32 ulps, elementwise, as a monotone integer.

    Reinterpreting a float's bits as an int makes adjacent representable
    values adjacent integers, so subtracting them counts representable steps.
    Negative floats are folded so the ordering stays monotone across zero.
    """
    def ordered(v):
        i = np.asarray(v, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(i < 0, np.int64(0x80000000) - i, i)

    return np.abs(ordered(a) - ordered(b))


def cmd_compare(args) -> int:
    with np.load(args.ref, allow_pickle=False) as data:
        recorded_on = str(data["__device"])
        keys = sorted({k.split("__")[0] for k in data.files
                       if "__" in k and not k.startswith("__")})
        failures = []
        print(f"reference recorded on: {recorded_on}\n")
        for key in keys:
            x, w = data[f"{key}__x"], data[f"{key}__w"]
            expected = data[f"{key}__out"]

            runs = []
            for _ in range(_DETERMINISM_RUNS):
                out, device = _run_kernel(x, w)
                runs.append(out)
            distinct = {r.tobytes() for r in runs}
            got = runs[0]

            if len(distinct) != 1:
                failures.append(
                    (key, f"{len(distinct)} distinct results in "
                          f"{_DETERMINISM_RUNS} runs"))
                print(f"  {key:24} on {device:5}  NON-DETERMINISTIC "
                      f"({len(distinct)} distinct in {_DETERMINISM_RUNS})")
                continue

            if x.dtype == np.float32:
                truth = _oracle(x, w).astype(np.float32)
                ours = int(_ulp_distance(got, truth).max())
                theirs = int(_ulp_distance(expected, truth).max())
                identical = np.array_equal(got, expected)
                ok = ours <= theirs
                verdict = (f"det x{_DETERMINISM_RUNS}  ulp_vs_oracle "
                           f"{ours} (reference {theirs})")
                if identical:
                    verdict += "  BIT-IDENTICAL"
                if not ok:
                    verdict += "  WORSE THAN REFERENCE"
                    failures.append(
                        (key, f"{ours} ulp against the oracle, "
                              f"reference achieves {theirs}"))
            else:
                diff = float(np.abs(got.astype(np.float64)
                                    - expected.astype(np.float64)).max())
                scale = float(np.abs(expected.astype(np.float64)).max()) or 1.0
                drift = diff / scale
                ok = drift < _FP16_DRIFT_BAR
                verdict = (f"det x{_DETERMINISM_RUNS}  drift {drift:.2e}"
                           + ("" if ok else "  OVER BAR"))
                if not ok:
                    failures.append((key, f"drift {drift:.3e}"))
            print(f"  {key:24} on {device:5}  {verdict}")

    print()
    if failures:
        print(f"FIRST LIGHT FAILED on {len(failures)} case(s):")
        for key, value in failures:
            print(f"  {key}: {value}")
        print("\nThat is the port's first bug report: the shape and dtype above "
              "are where\nthis device is either unstable or further from the "
              "truth than CUDA is.")
        return 1
    print("FIRST LIGHT PASSED — deterministic at every shape, and no further "
          "from the\nfp64 oracle than the CUDA reference is. fp16 within the "
          "drift gate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="record the reference on this machine")
    rec.add_argument("--out", default="metal_reference.npz")
    cmp_ = sub.add_parser("compare", help="re-run and compare against a reference")
    cmp_.add_argument("--ref", default="metal_reference.npz")
    args = ap.parse_args()
    return cmd_record(args) if args.cmd == "record" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
