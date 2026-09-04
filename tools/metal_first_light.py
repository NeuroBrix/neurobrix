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

* **fp32 is compared BIT-EXACTLY.** RMSNorm in fp32 is a deterministic
  sequence of IEEE operations; a correct port reproduces it exactly. "Close"
  is not the bar for a first-light check — a backend that is merely close is
  a backend whose reduction order or accumulation width differs, and that is
  precisely what this exists to detect.
* **fp16 is drift-gated**, because the accumulation happens in fp32 and the
  final narrowing can legitimately differ by an ulp.

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
            got, device = _run_kernel(x, w)

            if x.dtype == np.float32:
                identical = np.array_equal(got, expected)
                verdict = "BIT-IDENTICAL" if identical else "DIVERGED"
                if not identical:
                    failures.append((key, float(np.abs(got - expected).max())))
            else:
                diff = float(np.abs(got.astype(np.float64)
                                    - expected.astype(np.float64)).max())
                scale = float(np.abs(expected.astype(np.float64)).max()) or 1.0
                ok = diff / scale < 1e-3
                verdict = f"drift {diff/scale:.2e}" + ("" if ok else "  OVER BAR")
                if not ok:
                    failures.append((key, diff / scale))
            print(f"  {key:24} on {device:5}  {verdict}")

    print()
    if failures:
        print(f"FIRST LIGHT FAILED on {len(failures)} case(s):")
        for key, value in failures:
            print(f"  {key}: {value:.3e}")
        print("\nThat is the port's first bug report: the shape and dtype above "
              "are where\nthe backend's reduction differs from ours.")
        return 1
    print("FIRST LIGHT PASSED — the kernel produces our result on this device.")
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
