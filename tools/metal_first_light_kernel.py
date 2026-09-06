#!/usr/bin/env python3
"""First light, measured at the KERNEL, when the engine's container cannot yet
reach the device.

This is **not** the milestone. The milestone is
`tools/metal_first_light.py compare`, which runs `rms_norm` through
`NBXTensor` — deliberately, because mode 2 is what ports to Metal. That tool
is the one that decides, and nothing here substitutes for it.

It exists because `NBXTensor` has no Metal `DeviceAllocator` yet (step 5 of
the adoption plan, deliberately not written blind), so `from_numpy` cannot
allocate on an Apple GPU and the milestone tool stops before the kernel is
ever reached. That leaves the question the milestone was asked to answer
—*does our rms_norm kernel produce on an Apple GPU the numbers it produces on
CUDA?*— unanswered for a reason that has nothing to do with the kernel.

So this harness asks exactly that one question and nothing more:

* it imports `rms_norm_forward_kernel` **unmodified** from
  `kernels/ops/rmsnorm.py` — the same decorated object the engine launches;
* it launches it with the same grid, the same `num_warps=4` and the same
  `scale_by_weight` the wrapper passes (`kernels/wrappers.py::rms_norm`);
* the buffers are plain torch tensors instead of NBXTensor, because a buffer
  is all Triton needs and the allocator is the thing that does not exist yet;
* it holds the result to the **same bar** as the milestone tool: fp32
  bit-exact, fp16 drift-gated at 1e-3 relative;
* it reads the SAME reference file recorded on the CUDA machine.

What a pass here means, stated narrowly: the kernel's arithmetic survives the
port. What it does NOT mean: that `--triton` runs on Apple, that NBXTensor
works, or that any Apple support exists. Those need the allocator and the
milestone tool.

R33 preserved — the engine tree stays torch-free; the torch import lives here,
in a Metal tool, exactly as `metal_first_light.py` keeps numpy here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The same bar as tools/metal_first_light.py.
_FP16_DRIFT_BAR = 1e-3

# The kernel's own signature, spelled once. Types are read from what the
# wrapper passes in `kernels/wrappers.py::rms_norm`: three buffers, four
# strides and two dims as i32, `eps` as a float, and the three constexprs.
_RMS_NORM_SIGNATURE = {
    "input_ptr": "*fp32", "weight_ptr": "*fp32", "output_ptr": "*fp32",
    "batch_dim": "i32", "feat_dim": "i32",
    "input_batch_stride": "i32", "input_feat_stride": "i32",
    "output_batch_stride": "i32", "output_feat_stride": "i32",
    "eps": "fp32",
    "scale_by_weight": "constexpr",
    "BLOCK_SIZE_BATCH": "constexpr", "BLOCK_SIZE_FEAT": "constexpr",
}


def cmd_compile_only(ref_path) -> int:
    """Can the milestone kernel be LOWERED, at the milestone's own shapes?

    A separate question from whether it runs, and answerable on a machine
    where nothing runs. Apple's offline shader compiler is needed only by the
    pipeline's last stage (`metallib`); everything that decides *coverage*
    happens in `msl`, one stage earlier. So this compiles `rms_norm` through
    the `msl` stage at the exact tile sizes the engine would choose for each
    shape in the reference file — `BLOCK_SIZE_BATCH` from
    `wrappers._batch_block`, `BLOCK_SIZE_FEAT` from `next_power_of_2(feat)` —
    and reports whether the backend can lower it.

    It exists because the compile census, which has to invent tile sizes for
    kernels whose real ones come from a runtime heuristic, refuses this kernel
    at ITS invented (64, 64) tile. That refusal says nothing about the
    milestone, and the difference is worth measuring rather than assuming.
    """
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.backend.driver import _detect_metal_arch

    from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel
    from neurobrix.kernels.wrappers import _batch_block

    jit_fn = rms_norm_forward_kernel.fn          # peel @triton.heuristics
    target = GPUTarget("metal", _detect_metal_arch(), 32)

    class _StopAfterMSL(Exception):
        pass

    original = MetalBackend.add_stages

    def stop_after_msl(self, stages, options, language=None):
        original(self, stages, options, language)
        msl_stage = stages["msl"]

        def wrapped(src, metadata):
            msl_stage(src, metadata)
            raise _StopAfterMSL()

        stages["msl"] = wrapped

    MetalBackend.add_stages = stop_after_msl
    print(f"target: {target}")
    print("compiling rms_norm through the `msl` stage only — no execution, "
          "and no offline shader compiler needed.\n")

    shapes = _shapes_in(ref_path)
    failures = []
    try:
        for batch_dim, feat_dim in shapes:
            bsb = _batch_block(batch_dim, feat_dim)
            bsf = triton.next_power_of_2(feat_dim)
            constexprs = {"scale_by_weight": True,
                          "BLOCK_SIZE_BATCH": bsb, "BLOCK_SIZE_FEAT": bsf}
            label = f"{batch_dim}x{feat_dim}"
            try:
                triton.compile(
                    ASTSource(fn=jit_fn, signature=_RMS_NORM_SIGNATURE,
                              constexprs=constexprs),
                    target=target, options={"num_warps": 4})
                verdict = "reached metallib (shader compiler present)"
            except _StopAfterMSL:
                verdict = "LOWERS to MSL"
            except Exception as exc:
                verdict = f"REFUSED — {type(exc).__name__}: " \
                          f"{str(exc).splitlines()[0][:110]}"
                failures.append((label, verdict))
            print(f"  {label:>12}  BLOCK_SIZE_BATCH={bsb:<5} "
                  f"BLOCK_SIZE_FEAT={bsf:<6} {verdict}")
    finally:
        MetalBackend.add_stages = original

    print()
    if failures:
        print(f"LOWERING FAILED on {len(failures)} shape(s). The milestone "
              f"cannot pass at these shapes even once the shader compiler is "
              f"installed.")
        return 1
    print("The milestone kernel LOWERS at every shape in the reference. "
          "Nothing in\nthe backend's coverage blocks it — what blocks it is "
          "the toolchain and the\nNBXTensor allocator, neither of which is "
          "about this kernel.")
    return 0


def _shapes_in(ref_path):
    """The distinct (batch, feat) shapes the reference file records."""
    with np.load(ref_path, allow_pickle=False) as data:
        shapes = set()
        for key in data.files:
            if key.startswith("__") or "__" not in key:
                continue
            head = key.split("__")[0]
            dims = head.split("_")[-1]
            if "x" in dims:
                b, f = dims.split("x")
                shapes.add((int(b), int(f)))
    return sorted(shapes)


def _run_kernel(x_np, w_np, device, eps=1e-6):
    """Launch the engine's own rms_norm kernel, unmodified, on `device`."""
    import torch
    import triton
    from neurobrix.kernels.ops.rmsnorm import rms_norm_forward_kernel
    from neurobrix.kernels.wrappers import _batch_block

    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    w = torch.from_numpy(np.ascontiguousarray(w_np)).to(device)
    out = torch.empty_like(x)

    batch_dim, feat_dim = x.shape
    bsb = _batch_block(batch_dim, feat_dim)
    grid = (triton.cdiv(batch_dim, bsb),)

    rms_norm_forward_kernel[grid](
        x, w,
        out,
        batch_dim, feat_dim,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        eps,
        scale_by_weight=True,
        num_warps=4,
    )
    return out.cpu().numpy()


def _device_label() -> str:
    try:
        import triton
        return str(triton.runtime.driver.active.get_current_target().backend)
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True,
                    help="the metal_reference.npz recorded on the CUDA box")
    ap.add_argument("--compile-only", action="store_true",
                    help="do not run anything: compile the kernel through the "
                         "`msl` lowering stage at the reference's own shapes. "
                         "Answers whether the backend CAN lower the milestone "
                         "kernel on a machine with no offline shader compiler.")
    ap.add_argument("--device", default="cpu",
                    help="torch device holding the buffers (cpu or mps). The "
                         "KERNEL runs on the Apple GPU either way — the Metal "
                         "backend compiles and dispatches it; this only says "
                         "where the buffers live.")
    args = ap.parse_args()

    if args.compile_only:
        return cmd_compile_only(args.ref)

    backend = _device_label()
    print(f"triton target backend : {backend}")
    print(f"buffers on            : {args.device}")
    if backend != "metal":
        print(f"\nREFUSING: the active Triton target is '{backend}', not "
              f"'metal'. Comparing against the CUDA reference on any other "
              f"target would measure nothing about the port.", file=sys.stderr)
        return 2

    with np.load(args.ref, allow_pickle=False) as data:
        recorded_on = str(data["__device"])
        keys = sorted({k.split("__")[0] for k in data.files
                       if "__" in k and not k.startswith("__")})
        print(f"reference recorded on : {recorded_on}\n")

        failures = []
        for key in keys:
            x, w = data[f"{key}__x"], data[f"{key}__w"]
            expected = data[f"{key}__out"]
            try:
                got = _run_kernel(x, w, args.device)
            except Exception as exc:
                failures.append((key, f"{type(exc).__name__}: {exc}"))
                print(f"  {key:24} ERROR  {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:90]}")
                continue

            if x.dtype == np.float32:
                identical = np.array_equal(got, expected)
                verdict = "BIT-IDENTICAL" if identical else "DIVERGED"
                if not identical:
                    delta = float(np.abs(got.astype(np.float64)
                                         - expected.astype(np.float64)).max())
                    failures.append((key, f"max abs delta {delta:.3e}"))
                    verdict += f"  (max abs delta {delta:.3e})"
            else:
                diff = float(np.abs(got.astype(np.float64)
                                    - expected.astype(np.float64)).max())
                scale = float(np.abs(expected.astype(np.float64)).max()) or 1.0
                ok = diff / scale < _FP16_DRIFT_BAR
                verdict = f"drift {diff / scale:.2e}"
                if not ok:
                    verdict += "  OVER BAR"
                    failures.append((key, f"drift {diff / scale:.3e}"))
            print(f"  {key:24} {verdict}")

    print()
    if failures:
        print(f"KERNEL COMPARISON FAILED on {len(failures)} case(s):")
        for key, why in failures:
            print(f"  {key}: {why}")
        return 1
    print("KERNEL COMPARISON PASSED — the unmodified rms_norm kernel "
          "reproduces the CUDA reference on this device.")
    print("This is NOT the milestone: the milestone runs through NBXTensor "
          "(tools/metal_first_light.py) and needs the Metal DeviceAllocator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
