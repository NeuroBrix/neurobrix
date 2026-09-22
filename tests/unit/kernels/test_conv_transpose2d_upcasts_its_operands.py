"""conv_transpose2d multiplies in fp32, whatever dtype its operands arrive in.

It used to multiply a MASKED-LOADED operand in the operands' own dtype:

    in_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0)   # no upcast
    w_val  = tl.load(weight_ptr + w_offset)                          # no upcast
    acc   += tl.where(valid, in_val * w_val, 0.0)

That is, line for line, what `depthwise_conv2d` did before metal-first-light 393570c6 —
where the fp16 arm upcast and every other dtype took the native path, and bf16 with padding
returned 0.754 relative error against an fp64 oracle on Metal. This kernel had no fp16 arm at
all, so BOTH operands stayed narrow for every dtype. Found by sweeping the kernel family for
that shape after the Mac reported theirs: of seven candidates, five upcast at the LOAD line
(gemv_vec, mv_op, addmv_op, conv_depthwise2d, moe_decode_vec) and this was the only match.

WHAT IT IS WORTH ON CUDA, measured on a V100 (Cin=Cout=8, 16x16, k=3, stride 1):

    dtype       before      after
    float32     2.411e-07   2.411e-07     unchanged
    float16     4.802e-04   3.700e-04     -23 %
    bfloat16    3.345e-03   2.687e-03     -20 %

So it is not a Metal-only courtesy: an fp32 product is strictly more accurate than a narrow
one, and the accumulator was ALREADY fp32, so it costs nothing on any backend (R23).

THE INVARIANT, which is the part that transfers
-----------------------------------------------
Padding must not move a dtype's error floor. The floor is set by the mantissa; padding adds
masked taps that contribute zero. A dtype whose pad=1 and pad=0 disagree is a dtype the
kernel is treating differently from the others — which is how the Metal defect was finally
located, after an error map and a dtype table had failed to name it. Asserted here for
every dtype, so this kernel cannot acquire that asymmetry silently.

On CUDA the invariant held BEFORE the fix too (3.345e-03 at both paddings), which is exactly
why the defect was invisible here and why this cell is about the invariant rather than about
a number going down.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

CI = CO = 8
H = W = 16
K = 3


def _dev_ok():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        return DeviceAllocator.device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _dev_ok(), reason="no accelerator visible")


def _reference(x, w, pad):
    """float64 transposed convolution from the definition — no library conv, so it
    cannot share a defect with the kernel under test."""
    x = x.astype(np.float64); w = w.astype(np.float64)
    _, ci, h, wd = x.shape
    _, co, kh, kw = w.shape
    oh, ow = (h - 1) - 2 * pad + kh, (wd - 1) - 2 * pad + kw
    full = np.zeros((1, co, oh + 2 * pad, ow + 2 * pad))
    for c in range(ci):
        for o in range(co):
            for i in range(h):
                for j in range(wd):
                    full[0, o, i:i + kh, j:j + kw] += x[0, c, i, j] * w[c, o]
    return full[:, :, pad:pad + oh, pad:pad + ow]


def _run(dtype, pad):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as WR
    rng = np.random.default_rng(20260922)
    x = rng.standard_normal((1, CI, H, W), dtype=np.float32)
    w = rng.standard_normal((CI, CO, K, K), dtype=np.float32)
    out = WR.conv_transpose_wrapper(
        NBXTensor.from_numpy(x).to(dtype), NBXTensor.from_numpy(w).to(dtype), None,
        stride=1, padding=pad, dilation=1, output_padding=0, groups=1)
    got = out.to("float32").numpy().astype(np.float64)
    ref = _reference(x, w, pad)
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))


@pytest.mark.parametrize("dtype,floor", [("float32", 1e-6), ("float16", 1e-3), ("bfloat16", 6e-3)])
def test_each_dtype_sits_on_its_mantissa_floor(dtype, floor):
    for pad in (0, 1):
        assert _run(dtype, pad) < floor, f"{dtype} pad={pad} is above its mantissa floor"


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_padding_does_not_move_the_error_floor(dtype):
    """THE invariant. A dtype whose padded and unpadded error differ is a dtype this
    kernel treats differently from the others."""
    a, b = _run(dtype, 0), _run(dtype, 1)
    ratio = max(a, b) / max(min(a, b), 1e-30)
    assert ratio < 3.0, (
        f"{dtype}: padding moved the floor, pad0={a:.3e} pad1={b:.3e} (ratio {ratio:.2f}). "
        f"That is the signature of a narrow-dtype product on a masked-loaded operand.")


def test_the_source_upcasts_both_operands_at_the_load():
    """A door, not a census: the numbers above pass on CUDA even with the defect present,
    so the one thing this rack CAN assert universally is that neither operand is multiplied
    in its own dtype."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[3]
           / "src/neurobrix/kernels/ops/conv_transpose2d.py").read_text()
    assert "in_val = tl.load(input_ptr + in_offset, mask=valid, other=0.0).to(tl.float32)" in src, \
        "the masked input operand is no longer upcast at the load"
    assert "w_val = tl.load(weight_ptr + w_offset).to(tl.float32)" in src, \
        "the weight operand is no longer upcast at the load"
