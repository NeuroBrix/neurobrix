"""Complex-valued ops for the Triton zero-Torch execution branch."""

from __future__ import annotations

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops import elementwise_wrappers as _ew
from nbxtensor.tensor import NBXTensor


def _as_fp32(tensor):
    """Ensure tensor operands are float32 (narrow float64, widen half/int)."""
    if hasattr(tensor, "nbx_dtype") and tensor.nbx_dtype != NBXDtype.float32:
        return tensor.to(NBXDtype.float32)
    return tensor


def polar(abs_tensor, angle_tensor):
    """aten::polar(abs, angle) -> complex64 tensor (interleaved [re, im] pairs).

    Computes ``abs * cos(angle) + i * abs * sin(angle)`` via nbxtensor Triton ops,
    preserving broadcasting and matching ``aten::view_as_complex`` layout.
    """
    abs_fp32 = _as_fp32(abs_tensor)
    angle_fp32 = _as_fp32(angle_tensor)
    if not hasattr(angle_fp32, "nbx_dtype") and hasattr(abs_fp32, "nbx_dtype"):
        from nbxtensor.ops.dispatch import dispatch

        full_op = dispatch("full")
        angle_fp32 = full_op(
            (),
            float(angle_fp32),
            dtype=NBXDtype.float32,
            device=getattr(abs_fp32, "device", "cuda"),
        )
    real = _ew.mul(abs_fp32, _ew.cos(angle_fp32))
    imag = _ew.mul(abs_fp32, _ew.sin(angle_fp32))
    if getattr(real, "numel", lambda: 1)() == 0:
        return NBXTensor.empty(
            real.shape,
            NBXDtype.complex64,
            device=getattr(real, "device", "cuda"),
        )
    return _ew.complex_wrapper(real, imag)
