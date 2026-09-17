"""Power — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def pow_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    exponent,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    # `tl.extra.CUDA.libdevice.pow` used to stand here. It is NVIDIA's device
    # library, and asking any other backend to lower it fails at compile time —
    # measured on triton-ext, Kokoro-82M dying at aten.pow::0 (1, 55, 2048) with
    # "PassManager::run failed", and reproduced with a bare kernel containing no
    # engine code (repro_pow_libdevice_cuda_namespace.py):
    #     tl.extra.cuda.libdevice.pow   FAILED   PassManager::run failed
    #     portable exp/log/sign         COMPILED max rel err 2.522e-07
    # Triton 3.8.0 exposes no backend-neutral `pow`, so the semantics are built
    # here rather than borrowed from one vendor.
    #
    # libm semantics, which is what the libdevice call was chosen for:
    #   x > 0                     exp(e * ln x)
    #   x = 0                     0            (for e > 0)
    #   x < 0, e an integer       sign * exp(e * ln|x|), sign = -1 when e is odd
    #   x < 0, e not an integer   NaN          (no real root)
    _ax = tl.abs(x_fp32)
    _mag = tl.exp(exponent * tl.log(_ax))
    _is_int = tl.floor(exponent) == exponent
    _is_odd = _is_int & ((tl.floor(exponent * 0.5) * 2.0) != exponent)
    _sign = tl.where(_is_odd, -1.0, 1.0)
    _neg = tl.where(_is_int, _sign * _mag, float("nan"))
    out = tl.where(x_fp32 > 0, _mag, tl.where(x_fp32 < 0, _neg, 0.0))
    tl.store(output_ptr + offset, out, mask=mask)
