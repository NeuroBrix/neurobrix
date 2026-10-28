"""atan2(y, x) — pure @triton.jit kernel (element-wise, fp32).

`tl.extra.CUDA.libdevice.atan2` stood here. That namespace is NVIDIA's device
library and does not lower on any other backend: Kokoro-82M died at
`aten.angle::0` (1, 11, 15361) on triton-ext with "PassManager::run failed",
`angle` being atan2(imag, real). Triton 3.8.0 exposes no portable `atan` either
(`tl.math` has abs, ceil, cos, div_rn, erf, exp, exp2, fdiv, floor, fma, log,
log2, rsqrt, sin, sqrt, sqrt_rn, umulhi), so the function is built here.
"""

import triton
import triton.language as tl

# Triton only lets a @jit kernel read a global that is a constexpr.
_PI = tl.constexpr(3.14159265358979323846)
_PI_2 = tl.constexpr(1.57079632679489661923)


@triton.jit
def _atan_unit(z):
    """atan(z) for |z| <= 1, odd minimax polynomial in z^2.

    Degree-17 odd fit (the classic SunPro/Cephes coefficients). A degree-9 fit
    was tried first and measured 1.167e-05 max absolute error against numpy over
    all four quadrants — too coarse for a phase feeding audio synthesis, which
    is what `aten::angle` is here. This one is checked in the probe beside this
    file, not asserted in the kernel.
    """
    z2 = z * z
    p = 0.0028662257
    p = -0.0161657367 + z2 * p
    p = 0.0429096138 + z2 * p
    p = -0.0752896400 + z2 * p
    p = 0.1065626393 + z2 * p
    p = -0.1420889944 + z2 * p
    p = 0.1999355085 + z2 * p
    p = -0.3333314528 + z2 * p
    return z + z * z2 * p


@triton.jit
def _atan(z):
    """atan(z) over the whole line: fold |z| > 1 onto the unit interval."""
    a = tl.abs(z)
    big = a > 1.0
    # for |z| > 1, atan(z) = pi/2 - atan(1/z), with the sign carried outside
    u = tl.where(big, 1.0 / a, a)
    r = _atan_unit(u)
    r = tl.where(big, _PI_2 - r, r)
    return tl.where(z < 0, -r, r)


@triton.jit
def atan2_kernel(y_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offs < n_elements
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    # Quadrant reconstruction, matching numpy/libm atan2:
    #   x > 0            atan(y/x)
    #   x < 0, y >= 0    atan(y/x) + pi
    #   x < 0, y <  0    atan(y/x) - pi
    #   x = 0, y > 0     +pi/2        x = 0, y < 0   -pi/2      x = y = 0   0
    safe_x = tl.where(x == 0, 1.0, x)
    base = _atan(y / safe_x)
    shifted = base + tl.where(y >= 0, _PI, -_PI)
    on_axis = tl.where(y > 0, _PI_2, tl.where(y < 0, -_PI_2, 0.0))
    out = tl.where(x > 0, base, tl.where(x < 0, shifted, on_axis))
    tl.store(out_ptr + offs, out, mask=mask)
