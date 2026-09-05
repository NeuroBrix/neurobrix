"""The float policy the Metal compile runs under is declared, not defaulted.

`MTLCompileOptions` comes out of `init()` with mathMode FAST. Fast math lets
the compiler reassociate float arithmetic and swap in fast approximations for
divide, rsqrt and the transcendentals — a numerical policy, taken silently,
that nobody in this engine chose.

It changed real results: with fast math on, rms_norm fp32 at 2x4096 sat 3 ULP
from the fp64 oracle where the CUDA reference sits at 1, failing the first
light bar at that shape. With safe math the same kernel is BIT-IDENTICAL to
CUDA. So this is pinned, and pinned at the level that matters — the options
object the driver actually compiles with, not a constant.

Skipped where there is no Apple GPU; the pin is about Metal.
"""

from __future__ import annotations

import pytest

metal_driver = pytest.importorskip("neurobrix.triton.metal_driver")


def _has_metal():
    try:
        from neurobrix.kernels import nbx_tensor
        return nbx_tensor._detect_gpu_backend() == "metal"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="no Apple GPU here")


def test_the_default_metal_options_really_are_fast_math():
    """The negative control for the pin below.

    If Apple ever changes the default, this fails and the comment above stops
    being true — which is the moment to re-read the policy, not to relax it.
    """
    import Metal

    options = Metal.MTLCompileOptions.alloc().init()
    assert options.mathMode() != metal_driver.METAL_MATH_MODE_SAFE, (
        "MTLCompileOptions no longer defaults to fast math; the reasoning "
        "recorded in metal_driver.METAL_MATH_MODE_SAFE needs re-reading")


def test_kernels_are_compiled_under_safe_math():
    """The options object the driver actually compiles every kernel with."""
    options = metal_driver.compile_options()
    assert options.mathMode() == metal_driver.METAL_MATH_MODE_SAFE, (
        f"kernels compiled under mathMode {options.mathMode()}; the engine's "
        f"float results would be decided by a flag nobody chose")


def test_a_kernel_really_compiles_under_that_policy():
    """Not just the options: a real MSL source through the real call."""
    from neurobrix.kernels.metal_device import runtime

    source = ("#include <metal_stdlib>\n"
              "using namespace metal;\n"
              "kernel void nbx_probe(device float* out [[buffer(0)]],\n"
              "                      uint gid [[thread_position_in_grid]])\n"
              "{ out[gid] = 1.0f / sqrt(out[gid]); }\n")
    library = metal_driver.library_from_source(runtime()._device, source)
    assert "nbx_probe" in list(library.functionNames())
