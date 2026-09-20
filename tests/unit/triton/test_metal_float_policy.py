"""The float policy the Metal compile runs under is declared, not defaulted.

`MTLCompileOptions` comes out of `init()` with mathMode FAST. Fast math lets
the compiler reassociate float arithmetic and swap in fast approximations for
divide, rsqrt and the transcendentals — a numerical policy, taken silently,
that nobody in this engine chose.

It changed real results: with fast math on, rms_norm fp32 at 2x4096 sat 3 ULP
from the fp64 oracle where the CUDA reference sits at 1, failing the first
light bar at that shape. With safe math the same kernel is BIT-IDENTICAL to
CUDA.

WHOSE POLICY IT IS NOW. This file pinned the archived fork's own
`compile_options()`. The backend is triton-ext, and it makes the same choice —
measured 2026-09-17 in the pinned build, in BOTH of its compile routes:

    metal_native.m:501-507   MTLCompileOptions -> opt.mathMode = MTLMathModeSafe
                             (and fastMathEnabled = NO on older systems)
    compiler.py:65           xcrun metal -c -fmetal-math-mode=safe

That is a third party's decision, and we pin triton-ext at a commit precisely so
that a third party's decision cannot change under us without being seen. So the
pin moves here rather than disappearing with the fork: it reads the backend we
actually ship and fails if a pin bump ever flips the policy, which is cheaper to
find here than in a model's numbers.

Skipped where there is no Apple GPU; the pin is about Metal.
"""

from __future__ import annotations

import re

import pytest

_ext = pytest.importorskip("triton_apple_backend",
                           reason="the Metal backend in force is not installed here")


def _has_metal():
    nbx_tensor = pytest.importorskip("neurobrix.kernels.nbx_tensor")
    detect = nbx_tensor._detect_gpu_backend   # a renamed probe raises here, loudly
    try:
        return detect() == "metal"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_metal(), reason="no Apple GPU here")


def test_the_default_metal_options_really_are_fast_math():
    """The negative control for the pins below.

    If Apple ever changes the default, this fails and the reasoning above stops
    being true — which is the moment to re-read the policy, not to relax it.
    """
    import Metal

    options = Metal.MTLCompileOptions.alloc().init()
    safe = getattr(Metal, "MTLMathModeSafe", 0)
    assert options.mathMode() != safe, (
        "MTLCompileOptions no longer defaults to fast math; the reasoning "
        "recorded at the top of this file needs re-reading")


def _source(name):
    import pathlib
    return (pathlib.Path(_ext.__file__).parent / name).read_text()


def test_the_offline_route_compiles_safe():
    """`xcrun metal` is where every kernel of a cold run is compiled."""
    src = _source("compiler.py")
    assert "-fmetal-math-mode=safe" in src, (
        "the backend's offline compile no longer asks for safe math; the "
        "engine's float results would be decided by a flag nobody chose")
    assert "-ffast-math" not in src


def test_the_runtime_route_compiles_safe():
    """`MTLCompileOptions` is the other route, used when MSL is compiled in
    process rather than through the offline tool."""
    src = _source("metal_native.m")
    assert re.search(r"mathMode\s*=\s*MTLMathModeSafe", src), (
        "the backend's in-process compile no longer sets MTLMathModeSafe")
    assert re.search(r"fastMathEnabled\s*=\s*NO", src), (
        "the older-OS fallback no longer disables fast math")


def test_a_kernel_really_compiles_under_that_policy():
    """Not just the flags: a real MSL source through a real compile, with the
    reciprocal square root that fast math is most eager to approximate."""
    import Metal

    from neurobrix.kernels.metal_device import runtime

    source = ("#include <metal_stdlib>\n"
              "using namespace metal;\n"
              "kernel void nbx_probe(device float* out [[buffer(0)]],\n"
              "                      uint gid [[thread_position_in_grid]])\n"
              "{ out[gid] = 1.0f / sqrt(out[gid]); }\n")
    options = Metal.MTLCompileOptions.alloc().init()
    if options.respondsToSelector_("setMathMode:"):
        options.setMathMode_(getattr(Metal, "MTLMathModeSafe", 0))
    else:                                              # pragma: no cover
        options.setFastMathEnabled_(False)
    library, error = runtime()._device.newLibraryWithSource_options_error_(
        source, options, None)
    assert library is not None, f"the probe did not compile: {error}"
    assert "nbx_probe" in list(library.functionNames())
