"""The staged-dot shape must COMPUTE correctly, not merely stop refusing.

Everything measured on this chantier so far is a question of compilation: the
refusal falls, another appears, the MSL is emitted, the shader fails with nine
errors. The day those nine fall there will be a kernel that compiles, and
nothing whatever will have said that it computes the right thing. Fifteen
models would start running again and produce outputs no one had confronted
with anything.

So the success criterion is not "it compiles". It is "it compiles AND the
output agrees with an fp64 oracle inside the profile's tolerance, on the real
shape". This file is that second half, and it is written NOW -- while the
refusal still stands and the TTGIR fixture is still in hand -- rather than
after the nine errors fall, when the temptation to call it done will be at its
peak.

It must be red today for the RIGHT reason: the refusal holds, the kernel falls
back, so nothing on the Triton path computes anything. And it must turn green
BY CALCULATION, not by compilation -- which is why the oracle is an
independent fp64 convolution over the same inputs and not a second call into
the same code.

The first test is the template control. Three synthetic kernels written to
resemble this shape all compiled cleanly, each through a template the real
kernel never reaches -- a green that said something true about the wrong
object. A shape used to measure a compilation defect proves nothing until it
is shown to take the same path as the real case.

Runnable: PYTHONPATH=src python3 -m pytest \
    tests/unit/kernels/test_staged_dot_computes_not_merely_compiles.py -v
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from check_measurement_environment import owned_cache_env       # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="the Metal staged-dot path")


# ── the shape, chosen so the dot's tile is exactly the refused width ───────
#
# BLOCK_SIZE_BHW=32 x BLOCK_SIZE_OUTF=64 = 2048, which is the tile the captured
# IR carries and the width the refusal names. Both come from the kernel's own
# autotune config list, so this is a configuration a model can land on, not one
# invented for the test.
N, IN_C, IN_H, IN_W = 1, 32, 16, 16
OUT_C, KH, KW = 64, 3, 3
STRIDE, PAD = 1, 1
OUT_H = (IN_H + 2 * PAD - KH) // STRIDE + 1
OUT_W = (IN_W + 2 * PAD - KW) // STRIDE + 1
#: Every config `conv2d_forward_kernel` declares whose dimensions are all at
#: 64 or below -- the ones the generic dot path does not refuse, and therefore
#: the ones the tuner can now actually choose since a refused config stopped
#: ending the sweep. Six of the eight put a tile wider than the threadgroup
#: through the cooperative staged path.
#:
#: Validating ONE of them and calling the path correct is the mistake this
#: file exists to prevent, one level up: the fix opened degrees of freedom and
#: each is a shape a model can land on. Read from the kernel's own config list
#: so a config added there cannot stay unmeasured.
def _servable_configs():
    import re
    src = Path(__import__("neurobrix.kernels.ops.conv2d", fromlist=["x"]).__file__).read_text()
    found = re.findall(
        r"BLOCK_SIZE_BHW':\s*(\d+),\s*'BLOCK_SIZE_OUTF':\s*(\d+),\s*'BLOCK_SIZE_INF':\s*(\d+)",
        src)
    return sorted({tuple(int(x) for x in c) for c in found if max(int(x) for x in c) <= 64})


BLOCK_BHW, BLOCK_OUTF, BLOCK_INF = 32, 64, 32

_NESTING_REFUSAL = "per-element wrap loop is emitted OUTSIDE that loop"


@pytest.fixture(autouse=True)
def _owns_its_cache(tmp_path, monkeypatch):
    """This test compiles, so it owns its caches or it is not a measurement."""
    for var, value in owned_cache_env(tmp_path).items():
        monkeypatch.setenv(var, value)


def _profile_tolerance(dtype: str) -> float:
    """The profile's own screening tolerance for this dtype.

    Read from the profile file through the same function the certification
    gate uses -- not a number written here, and not a second table that could
    drift from the one the engine trusts.
    """
    from neurobrix.kernels.autotune_certified import _tolerance_for, active_profile

    prof = active_profile()
    if prof is None:
        pytest.skip("no active profile on this machine")
    vendor, profile = prof
    tol = _tolerance_for(vendor, profile, dtype)
    if tol is None:
        pytest.skip(f"{vendor}/{profile} declares no tolerance for {dtype}")
    return tol


def _deviation(got, want) -> float:
    """The deviation the certification gate itself computes.

    Imported rather than written here. The first version of this test scored
    `max |got - want| / max(|want|, 1e-6)` per element and reported 222 against
    a tolerance of 0.04 for a kernel whose correlation with the oracle was
    0.9999959: a relative error taken element-wise explodes wherever the oracle
    passes through zero, which a convolution's output does constantly. A
    verdict is only as good as its metric, and inventing a second metric beside
    the one the engine trusts is how a correct kernel gets rejected -- or a
    wrong one accepted, since the same floor can hide a real error too.
    """
    from neurobrix.kernels.autotune_certify import oracle_deviation

    return oracle_deviation(np.asarray(got), np.asarray(want))


def _inputs(seed=20260912):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N, IN_C, IN_H, IN_W), dtype=np.float32)
    w = rng.standard_normal((OUT_C, IN_C, KH, KW), dtype=np.float32) * 0.1
    return x, w


def _oracle_fp64(x32, w32):
    """A direct convolution in float64 over the SAME inputs.

    Independent of the kernel by construction: it shares no code path with it,
    only the numbers. A reference computed by calling the engine again would
    agree with a wrong kernel exactly as happily as with a right one.
    """
    x = x32.astype(np.float64)
    w = w32.astype(np.float64)
    xp = np.pad(x, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    out = np.zeros((N, OUT_C, OUT_H, OUT_W), dtype=np.float64)
    for kh in range(KH):
        for kw in range(KW):
            patch = xp[:, :, kh:kh + OUT_H * STRIDE:STRIDE,
                       kw:kw + OUT_W * STRIDE:STRIDE]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, kh, kw])
    return out


def _run_pinned(x32, w32, blocks=None):
    """Launch the real kernel with the config that gives the refused tile.

    `conv2d_forward_kernel` is autotuned; `.fn` is the jit function underneath,
    so the config is pinned instead of chosen. Pinning is what makes the test
    about ONE shape rather than about whatever the tuner picks today.

    Returns (output_as_float64, warnings_raised).
    """
    import triton
    from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
    from neurobrix.kernels.ops.conv2d import conv2d_forward_kernel
    from neurobrix.kernels.autotune_certify import f32_to_bf16_bits, bf16_bits_to_f32

    def _bf16(arr):
        # bf16 travels in a uint16 container, and `from_numpy`'s `dtype`
        # argument names what the BITS already are -- it does not convert.
        # Letting the element size be guessed from the carrier is what made
        # every bf16 shape uncertifiable before.
        bits = f32_to_bf16_bits(np.ascontiguousarray(arr))
        return NBXTensor.from_numpy(bits, dtype=NBXDtype.bfloat16)

    bhw, outf, inf = blocks or (BLOCK_BHW, BLOCK_OUTF, BLOCK_INF)
    x, w = _bf16(x32), _bf16(w32)
    out = NBXTensor.empty((N, OUT_C, OUT_H, OUT_W), device=x.device,
                          dtype=NBXDtype.bfloat16)
    grid = (triton.cdiv(N * OUT_H * OUT_W, bhw), triton.cdiv(OUT_C, outf), 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conv2d_forward_kernel.fn[grid](
            x, w, out,
            N, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
            *x.stride(), *w.stride(), *out.stride(),
            kernel_height=KH, kernel_width=KW,
            stride_height=STRIDE, stride_width=STRIDE,
            padding_height=PAD, padding_width=PAD,
            dilation_height=1, dilation_width=1,
            groups=1, fp16=False,
            BLOCK_SIZE_BHW=bhw, BLOCK_SIZE_OUTF=outf, BLOCK_SIZE_INF=inf,
        )
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.sync_device()
    got = bf16_bits_to_f32(out.numpy().view(np.uint16)).reshape(
        (N, OUT_C, OUT_H, OUT_W))
    return got.astype(np.float64), [str(c.message) for c in caught]


def _emitted_msl(fn):
    """The MSL this run actually emitted, captured at the emitter.

    Read from the emitter rather than from a warning: once the path works
    there are no warnings, and a control that keys on a refusal's text stops
    controlling anything the moment the refusal is fixed -- which is exactly
    when it is needed most.
    """
    import triton_msl.codegen.msl_emitter as EM

    seen = []
    original = EM.emit_msl

    def spy(mod, metadata, options):
        out = original(mod, metadata, options)
        seen.append(out)
        return out

    # Only the emitter module is touched. Importing the backend module here
    # re-runs triton's subclass discovery over a half-initialised module and
    # it finds zero backends -- an import that breaks what it observes.
    EM.emit_msl = spy
    try:
        fn()
    finally:
        EM.emit_msl = original
    return seen


def test_this_shape_takes_the_staged_dot_path():
    """The template control, and the reason the verdict below means anything.

    Three synthetic kernels written for this refusal all compiled cleanly
    through a template the real kernel never reaches, and every assertion on
    them was correct about the wrong object. So the shape must be shown to
    take the cooperative staged path, not merely to produce numbers.

    It reads the emitted MSL, which works on both sides of the fix: the
    cooperative fill loop and the barrier before the dot are what that path
    emits, and no other path emits them.
    """
    x32, w32 = _inputs()
    msls = _emitted_msl(lambda: _run_pinned(x32, w32))
    if not msls:
        pytest.fail("no MSL was emitted at all; this measured nothing")
    blob = "\n".join(msls)
    staged = ("_sa" in blob) and ("threadgroup_barrier" in blob)
    assert staged, (
        "the emitted MSL carries no cooperative staged fill. This shape was "
        "taken by some other path, so the oracle verdict below is about a "
        "different object than the one fifteen models are blocked on.")
    assert "_loop_e" in blob, (
        "no per-element wrap loop: a tile wider than the threadgroup must be "
        "covered by one, and its absence means the tile is not the wide one")


@pytest.mark.parametrize("blocks", _servable_configs(),
                         ids=lambda b: "x".join(str(v) for v in b))
def test_the_output_agrees_with_the_fp64_oracle(blocks):
    """Every config the tuner can now choose, judged by its numbers.

    Parametrised rather than pinned to one, because the config-exclusion fix
    is what made the other seven reachable: before it, a sweep died on the
    first refused config and the tuner never got to them. A fix that opens
    choices owes a verdict on each choice.
    """
    tol = _profile_tolerance("bf16")
    x32, w32 = _inputs()
    got, msgs = _run_pinned(x32, w32, blocks)

    fell_back = [m for m in msgs if "fall back" in m or "codegen failed" in m]
    assert not fell_back, (
        "the kernel fell back off the Triton path, so this output was not "
        f"produced by the code under test: {fell_back[0][:200]}")

    want = _oracle_fp64(x32, w32)
    dev = _deviation(got, want)
    assert dev <= tol, (
        f"max relative deviation {dev:.3e} exceeds the profile's bf16 "
        f"tolerance {tol:.3e} at config {blocks}. The kernel compiles and "
        f"computes the wrong "
        f"thing, which is the outcome this file exists to make impossible to "
        f"mistake for success.")


# ── the oracle, proven before it is trusted ───────────────────────────────


def _oracle_by_hand(x32, w32):
    """The same convolution written the dumbest way there is.

    Seven nested loops, no einsum, no stride tricks. It is slow and it is
    obviously correct, which is the whole point: the vectorised oracle above is
    the one the verdict rests on, and a vectorised expression is exactly where
    an off-by-one in the padding or a transposed weight axis hides without
    changing the shape of the result.
    """
    x = x32.astype(np.float64)
    w = w32.astype(np.float64)
    out = np.zeros((N, OUT_C, OUT_H, OUT_W), dtype=np.float64)
    for n in range(N):
        for oc in range(OUT_C):
            for oh in range(OUT_H):
                for ow in range(OUT_W):
                    acc = 0.0
                    for ic in range(IN_C):
                        for kh in range(KH):
                            for kw in range(KW):
                                ih = oh * STRIDE - PAD + kh
                                iw = ow * STRIDE - PAD + kw
                                if 0 <= ih < IN_H and 0 <= iw < IN_W:
                                    acc += x[n, ic, ih, iw] * w[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = acc
    return out


def test_the_oracle_agrees_with_a_hand_written_convolution():
    """The verdict rests on the oracle, so the oracle is checked first.

    A small shape, because seven Python loops over the real one would take
    minutes -- but the SAME code path through `_oracle_fp64`, so an error in
    its padding or its axis order shows here.
    """
    global N, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W
    saved = (N, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W)
    N, IN_C, IN_H, IN_W, OUT_C = 1, 3, 5, 5, 4
    OUT_H = (IN_H + 2 * PAD - KH) // STRIDE + 1
    OUT_W = (IN_W + 2 * PAD - KW) // STRIDE + 1
    try:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((N, IN_C, IN_H, IN_W), dtype=np.float32)
        w = rng.standard_normal((OUT_C, IN_C, KH, KW), dtype=np.float32)
        fast, slow = _oracle_fp64(x, w), _oracle_by_hand(x, w)
        assert fast.shape == slow.shape
        assert np.allclose(fast, slow, rtol=0, atol=1e-12), (
            f"the vectorised oracle disagrees with the hand-written one by "
            f"{np.max(np.abs(fast - slow)):.3e}; every verdict below it would "
            f"be measured against the wrong numbers")
    finally:
        N, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W = saved


def test_the_oracle_is_not_trivially_satisfiable():
    """Both directions: it must also DISAGREE when the answer is wrong.

    An oracle that returns something close to anything would pass the test
    above and pass the kernel too. Perturbing one weight must move the result
    by more than the tolerance the verdict uses, or the comparison cannot
    detect a kernel that computes almost-right.
    """
    tol = _profile_tolerance("bf16")
    x32, w32 = _inputs()
    want = _oracle_fp64(x32, w32)
    w_bad = w32.copy()
    w_bad[0, 0, 0, 0] += 1.0
    dev = _deviation(_oracle_fp64(x32, w_bad), want)
    assert dev > tol, (
        f"changing one weight moved the oracle by {dev:.3e}, within the "
        f"tolerance {tol:.3e} the verdict uses. The comparison would accept a "
        f"kernel that is wrong by at least that much.")
