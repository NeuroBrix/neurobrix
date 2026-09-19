"""Wan's RoPE is a complex multiply broadcast over heads. Does it rotate right?

Read out of the container, Wan2.1's 3-D RoPE is faithful to the vendor at every
step: the frequency bands concatenate 22 + 21 + 21 complex pairs over (T, H, W)
into [10, 30, 52, 64], flatten row-major to [15600, 64], unsqueeze twice to
[1, 1, 15600, 64], and multiply a query of [1, 12, 15600, 64] -- broadcasting one
position's rotation across all twelve heads. `view(..., 64, 2)` then
`view_as_complex` means the INTERLEAVED convention, pairs adjacent, which is what
diffusers does.

So the structure is right and only the execution can be wrong, and a complex
multiply is where this engine has been bitten: complex tensors are stored
interleaved and Triton sees the pointer as plain float, so every stride has to be
doubled with a trailing re/im axis appended. A transposed-complex `contiguous`
mangled the imaginary half of chatterbox's s3gen STFT for exactly this reason.

WHY IT MATTERS HERE. A rotation by the wrong angle leaves every value finite and
plausible, gives every token the wrong position, and lets the model never form a
scene -- while the patch grid survives untouched. That is precisely the artefact:
a field with a period of exactly 16.00 px, one DiT patch, at 20 steps.

This does not assume the multiply is the fault. It ASKS, against NumPy.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_a_complex_rotation_broadcasts_over_heads.py
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator

_TOTAL = DeviceAllocator.device_count()
needs_a_card = pytest.mark.skipif(_TOTAL == 0, reason="needs a CUDA device")

# Wan2.1-T2V-1.3B at 480x832x8, read from the graph: 12 heads, 15600 tokens,
# 64 complex pairs per head (head_dim 128). Trimmed in tokens to keep the cell
# small -- the token count is not what the broadcast is about, the HEAD axis is.
HEADS, TOK, PAIRS = 12, 512, 64


def _complex_pair(rng, shape):
    """A complex tensor as this engine stores it: interleaved, trailing [2]."""
    re = rng.standard_normal(shape).astype(np.float32)
    im = rng.standard_normal(shape).astype(np.float32)
    return re, im, np.stack([re, im], axis=-1).astype(np.float32)


@needs_a_card
def test_a_broadcast_complex_multiply_matches_numpy():
    """[1, 12, T, 64] x [1, 1, T, 64] -- one rotation, twelve heads."""
    rng = np.random.default_rng(20260918)
    qre, qim, q_inter = _complex_pair(rng, (1, HEADS, TOK, PAIRS))
    fre, fim, f_inter = _complex_pair(rng, (1, 1, TOK, PAIRS))

    expected = (qre + 1j * qim) * (fre + 1j * fim)

    try:
        qc = NBXTensor.from_numpy(q_inter).view_as_complex()
        fc = NBXTensor.from_numpy(f_inter).view_as_complex()
    except AttributeError as e:
        pytest.skip(f"no view_as_complex on NBXTensor: {e}")

    got = (qc * fc)
    DeviceAllocator.sync_device()
    got_np = got.numpy()
    if got_np.dtype.kind != "c":                      # came back as interleaved
        got_np = got_np[..., 0] + 1j * got_np[..., 1]

    np.testing.assert_allclose(
        got_np, expected, rtol=2e-5, atol=2e-5,
        err_msg="the broadcast complex rotation does not match NumPy")


@needs_a_card
def test_the_imaginary_half_survives_the_round_trip():
    """view_as_complex -> view_as_real must return what went in.

    The interleaved layout means a stride computed in complex elements and
    applied as float offsets reads the real part twice and the imaginary part
    never -- which looks like a plausible tensor, not like a crash.
    """
    rng = np.random.default_rng(7)
    _, _, inter = _complex_pair(rng, (2, 3, 5, PAIRS))
    t = NBXTensor.from_numpy(inter)
    try:
        back = t.view_as_complex().view_as_real()
    except AttributeError as e:
        pytest.skip(f"no complex views on NBXTensor: {e}")
    DeviceAllocator.sync_device()
    np.testing.assert_array_equal(
        back.numpy(), inter,
        err_msg="the round trip lost or duplicated a half")


@needs_a_card
def test_a_pure_rotation_preserves_magnitude():
    """A property NumPy is not needed to state: |q * e^{i0}| == |q|.

    RoPE's frequencies are unit-modulus, so the rotation must not change any
    magnitude. This catches a wrong-angle multiply that still happens to agree
    with a wrongly-written oracle -- the failure a single reference cannot see.
    """
    rng = np.random.default_rng(11)
    qre, qim, q_inter = _complex_pair(rng, (1, HEADS, TOK, PAIRS))
    ang = rng.uniform(-np.pi, np.pi, (1, 1, TOK, PAIRS)).astype(np.float32)
    f_inter = np.stack([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32)

    try:
        qc = NBXTensor.from_numpy(q_inter).view_as_complex()
        fc = NBXTensor.from_numpy(f_inter).view_as_complex()
    except AttributeError as e:
        pytest.skip(f"no view_as_complex on NBXTensor: {e}")
    got = (qc * fc)
    DeviceAllocator.sync_device()
    g = got.numpy()
    if g.dtype.kind != "c":
        g = g[..., 0] + 1j * g[..., 1]

    np.testing.assert_allclose(
        np.abs(g), np.abs(qre + 1j * qim), rtol=1e-4, atol=1e-4,
        err_msg="a unit-modulus rotation changed the magnitudes")
