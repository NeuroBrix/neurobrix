"""A bf16-output kernel's screen oracle must be encoded in the buffer's own width.

`screen_oracle.provider` returns the fp64 reference cast to the OUTPUT buffer's
dtype so the screen can compare it, byte-for-byte length, against each
candidate's output. bfloat16 is not a numpy dtype, so the cast table (`_NP`)
has no entry for it — and the fall-through was `np.float32`, four bytes per
element against the buffer's two. The reference then measured exactly DOUBLE
the output buffer, the size check rejected it, and the provider announced
"no oracle" and fell back to the bare consensus vote.

Measured 2026-09-14 on `conv2d_forward_kernel` at (1,180,448,448) bf16: the
fp64 conv oracle was wired and computing, yet the reference (144506880 bytes)
did not match the bf16 output (72253440), so every upscaler convolution — and
any bf16-output kernel — was seated UNSCREENED, a silent-wrong-in-waiting on a
public path, while the oracle sat unused.

This pins the fix: for a bf16 output the provider rounds the reference to
bf16 bits (round-to-nearest-even, the buffer's width) so it matches and the
oracle is consulted. A dtype property, not a backend one — so this runs
without a GPU.
"""
from __future__ import annotations

import numpy as np
from neurobrix.kernels import launcher as L

from neurobrix.kernels import screen_oracle as S
from neurobrix.kernels.autotune_certify import f32_to_bf16_bits
from neurobrix.kernels.nbx_tensor import NBXDtype


class _FakeTensor:
    """Enough of an NBXTensor for the provider: a host-readable bf16 carrier
    (uint16 bits) with a data_ptr and the bfloat16 dtype tag."""

    def __init__(self, bits: np.ndarray, addr: int):
        self._a = np.ascontiguousarray(bits, dtype=np.uint16)
        self._device = "cpu"
        self.shape = bits.shape
        self._addr = addr
        self._dtype = NBXDtype.bfloat16

    def numpy(self):
        return self._a

    def to_cpu(self):
        return self

    def data_ptr(self):
        return self._addr


def _conv_kernel_stub():
    def conv2d_forward_kernel():  # name is what the provider dispatches on
        raise AssertionError("not meant to run")
    return conv2d_forward_kernel


def test_bf16_conv_output_gets_a_matching_width_reference():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 3, 8, 8)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)
    out_bits = np.zeros((1, 4, 8, 8), dtype=np.uint16)   # bf16 output carrier

    x_t = _FakeTensor(f32_to_bf16_bits(x), 0x1000)
    w_t = _FakeTensor(f32_to_bf16_bits(w), 0x2000)
    o_t = _FakeTensor(out_bits, 0x3000)

    named = dict(
        input_pointer=x_t, weight_pointer=w_t, output_pointer=o_t,
        batch_dim=1, in_feat_dim=3, in_height=8, in_width=8,
        out_feat_dim=4, out_height=8, out_width=8,
        kernel_height=3, kernel_width=3,
        stride_height=1, stride_width=1,
        padding_height=1, padding_width=1,
        dilation_height=1, dilation_width=1, groups=1,
    )

    class _Tuner:
        base_fn = _conv_kernel_stub()
        nargs = named

    buffers = [L.ScreenedBuffer(0x3000, out_bits.nbytes, "bf16")]     # the output, bf16 width
    ref = S.provider(_Tuner(), ("conv-bf16-key",), buffers, meta=named)

    # The oracle must be CONSULTED, not discarded to the vote: a reference is
    # returned, one buffer, and it is the bf16 width of the output — not the
    # fp32 double that the missing `_NP["bf16"]` entry used to produce.
    assert ref is not None, (
        "provider fell back to the consensus vote for a bf16 conv output — the "
        "reference width did not match the output buffer")
    assert len(ref) == 1
    assert len(ref[0]) == out_bits.nbytes, (
        f"reference is {len(ref[0])} bytes for a {out_bits.nbytes}-byte bf16 "
        f"output buffer; a mismatch is what sent conv to the bare vote")


def test_fp32_output_still_matches():
    """The else-branch is unchanged: an fp32 output still gets an fp32-width
    reference. Guards against the bf16 branch stealing the general path."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((1, 3, 8, 8)) * 0.1).astype(np.float32)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)
    out = np.zeros((1, 4, 8, 8), dtype=np.float32)

    class _Fp32Tensor(_FakeTensor):
        def __init__(self, arr, addr):
            self._a = np.ascontiguousarray(arr, dtype=np.float32)
            self._device = "cpu"; self.shape = arr.shape; self._addr = addr
            self._dtype = None                          # not bf16

    x_t, w_t, o_t = _Fp32Tensor(x, 0x1000), _Fp32Tensor(w, 0x2000), _Fp32Tensor(out, 0x3000)
    named = dict(
        input_pointer=x_t, weight_pointer=w_t, output_pointer=o_t,
        batch_dim=1, in_feat_dim=3, in_height=8, in_width=8,
        out_feat_dim=4, out_height=8, out_width=8,
        kernel_height=3, kernel_width=3, stride_height=1, stride_width=1,
        padding_height=1, padding_width=1, dilation_height=1, dilation_width=1, groups=1,
    )

    class _Tuner:
        base_fn = _conv_kernel_stub()
        nargs = named

    ref = S.provider(_Tuner(), ("conv-fp32-key",), [L.ScreenedBuffer(0x3000, out.nbytes, "fp32")], meta=named)
    assert ref is not None and len(ref[0]) == out.nbytes
