"""A convolution's declared cost (`conv_wrappers.conv2d_cost`, `conv3d_cost`, M3) is what its wrapper allocates on the
device: each path run once with the allocator's pool off, the allocator's own peak above its level at entry against the
declaration's `peak_bytes` — the plain launch and the matrix unit's (its contiguous input and its dense result for a
caller's view), a strided input, the conv3d's temporal pad, its taps added in place, its bias, and a temporal stride's
folds. mochi-1-preview's VAE replays x1.01 through the pool and its card died at 1.37x the walk (2026-09): the
transients a planner did not see are these.

What it would do if the code were wrong: a declaration missing a buffer the wrapper takes (the dense result, the
pad, a fold), or holding one it drops, differs from the measured peak (red)."""
import numpy as np
import pytest

from tests.unit.kernels.test_an_op_allocates_what_it_declares import (_device, _measure, _no_pool,  # noqa: F401
                                                                    _spec, _this_machine)
from nbxtensor.ops._cost import Spec

pytestmark = pytest.mark.skipif(not _device(), reason="the allocator's peak needs a device")


@pytest.fixture(autouse=True)
def _own_tuning_cache(tmp_path, monkeypatch):
    """A key this test meets uncertified is swept: into its own cache, never the machine's (`~/.neurobrix/replay_cache`)."""
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay_cache"))


def _t(shape, dtype, seed=0):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    t = NBXTensor.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(np.float32) * 0.1)
    return t if dtype == "float32" else t.to(getattr(NBXDtype, dtype))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("bias,stride,pad", [(False, 1, 1), (True, 1, 1), (False, 1, 0), (False, 2, 1)])
def test_conv3d(dtype, bias, stride, pad):
    from nbxtensor.ops import conv_wrappers as CW
    x = _t((1, 32, 9, 32, 64), dtype)
    w = _t((32, 32, 3, 3, 3), dtype, 1)
    b = _t((32,), dtype, 2) if bias else None
    st = (stride, 1, 1)
    measured = _measure(lambda: CW.conv2d_wrapper(x, w, b, st, (pad, 1, 1), 1, False, 0, 1))
    declared = CW.conv3d_cost(_spec(x), _spec(w), st, (pad, 1, 1), 1, 1, bias).peak_bytes
    assert measured == declared, (measured, declared)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("strided", [False, True])
def test_conv2d(dtype, strided):
    from nbxtensor.ops import conv_wrappers as CW
    x = _t((4, 32, 64, 32), dtype)
    if strided:
        x = x.permute(1, 0, 2, 3)                # a channel-plane view, read where it lies (or copied)
    w = _t((48, x.shape[1], 3, 3), dtype, 1)
    if strided:
        # the key tuned on a dense input of the same shape first: the sweep's screen compares a strided input with an
        # oracle that reads it dense and refuses every candidate (an autotune screen defect, not this cost's)
        CW.conv2d_wrapper(x.contiguous(), w, None, 1, 1, 1, False, 0, 1)
    measured = _measure(lambda: CW.conv2d_wrapper(x, w, None, 1, 1, 1, False, 0, 1))
    declared = CW.conv2d_cost(_spec(x), _spec(w), 1, 1, 1, 1).peak_bytes
    assert measured == declared, (measured, declared)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("strided_b,allow", [(False, False), (True, False), (True, True)])
def test_bmm(dtype, strided_b, allow):
    """`linalg_wrappers.bmm_cost`: the GEMM family's batched product — the matrix unit's dense operands, the casts of
    the SIMT route, the store dtype (fp32 for half on hardware without native bf16)."""
    from nbxtensor.ops import linalg_wrappers as LW
    a = _t((8, 128, 64), dtype)
    b = _t((8, 96, 64), dtype, 1).transpose(1, 2) if strided_b else _t((8, 64, 96), dtype, 1)
    measured = _measure(lambda: LW.bmm(a, b, allow_strided_b=allow))
    declared = LW.bmm_cost(_spec(a), _spec(b), allow_strided_b=allow).peak_bytes
    assert measured == declared, (measured, declared)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("mask,causal,hk", [(None, False, 4), ("2d", False, 4), ("4d", False, 4), (None, True, 4),
                                            (None, False, 2)])
def test_math_attention(dtype, mask, causal, hk):
    """`attention_wrappers.math_attention_cost`: the scores' product, the scale, the mask made dense, the softmax, the
    probabilities' cast, the output's product — as `_math_attention` allocates them (GQA included)."""
    from nbxtensor.ops import attention_wrappers as AW
    B, H, Tq, Tk, D = 2, 4, 96, 80, 64
    q = _t((B, H, Tq, D), dtype)
    k = _t((B, hk, Tk, D), dtype, 1)
    v = _t((B, hk, Tk, D), dtype, 2)
    m = None
    if mask == "2d":
        m = _t((Tq, Tk), dtype, 3)
    elif mask == "4d":
        m = _t((B, 1, Tq, Tk), dtype, 3)
    measured = _measure(lambda: AW._math_attention(q, k, v, attn_mask=m, is_causal=causal))
    declared = AW.math_attention_cost(_spec(q), _spec(k), _spec(v), None if m is None else _spec(m), causal).peak_bytes
    assert measured == declared, (measured, declared)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("causal", [False, True])
def test_chunked_math_attention(dtype, causal):
    from nbxtensor.ops import attention_wrappers as AW
    B, H, Tq, Tk, D = 2, 4, 96, 80, 64
    q, k, v = _t((B, H, Tq, D), dtype), _t((B, H, Tk, D), dtype, 1), _t((B, H, Tk, D), dtype, 2)
    measured = _measure(lambda: AW._math_attention_chunked(q, k, v, None, causal, None, 32))
    declared = AW.chunked_math_attention_cost(_spec(q), _spec(k), _spec(v), 32, None, causal).peak_bytes
    assert measured == declared, (measured, declared)
