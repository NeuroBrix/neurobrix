"""`NBXDtype` is an IntEnum and `NBXDtype.float16` is its member 0 — falsy. A `dtype` argument
tested by truthiness (`dtype if dtype else other._dtype`, `dtype or NBXDtype.float32`) drops an
explicit fp16 request and keeps the template's dtype (or fp32).

Found by the derived census (2026-10-03): MiniCPM-o's resampler builds its additive attention
mask as zeros_like(bool, dtype=bfloat16) -> masked_fill(-inf); under the fp16 contract the dtype
becomes NBXDtype.float16, `zeros_like` kept the bool template, and the walked baddbmm key
carries its bias as 'uint8' (a bool tensor) where the width pass derives 'fp16'. By construction
a masked -inf stored in a bool reads back as 1: an additive mask that adds +1 instead of masking.

Injection: this test is RED on the tree before the fix (the four sites tested `dtype` by
truthiness) — the fix's own proof; restoring any one site turns its case RED."""
from types import SimpleNamespace

import pytest

from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor


def _capture(monkeypatch, name):
    seen = {}

    def factory(shape, dtype=None, device=None):
        seen["dtype"] = dtype
        return SimpleNamespace(shape=shape, dtype=dtype)
    monkeypatch.setattr(NBXTensor, name, staticmethod(factory))
    return seen


@pytest.mark.parametrize("like,factory", [("zeros_like", "zeros"), ("ones_like", "ones"),
                                          ("empty_like", "empty")])
@pytest.mark.parametrize("want", [NBXDtype.float16, NBXDtype.bfloat16, NBXDtype.float32])
def test_a_like_op_honours_the_dtype_it_is_given(monkeypatch, like, factory, want):
    assert not bool(NBXDtype.float16)          # the trap itself: fp16 is the IntEnum's 0
    seen = _capture(monkeypatch, factory)
    other = SimpleNamespace(_shape=(2, 3), _dtype=NBXDtype.bool_, _device="cuda:0", _device_idx=0)
    getattr(NBXTensor, like)(other, dtype=want)
    assert seen["dtype"] == want


def test_a_like_op_without_a_dtype_follows_its_template(monkeypatch):
    seen = _capture(monkeypatch, "zeros")
    other = SimpleNamespace(_shape=(2,), _dtype=NBXDtype.bool_, _device="cuda:0", _device_idx=0)
    NBXTensor.zeros_like(other)
    assert seen["dtype"] == NBXDtype.bool_


def test_linspace_honours_an_explicit_fp16(monkeypatch):
    from neurobrix.kernels import dispatch
    seen = _capture(monkeypatch, "empty")
    dispatch._create_linspace(0.0, 1.0, 0, dtype=NBXDtype.float16)     # steps 0: no launch
    assert seen["dtype"] == NBXDtype.float16
