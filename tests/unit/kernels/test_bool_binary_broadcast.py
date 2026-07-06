"""Unit test — broadcast support in the bool binary wrappers.

Guard for the T5 extended-attention-mask outer product
(`mask[:, None, :, None] & mask[:, None, None, :]` -> [1,1,S,S], Allegro
text_encoder). The former wrappers ran the flat elementwise kernel over
`a.numel()` with no broadcast: `bitwise_and([1,1,S,1], [1,1,1,S])` returned
the elementwise diagonal [1,1,S,1] (same numel — no OOB crash, silently
wrong), and the downstream expand stretched the q-valid column over the key
axis — NO key masking, every real token attended every pad token in every
T5 layer (triton cond-branch 35.8% drift vs the compiled oracle at Allegro
native; uncond clean because the empty-prompt mask is all-ones = inert).

Also guards the 0-dim operand variant (`mask.all() & qside`, the graph's
aten.bitwise_and::0): the former code read the 0-dim tensor flat over
a.numel() elements — out-of-bounds past its 1-element allocation.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_bool_binary_broadcast.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_bool_binary_broadcast.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

    pytest = _NoPytest()  # type: ignore

import numpy as np
import torch

from neurobrix.kernels import wrappers as w
from neurobrix.kernels.nbx_tensor import NBXTensor, nbx_to_torch


def _nbx(arr):
    return NBXTensor.from_numpy(np.ascontiguousarray(arr))


_S = 13  # small odd size; outer product 13x13 != 13 catches any flat path

_CASES = [
    # (op-id, wrapper, torch_fn)
    ("bitwise_and", w.bitwise_and_wrapper, torch.bitwise_and),
    ("bitwise_or", w.bitwise_or_wrapper, torch.bitwise_or),
    ("logical_and", w.logical_and_wrapper, torch.logical_and),
    ("logical_or", w.logical_or_wrapper, torch.logical_or),
]


def _pad_mask():
    m = np.zeros(_S, dtype=bool)
    m[:4] = True
    return m


@pytest.mark.parametrize("cid,wrap,tfn", _CASES)
def test_outer_product_broadcast(cid, wrap, tfn):
    """[1,1,S,1] op [1,1,1,S] -> [1,1,S,S] (the T5 extended-mask shape)."""
    m = _pad_mask()
    a = m.reshape(1, 1, _S, 1)
    b = m.reshape(1, 1, 1, _S)
    ref = tfn(torch.from_numpy(a), torch.from_numpy(b))
    got = nbx_to_torch(wrap(_nbx(a), _nbx(b))).cpu()
    assert tuple(got.shape) == (1, 1, _S, _S), f"{cid}: shape {tuple(got.shape)}"
    assert torch.equal(got.bool(), ref.bool()), f"{cid}: values diverge"


@pytest.mark.parametrize("cid,wrap,tfn", _CASES)
def test_same_shape_unchanged(cid, wrap, tfn):
    """Anti-regression: the non-broadcast path stays exact."""
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=(2, _S)).astype(bool)
    b = rng.integers(0, 2, size=(2, _S)).astype(bool)
    ref = tfn(torch.from_numpy(a), torch.from_numpy(b))
    got = nbx_to_torch(wrap(_nbx(a), _nbx(b))).cpu()
    assert torch.equal(got.bool(), ref.bool()), f"{cid}: values diverge"


@pytest.mark.parametrize("scalar_val", [True, False])
def test_zero_dim_operand(scalar_val):
    """0-dim tensor operand (`mask.all() & qside`): correct, no OOB read."""
    m = _pad_mask().reshape(1, 1, _S, 1)
    s = np.array(scalar_val, dtype=bool)  # 0-dim
    ref = torch.bitwise_and(torch.from_numpy(s), torch.from_numpy(m))
    got = nbx_to_torch(w.bitwise_and_wrapper(_nbx(s), _nbx(m))).cpu()
    assert tuple(got.shape) == (1, 1, _S, 1)
    assert torch.equal(got.bool(), ref.bool())


def test_new_ones_fills_ones():
    """aten::new_ones must FILL ones — it returned UNINITIALIZED memory
    (leftover TODO). The Allegro/T5 extended-mask chain opens with
    `new_ones([], dtype=bool) & (positions >= 0)`; the uninitialized 0-dim
    byte read 0 -> False -> the AND zeroed the q-side -> extended mask
    masked ALL keys -> every T5 self-attention row uniform -> cond-branch
    embeds washed toward the pad mean (triton cond corr 0.9288 vs the
    sequential oracle, uncond clean)."""
    from neurobrix.kernels.nbx_tensor import NBXDtype
    base = _nbx(np.zeros((1, 1, _S, 1), dtype=np.int64))
    scalar = base.new_ones([], dtype=NBXDtype.bool_)   # the exact graph call
    assert tuple(scalar.shape) == ()
    assert bool(nbx_to_torch(scalar).item()) is True
    full = base.new_ones([3, 2])
    assert nbx_to_torch(full).cpu().tolist() == [[1, 1], [1, 1], [1, 1]]
    fp = base.new_ones([4], dtype=NBXDtype.float16)
    assert nbx_to_torch(fp).cpu().tolist() == [1.0, 1.0, 1.0, 1.0]


if __name__ == "__main__":
    for cid, wrap, tfn in _CASES:
        test_outer_product_broadcast(cid, wrap, tfn)
        test_same_shape_unchanged(cid, wrap, tfn)
    test_zero_dim_operand(True)
    test_zero_dim_operand(False)
    print("PASS")
