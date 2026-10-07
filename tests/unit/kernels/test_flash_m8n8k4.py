"""Unit test — the flash forward on the m8n8k4 matrix unit holds the wrapper's whole flash contract.

Where the hardware profile declares `matrix_unit` (volta.yml), the SDPA wrapper's flash route runs
`kernels/ops/flash_attention_m8n8k4.py` instead of the `tl.dot` kernel. That kernel re-implements every
clause of the contract — Q scaled and rounded to its dtype, the additive memory-resident bias (no mask,
key padding, causal), GQA groups, query and key tails, a head dim cut into two powers of two, and a
fully-masked row left to the wrapper's guard — so each clause is proven here against an fp64 oracle
computed in-test, and the route is asserted TAKEN (a test that ran the `tl.dot` kernel would prove
nothing about this one). Skipped where the profile declares no matrix unit.

  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_flash_m8n8k4.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor


def _unit_declared() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return bool(W._matrix_unit())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _unit_declared(), reason="the hardware profile declares no matrix_unit")


def _oracle(q, k, v, bias, scale):
    groups = q.shape[1] // k.shape[1]
    k = np.repeat(k, groups, axis=1)
    v = np.repeat(v, groups, axis=1)
    qs = (q * scale).astype(np.float16).astype(np.float64)       # the contract: Q rounded to its dtype
    s = qs @ k.swapaxes(-1, -2) + bias
    m = s.max(-1, keepdims=True)
    full = np.isneginf(m)                                        # a fully-masked row: the guard gives 0
    p = np.exp(np.where(full, 0.0, s - np.where(full, 0.0, m)))
    p = np.where(full, 0.0, p)
    l = p.sum(-1, keepdims=True)
    return np.where(full, 0.0, (p @ v) / np.where(l == 0, 1.0, l))


CASES = [
    # B, H, Hk, Tq, Tk, D, mask
    (1, 2, 2, 777, 777, 96, "none"),      # D split 64 + 32, both tails
    (2, 3, 3, 300, 300, 64, "none"),
    (1, 2, 2, 129, 129, 128, "causal"),
    (1, 4, 2, 65, 200, 80, "keypad"),      # GQA 2, Tq != Tk, D 80 -> 64 + 16
    (1, 2, 2, 50, 70, 64, "fullrow"),      # one query row with every key masked
    (1, 8, 1, 33, 513, 256, "none"),       # MQA, the largest head dim of the measured row
]


@pytest.mark.parametrize("B,H,Hk,Tq,Tk,D,mask", CASES)
def test_m8n8k4_flash_matches_fp64(B, H, Hk, Tq, Tk, D, mask, monkeypatch):
    calls = []
    real = W._flash_m8n8k4
    monkeypatch.setattr(W, "_flash_m8n8k4", lambda *a, **kw: (calls.append(a[0].shape), real(*a, **kw))[1])
    monkeypatch.setattr(W._lk, "sdpa_route", lambda *a, **kw: ("flash", 0))
    rng = np.random.default_rng(B * 1000 + Tq + D)
    q = (rng.standard_normal((B, H, Tq, D)) * 0.5).astype(np.float16)
    k = (rng.standard_normal((B, Hk, Tk, D)) * 0.5).astype(np.float16)
    v = (rng.standard_normal((B, Hk, Tk, D)) * 0.5).astype(np.float16)
    bias = np.zeros((B, H, Tq, Tk))
    nb = lambda a: NBXTensor.from_numpy(a).to("cuda:0")
    kw = {}
    if mask == "causal":
        bias = np.where(np.tril(np.ones((Tq, Tk), bool)), 0.0, -np.inf)[None, None].repeat(B, 0).repeat(H, 1)
        kw["is_causal"] = True
    elif mask in ("keypad", "fullrow"):
        m = np.zeros((B, 1, Tq, Tk), np.float16)
        m[..., Tk - Tk // 3:] = -np.inf                         # padded keys
        if mask == "fullrow":
            m[..., 7, :] = -np.inf
        bias = np.broadcast_to(m.astype(np.float64), (B, H, Tq, Tk))
        kw["attn_mask"] = nb(m)
    out = W.scaled_dot_product_attention_wrapper(nb(q), nb(k), nb(v), k_pre_transposed=False, **kw)
    assert calls, "the m8n8k4 route was not taken"
    got = out.numpy().astype(np.float64)
    ref = _oracle(q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), bias, 1.0 / D ** 0.5)
    assert np.isfinite(got).all()
    err = np.abs(got - ref).max()
    assert err < 2e-3, f"max|diff| {err:.2e} vs fp64 ({mask}, D {D})"


def test_over_the_scores_budget_the_route_is_the_units_flash_and_deterministic(monkeypatch):
    """No route forced: once the fp32 scores exceed the device's budget, `sdpa_route` gives the unit's flash
    (`unit_flash_takes`) instead of the chunked math — and the kernel is deterministic by construction (no atomics,
    a fixed reduction order), the property the math route exists for on this arch: three runs, byte-identical.
    The budget is set small so a test-sized call is over it (the data the route reads, not the route)."""
    from neurobrix.kernels import launch_keys as LK
    from neurobrix.kernels.nbx_tensor import NBXDtype
    h = NBXDtype.float16
    assert LK.unit_flash_takes(96, h, h, h) and not LK.unit_flash_takes(96, NBXDtype.float32, h, h)
    calls = []
    real = W._flash_m8n8k4
    monkeypatch.setattr(W, "_flash_m8n8k4", lambda *a, **kw: (calls.append(a[0].shape), real(*a, **kw))[1])
    monkeypatch.setattr(W, "_sdpa_math_scores_budget_bytes_for", lambda *_: 1 << 20)
    rng = np.random.default_rng(5)
    B, H, T, D = 1, 2, 1024, 96
    q, k, v = ((rng.standard_normal((B, H, T, D)) * 0.5).astype(np.float16) for _ in range(3))
    nb = lambda x: NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")
    outs = [W.scaled_dot_product_attention_wrapper(nb(q), nb(k), nb(v)).numpy() for _ in range(3)]
    assert len(calls) == 3, f"the unit's flash ran {len(calls)} of 3 times"
    assert all(np.array_equal(o.view(np.uint8), outs[0].view(np.uint8)) for o in outs[1:])
    ref = _oracle(q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), 0.0, 1.0 / np.sqrt(D))
    assert np.abs(outs[0].astype(np.float64) - ref).max() < 2e-3
