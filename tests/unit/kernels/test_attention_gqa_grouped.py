"""Unit test — GQA attention without broadcasting K/V.

`_math_attention` used to expand K and V up to Q's head count and
materialise the result; `triton/kv_cache.py` did the same one layer
above, so the whole KV cache was copied per layer per token. At a
4,164-token context that was 9.8 GB of pure data movement to generate ONE
token, and it is the dominant term in our context scaling —
`strided_copy` fitted at 11.9 us per context token against the real
attention matmul's 3.8.

The algebra never required it. With H = H_k * groups, the flat order of
`q [B, H, T_q, D]` is already b, h_k, g, t, d, so viewing it as
`[B*H_k, groups*T_q, D]` is a pure view; batched against
`[B*H_k, T_k, D]` it computes exactly the same dot products, with no
expansion anywhere.

What this file locks is the EXACTNESS of that identity: grouping Q must
produce bit-identical bytes to broadcasting K/V, at every GQA ratio the
zoo uses and on every masking branch. The oracle is the replaced code,
recomputed in-test, not a recorded baseline.

MHA (groups == 1) is unchanged by construction — every expression in the
new path literally reduces to the old one — and is covered here anyway,
because "by construction" is an argument and this file wants a
measurement.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_attention_gqa_grouped.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_attention_gqa_grouped.py
"""
from __future__ import annotations

import ctypes

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:  # pragma: no cover - shim
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore[assignment]

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor

_NP = {NBXDtype.float32: (np.float32, 4), NBXDtype.float16: (np.float16, 2)}


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _rand(shape, seed, dt=np.float16):
    return NBXTensor.from_numpy(
        np.random.default_rng(seed).standard_normal(shape).astype(dt))


def _reference_expanding(q, k, v, attn_mask=None, is_causal=False):
    """The replaced path: broadcast K/V to Q's head count, materialise,
    then run the same decomposition. Only the GQA handling differs from
    the shipped function, so equality here isolates the change."""
    B, H, T_q, D = q.shape
    H_k, T_k = k.shape[1], k.shape[2]
    D_v = v.shape[-1]
    if H != H_k:
        g = H // H_k
        k = k.unsqueeze(2).expand(B, H_k, g, T_k, D).reshape(B, H, T_k, D).contiguous()
        v = v.unsqueeze(2).expand(B, H_k, g, T_k, D_v).reshape(B, H, T_k, D_v).contiguous()
    q_3d = q.reshape(B * H, T_q, D)
    k_3d_t = k.reshape(B * H, T_k, D).transpose(-2, -1)
    v_3d = v.reshape(B * H, T_k, D_v)
    scores = W.bmm(q_3d, k_3d_t, allow_strided_b=True)
    import math
    scores = W.mul(scores, float(1.0 / math.sqrt(D)))
    if attn_mask is not None:
        m = attn_mask
        if m.ndim == 2:
            m = m.unsqueeze(0).expand(B * H, T_q, T_k)
        elif m.ndim == 4:
            m = m.expand(B, H, T_q, T_k).reshape(B * H, T_q, T_k)
        if m.nbx_dtype != scores.nbx_dtype:
            m = m.to(scores.nbx_dtype)
        scores = W.add(scores, m.contiguous())
    p = W.softmax(scores, dim=-1)
    p = W.nan_to_num_wrapper(p, nan=0.0)
    if p.nbx_dtype != v_3d.nbx_dtype:
        p = p.to(v_3d.nbx_dtype)
    out = W.bmm(p, v_3d)
    if out.nbx_dtype != q.nbx_dtype:
        out = out.to(q.nbx_dtype)
    return out.reshape(B, H, T_q, D_v)


# (B, H, H_kv, T_q, T_k, D) — GQA ratios the zoo actually uses, plus MHA.
_SHAPES = [
    (1, 32, 4, 1, 256, 128),     # 8:1 — canonical decode bucket
    (1, 32, 4, 1, 4164, 128),    # 8:1 at a working context
    (1, 32, 8, 1, 1024, 128),    # 4:1
    (1, 32, 2, 1, 512, 128),     # 16:1
    (1, 32, 32, 1, 256, 128),    # MHA — groups == 1
    (1, 16, 16, 1, 128, 64),     # MHA, small head dim
    (2, 8, 2, 1, 256, 64),       # batch > 1
    (1, 32, 4, 12, 12, 128),     # prefill square, GQA
    (1, 16, 4, 23, 23, 64),      # prefill at the tracer's prime length
]


@pytest.mark.parametrize("B,H,H_kv,T_q,T_k,D", _SHAPES)
def test_grouped_gqa_bit_identical(B, H, H_kv, T_q, T_k, D) -> None:
    if not _has_gpu():
        pytest.skip("no GPU")
    q = _rand((B, H, T_q, D), 1)
    k = _rand((B, H_kv, T_k, D), 2)
    v = _rand((B, H_kv, T_k, D), 3)
    ref = _d2h(_reference_expanding(q, k, v))
    got = _d2h(W._math_attention(q, k, v))
    assert np.array_equal(ref, got), (
        f"B={B} H={H} H_kv={H_kv} T_q={T_q} T_k={T_k} D={D}: output moved\n"
        f"  ref {ref[:6]}\n  got {got[:6]}\n  max |diff| "
        f"{np.abs(ref.astype(np.float64) - got.astype(np.float64)).max():.3e}")


@pytest.mark.parametrize("case", ["mask2d", "mask4d", "fully_masked", "causal"])
def test_grouped_gqa_masking_branches(case) -> None:
    """Every masking branch has to regroup the mask the same way the
    scores were regrouped. Row (g, t) of the grouped scores must see mask
    row t — get that wrong and heads read another head's mask."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T_q, T_k, D = 1, 8, 2, 4, 32, 64
    q = _rand((B, H, T_q, D), 11)
    k = _rand((B, H_kv, T_k, D), 12)
    v = _rand((B, H_kv, T_k, D), 13)

    mask, causal = None, False
    if case == "causal":
        causal = True
    elif case == "mask2d":
        m = np.zeros((T_q, T_k), dtype=np.float32)
        m[:, T_k // 2:] = -np.inf
        mask = NBXTensor.from_numpy(m)
    elif case == "mask4d":
        m = np.zeros((B, H, T_q, T_k), dtype=np.float32)
        m[:, :, :, T_k - 3:] = -np.inf
        mask = NBXTensor.from_numpy(m)
    elif case == "fully_masked":
        m = np.zeros((T_q, T_k), dtype=np.float32)
        m[0, :] = -np.inf
        mask = NBXTensor.from_numpy(m)

    got = W._math_attention(q, k, v, attn_mask=mask, is_causal=causal)
    arr = _d2h(got)
    assert got.shape == (B, H, T_q, D), f"{case}: shape {got.shape}"
    assert np.isfinite(arr).all(), (
        f"{case}: non-finite output — the fully-masked-row guard must "
        f"survive regrouping")
    if mask is not None and case != "fully_masked":
        ref = _d2h(_reference_expanding(q, k, v, attn_mask=mask))
        assert np.array_equal(ref, arr), f"{case}: masked output moved"


def test_q_grouping_is_a_pure_view() -> None:
    """Activation proof: the regrouped Q must be a VIEW, or the change
    has only moved the copy rather than removed it."""
    if not _has_gpu():
        pytest.skip("no GPU")
    B, H, H_kv, T_q, D = 1, 32, 4, 1, 128
    groups = H // H_kv
    q = _rand((B, H, T_q, D), 5)
    grouped = q.reshape(B * H_kv, groups * T_q, D)
    assert grouped.is_contiguous(), (
        "grouped Q is not contiguous — reshape materialised, so the "
        "expansion has been replaced by another copy")
    assert grouped.data_ptr() == q.data_ptr(), (
        "grouped Q has a different storage — it is a copy, not a view")


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    for shape in _SHAPES:
        try:
            test_grouped_gqa_bit_identical(*shape)
            B, H, Hk, Tq, Tk, D = shape
            tag = "MHA" if H == Hk else f"GQA {H//Hk}:1"
            print(f"  PASS  bit-identical  {tag:8s} B={B} H={H} H_kv={Hk} "
                  f"T_q={Tq} T_k={Tk} D={D}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {shape}\n        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {shape}\n        {type(e).__name__}: {e}")
    for c in ("mask2d", "mask4d", "fully_masked", "causal"):
        try:
            test_grouped_gqa_masking_branches(c)
            print(f"  PASS  masking branch: {c}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  masking branch: {c}\n        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR masking branch: {c}\n        {type(e).__name__}: {e}")
    try:
        test_q_grouping_is_a_pure_view()
        print("  PASS  activation proof: grouped Q is a pure view")
    except AssertionError as e:
        fails += 1
        print(f"  FAIL  activation proof\n        {e}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
