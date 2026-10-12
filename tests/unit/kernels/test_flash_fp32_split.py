"""Unit test — an fp32 attention on the m8n8k4 matrix unit through the hi + lo split keeps the fp32 path's accuracy.

Where the hardware profile declares `matrix_unit.fp32_split.flash` (volta.yml), the DtypeEngine
(`triton/dtype.matrix_unit_flash_representation`) carries an fp32 attention to the fp16 unit as hi + lo (Q, K, V and
the softmax's P, each under a power-of-two scale constant along its reduced axis): three MMAs per product, fp32
accumulation (ops/flash_attention_m8n8k4.py, SPLIT). Before it, an fp32 attention ran the scalar FMA flash
(`tl.dot(input_precision="ieee")`): 22.1 s/op at Allegro-TI2V's L 79 200, H 24, D 96 on V100, 2.6 TFLOP/s.

THE GATE (bounds written before the first measurement, 2026-10-12): per sampled query row, the error is
max_d |o - o64| / max_d |o64| against an fp64 oracle computed in-test on those rows over EVERY key. With E_fma the
same measure for the FMA path (the unit withheld) on the same inputs:
  E_split <= 4 E_fma + 4 eps32   (hi + lo carries 22 significant bits where fp32 carries 24: a factor 4), and
  E_split <= 1e-5                (fp32-level; the fp16 unit kernel is at ~1e-3, so a lost lo term cannot pass).
The absolute clause holds for well-conditioned scores (lengths, contract clauses). The first measurement showed fp32
ITSELF above 1e-5 at two range edges: E_fma 7.0e-4 beyond fp16 (|s| ~ 1e4, eps32 |s| ~ 1e-3) and 2.8e-5 for a peaked
softmax (|s| ~ 30); there the relative clause alone judges (split measured 2.8e-4 and 2.1e-5, below the FMA path).
At L in {8192, 79 200 (Allegro-TI2V's token count at 720x1280, 88 frames), 12 289 (prime-ish, no power of two)}, plus
the range edges (beyond fp16's 65504, below its normals, a peaked softmax) and the contract clauses (mask, GQA, a
head dim cut 64 + 16, tails). Removing the lo terms (seen failing on the injection recorded in the commit) fails it.

  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_flash_fp32_split.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from neurobrix.triton.dtype import matrix_unit_flash_representation as rep

_EPS32 = float(np.finfo(np.float32).eps)

_UNIT = {"operand_dtype": "float16", "flash": [{"block_m": 128}],
         "fp32_split": {"operands": ["float32", "bfloat16"], "mm": [{}], "flash": [{"block_m": 64}]}}


# ── the decision (CPU) ────────────────────────────────────────────────────────────────────────────────────────────
def test_the_representation_is_the_profiles_decision():
    assert rep("float16", "float16", "float16", _UNIT) == "native"
    assert rep("float32", "float32", "float32", _UNIT) == "split"
    assert rep("bfloat16", "bfloat16", "bfloat16", _UNIT) == "split"
    assert rep("float32", "float16", "float16", _UNIT) is None          # unaligned operands: never guessed
    assert rep("float64", "float64", "float64", _UNIT) is None
    no_split_flash = {**_UNIT, "fp32_split": {k: v for k, v in _UNIT["fp32_split"].items() if k != "flash"}}
    assert rep("float32", "float32", "float32", no_split_flash) is None  # the mm split alone does not take attention
    assert rep("float16", "float16", "float16", {k: v for k, v in _UNIT.items() if k != "flash"}) is None
    assert rep("float32", "float32", "float32", {}) is None


def test_a_split_flash_row_without_a_power_of_two_block_k_is_refused(monkeypatch):
    from neurobrix.kernels.ops import _configs as C
    prof = {"matrix_unit": {"shape": "m8n8k4", "operand_dtype": "float16", "accumulator_dtype": "float32",
                            "fp32_split": {"operands": ["float32"],
                                           "mm": [{"block_m": 64, "block_n": 64, "block_k": 32, "warps_m": 2,
                                                   "warps_n": 2, "group_m": 8}],
                                           "flash": [{"block_m": 64, "block_n": 32, "warps_m": 4, "block_k": 24}]}}}
    monkeypatch.setattr(C, "active_vendor_profile", lambda: prof)
    with pytest.raises(ValueError, match="block_k 24"):
        C.matrix_unit()
    prof["matrix_unit"]["fp32_split"]["flash"][0]["block_k"] = 32
    tile = C.matrix_unit_flash_tile(96, C.matrix_unit()["fp32_split"])
    assert tile == {"block_m": 64, "block_n": 32, "warps_m": 4, "block_k": 32}


# ── the kernel (GPU) ──────────────────────────────────────────────────────────────────────────────────────────────
def _split_declared() -> bool:
    try:
        from neurobrix.kernels import wrappers as W
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.set_device(0)
        return bool(((W._matrix_unit() or {}).get("fp32_split") or {}).get("flash"))
    except Exception:
        return False


gpu = pytest.mark.skipif(not _split_declared(), reason="the hardware profile declares no matrix_unit.fp32_split.flash")


def _oracle_rows(q, k, v, rows, bias, scale):
    """fp64 attention of the query rows `rows` over every key ([B, H, R, D]); bias [B, H, Tq, Tk] or 0."""
    groups = q.shape[1] // k.shape[1]
    k = np.repeat(k, groups, axis=1).astype(np.float64)
    v = np.repeat(v, groups, axis=1).astype(np.float64)
    qs = q[:, :, rows].astype(np.float64) * scale
    s = qs @ k.swapaxes(-1, -2) + (bias[:, :, rows] if np.ndim(bias) else 0.0)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    return (p @ v) / p.sum(-1, keepdims=True)


def _per_row(got, ref):
    got, ref = got.reshape(-1, ref.shape[-1]), ref.reshape(-1, ref.shape[-1])
    return float((np.abs(got - ref).max(axis=1) / np.maximum(np.abs(ref).max(axis=1), 1e-300)).max())


def _run(q, k, v, mask=None, unit=True):
    """(output, representation taken or None): the wrapper on the flash route, the unit withheld when `unit` is
    False (the FMA path an fp32 attention took before the split)."""
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.nbx_tensor import NBXTensor
    taken = []
    real_flash, real_route, real_unit = W._flash_m8n8k4, W._lk.sdpa_route, W._lk.unit_flash
    W._flash_m8n8k4 = lambda *a, **kw: (taken.append(kw.get("split")), real_flash(*a, **kw))[1]
    W._lk.sdpa_route = lambda *a, **kw: ("flash", 0)
    if not unit:
        W._lk.unit_flash = lambda *a, **kw: None
    nb = lambda x: NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")
    try:
        kw = {"attn_mask": nb(mask)} if mask is not None else {}
        out = W.scaled_dot_product_attention_wrapper(nb(q), nb(k), nb(v), k_pre_transposed=False, **kw)
        return out.numpy().astype(np.float64), (taken[0] if taken else None)
    finally:
        W._flash_m8n8k4, W._lk.sdpa_route, W._lk.unit_flash = real_flash, real_route, real_unit


def _gate(q, k, v, mask=None, bias=0.0, rows=None, conditioned=True):
    B, H, Tq, D = q.shape
    rows = np.arange(Tq) if rows is None else rows
    ref = _oracle_rows(q, k, v, rows, bias, 1.0 / np.sqrt(D))
    split, how = _run(q, k, v, mask)
    assert how is True, "the fp32 attention did not take the unit's split flash"
    fma, how = _run(q, k, v, mask, unit=False)
    assert how is None
    es, ef = _per_row(split[:, :, rows], ref), _per_row(fma[:, :, rows], ref)
    print(f"\n  E_split {es:.2e}  E_fma {ef:.2e}  ({B}x{H}x{Tq}x{D})")
    assert np.isfinite(split).all()
    assert es <= 4 * ef + 4 * _EPS32, f"split {es:.2e} vs fma {ef:.2e}"
    if conditioned:
        assert es <= 1e-5, f"split {es:.2e} is not fp32-level"


@gpu
@pytest.mark.parametrize("L", [8192, 12289, 79200])
def test_the_split_flash_keeps_fp32_accuracy_at_length(L):
    rng = np.random.default_rng(L)
    q, k, v = (rng.standard_normal((1, 1, L, 96)).astype(np.float32) for _ in range(3))
    _gate(q, k, v, rows=np.unique(np.r_[0, L - 1, rng.integers(0, L, 126)]))


_EDGES = {   # (Q, K, V factors, scores well-conditioned for fp32)
    "beyond fp16 (Q 1e5, V 1e6)": (1e5 / 96, 1.0, 1e6, False),
    "below fp16 normals (1e-8)": (1e-8, 1e-8, 1e-8, True),
    "a peaked softmax (scores ~ 30)": (5.5, 5.5, 1.0, False),
}


@gpu
@pytest.mark.parametrize("edge", list(_EDGES))
def test_range_edges(edge):
    rng = np.random.default_rng(len(edge))
    fq, fk, fv, conditioned = _EDGES[edge]
    q, k, v = (rng.standard_normal((1, 2, 1000, 96)).astype(np.float32) * f for f in (fq, fk, fv))
    _gate(q, k, v, conditioned=conditioned)


@gpu
@pytest.mark.parametrize("B,H,Hk,Tq,Tk,D,mask", [
    (1, 4, 2, 65, 200, 80, "keypad"),      # GQA 2, Tq != Tk, D 80 -> 64 + 16
    (2, 3, 3, 300, 300, 64, "none"),
    (1, 2, 2, 129, 129, 128, "none"),
    (1, 2, 1, 100, 77, 192, "none"),       # the profile's 192 and 256 rows
    (1, 1, 1, 70, 90, 256, "none"),
])
def test_the_contract_clauses(B, H, Hk, Tq, Tk, D, mask):
    rng = np.random.default_rng(Tq + D)
    q = rng.standard_normal((B, H, Tq, D)).astype(np.float32)
    k = rng.standard_normal((B, Hk, Tk, D)).astype(np.float32)
    v = rng.standard_normal((B, Hk, Tk, D)).astype(np.float32)
    m, bias = None, 0.0
    if mask == "keypad":
        m = np.zeros((B, 1, Tq, Tk), np.float32)
        m[..., Tk - Tk // 3:] = -np.inf
        bias = np.broadcast_to(m.astype(np.float64), (B, H, Tq, Tk))
    _gate(q, k, v, mask=m, bias=bias)
