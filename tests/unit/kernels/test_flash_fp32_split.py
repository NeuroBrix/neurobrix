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


def _profile(flash_row, smem=98304):
    return {"memory": {"max_shared_memory_per_block": smem},
            "matrix_unit": {"shape": "m8n8k4", "operand_dtype": "float16", "accumulator_dtype": "float32",
                            "fp32_split": {"operands": ["float32"],
                                           "mm": [{"block_m": 64, "block_n": 64, "block_k": 32, "warps_m": 2,
                                                   "warps_n": 2, "group_m": 8}],
                                           "flash": [flash_row]}}}


def test_a_split_flash_row_without_a_power_of_two_block_k_is_refused(monkeypatch):
    from nbxtensor.ops import _configs as C
    prof = _profile({"head_dim_le": 96, "block_m": 64, "block_n": 32, "warps_m": 4, "block_k": 24})
    monkeypatch.setattr(C, "active_vendor_profile", lambda: prof)
    with pytest.raises(ValueError, match="block_k 24"):
        C.matrix_unit()
    del prof["matrix_unit"]["fp32_split"]["flash"][0]["block_k"]
    with pytest.raises(ValueError, match="names no block_k"):          # a missing field is refused, not a KeyError
        C.matrix_unit()
    prof["matrix_unit"]["fp32_split"]["flash"][0]["block_k"] = 32
    tile = C.matrix_unit_flash_tile(96, C.matrix_unit()["fp32_split"])
    assert tile == {"block_m": 64, "block_n": 32, "warps_m": 4, "block_k": 32}


def test_a_flash_row_over_the_profiles_shared_memory_never_loads(monkeypatch):
    """The door of 2026-10-12: a row the kernel cannot stage within `memory.max_shared_memory_per_block` is refused
    by name when the profile is read, not by the compiler at the first launch that reaches it. The split doubles the
    staged tiles (Q, K, V as hi + lo): head dim 256 at 64 x 16 stages 4 x 256 x (64 + 32) = 98 304 bytes — the whole
    V100 budget — and 64 x 32 is over it; the native row at the same tile is half that."""
    from nbxtensor.ops import _configs as C
    from nbxtensor.ops.flash_attention_m8n8k4 import head_parts, staged_shared_bytes
    assert head_parts(256) == (256, 0) and head_parts(96) == (64, 32) and head_parts(80) == (64, 16)
    assert staged_shared_bytes(64, 16, 256, 0, split=True) == 98304
    assert staged_shared_bytes(64, 16, 256, 0, split=False) == 49152
    assert staged_shared_bytes(64, 32, 64, 32, split=True) == 4 * 96 * (64 + 64)
    prof = _profile({"head_dim_le": 256, "block_m": 64, "block_n": 32, "warps_m": 4, "block_k": 32})
    monkeypatch.setattr(C, "active_vendor_profile", lambda: prof)
    with pytest.raises(ValueError, match="stages 131072 bytes of shared memory, over memory.max_shared_memory_per_block 98304"):
        C.matrix_unit()
    prof["matrix_unit"]["fp32_split"]["flash"][0]["block_n"] = 16                 # exactly the budget: loads
    assert C.matrix_unit_flash_tile(256, C.matrix_unit()["fp32_split"])["block_n"] == 16
    prof["matrix_unit"]["flash"] = [{"head_dim_le": 256, "block_m": 128, "block_n": 64, "warps_m": 4}]   # native, 131072
    with pytest.raises(ValueError, match=r"matrix_unit\.flash row head_dim_le 256 128x64 stages 131072"):
        C.matrix_unit()
    prof["matrix_unit"]["flash"] = [{"block_m": 64, "block_n": 32, "warps_m": 4}]                       # unpriceable
    with pytest.raises(ValueError, match="names no head_dim_le"):
        C.matrix_unit()
    prof["matrix_unit"]["flash"] = [{"head_dim_le": 64, "block_m": 128, "block_n": 64, "warps_m": 4}]
    del prof["memory"]                      # a profile that did not say its budget: the rows are not priced here
    prof["matrix_unit"]["fp32_split"]["flash"][0]["block_n"] = 32
    assert C.matrix_unit()["fp32_split"]["flash"][0]["block_n"] == 32


def test_the_committed_volta_rows_all_load_under_their_own_budget():
    """Every row volta.yml declares, native and split, stages within the 96 KB it declares (the door above would
    otherwise refuse the shipped profile at import)."""
    import yaml
    from pathlib import Path
    from nbxtensor.ops import _configs as C
    from nbxtensor.ops.flash_attention_m8n8k4 import head_parts, staged_shared_bytes
    import neurobrix
    prof = yaml.safe_load((Path(neurobrix.__file__).parent / "config/vendors/nvidia/volta.yml").read_text())
    budget = prof["memory"]["max_shared_memory_per_block"]
    mu = prof["matrix_unit"]
    rows = [(r, False) for r in mu["flash"]] + [(r, True) for r in mu["fp32_split"]["flash"]]
    assert mu["flash"] and mu["fp32_split"]["flash"], "both sections hold rows"
    for r, split in rows:
        assert staged_shared_bytes(r["block_m"], r["block_n"], *head_parts(r["head_dim_le"]), split=split) <= budget, r


# ── the kernel (GPU) ──────────────────────────────────────────────────────────────────────────────────────────────
def _no_gpu_reason():
    """Why the GPU cells cannot run here, or None. Only the ABSENCE of a device or of the profile's split rows skips:
    a broken kernel module, a refused profile or a failing allocator raises at collection, never reads as a skip."""
    from nbxtensor.ops import wrappers as W
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.backend.runtime import _gpu_runtime
    try:
        _gpu_runtime()
    except (OSError, RuntimeError) as exc:           # no CUDA/HIP runtime library on this host
        return f"no GPU runtime: {exc}"
    if DeviceAllocator.device_count() < 1:           # a runtime with no visible device (CUDA_VISIBLE_DEVICES="")
        return "the process sees no device"
    DeviceAllocator.set_device(0)
    if not ((W._matrix_unit() or {}).get("fp32_split") or {}).get("flash"):
        return "the hardware profile declares no matrix_unit.fp32_split.flash"
    return None


_SKIP = _no_gpu_reason()
gpu = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


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


def _run(q, k, v, mask=None, unit=True, dtype=None):
    """(output, representation taken or None): the wrapper on the flash route, the unit withheld when `unit` is
    False (the FMA path an fp32 attention took before the split). `dtype`: the operands cast on the device to it
    (bfloat16: the values must already be bf16-representable), the output read back through float32."""
    from nbxtensor.ops import wrappers as W
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    taken = []
    real_flash, real_route, real_unit = W._flash_m8n8k4, W._lk.sdpa_route, W._lk.unit_flash
    W._flash_m8n8k4 = lambda *a, **kw: (taken.append(kw.get("split")), real_flash(*a, **kw))[1]
    W._lk.sdpa_route = lambda *a, **kw: ("flash", 0)
    if not unit:
        W._lk.unit_flash = lambda *a, **kw: None

    def nb(x):
        t = NBXTensor.from_numpy(np.ascontiguousarray(x)).to("cuda:0")
        return t.to(dtype) if dtype is not None else t
    try:
        kw = {"attn_mask": nb(mask)} if mask is not None else {}
        out = W.scaled_dot_product_attention_wrapper(nb(q), nb(k), nb(v), k_pre_transposed=False, **kw)
        if dtype is not None:
            assert out.nbx_dtype == dtype, f"the output is {out.nbx_dtype}, not the operands' {dtype}"
            out = out.to(NBXDtype.float32)
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


_BF16_ULP = 2.0 ** -7      # bfloat16: 7 stored significand bits (8 with the implicit one): values 2^e apart by 2^(e-7)


def _bf16_round(x):
    from nbxtensor.dtype.convert import bf16_bits_to_f32 as bf16_carrier_to_float32, f32_to_bf16_bits as float32_to_bf16_bits
    bits = float32_to_bf16_bits(np.ascontiguousarray(x, dtype=np.float32))
    return bf16_carrier_to_float32(np.asarray(bits, dtype=np.uint16).view("V2")).astype(np.float64)


@gpu
@pytest.mark.parametrize("L,D", [(4096, 96), (1000, 128)])
def test_a_bfloat16_attention_through_the_split_is_fp32_level_inside(L, D):
    """`fp32_split.operands` lists bfloat16: a bf16 attention takes the split too (its output is bf16, so the fp32
    clauses above cannot read it). THE GATE, written before the measurement: an output element may differ from the
    correctly rounded bf16 of the fp64 oracle only where the oracle lies within the kernel's internal error of a
    rounding boundary. With an fp32-level inside (~4 eps32 relative) that is a fraction ~ 2 x 4 eps32 / ulp_bf16
    = 1e-3 of the elements, every one by ONE bf16 ulp; with an fp16-level inside (the lo terms dropped, ~1e-3
    relative) half the elements would differ. Bound: at most 1 % of the elements differ, each by at most one ulp
    of the binade the oracle OR the result lies in (a value rounded up into the next binade is one ulp of THAT
    binade away) plus the inside's fp32-level error taken against the UNCANCELLED weighted sum sum_k p_k |v_k| /
    sum p — a dot product's error scales with that sum, not with its result, so an element that cancels to near
    zero keeps the row's absolute error, far above its own ulp (first measurements 2026-10-12: 0.02-0.04 % off,
    maxima exactly 2.00 'ulp' and 10 'ulp' on a near-zero element — the first because the gate's ulp constant was
    2^-8 where bfloat16's spacing is 2^-7 (7 stored bits), so every one-ulp element read as two; the second the
    cancellation above). The FMA path (unit withheld) on the same bf16 inputs is the Dao kernel with bf16 operands: it
    rounds the scaled Q and the probabilities P to bf16 before their products (ops/flash_attention.py, `p = p.to(v.dtype)`),
    as the vendor's own bf16 attention does on this hardware (the math path: bf16 scores, bf16 P) — a bf16-level inside,
    so about half its elements differ from the correctly rounded oracle (measured 2026-10-12: 51.6 %, max 478 ulp of a
    cancelling element's binade). It is held to THAT inside: one ulp plus 2^-8 of the uncancelled sum, no fraction
    clause. The split path is the one that is fp32-level inside."""
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    rng = np.random.default_rng(L + D)
    q32, k32, v32 = (rng.standard_normal((1, 2, L, D)).astype(np.float32) for _ in range(3))
    q, k, v = (_bf16_round(x).astype(np.float32) for x in (q32, k32, v32))     # the bf16 values, exactly
    rows = np.unique(np.r_[0, L - 1, rng.integers(0, L, 126)])
    ref = _oracle_rows(q, k, v, rows, 0.0, 1.0 / np.sqrt(D))
    uncancelled = _oracle_rows(q, k, np.abs(v), rows, 0.0, 1.0 / np.sqrt(D))   # same p, |v|: the sum before cancellation
    for unit in (True, False):
        got, how = _run(q, k, v, unit=unit, dtype=NBXDtype.bfloat16)
        assert (how is True) if unit else (how is None)
        g = got[:, :, rows]
        off = np.abs(g - _bf16_round(ref))
        ulp = _BF16_ULP * 2.0 ** np.floor(np.log2(np.maximum(np.maximum(np.abs(ref), np.abs(g)), 1e-300)))
        inside = 4 * _EPS32 if unit else 2.0 ** -8          # fp32-level through the split; bf16-level Q and P on the FMA path
        bound = ulp + inside * uncancelled
        frac = float((off > 0).mean())
        print(f"\n  bf16 {'split' if unit else 'FMA'} {L}x{D}: {frac * 100:.3f} % of elements off the rounded oracle, "
              f"max {float((off / ulp).max()):.2f} ulp of the binade, max {float((off / bound).max()):.2f} of the bound")
        assert np.isfinite(g).all()
        assert (off <= bound * (1 + 1e-9)).all(), \
            "an element further from the rounded oracle than one ulp of its binade plus the inside's error on the uncancelled sum"
        if unit:
            assert frac <= 0.01, f"{frac * 100:.2f} % of the elements off the correctly rounded oracle"


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
