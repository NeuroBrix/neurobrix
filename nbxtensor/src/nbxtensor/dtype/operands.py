"""The operand decisions of the kernels, as dtype names (DESIGN rule 12: the mechanism in the library): the dtype a
contraction's partials are summed in, how a matrix unit carries an operand, how an attention's operands are aligned
when they disagree, the scale of a split representation. Pure functions of dtype names and of a device's matrix unit
(`ops._configs.matrix_unit`); the application decides which device and which policy, never these rules.
"""
from __future__ import annotations

from typing import Optional

from nbxtensor.dtype import itemsize as _itemsize

__all__ = ["contraction_accumulator_dtype", "contraction_accumulator_bytes", "matrix_unit_representation",
           "matrix_unit_flash_representation", "ATTENTION_OPERAND_ALIGNMENTS", "attention_operand_alignment",
           "attention_operand_dtypes", "matrix_unit_split_scale", "constant_load_dtype"]


def contraction_accumulator_dtype(dtype: str) -> str:
    """The dtype (a name) the partial results of a contraction split over the axis it reduces
    are summed in, when a stretch runs in slices of that axis (`core/strategies/chunked_piece`):
    the split contraction is the SAME contraction, so its partials are summed in the accumulator
    its kernel keeps — float32 for a half dtype (every mm/bmm/conv kernel accumulates fp32, and
    `sum` is an AMP_FP32 op), the dtype itself for float32 and float64 — and the sum is stored
    once, in the dtype the contraction's own output was given. A partial summed at the pass
    dtype rounds once per slice where the whole op rounds once. Any other dtype is refused by
    name: a contraction over a token axis yields a floating value.

    The one definition both engines read."""
    if dtype in ("float16", "bfloat16"):
        return "float32"
    if dtype in ("float32", "float64"):
        return dtype
    raise ValueError(f"contraction_accumulator_dtype: {dtype!r} is not a floating dtype a "
                     f"contraction's partials are summed in")


def contraction_accumulator_bytes(store_bytes: int) -> int:
    """`contraction_accumulator_dtype` read on a WIDTH (bytes per element), for Prism, which prices
    a plan in widths: a 2-byte store (fp16 or bf16 alike) accumulates in 4 bytes; 4 and 8 are their
    own accumulator. The same rule (a unit test holds the two equal over every floating dtype)."""
    if store_bytes == 2:
        return 4
    if store_bytes in (4, 8):
        return store_bytes
    raise ValueError(f"contraction_accumulator_bytes: a {store_bytes}-byte store is no floating "
                     f"dtype a contraction's partials are summed in")


def matrix_unit_representation(dtype: str, unit: dict) -> Optional[str]:
    """How the hardware profile's matrix unit (`matrix_unit`, nbxtensor.ops._configs.matrix_unit) carries one GEMM
    operand whose dtype in memory is `dtype` (a name): "native" when it IS the unit's operand dtype (one MMA term
    per operand), "split" when the profile declares `matrix_unit.fp32_split` with this dtype among its `operands`
    (the value, widened to fp32, is carried as hi + lo of the operand dtype under a power-of-two scale, see
    `matrix_unit_split_scale`; hi*hi + hi*lo + lo*hi with fp32 accumulation outside the unit, the vendor's fp32
    arithmetic kept), else None (the GEMM keeps its tl.dot kernel). The ONE decision: the wrappers launch with it
    and the derived census keys with it (nbxtensor.ops.launch_keys.matrix_unit_operands)."""
    if not unit or not unit.get("mm"):
        return None
    if dtype == unit["operand_dtype"]:
        return "native"
    split = unit.get("fp32_split") or {}
    if split.get("mm") and dtype in (split.get("operands") or ()):
        return "split"
    return None


def matrix_unit_flash_representation(q: str, k: str, v: str, unit: dict) -> Optional[str]:
    """How the profile's matrix unit carries an attention's three operands (names, already aligned by
    `attention_operand_dtypes`) in its flash kernel (nbxtensor/ops/flash_attention_m8n8k4.py): "native" when all three
    ARE the unit's operand dtype and the profile lists `matrix_unit.flash` rows; "split" when all three are among
    `matrix_unit.fp32_split.operands` and the profile lists `matrix_unit.fp32_split.flash` rows — Q, K, V and the
    softmax's P carried as hi + lo as `matrix_unit_representation` carries a GEMM operand, the vendor's fp32
    arithmetic kept; else None (the attention keeps its tl.dot kernel). The ONE decision the wrapper launches with
    and the derived census routes with (nbxtensor.ops.launch_keys.unit_flash)."""
    if not unit or not (q == k == v):
        return None
    if q == unit["operand_dtype"]:
        return "native" if unit.get("flash") else None
    split = unit.get("fp32_split") or {}
    if split.get("flash") and q in (split.get("operands") or ()):
        return "split"
    return None


#: The ways a hardware profile may align attention operands that disagree in dtype
#: (`precision.attention_operands`): every operand takes the narrowest of the three, or the widest.
ATTENTION_OPERAND_ALIGNMENTS = ("narrowest", "widest")
#: The operand dtypes the alignment chooses among; their widths are the dtype table's.
_ATTENTION_OPERAND_DTYPES = ("float16", "bfloat16", "float32", "float64")


def attention_operand_alignment(profile: dict) -> str:
    """The alignment the hardware profile declares for attention operands that disagree in dtype
    (`precision.attention_operands`, one of ATTENTION_OPERAND_ALIGNMENTS). A profile that declares
    none, or another word, is refused by name: the choice moves a whole attention between the
    matrix unit and the scalar units, so it is never guessed.

    The one definition both engines read."""
    v = (profile.get("precision") or {}).get("attention_operands")
    if v not in ATTENTION_OPERAND_ALIGNMENTS:
        raise ValueError(f"hardware profile {profile.get('architecture')!r}: "
                         f"precision.attention_operands is {v!r}, not one of "
                         f"{ATTENTION_OPERAND_ALIGNMENTS}")
    return v


def attention_operand_dtypes(q: str, k: str, v: str, alignment: str,
                             q_round: Optional[str] = None) -> tuple:
    """The (q, k, v, q_round) dtypes (names) an attention computes with — the ONE decision both
    engines' attention reads (R30): the Triton wrapper (`launch_keys.sdpa_operand_dtypes`), the
    derived census and Prism's width estimate through it, and the compiled engine through its
    twin. The kernels need matching operands: three that already match (Q judged at its KV-cache
    rounding `q_round` when the cache asks for one) are returned unchanged; three that disagree
    all take the narrowest or the widest of them, as the profile's `alignment` declares
    (`attention_operand_alignment`), the first in (q, k, v) order on a tie, and q_round is
    dropped. A non-floating operand is refused by name.

    Measured on V100 (2026-10-08, nbx/campaigns/2026_10_08_allegro_prof/probe): an fp16 attention
    whose V alone came out of an fp32-stored GEMM ran 22.0 s/op on the scalar flash when aligned
    to the widest, 1.16 s/op on the matrix-unit flash when aligned to the narrowest.

    The one definition both engines read."""
    qj = q_round if q_round is not None else q
    for d in (qj, k, v):
        if d not in _ATTENTION_OPERAND_DTYPES:
            raise ValueError(f"attention_operand_dtypes: {d!r} is not a floating dtype an "
                             f"attention computes in")
    if qj == k == v:
        return q, k, v, q_round
    if alignment not in ATTENTION_OPERAND_ALIGNMENTS:
        raise ValueError(f"attention_operand_dtypes: alignment {alignment!r} is not one of "
                         f"{ATTENTION_OPERAND_ALIGNMENTS}")
    pick = min if alignment == "narrowest" else max
    t = pick((qj, k, v), key=_itemsize.itemsize)
    return t, t, t, None


def matrix_unit_split_scale(unit: dict) -> tuple:
    """(hi_exp, lo_shift) of the split representation, derived from the unit's operand dtype (never written in a
    kernel): each operand tile is scaled by the power of two that brings its largest magnitude into
    [2^hi_exp, 2^(hi_exp+1)) — the top binade below the operand dtype's overflow, so no hi overflows and every
    value down to 2^-(hi_exp+1+|min exponent|) of the tile's largest stays normal — and lo, the residual
    x - hi, is stored times 2^lo_shift (the operand dtype's significand width) so it does not underflow where hi
    does not (Ootomo & Yokota 2022, arXiv 2203.03341, eq. 19-24)."""
    import numpy as np
    fi = np.finfo(np.dtype(unit["operand_dtype"]))
    return int(fi.maxexp) - 2, int(fi.nmant) + 1


def constant_load_dtype(traced: str, compute: str, stores_fp64: bool) -> str:
    """The dtype (a name) an embedded graph constant is bound in by the Triton engines
    (`GraphExecutor._load_constant_triton`): a bfloat16 constant is decoded to the compute dtype
    when that is half (fp16 bits from bf16 on hardware computing fp16, the bf16 bits kept under
    bf16), to float32 otherwise; float64 / complex128 are held as the branch holds them
    (`dtype.itemsize.storage_dtype`); every other dtype is kept as traced. The loader and Prism's width
    pass both ask it."""
    if traced == "bfloat16":
        return compute if compute in ("float16", "bfloat16") else "float32"
    return _itemsize.storage_dtype(traced, stores_fp64)
