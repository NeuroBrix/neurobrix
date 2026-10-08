"""The TilingEngine's sizing half — torch-free, imported by Prism, the launch keys, the Triton
wrappers and the TilingEngine's executor.

This copy serves the PyTorch branch (Prism, the core executors and the TilingEngine). Its twin
`neurobrix/triton/tiling_sizes.py` serves the Triton branch (the wrappers, the launch keys and
the derived census through them): the two engines share no compute code, so each carries its own
copy, held equal function by function by `tests/unit/tiling/test_the_two_tiling_sizes_are_one.py`.

Every size a memory split is cut by is computed here, from `config/tiling.yml` (engine-wide
values, each with its source) and from the values the caller reads in the active hardware
profile (per-architecture values stay there). Pure arithmetic: Prism prices with it at plan time,
the launch keys and the census derive with it, the wrappers and the executor cut with it — one
definition each, so the plan, the key and the run agree.

What it owns:
  * the conv3d one-shot peak and its temporal chunk (a PLAN decision: `need` against the op
    budget; the wrapper sizes the chunks with the same function);
  * the conv2d band cut;
  * the attention scores bound and its chunk rows;
  * Prism's op budget and its op-level band factors (conv, fusion pair, rms_norm, residual chain);
  * the tile overlaps (spatial and temporal);
  * the in-place and residual-chain thresholds.
"""
from __future__ import annotations

import math
from typing import Any, Sequence, Tuple


def _value(*path: str) -> Any:
    """One value of config/tiling.yml, refused by name when absent."""
    from neurobrix.core.config.loader import get_tiling_policy
    node: Any = get_tiling_policy()
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"ZERO FALLBACK: config/tiling.yml states no {'.'.join(path[:i + 1])}")
        node = node[key]
    return node


def _pow2_at_most(factor: int, cap: int) -> int:
    """The smallest power of two >= `factor`, at most `cap` (a band factor aligned for halos)."""
    p = 1
    while p < factor and p < cap:
        p *= 2
    return p


# --- conv3d: the one-shot peak and the temporal chunk ---------------------------------------------

def conv3d_chunk_bytes() -> int:
    """The per-transient bound of the chunked conv3d path (`conv3d.chunk_bytes`)."""
    return int(_value("conv3d", "chunk_bytes"))


def conv2d_band_bytes() -> int:
    """The conv2d band-streaming threshold (`conv2d.band_bytes`)."""
    return int(_value("conv2d", "band_bytes"))


def _triple(v) -> Tuple[int, int, int]:
    if isinstance(v, (list, tuple)):
        v = list(v) + [v[-1]] * (3 - len(v))
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


def conv3d_need(x_shape: Sequence[int], w_shape: Sequence[int], stride, padding, dilation,
                in_bytes: int, out_bytes: int) -> Tuple[int, int]:
    """(bytes the ONE-SHOT path holds at its peak, folded bytes per output frame). `need` is 0 when
    no fold exceeds `conv3d_chunk_bytes()` — the one-shot path is taken whatever the budget. The
    peak: the temporal pad copy, then the larger of (input fold + output fold [+ a band when the
    output bands] [+ the kt accumulator]) and (two output folds [+ the accumulator]), and a fixed
    slack (`conv3d.slack_bytes`). The wrapper runs conv3d as kt conv2d launches over the folded
    batch axis (B*T_out frames); whether it chunks is Prism's decision, made from this `need`
    against the op budget (decision 2, 2026-09-29: never from the driver's free bytes)."""
    B, Cin, T, H, W = (int(d) for d in x_shape)
    Cout, _Cg, kt, kh, kw = (int(d) for d in w_shape)
    st, sh, sw = _triple(stride)
    pt, ph, pw = _triple(padding)
    dt, dh, dw = _triple(dilation)
    t_out = (T + 2 * pt - dt * (kt - 1) - 1) // st + 1
    if t_out <= 0:
        return 0, 0
    oh = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    fold_in = B * t_out * Cin * H * W * in_bytes
    fold_out = B * t_out * Cout * oh * ow * out_bytes
    frame = max(fold_in, fold_out) // max(1, t_out)
    if max(fold_in, fold_out) <= conv3d_chunk_bytes():
        return 0, frame
    band = conv2d_band_bytes()
    pad_bytes = B * Cin * (T + 2 * pt) * H * W * in_bytes if pt > 0 else 0
    band_extra = band if fold_out > band else 0
    acc_extra = fold_out if kt >= 2 else 0
    need = (pad_bytes + max(fold_out + fold_in + band_extra + acc_extra, 2 * fold_out + acc_extra)
            + int(_value("conv3d", "slack_bytes")))
    return need, frame


def chunk_frames(frame_bytes: int) -> int:
    """Output frames per chunk: each chunk's folded transient under `conv3d_chunk_bytes()`."""
    return max(1, int(conv3d_chunk_bytes() // max(1, frame_bytes)))


# --- conv2d: the band cut -------------------------------------------------------------------------

def conv2d_band_rows(N: int, out_c: int, out_h: int, out_w: int, out_bytes: int,
                     band_bytes: int) -> Tuple[int, int]:
    """(output rows of one band, bytes of one output row) of conv2d band streaming: bands of
    about `band_bytes / conv2d.band_budget_divisor`, evenly cut. A band >= `out_h` means one row
    is already over the budget — the caller refuses it."""
    band_target = max(1, band_bytes // int(_value("conv2d", "band_budget_divisor")))
    row_bytes = N * out_c * out_w * out_bytes
    rows_per_band = max(1, band_target // max(1, row_bytes))
    tile_factor = max(1, (out_h + rows_per_band - 1) // rows_per_band)
    return (out_h + tile_factor - 1) // tile_factor, row_bytes


# --- attention: the scores bound, the chunk rows, the device budget -------------------------------

#: The math route's scores are fp32: four bytes a score (the width of the dtype, not a size choice).
_SCORES_BYTES = 4


def sdpa_scores_bytes(batch: int, nheads: int, Tq: int, Tk: int) -> int:
    """The bytes of an attention's fp32 scores."""
    return batch * nheads * Tq * Tk * _SCORES_BYTES


def sdpa_scores_bound(budget_bytes: int, head_dim_pow2: bool) -> int:
    """The fp32 scores bound an attention routes on: the device's budget; a non-power-of-two head
    dim whose architecture declares none routes on `sdpa.non_pow2_head_scores_bytes`."""
    if head_dim_pow2:
        return budget_bytes
    return budget_bytes or int(_value("sdpa", "non_pow2_head_scores_bytes"))


def sdpa_chunk_rows(bound: int, batch: int, nheads: int, Tq: int, Tk: int,
                    min_chunk_rows: int, max_chunks: int) -> int:
    """Query rows per chunk that keep each chunk's fp32 scores within `bound`, aligned to the
    arch's row block (`memory.sdpa_math_min_chunk_rows`); 0 when the shape cannot chunk within
    the arch's chunk ceiling (`memory.sdpa_math_max_chunks`)."""
    if not min_chunk_rows:
        return 0
    rows = (bound // (batch * nheads * Tk * _SCORES_BYTES)) // min_chunk_rows * min_chunk_rows
    if rows >= min_chunk_rows and -(-Tq // rows) <= max_chunks:
        return rows
    return 0


def sdpa_device_scores_budget(base_bytes: int, device_fraction: float, device_memory_mb) -> int:
    """The scores budget on one device: min(the arch's byte budget, `device_fraction` of THIS
    device's memory) — the 2026-08-31 xlong-prefill OOM proved a per-arch byte budget alone is
    wrong on heterogeneous rigs. No fraction, or no device memory known: the arch's budget."""
    if not base_bytes:
        return 0
    if not device_fraction or device_memory_mb is None:
        return base_bytes
    return min(base_bytes, int(device_memory_mb * 1024 * 1024 * device_fraction))


# --- Prism: the op budget and the op-level band factors -------------------------------------------

def op_budget_fraction() -> float:
    """The fraction of a card an op's transient is measured against (`op_level.op_budget_card_fraction`)."""
    return float(_value("op_level", "op_budget_card_fraction"))


def op_budget_bytes(card_bytes: int, resident_bytes: int) -> int:
    """An op's budget on its card: the op budget fraction of the card minus the component's
    resident weights."""
    return int(card_bytes * op_budget_fraction()) - resident_bytes


def conv_band_factor(total_bytes: int, kept_bytes: int, card_bytes: int) -> int:
    """Bands an overflowing convolution (or upsample->conv pair) runs in: its tiled bytes over
    one band's budget (`conv_band_card_fraction` of the card minus the output kept whole, at
    least `band_budget_floor_bytes`), at least `min_band_factor`, rounded up to a power of two
    at most `max_band_factor`."""
    budget = int(float(_value("op_level", "conv_band_card_fraction")) * card_bytes) - kept_bytes
    budget = max(budget, int(_value("op_level", "band_budget_floor_bytes")))
    factor = max(int(_value("op_level", "min_band_factor")), math.ceil(total_bytes / budget))
    return _pow2_at_most(factor, int(_value("op_level", "max_band_factor")))


def upsample_overflows(out_bytes: int, card_bytes: int) -> bool:
    """An upsample output worth fusing into its conv even when the conv does not overflow."""
    return out_bytes > float(_value("op_level", "upsample_overflow_card_fraction")) * card_bytes


def rms_norm_overflows(out_bytes: int, card_bytes: int) -> bool:
    """An rms_norm whose output runs in H bands."""
    return out_bytes > float(_value("op_level", "rms_norm", "overflow_card_fraction")) * card_bytes


def rms_norm_band_factor(out_bytes: int, card_bytes: int) -> int:
    """Bands an overflowing rms_norm runs in: its output over `rms_norm.band_card_fraction` of
    the card, at least `rms_norm.min_band_factor`, a power of two, at most `rms_norm.max_band_factor`."""
    budget = int(float(_value("op_level", "rms_norm", "band_card_fraction")) * card_bytes)
    factor = max(int(_value("op_level", "rms_norm", "min_band_factor")), math.ceil(out_bytes / budget))
    factor = _pow2_at_most(factor, int(_value("op_level", "max_band_factor")))
    return min(factor, int(_value("op_level", "rms_norm", "max_band_factor")))


def residual_chain_band_factor(bytes_fp32: int) -> int:
    """Bands a residual chain streams in: its half-width bytes (the fp32 estimate halved) over
    `residual_chain.band_target_bytes`, at least `residual_chain.min_band_factor`, a power of two at most
    `residual_chain.max_band_factor`."""
    half_bytes = bytes_fp32 // 2
    factor = max(int(_value("op_level", "residual_chain", "min_band_factor")),
                 math.ceil(half_bytes / int(_value("op_level", "residual_chain", "band_target_bytes"))))
    return _pow2_at_most(factor, int(_value("op_level", "residual_chain", "max_band_factor")))


def residual_chain_min_base_bytes_fp32() -> int:
    """The base-tensor size (fp32) from which a residual chain is detected."""
    return int(_value("op_level", "residual_chain", "min_base_bytes_fp32"))


# --- overlaps -------------------------------------------------------------------------------------

def spatial_halo(extent: int) -> int:
    """The proportional spatial overlap of a tile of `extent`: extent // `overlap.spatial_divisor`."""
    return extent // int(_value("overlap", "spatial_divisor"))


def spatial_overlap(extent: int) -> int:
    """A spatial tile's overlap: its proportional halo, at least `overlap.spatial_min`."""
    return max(int(_value("overlap", "spatial_min")), spatial_halo(extent))


def temporal_halo(t_tile: int) -> int:
    """The proportional temporal overlap of a tile of `t_tile` frames: t_tile // `overlap.temporal_divisor`."""
    return t_tile // int(_value("overlap", "temporal_divisor"))


# --- in place -------------------------------------------------------------------------------------

def inplace_min_bytes() -> int:
    """The output size from which an add or an element-wise unary writes into its dead input."""
    return int(_value("inplace", "min_bytes"))
