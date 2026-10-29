"""A component's VENDOR tiling: the cut its vendor's own reference makes, declared with its source — the one exception
to "Prism tiles only where a tile is the untiled component" (decision of 2026-10-16 18:40 / 21:30).

Two vendor tilings exist in the catalogue, each reproduced here as its vendor computes it, from the numbers the
declaration carries (`components/<c>/vendor_tiling.json`, written by Forge from the registry with the vendor file and
line) — never from a model name:

* `ramp_crop` (diffusers' AutoencoderKL family: CogVideoX's `tiled_decode`, DC-AE's): latent tiles of `tile` at
  `stride`, each decoded whole; each tile's top rows blended IN PLACE with the bottom rows of the tile above
  (`blend_v`), then its left columns with the right columns of the tile on its left as that one stands
  (`blend_h`), over `blend` output pixels with the weights (1 - y / blend, y / blend); every tile cropped to `crop`
  output pixels and the crops concatenated. The output is exactly the vendor's: the crop of the last row and column
  keeps what remains.
* `mask_sum` (Allegro's `tiled_decode` / `tiled_encode`): tiles of `tile_in` at `stride_in` over the input, only as
  many as fit whole (floor((N - tile) / stride) + 1 per axis); each tile's output weighted on its overlaps by linear
  ramps — rising over its head unless it is the first on that axis, falling over its tail unless it is the last,
  time then height then width — and summed into an output of zeros at `stride_out`. Where no tile reaches, the output
  is zero, as the vendor's.

Mode-polymorphic (torch or NBXTensor, the run's own kind); R33: no torch at import. The weights are computed as the
vendor computes them: a ramp entry is float32(i) / float32(n) (`torch.arange(n).float() / n`), a blend weight the
Python float (1 - y / blend).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np


def _is_nbx(t: Any) -> bool:
    return hasattr(t, "to_cuda") and hasattr(t, "_device")


def _zeros(ref: Any, shape: Sequence[int]) -> Any:
    if _is_nbx(ref):
        from nbxtensor.tensor import NBXTensor
        dev = f"cuda:{getattr(ref, '_device_idx', 0)}" if ref._device == "cuda" else ref._device
        return NBXTensor.zeros(list(shape), dtype=ref.dtype, device=dev)
    import torch
    return torch.zeros(*shape, dtype=ref.dtype, device=ref.device)


def _cat(parts: List[Any], axis: int) -> Any:
    shape = list(parts[0].shape)
    shape[axis] = sum(int(p.shape[axis]) for p in parts)
    out = _zeros(parts[0], shape)
    pos = 0
    for p in parts:
        sl = [slice(None)] * len(shape)
        sl[axis] = slice(pos, pos + int(p.shape[axis]))
        out[tuple(sl)] = p
        pos += int(p.shape[axis])
    return out


def _at(rank: int, axis: int, index) -> tuple:
    sl = [slice(None)] * rank
    sl[axis] = index
    return tuple(sl)


def ramp_crop(z: Any, decl: Dict[str, Any], run_tile: Callable[[Any], Any]) -> Any:
    """The vendor's `tiled_decode` of the AutoencoderKL family. `decl`: tile, stride (latent rows / columns), blend,
    crop (output pixels), in_axes (the latent's height and width axes), out_axes (the output's)."""
    th, tw = decl["tile"]
    sh, sw = decl["stride"]
    bh, bw = decl["blend"]
    ch, cw = decl["crop"]
    iy, ix = decl["in_axes"]
    oy, ox = decl["out_axes"]
    H, W = int(z.shape[iy]), int(z.shape[ix])
    rank_in = len(z.shape)
    rows: List[List[Any]] = []
    for i in range(0, H, sh):
        row = []
        for j in range(0, W, sw):
            sl = [slice(None)] * rank_in
            sl[iy], sl[ix] = slice(i, i + th), slice(j, j + tw)
            # copied out of the run that made it: an engine's arena may reuse that memory at the next tile
            row.append(run_tile(z[tuple(sl)].contiguous()).clone())
        rows.append(row)
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            r = len(tile.shape)
            if i > 0:
                a = rows[i - 1][j]
                e = min(int(a.shape[oy]), int(tile.shape[oy]), bh)
                for y in range(e):
                    tile[_at(r, oy, y)] = a[_at(r, oy, y - e)] * (1 - y / e) + tile[_at(r, oy, y)] * (y / e)
            if j > 0:
                a = row[j - 1]
                e = min(int(a.shape[ox]), int(tile.shape[ox]), bw)
                for x in range(e):
                    tile[_at(r, ox, x)] = a[_at(r, ox, x - e)] * (1 - x / e) + tile[_at(r, ox, x)] * (x / e)
            sl = [slice(None)] * r
            sl[oy], sl[ox] = slice(0, ch), slice(0, cw)
            result_row.append(tile[tuple(sl)].contiguous())
        result_rows.append(_cat(result_row, ox))
    return _cat(result_rows, oy)


def _ramp(n: int, rising: bool) -> List[float]:
    w = np.arange(0, n, dtype=np.float32) / np.float32(n)
    return [float(v) for v in (w if rising else np.float32(1) - w)]


def mask_sum(x: Any, decl: Dict[str, Any], run_tile: Callable[[Any], Any]) -> Any:
    """Allegro's `tiled_decode` / `tiled_encode`. `decl`: tile_in, stride_in (the input's frames, rows, columns),
    tile_out, stride_out (the output's), in_axes, out_axes (time, height, width), out_scale (the output extent per
    input unit on each axis, as a fraction [num, den])."""
    ki, si, ko, so = decl["tile_in"], decl["stride_in"], decl["tile_out"], decl["stride_out"]
    in_axes, out_axes = decl["in_axes"], decl["out_axes"]
    counts = [(int(x.shape[a]) - k) // s + 1 for a, k, s in zip(in_axes, ki, si)]
    if min(counts) < 1:
        raise ValueError(f"vendor tiling: an input of {[int(x.shape[a]) for a in in_axes]} on its tiled axes is "
                         f"smaller than one tile of {list(ki)}")
    overlap = [k - s for k, s in zip(ko, so)]
    out = None
    rank_in = len(x.shape)
    for n in range(counts[0]):
        for h in range(counts[1]):
            for w in range(counts[2]):
                idx = (n, h, w)
                sl = [slice(None)] * rank_in
                for a, k, s, p in zip(in_axes, ki, si, idx):
                    sl[a] = slice(p * s, p * s + k)
                y = run_tile(x[tuple(sl)].contiguous()).clone()
                r = len(y.shape)
                if out is None:
                    shape = list(y.shape)
                    for a, (num, den) in zip(out_axes, decl["out_scale"]):
                        shape[a] = int(x.shape[in_axes[out_axes.index(a)]]) * num // den
                    out = _zeros(y, shape)
                for ax, p, cnt, ov in zip(out_axes, idx, counts, overlap):
                    if ov <= 0:
                        continue
                    if p > 0:
                        for t, wgt in enumerate(_ramp(ov, True)):
                            y[_at(r, ax, t)] = y[_at(r, ax, t)] * wgt
                    if p < cnt - 1:
                        for t, wgt in enumerate(_ramp(ov, False)):
                            y[_at(r, ax, t - ov)] = y[_at(r, ax, t - ov)] * wgt
                osl = [slice(None)] * r
                for a, k, s, p in zip(out_axes, ko, so, idx):
                    osl[a] = slice(p * s, p * s + k)
                out[tuple(osl)] = out[tuple(osl)] + y
    return out


METHODS = {"ramp_crop": ramp_crop, "mask_sum": mask_sum}


def run_vendor_tiling(x: Any, decl: Dict[str, Any], run_tile: Callable[[Any], Any]) -> Any:
    method = decl.get("method")
    if method not in METHODS:
        raise ValueError(f"vendor tiling: method {method!r} is not one this engine reproduces ({sorted(METHODS)})")
    return METHODS[method](x, decl, run_tile)
