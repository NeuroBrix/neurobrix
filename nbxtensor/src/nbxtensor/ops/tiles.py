"""A large tensor computed tile by tile: each tile read as a padded window, each tile's result crossfaded into the
whole over the overlap it shares with its neighbours, the whole divided by the summed weights once at the end.

Where the tiles sit, their size and stride, and which axes are tiled are the caller's decisions; this module is the
mechanism. `tile_window` reads one tile of an NBXTensor; `TileBlend` places results of either engine's tensors (it
calls only the methods a tensor shares with torch's: `new_zeros`, `new_tensor`, indexing, `+`, `*`, `/`).

Feathered, not boxed: a tile weighs 1 in its interior and ramps down linearly across each overlap it shares with a
neighbour, (i + 0.5) / overlap, so two tiles' weights sum to exactly 1 at every overlapped element (a crossfade). A box
weight (every tile 1, then divide) switched the average abruptly wherever the number of covering tiles changed: a
line at every tile stride (CogVideoX-2b's 64-px grid and Wan2.1-T2V's 96-px grid, both engines, 2026-09-27). It is not
the diffusers formula (blend_h / blend_v ramp the previous tile's tail into the next tile's head and crop each tile
to the stride), so a tile-level comparison with a vendor's tiled decode keeps a residual by construction.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

__all__ = ["crossfade_ramp", "tile_window", "TileBlend"]


def crossfade_ramp(n: int, before: int, after: int) -> np.ndarray:
    """The float32 weights of `n` elements along one axis: 1 inside, a linear ramp (i + 0.5) / overlap across the
    `before` elements shared with the previous tile and the `after` elements shared with the next."""
    r = np.ones(n, dtype=np.float32)
    if before > 0:
        k = min(before, n)
        r[:k] = np.minimum(r[:k], (np.arange(k, dtype=np.float32) + 0.5) / before)
    if after > 0:
        k = min(after, n)
        r[n - k:] = np.minimum(r[n - k:], (np.arange(k, 0, -1, dtype=np.float32) - 0.5) / after)
    return r


def tile_window(x, starts: Sequence[int], sizes: Sequence[int]):
    """The tile of NBXTensor `x` starting at `starts` on its trailing `len(starts)` axes, `sizes` long, CONTIGUOUS.
    Cut short by the end of the tensor, its two trailing axes are padded up to `sizes` by replicating their edge; a
    leading tiled axis (time, before height and width) is never padded, only cut.

    Contiguous because a tile is a slice, and a slice strides across its parent's rows: an interior 32-wide tile of a
    64-wide image has row stride 64. A flat-indexed kernel handed it reads the wrong addresses — the Sana 4Kpx conv::55
    incident, 94 % of elements past 1.0 absolute. Edge tiles came back from the pad packed and right while interior
    tiles did not: a grid, tile interiors wrong and seams clean."""
    from nbxtensor.ops.shape_wrappers import pad_wrapper
    shape = tuple(int(d) for d in x.shape)
    lead = len(shape) - len(starts)
    stops = [min(s + n, shape[lead + i]) for i, (s, n) in enumerate(zip(starts, sizes))]
    tile = x[(slice(None),) * lead + tuple(slice(s, e) for s, e in zip(starts, stops))]
    pad_h = sizes[-2] - (stops[-2] - starts[-2])
    pad_w = sizes[-1] - (stops[-1] - starts[-1])
    if pad_h or pad_w:
        tile = pad_wrapper(tile, [0, pad_w, 0, pad_h], mode="replicate")
    return tile.contiguous()


class TileBlend:
    """The whole a set of overlapping tiles' results is crossfaded into.

    `like`: a tile's result (the accumulators take its dtype and device). `out_shape`: the whole. `weight_shape`: the
    summed weights, of the whole's rank (size 1 on an axis every tile covers whole, so the final divide broadcasts).
    `axes`: one `(axis, starts, length)` per tiled axis of the result — the output start of every tile along it, in
    order, and a tile's output length. `add(result, index)` places a tile given its position index along each of
    `axes`; `result()` is the whole."""

    def __init__(self, like: Any, out_shape: Sequence[int], weight_shape: Sequence[int],
                 axes: Sequence[Tuple[int, Sequence[int], int]]):
        self._ndim = len(out_shape)
        self._axes: List[Tuple[int, List[int], int, int]] = [
            (axis, [int(s) for s in starts], int(length), int(out_shape[axis])) for axis, starts, length in axes]
        self._like = like
        self.output = like.new_zeros(list(out_shape), device=like.device)
        self.weight = like.new_zeros(list(weight_shape), device=like.device)

    def add(self, result: Any, index: Sequence[int]) -> None:
        osl, rsl = [slice(None)] * self._ndim, [slice(None)] * self._ndim
        shape, ramps = [1] * self._ndim, []
        for (axis, starts, length, extent), i in zip(self._axes, index):
            o = starts[i]
            n = min(length, extent - o)
            osl[axis], rsl[axis], shape[axis] = slice(o, o + n), slice(0, n), n
            before = max(0, starts[i - 1] + length - o) if i > 0 else 0
            after = max(0, o + length - starts[i + 1]) if i + 1 < len(starts) else 0
            ramps.append((axis, crossfade_ramp(n, before, after)))
        window = np.ones(shape, dtype=np.float32)
        for axis, r in ramps:
            along = [1] * self._ndim
            along[axis] = shape[axis]
            window = window * r.reshape(along)
        w = self._like.new_tensor(np.ascontiguousarray(window))
        osl, rsl = tuple(osl), tuple(rsl)
        self.output[osl] = self.output[osl] + result[rsl] * w
        self.weight[osl] = self.weight[osl] + w

    def result(self) -> Any:
        """The whole: the weighted sum divided by the summed weights."""
        return self.output / self.weight
