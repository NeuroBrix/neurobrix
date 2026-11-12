"""Slices of one axis of a tensor, read and written as views; partial sums kept in an accumulator dtype.

An axis of extent `P * total * inner` is split as the view `(P, total, inner)`: a slice `[start, start + length)` of
its `total` rows is cut from the middle factor, so a strided input is copied a slice at a time, never whole first. An
`inner` of None is a broadcast along the axis: its first extent is every slice. The stitch of a computation run in
slices (an application's chunked region) is these functions: read each slice (`take_slice`, `take_slice_cyclic` for a
read that runs off an end and continues from the other), write each slice's output into the whole (`whole_shape`,
`put_slice`), keep a copy the next run cannot overwrite (`own`), and sum a contraction's partials (`accumulate`,
`store`).

Written over the methods a tensor shares with torch's (`unflatten`, `narrow`, `contiguous`, `reshape`, `view`,
`new_empty`, `copy_`, `to`, `+`): an NBXTensor, or the tensor of an engine whose library is torch. No kernel is
autotuned here, and nothing is imported to tell the two apart.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

__all__ = ["take_slice", "take_slice_cyclic", "put_slice", "whole_shape", "own", "accumulate", "store"]


def take_slice(x: Any, axis: int, total: int, inner: Optional[int], start: int, length: int) -> Any:
    """The slice [start, start+length) of `axis` of a WHOLE tensor `x`, contiguous. A broadcast (`inner` None) is
    constant along the axis: its first extent at `length` is the slice."""
    shape = tuple(int(d) for d in x.shape)
    if inner is None:
        return x.narrow(axis, 0, shape[axis] // total * length).contiguous()
    p = shape[axis] // (total * inner)
    if p * total * inner != shape[axis]:
        raise RuntimeError(f"take_slice: extent {shape[axis]} on axis {axis} is not P*{total}*{inner}")
    pre, post = shape[:axis], shape[axis + 1:]
    # the axis split as a VIEW: only the slice is copied, never the whole input
    v = x.unflatten(axis, (p, total, inner)).narrow(axis + 1, start, length).contiguous()
    return v.reshape(*pre, p * length * inner, *post)


def take_slice_cyclic(x: Any, axis: int, total: int, inner: Optional[int], start: int, length: int) -> Any:
    """`take_slice` of the rows [start, start + length) taken modulo the axis' extent: a read that runs off either
    end continues from the other."""
    if 0 <= start and start + length <= total:
        return take_slice(x, axis, total, inner, start, length)
    if inner is None:                       # a broadcast is the same along the axis
        return take_slice(x, axis, total, inner, 0, length)
    pieces, at, left = [], start % total, length
    while left > 0:
        n = min(left, total - at)
        pieces.append((take_slice(x, axis, total, inner, at, n), n))
        left -= n
        at = 0
    first, n0 = pieces[0]
    whole = first.new_empty(whole_shape(first.shape, axis, length, n0))
    off = 0
    for part, n in pieces:
        put_slice(whole, part, axis, length, inner, off, n)
        off += n
    return whole


def put_slice(full: Any, part: Any, axis: int, total: int, inner: int, start: int, length: int) -> None:
    """Write a slice's output `part` into the whole output `full` (allocated contiguous)."""
    shape = tuple(int(d) for d in full.shape)
    p = shape[axis] // (total * inner)
    pre, post = shape[:axis], shape[axis + 1:]
    dst = full.view(*pre, p, total, inner, *post).narrow(axis + 1, start, length)
    dst.copy_(part.reshape(*pre, p, length, inner, *post))


def whole_shape(part_shape: Sequence[int], axis: int, total: int, length: int) -> Tuple[int, ...]:
    """The shape of the whole whose slice of `length` rows (out of `total`) has `part_shape`."""
    s = [int(d) for d in part_shape]
    s[axis] = s[axis] // length * total
    return tuple(s)


def own(x: Any, dtype: Any = None) -> Any:
    """A copy the next run of the same executor cannot overwrite (an arena slot is reused), in `dtype` when given
    (`copy_` converts on the store)."""
    shape = tuple(int(d) for d in x.shape)
    o = x.new_empty(shape) if dtype is None else x.new_empty(shape, dtype=dtype)
    o.copy_(x)
    return o


def accumulate(acc: Any, part: Any, dtype: Any) -> Any:
    """The running sum of a contraction's partials over the slices: `part` added to `acc` (None at the first slice)
    in `dtype`, the accumulator dtype, never the dtype the partial arrives in."""
    if acc is None:
        return own(part, dtype)
    return acc + (part if part.dtype == dtype else part.to(dtype))


def store(acc: Any, dtype: Any) -> Any:
    """A contraction's sum stored once in `dtype`, the dtype its own output was given."""
    return acc if acc.dtype == dtype else own(acc, dtype)
