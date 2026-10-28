"""Shapes, strides and views as pure functions on integers.

Every rule a tensor's layout follows lives here once, and `Tensor` calls it: how many elements a shape holds,
the row-major strides of a shape, whether a layout is contiguous, the strides of a view (ATen's `computeStride`),
broadcasting, slice bounds, and the largest offset a strided layout reaches. Integers in, integers out: the
functions read no device, no dtype and no global state (R-NBXT 11), so a Rust core can take them over unchanged.

Strides are in ELEMENTS, never bytes.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

__all__ = ["numel", "contiguous_strides", "is_contiguous", "view_strides", "broadcast_shapes",
           "broadcast_strides", "slice_bounds", "max_linear_offset", "MAX_NDIM"]

# PyTorch's bound on the rank an op iterates (aten/src/ATen/TensorIterator.h, `MAX_DIMS == 25`); a kernel launch
# beyond it is refused, never truncated.
MAX_NDIM = 25


def numel(shape: Sequence[int]) -> int:
    """The number of elements of `shape`: the product of its extents, 1 for the empty shape (a scalar)."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    """The row-major strides of `shape`, last dim fastest."""
    if len(shape) == 0:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """PyTorch's contiguity: the elements lie row-major without gaps; a dim of extent 1 may carry any stride (it is
    never walked) and an empty tensor is contiguous. Comparing the strides to the canonical ones instead called a
    decode-step view such as (1, 32, 64, 1) with strides (2048, 64, 1, 64) — a transposed single-token head block —
    non-contiguous and copied it before every element-wise kernel (NeuroBrix's copy census of 2026-09-07)."""
    expected = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent == 0:
            return True
        if extent == 1:
            continue
        if stride != expected:
            return False
        expected *= extent
    return True


def view_strides(old_shape: Sequence[int], old_strides: Sequence[int],
                 new_shape: Sequence[int]) -> Optional[list]:
    """The strides `new_shape` takes as a VIEW of a tensor laid out as (`old_shape`, `old_strides`), or None when no
    view exists — ATen's own rule (aten/src/ATen/TensorUtils.cpp `computeStride`): the old dims are walked in chunks
    of contiguous runs, and a new dim must fall inside one chunk."""
    total = numel(old_shape)
    if total == 0 or not old_shape:
        return None
    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base = old_strides[-1]
    tensor_numel = view_numel = 1
    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]
        if tensor_d == 0 or (old_shape[tensor_d - 1] != 1
                             and old_strides[tensor_d - 1] != tensor_numel * chunk_base):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_strides[view_d] = view_numel * chunk_base
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if tensor_d > 0:
                chunk_base = old_strides[tensor_d - 1]
                tensor_numel = view_numel = 1
    return new_strides if view_d == -1 else None


def broadcast_shapes(*shapes: Sequence[int]) -> Tuple[int, ...]:
    """The shape two or more shapes broadcast to (numpy's rule: right-aligned, an extent of 1 stretches); shapes
    that do not broadcast are refused by name."""
    if len(shapes) == 1:
        return tuple(shapes[0])
    result = tuple(shapes[0])
    for s in shapes[1:]:
        s = tuple(s)
        ndim = max(len(result), len(s))
        a = (1,) * (ndim - len(result)) + result
        b = (1,) * (ndim - len(s)) + s
        new = []
        for sa, sb in zip(a, b):
            if sa == sb:
                new.append(sa)
            elif sa == 1:
                new.append(sb)
            elif sb == 1:
                new.append(sa)
            else:
                raise ValueError(f"Cannot broadcast {result} and {s}")
        result = tuple(new)
    return result


def broadcast_strides(shape: Sequence[int], strides: Sequence[int],
                      target: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """(shape, strides) of the layout (`shape`, `strides`) EXPANDED to `target` without a copy: leading dims are
    prepended with extent 1, a dim of extent 1 stretches with stride 0, an entry of -1 keeps the dim as it is
    (ATen `expand`)."""
    pad = len(target) - len(shape)
    old_s = (1,) * pad + tuple(shape)
    old_st = (0,) * pad + tuple(strides)
    new_s, new_st = [], []
    for i, s in enumerate(target):
        if s == -1:
            new_s.append(old_s[i])
            new_st.append(old_st[i])
        elif old_s[i] == 1:
            new_s.append(s)
            new_st.append(0)
        else:
            new_s.append(s)
            new_st.append(old_st[i])
    return tuple(new_s), tuple(new_st)


def slice_bounds(start: Optional[int], stop: Optional[int], size: int, step: Optional[int] = None) -> Tuple[int, int]:
    """(start, stop) of a slice over an axis of `size`, as torch and numpy read it: negative bounds count from the
    end, and both bounds are CLAMPED to [0, size] — [50:80] over 60 rows is rows 50..59. Unclamped, a stop past the
    extent read past the tensor's storage (CogVideoX's last VAE tiles in NeuroBrix, 2026-10-17). A step other than 1
    is refused by name."""
    if step not in (None, 1):
        raise RuntimeError(f"ZERO FALLBACK: slicing with step {step} is not supported (narrow only)")
    start = 0 if start is None else int(start)
    stop = size if stop is None else int(stop)
    if start < 0:
        start += size
    if stop < 0:
        stop += size
    start = min(max(start, 0), size)
    stop = min(max(stop, start), size)
    return start, stop


def max_linear_offset(shape: Sequence[int], strides: Sequence[int]) -> int:
    """The largest element offset a strided layout reaches — what a kernel's index arithmetic must be wide enough
    for."""
    total = 0
    for extent, stride in zip(shape, strides):
        if extent > 0:
            total += (int(extent) - 1) * abs(int(stride))
    return total
