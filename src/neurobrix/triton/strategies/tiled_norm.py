"""The two passes of an exact tiled group norm (the supervisor's decision 2026-10-27 11:15): the statistics of a group
norm over a sliced axis are taken ONCE over its whole input (`norm_statistics`, fp32, a fixed summation order: frame by
frame, group by group), then each slice is normalised with them (`group_norm_apply`). The result is the whole op's
up to the summation order of the statistics. No model, no family is named; the chunked piece calls it for every
`NORM` its stretch starts at (`triton/prism/chunked_region.py`). The Triton branch's twin of
`pytorch/strategies/tiled_norm.py`, Torch-free (RED_LINES.md lines 16-19): the library's NBXTensor passes
(`nbxtensor.ops.norm_wrappers.group_norm_statistics` / `group_norm_apply`, the first and second pass of its own group
norm kernel, the statistics taken in the same fp32 sum and sum of squares); a value that is no NBXTensor is refused
by name."""
from __future__ import annotations

from typing import Any, Tuple


def _is_nbx(t: Any) -> bool:
    return hasattr(t, "to_cuda") and hasattr(t, "_device")


def norm_statistics(x: Any, group: int, eps: float) -> Tuple[Any, Any]:
    """(mean, rstd), each [N, G] fp32, of `x` [N, C, ...] over each group's channels and every other position — what
    `aten::native_group_norm` computes — taken by the library's first pass."""
    if not _is_nbx(x):
        raise RuntimeError(f"ZERO FALLBACK: the Triton tiled group norm takes an NBXTensor, got {type(x).__name__}")
    from nbxtensor.ops.norm_wrappers import group_norm_statistics
    return group_norm_statistics(x, int(group), float(eps))


def group_norm_apply(x: Any, weight: Any, bias: Any, group: int, mean: Any, rstd: Any) -> Tuple[Any, Any, Any]:
    """`aten::native_group_norm`'s three outputs for a slice `x` [N, C, ...] given the whole input's statistics: the
    normalised slice in the slice's dtype, as the Triton engine's group norm writes it (the DtypeEngine's wrap around
    the interceptor gives it the op's operands, `ChunkedPiece._norm_interceptor`), mean and rstd [N, G]."""
    if not _is_nbx(x):
        raise RuntimeError(f"ZERO FALLBACK: the Triton tiled group norm takes an NBXTensor, got {type(x).__name__}")
    from nbxtensor.ops.norm_wrappers import group_norm_apply as _apply
    return _apply(x, weight, bias, int(group), mean, rstd)
