"""The two passes of an exact tiled group norm (the supervisor's decision 2026-10-27 11:15): the statistics of a group
norm over a sliced axis are taken ONCE over its whole input (`norm_statistics`, fp32, a fixed summation order: frame by
frame, group by group), then each slice is normalised with them (`group_norm_apply`). The result is the whole op's
up to the summation order of the statistics. No model, no family is named; the chunked piece calls it for every
`NORM` its stretch starts at (`core/prism/chunked_region.py`). Compiled engine (torch); a Triton plan is refused by
name until its NBXTensor twin is written (`ChunkedPiece`)."""
from __future__ import annotations

from typing import Any, Tuple


def norm_statistics(x: Any, group: int, eps: float) -> Tuple[Any, Any]:
    """(mean, rstd), each [N, G] fp32, of `x` [N, C, ...] over each group's channels and every other position — what
    `aten::native_group_norm` computes — taken one sample at a time so no fp32 copy of the whole input is held."""
    import torch
    if not isinstance(x, torch.Tensor):
        raise RuntimeError("tiled group norm: the Triton engine's NBXTensor statistics are not written yet; the plan "
                           "may not slice a group norm over its axis on that engine")
    n, c = int(x.shape[0]), int(x.shape[1])
    if c % group:
        raise ValueError(f"tiled group norm: {c} channels in {group} groups")
    mean = torch.empty(n, group, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n, group, dtype=torch.float32, device=x.device)
    for i in range(n):
        xi = x[i].reshape(group, -1).to(torch.float32)
        m = xi.mean(dim=1)
        v = (xi - m[:, None]).pow_(2).mean(dim=1)
        mean[i], rstd[i] = m, torch.rsqrt(v + eps)
        del xi
    return mean, rstd


def group_norm_apply(x: Any, weight: Any, bias: Any, group: int, mean: Any, rstd: Any) -> Tuple[Any, Any, Any]:
    """`aten::native_group_norm`'s three outputs for a slice `x` [N, C, ...] given the whole input's statistics: the
    normalised slice in fp32 (the engine's fp32-class norm), mean and rstd [N, G]."""
    import torch
    n, c = int(x.shape[0]), int(x.shape[1])
    m, r = mean.to(x.device), rstd.to(x.device)
    y = x.to(torch.float32).reshape(n, group, -1)
    y = (y - m[:, :, None]) * r[:, :, None]
    y = y.reshape(x.shape)
    shape = [1, c] + [1] * (x.dim() - 2)
    if weight is not None:
        y = y * weight.to(torch.float32).reshape(shape)
    if bias is not None:
        y = y + bias.to(torch.float32).reshape(shape)
    return y, m, r
