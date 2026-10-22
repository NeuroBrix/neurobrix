"""The compiled engine's `aten::roll` (`CompiledOpResolver._make_roll`) equals `torch.roll` and makes ONE output: over
several dims `torch.roll` rolls one dim after the other and holds an intermediate — HAT-S at 2048 px on 16 GB, three
2 304 MB maps at the shifted-window roll where its price holds two (2026-10-22). CPU tensors judge the values; a
CUDA device judges the peak.

Injection (seen red, then restored green, 2026-10-22): `_make_roll` returning `torch.roll` -> the peak cell red."""
from __future__ import annotations

import pytest
import torch

from neurobrix.core.runtime.graph.compiled_ops import CompiledOpResolver


def _roll():
    return CompiledOpResolver._make_roll(None)


@pytest.mark.parametrize("shifts,dims", [((-8, -8), (1, 2)), ((3,), (2,)), ((5, -2), (2, 1)), ((0, 4), (1, 2)), (2, ())])
def test_it_equals_torch_roll(shifts, dims):
    x = torch.randn(2, 12, 20, 6).permute(0, 2, 1, 3)
    assert torch.equal(_roll()(x, list(shifts) if isinstance(shifts, tuple) else shifts, list(dims)),
                       torch.roll(x, shifts, dims) if dims else torch.roll(x, shifts))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_two_dim_roll_holds_one_output():
    x = torch.ones(1, 256, 256, 32, device="cuda")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = _roll()(x, [-8, -8], [1, 2])
    peak = torch.cuda.max_memory_allocated() - base
    assert peak <= out.numel() * 4 + (1 << 20), (peak, out.numel() * 4)
