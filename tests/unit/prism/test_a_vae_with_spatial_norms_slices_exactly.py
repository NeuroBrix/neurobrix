"""A component whose group norms read the whole image is sliced in space EXACTLY (the supervisor's decision 2026-10-27
11:15, Hocine 11:01: mochi must pass on 16 GB and 8 GB). Each group norm over the sliced axis is a synchronisation point:
its statistics over the WHOLE input (fp32, a fixed order), then each slice normalised with them; a padding eaten by its
valid convolution is one stencil with its halo. Before: mochi-1-preview's VAE mixed the axis at every norm and every
padding (272 of 407 ops), refused at an 8 GB rung ("activations alone peak at 15 345 MB")."""
import json
import os
from pathlib import Path

import pytest
import torch

from neurobrix.core.prism.chunked_region import MIX, NORM, TokenAxis
from neurobrix.core.strategies.tiled_norm import group_norm_apply, norm_statistics
from neurobrix.core.strategies.value_parking import to_host

CACHE = Path(os.environ.get("NEUROBRIX_CACHE", os.path.expanduser("~/.neurobrix/cache")))
MOCHI_VAE = CACHE / "mochi-1-preview" / "components" / "vae" / "graph.json"


@pytest.mark.skipif(not MOCHI_VAE.exists(), reason="mochi-1-preview's container is not in this cache")
@pytest.mark.parametrize("sym", ["s2", "s3"])
def test_mochi_s_vae_mixes_neither_spatial_axis(sym):
    ax = TokenAxis(json.loads(MOCHI_VAE.read_text()), sym, 1)
    kinds = [v.kind for v in ax.verdicts]
    assert MIX not in kinds, [v.reason for v in ax.verdicts if v.kind == MIX][:3]
    assert kinds.count(NORM) == 38


@pytest.mark.parametrize("shape,group", [((3, 64, 24, 20), 32), ((2, 32, 5, 12, 16), 8)])
def test_a_sliced_group_norm_with_whole_statistics_is_the_whole_norm(shape, group):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32) * 3 + 1
    w, b = torch.randn(shape[1]), torch.randn(shape[1])
    n, c = shape[0], shape[1]
    hw = int(torch.tensor(shape[2:]).prod())
    ref = torch.ops.aten.native_group_norm(x, w, b, n, c, hw, group, 1e-6)[0]
    mean, rstd = norm_statistics(x, group, 1e-6)
    H = shape[-2]
    parts = [group_norm_apply(x[..., a:a + 7, :], w, b, group, mean, rstd)[0] for a in range(0, H, 7)]
    got = torch.cat(parts, dim=-2)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), (got - ref).abs().max()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Metal device")
def test_a_value_parked_on_the_host_leaves_metal():
    v = torch.ones(4, device="mps")
    assert to_host(v).device.type == "cpu"
