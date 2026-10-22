"""The GEMV and addmv candidates a backend can stage (workstream 1): on a backend whose lowering stages a 2-D reduce
one element per thread (Metal, measured 2026-09-07), a config's reduced extent — the GEMV's K chunk, the addmv's
reduction tile — never exceeds its program's threads (`nbx_tensor.reduced_extent_max`), and its tile never the
backend's reduce-tile cap; on cuda every candidate within the shapes stays. The rule moved from the Apple profiles'
hand tiles into the configs' prune.

Injection (seen red, 2026-10-22): the per-thread filter removed from `prune_gemv_vec` -> red on the Metal case."""
from __future__ import annotations

import pytest

from neurobrix.kernels import nbx_tensor as NT
from neurobrix.kernels.ops import _configs as CF


def _backend(monkeypatch, name):
    monkeypatch.setattr(NT, "_detect_gpu_backend", lambda: name)


@pytest.mark.parametrize("prune,configs,red,args", [
    (CF.prune_gemv_vec, CF.gemv_vec_configs(), "BLOCK_K", {"N": 32000, "K": 2048}),
    (CF.prune_addmv, CF.addmv_configs(), "BLOCK_M", {"N": 2560, "M": 2048}),
])
def test_on_a_per_thread_backend_the_reduced_extent_fits_the_threads(monkeypatch, prune, configs, red, args):
    _backend(monkeypatch, "metal")
    kept = prune(configs, args)
    assert kept
    for c in kept:
        assert c.kwargs[red] <= c.num_warps * 32, (c.kwargs, c.num_warps)
        assert c.kwargs["BLOCK_N"] * c.kwargs[red] <= NT.reduce_tile_max()
    # the tiles Apple adopted by hand are candidates it keeps (8 x 128: 4 warps for the GEMV, 32 for addmv)
    assert any(c.kwargs["BLOCK_N"] == 8 and c.kwargs[red] == 128 for c in kept)


def test_on_cuda_every_candidate_within_the_shapes_stays(monkeypatch):
    _backend(monkeypatch, "cuda")
    kept = CF.prune_gemv_vec(CF.gemv_vec_configs(), {"N": 32000, "K": 2048})
    assert len(kept) == len([c for c in CF.gemv_vec_configs() if c.kwargs["BLOCK_K"] <= 2048])
    assert any(c.kwargs == {"BLOCK_N": 8, "BLOCK_K": 256} and c.num_warps == 4 and c.num_stages == 1 for c in kept)
