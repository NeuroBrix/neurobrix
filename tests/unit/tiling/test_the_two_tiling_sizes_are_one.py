"""The TilingEngine's sizing half exists twice, once per engine branch, and the two are one.

`neurobrix/core/module/tiling_sizes.py` sizes the PyTorch branch's splits (Prism, the core
executors, the TilingEngine); `neurobrix/triton/tiling_sizes.py` sizes the Triton branch's (the
wrappers, the launch keys, the derived census). The engines share no compute code, so each keeps
its own copy — and a plan priced by one copy and cut by the other is only sound while the two
answer the same on every input. This file holds them to it: the same public functions, and the
same answer from each over a grid of inputs that crosses every threshold, floor and cap.

What this test does if the code were wrong: a one-value change in either copy (a divisor, a floor,
the scores width) answers differently on some grid point -> red, naming the function and the input
(seen red with `_SCORES_BYTES` 4 -> 2 in the Triton copy and `max(1, ...)` -> `max(2, ...)` in
`chunk_frames` of the core copy, green again once restored).
"""
from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from neurobrix.core.module import tiling_sizes as CORE
from neurobrix.triton import tiling_sizes as TRITON

MB = 1024 ** 2
GiB = 1024 ** 3


def _public(mod):
    return {n for n, f in inspect.getmembers(mod, inspect.isfunction)
            if f.__module__ == mod.__name__ and not n.startswith("_")}


_BYTES = [0, 1, 1000, 255 * MB, 256 * MB, GiB - 1, GiB, GiB + 1, 3 * GiB, 4 * GiB, 9 * GiB,
          40 * GiB, 300 * GiB]
_CARDS = [3072 * MB, 4096 * MB, 16384 * MB, 32768 * MB, 80 * 1024 * MB]
_EXTENTS = [0, 1, 4, 7, 8, 16, 31, 64, 200, 351, 1024]

_CONV3D_X = [[1, 256, 13, 240, 360], [2, 64, 9, 1024, 1024], [1, 4, 1, 8, 8], [1, 128, 81, 480, 832],
             [1, 16, 3, 2048, 2048]]
_CONV3D_W = [[256, 256, 3, 3, 3], [64, 64, 3, 3, 3], [128, 128, 1, 3, 3], [16, 16, 3, 1, 1]]

GRID = {
    "conv3d_chunk_bytes": [()],
    "conv2d_band_bytes": [()],
    "conv3d_need": [(x, w, s, p, d, ib, ob)
                    for x in _CONV3D_X for w in _CONV3D_W if w[1] == x[1]
                    for s in (1, [1, 2, 2]) for p in (0, 1, [0, 1, 1]) for d in (1,)
                    for ib, ob in ((2, 2), (4, 4), (2, 4))],
    "chunk_frames": [(b,) for b in _BYTES],
    "conv2d_band_rows": [(n, c, h, w, ob, band)
                         for n in (1, 2) for c in (3, 64, 512) for h in (1, 64, 1024, 8192)
                         for w in (64, 8192) for ob in (2, 4) for band in (GiB, 4 * GiB)],
    "sdpa_scores_bytes": [(b, h, tq, tk) for b in (1, 2) for h in (8, 32) for tq in (1, 4096)
                          for tk in (1, 4096, 65536)],
    "sdpa_scores_bound": [(b, p) for b in (0, GiB, 2 * GiB) for p in (True, False)],
    "sdpa_chunk_rows": [(bound, b, h, tq, tk, mr, mc)
                        for bound in (0, GiB, 2 * GiB) for b in (1, 2) for h in (8, 32)
                        for tq in (1024, 16384) for tk in (4096, 65536)
                        for mr in (0, 128) for mc in (0, 16, 64)],
    "sdpa_device_scores_budget": [(b, f, m) for b in (0, GiB, 2 * GiB) for f in (0.0, 0.07, 0.5)
                                  for m in (None, 16384, 32768)],
    "conv3d_transient_bytes": [(x, w, s, p, 1, ib, ob, ch)
                               for x in _CONV3D_X for w in _CONV3D_W if w[1] == x[1]
                               for s in (1, [1, 2, 2]) for p in (0, [0, 1, 1])
                               for ib, ob in ((2, 2), (2, 4)) for ch in (False, True)],
    "conv2d_band_transient_bytes": [(n, ci, iw, ib, co, oh, ow, ob, kh, sh, dh, band)
                                    for n in (1, 2) for ci, co in ((3, 64), (512, 512))
                                    for iw, oh, ow in ((64, 64, 64), (2048, 1024, 1024), (8192, 8192, 8192))
                                    for ib, ob in ((2, 2), (4, 4)) for kh, sh, dh in ((1, 1, 1), (3, 1, 1), (3, 2, 1))
                                    for band in (GiB, 4 * GiB)],
    "tiled_conv2d_bands": [(ih, oh, kh, sh, 1, ph, tf)
                           for ih, oh, kh, sh, ph in ((64, 64, 3, 1, 1), (1024, 512, 3, 2, 1), (8, 8, 1, 1, 0),
                                                      (4096, 4096, 7, 1, 3))
                           for tf in (1, 2, 4, 16, 64)],
    "tiled_conv2d_transient_bytes": [(n, ci, ih, iw, ib, co, oh, ow, ob, kh, sh, 1, ph, pw, hb, tf)
                                     for n in (1, 2) for ci, co in ((3, 64), (512, 256))
                                     for ih, iw, oh, ow, kh, sh, ph, pw in ((1024, 1024, 1024, 1024, 3, 1, 1, 1),
                                                                            (2048, 2048, 1024, 1024, 3, 2, 1, 1),
                                                                            (512, 512, 512, 512, 1, 1, 0, 0))
                                     for ib, ob in ((2, 2), (4, 4)) for hb in (False, True) for tf in (1, 4, 16)],
    "sdpa_route": [(b, h, tq, tk, d, dv, bud, mr, mc, fm, uf)
                   for b in (1, 2) for h in (8, 24) for tq, tk in ((1, 4096), (4096, 4096), (16156, 16156))
                   for d, dv in ((64, 64), (128, 128), (80, 80), (64, 128))
                   for bud in (0, 2 * GiB) for mr, mc in ((0, 0), (128, 16))
                   for fm in (False, True) for uf in (False, True)],
    "sdpa_transient_bytes": [(r, rows, b, h, tq, tk) for r in ("math", "chunked", "flash")
                             for rows in (0, 128, 1024) for b in (1, 2) for h in (8, 24)
                             for tq in (1, 4096, 16156) for tk in (4096, 16156)],
    "op_budget_fraction": [()],
    "op_budget_bytes": [(c, r) for c in _CARDS for r in (0, 300 * MB, 9 * GiB)],
    "conv_band_factor": [(t, k, c) for t in _BYTES for k in (0, GiB, 9 * GiB) for c in _CARDS],
    "upsample_overflows": [(b, c) for b in _BYTES for c in _CARDS],
    "rms_norm_overflows": [(b, c) for b in _BYTES for c in _CARDS],
    "rms_norm_band_factor": [(b, c) for b in _BYTES for c in _CARDS],
    "residual_chain_band_factor": [(b,) for b in _BYTES],
    "residual_chain_min_base_bytes_fp32": [()],
    "spatial_halo": [(e,) for e in _EXTENTS],
    "spatial_overlap": [(e,) for e in _EXTENTS],
    "temporal_halo": [(e,) for e in _EXTENTS],
    "inplace_min_bytes": [()],
}


def test_the_two_copies_carry_the_same_functions():
    assert _public(CORE) == _public(TRITON), _public(CORE) ^ _public(TRITON)


def test_every_function_has_a_grid():
    assert _public(CORE) == set(GRID), _public(CORE) ^ set(GRID)


@pytest.mark.parametrize("name", sorted(GRID))
def test_the_two_copies_answer_the_same(name):
    core_f, triton_f = getattr(CORE, name), getattr(TRITON, name)
    for args in GRID[name]:
        assert core_f(*args) == triton_f(*args), f"{name}{args}: core {core_f(*args)} != triton {triton_f(*args)}"


def test_the_triton_copy_imports_no_torch_no_numpy_no_third_party():
    src = str(Path(__file__).resolve().parents[3] / "src")
    probe = ("import sys; import neurobrix.triton.tiling_sizes as T; T.conv3d_chunk_bytes(); "
             "print('torch' in sys.modules, 'numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                         env={"PYTHONPATH": src, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout
    import ast
    tree = ast.parse(Path(TRITON.__file__).read_text())
    imported = {(n.module or "") if isinstance(n, ast.ImportFrom) else a.name
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    assert imported <= {"__future__", "math", "typing", "neurobrix.core.config.loader"}, imported
