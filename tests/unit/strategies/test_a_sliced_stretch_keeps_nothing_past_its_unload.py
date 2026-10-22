"""A sliced stretch (`ChunkedPiece`) keeps what its last run made only until its unload — the executor's contract:
`tensors_of_last_run` is read BEFORE `unload_weights`, which drops the arena and the capture both. Kept past it,
every stretch held its assembled output on the device for the rest of the run: HAT-S at 2048 px on 32 GB, four
stretches' 2 304 MB outputs beside the ninth piece, which OOMed (2026-10-22).

Injection (seen red, then restored green, 2026-10-22): the reset in `unload_weights` removed -> red."""
from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.strategies.chunked_piece import ChunkedPiece


class _Out:
    pass


def test_the_last_runs_output_is_released_at_unload():
    piece = ChunkedPiece.__new__(ChunkedPiece)
    piece._passes = []
    piece._holder = SimpleNamespace(unload_weights=lambda: None)
    out = _Out()
    alive = weakref.ref(out)
    piece._last_run = {"aten.add::3::out_0": out}
    assert piece.tensors_of_last_run(["aten.add::3::out_0"]) == {"aten.add::3::out_0": out}   # read before
    del out
    piece.unload_weights()
    gc.collect()
    assert alive() is None
    assert piece.tensors_of_last_run(["aten.add::3::out_0"]) == {}
