"""A piece takes over the values it is the last piece to read and frees each at its last use inside it, as the whole
run does and as Prism's live curve prices it. mochi-1-preview native at the 8 GB rung (2026-10-28): each transformer
piece peaked above its price by what it received (+719 / +219 / +408 MB for 782 / 283 / 472 MB received) — the
strategy's `values` held every received value to the piece's end.

What this test would do if the code were wrong: the strategy keeping its reference while the piece runs — the value
alive after the piece let go of it (red); a value a LATER piece reads handed over — the next piece finds nothing (red)."""
from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class _Tensor:
    def __init__(self, name):
        self.name = name


class _Strategy(LayerStreamingStrategy):
    def _nbx_path(self, component_name):
        return "<no container>"


class _Piece:
    """Runs as GraphExecutor does with handed inputs: takes them out of the caller's feed, then lets go of each after
    its last use (here: before its own output is made) and records whether anything else still holds it."""
    device = "cuda:0"

    def __init__(self, index, reads, makes, consumed, alive):
        self._dag = {"segment_index": index, "segment_input_names": list(reads),
                     "ops": {f"op{index}": {"output_tensor_ids": list(makes)}}, "output_tensor_ids": list(makes)}
        self._consumed_input_tids = frozenset(f"input::{n}" for n in consumed)
        self.reads, self.makes, self.alive = reads, makes, alive

    def protect_tensor_id(self, tid):
        pass

    def load_weights(self, *a, **k):
        pass

    def unload_weights(self):
        pass

    def tensors_of_last_run(self, tids):
        return {}

    def run(self, feed, *a, **k):
        own = dict(feed)
        for tid in self._consumed_input_tids:
            feed.pop(tid[len("input::"):], None)
        refs = {n: weakref.ref(own[n]) for n in (t[len("input::"):] for t in self._consumed_input_tids)}
        assert all(own[n] is not None for n in self.reads), "a value this piece reads was not handed"
        own.clear()                                    # the last use inside the piece
        gc.collect()
        self.alive.update({n: r() is not None for n, r in refs.items()})
        return {m: _Tensor(m) for m in self.makes}


def test_a_received_value_is_freed_at_its_last_use_inside_the_piece(monkeypatch):
    import neurobrix.core.strategies.layer_streaming as LS
    monkeypatch.setattr(LS, "_segment_memory_line", lambda seg: "")
    monkeypatch.setattr(LS, "_reset_segment_peak", lambda seg: None)
    alive = {}
    # piece 0 makes `h` (read by piece 1 alone) and `mask` (read by pieces 1 and 2): piece 1 takes over `h` only
    pieces = [_Piece(0, ["x"], ["h", "mask"], [], alive), _Piece(1, ["h", "mask"], ["g"], ["h"], alive),
              _Piece(2, ["mask", "g"], ["z"], ["mask", "g"], alive)]
    base = SimpleNamespace(_dag={"tensors": {}, "output_tensor_ids": ["z"]}, _persistent_tensor_ids=set())
    ctx = SimpleNamespace(component_executors={}, layer_segments={}, layer_graphs={}, layer_moe={},
                          layer_chunks={}, layer_swapped=frozenset())
    out = _Strategy(ctx, "layer_streaming")._run_pieces("comp", base, pieces, {"x": _Tensor("x")})
    assert alive == {"h": False, "mask": False, "g": False}, f"a handed value outlived its last use: {alive}"
    assert set(out) == {"z"}
