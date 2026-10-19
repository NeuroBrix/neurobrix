"""A streamed component's pieces hand values to each other by name; a value lives only until its last reader (the later
pieces' inputs, the component's declared outputs). Kept to the end of the run, every seam tensor stayed alive: Allegro
native on the Mac (720p, 5 pieces) held 1.5, 5.0, 7.1, 7.8 GB after pieces 0-3 with each piece's weights released, and
the 5th met MPS out of memory (2026-10-20).

What this test does if the code is wrong: the liveness cell fails when piece 0's output, read only by piece 1, is still
referenced while piece 2 runs; the result cell fails when a value the later pieces or the declared outputs still read is
dropped (the run then refuses "needs ... nothing before it produced them" or "declares output ... no piece returned it").
"""
import gc
import weakref
from types import SimpleNamespace

import torch

from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class _Piece:
    def __init__(self, index, needs, makes, probe=None):
        self._dag = {"segment_index": index, "segment_input_names": needs,
                     "ops": {f"op{index}": {"output_tensor_ids": makes}}}
        self.makes, self.probe, self.device, self.mode = makes, probe, "cpu", "compiled"

    def protect_tensor_id(self, tid): pass
    def load_weights(self, *a, **k): pass
    def unload_weights(self): pass
    def tensors_of_last_run(self, tids): return {}

    def run(self, feed, *a, **k):
        if self.probe:
            self.probe()
        total = sum(v.sum() for v in feed.values() if torch.is_tensor(v))
        return {m: torch.full((4,), float(total) + 1.0) for m in self.makes}


class _Strategy(LayerStreamingStrategy):
    def __init__(self): self._produced = {}
    def _nbx_path(self, component_name): return "<none>"


def _run(pieces):
    base = SimpleNamespace(_dag={"tensors": {}, "output_tensor_ids": ["out"]}, _persistent_tensor_ids=())
    return _Strategy()._run_pieces("c", base, pieces, {"x": torch.ones(4)})


def test_a_seam_value_is_released_once_its_last_reader_ran():
    seen = {}
    def keep_ref():                           # piece 1 runs: piece 0's "a" is its input, alive here
        pass
    def check():                              # piece 2 runs: "a" had no reader after piece 1
        gc.collect()
        seen["alive"] = seen["ref"]() is not None
    p0 = _Piece(0, ["x"], ["a"])
    orig = p0.run
    def run0(feed, *a, **k):
        out = orig(feed)
        seen["ref"] = weakref.ref(out["a"])
        return out
    p0.run = run0
    pieces = [p0, _Piece(1, ["a"], ["b"], keep_ref), _Piece(2, ["b"], ["out"], check)]
    _run(pieces)
    assert seen["alive"] is False, "piece 0's output, read only by piece 1, is still held while piece 2 runs"


def test_values_still_read_later_or_declared_are_kept():
    pieces = [_Piece(0, ["x"], ["a", "skip"]), _Piece(1, ["a"], ["b"]), _Piece(2, ["b", "skip"], ["out"])]
    out = _run(pieces)
    assert set(out) == {"out"} and out["out"].shape == (4,)
