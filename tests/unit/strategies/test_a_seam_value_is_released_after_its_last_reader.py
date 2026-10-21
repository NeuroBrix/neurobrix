"""A streamed component's pieces hand their outputs to the later pieces by name
(`LayerStreamingStrategy._run_pieces`); a value is released once the last piece that reads it has run,
as the whole run frees a tensor after its last use. Held to the end, every seam the component ever
crossed stayed on the card beside the pieces Prism priced without it (its live curve frees each one
after its last reader): HAT-S at 2048 px plans 42 stretches whose whole inputs and outputs, 2.25 GB
each, would all have stayed live.

Injection (seen red, then restored green, 2026-10-21): the release loop removed -> the first piece's
output is still alive while the third piece runs."""
from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import torch

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class _Strategy(LayerStreamingStrategy):
    def _nbx_path(self, component_name):
        return "<no container>"


class _Piece:
    """A piece reading `reads`, producing `makes` from them; `check` runs inside its run."""

    def __init__(self, index, reads, makes, check=None):
        self._dag = {"segment_index": index, "segment_input_names": list(reads),
                     "ops": {f"op{index}": {"output_tensor_ids": [makes]}},
                     "output_tensor_ids": [makes]}
        self.makes, self.check = makes, check

    def protect_tensor_id(self, tid):
        pass

    def load_weights(self, *a, **k):
        pass

    def unload_weights(self):
        pass

    def tensors_of_last_run(self, tids):
        return {}

    def peak_memory_bytes(self):
        return 0

    def run(self, feed, *a, **k):
        if self.check:
            self.check()
        return {self.makes: sum(feed.values()) + 1}


def test_a_value_no_later_piece_reads_is_released():
    refs = {}

    class _First(_Piece):
        def run(self, feed, *a, **k):
            out = torch.ones(4) + feed["x"]
            refs["a"] = weakref.ref(out)
            return {"a": out}

    def a_is_gone():
        gc.collect()
        assert refs["a"]() is None, "the first piece's output outlived its last reader"

    pieces = [_First(0, ["x"], "a"), _Piece(1, ["a"], "b"), _Piece(2, ["b"], "c", check=a_is_gone)]
    base = SimpleNamespace(_dag={"tensors": {}, "output_tensor_ids": ["c"]}, _persistent_tensor_ids=set())
    ctx = SimpleNamespace(component_executors={}, layer_segments={}, layer_graphs={}, layer_moe={},
                          layer_chunks={})
    out = _Strategy(ctx, "layer_streaming")._run_pieces("c", base, pieces, {"x": torch.zeros(4)})
    assert torch.equal(out["c"], torch.full((4,), 3.0))
