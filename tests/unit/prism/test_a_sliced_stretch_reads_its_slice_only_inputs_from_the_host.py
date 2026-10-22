"""With the values between pieces parked, a sliced stretch's inputs it reads ONLY in slices (`chunked_region.
sliced_reads`: a layout on the axis, no output of it, read by no pass that runs whole) stay on the host — Prism prices
their slices only, the strategy leaves them there, each pass uploads its slice. An input a whole pass reads comes
to the device whole. HAT-L at 2048 px on 16 GB: its last stretch read the long skips whole, 15.3 GB at a slice of
one window row against 14.1 GB (2026-10-22).

Injection (seen red, then restored green, 2026-10-22): `sliced_reads` returning every input -> red; and the strategy
bringing every read to the device -> red."""
from __future__ import annotations

from types import SimpleNamespace

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.prism.chunked_region import ChunkPass, RegionPlan, sliced_reads
from neurobrix.core.strategies import value_parking as VP
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


def _plan():
    ops = {"a": {"input_tensor_ids": ["x", "w_whole"], "output_tensor_ids": ["y"]},
           "b": {"input_tensor_ids": ["y", "skip"], "output_tensor_ids": ["z"]},
           "c": {"input_tensor_ids": ["w_whole"], "output_tensor_ids": ["s"]}}
    lay = object()
    plan = RegionPlan(symbol="s1", first=0, last=2, first_op="a", last_op="c",
                      passes=[ChunkPass(ops=["c"], targets=["s"], chunked=False),
                              ChunkPass(ops=["a", "b"], targets=["z"], chunked=True)],
                      inputs=["x", "skip", "w_whole"], outputs=["z", "s"],
                      layouts={"x": lay, "skip": lay, "w_whole": lay, "y": lay, "z": lay})
    return plan, ops


def test_only_the_inputs_read_in_slices_alone_stay_on_the_host():
    plan, ops = _plan()
    assert sliced_reads(plan, ops) == {"x", "skip"}          # w_whole: a whole pass reads it


class _T:
    def __init__(self, name, device="cuda:0"):
        self.name, self.device = name, device

    @property
    def is_cuda(self):
        return self.device.startswith("cuda")

    def to(self, device):
        return _T(self.name, str(device))


class _Strategy(LayerStreamingStrategy):
    def _nbx_path(self, component_name):
        return "<no container>"


class _Piece:
    device = "cuda:0"

    def __init__(self, index, reads, makes, sliced=()):
        self._dag = {"segment_index": index, "segment_input_names": list(reads),
                     "ops": {f"op{index}": {"output_tensor_ids": list(makes)}}, "output_tensor_ids": list(makes)}
        self.makes, self.fed, self._sliced = makes, {}, set(sliced)
        if sliced:
            self.sliced_input_names = lambda: set(self._sliced)

    def protect_tensor_id(self, tid):
        pass

    def load_weights(self, *a, **k):
        pass

    def unload_weights(self):
        pass

    def tensors_of_last_run(self, tids):
        return {}

    def run(self, feed, *a, **k):
        self.fed = {n: v.device for n, v in feed.items()}
        return {m: _T(m) for m in self.makes}


def test_the_strategy_leaves_a_slice_only_input_on_the_host(monkeypatch):
    import neurobrix.core.strategies.layer_streaming as LS
    monkeypatch.setattr(LS, "_segment_memory_line", lambda seg: "")
    monkeypatch.setattr(LS, "_reset_segment_peak", lambda seg: None)
    pieces = [_Piece(0, ["x"], ["skip", "h"]), _Piece(1, ["skip", "h"], ["z"], sliced=["skip"])]
    base = SimpleNamespace(_dag={"tensors": {}, "output_tensor_ids": ["z"]}, _persistent_tensor_ids=set())
    ctx = SimpleNamespace(component_executors={}, layer_segments={}, layer_graphs={}, layer_moe={},
                          layer_chunks={}, layer_swapped=frozenset({"comp"}))
    _Strategy(ctx, "layer_streaming")._run_pieces("comp", base, pieces, {"x": _T("x")})
    assert pieces[1].fed == {"skip": "cpu", "h": "cuda:0"}, pieces[1].fed
