"""A swapped component (`StrategyContext.layer_swapped`, Prism `layer_stream_swapped`) parks every value the next
piece does not read on the host, and brings each value a piece reads to that piece's device first
(`LayerStreamingStrategy._run_pieces`, `core/strategies/value_parking`): the rule Prism priced in
`LayerPartitioner._greedy_swapped`. The skip `a` below is made by the first piece and read by the last.

Injection (seen red, then restored green, 2026-10-22): the parking loop skipped -> `a` is never parked (red)."""
from __future__ import annotations

from types import SimpleNamespace

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.strategies import value_parking as VP
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class _Tensor:
    """A tensor that says where it lives, moved as a torch tensor is (`.to`)."""

    def __init__(self, name, device="cuda:0"):
        self.name, self.device = name, device

    @property
    def is_cuda(self):
        return self.device.startswith("cuda")

    def to(self, device):
        return _Tensor(self.name, str(device))


class _Strategy(LayerStreamingStrategy):
    def _nbx_path(self, component_name):
        return "<no container>"


class _Piece:
    device = "cuda:0"

    def __init__(self, index, reads, makes):
        self._dag = {"segment_index": index, "segment_input_names": list(reads),
                     "ops": {f"op{index}": {"output_tensor_ids": [makes]}}, "output_tensor_ids": [makes]}
        self.makes, self.fed = makes, {}

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
        return {self.makes: _Tensor(self.makes)}


def _run(swapped, monkeypatch):
    import neurobrix.core.strategies.layer_streaming as LS
    # the per-piece memory report reads a real device; what it prints is not what is judged here
    monkeypatch.setattr(LS, "_segment_memory_line", lambda seg: "")
    monkeypatch.setattr(LS, "_reset_segment_peak", lambda seg: None)
    parked = []
    real = VP.to_host

    def to_host(v):
        out = real(v)
        if out is not v:
            parked.append(v.name)
        return out
    monkeypatch.setattr(VP, "to_host", to_host)
    pieces = [_Piece(0, ["x"], "a"), _Piece(1, ["a"], "b"), _Piece(2, ["b"], "c"), _Piece(3, ["c", "a"], "z")]
    base = SimpleNamespace(_dag={"tensors": {}, "output_tensor_ids": ["z"]}, _persistent_tensor_ids=set())
    ctx = SimpleNamespace(component_executors={}, layer_segments={}, layer_graphs={}, layer_moe={},
                          layer_chunks={}, layer_swapped=frozenset({"comp"} if swapped else ()))
    out = _Strategy(ctx, "layer_streaming")._run_pieces("comp", base, pieces, {"x": _Tensor("x")})
    return out, pieces, parked


def test_the_skip_waits_on_the_host_and_comes_back_for_its_reader(monkeypatch):
    out, pieces, parked = _run(True, monkeypatch)
    assert "a" in parked, parked                     # parked before the pieces that do not read it
    assert "z" not in parked                          # a declared output is never parked
    assert all(d == "cuda:0" for p in pieces for d in p.fed.values()), [p.fed for p in pieces]
    assert out["z"].device == "cuda:0"


def test_without_the_plan_nothing_moves(monkeypatch):
    _out, pieces, parked = _run(False, monkeypatch)
    assert parked == [], parked
