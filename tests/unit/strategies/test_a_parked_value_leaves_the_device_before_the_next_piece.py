"""A value parked on the host leaves the DEVICE before the next piece runs: nothing of `_run_pieces` keeps its device
tensor bound while that piece computes — not the previous piece's result dict, not its feed. HAT-S at 2048 px on a
16 GB V100 (2026-10-22): piece 1 held piece 0's 4 096 MB attention mask and two 2 304 MB maps the plan had parked,
through the previous piece's `out` still bound, and OOMed at its first copy.

Injection (seen red, then restored green, 2026-10-22): `del out, feed` removed -> the parked device tensor is alive
during the next piece (red)."""
from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class _Tensor:
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

    def __init__(self, index, reads, makes, seen):
        self._dag = {"segment_index": index, "segment_input_names": list(reads),
                     "ops": {f"op{index}": {"output_tensor_ids": list(makes)}}, "output_tensor_ids": list(makes)}
        self.makes, self.seen = makes, seen

    def protect_tensor_id(self, tid):
        pass

    def load_weights(self, *a, **k):
        pass

    def unload_weights(self):
        pass

    def tensors_of_last_run(self, tids):
        return {}

    def run(self, feed, *a, **k):
        gc.collect()
        # the device tensors every earlier piece made, alive or not while THIS piece computes
        self.seen.append({n: r() is not None for n, r in _made.items()})
        out = {}
        for m in self.makes:
            t = _Tensor(m)
            _made[m] = weakref.ref(t)
            out[m] = t
        return out


_made: dict = {}


def test_a_parked_value_is_off_the_device_while_the_next_piece_runs(monkeypatch):
    import neurobrix.core.strategies.layer_streaming as LS
    monkeypatch.setattr(LS, "_segment_memory_line", lambda seg: "")
    monkeypatch.setattr(LS, "_reset_segment_peak", lambda seg: None)
    _made.clear()
    seen = []
    # piece 0 makes `mask` (read only by piece 2) and `h`; piece 1 reads `h` alone; piece 2 reads `mask` and `g`
    pieces = [_Piece(0, ["x"], ["mask", "h"], seen), _Piece(1, ["h"], ["g"], seen), _Piece(2, ["mask", "g"], ["z"], seen)]
    base = SimpleNamespace(_dag={"tensors": {}, "output_tensor_ids": ["z"]}, _persistent_tensor_ids=set())
    ctx = SimpleNamespace(component_executors={}, layer_segments={}, layer_graphs={}, layer_moe={},
                          layer_chunks={}, layer_swapped=frozenset({"comp"}))
    _Strategy(ctx, "layer_streaming")._run_pieces("comp", base, pieces, {"x": _Tensor("x")})
    assert seen[1]["mask"] is False, seen      # parked: its device tensor gone while piece 1 computes
    assert seen[1]["h"] is True, seen          # piece 1 reads it: on the device
    assert seen[2]["h"] is False, seen         # no later reader: released, as the whole run frees it
