"""A piece Prism holds between the component's runs (`layer_stream_resident`) loads once, is never unloaded by a run, and
drops what its run left in its arena after every run — the weights alone are what the plan holds for it. A piece not held
loads and unloads around each run, as before. Before (the Mac, 2026-10-27 23:26): the held pieces kept their run tensors,
mochi-1-preview's transformer held 15.3 GB with 10.8 GB of held weights and MPS refused at the 27th piece. No card."""
from types import SimpleNamespace

from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy


class Piece:
    def __init__(self, i):
        self._dag = {"ops": {f"op{i}": {"output_tensor_ids": [f"t{i}"]}}, "segment_input_names": [f"t{i - 1}"],
                     "segment_index": i, "tensors": {}}
        self.i, self.calls = i, []
        self._weights_loaded = False

    def load_weights(self, *_):
        self.calls.append("load"); self._weights_loaded = True

    def unload_weights(self):
        self.calls.append("unload"); self._weights_loaded = False

    def release_run_tensors(self):
        self.calls.append("release")

    def protect_tensor_id(self, tid):
        pass

    def run(self, feed, *a, **k):
        self.calls.append("run")
        return {f"t{self.i}": feed[f"t{self.i - 1}"] + 1}

    def tensors_of_last_run(self, tids):
        return {}


def _strategy(held):
    ctx = SimpleNamespace(layer_resident={"lm": frozenset(held)}, layer_swapped=frozenset(), layer_segments={},
                          component_executors={}, topology={})
    s = LayerStreamingStrategy.__new__(LayerStreamingStrategy)
    s.context = ctx
    s._produced, s._last_read = {}, {}
    s._nbx_path = lambda name: "/nowhere"
    return s


def test_held_pieces_load_once_and_drop_their_run_the_others_stream():
    s = _strategy({0, 1})
    pieces = [Piece(1), Piece(2), Piece(3)]
    base = SimpleNamespace(_dag={"output_tensor_ids": ["t3"], "tensors": {}}, _persistent_tensor_ids=())
    for step in range(3):
        out = s._run_pieces("lm", base, pieces, {"t0": step})
        assert out == {"t3": step + 3}
    for p in pieces[:2]:
        assert p.calls == ["load", "run", "release", "run", "release", "run", "release"], p.calls
    assert pieces[2].calls == ["load", "run", "unload"] * 3, pieces[2].calls


def test_releasing_the_component_releases_its_held_pieces():
    s = _strategy({0})
    pieces = [Piece(1), Piece(2)]
    s._segment_executors, s._installed_pieces = {"lm": pieces}, {}
    base = SimpleNamespace(_dag={"output_tensor_ids": ["t2"], "tensors": {}}, _persistent_tensor_ids=())
    s._run_pieces("lm", base, pieces, {"t0": 0})
    s.uninstall_for_executor("lm", base)
    assert pieces[0].calls[-1] == "unload" and not pieces[0]._weights_loaded, pieces[0].calls
