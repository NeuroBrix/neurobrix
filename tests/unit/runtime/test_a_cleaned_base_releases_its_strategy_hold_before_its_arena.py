"""A non-persistent executor's `cleanup()` (the sessions' end of request: the Janus image path, score mode, a ctx
persistent over executors built non-persistent) releases the base's weight dict, so its arena. Before that free, the
strategy that installed itself on the executor drops what it holds into that arena: the held layer-streaming pieces
unload (their borrows), the zero3 ratchet and its callbacks go — the order `RuntimeExecutor._unload_component_weights`
keeps (uninstall, then unload). Before: `cleanup()` freed the base with the install armed, cleared `_weights_loaded`,
and the later ordered unload returned at its first guard, so held pieces kept borrows into the freed arena (the alias
hazard reading of 2026-11-24, case 3). Both branches. No card."""
from types import SimpleNamespace

import pytest

from neurobrix.core.memory import MemoryManager
from neurobrix.core.runtime.graph_executor import GraphExecutor


class Piece:
    """A held piece: loaded, its table borrowing the base's weights until it unloads."""

    def __init__(self, order):
        self.order, self._weights_loaded = order, True

    def unload_weights(self):
        """The piece drops its table, borrows included: recorded in the shared order."""
        self.order.append("piece unloads")
        self._weights_loaded = False


def _base(order, monkeypatch):
    """A non-persistent base executor holding one weight; the free of its dict is recorded in `order`."""
    base = GraphExecutor.__new__(GraphExecutor)
    base._persistent, base._compiled_seq, base._triton_seq = False, None, None
    base._pieces_capture, base._ctx, base._weights = None, None, {"w": object()}
    base._weights_loaded = True
    monkeypatch.setattr(MemoryManager, "unload_weights",
                        staticmethod(lambda weights: (order.append("base arena freed"), weights.clear())))
    return base


@pytest.mark.parametrize("branch", ["triton", "pytorch"])
def test_cleanup_unloads_the_held_pieces_before_it_frees_the_base(branch, monkeypatch):
    mod = __import__(f"neurobrix.{branch}.strategies.layer_streaming", fromlist=["LayerStreamingStrategy"])
    order = []
    base = _base(order, monkeypatch)
    s = mod.LayerStreamingStrategy.__new__(mod.LayerStreamingStrategy)
    s.context = SimpleNamespace(layer_segments={"lm": [["op0", "op1"]]})
    s._installed, s._installed_pieces, s._segment_executors = {"lm"}, {}, {"lm": [Piece(order)]}
    s._ensure_flow_reads = lambda name, executor: None
    assert s.install_for_executor("lm", base) is True
    base.cleanup()
    assert order == ["piece unloads", "base arena freed"], order


@pytest.mark.parametrize("branch", ["triton", "pytorch"])
def test_cleanup_disarms_the_zero3_ratchet_before_it_frees_the_base(branch, monkeypatch):
    mod = __import__(f"neurobrix.{branch}.strategies.zero3", fromlist=["Zero3Strategy"])
    order = []
    base = _base(order, monkeypatch)
    s = mod.Zero3Strategy.__new__(mod.Zero3Strategy)
    s._installed, s._ratchet, s._loaded_components, s._pinned_components = {}, {}, {"lm"}, {"lm"}
    s._install("lm", base)
    monkeypatch.setattr(MemoryManager, "unload_weights", staticmethod(lambda weights: order.append(
        ("base arena freed", "lm" in s._installed, base._persistent_pre_op_callback is not None))))
    base.cleanup()
    assert order == [("base arena freed", False, False)], order
    assert not s._pinned_components and not s._loaded_components
    assert getattr(base, "strategy_uninstall", None) is None      # disarmed: the next load installs afresh
