"""A component says when it leaves the device, as it says when it arrives.

The load prints "'<component>': loading N weights"; nothing said when a component was released. The Mac's unified
overruns (2026-10-14) peaked at a component load — Allegro's transformer, VACE's text encoder with its VAE encoder
— and a per-second memory trace could not tell whether the previous component had left. The release line is the
second anchor of that trace, in both engines (GraphExecutor.unload_weights is theirs).

What this test does if the code is wrong: the release cell fails when an unload prints no line naming the component
and the weights it released (red on main); the empty cell fails when an executor holding nothing prints one.

Run: python -m pytest tests/unit/runtime/test_a_component_says_when_it_is_released.py
"""
import pytest

from neurobrix.core.runtime.graph_executor import GraphExecutor


def _executor(mode, weights, monkeypatch):
    from neurobrix.core.memory.manager import MemoryManager
    monkeypatch.setattr(MemoryManager, "cleanup_context", staticmethod(lambda ctx: None))
    monkeypatch.setattr(MemoryManager, "unload_weights", staticmethod(lambda w, *a, **k: w.clear()))
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._persistent = False
    ex._compiled_seq = None
    ex._triton_seq = None
    ex._pieces_capture = None
    ex._weights = weights
    ex._ctx = None
    ex.mode = mode
    ex._component_name = "transformer"
    return ex


@pytest.mark.parametrize("mode,label", [("compiled", "Compiled"), ("triton", "Triton")])
def test_an_unload_says_which_component_left_and_what_it_released(mode, label, monkeypatch, capsys):
    _executor(mode, {"a": 1, "b": 2, "c": 3}, monkeypatch).unload_weights()
    assert f"[{label}] 'transformer': released 3 weights" in capsys.readouterr().out


def test_an_executor_holding_nothing_says_nothing(monkeypatch, capsys):
    _executor("compiled", {}, monkeypatch).unload_weights()
    assert "released" not in capsys.readouterr().out


def test_a_load_says_what_it_holds_each_storage_once_and_what_sits_on_the_host():
    """The load's second line: bytes of the loaded tensors (a view and its base counted once), how many are on the
    host — the discriminator between a host copy, allocator overhead and a wider dtype on the Mac's 2x loads."""
    import torch
    from neurobrix.core.runtime.graph_executor import _held_line
    base = torch.zeros(1024, dtype=torch.bfloat16)
    line = _held_line({"w": base, "w_view": base[:512], "b": torch.zeros(256, dtype=torch.float32)}, "cpu")
    assert line.startswith("3 tensors, 0 MB, 3 on the host"), line
    big = torch.zeros(1 << 20, dtype=torch.bfloat16)          # 2 MB
    assert _held_line({"x": big, "y": big[1:]}, "cpu").startswith("2 tensors, 2 MB"), "one storage, counted once"
