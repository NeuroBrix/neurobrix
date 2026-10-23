"""A grid with no program launches nothing (kernels/launcher.launch), as Triton's own launchers skip a zero grid: an
element-wise op on an empty tensor computes nothing, and the tensor may hold no buffer, which triton-ext's Metal
driver refuses as a null pointer (Qwen3-VL-30B-A3B-Thinking triton, Mac 2026-10-23: an add over a (0, 2048) tensor).
What this test would do if the code were wrong: the launch reaches prepare and the driver -> red."""
import pytest

from neurobrix.kernels import launcher as L


def test_a_zero_grid_never_reaches_the_driver(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a zero-program launch reached prepare/the driver")
    monkeypatch.setattr(L, "prepare", boom)
    monkeypatch.setattr(L, "active_driver", boom)
    assert L.launch(object(), (0,), 1, 2) is None
    assert L.launch(object(), (4, 0), 1, 2) is None


def test_a_nonzero_grid_still_launches(monkeypatch):
    seen = []
    def prep(*a, **k):
        seen.append("prepare")
        raise RuntimeError("stop after prepare")
    monkeypatch.setattr(L, "prepare", prep)
    with pytest.raises(RuntimeError, match="stop after prepare"):
        L.launch(object(), (1,), 1)
    assert seen == ["prepare"]
