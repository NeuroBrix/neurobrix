"""`MemoryManager.unload_weights` releases the MoE pointer tables that pin weights (Metal) — and a release
that FAILS is raised, never swallowed: a table that could not release its pins keeps the unloaded weights in
memory, which is the defect the step exists to remove (deepseek-moe-16b streamed on the Mac, 2026-10-04:
swap 3.4 -> 10.2 GB while the second segment loaded). The step first landed inside `except Exception: pass`."""
import sys
import types

import pytest

from neurobrix.core.memory.manager import MemoryManager


def _stub(monkeypatch, fn):
    monkeypatch.setitem(sys.modules, "neurobrix.triton.moe", types.SimpleNamespace(release_pinned_tables=fn))


def test_the_unload_asks_the_loaded_moe_module_to_release_its_pinned_tables(monkeypatch):
    calls = []
    _stub(monkeypatch, lambda: calls.append(1) or 0)
    MemoryManager.unload_weights({}, clear_cuda_cache=False)
    assert calls == [1]


def test_a_release_that_fails_is_raised(monkeypatch):
    def boom():
        raise RuntimeError("a pin would not exit")
    _stub(monkeypatch, boom)
    with pytest.raises(RuntimeError, match="a pin would not exit"):
        MemoryManager.unload_weights({}, clear_cuda_cache=False)


def test_a_run_that_never_loaded_the_moe_module_imports_nothing(monkeypatch):
    monkeypatch.delitem(sys.modules, "neurobrix.triton.moe", raising=False)
    MemoryManager.unload_weights({}, clear_cuda_cache=False)
    assert "neurobrix.triton.moe" not in sys.modules
