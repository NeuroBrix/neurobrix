"""A released component collects the heap ONCE (the Dell 2026-11-04 08:14): mochi-1-preview native on Metal met 1-2 GB of
device memory held by Python cycles after its transformer's release (a gc + a flush gave them back, RUN 19), because
4d1187f9 took the per-segment collect out of `MemoryManager.unload_weights` — rightly: it cost every segment of every
decode step. The collect comes back at the COMPONENT's release, through each engine's own flow release.

What this test does if the code is wrong: no release at a component's unload (red on 6795b0d6), the compiled release on
the Triton engine (R33), or a release per segment (a streamed piece's own unload)."""
import types

import pytest

from neurobrix.core.runtime.executor import RuntimeExecutor


@pytest.mark.parametrize("mode,where", [("compiled", "neurobrix.core.memory.manager"),
                                        ("triton", "nbxtensor.memory.memory_pool")])
def test_a_component_release_runs_its_engines_flow_release_once(mode, where, monkeypatch):
    import importlib
    released = []
    for mod in ("neurobrix.core.memory.manager", "nbxtensor.memory.memory_pool"):
        m = importlib.import_module(mod)
        monkeypatch.setattr(m, "release_flow_memory", lambda d, _m=mod: released.append((_m, d)))
    unloads = []
    ex = types.SimpleNamespace(_weights_loaded=True, device="mps:0", unload_weights=lambda: unloads.append(1))
    rt = RuntimeExecutor.__new__(RuntimeExecutor)
    rt.executors, rt.strategy, rt.mode = {"transformer": ex}, None, mode
    rt._unload_component_weights("transformer")
    assert unloads == [1] and released == [(where, "mps:0")], released
    rt._unload_component_weights("transformer")             # nothing loaded any more: no second release
    assert released == [(where, "mps:0")]
