"""A temporal stream's chunk, once its outputs are copied out, returns the memory its run freed to the device.

mochi-1-preview's VAE in five 1-frame chunks on the M4 Pro (2026-11-09): each chunk's run was dropped
(`release_run_tensors`) but the MPS allocator kept its freed heaps cached; at the 4th chunk's conv::33 the heaps held
14.42 GiB over 8.08 GiB live and the run ran out of memory (RUN 87). With the engine's device release after each
chunk the same plan ran green (RUN 88). Card-less: the stream's release callback is captured and called."""
from types import SimpleNamespace

import pytest

from neurobrix.core.runtime import executor as ex


def _executor(monkeypatch, mode):
    calls = []
    run = object.__new__(ex.RuntimeExecutor)
    run.mode = mode
    run._nbx_path_str = "unused.nbx"
    piece = SimpleNamespace(release_run_tensors=lambda: calls.append("drop run"),
                            run=lambda inputs: {}, load_weights=lambda *a: None, unload_weights=lambda: None,
                            register_op_uid_interceptors=lambda i: None)
    decl = {"dags": {"first": {"tensors": {"x": {"input_name": "z"}}}, "next": {}}, "input": "x", "states": []}
    monkeypatch.setattr(ex, "ensure_extracted", lambda p: p)
    monkeypatch.setattr(ex, "_release_device_memory", lambda mode, device: calls.append(("device", mode, device)))
    from neurobrix.core.prism import temporal_stream as ts
    from neurobrix.core.module import temporal_stream_runner as tsr
    from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy
    monkeypatch.setattr(ts, "load", lambda cache, comp: decl)
    monkeypatch.setattr(tsr, "handed_inputs", lambda dag: frozenset())
    monkeypatch.setattr(tsr, "state_interceptors", lambda decl, park: {})
    monkeypatch.setattr(LayerStreamingStrategy, "_piece_executor", staticmethod(lambda *a, **k: piece))

    def stream(decl, schedule, x, name, first, nxt, release=None):
        for i, _ in enumerate(schedule):
            release("first" if i == 0 else "next")
        return "out"
    monkeypatch.setattr(tsr, "run_stream", stream)
    return run, calls


@pytest.mark.parametrize("mode", ["compiled", "triton"])
def test_each_chunk_drops_its_run_then_returns_its_memory_to_the_device(monkeypatch, mode):
    run, calls = _executor(monkeypatch, mode)
    component = SimpleNamespace(device="mps:0")
    out = run._run_temporal_stream("vae", component, {"z": "latents"}, {"schedule": [1, 1, 1]})
    assert out == "out"
    assert calls == ["drop run", ("device", mode, "mps:0")] * 3, calls
