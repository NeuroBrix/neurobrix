"""A causal video component run in temporal chunks by the ENGINE equals the same component run whole, in every engine.

Forge derives the stream from the clip graph (`components/<c>/stream_first.json`, `stream_next.json`, `stream.json`)
and proves it in float64 with ATen. This file proves the engine's half: the real `GraphExecutor` running the clip graph
whole, and the stream brick (`core/module/temporal_stream_runner.run_stream`) driving the two chunk graphs on piece
executors that BORROW the whole executor's weights (`LayerStreamingStrategy._piece_executor`), on the container's own
weights, at a small extent, give the same output to the engine's float32 noise.

What it would do if the code were wrong:
  * a state not fed back, fed under the wrong name, or read from a memory the next run overwrote: the frames after the
    first chunk read other context — off by order 1 (the Forge proof's zero-state injection measured 1.4);
  * a chunk written at the wrong place on the output's time axis: the output differs at those frames;
  * the piece executors not holding the weights: the run refuses by name.

The containers are the cache's (`neurobrix.core.paths.cache_dir`, NEUROBRIX_CACHE for a stage); a machine without the
container, or a container not yet carrying its stream, skips the case by name. CogVideoX's stream is the vendor's
chunked function, not its clip graph (its group norms read the chunk): proven against the vendor in Forge
(tests/test_a_vendor_chunked_vae_is_the_declared_stream.py), not here.
"""
from __future__ import annotations

import json

import pytest

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.module.temporal_stream_runner import run_stream
from neurobrix.core.paths import cache_dir
from neurobrix.core.prism import temporal_stream as ts
from neurobrix.core.runtime.graph_executor import GraphExecutor
from neurobrix.core.strategies.layer_streaming import LayerStreamingStrategy

torch = pytest.importorskip("torch")

# (container, component, input shape, schedule in input frames)
CASES = [
    ("Wan2.1-T2V-1.3B", "vae", [1, 16, 5, 4, 4], [2, 1, 2]),
    ("Wan2.2-I2V-A14B", "vae_encoder", [1, 3, 9, 32, 32], [5, 4]),
    ("mochi-1-preview", "vae", [1, 12, 4, 4, 4], [1, 2, 1]),
]
BOUND = 1e-4      # float32 convolutions over different time extents: the engine's own noise


def _case(model, comp):
    c = cache_dir() / model
    if not (c / "components" / comp / "graph.json").exists():
        pytest.skip(f"{model}/{comp}: not in this machine's cache ({c})")
    decl = ts.load(c, comp)
    if decl is None:
        pytest.skip(f"{model}/{comp}: the container carries no stream yet (Forge tools/derive_temporal_stream.py)")
    return c, decl


def _executor(container, comp, dag, mode, device):
    ex = GraphExecutor(family="video", vendor="nvidia", arch="volta", device=device, dtype="float32", mode=mode)
    ex.load_graph_from_dict(dag)
    ex._component_name = comp
    return ex


def _cuda() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


MODES = [("compiled", "cpu"), ("sequential", "cpu")] + (
    [("compiled", "cuda"), ("triton", "cuda"), ("triton_sequential", "cuda")] if _cuda() else [])


def _input(x, mode, device):
    """The run's own tensor kind: torch for the two torch engines, NBXTensor for the two Triton ones (R30/R33)."""
    if mode.startswith("triton"):
        from neurobrix.kernels.nbx_tensor import NBXTensor
        return NBXTensor.from_numpy(x.numpy()).to_cuda()
    return x.to(device)


def _host(t):
    if hasattr(t, "to_cuda") and hasattr(t, "_device"):          # an NBXTensor: read back to the host
        return torch.from_numpy(t.numpy()).float()
    return t.detach().float().cpu()


@pytest.mark.parametrize("mode,device", MODES)
@pytest.mark.parametrize("model,comp,shape,schedule", CASES, ids=[f"{m}/{c}" for m, c, _, _ in CASES])
def test_the_engine_runs_the_stream_as_the_clip(model, comp, shape, schedule, mode, device):
    container, decl = _case(model, comp)
    assert not ts.broken(decl, schedule), ts.broken(decl, schedule)
    clip_dag = json.loads((container / "components" / comp / "graph.json").read_text())
    whole = _executor(container, comp, clip_dag, mode, device)
    whole.load_weights(str(container), comp)
    pieces = {k: LayerStreamingStrategy._piece_executor(whole, decl["dags"][k], comp, whole) for k in ("first", "next")}
    from neurobrix.core.module.temporal_stream_runner import state_interceptors
    for p in pieces.values():
        p.register_op_uid_interceptors(state_interceptors(decl))       # as the runtime runs them
    for p in pieces.values():
        p.load_weights(str(container), comp)
    input_name = clip_dag["tensors"][decl["input"]]["input_name"]
    out_name = clip_dag["tensors"][decl["output"]].get("output_name") or decl["output"]
    torch.manual_seed(0)
    x = _input(torch.randn(shape, dtype=torch.float32), mode, device)
    ref = _host(whole.run({input_name: x})[out_name])
    got = _host(run_stream(decl, schedule, x, input_name, pieces["first"].run, pieces["next"].run))
    assert tuple(got.shape) == tuple(ref.shape), (tuple(got.shape), tuple(ref.shape))
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel < BOUND, f"{model}/{comp} {mode}: the stream differs from the clip by {rel:.2e}"


def test_a_schedule_that_does_not_cover_the_input_is_refused():
    container, decl = _case("Wan2.1-T2V-1.3B", "vae")
    x = torch.zeros(1, 16, 5, 4, 4)
    with pytest.raises(ValueError, match="covers 4 frames of an input of 5"):
        run_stream(decl, [2, 2], x, "z", lambda i: {}, lambda i: {})


@pytest.mark.parametrize("mode", ["compiled", "sequential"])
def test_every_state_is_copied_by_its_own_op(mode):
    """A state the runtime carries is COPIED where its op produces it (`state_interceptors`): as a slice view it kept
    its whole source — a causal site's full input — alive to the end of the chunk (mochi: 14.5 GB at a chunk priced at
    10.2 GB). The copy is the interceptor's: each state's op must run through it once per chunk. Red if the
    interceptor is not registered or not reached (the run's outputs alone cannot tell: an executor may copy its
    outputs at the end, after the source was pinned the whole run)."""
    container, decl = _case("Wan2.1-T2V-1.3B", "vae")
    from neurobrix.core.module import temporal_stream_runner as R
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return R.materialize_state(*a, **k)
    spy.self_manages_dtype = True
    clip_dag = json.loads((container / "components" / "vae" / "graph.json").read_text())
    whole = _executor(container, "vae", clip_dag, mode, "cpu")
    whole.load_weights(str(container), "vae")
    first = LayerStreamingStrategy._piece_executor(whole, decl["dags"]["first"], "vae", whole)
    first.register_op_uid_interceptors({uid: spy for uid in R.state_interceptors(decl)})
    first.load_weights(str(container), "vae")
    first.run({"z": torch.randn(1, 16, 2, 4, 4)})
    assert len(calls) == len(decl["states"]), (len(calls), len(decl["states"]))
