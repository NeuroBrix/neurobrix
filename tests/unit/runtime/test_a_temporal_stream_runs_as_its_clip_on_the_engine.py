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
        from nbxtensor.tensor import NBXTensor
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
    from neurobrix.core.module.temporal_stream_runner import handed_inputs, state_interceptors
    pieces = {k: LayerStreamingStrategy._piece_executor(whole, decl["dags"][k], comp, whole,
                                                        consumed_inputs=handed_inputs(decl["dags"][k]))
              for k in ("first", "next")}     # as the runtime builds them (`Executor._run_temporal_stream`)
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


def test_a_states_source_dies_at_its_last_consumer_in_the_compiled_engine():
    """The compiled engine pins the storage chain of each graph output through view ops (UMT5's reassembled norm,
    graph_executor's output protection). A state's slice is copied by its interceptor, so its source must still be
    freed after it: each `stream.state_out::<k>` op kills its source's slot. Red without the interceptor stop in
    `_protect_storage_chain`: the kills are removed, and every causal site's input lives to the chunk's end (Wan's
    VAE, a 2-frame chunk: 17.3 GB on the card against 6.0 GB with the stop)."""
    container, decl = _case("Wan2.1-T2V-1.3B", "vae")
    from neurobrix.core.module.temporal_stream_runner import state_interceptors
    clip_dag = json.loads((container / "components" / "vae" / "graph.json").read_text())
    whole = _executor(container, "vae", clip_dag, "compiled", "cpu")
    whole.load_weights(str(container), "vae")
    first = LayerStreamingStrategy._piece_executor(whole, decl["dags"]["first"], "vae", whole)
    first.register_op_uid_interceptors(state_interceptors(decl))
    first.load_weights(str(container), "vae")
    first.run({"z": torch.randn(1, 16, 2, 4, 4)})
    seq = first._compiled_seq
    order = first._dag["execution_order"]
    killed = 0
    for k in range(len(decl["states"])):
        idx = order.index(f"stream.state_out::{k}")
        src = first._dag["ops"][f"stream.state_out::{k}"]["input_tensor_ids"][0]
        if seq._tensor_id_to_slot.get(src) in seq._ops[idx].kill_slots:
            killed += 1
    assert killed > 0, "no state's source is freed after its state op: the output protection pinned them all"


@pytest.mark.parametrize("mode,device", MODES)
def test_a_handed_state_is_freed_at_its_last_use(mode, device):
    """The runner HANDS a chunk and its states to the run (`handed_inputs`, `GraphExecutor._consumed_input_tids`): the
    run takes them out of the caller's dict and each is freed at its last use, as Prism prices them
    (`handed_inputs=True`). Held as protected inputs they lived through the whole chunk: mochi's 5 GB of states
    beside a 13 GB chunk, 18.2 GB measured on a card the plan had priced at 13.1 (2026-10-17). Red if the run keeps
    them: torch engines — a state still referenced once the run returned, before any release; Triton engines — the
    device still holding the states' bytes then."""
    container, decl = _case("Wan2.1-T2V-1.3B", "vae")
    from neurobrix.core.module.temporal_stream_runner import handed_inputs, state_interceptors
    clip_dag = json.loads((container / "components" / "vae" / "graph.json").read_text())
    whole = _executor(container, "vae", clip_dag, mode, device)
    whole.load_weights(str(container), "vae")
    pieces = {k: LayerStreamingStrategy._piece_executor(whole, decl["dags"][k], "vae", whole,
                                                        consumed_inputs=handed_inputs(decl["dags"][k]))
              for k in ("first", "next")}
    for p in pieces.values():
        p.register_op_uid_interceptors(state_interceptors(decl))
        p.load_weights(str(container), "vae")
    torch.manual_seed(0)

    def chunk_inputs():
        out = pieces["first"].run({"z": _input(torch.randn(1, 16, 2, 4, 4), mode, device)})
        ins = {s["name"]: out.pop(s["name"]) for s in decl["states"]}
        pieces["first"].release_run_tensors()
        ins["z"] = _input(torch.randn(1, 16, 1, 4, 4), mode, device)
        return ins
    # one warm run first: what a piece builds once and keeps across runs (its bound constants) is no run's tensor
    pieces["next"].run(chunk_inputs())
    pieces["next"].release_run_tensors()
    inputs = chunk_inputs()
    if mode.startswith("triton"):
        from nbxtensor.memory.allocator import DeviceAllocator as DA
        state_bytes = sum(int(v.numel()) * int(v.element_size()) for k, v in inputs.items() if k != "z")
        DA.empty_cache_pool()
        before = DA.memory_allocated()
        result = pieces["next"].run(inputs)
        DA.empty_cache_pool()
        kept = DA.memory_allocated() - before
        out_bytes = sum(int(v.numel()) * int(v.element_size()) for v in result.values())
        assert not inputs, f"the run left {sorted(inputs)} in its caller's dict"
        assert kept <= out_bytes, (f"{mode}: the device holds {kept} B after the run, its outputs are {out_bytes} B: "
                                   f"the handed states ({state_bytes} B) were kept")
        # and once the caller holds its outputs and releases the run, the run holds NOTHING of the pass: the
        # triton-sequential weight view kept the whole store bound until the next pass (3.6 GB of Wan states)
        kept_out = {k: v for k, v in result.items()}
        del result
        pieces["next"].release_run_tensors()
        DA.empty_cache_pool()
        left = DA.memory_allocated() - before - sum(int(v.numel()) * int(v.element_size()) for v in kept_out.values())
        assert left <= 0, f"{mode}: {left} B of the pass still held after its release, beyond the outputs the caller keeps"
        return
    import weakref
    refs = {k: weakref.ref(v) for k, v in inputs.items()}
    result = pieces["next"].run(inputs)
    assert not inputs, f"the run left {sorted(inputs)} in its caller's dict"
    alive = sorted(k for k, r in refs.items() if r() is not None)
    assert not alive, f"{mode}: {len(alive)} handed input(s) still alive once the run returned, e.g. {alive[:3]}"
    assert result


@pytest.mark.parametrize("mode,device", MODES)
@pytest.mark.parametrize("model,comp,shape,schedule", CASES, ids=[f"{m}/{c}" for m, c, _, _ in CASES])
def test_the_engine_runs_the_parked_stream_as_the_clip(model, comp, shape, schedule, mode, device):
    """States PARKED on the host between chunks (strategy D, Hocine 2026-10-27 11:01; the plan's `park_states`): each
    state is copied to the host by the op that makes it (`park_state`) and comes back to the card right before the
    first op of the next chunk that reads it (`ParkedTensor`, the sequences' `defer_inputs`). The stream must still
    be the clip. Red if a parked state is bound to the wrong tensor, never bound (the run reads nothing there), or
    bound after its reader."""
    container, decl = _case(model, comp)
    clip_dag = json.loads((container / "components" / comp / "graph.json").read_text())
    whole = _executor(container, comp, clip_dag, mode, device)
    whole.load_weights(str(container), comp)
    from neurobrix.core.module.temporal_stream_runner import ParkedTensor, handed_inputs, state_interceptors
    pieces = {k: LayerStreamingStrategy._piece_executor(whole, decl["dags"][k], comp, whole,
                                                        consumed_inputs=handed_inputs(decl["dags"][k]))
              for k in ("first", "next")}
    for p in pieces.values():
        p.register_op_uid_interceptors(state_interceptors(decl, park=True))
        p.load_weights(str(container), comp)
    input_name = clip_dag["tensors"][decl["input"]]["input_name"]
    out_name = clip_dag["tensors"][decl["output"]].get("output_name") or decl["output"]
    torch.manual_seed(0)
    x = _input(torch.randn(shape, dtype=torch.float32), mode, device)
    ref = _host(whole.run({input_name: x})[out_name])
    parked = []

    def run_first(inputs):
        out = pieces["first"].run(inputs)
        parked.extend(type(out[s["name"]]) is ParkedTensor for s in decl["states"])
        return out
    got = _host(run_stream(decl, schedule, x, input_name, run_first, pieces["next"].run))
    assert parked and all(parked), "the first chunk's states did not leave the run parked on the host"
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel < BOUND, f"{model}/{comp} {mode}: the parked stream differs from the clip by {rel:.2e}"


def test_a_parked_state_comes_back_at_its_first_reader():
    """The compiled sequence binds each parked state right before the FIRST op that reads it (`defer_inputs`), never
    at the chunk's start — where Prism prices it (`deferred_inputs`). Red if the states are bound with the chunk (no
    deferred binding at all) or at another op than their first reader."""
    container, decl = _case("Wan2.1-T2V-1.3B", "vae")
    from neurobrix.core.module.temporal_stream_runner import handed_inputs, state_interceptors
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence
    clip_dag = json.loads((container / "components" / "vae" / "graph.json").read_text())
    whole = _executor(container, "vae", clip_dag, "compiled", "cpu")
    whole.load_weights(str(container), "vae")
    pieces = {k: LayerStreamingStrategy._piece_executor(whole, decl["dags"][k], "vae", whole,
                                                        consumed_inputs=handed_inputs(decl["dags"][k]))
              for k in ("first", "next")}
    for p in pieces.values():
        p.register_op_uid_interceptors(state_interceptors(decl, park=True))
        p.load_weights(str(container), "vae")
    out = pieces["first"].run({"z": torch.randn(1, 16, 2, 4, 4)})
    ins = {s["name"]: out[s["name"]] for s in decl["states"]}
    ins["z"] = torch.randn(1, 16, 1, 4, 4)
    seen = []
    real = CompiledSequence._bind_deferred

    def spy(self, op_idx):
        seen.extend((op_idx, tid) for tid, _ in self._deferred_at.get(op_idx, ()))
        return real(self, op_idx)
    CompiledSequence._bind_deferred = spy
    try:
        pieces["next"].run(ins)
    finally:
        CompiledSequence._bind_deferred = real
    seq = pieces["next"]._compiled_seq
    want = set()
    for s in decl["states"]:
        slot = seq._tensor_id_to_slot[f"input::{s['name']}"]
        want.add((next(i for i, op in enumerate(seq._ops) if slot in (op.all_input_slots or ())), f"input::{s['name']}"))
    assert set(seen) == want and len(seen) == len(want), (sorted(set(seen) ^ want)[:4], len(seen), len(want))
