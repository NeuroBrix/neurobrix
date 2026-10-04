"""A flow feeds its first stage the recording's REAL length — never the trace's — and a container
that cannot take it is refused by name.

parakeet-tdt-1.1b and canary-qwen-2.5b, 2026-10-04: both encoders carry their frame axis as a
symbol at the input and froze dims downstream of it at the trace's audio length (a pad mask
repeated 375 times, a relative-position window sliced at 375). No gate saw it, because the engine
zero-padded every recording to the 3 000 mel frames of the trace (`core/flow/rnnt.py`, its Triton
mirror, `audio_utils._fit_to_trace`, `audio_frontend.fit_features`, the audio flow's inline fit):
no clip ever reached a graph at another length. The padding compensated a build-side defect at
runtime; it is removed, and the door that replaces it (`core/flow/input_extent.admit`) reads the
graph's own symbol table:

* a symbolic axis takes the fed extent — proven here at two lengths away from the trace, one far;
* an axis the graph froze AT THE INPUT is refused by name (container, component, input, axis,
  trace value, extent fed, the repair);
* a dim the graph froze DOWNSTREAM of a symbolic input is refused by name (the op, the tensor,
  the dim, its trace value) — the two containers' actual defect;
* an axis frozen by the VENDOR's contract (the whisper extractor pads to its own 30 s window)
  reaches the graph at that extent, produced by the front end, and is admitted;
* the RNNT decode walks the encoder's own frame axis, and the long-form window is the family
  profile's value, never the trace extent;
* Prism binds the encoder's symbols to the recording's frames (the plan and the derived census).

Both engines (R30): every flow cell runs on the ATen flow and on its Triton mirror. CPU only: the
Triton cells stand a numpy carrier in for NBXTensor (no card), which is the flows' shape logic.

Injections (each seen RED, then restored GREEN — recorded in the commit message):
the pad restored in the rnnt flows; the literal-axis refusal removed from `admit`; the broadcast
scan removed from `admit`; the decode bound put back on `ceil(length / 8)`; the audio rule removed
from `FlowBindings.overrides`.

Run: python -m pytest tests/unit/flow/test_a_flow_feeds_the_real_length_never_the_traces.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from neurobrix.core.flow import input_extent as IE

TRACE_FRAMES = 3000
SR, HOP = 16000, 160                      # the NeMo extractor's own rate and hop (mel_dsp defaults)


def _sym(i, t):
    return {"type": "symbol", "id": i, "trace": t}


def _sub(e, k=8):
    """ceil(e / k) as the conformer's subsampling writes it: ((e - 1) // k) + 1."""
    tv = (e["trace"] - 1) // k + 1
    return {"type": "add", "trace": tv, "right": 1,
            "left": {"type": "floordiv", "trace": tv - 1, "right": k,
                     "left": {"type": "add", "left": e, "right": -1, "trace": e["trace"] - 1}}}


def encoder_graph(*, frozen_input=False, frozen_downstream=False, feat="audio_signal",
                  length="length", mels=80, layout="mel_first"):
    """A small encoder graph in the container's own encoding: the features input, its length,
    and one elementwise op between a mask built from the frame axis and the subsampled frames.
    `frozen_downstream` keeps the mask at the trace's 375 frames — the two containers' defect."""
    s0, s1, s2 = _sym("s0", 1), _sym("s1", TRACE_FRAMES), _sym("s2", 1)
    frames = TRACE_FRAMES if frozen_input else s1
    t_enc = TRACE_FRAMES // 8 if frozen_input else _sub(s1)
    mask = TRACE_FRAMES // 8 if (frozen_downstream or frozen_input) else _sub(s1)
    dims = [s0, mels, frames] if layout == "mel_first" else [s0, frames, mels]
    shape = [1, mels, TRACE_FRAMES] if layout == "mel_first" else [1, TRACE_FRAMES, mels]
    symbols = {"s0": {"name": "batch", "trace_value": 1, "source": f"input::{feat}::dim_0",
                      "constraints": {"min": 1}},
               "s2": {"name": "batch", "trace_value": 1, "source": f"input::{length}::dim_0",
                      "constraints": {"min": 1}}}
    if not frozen_input:
        axis = 2 if layout == "mel_first" else 1
        symbols["s1"] = {"name": "seq_len", "trace_value": TRACE_FRAMES,
                         "source": f"input::{feat}::dim_{axis}", "constraints": {"min": 1}}
    return {
        "component_name": "encoder",
        "input_tensor_ids": [f"input::{feat}", f"input::{length}"],
        "output_tensor_ids": ["aten.logical_and::0::out_0"],
        "execution_order": ["aten.logical_and::0"],
        "symbolic_context": {"symbols": symbols},
        "tensors": {
            f"input::{feat}": {"input_name": feat, "shape": shape, "dtype": "float32",
                               "symbolic_shape": {"dims": dims, "concrete": shape}},
            f"input::{length}": {"input_name": length, "shape": [1], "dtype": "int64",
                                 "symbolic_shape": {"dims": [s2], "concrete": [1]}},
            "frames::out_0": {"shape": [1, 375, 375], "dtype": "bool",
                              "symbolic_shape": {"dims": [1, t_enc, t_enc]}},
            "mask::out_0": {"shape": [1, 375, 375], "dtype": "bool",
                            "symbolic_shape": {"dims": [s2, mask, mask]}},
            "aten.logical_and::0::out_0": {"shape": [1, 375, 375], "dtype": "bool",
                                           "symbolic_shape": {"dims": [s2, t_enc, t_enc]}},
        },
        "ops": {"aten.logical_and::0": {
            "op_uid": "aten.logical_and::0", "op_type": "aten::logical_and",
            "input_tensor_ids": ["mask::out_0", "frames::out_0"],
            "output_tensor_ids": ["aten.logical_and::0::out_0"], "parent_module": "ConformerEncoder",
            "attributes": {"args": [{"type": "tensor", "tensor_id": "mask::out_0"},
                                    {"type": "tensor", "tensor_id": "frames::out_0"}]}}},
    }


def _wav(tmp_path, seconds, name=None):
    rng = np.random.default_rng(int(seconds * 1000))
    path = tmp_path / (name or f"clip_{seconds}s.wav")
    sf.write(str(path), (0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32), SR,
             subtype="PCM_16")
    return path


def _mel_frames(seconds):
    return 1 + int(seconds * SR) // HOP


def _container(tmp_path, graph, *, flow_type="rnnt", preprocessing="nemo_mel", name="toy-stt",
               family="stt", stage="encoder", feat="audio_signal"):
    root = tmp_path / name
    (root / "modules" / "tokenizer").mkdir(parents=True)
    (root / "components" / stage).mkdir(parents=True)
    graph = dict(graph, component_name=stage)
    (root / "components" / stage / "graph.json").write_text(json.dumps(graph))
    topo = {"flow": {"type": flow_type,
                     "audio": {"input": {"modality": "audio", "preprocessing": preprocessing,
                                         "variable": "global.input_features"},
                               "stages": [{"component": stage, "execution": "forward"}]}},
            "connections": [{"from": "global.input_features", "to": f"{stage}.{feat}"}]}
    manifest = {"model_name": name, "family": family, "dtype": "float32"}
    (root / "topology.json").write_text(json.dumps(topo))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, topo, manifest, graph


def _ctx(root, topo, manifest, graph, audio, *, stage="encoder", device="cpu", defaults=None):
    return SimpleNamespace(
        pkg=SimpleNamespace(topology=topo, manifest=manifest, defaults=dict(defaults or {}),
                            cache_path=root),
        variable_resolver=SimpleNamespace(resolved={"global.audio_path": str(audio)}),
        executors={stage: SimpleNamespace(_dag=graph, _weights={})},
        modules={}, plan=None, primary_device=device, nbx_path_str=str(root),
        persistent_mode=False, mode="compiled",
        compute_dtype=lambda component=None: torch.float32)


class _Carrier:
    """The numpy stand-in for NBXTensor on a host with no card: a shape and its array."""

    def __init__(self, arr):
        self._a = np.asarray(arr)
        self.shape = tuple(self._a.shape)

    @classmethod
    def from_numpy(cls, arr):
        return cls(arr)

    def numpy(self):
        return self._a


def _fn(*_a, **_k):
    return None


def _compiled_rnnt(ctx):
    from neurobrix.core.flow.rnnt import RNNTEngine
    return RNNTEngine(ctx, _fn, _fn, _fn, _fn)


def _triton_rnnt(ctx, monkeypatch):
    import neurobrix.triton.flow.rnnt as TR
    monkeypatch.setattr(TR, "NBXTensor", _Carrier)
    monkeypatch.setattr(TR.DeviceAllocator, "set_device", staticmethod(lambda idx: None))
    monkeypatch.setattr(TR, "parse_device_idx", lambda dev: 0)
    return TR.TritonRNNTEngine(ctx, _fn, _fn, _fn, _fn)


RNNT_ENGINES = ["compiled", "triton"]


def _rnnt(engine, ctx, monkeypatch):
    return _compiled_rnnt(ctx) if engine == "compiled" else _triton_rnnt(ctx, monkeypatch)


def _length_value(t):
    return int(t.item()) if hasattr(t, "item") else int(np.asarray(t.numpy()).reshape(-1)[0])


# ---------------------------------------------------------------------------------------------
# the door: the graph's own symbol table
# ---------------------------------------------------------------------------------------------
def test_the_reader_tells_a_symbolic_axis_from_a_frozen_one():
    axes = IE.input_axes(encoder_graph())
    assert axes.name == "audio_signal" and axes.trace_shape == (1, 80, TRACE_FRAMES)
    assert axes.symbols == {0: "s0", 2: "s1"} and axes.literal == (1,)
    frozen = IE.input_axes(encoder_graph(frozen_input=True))
    assert frozen.symbols == {0: "s0"} and frozen.literal == (1, 2)
    assert IE.input_axes(encoder_graph(), "length").trace_shape == (1,)
    # the older encoding keeps the symbolic dicts inside `shape`
    legacy = {"tensors": {"input::x": {"type": "input", "shape": [
        _sym("s0", 1), {"type": "symbol", "id": "s1", "trace_value": 700}, 160]}}}
    assert IE.input_axes(legacy) == IE.InputAxes("input::x", "x", (1, 700, 160), {0: "s0", 1: "s1"}, (2,))
    assert IE.input_axes(None) is None and IE.input_axes({"tensors": {}}) is None


@pytest.mark.parametrize("frames", [101, 1101, 2001, 6001, TRACE_FRAMES])
def test_a_symbolic_axis_takes_the_fed_extent(frames):
    """Two lengths away from the trace, one far beyond it, and the trace itself."""
    b = IE.admit("toy-stt", "encoder", encoder_graph(), {"audio_signal": (1, 80, frames),
                                                        "length": (1,)})
    assert b == {"s0": 1, "s1": frames, "s2": 1}


def test_an_input_axis_the_graph_froze_is_refused_by_name():
    with pytest.raises(IE.FrozenTraceExtent) as e:
        IE.admit("toy-stt", "encoder", encoder_graph(frozen_input=True),
                 {"audio_signal": (1, 80, 1101), "length": (1,)})
    msg = str(e.value)
    for said in ("'toy-stt'", "'encoder'", "'audio_signal'", "axis 2", "3000", "1101",
                 "single-write pass"):
        assert said in msg, (said, msg)
    # at its own extent the frozen axis is admitted: a literal is refused only when fed another
    IE.admit("toy-stt", "encoder", encoder_graph(frozen_input=True),
             {"audio_signal": (1, 80, TRACE_FRAMES), "length": (1,)})


def test_a_dim_frozen_downstream_of_a_symbolic_input_is_refused_by_name():
    """parakeet's and canary's defect: the input axis carries its symbol, a mask keeps 375."""
    g = encoder_graph(frozen_downstream=True)
    with pytest.raises(IE.FrozenTraceExtent) as e:
        IE.admit("toy-stt", "encoder", g, {"audio_signal": (1, 80, 1101), "length": (1,)})
    msg = str(e.value)
    for said in ("'toy-stt'", "'encoder'", "aten.logical_and::0", "mask::out_0", "literal 375",
                 "s1 (seq_len) 3000 -> 1101", "138", "single-write pass"):
        assert said in msg, (said, msg)
    # the trace point itself is the one extent that container has witnessed: admitted
    IE.admit("toy-stt", "encoder", g, {"audio_signal": (1, 80, TRACE_FRAMES), "length": (1,)})


def test_an_expression_the_resolver_cannot_evaluate_is_refused_not_passed():
    """The door does not admit what it could not read: an elementwise op whose annotation carries a
    node the runtime's resolver has no rule for is refused, never counted and waved through."""
    g = encoder_graph()
    g["tensors"]["mask::out_0"]["symbolic_shape"]["dims"][1] = {"type": "no_such_node", "trace": 375}
    with pytest.raises(IE.FrozenTraceExtent, match="cannot evaluate"):
        IE.admit("toy-stt", "encoder", g, {"audio_signal": (1, 80, 1101), "length": (1,)})


def test_the_scan_is_the_tools_scan():
    """One implementation: the catalogue sweep (tools/symbolic_broadcast_scan.py) calls the door's
    own function."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import symbolic_broadcast_scan as S
    assert S.broadcast_breaks is IE.broadcast_breaks
    found, unknown = S.scan_graph(encoder_graph(frozen_downstream=True))
    assert [f["op"] for f in found] == ["aten.logical_and::0"] and unknown == 0
    assert found[0]["breaks_when"] == ["seq_len"]
    assert S.scan_graph(encoder_graph())[0] == []


def test_the_door_and_the_feeds_import_without_torch():
    """R33: the Triton flows, the plan and the census import the door; it brings no torch."""
    import subprocess
    import sys
    from pathlib import Path
    src = str(Path(__file__).resolve().parents[3] / "src")
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import neurobrix.core.flow.input_extent as IE\n"
         "import neurobrix.core.module.audio.feeds\n"
         "IE.admit('m', 'c', {'tensors': {'input::x': {'input_name': 'x', 'shape': [1, 4]}}}, {'x': (1, 4)})\n"
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"}, timeout=120)
    assert out.stdout.strip() == "False", (out.stdout, out.stderr[-600:])


# ---------------------------------------------------------------------------------------------
# the RNNT flow, both engines
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("engine", RNNT_ENGINES)
@pytest.mark.parametrize("seconds", [1.0, 11.0, 25.0])
def test_the_rnnt_flow_feeds_the_recordings_own_frames(tmp_path, monkeypatch, engine, seconds):
    root, topo, manifest, g = _container(tmp_path, encoder_graph())
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, seconds))
    eng = _rnnt(engine, ctx, monkeypatch)
    eng._preprocess_audio({})
    res = ctx.variable_resolver.resolved
    frames = _mel_frames(seconds)
    assert frames != TRACE_FRAMES
    assert tuple(res["global.audio_signal"].shape) == (1, 80, frames)      # never 3 000
    assert _length_value(res["global.length"]) == frames
    assert eng._stt_windows is None and eng._stt_window_mels == [frames]


@pytest.mark.parametrize("engine", RNNT_ENGINES)
def test_a_long_recording_runs_the_family_window_and_its_last_window_unpadded(
        tmp_path, monkeypatch, engine):
    """45 s: the family profile's 30 s window (3 000 frames at the extractor's hop) with its 2 s
    overlap, then the remainder at ITS OWN length (it was padded to the window)."""
    root, topo, manifest, g = _container(tmp_path, encoder_graph())
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, 45.0))
    eng = _rnnt(engine, ctx, monkeypatch)
    eng._preprocess_audio({})
    assert [tuple(f.shape) for f, _l in eng._stt_windows] == [(1, 80, 3000), (1, 80, 1701)]
    assert [_length_value(l) for _f, l in eng._stt_windows] == [3000, 1701]
    assert eng._stt_window_mels == [3000, 1701] and eng._stt_overlap_mel == 200


@pytest.mark.parametrize("engine", RNNT_ENGINES)
@pytest.mark.parametrize("frozen", ["input", "downstream"])
def test_the_rnnt_flow_refuses_a_frozen_encoder_by_name(tmp_path, monkeypatch, engine, frozen):
    g = encoder_graph(frozen_input=frozen == "input", frozen_downstream=frozen == "downstream")
    root, topo, manifest, g = _container(tmp_path, g, name="toy-frozen")
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, 11.0))
    eng = _rnnt(engine, ctx, monkeypatch)
    with pytest.raises(IE.FrozenTraceExtent) as e:
        eng._preprocess_audio({})
    msg = str(e.value)
    assert "'toy-frozen'" in msg and "'encoder'" in msg and "single-write pass" in msg
    assert ("axis 2 is frozen at its trace extent 3000" in msg) if frozen == "input" \
        else ("literal 375" in msg and "aten.logical_and::0" in msg)
    assert "global.audio_signal" not in ctx.variable_resolver.resolved      # nothing was fed


@pytest.mark.parametrize("engine", RNNT_ENGINES)
@pytest.mark.parametrize("last", [201, 202, 203, 345, 1701, 3000])
def test_the_long_form_seams_tile_the_encoder_timeline(tmp_path, monkeypatch, engine, last):
    """Every encoder frame of a long recording belongs to exactly ONE window — whatever the length
    of the last one. A decode that emits one token per frame must come back as the whole timeline,
    in order, nothing twice, nothing missing. With the seam overlap measured per window, a last
    window of 201 mel frames rounded it to 26 encoder frames against the full windows' 25: a
    30.00 s recording (3 001 frames: one window and 201 more) was refused at the merge, and 202 or
    203 lost an encoder frame."""
    total = 2800 + last                                   # one full window, then `last` frames
    root, topo, manifest, g = _container(tmp_path, encoder_graph())
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, (total - 1) * HOP / SR, "long.wav"))
    ctx.variable_resolver.resolve_all = lambda: dict(ctx.variable_resolver.resolved)
    eng = _rnnt(engine, ctx, monkeypatch)
    import neurobrix.core.flow.rnnt as CR
    import neurobrix.triton.flow.rnnt as TR
    monkeypatch.setattr(CR if engine == "compiled" else TR, "release_flow_memory", lambda dev: None)
    state = {"window": -1}

    def encoder_output(_comp):
        state["window"] += 1
        frames = _length_value(ctx.variable_resolver.resolved["global.length"])
        return SimpleNamespace(shape=(1, 5, (frames - 1) // 8 + 1))

    def one_token_per_frame(enc_output):
        t = enc_output.shape[2]
        eng._enc_frames = t
        eng._token_frames = list(range(t))
        return [350 * state["window"] + f for f in range(t)]     # the frame's place in the recording

    monkeypatch.setattr(eng, "_get_component_output", encoder_output)
    monkeypatch.setattr(eng, "_rnnt_greedy_decode", one_token_per_frame)
    monkeypatch.setattr(eng, "_decode_tokens", lambda tokens: "")
    out = eng.execute()
    n_enc = 350 + (last - 1) // 8 + 1
    assert out["global.generated_token_ids"] == list(range(n_enc))


@pytest.mark.parametrize("engine", RNNT_ENGINES)
def test_an_audio_flow_does_not_chunk_the_stage_it_just_admitted(tmp_path, monkeypatch, engine):
    """The `audio` flow's fixed-length chunking (a codec decoder run in trace-length blocks, the
    last one zero-padded) must not reach the first stage: it would pad the recording back to the
    trace one step after the door admitted it at its own length."""
    root, topo, manifest, g = _container(tmp_path, encoder_graph(), flow_type="audio")
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, 4.0))
    ctx.connections_index = {"encoder": {"audio_signal": ["global.input_features"]}}
    ctx.persistent_mode = True
    ran, chunked = [], []
    execute = lambda comp, phase, x: ran.append(comp)  # noqa: E731
    audio = topo["flow"]["audio"]
    if engine == "compiled":
        import neurobrix.core.flow.audio as A
        eng = A.AudioEngine(ctx, execute, _fn, _fn, _fn)
    else:
        import neurobrix.kernels.nbx_tensor as NT
        import neurobrix.triton.audio_frontend as AF
        import neurobrix.triton.flow.audio as A
        monkeypatch.setattr(AF, "NBXTensor", _Carrier)
        monkeypatch.setattr(A, "NBXTensor", _Carrier)
        monkeypatch.setattr(NT.DeviceAllocator, "set_device", staticmethod(lambda idx: None))
        ctx.primary_device = "cuda:0"
        eng = A.TritonAudioEngine(ctx, execute, _fn, _fn, _fn)
    eng._preprocess_audio_input(audio["input"], audio["stages"])
    assert tuple(ctx.variable_resolver.resolved["global.input_features"].shape) == (1, 80, 401)
    monkeypatch.setattr(eng, "_try_chunked_forward", lambda comp: (chunked.append(comp), True)[1])
    for name in ("_store_stage_output", "_reshape_output_for_connections"):
        if hasattr(eng, name):
            monkeypatch.setattr(eng, name, lambda comp: None)
    eng._execute_forward_stage({"component": "encoder"})
    assert ran == ["encoder"] and chunked == []          # run whole, at its own 401 frames
    eng._execute_forward_stage({"component": "codec"})   # any other stage keeps the chunking rule
    ctx.connections_index["codec"] = {"x": ["global.input_features"]}
    eng._execute_forward_stage({"component": "codec"})
    assert chunked == ["codec"] and ran == ["encoder"]


def _decoder_stubs(ctx, vocab=6, hidden=4, d_enc=5):
    rng = np.random.default_rng(0)
    f = lambda *s: torch.from_numpy(rng.standard_normal(s).astype(np.float32))  # noqa: E731
    out_bias = torch.zeros(vocab + 1)
    out_bias[vocab] = 1e4                           # every frame emits BLANK: one frame per step
    ctx.executors["decoder"] = SimpleNamespace(_weights={
        "prediction.embed.weight": f(vocab + 1, hidden),
        "dec_rnn.lstm.weight_ih_l0": f(4 * hidden, hidden), "dec_rnn.lstm.weight_hh_l0": f(4 * hidden, hidden),
        "dec_rnn.lstm.bias_ih_l0": f(4 * hidden), "dec_rnn.lstm.bias_hh_l0": f(4 * hidden)})
    ctx.executors["joint"] = SimpleNamespace(_weights={
        "enc.weight": f(3, d_enc), "enc.bias": f(3), "pred.weight": f(3, hidden),
        "pred.bias": f(3), "joint.2.weight": torch.zeros(vocab + 1, 3), "joint.2.bias": out_bias})
    ctx.pkg.defaults.update(vocab_size=vocab, blank_id=vocab, num_tdt_durations=1)
    return d_enc, hidden


@pytest.mark.parametrize("engine", RNNT_ENGINES)
@pytest.mark.parametrize("enc_frames", [4, 9])
def test_the_decode_walks_the_encoders_own_frame_axis(tmp_path, monkeypatch, engine, enc_frames):
    """The decode bound is the frame axis of what the encoder returned. It was
    `ceil(global.length / subsampling_factor)` with a literal 8 when the container carried no
    factor: a 16-frame input gave 2 decode frames whatever the encoder returned."""
    root, topo, manifest, g = _container(tmp_path, encoder_graph())
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, 1.0))
    d_enc, hidden = _decoder_stubs(ctx)
    eng = _rnnt(engine, ctx, monkeypatch)
    visited = []
    if engine == "compiled":
        ctx.variable_resolver.resolved["global.length"] = torch.tensor([16])
        joint = eng._run_joint
        monkeypatch.setattr(eng, "_run_joint",
                            lambda enc, dec, jw: (visited.append(1), joint(enc, dec, jw))[1])
        enc_out = torch.zeros(1, d_enc, enc_frames)
    else:
        ctx.variable_resolver.resolved["global.length"] = _Carrier(np.array([16]))
        ctx.executors["decoder"]._weights = {k: v.numpy() for k, v in ctx.executors["decoder"]._weights.items()}
        ctx.executors["joint"]._weights = {k: v.numpy() for k, v in ctx.executors["joint"]._weights.items()}
        joint = eng._run_joint_np
        monkeypatch.setattr(eng, "_run_joint_np",
                            lambda enc, dec, jw: (visited.append(1), joint(enc, dec, jw))[1])
        # the LSTM kernel needs a card; the bound under test does not depend on its values
        monkeypatch.setattr(eng, "_run_lstm_np", lambda x, hx, *a: (
            np.zeros((1, 1, hidden), np.float32), (hx[0], hx[1])))
        enc_out = np.zeros((1, d_enc, enc_frames), np.float32)
    assert eng._rnnt_greedy_decode(enc_out) == []
    assert len(visited) == enc_frames and eng._enc_frames == enc_frames


def test_the_rnnt_window_is_the_family_profiles_value():
    from neurobrix.core.config.loader import get_family_config
    from neurobrix.core.module.audio.stt_longform import rnnt_feed_plan
    lf = get_family_config("stt")["long_form"]
    assert rnnt_feed_plan(1101, lf, SR, HOP) == ([(0, 1101)], 0)
    assert rnnt_feed_plan(3000, lf, SR, HOP) == ([(0, 3000)], 0)
    assert rnnt_feed_plan(4501, lf, SR, HOP) == ([(0, 3000), (2800, 1701)], 200)
    # the window follows the extractor's hop: the same 30 s is 1 500 frames at a 20 ms hop
    assert rnnt_feed_plan(2000, lf, SR, 320)[0][0] == (0, 1500)
    for missing in ("rnnt_window_seconds", "rnnt_overlap_seconds"):
        with pytest.raises(RuntimeError, match=missing):
            rnnt_feed_plan(1101, {k: v for k, v in lf.items() if k != missing}, SR, HOP)


# ---------------------------------------------------------------------------------------------
# the generic audio front ends (audio_llm / encoder_decoder / audio flows), both engines
# ---------------------------------------------------------------------------------------------
def _front_end(engine, ctx, topo, monkeypatch):
    audio, stages = topo["flow"]["audio"], topo["flow"]["audio"]["stages"]
    if engine == "compiled":
        from neurobrix.core.flow.audio_utils import preprocess_audio_input
        preprocess_audio_input(ctx, audio, stages)
    else:
        import neurobrix.kernels.nbx_tensor as NT
        import neurobrix.triton.audio_frontend as AF
        monkeypatch.setattr(AF, "NBXTensor", _Carrier)
        monkeypatch.setattr(NT.DeviceAllocator, "set_device", staticmethod(lambda idx: None))
        ctx.primary_device = "cuda:0"
        AF.preprocess_audio_input_np(ctx, audio, stages)
    return ctx.variable_resolver.resolved


@pytest.mark.parametrize("engine", RNNT_ENGINES)
@pytest.mark.parametrize("seconds", [4.0, 11.0, 40.0])
def test_a_nemo_front_end_feeds_the_recordings_own_frames(tmp_path, monkeypatch, engine, seconds):
    """canary-qwen's class (audio_llm, nemo_mel): 40 s is BEYOND the 30 s of the trace — it was
    cut to 3 000 frames; 4 s and 11 s were zero-padded to them, and the length said 3 000."""
    g = encoder_graph(feat="audio_signal", length="audio_signal_length", mels=128)
    root, topo, manifest, g = _container(tmp_path, g, flow_type="audio_llm", stage="perception")
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, seconds), stage="perception")
    res = _front_end(engine, ctx, topo, monkeypatch)
    frames = _mel_frames(seconds)
    assert tuple(res["global.input_features"].shape) == (1, 128, frames)
    assert _length_value(res["global.audio_signal_length"]) == frames


@pytest.mark.parametrize("engine", RNNT_ENGINES)
def test_a_conformer_front_end_feeds_the_recordings_own_frames(tmp_path, monkeypatch, engine):
    """granite-speech's class (frame-stacked conformer features, the frame axis on dim 1)."""
    g = encoder_graph(feat="hidden_states", mels=160, layout="frames_first")
    root, topo, manifest, g = _container(tmp_path, g, flow_type="audio_llm",
                                         preprocessing="conformer", feat="hidden_states")
    for seconds in (4.0, 11.0):
        ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, seconds))
        res = _front_end(engine, ctx, topo, monkeypatch)
        stacked = _mel_frames(seconds) // 2
        assert tuple(res["global.input_features"].shape) == (1, stacked, 160)


@pytest.mark.parametrize("engine", RNNT_ENGINES)
def test_a_front_end_refuses_a_frozen_first_stage_by_name(tmp_path, monkeypatch, engine):
    g = encoder_graph(feat="audio_signal", length="audio_signal_length", mels=128,
                      frozen_downstream=True)
    root, topo, manifest, g = _container(tmp_path, g, flow_type="audio_llm", stage="perception",
                                         name="toy-frozen")
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, 11.0), stage="perception")
    with pytest.raises(IE.FrozenTraceExtent) as e:
        _front_end(engine, ctx, topo, monkeypatch)
    msg = str(e.value)
    assert "'toy-frozen'" in msg and "'perception'" in msg and "literal 375" in msg
    assert "global.input_features" not in ctx.variable_resolver.resolved


@pytest.mark.parametrize("engine", RNNT_ENGINES)
def test_the_vendors_own_window_reaches_a_frozen_axis_unrefused(tmp_path, monkeypatch, engine):
    """The whisper extractor pads to ITS OWN 30 s window (`chunk_length`, the vendor's contract):
    the front end produces 3 000 frames for a 4 s clip, the graph's literal axis takes them, and
    nothing in the flow pads. The frozen axis is refused only when fed another extent."""
    g = encoder_graph(feat="input_features", frozen_input=True, mels=80)
    root, topo, manifest, g = _container(tmp_path, g, flow_type="audio_llm",
                                         preprocessing="mel_spectrogram", feat="input_features")
    (root / "modules" / "processor").mkdir()
    (root / "modules" / "processor" / "preprocessor_config.json").write_text(json.dumps(
        {"sampling_rate": SR, "hop_length": HOP, "n_fft": 400, "chunk_length": 30,
         "feature_size": 80}))
    ctx = _ctx(root, topo, manifest, g, _wav(tmp_path, 4.0))
    res = _front_end(engine, ctx, topo, monkeypatch)
    assert tuple(res["global.input_features"].shape) == (1, 80, TRACE_FRAMES)


# ---------------------------------------------------------------------------------------------
# the plan and the census bind what the flow feeds
# ---------------------------------------------------------------------------------------------
def test_the_feeds_are_the_flows_own(tmp_path):
    from neurobrix.core.module.audio.feeds import first_audio_stage, request_feeds
    root, topo, manifest, g = _container(tmp_path, encoder_graph())
    assert first_audio_stage(topo) == "encoder"
    assert first_audio_stage({"flow": {"audio": {"input": {"modality": "text"},
                                                  "stages": [{"component": "bert"}]}}}) is None
    # the flows' own rule: an undeclared modality is read off the direction; rnnt always listens
    assert first_audio_stage({"flow": {"audio": {"direction": "stt",
                                                  "stages": [{"component": "enc"}]}}}) == "enc"
    assert first_audio_stage({"flow": {"audio": {"direction": "tts",
                                                  "stages": [{"component": "bert"}]}}}) is None
    assert first_audio_stage({"flow": {"type": "rnnt",
                                       "audio": {"stages": [{"component": "enc"}]}}}) == "enc"
    for seconds in (4.0, 11.0):
        assert request_feeds(topo, root, _wav(tmp_path, seconds), g, "toy-stt", "stt") == [
            {"audio_signal": (1, 80, _mel_frames(seconds)), "length": (1,)}]
    assert request_feeds(topo, root, _wav(tmp_path, 45.0), g, "toy-stt", "stt") == [
        {"audio_signal": (1, 80, 3000), "length": (1,)},
        {"audio_signal": (1, 80, 1701), "length": (1,)}]
    frozen = encoder_graph(frozen_downstream=True)
    with pytest.raises(IE.FrozenTraceExtent, match="literal 375"):
        request_feeds(topo, root, _wav(tmp_path, 11.0), frozen, "toy-stt", "stt")


def test_shape_only_planning_leaves_the_runs_noise_stream_alone(tmp_path):
    """The NeMo extractor's dither draws noise; the plan's shape-only pass draws from its own
    generator, so planning a request never moves the stream the run's features draw from."""
    from neurobrix.core.module.audio.feeds import request_feeds
    root, topo, manifest, g = _container(tmp_path, encoder_graph())
    wav = _wav(tmp_path, 4.0)
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    request_feeds(topo, root, wav, g, "toy-stt", "stt")
    assert np.array_equal(np.random.get_state()[1], before)


def test_prism_binds_the_encoder_to_the_recordings_frames(tmp_path):
    """The plan and the derived census read one map (`FlowBindings`): the encoder's length symbol
    is the recording's mel frames — the largest feed for a long-form run — never the trace's."""
    from neurobrix.core.prism.flow_bindings import FlowBindings
    from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig
    root, topo, manifest, g = _container(tmp_path, encoder_graph())

    def bound(seconds):
        fb = FlowBindings(topo, root, "toy-stt", audio_path=str(_wav(tmp_path, seconds)), family="stt")
        ic = InputConfig(batch_size=1, dtype="float32", flow=fb)
        return ActivationProfiler(g).build_symbol_map(ic)

    assert bound(4.0)["s1"] == _mel_frames(4.0) == 401
    assert bound(11.0)["s1"] == _mel_frames(11.0) == 1101
    assert bound(45.0)["s1"] == 3000                      # the long-form run's full window
    for m in (bound(4.0), bound(45.0)):
        assert m["s0"] == 1 and m["s2"] == 1              # the batch is not the recording's
    # a request without a recording keeps the name-driven map: the trace, a witnessed extent
    ic = InputConfig(batch_size=1, dtype="float32", flow=FlowBindings(topo, root, "toy-stt"))
    assert ActivationProfiler(g).build_symbol_map(ic)["s1"] == TRACE_FRAMES
