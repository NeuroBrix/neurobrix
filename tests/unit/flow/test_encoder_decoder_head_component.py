"""Encoder-decoder logits computation when logits_source == 'head' with head_component (e.g. proj_out),
plus regression coverage for 'tied', 'embed_weight_tied', 'lm_head', and 'self' across both branches."""
from types import SimpleNamespace

import numpy as np
import pytest


class _HostNBXTensor:
    """Host stand-in for NBXTensor supporting select, unsqueeze, contiguous, to, and numpy."""

    def __init__(self, arr, device_idx: int = 0):
        self._arr = np.asarray(arr)
        self.shape = self._arr.shape
        self.dtype = self._arr.dtype
        self._dtype = self._arr.dtype
        self._device_idx = device_idx
        self.ndim = self._arr.ndim

    def select(self, dim: int, idx: int):
        return _HostNBXTensor(np.take(self._arr, idx, axis=dim), device_idx=self._device_idx)

    def unsqueeze(self, dim: int):
        return _HostNBXTensor(np.expand_dims(self._arr, axis=dim), device_idx=self._device_idx)

    def is_contiguous(self) -> bool:
        return self._arr.flags["C_CONTIGUOUS"]

    def contiguous(self):
        return _HostNBXTensor(np.ascontiguousarray(self._arr), device_idx=self._device_idx)

    def to(self, *args, **kwargs):
        return _HostNBXTensor(self._arr.astype(np.float32), device_idx=self._device_idx)

    def numpy(self):
        return self._arr

    @classmethod
    def from_numpy(cls, arr):
        return cls(arr)


def _make_ctx(stage, extra_flow=None, persistent=True):
    flow_cfg = {"audio": {"stages": [
        {"component": "model.encoder", "execution": "forward"},
        stage,
    ]}}
    if extra_flow:
        flow_cfg.update(extra_flow)
    defaults = {
        "max_tokens": 3,
        "temperature": 0.0,
        "eos_token_id": 50257,
        "decoder_start_token_id": 50258,
        "suppress_tokens": [1350],
        "begin_suppress_tokens": [],
    }
    return SimpleNamespace(
        pkg=SimpleNamespace(topology={"flow": flow_cfg}, defaults=defaults),
        plan=SimpleNamespace(strategy="first"),
        executors={},
        modules={},
        persistent_mode=persistent,
        primary_device="cpu",
        variable_resolver=SimpleNamespace(resolved={}, resolve_all=lambda: {}),
    )


def test_pytorch_encoder_decoder_runs_head_component_proj_out(monkeypatch):
    import torch
    import neurobrix.core.flow.encoder_decoder as flow
    import neurobrix.core.module.sampling_contract as rules

    monkeypatch.setattr(rules, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(rules, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(flow, "_effective_sampling", lambda ctx, d: ({}, set()))

    hidden_dim, vocab_size = 1280, 51866
    stage = {
        "component": "model.decoder",
        "execution": "autoregressive",
        "cross_attention_from": "model.encoder",
        "logits_source": "head",
        "head_component": "proj_out",
    }
    ctx = _make_ctx(stage)
    loaded, head_calls = [], []

    class _HeadExec:
        _dag = {"input_tensor_ids": ["input::input"], "tensors": {"input::input": {}}}

        def run(self, inputs):
            x = inputs["input"]
            head_calls.append(tuple(x.shape))
            logits = torch.full((1, 1, vocab_size), -10.0, dtype=torch.float32)
            # Pick token 1350 (suppressed -> must be ignored) and token 4242 (expected)
            logits[0, 0, 1350] = 99.0
            logits[0, 0, 4242 if len(head_calls) == 1 else 50257] = 50.0
            return {"aten._unsafe_view::0::out_0": logits}

    ctx.executors["model.decoder"] = SimpleNamespace(_weights={})
    ctx.executors["proj_out"] = _HeadExec()

    def execute_fn(comp, phase, t):
        assert comp == "model.decoder"
        # Decoder outputs hidden states [1, S, 1280], NOT vocab logits
        ctx.variable_resolver.resolved["model.decoder.output_0"] = torch.zeros((1, 2, hidden_dim))

    engine = flow.EncoderDecoderEngine(
        ctx, execute_fn, None, lambda n: loaded.append(n), lambda n: None,
    )
    monkeypatch.setattr(engine, "_decoder_kv_wrapper", lambda *a: None)
    all_ids, all_texts = [], []
    engine._decode_one_window(stage, "model.decoder", ctx.pkg.defaults, all_ids, all_texts)

    assert "proj_out" in loaded
    assert head_calls == [(1, 1, hidden_dim), (1, 1, hidden_dim)]
    assert all_ids == [50258, 4242, 50257]


def test_triton_encoder_decoder_runs_head_component_proj_out_and_transfers_device(monkeypatch):
    import neurobrix.triton.flow.encoder_decoder as flow
    import neurobrix.core.module.sampling_contract as rules

    monkeypatch.setattr(rules, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(rules, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(flow, "_effective_sampling", lambda ctx, d: ({}, set()))
    monkeypatch.setattr(flow, "NBXTensor", _HostNBXTensor)
    monkeypatch.setattr(flow._nbx_device, "select", lambda idx: None)
    transfers = []
    monkeypatch.setattr(
        flow, "transfer_tensor",
        lambda t, dev: (transfers.append(dev), _HostNBXTensor(t.numpy(), device_idx=dev))[1],
    )

    hidden_dim, vocab_size = 1280, 51866
    stage = {
        "component": "model.decoder",
        "execution": "autoregressive",
        "cross_attention_from": "model.encoder",
        "logits_source": "head",
        "head_component": "proj_out",
    }
    ctx = _make_ctx(stage)
    ctx.primary_device = "cuda:0"
    loaded, head_calls = [], []

    class _HeadExec:
        device = "cuda:1"
        _dag = {"input_tensor_ids": ["input::input"], "tensors": {"input::input": {}}}

        def run(self, inputs):
            x = inputs["input"]
            assert x._device_idx == 1
            head_calls.append(tuple(x.shape))
            arr = np.full((1, 1, vocab_size), -10.0, dtype=np.float32)
            arr[0, 0, 1350] = 99.0
            arr[0, 0, 4242 if len(head_calls) == 1 else 50257] = 50.0
            return {"aten._unsafe_view::0::out_0": _HostNBXTensor(arr, device_idx=1)}

    ctx.executors["model.decoder"] = SimpleNamespace(_weights={})
    ctx.executors["proj_out"] = _HeadExec()

    def execute_fn(comp, phase, t):
        assert comp == "model.decoder"
        ctx.variable_resolver.resolved["model.decoder.output_0"] = _HostNBXTensor(
            np.zeros((1, 2, hidden_dim), dtype=np.float16), device_idx=0
        )

    engine = flow.TritonEncoderDecoderEngine(
        ctx, execute_fn, None, lambda n: loaded.append(n), lambda n: None,
    )
    monkeypatch.setattr(engine, "_decoder_kv_interceptor", lambda *a: None)
    all_ids, all_texts = [], []
    engine._decode_one_window(stage, "model.decoder", ctx.pkg.defaults, all_ids, all_texts)

    assert "proj_out" in loaded
    assert transfers == [1, 1]
    assert head_calls == [(1, 1, hidden_dim), (1, 1, hidden_dim)]
    assert all_ids == [50258, 4242, 50257]


def test_execute_unloads_head_component_when_non_persistent(monkeypatch):
    import torch
    import neurobrix.core.flow.encoder_decoder as core_flow
    import neurobrix.triton.flow.encoder_decoder as triton_flow
    import neurobrix.core.flow.audio_utils as audio_utils
    import neurobrix.triton.audio_frontend as audio_frontend

    monkeypatch.setattr(audio_utils, "preprocess_audio_input", lambda *a: None)
    monkeypatch.setattr(audio_utils, "postprocess_text_output", lambda *a: None)
    monkeypatch.setattr(core_flow, "release_flow_memory", lambda *a: None)
    monkeypatch.setattr(audio_frontend, "preprocess_audio_input_np", lambda *a: None)
    monkeypatch.setattr(audio_frontend, "postprocess_text_output_np", lambda *a: None)
    monkeypatch.setattr(triton_flow, "release_flow_memory", lambda *a: None)

    stage = {
        "component": "model.decoder",
        "execution": "autoregressive",
        "logits_source": "head",
        "head_component": "proj_out",
    }

    for is_triton in (False, True):
        ctx = _make_ctx(stage, persistent=False)
        ctx.variable_resolver.resolved["global.input_features"] = np.zeros((1, 128, 3000), dtype=np.float32)
        ctx.executors = {"model.encoder": SimpleNamespace(), "model.decoder": SimpleNamespace(), "proj_out": SimpleNamespace()}
        unloaded = []
        if is_triton:
            eng = triton_flow.TritonEncoderDecoderEngine(
                ctx, lambda *a: None, None, lambda n: None, lambda n: unloaded.append(n)
            )
            monkeypatch.setattr(eng, "_get_component_output", lambda *a: None, raising=False)
        else:
            eng = core_flow.EncoderDecoderEngine(
                ctx, lambda *a: None, None, lambda n: None, lambda n: unloaded.append(n)
            )
            ctx.variable_resolver.resolved["model.encoder.output_0"] = torch.zeros((1, 10, 1280))
        monkeypatch.setattr(eng, "_decode_one_window", lambda *a, **k: None)
        eng.execute()
        assert unloaded == ["model.encoder", "model.decoder", "proj_out"]


def test_logits_source_head_refuses_missing_head_or_executor():
    import torch
    import neurobrix.core.flow.encoder_decoder as core_flow
    import neurobrix.triton.flow.encoder_decoder as triton_flow

    ctx = _make_ctx({"component": "model.decoder", "execution": "autoregressive", "logits_source": "head"})
    core_eng = core_flow.EncoderDecoderEngine(ctx, lambda *a: None, None, lambda n: None, lambda n: None)

    with pytest.raises(RuntimeError, match="ZERO FALLBACK.*head_component"):
        core_eng._compute_logits(torch.zeros((1, 1, 8)), None, "head", head_name=None)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK.*head_component"):
        triton_flow._compute_logits(_make_ctx({}), _HostNBXTensor(np.zeros((1, 1, 8))), None, "head", head_name=None)

    with pytest.raises(RuntimeError, match="ZERO FALLBACK.*missing from executors"):
        core_eng._compute_logits(torch.zeros((1, 1, 8)), None, "head", head_name="proj_out")
    with pytest.raises(RuntimeError, match="ZERO FALLBACK.*missing from executors"):
        triton_flow._compute_logits(ctx, _HostNBXTensor(np.zeros((1, 1, 8))), None, "head", head_name="proj_out")


def test_head_input_name_and_stage_section_resolution():
    from neurobrix.core.flow.shared_steps import head_input_name, resolve_logits_head

    flow_stage = {"audio": {"head_component": "sec_head", "stages": []}}
    assert resolve_logits_head(flow_stage, {"logits_source": "head"}) == "sec_head"
    assert resolve_logits_head(flow_stage, {"logits_source": "head", "head_component": "proj_out"}) == "proj_out"
    assert resolve_logits_head({}, {"logits_source": "lm_head"}) == "lm_head"
    assert resolve_logits_head({}, {"logits_source": "tied"}) is None
    assert resolve_logits_head({}, {"logits_source": "self"}) is None

    exec_custom = SimpleNamespace(_dag={"input_tensor_ids": ["input::hidden_states"], "tensors": {}})
    assert head_input_name(exec_custom) == "hidden_states"
    assert head_input_name(SimpleNamespace()) == "input"


def test_tied_and_self_logits_sources_preserved(monkeypatch):
    import torch
    import neurobrix.core.flow.encoder_decoder as core_flow
    import neurobrix.triton.flow.encoder_decoder as triton_flow
    import nbxtensor.ops.dispatch as nbx_dispatch

    ctx = _make_ctx({"component": "model.decoder", "execution": "autoregressive"})
    core_eng = core_flow.EncoderDecoderEngine(ctx, lambda *a: None, None, lambda n: None, lambda n: None)

    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    embed = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    for src in ("tied", "embed_weight_tied"):
        out = core_eng._compute_logits(hidden, embed, src)
        assert torch.allclose(out, torch.tensor([[[3.0, 8.0, 7.0]]]))
    assert torch.allclose(core_eng._compute_logits(hidden, embed, "self"), hidden[:, -1:, :])

    mm_calls = []
    monkeypatch.setattr(
        nbx_dispatch, "dispatch",
        lambda op: (lambda a, b: (mm_calls.append(op), _HostNBXTensor(np.matmul(a.numpy(), b.numpy())))[1]),
    )
    h_nbx = _HostNBXTensor(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    e_nbx = _HostNBXTensor(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32))
    # Add reshape / transpose on _HostNBXTensor for the triton tied path
    _HostNBXTensor.reshape = lambda self, *s: _HostNBXTensor(self._arr.reshape(*s), device_idx=self._device_idx)
    _HostNBXTensor.transpose = lambda self, d0, d1: _HostNBXTensor(np.swapaxes(self._arr, d0, d1), device_idx=self._device_idx)
    for src in ("tied", "embed_weight_tied"):
        out_t = triton_flow._compute_logits(ctx, h_nbx, e_nbx, src)
        assert np.allclose(out_t.numpy(), np.array([[[3.0, 8.0, 7.0]]]))
    assert mm_calls == ["mm", "mm"]
    assert np.allclose(triton_flow._compute_logits(ctx, h_nbx, e_nbx, "self").numpy(), np.array([[[3.0, 4.0]]]))
