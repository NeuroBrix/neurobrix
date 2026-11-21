"""The speech leg receives the hidden rows of the whole tts span, on the PyTorch and the Triton branch.

A staged vlm flow decodes on the decoder cache: the first forward covers the whole context (the prefill), every
later one the single new token. The speech leg conditions on the hidden row of every position of its tts span, so
the flow keeps the rows of each forward as it happens (the prefill's, then one per cached step) and hands the leg
the concatenation `[1, S, H]` in position order. Before, it was handed the last forward's output alone, `[1, 1, H]`,
and refused: "the final decode forward covers 1 positions but the tts span ends at 34 — hidden rows unavailable".

The decode loop of the flow runs for real, with the real step protocol (`DecodeSteps`) over a fake cache; the LM
is a fake whose row for position p is the number p, so the rows the leg receives say which positions they cover.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

PROMPT_IDS = [5, 6, 7, 8]
TTS_BOS, TTS_EOS = 100, 101
GENERATED = [21, 22, 23, TTS_EOS]      # the tts-eos ends the decode: it is sampled, never fed
HIDDEN = 3


class _Cache:   # the decoder cache the step protocol drives
    def use_branch(self, _b):
        pass

    def update_position_offset(self):
        pass

    def set_decode_mode(self, actual_seq_len=None):
        pass


class _Tokenizer:
    pass


def _ctx(tmp_path):
    topology = {"flow": {"type": "vlm",
                         "vlm": {"lm_component": "lm"},
                         "speech": {"condition_type": "hidden_text_merge", "tts_bos_token_id": TTS_BOS,
                                    "tts_eos_token_id": TTS_EOS}}}
    tensors = {"input::position_ids": {"shape": ["B", "S"]}}
    (tmp_path / "components" / "lm").mkdir(parents=True)
    (tmp_path / "components" / "lm" / "graph.json").write_text(json.dumps({"tensors": tensors}))
    lm = types.SimpleNamespace(_dag={"tensors": tensors})
    return types.SimpleNamespace(
        pkg=types.SimpleNamespace(topology=topology, defaults={"max_tokens": 16, "eos_token_id": [0, TTS_EOS]},
                                  cache_path=str(tmp_path)),
        variable_resolver=types.SimpleNamespace(resolved={"global.prompt": "hi", "global.mode": _mode()}, resolve_all=lambda: {}),
        modules={}, executors={"lm": lm}, persistent_mode=True, primary_device="cpu")


def _mode():
    from neurobrix.core.flow.base import SPEECH_MODE
    return SPEECH_MODE


def _lm_forward(engine, to_numpy, make):
    """The LM fake: rows of the fed positions, each row filled with its absolute position."""
    resolved = engine.ctx.variable_resolver.resolved

    def _execute_component(name, *_a, **_k):
        if name != "lm":
            return
        pos = to_numpy(resolved["position_ids"]).reshape(-1)
        rows = np.broadcast_to(pos.reshape(1, -1, 1).astype(np.float32), (1, len(pos), HIDDEN)).copy()
        resolved["lm.output_0"] = make(rows)
    engine._execute_component = _execute_component
    engine._get_component_output = lambda name: resolved.get(f"{name}.output_0")
    engine._compute_logits = lambda *a, **k: None
    engine._ensure_weights_loaded = lambda name: None
    engine._get_embed_weight = lambda name: object()
    engine._tokenize_with_placeholder_run = lambda *a, **k: (list(PROMPT_IDS), 0, 0)


def _sampled(monkeypatch, module):
    ids = iter(GENERATED)
    monkeypatch.setattr(module, "sample_next_id", lambda sampler, logits, generated: next(ids))


def _check(received):
    assert len(received) == 1
    rows = received[0]["lm_hidden"]
    rows = rows.numpy() if hasattr(rows, "numpy") else rows.data
    n_ids = len(PROMPT_IDS) + 1 + len(GENERATED)           # + the tts-bos the audio-mode prompt ends with
    span_end = n_ids - 1                                   # the tts-eos position: the span stops before it
    assert rows.shape == (1, span_end, HIDDEN)             # a row for every position the LM has run
    assert rows[0, :, 0].tolist() == list(range(span_end))  # ... in position order, none twice
    assert received[0]["full_ids"] == PROMPT_IDS + [TTS_BOS] + GENERATED


def test_the_pytorch_branch_hands_the_speech_leg_the_rows_of_every_position(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import neurobrix.core.flow.decoder_cache as decoder_cache
    import neurobrix.core.flow.speech_cfm as speech_cfm
    import neurobrix.core.flow.vlm as cvlm
    import neurobrix.core.module.autoregressive.samplers as samplers
    import neurobrix.core.module.sampling_contract as sampling_contract

    received = []
    monkeypatch.setattr(speech_cfm, "SpeechCFMLeg", lambda engine: types.SimpleNamespace(run=received.append))
    monkeypatch.setattr(sampling_contract, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(sampling_contract, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(samplers, "CombinedSampler", lambda cfg: None)
    _sampled(monkeypatch, samplers)
    monkeypatch.setattr(cvlm, "_effective_sampling", lambda ctx, defaults: ({}, set()))
    monkeypatch.setattr(cvlm, "gather_rows", lambda w, ids, device, dtype: torch.zeros(1, len(ids[0]), HIDDEN))
    monkeypatch.setattr(decoder_cache, "open_decoder_cache", lambda *a: _Cache())

    engine = object.__new__(cvlm.VLMEngine)
    engine.ctx = _ctx(tmp_path)
    engine._get_compute_dtype = lambda: torch.float32
    _lm_forward(engine, lambda t: t.numpy(), torch.from_numpy)
    engine._execute_staged_splice(engine.ctx.pkg.topology["flow"]["vlm"])
    _check(received)


class _T:   # the slice of NBXTensor the staged decode loop touches, over numpy
    def __init__(self, a):
        self.data = np.asarray(a)

    shape = property(lambda self: self.data.shape)

    def to(self, *_a):
        return self

    def narrow(self, dim, start, length):
        return _T(np.take(self.data, range(start, start + length), axis=dim))

    def numpy(self):
        return self.data

    @staticmethod
    def from_numpy(a):
        return _T(a)

    @staticmethod
    def cat(tensors, dim=0):
        return _T(np.concatenate([t.data for t in tensors], axis=dim))


def test_the_triton_branch_hands_the_speech_leg_the_rows_of_every_position(tmp_path, monkeypatch):
    import neurobrix.triton.flow.decoder_cache as decoder_cache
    import neurobrix.triton.flow.speech_cfm as speech_cfm
    import neurobrix.triton.flow.vlm as tvlm

    received = []
    monkeypatch.setattr(speech_cfm, "SpeechCFMLeg", lambda engine: types.SimpleNamespace(run=received.append))
    monkeypatch.setattr(tvlm, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(tvlm, "CombinedSampler", lambda cfg: None)
    monkeypatch.setattr(tvlm, "_effective_sampling", lambda ctx, defaults: ({}, set()))
    monkeypatch.setattr(tvlm, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(tvlm, "NBXTensor", _T)
    _sampled(monkeypatch, tvlm)
    monkeypatch.setattr(decoder_cache, "open_decoder_cache", lambda *a: _Cache())

    engine = object.__new__(tvlm.TritonVLMEngine)
    engine.ctx = _ctx(tmp_path)
    engine._compute_dtype = lambda: "float32"
    engine._embed_ids = lambda ids, weight, dtype: _T(np.zeros((1, len(ids), HIDDEN), dtype=np.float32))
    _lm_forward(engine, lambda t: t.data, _T)
    engine._execute_staged_splice(engine.ctx.pkg.topology["flow"]["vlm"])
    _check(received)
