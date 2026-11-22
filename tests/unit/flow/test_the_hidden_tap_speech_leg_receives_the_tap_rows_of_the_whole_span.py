"""The hidden_tap speech leg receives the tap rows of the whole span, on the PyTorch and the Triton branch.

A vlm flow decodes on the decoder cache: the first forward covers the whole context (the prefill), every later one
the single new token. The speech leg of the tap class reads the LM's `hidden_tap` output as the rows `[1, S0+n-1]` of
the whole context (it slices the rows of the user turns from it), so the flow keeps the tap rows of each forward as it
happens (the prefill's, then one per cached step) and hands the leg their concatenation in position order. Before, it
read the final forward's tap alone, `[1, 1, H]`, which covers one position under the cache.

The decode loop of the flow runs for real, with the real step protocol (`DecodeSteps`) over a fake cache; the LM is a
fake whose tap row for position p is the number 1000 + p (its ordinary output is p, so a leg handed the wrong tensor
shows), so the rows the leg receives say which positions they cover. Both decodes are run: on the cache, and
without it (every forward covers the whole context, so each one's rows REPLACE what was kept).
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

PROMPT_IDS = [5, 6, 7, 8, 9]
GENERATED = [21, 22, 23, 0]      # the eos ends the decode: it is sampled, never fed
HIDDEN = 3
TAP = 1000


class _Cache:   # the decoder cache the step protocol drives
    def use_branch(self, _b):
        pass

    def update_position_offset(self):
        pass

    def set_decode_mode(self, actual_seq_len=None):
        pass


class _Tokenizer:
    pass


def _ctx(tmp_path, with_tap=True):
    topology = {"flow": {"type": "vlm",
                         "vlm": {"vision_component": "vision", "lm_component": "lm", "image_token_id": 9,
                                 "spatial_merge_size": 2},
                         "speech": {"condition_type": "hidden_tap"}}}
    tensors = {"input::inputs_embeds": {"shape": ["B", "S", HIDDEN]}, "input::position_ids": {"shape": ["B", "S"]}}
    (tmp_path / "components" / "lm").mkdir(parents=True)
    (tmp_path / "components" / "lm" / "graph.json").write_text(json.dumps({"tensors": tensors}))
    lm = types.SimpleNamespace(_dag={"tensors": tensors})
    return types.SimpleNamespace(
        pkg=types.SimpleNamespace(topology=topology, defaults={"max_tokens": 16, "eos_token_id": 0},
                                  cache_path=str(tmp_path)),
        variable_resolver=types.SimpleNamespace(
            resolved={"global.prompt": "hi", "global.mode": _mode()}, resolve_all=lambda: {}),
        modules={}, executors={"lm": lm}, persistent_mode=True, primary_device="cpu")


def _mode():
    from neurobrix.core.flow.base import SPEECH_MODE
    return SPEECH_MODE


def _lm_forward(engine, to_numpy, make, with_tap=True):
    """The LM fake: the rows of the fed positions, each filled with its absolute position (output) and 1000 + it (tap)."""
    resolved = engine.ctx.variable_resolver.resolved

    def _execute_component(name, *_a, **_k):
        if name != "lm":
            return
        pos = to_numpy(resolved["position_ids"]).reshape(-1)
        rows = np.broadcast_to(pos.reshape(1, -1, 1).astype(np.float32), (1, len(pos), HIDDEN)).copy()
        resolved["lm.output_0"] = make(rows)
        if with_tap:
            resolved["lm.hidden_tap"] = make(rows + TAP)
    engine._execute_component = _execute_component
    engine._get_component_output = lambda name: resolved.get(f"{name}.output_0")
    engine._compute_logits = lambda *a, **k: None
    engine._ensure_weights_loaded = lambda name: None
    engine._get_embed_weight = lambda name: object()
    engine._tokenize_text_only = lambda prompt: list(PROMPT_IDS)


def _sampled(monkeypatch, module):
    ids = iter(GENERATED)
    monkeypatch.setattr(module, "sample_next_id", lambda sampler, logits, generated: next(ids))


def _check(received):
    assert len(received) == 1
    tap = received[0]["hidden_tap"]
    tap = tap.numpy() if hasattr(tap, "numpy") else tap.data
    n_rows = len(PROMPT_IDS) + len(GENERATED) - 1          # the eos is sampled, never fed: S0 + n - 1 positions
    assert tap.shape == (1, n_rows, HIDDEN)                # a tap row for every position the LM has run
    assert tap[0, :, 0].tolist() == [TAP + p for p in range(n_rows)]   # ... in position order, none twice
    assert received[0]["generated_ids"] == GENERATED


CACHE_OR_NOT = pytest.mark.parametrize("cached", [True, False], ids=["on_the_cache", "recomputing_the_whole_context"])


@CACHE_OR_NOT
def test_the_pytorch_branch_hands_the_speech_leg_the_tap_rows_of_every_position(cached, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import neurobrix.core.flow.decoder_cache as decoder_cache
    import neurobrix.core.flow.speech as speech
    import neurobrix.core.flow.vlm as cvlm
    import neurobrix.core.module.autoregressive.samplers as samplers
    import neurobrix.core.module.sampling_contract as sampling_contract

    received = []
    monkeypatch.setattr(speech, "SpeechLeg", lambda engine: types.SimpleNamespace(run=received.append))
    monkeypatch.setattr(sampling_contract, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(sampling_contract, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(samplers, "CombinedSampler", lambda cfg: None)
    _sampled(monkeypatch, samplers)
    monkeypatch.setattr(cvlm, "_effective_sampling", lambda ctx, defaults: ({}, set()))
    monkeypatch.setattr(cvlm, "gather_rows", lambda w, ids, device, dtype: torch.zeros(1, len(ids[0]), HIDDEN))
    monkeypatch.setattr(decoder_cache, "open_decoder_cache", lambda *a: _Cache() if cached else None)

    engine = object.__new__(cvlm.VLMEngine)
    engine.ctx = _ctx(tmp_path)
    engine._get_compute_dtype = lambda: torch.float32
    _lm_forward(engine, lambda t: t.numpy(), torch.from_numpy)
    engine.execute()
    _check(received)


class _T:   # the slice of NBXTensor the vlm decode loop touches, over numpy
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


@CACHE_OR_NOT
def test_the_triton_branch_hands_the_speech_leg_the_tap_rows_of_every_position(cached, tmp_path, monkeypatch):
    import neurobrix.triton.flow.decoder_cache as decoder_cache
    import neurobrix.triton.flow.speech as speech
    import neurobrix.triton.flow.vlm as tvlm

    received = []
    monkeypatch.setattr(speech, "SpeechLeg", lambda engine: types.SimpleNamespace(run=received.append))
    monkeypatch.setattr(tvlm, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(tvlm, "CombinedSampler", lambda cfg: None)
    monkeypatch.setattr(tvlm, "_effective_sampling", lambda ctx, defaults: ({}, set()))
    monkeypatch.setattr(tvlm, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(tvlm, "NBXTensor", _T)
    _sampled(monkeypatch, tvlm)
    monkeypatch.setattr(decoder_cache, "open_decoder_cache", lambda *a: _Cache() if cached else None)

    engine = object.__new__(tvlm.TritonVLMEngine)
    engine.ctx = _ctx(tmp_path)
    engine._compute_dtype = lambda: "float32"
    engine._embed_ids = lambda ids, weight, dtype: _T(np.zeros((1, len(ids), HIDDEN), dtype=np.float32))
    _lm_forward(engine, lambda t: t.data, _T)
    engine.execute()
    _check(received)


def test_a_graph_without_a_hidden_tap_is_refused_by_name(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import neurobrix.core.flow.decoder_cache as decoder_cache
    import neurobrix.core.flow.vlm as cvlm
    import neurobrix.core.module.autoregressive.samplers as samplers
    import neurobrix.core.module.sampling_contract as sampling_contract

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
    _lm_forward(engine, lambda t: t.numpy(), torch.from_numpy, with_tap=False)
    with pytest.raises(RuntimeError, match="hidden_tap"):
        engine.execute()
