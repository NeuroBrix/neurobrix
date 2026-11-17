"""A text request walks a vlm container's flow to its language model, on the PyTorch and the Triton branch.

A request with no image, video or audio is the text-only form: no modality tower runs, the prompt is
chat-templated alone (no span placeholder is looked up), and the language model is handed the positions its
graph declares: rank 2 `[1, S]` one index per token, rank 3 `[3, 1, S]` the three M-RoPE planes, which for a
text-only context are three equal rows 0..S-1. The build below declares no `audio_token_id` and no audio tower,
as a vision-language container without speech does: the span-token read of the audio branch must never run for
a text request (the PyTorch branch keeps it inside its audio branch; the Triton mirror read it after the
modality branches, so a text request stopped there).

The whole flow from the request's modality dispatch to the first language-model call runs for real; only what
needs weights, a tokenizer file or a card is a stub: the tower and LM calls, the embedding rows, the sampler.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

PROMPT = "Write one sentence about rivers."
PROMPT_IDS = [11, 12, 13, 14, 15]


class _ReachedTheLanguageModel(Exception):
    pass


class _Tokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        self.messages.append(messages)
        return list(PROMPT_IDS)


class _Steps:  # the decoder cache's feed: the prefill hands the whole context
    one_token = False

    def __init__(self, *_a):
        pass

    def feed(self, full, _one):
        return full

    def before_run(self):
        pass


def _lm_tensors(rank):
    return {"input::inputs_embeds": {"shape": ["B", "S", 8]},
            "input::position_ids": {"shape": ["B", "S"] if rank == 2 else [3, "B", "S"]}}


def _ctx(tmp_path, rank, executors):
    lm_dir = tmp_path / "components" / "lm"
    lm_dir.mkdir(parents=True)
    (lm_dir / "graph.json").write_text(json.dumps({"tensors": _lm_tensors(rank)}))
    topology = {"flow": {"type": "vlm", "vlm": {
        "vision_component": "vision", "lm_component": "lm", "image_token_id": 9, "spatial_merge_size": 2}}}
    return types.SimpleNamespace(
        pkg=types.SimpleNamespace(topology=topology, defaults={"max_tokens": 4, "eos_token_id": 0},
                                  cache_path=str(tmp_path)),
        variable_resolver=types.SimpleNamespace(resolved={"global.prompt": PROMPT}),
        modules={"tokenizer": _Tokenizer()}, executors=executors, persistent_mode=False, primary_device="cpu")


def _walk(engine, calls):
    def _execute_component(name, *_a, **_k):
        calls.append(("run", name))
        if name == "lm":
            raise _ReachedTheLanguageModel
    engine._ensure_weights_loaded = lambda name: calls.append(("load", name))
    engine._execute_component = _execute_component
    engine._tokenize_around_span = lambda *a, **k: calls.append(("span", a)) or ([], [])
    engine._get_embed_weight = lambda name: object()
    with pytest.raises(_ReachedTheLanguageModel):
        engine.execute()
    return engine.ctx.variable_resolver.resolved["global.position_ids"]


def _expected(rank):
    row = np.arange(len(PROMPT_IDS), dtype=np.int64)
    return row.reshape(1, -1) if rank == 2 else np.broadcast_to(row.reshape(1, 1, -1), (3, 1, len(row)))


def _assert_text_only(calls, tokenizer, positions, rank):
    assert calls == [("load", "lm"), ("run", "lm")]          # no tower loaded or run, no span looked up
    assert tokenizer.messages == [[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]]
    assert positions.shape == _expected(rank).shape
    assert np.array_equal(positions, _expected(rank))


@pytest.mark.parametrize("rank", [2, 3])
def test_the_triton_branch_walks_a_text_request_to_its_language_model(rank, tmp_path, monkeypatch):
    import neurobrix.triton.flow.decoder_cache as decoder_cache
    import neurobrix.triton.flow.vlm as tvlm

    class _Embeds:
        def __init__(self, n):
            self.shape = (1, n, 8)

        def narrow(self, _dim, _start, _length):
            return self

    monkeypatch.setattr(tvlm, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(tvlm, "CombinedSampler", lambda cfg: None)
    monkeypatch.setattr(tvlm, "_effective_sampling", lambda ctx, defaults: ({}, set()))
    monkeypatch.setattr(tvlm, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(tvlm, "NBXTensor", types.SimpleNamespace(from_numpy=lambda a: a))
    monkeypatch.setattr(decoder_cache, "DecodeSteps", _Steps)
    monkeypatch.setattr(decoder_cache, "open_decoder_cache", lambda *a: None)

    engine = object.__new__(tvlm.TritonVLMEngine)
    engine.ctx = _ctx(tmp_path, rank, executors={})
    engine._compute_dtype = lambda: "float32"
    engine._embed_ids = lambda ids, weight, dtype: _Embeds(len(ids))
    calls = []
    positions = _walk(engine, calls)
    _assert_text_only(calls, engine.ctx.modules["tokenizer"], np.asarray(positions), rank)


@pytest.mark.parametrize("rank", [2, 3])
def test_the_pytorch_branch_walks_a_text_request_to_its_language_model(rank, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import neurobrix.core.flow.decoder_cache as decoder_cache
    import neurobrix.core.flow.vlm as cvlm
    import neurobrix.core.module.autoregressive.samplers as samplers
    import neurobrix.core.module.sampling_contract as sampling_contract

    monkeypatch.setattr(sampling_contract, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(sampling_contract, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(samplers, "CombinedSampler", lambda cfg: None)
    monkeypatch.setattr(cvlm, "_effective_sampling", lambda ctx, defaults: ({}, set()))
    monkeypatch.setattr(cvlm, "gather_rows", lambda w, ids, device, dtype: torch.zeros(1, len(ids[0]), 8))
    monkeypatch.setattr(decoder_cache, "DecodeSteps", _Steps)
    monkeypatch.setattr(decoder_cache, "open_decoder_cache", lambda *a: None)

    lm = types.SimpleNamespace(_dag={"tensors": _lm_tensors(rank)})
    engine = object.__new__(cvlm.VLMEngine)
    engine.ctx = _ctx(tmp_path, rank, executors={"lm": lm})
    engine._get_compute_dtype = lambda: torch.float32
    engine._get_tokenizer = lambda: engine.ctx.modules["tokenizer"]
    calls = []
    positions = _walk(engine, calls)
    _assert_text_only(calls, engine.ctx.modules["tokenizer"], positions.numpy(), rank)
