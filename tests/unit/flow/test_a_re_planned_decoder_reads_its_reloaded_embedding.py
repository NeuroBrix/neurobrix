"""After a device out-of-memory re-plan (`RuntimeExecutor._replan_after_oom`: every component unloaded, the loaded ones
loaded again under the new plan, the failed call run again from the same frame), the audio_llm and encoder_decoder
decoders read the token embedding of the RELOADED table, never the one they read before the call: that local pointed
into an arena the re-plan freed (the alias hazard reading of 2026-11-24). A re-plan is known to the flow by its plan:
the re-plan gives the flow's ctx the new one (`self._flow_ctx.plan = self.plan`). Both branches. No card."""
from types import SimpleNamespace

import numpy as np
import pytest


class T:
    """A host stand-in for an NBXTensor: shape, narrow, cast, cat — what the Triton decode loops touch."""

    def __init__(self, shape=(1, 1, 4)):
        self.shape = shape

    def to(self, *a, **k):
        """A cast keeps the stand-in as it is."""
        return self

    def narrow(self, dim, start, length):
        """A slice of the context: one position wide."""
        return T((1, length, 4))

    @classmethod
    def cat(cls, parts, dim=1):
        """A concatenation along the sequence: the positions add up."""
        return T((1, sum(p.shape[1] for p in parts), 4))

    @classmethod
    def from_numpy(cls, a):
        """An id or position array, wrapped."""
        return T(tuple(np.asarray(a).shape))


class Run:
    """The LM executor and its re-plan: the first call re-plans (the plan replaced, the table loaded again)."""

    def __init__(self, ctx, name, key, old, new, output):
        self.ctx, self.name, self.key, self.new, self.output = ctx, name, key, new, output
        ctx.executors[name] = SimpleNamespace(_weights={key: old})
        self.calls, self.read = 0, []

    def execute(self, comp, phase, t):
        """The component call: the first one ran out of memory, re-planned and ran again under the new plan."""
        assert comp == self.name
        self.calls += 1
        if self.calls == 1:
            self.ctx.plan = SimpleNamespace(strategy="re-planned")
            self.ctx.executors[self.name]._weights = {self.key: self.new}
        self.ctx.variable_resolver.resolved[f"{self.name}.output_0"] = self.output


def _ctx(stages, defaults):
    """A flow ctx with the stages and defaults given, persistent so the flow keeps its components."""
    return SimpleNamespace(
        pkg=SimpleNamespace(topology={"flow": {"audio": {"stages": stages}}}, defaults=defaults),
        plan=SimpleNamespace(strategy="first"), executors={}, modules={}, persistent_mode=True,
        primary_device="cpu", variable_resolver=SimpleNamespace(resolved={}, resolve_all=lambda: {}))


def _sampling(monkeypatch, samplers, tokens):
    """The sampler stubbed to the token list given; the sampling rules accepted as is."""
    import neurobrix.core.module.sampling_contract as rules
    monkeypatch.setattr(rules, "sampler_config", lambda *a, **k: {})
    monkeypatch.setattr(rules, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(samplers, "CombinedSampler", lambda cfg: None)
    monkeypatch.setattr(samplers, "sample_next_id", lambda s, logits, ids: tokens[len(ids)])


AUDIO_STAGES = [{"component": "proj", "execution": "forward"},
                {"component": "lm", "execution": "autoregressive", "logits_source": "embed_weight_tied"}]
AUDIO_DEFAULTS = {"max_tokens": 4, "eos_token_id": 0, "stt_prefix_ids": [], "stt_suffix_ids": []}


def test_the_triton_audio_llm_decoder_reads_the_reloaded_embedding(monkeypatch):
    import neurobrix.triton.flow.audio_llm as flow
    import neurobrix.triton.audio_frontend as frontend
    import neurobrix.triton.samplers as samplers
    import neurobrix.triton.flow.decoder_cache as cache
    import neurobrix.triton.flow.decode_topk as topk
    monkeypatch.setattr(frontend, "preprocess_audio_input_np", lambda *a: None)
    monkeypatch.setattr(frontend, "postprocess_text_output_np", lambda *a: None)
    monkeypatch.setattr(cache, "open_decoder_cache", lambda *a: None)
    monkeypatch.setattr(topk, "log_topk", lambda *a: None)
    monkeypatch.setattr(flow, "NBXTensor", T)
    monkeypatch.setattr(flow, "_effective_sampling", lambda ctx, d: ({}, set()))
    _sampling(monkeypatch, samplers, [1, 0])
    ctx = _ctx(AUDIO_STAGES, AUDIO_DEFAULTS)
    ctx.variable_resolver.resolved["proj.output_0"] = T((1, 2, 4))
    old, new = SimpleNamespace(name="old", ndim=2), SimpleNamespace(name="new", ndim=2)
    run = Run(ctx, "lm", "model.embed_tokens.weight", old, new, T())
    engine = flow.TritonAudioLLMEngine(ctx, run.execute, None, lambda n: None, lambda n: None)
    engine._compute_dtype = lambda: None
    engine._get_component_output = lambda name: ctx.variable_resolver.resolved.get(f"{name}.output_0")
    engine._compute_logits = lambda out, w, src: (run.read.append(w.name), "logits")[1]
    engine._embed_ids = lambda ids, w, dtype: (run.read.append(w.name), T())[1]
    engine.execute()
    assert run.calls == 2 and run.read == ["new", "new", "new"], run.read


def test_the_compiled_audio_llm_decoder_reads_the_reloaded_embedding(monkeypatch):
    import torch
    import neurobrix.core.flow.audio_llm as flow
    import neurobrix.core.flow.audio_utils as utils
    import neurobrix.core.module.autoregressive.samplers as samplers
    import neurobrix.core.flow.decoder_cache as cache
    import neurobrix.core.flow.decode_topk as topk
    monkeypatch.setattr(utils, "preprocess_audio_input", lambda *a: None)
    monkeypatch.setattr(utils, "postprocess_text_output", lambda *a: None)
    monkeypatch.setattr(utils, "get_compute_dtype", lambda ctx: torch.float32)
    monkeypatch.setattr(cache, "open_decoder_cache", lambda *a: None)
    monkeypatch.setattr(topk, "log_topk", lambda *a: None)
    monkeypatch.setattr(flow, "_effective_sampling", lambda ctx, d: ({}, set()))
    _sampling(monkeypatch, samplers, [1, 0])
    ctx = _ctx(AUDIO_STAGES, AUDIO_DEFAULTS)
    ctx.variable_resolver.resolved["proj.output_0"] = torch.zeros(1, 2, 4)
    old, new = torch.zeros(8, 4), torch.ones(8, 4)
    names = {id(old): "old", id(new): "new"}
    run = Run(ctx, "lm", "model.embed_tokens.weight", old, new, torch.zeros(1, 1, 4))
    monkeypatch.setattr(flow, "gather_rows", lambda w, ids, device, dtype: (
        run.read.append(names[id(w)]), torch.zeros(1, len(ids[0]), 4))[1])
    engine = flow.AudioLLMEngine(ctx, run.execute, None, lambda n: None, lambda n: None)
    engine._get_component_output = lambda name: ctx.variable_resolver.resolved.get(f"{name}.output_0")
    engine._compute_logits = lambda out, w, src: (run.read.append(names[id(w)]), "logits")[1]
    engine.execute()
    assert run.calls == 2 and run.read == ["new", "new", "new"], run.read


@pytest.mark.parametrize("branch", ["triton", "compiled"])
def test_the_encoder_decoder_decoder_reads_the_reloaded_embedding(branch, monkeypatch):
    import neurobrix.core.module.audio.stt_longform as longform
    import neurobrix.core.module.sampling_contract as rules
    monkeypatch.setattr(rules, "enforce_sampling_support", lambda *a, **k: None)
    monkeypatch.setattr(longform, "whisper_forced_map", lambda *a, **k: {})
    monkeypatch.setattr(longform, "whisper_begin_index", lambda m: 1)
    tokens = [None, 1, 0]                 # after the start token: one token, then the end
    defaults = {"max_tokens": 4, "temperature": 0.0, "eos_token_id": 0, "decoder_start_token_id": 5}
    stage = {"component": "dec", "execution": "autoregressive", "logits_source": "embed_weight_tied"}
    ctx = _ctx([stage], defaults)
    old, new = SimpleNamespace(name="old", shape=(8, 4)), SimpleNamespace(name="new", shape=(8, 4))
    if branch == "triton":
        import neurobrix.triton.flow.encoder_decoder as flow
        run = Run(ctx, "dec", "embed", old, new, T())
        monkeypatch.setattr(flow, "NBXTensor", T)
        monkeypatch.setattr(flow, "_nbx_device", SimpleNamespace(select=lambda i: None))
        monkeypatch.setattr(flow, "_get_embed_weight", lambda c, n: c.executors[n]._weights["embed"])
        monkeypatch.setattr(flow, "_get_component_output", lambda c, n: c.variable_resolver.resolved[f"{n}.output_0"])
        monkeypatch.setattr(flow, "_compute_logits", lambda c, out, w, src: (run.read.append(w.name), "logits")[1])
        monkeypatch.setattr(flow, "_sample_token_nbx", lambda logits, t, generated_ids, **k: tokens[len(generated_ids)])
        engine = flow.TritonEncoderDecoderEngine(ctx, run.execute, None, lambda n: None, lambda n: None)
        engine._decoder_kv_interceptor = lambda name, n: None
    else:
        import torch
        import neurobrix.core.flow.encoder_decoder as flow
        run = Run(ctx, "dec", "embed", old, new, torch.zeros(1, 1, 4))
        engine = flow.EncoderDecoderEngine(ctx, run.execute, None, lambda n: None, lambda n: None)
        engine._decoder_kv_wrapper = lambda name, n: None
        engine._get_embed_weight = lambda n: ctx.executors[n]._weights["embed"]
        engine._get_component_output = lambda n: ctx.variable_resolver.resolved[f"{n}.output_0"]
        engine._compute_logits = lambda out, w, src: (run.read.append(w.name), "logits")[1]
        engine._sample_token = lambda logits, t, generated_ids, **k: tokens[len(generated_ids)]
    monkeypatch.setattr(flow, "_effective_sampling", lambda c, d: ({}, set()))
    engine._decode_one_window(stage, "dec", defaults, [], [])
    assert run.calls == 2 and run.read == ["new", "new"], run.read
