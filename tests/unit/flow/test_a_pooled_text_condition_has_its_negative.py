"""The unconditional pass reads the negative prompt through EVERY text condition, the pooled vector included.

Measured 2026-10-04 on Open-Sora-v2 (192x336, 51 frames, the vendor's code fed the engine's noise): unguided, the
engine's step-0 velocity equals the vendor's conditional row at cos 0.99992; guided it stood at cos 0.78. The flow
encoded the empty prompt only through the T5 (hidden states) and both CFG engines repeated the prompt's CLIP
pooled vector into the unconditional row ([pos, pos]); the vendor encodes `text + neg + neg` through T5 and CLIP
alike (opensora/utils/sampling.py, I2VDenoiser.prepare_guidance).

What each test would do if the code were wrong:
  * the rule tests fail if the pooled vector is not found on the real Open-Sora-v2 wiring, or if a shared image
    condition (Wan-I2V) or a hidden state is taken for one, or if a negative of another shape is used;
  * the flow tests run `_execute_negative_encoding` of BOTH engines' flows: a flow that does not record the
    negative pooled vector, or leaves the negative in place of the positive, fails;
  * the engine tests run the CFG engines' passes and read what the denoiser was fed: [pos, pos] (or the negative
    left in place for the conditional pass) fails;
  * the site walk fails if a flow's pre-loop gate or the Triton batched site stops reading the rule.
Seen failing on each injection: see the commit message.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neurobrix.core.runtime.resolution.negative_pooled_condition import (
    negative_port, negative_swaps, pooled_text_outputs)

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
FLOWS = ("neurobrix.core.flow.iterative_process:IterativeProcessHandler",
         "neurobrix.triton.flow.iterative_process:TritonIterativeProcessHandler")

# The Open-Sora-v2 container's own wiring (topology.json, 2026-10-04).
OPEN_SORA = {
    "flow": {"pre_loop": ["text_encoder", "text_encoder_2"],
             "loop": {"components": ["transformer"], "state_variable": "global.img"}},
    "connections": [
        {"from": "global.input_ids", "to": "text_encoder.input_ids"},
        {"from": "global.input_ids_2", "to": "text_encoder_2.input_ids"},
        {"from": "text_encoder.last_hidden_state", "to": "transformer.txt"},
        {"from": "text_encoder_2.pooler_output", "to": "transformer.y_vec"},
        {"from": "global.img", "to": "transformer.img"},
        {"from": "global.img_ids", "to": "transformer.img_ids"},
        {"from": "global.txt_ids", "to": "transformer.txt_ids"},
        {"from": "global.timesteps", "to": "transformer.timesteps"},
        {"from": "transformer.output_0", "to": "transformer.img"},
        {"from": "transformer.output_0", "to": "vae.z"},
        {"from": "global.cond", "to": "transformer.cond"},
    ]}

# The Wan2.1-I2V-14B-480P wiring: the CLIP IMAGE encoder's embedding is shared by both passes (no negative).
WAN_I2V = {
    "flow": {"pre_loop": ["text_encoder", "image_encoder"],
             "loop": {"components": ["transformer"], "state_variable": "global.latents"}},
    "connections": [
        {"from": "global.input_ids", "to": "text_encoder.input_ids"},
        {"from": "global.pixel_values", "to": "image_encoder.pixel_values"},
        {"from": "text_encoder.last_hidden_state", "to": "transformer.encoder_hidden_states"},
        {"from": "image_encoder.output_31", "to": "transformer.encoder_hidden_states_image"},
        {"from": "global.latents", "to": "transformer.hidden_states"},
    ]}


# ---------------------------------------------------------------- the rule

def test_the_clip_pooled_vector_is_a_text_condition():
    assert pooled_text_outputs(OPEN_SORA, "text_encoder_2") == ["pooler_output"]


def test_a_hidden_state_encoder_has_no_pooled_condition():
    assert pooled_text_outputs(OPEN_SORA, "text_encoder") == []


def test_an_image_condition_is_shared_not_negated():
    assert pooled_text_outputs(WAN_I2V, "image_encoder") == []
    assert pooled_text_outputs(WAN_I2V, "text_encoder") == []


def test_the_negative_is_recorded_beside_its_output():
    assert negative_port("text_encoder_2.pooler_output") == "text_encoder_2.negative_pooler_output"


def test_a_recorded_negative_is_swapped_in_and_nothing_else():
    neg, pos = torch.zeros(1, 768), torch.ones(1, 768)
    resolved = {"text_encoder_2.pooler_output": pos, "text_encoder_2.negative_pooler_output": neg,
                "global.cond": torch.ones(1, 4)}
    swaps = negative_swaps(OPEN_SORA, resolved, "transformer",
                           skip=("text_encoder.last_hidden_state", "global.img"))
    assert [k for k, _ in swaps] == ["text_encoder_2.pooler_output"] and swaps[0][1] is neg


def test_a_negative_of_another_shape_is_refused_by_name():
    resolved = {"text_encoder_2.pooler_output": torch.ones(1, 768),
                "text_encoder_2.negative_pooler_output": torch.ones(1, 1024)}
    with pytest.raises(RuntimeError) as exc:
        negative_swaps(OPEN_SORA, resolved, "transformer")
    msg = str(exc.value)
    assert "ZERO FALLBACK" in msg and "text_encoder_2.negative_pooler_output" in msg and "1024" in msg


# ---------------------------------------------------------------- both flows record the negative pooled vector

class _Resolver:
    def __init__(self, values):
        self.resolved = dict(values)
        self.loop_state = {}

    def get(self, key, default=None):
        return self.resolved.get(key, default)

    def set(self, key, value):
        self.resolved[key] = value


NEG_POOLED, POS_POOLED = torch.full((1, 768), -1.0), torch.full((1, 768), 1.0)


@pytest.mark.parametrize("module_name", FLOWS)
def test_the_flow_records_the_negative_pooled_vector_and_restores_the_prompts(monkeypatch, module_name):
    module_name, class_name = module_name.split(":")
    flow = importlib.import_module(module_name)
    handler_cls = getattr(flow, class_name)
    neg_ids, neg_mask = torch.zeros(1, 77, dtype=torch.int64), torch.ones(1, 77, dtype=torch.int64)

    class _TextProcessor:
        def __init__(self, **kwargs):
            pass

        def tokenize_negative(self, device, encoder_name="text_encoder", negative_prompt=""):
            assert encoder_name == "text_encoder_2"
            return neg_ids, neg_mask

    monkeypatch.setattr("neurobrix.core.module.text.processor.TextProcessor", _TextProcessor)
    if hasattr(flow, "_to_nbx"):
        monkeypatch.setattr(flow, "_to_nbx", lambda t, *a, **k: t)
    resolver = _Resolver({"global.input_ids_2": "POS_IDS", "global.attention_mask_2": "POS_MASK",
                          "text_encoder_2.pooler_output": POS_POOLED})

    def _encode(name, phase, arg):
        assert resolver.get("global.input_ids_2") is neg_ids, "the encoder ran on the prompt, not the negative"
        resolver.set("text_encoder_2.pooler_output", NEG_POOLED)

    handler = handler_cls.__new__(handler_cls)
    handler.ctx = SimpleNamespace(variable_resolver=resolver, modules={"tokenizer_2": object()},
                                  primary_device="cpu", pkg=SimpleNamespace(defaults={}, topology=OPEN_SORA),
                                  executors={})
    handler._tokenizer_config_with_flags = lambda encoder, tokenizer: {}
    handler._execute_component = _encode
    handler._execute_negative_encoding("text_encoder_2", ["pooler_output"], hidden=False)
    assert resolver.get("text_encoder_2.negative_pooler_output") is NEG_POOLED
    assert resolver.get("text_encoder_2.pooler_output") is POS_POOLED, "the prompt's pooled vector was not restored"
    assert resolver.get("global.input_ids_2") == "POS_IDS" and resolver.get("global.attention_mask_2") == "POS_MASK"
    assert "text_encoder_2.negative_hidden_state" not in resolver.resolved


def _gate_reads_the_rule(path: Path) -> bool:
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "pooled_text_outputs"
               for n in ast.walk(ast.parse(path.read_text())))


@pytest.mark.parametrize("rel", ("core/flow/iterative_process.py", "triton/flow/iterative_process.py"))
def test_each_flows_pre_loop_gate_reads_the_rule(rel):
    assert _gate_reads_the_rule(SRC / rel), f"{rel}: the pre-loop gate no longer asks for pooled conditions"


# ---------------------------------------------------------------- the CFG engines feed [neg, pos]

def _cfg_ctx():
    resolver = _Resolver({
        "text_encoder.last_hidden_state": torch.ones(1, 5, 8),
        "text_encoder.negative_hidden_state": torch.zeros(1, 5, 8),
        "text_encoder_2.pooler_output": POS_POOLED,
        "text_encoder_2.negative_pooler_output": NEG_POOLED,
        "global.attention_mask": torch.ones(1, 5, dtype=torch.int64),
        "text_encoder.negative_attention_mask": torch.ones(1, 5, dtype=torch.int64),
        "global.img_ids": torch.zeros(1, 6, 3), "global.txt_ids": torch.zeros(1, 5, 3),
        "global.cond": torch.zeros(1, 6, 4),
    })
    return SimpleNamespace(variable_resolver=resolver, loop_id="loop", strategy=None,
                           pkg=SimpleNamespace(topology=OPEN_SORA, defaults={}))


def _engine(module, ctx, fed):
    cls = module.CFGEngine if hasattr(module, "CFGEngine") else module.TritonCFGEngine

    def _run(name, phase):
        fed.append((phase, ctx.variable_resolver.get("text_encoder_2.pooler_output")))
        return {"output_0": torch.zeros(ctx.variable_resolver.get("global.img").shape)}

    return cls(ctx, _run, lambda name, out: out["output_0"], guidance_scale=7.0)


def test_the_batched_pass_feeds_the_negative_pooled_vector_to_the_unconditional_row():
    from neurobrix.core.cfg import engine as core_engine
    ctx, fed = _cfg_ctx(), []
    _engine(core_engine, ctx, fed)._execute_batched_cfg(
        "transformer", torch.zeros(1, 6, 64), torch.tensor(0.5), 7.0, torch.float32)
    (_, y_vec), = fed
    assert torch.equal(y_vec[0], NEG_POOLED[0]), "the unconditional row read the prompt's pooled vector"
    assert torch.equal(y_vec[1], POS_POOLED[0])
    assert ctx.variable_resolver.get("text_encoder_2.pooler_output") is POS_POOLED


class _NbxLike(torch.Tensor):
    """A host tensor answering the two NBXTensor attributes the Triton sequential pass reads."""
    @property
    def _dtype(self):
        return self.dtype

    @property
    def _device(self):
        return self.device


@pytest.mark.parametrize("module_name", ("neurobrix.core.cfg.engine", "neurobrix.triton.cfg.engine"))
def test_the_sequential_passes_feed_the_negative_then_the_prompt(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    if hasattr(module, "_ensure_nbx"):                 # the Triton engine wraps at its boundary; stand-ins pass through
        monkeypatch.setattr(module, "_ensure_nbx",
                            lambda t, *a, **k: t.as_subclass(_NbxLike) if isinstance(t, torch.Tensor) else t)
    ctx, fed = _cfg_ctx(), []
    _engine(module, ctx, fed)._execute_sequential_cfg("transformer", torch.zeros(1, 6, 64), torch.tensor(0.5), 7.0)
    assert [p for p, _ in fed] == ["cfg_uncond", "cfg_cond"]
    assert fed[0][1] is NEG_POOLED, "the unconditional pass read the prompt's pooled vector"
    assert fed[1][1] is POS_POOLED, "the conditional pass read the negative"


def test_the_triton_batched_site_takes_the_negative_for_the_unconditional_row():
    """The Triton batched forward needs device tensors; its one site is read instead (R30 mirror of the core test)."""
    tree = ast.parse((SRC / "triton/cfg/engine.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_execute_batched_cfg")
    cats = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "cat"
            and any(isinstance(m, ast.Name) and m.id == "_negatives" for m in ast.walk(n))]
    assert len(cats) == 1, "the extra-input batch no longer reads the recorded negatives"
    first_row = cats[0].args[0].elts[0]
    assert any(isinstance(m, ast.Name) and m.id == "_negatives" for m in ast.walk(first_row)), \
        "the negative is not in the unconditional (first) row"
