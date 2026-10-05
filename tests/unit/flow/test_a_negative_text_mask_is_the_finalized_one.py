"""The unconditional branch's text mask is the mask of the embedding it masks.

Measured 2026-10-04 on SANA-Video_2B_720p_diffusers (256x640, 4 steps, the vendor's SanaVideoPipeline in fp32
fed the engine's noise): the engine's batched text mask held 310 ones over [unconditional, conditional], the
vendor's 11. Both flows finalized the negative embedding (the instruction-prefix slice, 519 -> 300 positions)
and stored the TOKENIZER's mask (519) beside it; every CFG site then met a mask of another length than its
embedding and substituted ones, so the unconditional branch attended 299 padding positions. First divergent
op: block 0's cross-attention; step-0 guided output rel 0.557 from the vendor (cos 0.8825), 1.98e-4 after.

What each test would do if the code were wrong:
  * the flow tests run `_execute_negative_encoding` of BOTH engines' flows over stand-ins with the real
    finalizer: a flow that stores the tokenizer's mask leaves 519 (or 226) positions and fails the length;
  * the rule tests fail if a mask of another length is returned (or replaced) instead of refused;
  * the site walk fails if any CFG site reads the negative mask without passing through the rule,
    or if a second read path appears beside the batched one the plan's guidance split reuses.
Seen failing on each injection: see the commit message.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neurobrix.core.components.handlers.text_encoder_handler import TextEncoderComponentHandler
from neurobrix.core.runtime.resolution.negative_text_mask import finalized_mask, negative_mask_for

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
FLOWS = ("neurobrix.core.flow.iterative_process:IterativeProcessHandler",
         "neurobrix.triton.flow.iterative_process:TritonIterativeProcessHandler")
CFG_ENGINES = ("core/cfg/engine.py", "triton/cfg/engine.py")


# ---------------------------------------------------------------- the rule

def test_the_finalizers_mask_replaces_the_tokenizers():
    raw, cut = torch.ones(1, 519), torch.ones(1, 300)
    assert finalized_mask(raw, {"hidden_state": object(), "attention_mask": cut}) is cut


def test_a_finalizer_that_returns_no_mask_keeps_the_tokenizers():
    raw = torch.ones(1, 77)
    assert finalized_mask(raw, {"hidden_state": object()}) is raw
    assert finalized_mask(raw, None) is raw


def test_a_negative_mask_of_its_embeddings_length_passes_untouched():
    mask, hidden = torch.tensor([[1, 0, 0]]), torch.zeros(1, 3, 4)
    assert negative_mask_for(mask, hidden, "text_encoder") is mask


def test_no_recorded_negative_mask_is_none():
    assert negative_mask_for(None, torch.zeros(1, 3, 4), "text_encoder") is None


def test_a_negative_mask_of_another_length_is_refused_by_name():
    with pytest.raises(RuntimeError) as exc:
        negative_mask_for(torch.ones(1, 519), torch.zeros(1, 300, 4), "text_encoder_2")
    msg = str(exc.value)
    assert "ZERO FALLBACK" in msg and "text_encoder_2.negative_attention_mask" in msg
    assert "519" in msg and "300" in msg


# ---------------------------------------------------------------- both flows store the finalized mask

class _Resolver:
    def __init__(self, values):
        self.resolved = dict(values)

    def get(self, key, default=None):
        return self.resolved.get(key, default)

    def set(self, key, value):
        self.resolved[key] = value


def _run_negative_encoding(monkeypatch, module_name, tokenizer_cfg, encoded_len, attended):
    """`_execute_negative_encoding` of one engine's flow, over stand-ins and the REAL finalizer."""
    module_name, class_name = module_name.split(":")
    flow = importlib.import_module(module_name)
    handler_cls = getattr(flow, class_name)
    ids = torch.zeros(1, encoded_len, dtype=torch.int64)
    mask = torch.zeros(1, encoded_len, dtype=torch.int64)
    mask[:, :attended] = 1

    class _TextProcessor:
        def __init__(self, **kwargs):
            pass

        def tokenize_negative(self, device, encoder_name="text_encoder", negative_prompt=""):
            return ids, mask

    monkeypatch.setattr("neurobrix.core.module.text.processor.TextProcessor", _TextProcessor)
    if hasattr(flow, "_to_nbx"):                       # the Triton flow wraps at its boundary; the stand-ins pass through
        monkeypatch.setattr(flow, "_to_nbx", lambda t, *a, **k: t)

    finalizer = SimpleNamespace(
        finalize_embeddings=lambda **kw: TextEncoderComponentHandler.finalize_embeddings(None, **kw))
    resolver = _Resolver({"global.input_ids": "POS_IDS", "global.attention_mask": "POS_MASK",
                          "text_encoder.last_hidden_state": "POS_HIDDEN"})
    handler = handler_cls.__new__(handler_cls)
    handler.ctx = SimpleNamespace(
        variable_resolver=resolver, modules={"tokenizer": object()}, primary_device="cpu",
        pkg=SimpleNamespace(defaults={}, topology={}),
        executors={"text_encoder": SimpleNamespace(_component_handler=finalizer)})
    handler._tokenizer_config_with_flags = lambda encoder, tokenizer: dict(tokenizer_cfg)
    handler._execute_component = lambda name, phase, arg: resolver.set(
        "text_encoder.last_hidden_state", torch.randn(1, encoded_len, 8))
    handler._execute_negative_encoding("text_encoder")
    return resolver


@pytest.mark.parametrize("module_name", FLOWS)
def test_a_sliced_negative_embedding_carries_its_sliced_mask(monkeypatch, module_name):
    """The instruction-prefix slice (Sana): 519 encoded positions -> [BOS] + the last 299."""
    cfg = {"complex_human_instruction": ["an instruction"], "max_sequence_length": 300}
    r = _run_negative_encoding(monkeypatch, module_name, cfg, encoded_len=519, attended=1)
    hidden, mask = r.get("text_encoder.negative_hidden_state"), r.get("text_encoder.negative_attention_mask")
    assert tuple(hidden.shape) == (1, 300, 8)
    assert tuple(mask.shape) == (1, 300), f"the stored mask has {mask.shape[-1]} positions, its embedding 300"
    assert int(mask.sum()) == 1 and int(mask[0, 0]) == 1          # the vendor's unconditional mask: BOS alone
    assert negative_mask_for(mask, hidden, "text_encoder") is mask


@pytest.mark.parametrize("module_name", FLOWS)
def test_a_padded_negative_embedding_carries_its_padded_mask(monkeypatch, module_name):
    """The zero re-pad (T5 / UMT5): 226 encoded positions -> the model's 512, the mask padded with zeros."""
    cfg = {"zero_pad_embeddings": True, "max_sequence_length": 512}
    r = _run_negative_encoding(monkeypatch, module_name, cfg, encoded_len=226, attended=3)
    hidden, mask = r.get("text_encoder.negative_hidden_state"), r.get("text_encoder.negative_attention_mask")
    assert tuple(hidden.shape) == (1, 512, 8) and tuple(mask.shape) == (1, 512)
    assert int(mask.sum()) == 3


@pytest.mark.parametrize("module_name", FLOWS)
def test_an_unfinalized_negative_embedding_keeps_the_tokenizers_mask(monkeypatch, module_name):
    r = _run_negative_encoding(monkeypatch, module_name, {}, encoded_len=77, attended=5)
    mask = r.get("text_encoder.negative_attention_mask")
    assert tuple(mask.shape) == (1, 77) and int(mask.sum()) == 5


@pytest.mark.parametrize("module_name", FLOWS)
def test_the_positive_inputs_are_restored(monkeypatch, module_name):
    cfg = {"complex_human_instruction": ["an instruction"], "max_sequence_length": 300}
    r = _run_negative_encoding(monkeypatch, module_name, cfg, encoded_len=519, attended=1)
    assert r.get("global.input_ids") == "POS_IDS" and r.get("global.attention_mask") == "POS_MASK"
    assert r.get("text_encoder.last_hidden_state") == "POS_HIDDEN"


# ---------------------------------------------------------------- every CFG site goes through the rule

def _negative_mask_reads(tree):
    """(node, guarded) for every `<resolver>.get(f"...negative_attention_mask", ...)` in the module."""
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        if "negative_attention_mask" not in ast.unparse(node.args[0] if node.args else node):
            continue
        guarded, cur = False, node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.Call) and getattr(cur.func, "id", None) == "_negative_mask_for":
                guarded = True
                break
            if isinstance(cur, ast.stmt):
                break
        out.append((node, guarded))
    return out


@pytest.mark.parametrize("rel", CFG_ENGINES)
def test_every_cfg_site_reads_the_negative_mask_through_the_rule(rel):
    reads = _negative_mask_reads(ast.parse((SRC / rel).read_text()))
    # One site: the batched pass. The sequential CFG path is gone; a plan that cannot hold both branches splits the
    # batched inputs into halves, and each half is fed through this same read.
    assert len(reads) == 1, f"{rel}: expected the batched site only, found {len(reads)}"
    unguarded = [node.lineno for node, guarded in reads if not guarded]
    assert not unguarded, f"{rel}: the negative mask is read outside negative_mask_for at line(s) {unguarded}"


def test_the_site_walk_is_seen_failing_on_a_bare_read():
    bare = ast.parse(
        "def f(self, encoder_comp, neg_hidden):\n"
        "    neg_mask = self._ctx.variable_resolver.get(f'{encoder_comp}.negative_attention_mask', None)\n"
        "    if neg_mask is None or neg_mask.shape[-1] != neg_hidden.shape[1]:\n"
        "        neg_mask = 1\n")
    assert [guarded for _, guarded in _negative_mask_reads(bare)] == [False]
