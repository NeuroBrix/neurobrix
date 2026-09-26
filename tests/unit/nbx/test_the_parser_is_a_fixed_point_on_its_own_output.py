"""The NeuroTax parser is a fixed point on its own output, and strict agrees with permissive.

The key of a tensor is the parser's normalized form (the neurotaxe, 2026-09-26). A key the
parser rewrites when it reads it again is not a key: a tool that re-applies the parser moves
it. Measured over the cache on 2026-09-26: 63 841 keys rewritten `gate` -> `router` (the
registry mapped `gate_proj` -> `gate` for the SwiGLU gate and `gate` -> `router` for the MoE
router) and 5 164 `self_attn` -> `attn` (`attn1` -> `self_attn`, `self_attn` -> `attn`).

The parser translates token by token, so parse(parse(k)) == parse(k) for every k exactly when
every canonical token maps to itself. That is what is pinned here, over the whole registry —
no container needed. Strict mode is pinned to agree with permissive on every token it
accepts, and to accept every token the parser emits (it refused 65 of them, so it could not
validate a container).

What these would do on the old parser: the first fails naming `gate` and `self_attn`, the
second fails on the 65 canonical tokens, the third on `mlp`, `ln`, `lm_head`.
"""
from __future__ import annotations

import re

from neurobrix.nbx.neurotax import SynonymRegistry, normalize_tensor_name, normalize_tensor_name_strict


def test_every_canonical_token_maps_to_itself():
    moved = {v: SynonymRegistry.resolve(v) for v in SynonymRegistry.canonical_tokens()
             if SynonymRegistry.resolve(v) != v}
    assert moved == {}, f"canonical tokens the parser rewrites on a second pass: {moved}"


def test_strict_accepts_every_token_the_parser_emits():
    refused = []
    for v in sorted(SynonymRegistry.canonical_tokens()):
        try:
            SynonymRegistry.resolve_strict(v, v)
        except ValueError:
            refused.append(v)
    assert refused == []


def test_strict_and_permissive_give_one_key():
    tokens = set(SynonymRegistry._REGISTRY) | {
        p.strip("^$") for p in SynonymRegistry.PRESERVE_PATTERNS if re.fullmatch(r"\^[a-z0-9_]+\$", p)}
    split = {t: (SynonymRegistry.resolve(t), SynonymRegistry.resolve_strict(t, t))
             for t in sorted(tokens) if SynonymRegistry.resolve(t) != SynonymRegistry.resolve_strict(t, t)}
    assert split == {}, f"token -> (permissive, strict): {split}"


def test_the_two_collisions_are_told_apart():
    # The SwiGLU gate and the MoE router are two functions, two canonical tokens.
    swiglu = normalize_tensor_name("model.layers.3.mlp.gate_proj.weight")
    router = normalize_tensor_name("model.layers.3.mlp.gate.weight")
    assert swiglu != router
    assert normalize_tensor_name(swiglu) == swiglu and normalize_tensor_name(router) == router
    assert normalize_tensor_name_strict(swiglu) == swiglu and normalize_tensor_name_strict(router) == router
    # diffusers' attn1 and an LLM's self_attn are one function, one token, stable on a second pass.
    a = normalize_tensor_name("transformer_blocks.0.attn1.to_q.weight")
    b = normalize_tensor_name("model.layers.0.self_attn.q_proj.weight")
    assert a.split(".")[2] == b.split(".")[3] == "attn"
    assert normalize_tensor_name(a) == a and normalize_tensor_name(b) == b
