"""The vendor tokens NeuroTax 5.0's complete vocabulary keeps as canonical role names pass the collision and fixed-point gates.

Nineteen tokens of the catalogue already spell their role the way NeuroTax would (`attn_mask`,
`logit_scale`, `kv`, `pre_norm`, ...): the registry maps each to itself. The supervisor accepted
them (2026-10-04 01:57) PROVIDED each passes the gates, proven here:

* fixed point — the parser returns the token unchanged, strict and permissive, and with an index
  glued to it;
* collision — the registry keys that land on the token are exactly the spellings of that one role
  (pinned below): a later entry that maps a token of another meaning onto it goes red here. Over
  the 49 containers the rename makes no two keys of a component collide (the census gate of
  2026-10-04, `forge/tools/neurotax_rename.py` dry run: 0 collisions, 0 refused).

What these would do if one of them were remapped (e.g. `kv` -> `key`) or another meaning were
mapped onto one (e.g. `pooler` -> `pre_norm`): the fixed-point or the synonym cell fails.
"""
from __future__ import annotations

import pytest

from neurobrix.nbx.neurotax import SynonymRegistry as R

# token -> every registry key that lands on it (its own spelling and its vendors' synonyms)
KEPT = {
    "alpha": {"alpha"},
    "attn_mask": {"attn_mask"},
    "causal_mask": {"causal_mask"},
    "cond_proj": {"cond_proj"},
    "conv_block": {"conv_block", "conv_blocks"},
    "fast_freqs_cis": {"fast_freqs_cis"},
    "fast_norm": {"fast_norm"},
    "kv": {"kv", "kv_proj", "to_kv"},
    "logit_scale": {"logit_scale"},
    "mean": {"mean"},
    "pos_bias_u": {"pos_bias_u"},
    "pos_bias_v": {"pos_bias_v"},
    "post_norm": {"post_norm", "post_layernorm", "ln_post"},
    "pre_norm": {"pre_norm", "pre_layrnorm"},
    "ref_pos_embed": {"ref_pos_embed"},
    "speech_head": {"speech_head"},
    "text_head": {"text_head"},
    "vision_head": {"vision_head"},
    "window": {"window"},
}


def test_nineteen():
    assert len(KEPT) == 19


@pytest.mark.parametrize("token", sorted(KEPT))
def test_a_kept_token_is_a_fixed_point(token):
    assert R._REGISTRY[token] == token
    assert R.resolve(token) == token and R.resolve_strict(token, token) == token
    for glued in (f"{token}7", f"{token}_7"):
        assert R.resolve(glued) == glued and R.resolve_strict(glued, glued) == glued


@pytest.mark.parametrize("token", sorted(KEPT))
def test_a_kept_token_names_one_role(token):
    assert {k for k, v in R._REGISTRY.items() if v == token} == KEPT[token]
