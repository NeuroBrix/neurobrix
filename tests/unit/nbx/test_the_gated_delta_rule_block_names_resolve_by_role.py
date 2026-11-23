"""A linear-attention block's gated-delta-rule weights resolve, each to its role, by token name.

The nine weight names such a block carries (`linear_attn.{A_log, conv1d, dt_bias, in_proj_qkv,
in_proj_z, in_proj_a, in_proj_b, norm, out_proj}`) are translated by the vocabulary alone: no
model name, no model type, no layer count. Without the vocabulary the strict parser refuses all
of them (432 keys of a 48-layer recording). An invented token still refuses by name, and every
translated name is a fixed point of the parser. The block and its short convolution take roles
5.0 already names: `token_mixer` (the sublayer that mixes the tokens) and `depthwise_conv` (one
filter per channel, weight [C, 1, taps]); no second name for a role the vocabulary has.
"""
from __future__ import annotations

import pytest

from neurobrix.pytorch.neurotax import NeuroTaxParser, SynonymRegistry, first_non_canonical

ROLES = {
    "A_log": "decay_log",
    "conv1d.weight": "depthwise_conv.weight",
    "dt_bias": "step_bias",
    "in_proj_qkv.weight": "qkv.weight",
    "in_proj_z.weight": "out_gate_proj.weight",
    "in_proj_a.weight": "decay_gate_proj.weight",
    "in_proj_b.weight": "beta_proj.weight",
    "norm.weight": "norm.weight",
    "out_proj.weight": "out.weight",
}


@pytest.mark.parametrize("leaf", sorted(ROLES))
@pytest.mark.parametrize("block", [0, 47])
def test_each_gated_delta_name_resolves_to_its_role(leaf, block):
    name = f"model.layers.{block}.linear_attn.{leaf}"
    canon = f"model.block.{block}.token_mixer.{ROLES[leaf]}"
    assert NeuroTaxParser().normalize_strict(name) == canon
    assert NeuroTaxParser().normalize(name) == canon


def test_the_translated_names_are_fixed_points():
    names = [f"model.block.{i}.token_mixer.{v}" for i in range(3) for v in ROLES.values()]
    assert first_non_canonical(names) is None


def test_an_invented_token_in_the_block_still_refuses_by_name():
    with pytest.raises(ValueError, match="in_proj_w"):
        NeuroTaxParser().normalize_strict("model.layers.3.linear_attn.in_proj_w.weight")
    with pytest.raises(ValueError, match="B_log"):
        SynonymRegistry.resolve_strict("B_log", "model.layers.3.linear_attn.B_log")


def test_the_block_and_its_convolution_take_the_roles_the_vocabulary_already_names():
    assert SynonymRegistry.resolve("linear_attn") == SynonymRegistry.resolve("mixer") == "token_mixer"
    assert SynonymRegistry.resolve("conv1d") == SynonymRegistry.resolve("dwconv") == "depthwise_conv"
    assert {"lin_attn", "short_conv"}.isdisjoint(SynonymRegistry.canonical_tokens())
