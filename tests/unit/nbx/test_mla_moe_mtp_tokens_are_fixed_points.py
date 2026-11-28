"""MLA Q-latent, MoE router bias, and MTP structural tokens are accepted as fixed points by both NeuroTax twins."""
from __future__ import annotations

import pytest

from neurobrix.pytorch import neurotax as pytorch_neurotax
from neurobrix.triton import neurotax as triton_neurotax

MLA_MOE_MTP_TOKENS = (
    "q_a_proj",
    "q_a_layernorm",
    "q_b_proj",
    "e_score_correction_bias",
    "eh_proj",
    "enorm",
    "hnorm",
    "shared_head",
)

MLA_MOE_MTP_KEYS = (
    "model.block.0.attn.q_a_proj.weight",
    "model.block.0.attn.q_a_layernorm.weight",
    "model.block.0.attn.q_b_proj.weight",
    "model.block.0.ffn.router.e_score_correction_bias",
    "model.block.61.eh_proj.weight",
    "model.block.61.enorm.weight",
    "model.block.61.hnorm.weight",
    "model.block.61.shared_head.head.weight",
    "model.block.61.shared_head.norm.weight",
)

TWINS = (
    pytest.param(pytorch_neurotax, id="pytorch"),
    pytest.param(triton_neurotax, id="triton"),
)


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("token", MLA_MOE_MTP_TOKENS)
def test_mla_moe_mtp_tokens_are_self_mapped_fixed_points(twin, token):
    reg = twin.SynonymRegistry
    assert reg._REGISTRY[token] == token
    assert reg.resolve(token) == token
    assert reg.resolve_strict(token, token) == token
    for glued in (f"{token}7", f"{token}_7"):
        assert reg.resolve(glued) == glued
        assert reg.resolve_strict(glued, glued) == glued


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("key", MLA_MOE_MTP_KEYS)
def test_mla_moe_mtp_keys_are_accepted_and_fixed_points(twin, key):
    assert twin.normalize_tensor_name_strict(key) == key
    assert twin.normalize_tensor_name(key) == key
    assert twin.first_non_canonical([key]) is None


@pytest.mark.parametrize("twin", TWINS)
def test_mla_moe_mtp_vendor_keys_normalize_to_fixed_point(twin):
    vendor_to_canonical = (
        ("model.layers.0.self_attn.q_a_proj.weight", "model.block.0.attn.q_a_proj.weight"),
        ("model.layers.0.self_attn.q_a_layernorm.weight", "model.block.0.attn.q_a_layernorm.weight"),
        ("model.layers.0.self_attn.q_b_proj.weight", "model.block.0.attn.q_b_proj.weight"),
        ("model.layers.0.mlp.gate.e_score_correction_bias", "model.block.0.ffn.router.e_score_correction_bias"),
        ("model.layers.61.eh_proj.weight", "model.block.61.eh_proj.weight"),
        ("model.layers.61.enorm.weight", "model.block.61.enorm.weight"),
        ("model.layers.61.hnorm.weight", "model.block.61.hnorm.weight"),
        ("model.layers.61.shared_head.head.weight", "model.block.61.shared_head.head.weight"),
        ("model.layers.61.shared_head.norm.weight", "model.block.61.shared_head.norm.weight"),
    )
    for vendor_key, canonical in vendor_to_canonical:
        assert twin.normalize_tensor_name_strict(vendor_key) == canonical
        assert twin.normalize_tensor_name(vendor_key) == canonical
        assert twin.normalize_tensor_name_strict(canonical) == canonical
    twin.refuse_non_canonical(MLA_MOE_MTP_KEYS, "deepseek_v3", "transformer")
