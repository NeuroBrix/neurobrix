"""NextDiT structural tokens and patch-size pair indices are accepted as fixed points by both NeuroTax twins."""
from __future__ import annotations

import pytest

from neurobrix.pytorch import neurotax as pytorch_neurotax
from neurobrix.triton import neurotax as triton_neurotax

NEXTDIT_TOKENS = (
    "all_final_layer",
    "all_x_embedder",
    "cap_embedder",
    "cap_pad_token",
    "context_refiner",
    "x_pad_token",
)

NEXTDIT_KEYS = (
    "all_final_layer.2-1.adaln_mod.1.bias",
    "all_x_embedder.2-1.proj.weight",
    "context_refiner.0.attn.query.weight",
    "cap_embedder.0.weight",
    "cap_pad_token",
    "x_pad_token",
)

TWINS = (
    pytest.param(pytorch_neurotax, id="pytorch"),
    pytest.param(triton_neurotax, id="triton"),
)


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("token", NEXTDIT_TOKENS)
def test_nextdit_tokens_are_self_mapped_fixed_points(twin, token):
    reg = twin.SynonymRegistry
    assert reg._REGISTRY[token] == token
    assert reg.resolve(token) == token
    assert reg.resolve_strict(token, token) == token
    for glued in (f"{token}7", f"{token}_7"):
        assert reg.resolve(glued) == glued
        assert reg.resolve_strict(glued, glued) == glued


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("pair", ("2-1", "1-1", "4-2"))
def test_patch_pair_indices_are_preserved_and_fixed_points(twin, pair):
    reg = twin.SynonymRegistry
    assert reg.resolve(pair) == pair
    assert reg.resolve_strict(pair, pair) == pair


@pytest.mark.parametrize("twin", TWINS)
@pytest.mark.parametrize("key", NEXTDIT_KEYS)
def test_nextdit_keys_are_accepted_and_fixed_points(twin, key):
    assert twin.normalize_tensor_name_strict(key) == key
    assert twin.normalize_tensor_name(key) == key
    assert twin.first_non_canonical([key]) is None


@pytest.mark.parametrize("twin", TWINS)
def test_nextdit_vendor_key_normalizes_to_fixed_point(twin):
    vendor_key = "all_final_layer.2-1.adaLN_modulation.1.bias"
    canonical = "all_final_layer.2-1.adaln_mod.1.bias"
    assert twin.normalize_tensor_name_strict(vendor_key) == canonical
    assert twin.normalize_tensor_name(vendor_key) == canonical
    assert twin.normalize_tensor_name_strict(canonical) == canonical
    twin.refuse_non_canonical(NEXTDIT_KEYS, "nextdit_model", "transformer")
