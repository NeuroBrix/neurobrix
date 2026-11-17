"""A vision encoder's patch stem, position table and patch merger spellings resolve, each to its role.

Four tokens a vision encoder's weights may carry name roles the vocabulary already has:
`patch_conv` the patch embedding, `patch_positional_embedding` the position embedding,
`patch_merger` a patch merger's down-sampling (Swin's `downsample`) and `merging_layer` its linear.
Each is translated by the vocabulary alone, strict and permissive, inside a whole key, and the
translated key is a fixed point of the parser. Without its entry the strict parser refuses the key
by the token's name. (`ln_pre`, the fifth spelling, is pinned by the kept-tokens gate test.)
"""
from __future__ import annotations

import pytest

from neurobrix.nbx.neurotax import NeuroTaxParser, SynonymRegistry, first_non_canonical

# token -> (a stored key holding it, the role it names, the key the parser returns)
SPELLINGS = {
    "patch_conv": ("vision_tower.patch_conv.weight", "patch_embed", "vision.patch_embed.weight"),
    "patch_positional_embedding": (
        "vision_tower.patch_positional_embedding.weight", "pos_embed", "vision.pos_embed.weight"),
    "patch_merger": (
        "multi_modal_projector.patch_merger.weight", "down_sample", "multi_modal_projector.down_sample.weight"),
    "merging_layer": (
        "multi_modal_projector.patch_merger.merging_layer.weight", "proj",
        "multi_modal_projector.down_sample.proj.weight"),
}


@pytest.mark.parametrize("token", sorted(SPELLINGS))
def test_the_spelling_resolves_to_its_role(token):
    role = SPELLINGS[token][1]
    assert SynonymRegistry.resolve(token) == role
    assert SynonymRegistry.resolve_strict(token, token) == role


@pytest.mark.parametrize("token", sorted(SPELLINGS))
def test_a_key_holding_the_spelling_gets_its_standard_name(token):
    key, _, canon = SPELLINGS[token]
    assert NeuroTaxParser().normalize_strict(key) == canon
    assert NeuroTaxParser().normalize(key) == canon
    assert first_non_canonical([canon]) is None
