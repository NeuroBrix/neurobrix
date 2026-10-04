"""NeuroTax 5.0, complete: the vocabulary names every key of the catalogue, and stays a fixed point doing it.

The 2026-10-03 census of the 49 cached containers found 362 tokens the 5.0 registry refused
(23 649 keys, 43 containers): VAE and codec stages, vocoders, vision towers, speech front-ends.
`data/neurotax_5_0_refused_tokens.txt` is that list, read from the cache once; every token in it
must now resolve strictly. A token with a glued index (`tdnnd12`, `conv3`, `feed_forward1`)
resolves through its stem with the index kept, so the 25th dense layer is named like the first.

What these would do on the 5.0 parser: the first fails on 362 tokens; the glued-index cells fail
on `tdnnd12` / `layer_norm1`; the family table fails on every row; the PyTorch-spelling cell
fails on `weight_ih_l0_reverse` / `original0`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neurobrix.nbx.neurotax import (NEUROTAX_VERSION, SynonymRegistry, normalize_tensor_name,
                                    normalize_tensor_name_strict)

REFUSED_5_0 = (Path(__file__).parent / "data" / "neurotax_5_0_refused_tokens.txt").read_text().split()


def test_the_version_is_5_0():
    # The completed vocabulary IS 5.0 (the owner, 2026-10-04 02:05): no second public version.
    assert NEUROTAX_VERSION == "5.0"


def test_every_token_5_0_refused_resolves():
    assert len(REFUSED_5_0) == 362
    refused = []
    for t in REFUSED_5_0:
        try:
            SynonymRegistry.resolve_strict(t, t)
        except ValueError:
            refused.append(t)
    assert refused == []


def test_every_resolved_token_is_a_fixed_point():
    moved = {}
    for t in REFUSED_5_0:
        c = SynonymRegistry.resolve_strict(t, t)
        if SynonymRegistry.resolve_strict(c, c) != c or SynonymRegistry.resolve(c) != c:
            moved[t] = c
    assert moved == {}


@pytest.mark.parametrize("token,canonical", [
    ("tdnnd12", "dense_layer12"), ("tdnnd25", "dense_layer25"), ("conv3", "conv3"), ("conv5", "conv5"),
    ("layer_norm1", "norm1"), ("feed_forward2", "ffn2"), ("conv2d3", "conv3"), ("rdb1", "dense_block1"),
    ("convolution_0", "conv_0"), ("conv_up2", "up_sample_conv2"), ("pwconv1", "pointwise_conv1"),
    ("layer1", "block1"),
    # an explicit entry wins over the stem: `fc1` is the CLIP/OPT up projection, not `proj1`
    ("fc1", "up"), ("linear_1", "proj_1"), ("norm1", "norm1"), ("wi_0", "up_0"),
])
def test_a_glued_index_keeps_its_index_and_translates_its_stem(token, canonical):
    assert SynonymRegistry.resolve(token) == canonical
    assert SynonymRegistry.resolve_strict(token, token) == canonical
    assert SynonymRegistry.resolve(canonical) == canonical


def test_every_stem_derivation_is_a_fixed_point():
    stems = {t for t in SynonymRegistry._REGISTRY if t[-1].isalpha()} | {
        t for t in SynonymRegistry.canonical_tokens() if t[-1].isalpha()}
    moved = {}
    for s in sorted(stems):
        for glued in (f"{s}7", f"{s}_7"):
            c = SynonymRegistry.resolve(glued)
            if c != glued and (SynonymRegistry.resolve(c) != c or SynonymRegistry.resolve_strict(c, c) != c):
                moved[glued] = (c, SynonymRegistry.resolve(c))
    assert moved == {}


@pytest.mark.parametrize("token", ["weight_ih_l0", "bias_hh_l2", "weight_ih_l0_reverse", "in_proj_weight",
                                   "in_proj_bias", "weight_g", "weight_v", "parametrizations", "original0",
                                   "original1"])
def test_pytorchs_own_parameter_spellings_stay(token):
    assert SynonymRegistry.resolve_strict(token, token) == token


# One key per family, as the partial-vocabulary container holds it -> as the complete vocabulary names it (read from the cache, 2026-10-04).
FAMILY_KEYS = [
    ("decoder.mid.temp_convs.0.conv1.0.bias", "decoder.mid.temporal_conv.0.conv1.0.bias"),          # video VAE
    ("decoder.conv_norm_out.weight", "decoder.norm_out.weight"),
    ("double_blocks.0.img_attn.norm.key_norm.scale", "block.0.attn.norm.norm_k.scale"),            # MMDiT
    ("single_blocks.0.modulation.lin.bias", "single_block.0.mod.proj.bias"),
    ("model.block.0.attn.k_norm.weight", "model.block.0.attn.norm_k.weight"),                     # LLM
    ("block.1.ffn.shared_experts.ffn_gate.weight", "block.1.ffn.shared_expert.ffn_gate.weight"),
    ("block.0.attn.kv_a_proj_with_mqa.weight", "block.0.attn.kv_down.weight"),
    ("vision_tower.attn_pool.kv.bias", "vision.pool_attn.kv.bias"),                                 # vision
    ("merger.ln_q.weight", "mm_proj.norm_q.weight"),
    ("encoder.block.0.feed_forward1.linear1.bias", "encoder.block.0.ffn1.proj1.bias"),              # STT
    ("encoder.block.0.attn.linear_pos.weight", "encoder.block.0.attn.pos_proj.weight"),
    ("mel2wav.resblocks.0.convs1.0.parametrizations.weight.original0",                               # vocoder
     "vocoder.resblock.0.conv1.0.parametrizations.weight.original0"),
    ("speaker_encoder.xvector.block1.tdnnd1.nonlinear1.batchnorm.weight",
     "voice_encoder.trunk.block1.dense_layer1.bn_act1.bnorm.weight"),
    ("flow.decoder.estimator.mid_blocks.0.0.ffn.1.weight", "acoustic.decoder.denoiser.bottleneck_block.0.0.ffn.1.weight"),
    ("decode.0.conv1x1.weight_g", "dec_resnet.0.skip_conv.weight_g"),
    ("body.0.rdb1.conv1.weight", "trunk.0.dense_block1.conv1.weight"),                              # upscaler
    ("swin2sr.encoder.stages.0.block.0.attn.self.logit_scale", "model.encoder.stage.0.block.0.attn.attn.logit_scale"),
    ("decoder.stages.0.2.mixer.conv.conv.conv.bias", "decoder.stage.0.2.token_mixer.conv.conv.conv.bias"),  # codec
    ("pred.weight", "pred_proj.weight"),                                                             # transducer joint
]


@pytest.mark.parametrize("old,new", FAMILY_KEYS)
def test_a_5_0_key_is_renamed_by_the_parser(old, new):
    assert normalize_tensor_name_strict(old) == new
    assert normalize_tensor_name(old) == new
    assert normalize_tensor_name_strict(new) == new
