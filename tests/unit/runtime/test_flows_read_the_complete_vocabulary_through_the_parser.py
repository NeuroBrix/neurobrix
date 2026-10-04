"""The flows that read weights by name outside the graph read them through the parser (rule 8).

The complete vocabulary (2026-10-04) renames the transducer joint's `enc` / `pred` to `enc_proj` / `pred_proj` and the
speech vocoder's `input_embedding` to `token_embed`. The RNNT flows (both modes) matched the raw
suffixes `enc.weight` / `pred.weight`, and the TTS flows (both modes) found the vocoder's token
table by `"embedding" in weight_name`: on a renamed container the first refuses the joint, the second
finds no vocab size and lets the special tokens through to the vocoder. Both now name the
vendor's structure once and take its key from the parser.

What these would do on the old code: the joint cells fail (no role found for `enc_proj.weight`);
the vocoder cell fails (`input_embedding` is no NeuroTax token; `"embedding" in` misses `token_embed`).
"""
from __future__ import annotations

import pytest

from neurobrix.nbx.neurotax import normalize_tensor_name_strict

# parakeet-tdt-1.1b's joint component, partial keys -> complete keys (the cache, 2026-10-04)
JOINT_KEYS = {"pred_proj.weight": "dec_weight", "pred_proj.bias": "dec_bias",
             "enc_proj.weight": "enc_weight", "enc_proj.bias": "enc_bias",
             "joint.2.weight": "out_weight", "joint.2.bias": "out_bias"}


def _modules():
    import neurobrix.core.flow.rnnt as core_rnnt
    import neurobrix.triton.flow.rnnt as triton_rnnt
    return core_rnnt, triton_rnnt


@pytest.mark.parametrize("mode", [0, 1])
def test_the_joint_is_found_on_the_complete_keys(mode):
    mod = _modules()[mode]
    assert {k: mod.joint_role(k) for k in JOINT_KEYS} == JOINT_KEYS
    assert mod.joint_role("joint.enc_proj.weight") == "enc_weight"      # under a component prefix
    assert mod.joint_role("enc.weight") is None                          # the partial spelling is not a key


def test_the_joint_keys_are_the_parsers():
    core_rnnt, triton_rnnt = _modules()
    assert core_rnnt._JOINT_KEYS == triton_rnnt._JOINT_KEYS
    assert set(core_rnnt._JOINT_KEYS) == set(JOINT_KEYS)


def test_the_vocoder_token_table_is_found_by_its_canonical_token():
    import neurobrix.core.flow.tts_llm as core_tts
    import neurobrix.triton.flow.tts_llm as triton_tts
    for mod in (core_tts, triton_tts):
        # chatterbox s3gen: partial `flow.input_embedding.weight` -> `acoustic.token_embed.weight`
        new = normalize_tensor_name_strict("flow.input_embedding.weight")
        assert new == "acoustic.token_embed.weight"
        assert mod._TOKEN_EMBED in new.split(".")
        # the speaker projection beside it carries `embed` in no token of its own
        assert mod._TOKEN_EMBED not in normalize_tensor_name_strict("flow.spk_embed_affine_layer.weight").split(".")
