"""The KV-cached decoder of a flow is the component the flow NAMES — the VLM flow's
`lm_component` as much as the autoregressive flow's session LM — so Prism charges the
KV cache it plans to that component and an expert store beside it is granted the room
that remains, never the room the cache was promised.

Measured 2026-10-13 on Qwen3-Omni-30B-A3B-Instruct (32 GB V100, triton): the flow is
`vlm` with `lm_component: thinker.model`; `decode_lm_component` returned None, the
solver fell back to the names `language_model` / `model` / `llm` (none exist), the KV
plan was charged to nobody, and the expert store's grant took the whole usable card
(`_zero3_resident_budgets`); the cell died at the lm_head copy with 206 MB free.

What this test does if the code is wrong: the VLM cell fails when the named decoder is
not returned (red before this change); the unknown-name cell fails if a name the flow
gives but the container does not hold is taken; the autoregressive and encoder-decoder
cells fail if the earlier readings regress.

Run: python -m pytest tests/unit/runtime/test_the_decoder_a_flow_names_carries_the_kv_plan.py
"""
from neurobrix.core.runtime.lm_facts import decode_lm_component


def test_a_vlm_flow_names_its_decoder():
    topo = {"flow": {"type": "vlm", "vlm": {"lm_component": "thinker.model", "vision_component": "thinker.visual"}}}
    assert decode_lm_component(topo, ["thinker.visual", "thinker.model", "thinker.lm_head"]) == "thinker.model"


def test_a_name_the_container_does_not_hold_is_not_taken():
    topo = {"flow": {"type": "vlm", "vlm": {"lm_component": "language_model"}}}
    assert decode_lm_component(topo, ["model.visual", "model.language_model"]) is None


def test_the_earlier_readings_stand():
    ar = {"flow": {"type": "autoregressive_generation", "generation": {"lm_component": "model"}}}
    assert decode_lm_component(ar, ["model", "lm_head"]) == "model"
    ed = {"flow": {"type": "encoder_decoder", "stages": [
        {"component": "model.encoder", "execution": "once"},
        {"component": "model.decoder", "execution": "autoregressive", "cross_attention_from": "model.encoder"}]}}
    assert decode_lm_component(ed, ["model.encoder", "model.decoder"]) == "model.decoder"
    assert decode_lm_component({"flow": {"type": "iterative_process"}}, ["transformer"]) is None
