"""A daemon serves warm when its flow holds every component together during a request (2026-11-10: cold, a served
Qwen3-Coder-30B-A3B reloaded 57.6 GB and recopied 31 GB of experts at every request, ~150 s of a 170 s answer).

What this test would do if the code were wrong: a text decode (the language model and its head, no phases) reads as
not held together — RED; an image decode, which releases its language model before its decoder loads, reads as held
together — RED (it would keep both and overrun the plan)."""
from neurobrix.core.flow.base import holds_every_component

TEXT = {"flow": {"type": "autoregressive_generation", "generation": {"type": "autoregressive"}}}
IMAGE = {"flow": {"type": "autoregressive_generation",
                  "generation": {"type": "autoregressive_image", "lm_component": "language_model",
                                 "decoder_component": "gen_vision_model"}}}


def test_a_text_decode_holds_its_model_and_head_together():
    assert holds_every_component(TEXT, "compiled", ["model", "lm_head"])
    assert holds_every_component(TEXT, "triton", ["model", "lm_head"])


def test_an_image_decode_on_the_triton_engine_does_not_hold_its_model_beside_its_decoder_on_a_first_request():
    # a served plan has one phase on both engines (the module docstring of base.py); the rule reads the phases as
    # served, so it names exactly the components that phase holds
    from neurobrix.core.flow.base import resident_together
    phases = resident_together(IMAGE, "triton", served=True)
    held = set().union(*phases) if phases else set()
    assert holds_every_component(IMAGE, "triton", sorted(held)) == any(held <= set(p) for p in phases)
    assert not holds_every_component(IMAGE, "triton", sorted(held | {"a_component_no_phase_names"}))
