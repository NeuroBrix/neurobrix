"""A multimodal build is made for ONE generation type: a request asking another mode is refused by name, at the CLI and
at the serving door alike (`core/runtime/build_modes.build_refuses_mode`). Janus-Pro-7B (built `autoregressive_image`)
served by chat completions ran its image autoregression for a text request and answered the image tokens as text
(" inqu grav Store European reform...", both engines, 2026-11-13)."""
from types import SimpleNamespace

from neurobrix.core.runtime.build_modes import build_refuses_mode
from neurobrix.serving.openai_api.routes import Served

IMAGE_BUILD = {"flow": {"type": "autoregressive_generation", "generation": {"type": "autoregressive_image"}}}
TEXT_BUILD = {"flow": {"type": "autoregressive_generation", "generation": {"type": "autoregressive_text"}}}


def test_an_image_build_refuses_text_and_serves_images():
    why = build_refuses_mode(IMAGE_BUILD, "text")
    assert why and "only --mode image" in why and "autoregressive_image" in why
    assert build_refuses_mode(IMAGE_BUILD, "image") is None


def test_a_text_build_refuses_images_unless_it_declares_a_diffusion_leg():
    assert "only --mode text" in build_refuses_mode(TEXT_BUILD, "image")
    with_leg = {"flow": {**TEXT_BUILD["flow"], "image_gen": {"decoder": "x"}}}
    assert build_refuses_mode(with_leg, "image") is None


def test_a_build_declaring_no_generation_type_or_a_mode_outside_the_rule_is_not_refused():
    assert build_refuses_mode({"flow": {"type": "vlm"}}, "text") is None
    assert build_refuses_mode(IMAGE_BUILD, "audio") is None
    assert build_refuses_mode(IMAGE_BUILD, None) is None


def test_the_serving_door_reads_the_loaded_build():
    served = Served.__new__(Served)
    served._engine = SimpleNamespace(package=SimpleNamespace(topology=IMAGE_BUILD))
    assert "only --mode image" in served.refusal("text")
    assert served.refusal("image") is None
