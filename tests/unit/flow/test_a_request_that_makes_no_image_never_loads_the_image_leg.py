"""A VLM whose build declares a generative-image leg (`flow.image_gen`) runs it on `--mode image` alone (both engines'
handlers); a single request of any other known mode never loads that leg's own components, while the language model it
conditions on stays. A served session, or an unknown mode, prices every leg. Before: Ming-Lite-Omni-1.5 native, a text
request on a V100-16GB, priced its image denoiser and connector in every phase and ran zero3, twice as slow (lot gate
2026-10-28: 281 s vs 142). No card."""
from neurobrix.core.flow.base import unloaded_by_request

FLOW = {"type": "vlm",
        "vlm": {"vision_component": "vision", "lm_component": "model.model"},
        "generation": {"type": "autoregressive_text", "lm_component": "model.model", "head_component": "model.lm_head"},
        "image_gen": {"components": {"condition_lm": "model.model", "connector": "condition_connector",
                                     "denoiser": "image_denoiser", "vae": "image_vae"}}}


def test_a_text_request_never_loads_the_image_legs_own_components():
    assert unloaded_by_request({"flow": FLOW}, "text") == {"condition_connector", "image_denoiser", "image_vae"}


def test_an_image_request_a_served_session_and_an_unknown_mode_load_them():
    assert unloaded_by_request({"flow": FLOW}, "image") == set()
    assert unloaded_by_request({"flow": FLOW}, "text", served=True) == set()
    assert unloaded_by_request({"flow": FLOW}, None) == set()
