"""An autoregressive IMAGE decode (`generation.type: autoregressive_image`) loads, in both engines' handlers, exactly the
components its `generation` block names — the language model, the head, the embedding, the aligner, the decoder — on
every request and every mode; a component of the flow's order that block does not name (an understanding tower and its
aligner) is never loaded, and Prism prices it nowhere. Before: Janus-Pro-7B triton on a V100-16GB priced its
`vision_model` (723 MB) and `aligner` (42 MB) in both phases, the decode phase held 14 967 MB over the 14 965 permitted
and the plan went from lazy to zero3 (242 s instead of 113, 2026-10-28). A TEXT decode declares nothing. No card."""
from neurobrix.core.flow.base import unloaded_by_request

GEN = {"type": "autoregressive_image", "lm_component": "language_model", "decoder_component": "gen_vision_model",
       "head_component": "gen_head", "embed_component": "gen_embed", "aligner_component": "gen_aligner"}
ORDER = ["aligner", "gen_aligner", "gen_embed", "gen_head", "gen_vision_model", "language_model", "vision_model"]


def _topology(gen):
    return {"flow": {"type": "autoregressive_generation", "order": ORDER, "generation": gen}}


def test_the_components_the_generation_does_not_name_are_never_loaded():
    for mode in (None, "image"):
        for served in (False, True):
            assert unloaded_by_request(_topology(GEN), mode, served=served) == {"aligner", "vision_model"}, (mode, served)


def test_a_text_decode_and_a_block_that_leaves_a_component_unnamed_declare_nothing():
    assert unloaded_by_request(_topology({**GEN, "type": "autoregressive_text"})) == set()
    for k in ("lm_component", "decoder_component", "head_component", "embed_component", "aligner_component"):
        assert unloaded_by_request(_topology({n: v for n, v in GEN.items() if n != k})) == set(), k
