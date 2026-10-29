"""The derivation derives a VLM's vision tower and projector UNDER THE PROFILE it derives for — its
native-bf16 flag and its attention budget — never under a stub.

The VLM extent sites (the staged, masked and splice contracts) and the audio-LLM forward stages
called `run_at_inputs(..., False, (0, 1, 1), ...)`: no native bf16, a zero attention budget. On a
V100 that is the profile by coincidence; on the Mac's M4 Pro (bf16 native) the vision tower's
output was derived fp32 and the projector keyed with a fp32 activation — MiniCPM-o-4_5's run asked
`addmm (64, 4096, 4096) bf16,bf16,bf16,bf16` at the projector and the derived table held
`fp32,fp32,bf16,fp32` (the Mac, 2026-10-14 06:18; reproduced on the CPU with its profile).

What this test does if the code is wrong: each cell fails when a site hands `run_at_inputs` a flag
or a budget other than the profile's (red before this change: the staged and masked sites gave
False and (0, 1, 1) whatever the profile; the parameters did not exist).

Run: PYTHONPATH=src python -m pytest tests/unit/census/test_the_derivation_derives_a_vlm_leg_under_its_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import derived_census as D  # noqa: E402

# A decoding plan carries its KV cache (step 3, 2026-10-18: `lm_facts.decoder_cache_from_plan` refuses a decoder
# without one, and the derivation with it, `_decoder_sites`) — the LM's geometry here is the stub's.
PLAN = {"components": [{"name": "vpm", "dtype": "bfloat16"}, {"name": "resampler", "dtype": "bfloat16"},
                       {"name": "llm", "dtype": "bfloat16"}],
        "kv_cache": {"max_cache_len": 64, "dtype": "bfloat16", "num_kv_heads": 4}}
SDPA = (123456, 4, 5)


def _stubs(monkeypatch, seen):
    """The processor, the graphs, the tokeniser and `run_at_inputs` stubbed; `run_at_inputs` records the
    profile it was handed and answers a shaped output so the site builder finishes."""
    def run_at_inputs(model, comp, cdtype, mode, inputs, has_native_bf16, sdpa, unhandled, **kw):
        seen.append((comp, has_native_bf16, tuple(sdpa)))
        return [], {"output_0": [1, 64, 4096]}
    monkeypatch.setattr(D, "run_at_inputs", run_at_inputs)
    monkeypatch.setattr(D, "raw_graph", lambda model, comp: {"tensors": {"input::inputs_embeds": {"shape": [1, 8, 4096]}}})
    import neurobrix.core.module.vision.input_processor as IP
    monkeypatch.setattr(IP, "prepare_image_inputs", lambda topo, model, path, root: {
        "global.pixel_values": np.zeros((1, 3, 448, 448), np.float32), "global.image_grid_thw": np.array([[1, 32, 32]]),
        "global.all_pixel_values": np.zeros((1, 3, 448, 448), np.float32), "global.patch_attention_mask": np.ones((1, 1, 1024), bool),
        "global.tgt_sizes": np.array([[32, 32]])})
    import neurobrix.core.runtime.decode_bound as DB
    monkeypatch.setattr(DB, "decode_bound", lambda n: int(n or 8))
    import neurobrix.triton.flow.vlm as V
    monkeypatch.setattr(V.TritonVLMEngine, "_tokenize_with_placeholder_run", staticmethod(lambda *a, **k: ([1, 2, 3], 1, 2)))
    monkeypatch.setattr(V.TritonVLMEngine, "_build_masked_splice_ids", staticmethod(lambda *a, **k: ([1, 2, 3], 1, 2)), raising=False)
    monkeypatch.setattr(D, "_vlm_flow_stub", lambda model: SimpleNamespace(), raising=False)
    # the flow's logits after each LM step: the head's (V, H) weight and its GEMV, derived since the census reads the
    # head (the cells here look at the vision legs' profile; the head is the catalogue's usual one, no launch kept)
    monkeypatch.setattr(D, "_head_rows", lambda model, _u, head: (32000, 4096, "float16"))
    monkeypatch.setattr(D, "_flow_logits", lambda model, lm, rows, native: [])


def test_the_staged_contract_derives_the_tower_and_the_projector_under_the_profile(monkeypatch):
    seen = []
    _stubs(monkeypatch, seen)
    topo = {"flow": {"vlm": {"vision_component": "vpm", "vision_projection_component": "resampler",
                             "lm_component": "llm", "preprocessing": {}}}}
    D._vlm_staged_sites("M", topo, {"max_tokens": 8}, PLAN, "img.png", "describe", None, True, SDPA)
    assert seen == [("vpm", True, SDPA), ("resampler", True, SDPA)], seen


def test_the_sites_take_the_profile_from_extent_sites(monkeypatch):
    seen = []
    _stubs(monkeypatch, seen)
    topo = {"flow": {"type": "vlm", "vlm": {"vision_component": "vpm", "vision_projection_component": "resampler",
                                           "lm_component": "llm", "head_component": "lm_head",
                                           "preprocessing": {}}}}     # every VLM of the catalogue names its head
    monkeypatch.setattr(D, "vlm_contract", lambda model, topo: "staged")
    D.extent_sites("M", topo, {"max_tokens": 8}, PLAN, "describe", None, None, "img.png",
                   has_native_bf16=True, sdpa=SDPA)
    assert [s[1:] for s in seen] == [(True, SDPA), (True, SDPA)], seen
    seen.clear()
    D.extent_sites("M", topo, {"max_tokens": 8}, PLAN, "describe", None, None, "img.png",
                   has_native_bf16=False, sdpa=(7, 1, 1))
    assert [s[1:] for s in seen] == [(False, (7, 1, 1)), (False, (7, 1, 1))], seen
