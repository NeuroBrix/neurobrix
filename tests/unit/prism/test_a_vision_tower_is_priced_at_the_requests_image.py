"""A vision-language model's vision tower is priced at the request's image, not at its trace's.

The image reaches the tower as the build's processor makes it (`input_processor.prepare_image_inputs`): the patch
rows (`hidden_states`) and the grid (`grid_thw`, whose VALUES bind its grid symbols) — the flow's contract. Prism's
flow bindings had no rule for it, so every one of the tower's symbols took its trace extent: Ming-Lite-Omni-1.5's
tower, traced on a 5 x 46 x 74 grid (17 020 patches), was priced at 9.65 GiB of activations (four 17 020 x 17 020
fp32 attention matrices) for a 448x448 image of 1 024 patches, and its Triton plan was refused on the Mac's
profile and on a 16 GB card (2026-10-14).

What this test does if the code is wrong: the binding cell fails when the tower's patch rows and grid symbols are
not the request image's (red on main: 17 020 patches, a 46 x 74 grid); the no-image cell fails when a request
without an image binds anything for the tower; the other-component cell fails when the rule reaches a component
that is not the flow's vision tower; the plan cell fails when the tower's activations are not priced at the image.

Run: python -m pytest tests/unit/prism/test_a_vision_tower_is_priced_at_the_requests_image.py
"""
import json
from pathlib import Path

from tests.unit.prism._pinned_machine import container_root

MODEL = "Ming-Lite-Omni-1.5"
IMAGE = Path(__file__).resolve().parents[3] / "benchmarks" / "assets" / "apple_448.png"


def _bindings(image=IMAGE):
    from neurobrix.core.prism.flow_bindings import FlowBindings
    root = container_root(MODEL)
    topo = json.loads((root / "topology.json").read_text())
    name = json.loads((root / "manifest.json").read_text()).get("model_name")
    return FlowBindings(topo, root, name, image_path=str(image) if image else None), topo


def test_the_tower_binds_the_request_images_patches_and_grid():
    fb, topo = _bindings()
    vis = topo["flow"]["vlm"]["vision_component"]
    dag = fb.graph(vis)
    syms = dag["symbolic_context"]["symbols"]
    got = fb.overrides(dag, None)
    by_source = {syms[s]["source"]: v for s, v in got.items()}
    # 448 / 14 = 32 patches a side, one frame: 1 024 patch rows, the grid 1 x 32 x 32
    assert by_source.get("input::hidden_states::dim_0") == 1024, by_source
    assert by_source.get("input::grid_thw::val_1") == 32 and by_source.get("input::grid_thw::val_2") == 32
    assert by_source.get("input::grid_thw::val_0") == 1
    assert max(int(s["trace_value"]) for s in syms.values() if s["source"] == "input::hidden_states::dim_0") > 1024, \
        "precondition: the trace's patch count is not the request's"


def test_a_request_without_an_image_binds_nothing_for_the_tower():
    fb, topo = _bindings(image=None)
    vis = topo["flow"]["vlm"]["vision_component"]
    assert fb.vision_inputs(vis) is None


def test_the_rule_reaches_only_the_flows_vision_tower():
    fb, topo = _bindings()
    lm = topo["flow"]["vlm"]["lm_component"]
    assert fb.vision_inputs(lm) is None


def test_the_towers_activations_are_priced_at_the_image():
    from neurobrix.core.prism.profiler import ActivationProfiler
    from neurobrix.core.prism.solver import InputConfig
    fb, topo = _bindings()
    vis = topo["flow"]["vlm"]["vision_component"]
    cfg = InputConfig(batch_size=1, height=448, width=448, flow=fb, dtype="bfloat16", vae_scale=8, mode="text")
    ap = ActivationProfiler(fb.graph(vis)).estimate_peak_memory(input_config=cfg, dtype_bytes=2, placement_floor=True)
    assert ap.peak_bytes < 1 << 30, f"{ap.peak_bytes / 2**30:.2f} GiB for a 1 024-patch image"


def test_a_staged_tower_binds_the_processors_three_outputs():
    """The STAGED contract (the tower declares `all_pixel_values`, MiniCPM-o-4_5): the tower takes the processor's
    pixel rows, patch mask and `tgt_sizes`, whose values bind its grid symbols (the first version of the rule fed it
    the splice contract's two inputs and refused the request, 2026-10-14 23:46)."""
    from neurobrix.core.prism.flow_bindings import FlowBindings
    root = container_root("MiniCPM-o-4_5")
    topo = json.loads((root / "topology.json").read_text())
    name = json.loads((root / "manifest.json").read_text()).get("model_name")
    fb = FlowBindings(topo, root, name, image_path=str(IMAGE))
    vis = topo["flow"]["vlm"]["vision_component"]
    feed = fb.vision_inputs(vis)
    assert sorted(feed) == ["all_pixel_values", "patch_attention_mask", "tgt_sizes"]
    dag = fb.graph(vis)
    syms = dag["symbolic_context"]["symbols"]
    got = {syms[s]["source"]: v for s, v in fb.overrides(dag, None).items()}
    assert got.get("input::tgt_sizes::val_0") == 32 and got.get("input::tgt_sizes::val_1") == 32, got
