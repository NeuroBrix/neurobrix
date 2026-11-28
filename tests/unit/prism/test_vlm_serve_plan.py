"""VLM and autoregressive_text served planning (CPU-only, twin PyTorch + Triton).

Covers:
1. `autoregressive_text` containers carrying unused vision components (Devstral-Small-2-24B, Qwen3.8-27B):
   `unloaded_by_request` skips components outside `{lm_component, head_component, codec.decoder}` in both
   single-run and served mode, while keeping `codec.decoder` (Orpheus SNAC) loaded.
2. True `flow.type == "vlm"` containers in `serve_mode=True` whose vision tower declares `pixel_values`
   with symbolic `height`/`width`:
   - binds `height`/`width` from container-declared `vision_image_size` or `preprocessor_config.json`
     via `FlowBindings` on BOTH PyTorch and Triton without `MissingRuntimeValue`;
   - still raises `MissingRuntimeValue` when neither is declared (strict ZERO FALLBACK);
   - non-VLM spatial graphs with missing `height`/`width` still raise `MissingRuntimeValue`.
3. Real `Devstral-Small-2-24B-Instruct-2512` container metadata CPU-only `solve_served` on both PyTorch
   (`mode="compiled"`) and Triton (`mode="triton"`), skipped cleanly if container is absent.
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest

from neurobrix.core.flow.base import unloaded_by_request
from neurobrix.core.runtime_values import MissingRuntimeValue


def _vision_dag(comp: str = "model.vision_tower") -> dict:
    return {
        "component_name": comp,
        "tensors": {
            "input::pixel_values": {
                "shape": [1, 3, 168, 532],
                "symbolic_shape": ["s0", 3, "s1", "s2"],
                "dtype": "bfloat16",
            }
        },
        "ops": {},
        "execution_order": [],
        "input_tensor_ids": ["input::pixel_values"],
        "output_tensor_ids": [],
        "symbolic_context": {
            "symbols": {
                "s0": {"name": "batch", "trace_value": 1, "source": "input::pixel_values::dim_0"},
                "s1": {"name": "height", "trace_value": 168, "source": "input::pixel_values::dim_2"},
                "s2": {"name": "width", "trace_value": 532, "source": "input::pixel_values::dim_3"},
            },
            "expressions": {},
        },
    }


def _vlm_topology(*, vision_image_size: int | None = None) -> dict:
    ev = {"model.language_model": {"hidden_size": 5120}}
    if vision_image_size is not None:
        ev["model.language_model"]["vision_image_size"] = vision_image_size
    return {
        "flow": {
            "type": "vlm",
            "order": ["model.vision_tower", "model.multi_modal_projector", "model.language_model", "lm_head"],
            "vlm": {
                "vision_component": "model.vision_tower",
                "vision_projection_component": "model.multi_modal_projector",
                "lm_component": "model.language_model",
                "head_component": "lm_head",
            },
        },
        "connections": [
            {"from": "global.pixel_values", "to": "model.vision_tower.pixel_values"},
        ],
        "extracted_values": ev,
    }


def test_autoregressive_text_unloads_unused_vision_components_and_keeps_codec():
    topo = {
        "flow": {
            "type": "autoregressive_generation",
            "order": [
                "model.language_model",
                "model.multi_modal_projector",
                "model.vision_tower",
                "lm_head",
            ],
            "generation": {
                "type": "autoregressive_text",
                "lm_component": "model.language_model",
                "head_component": "lm_head",
            },
        }
    }
    for served in (False, True):
        assert unloaded_by_request(topo, "text", served=served) == {
            "model.multi_modal_projector",
            "model.vision_tower",
        }

    # Orpheus TTS codec.decoder stays loaded
    orpheus_topo = {
        "flow": {
            "type": "autoregressive_generation",
            "order": ["model", "lm_head", "codec.decoder"],
            "generation": {
                "type": "autoregressive_text",
                "lm_component": "model",
                "head_component": "lm_head",
            },
        }
    }
    for served in (False, True):
        assert unloaded_by_request(orpheus_topo, "text", served=served) == set()


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
def test_vlm_serve_binds_declared_vision_image_size(branch, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    topo = _vlm_topology(vision_image_size=1540)
    dag = _vision_dag()
    fb = fb_mod.FlowBindings(topo, tmp_path, "vlm-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    smap = profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)
    assert smap["s0"] == 1
    assert smap["s1"] == 1540
    assert smap["s2"] == 1540


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
def test_vlm_serve_binds_declared_preprocessor_config(branch, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    proc_dir = tmp_path / "modules" / "image_processor"
    proc_dir.mkdir(parents=True)
    (proc_dir / "preprocessor_config.json").write_text(json.dumps({"size": {"shortest_edge": 384}, "crop_size": 384}))

    topo = _vlm_topology(vision_image_size=None)
    dag = _vision_dag()
    fb = fb_mod.FlowBindings(topo, tmp_path, "vlm-proc-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    smap = profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)
    assert smap["s1"] == 384
    assert smap["s2"] == 384


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
def test_vlm_serve_without_declared_vision_size_still_refuses_zero_fallback(branch, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    topo = _vlm_topology(vision_image_size=None)
    dag = _vision_dag()
    fb = fb_mod.FlowBindings(topo, tmp_path, "vlm-undeclared-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    with pytest.raises(MissingRuntimeValue, match="height"):
        profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
@pytest.mark.parametrize(
    ("pc_payload", "expected_hw"),
    (
        ({"size": {"longest_edge": 1540}}, (1540, 1540)),
        ({"size": 448}, (448, 448)),
        ({"crop_size": {"height": 336, "width": 448}}, (336, 448)),
        ({"size": {"height": 512, "width": 640}}, (512, 640)),
    ),
)
def test_vlm_serve_binds_preprocessor_size_variants(branch, pc_payload, expected_hw, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    proc_dir = tmp_path / "modules" / "image_processor"
    proc_dir.mkdir(parents=True)
    (proc_dir / "preprocessor_config.json").write_text(json.dumps(pc_payload))

    topo = _vlm_topology(vision_image_size=None)
    dag = _vision_dag()
    fb = fb_mod.FlowBindings(topo, tmp_path, "vlm-proc-variant-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    smap = profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)
    assert (smap["s1"], smap["s2"]) == expected_hw


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
@pytest.mark.parametrize("pc_payload", ({}, {"min_pixels": 3136}))
def test_vlm_serve_undeclared_preprocessor_config_refuses_zero_fallback(branch, pc_payload, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    proc_dir = tmp_path / "modules" / "image_processor"
    proc_dir.mkdir(parents=True)
    (proc_dir / "preprocessor_config.json").write_text(json.dumps(pc_payload))

    topo = _vlm_topology(vision_image_size=None)
    dag = _vision_dag()
    fb = fb_mod.FlowBindings(topo, tmp_path, "vlm-empty-proc-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    with pytest.raises(MissingRuntimeValue, match="height"):
        profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
def test_vlm_serve_empty_preprocessor_config_uses_vision_image_size(branch, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    proc_dir = tmp_path / "modules" / "image_processor"
    proc_dir.mkdir(parents=True)
    (proc_dir / "preprocessor_config.json").write_text(json.dumps({}))

    topo = _vlm_topology(vision_image_size=1540)
    dag = _vision_dag()
    fb = fb_mod.FlowBindings(topo, tmp_path, "vlm-empty-proc-ev-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    smap = profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)
    assert smap["s1"] == 1540
    assert smap["s2"] == 1540


@pytest.mark.parametrize("branch", ("pytorch", "triton"))
def test_non_vlm_spatial_graph_without_height_still_refuses(branch, tmp_path):
    profiler_mod = __import__(f"neurobrix.{branch}.prism.profiler", fromlist=["ActivationProfiler", "InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])

    topo = {"flow": {"type": "iterative_process", "loop": {"components": ["transformer"]}}, "connections": []}
    dag = _vision_dag("transformer")
    fb = fb_mod.FlowBindings(topo, tmp_path, "diff-test")
    ic = profiler_mod.InputConfig(batch_size=1, height=None, width=None, dtype="bfloat16", flow=fb)

    with pytest.raises(MissingRuntimeValue, match="height"):
        profiler_mod.ActivationProfiler(dag).build_symbol_map(ic, placement_floor=True)


DEVSTRAL_CONTAINER = pathlib.Path(
    "/mnt/nvme_ai_work/built_models_1127/Devstral-Small-2-24B/Devstral-Small-2-24B-Instruct-2512"
)


@pytest.mark.skipif(not DEVSTRAL_CONTAINER.exists(), reason="Devstral-Small-2-24B container not present locally")
@pytest.mark.parametrize(("branch", "mode"), (("pytorch", "compiled"), ("triton", "triton")))
def test_real_devstral_container_solve_served_cpu_only(branch, mode):
    from neurobrix.nbx import NBXContainer
    from neurobrix.serving.engine import _container_extent, _temporal_axis

    container = NBXContainer.load(str(DEVSTRAL_CONTAINER))
    manifest = container.get_manifest() or {}
    defaults = json.loads((DEVSTRAL_CONTAINER / "runtime" / "defaults.json").read_text())
    topo = json.loads((DEVSTRAL_CONTAINER / "topology.json").read_text())
    h, w, vs = _container_extent(manifest, defaults, topo.get("components") or {})
    nf, tc = _temporal_axis(defaults)

    prism = __import__(f"neurobrix.{branch}.prism", fromlist=["InputConfig"])
    fb_mod = __import__(f"neurobrix.{branch}.prism.flow_bindings", fromlist=["FlowBindings"])
    rc_mod = __import__(f"neurobrix.{branch}.prism.request_config", fromlist=["served_plan_fields"])
    sp_mod = __import__(f"neurobrix.{branch}.prism.served_prefill", fromlist=["solve_served"])
    ld_mod = __import__(f"neurobrix.{branch}.prism.loader", fromlist=["load_profile"])
    mb_mod = __import__(f"neurobrix.{branch}.prism.memory_budget", fromlist=["DeviceReading"])

    hw = ld_mod.load_profile("a100-80g")
    hw.cpu = ld_mod.load_profile("c4140-4xv100-16GB-nvlink").cpu
    for d in hw.devices:
        if not getattr(d, "runtime_context_mb", None):
            d.runtime_context_mb = 512
        if not getattr(d, "runtime_total_mb", None):
            d.runtime_total_mb = int(d.memory_mb)
    ic = prism.InputConfig(
        **rc_mod.served_plan_fields(manifest.get("family"), defaults, manifest),
        height=h, width=w, vae_scale=vs, num_frames=nf, temporal_compression=tc,
        flow=fb_mod.FlowBindings(topo, DEVSTRAL_CONTAINER, manifest.get("model_name")),
        max_context=8192,
    )
    fake_reading = mb_mod.DeviceReading(
        kind="cuda", capacity_mb=81920.0, free_mb=81408.0,
        unified=False, display_active=False, held_by_others_mb=0.0,
        own_context_mb=0.0, measured=True, source="cpu_test",
    )
    with patch(f"neurobrix.{branch}.prism.solver.read_device_sharing", return_value=fake_reading), \
         patch(f"neurobrix.{branch}.prism.memory_budget.read_device_sharing", return_value=fake_reading):
        plan, _cfg = sp_mod.solve_served(container, hw, ic, mode=mode)

    assert plan.strategy is not None
    assert plan.kv_cache_plan is not None
    assert plan.kv_cache_plan.max_cache_len > 0
