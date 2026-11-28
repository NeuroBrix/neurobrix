"""On a heterogeneous multi-GPU machine (e.g. 16 GB + 32 GB), a streamed plan whose
`device_window_mb` is sized for the 32 GB card it places components on must not be
refused by `execution_door` against an unused 16 GB card that holds zero components,
while still refusing when the assigned 32 GB card itself has less free memory than
`device_window_mb`."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _run_hetero_solve(
    branch: str,
    monkeypatch,
    *,
    cuda0_free_mb: float,
    cuda2_free_mb: float,
    window_mb: float = 29000.0,
):
    if branch == "triton":
        import neurobrix.triton.prism.solver as mod
        from neurobrix.triton.prism.cpu_config import CPUConfig
        from neurobrix.triton.prism.structure import (
            DeviceBrand, DeviceSpec, PrismProfile,
        )
        from neurobrix.triton.prism import owed_facts as of_mod
        mode = "triton"
    else:
        import neurobrix.pytorch.prism.solver as mod
        from neurobrix.pytorch.prism.cpu_config import CPUConfig
        from neurobrix.pytorch.prism.structure import (
            DeviceBrand, DeviceSpec, PrismProfile,
        )
        from neurobrix.pytorch.prism import owed_facts as of_mod
        mode = "compiled"

    spec0 = DeviceSpec(
        index=0, name="Tesla V100-SXM2-16GB", memory_mb=16151,
        compute_capability="7.0", supports_dtypes=["float32", "float16", "bfloat16"],
        architecture="volta", brand=DeviceBrand.NVIDIA, unified_memory=False,
    )
    spec2 = DeviceSpec(
        index=2, name="Tesla V100-SXM2-32GB", memory_mb=32501,
        compute_capability="7.0", supports_dtypes=["float32", "float16", "bfloat16"],
        architecture="volta", brand=DeviceBrand.NVIDIA, unified_memory=False,
    )
    profile = PrismProfile(
        id="hetero-16g-32g",
        vendor="nvidia",
        devices=[spec0, spec2],
        cpu=CPUConfig(model="Xeon", cores=16, threads=32, ram_mb=257530, architecture="x86_64"),
    )

    comp = SimpleNamespace(name="dit", get_dominant_dtype=lambda: "bfloat16")

    class _FakeContainer:
        cache_path = None

        def get_manifest(self):
            return {"family": "video"}

        def get_neural_components(self):
            return [comp]

        def get_component(self, name):
            return comp if name == "dit" else None

        def get_graph(self, name):
            return {}

        def get_shard_sizes(self):
            return {}

    solver = mod.PrismSolver()

    def _fake_prepare_devices(_profile):
        return [
            mod.DeviceState(
                device_string="cuda:0", capacity_mb=16151.0, used_mb=0.0,
                live_free_mb=cuda0_free_mb, budget_mb=14000.0,
                recommended_mb=14000.0, spec=spec0,
            ),
            mod.DeviceState(
                device_string="cuda:2", capacity_mb=32501.0, used_mb=0.0,
                live_free_mb=cuda2_free_mb, budget_mb=30000.0,
                recommended_mb=30000.0, spec=spec2,
            ),
        ]

    def _fake_compute_memory(*a, **kw):
        return {
            "dit": mod.ComponentMemory(
                component_name="dit",
                weight_bytes=35 * (1024 ** 3),
                activation_bytes=4 * (1024 ** 3),
                overhead_bytes=0,
            )
        }

    def _fake_layer_streaming(sorted_comps, comp_mem, devices, *a, **kw):
        solver._layer_stream_window_bytes = int(window_mb * 1024 * 1024)
        solver._layer_stream_partitions = {"dit": SimpleNamespace(segments=[SimpleNamespace(first_op="op0", last_op="op1", weight_bytes=10 * 1024**3, weight_names=["w"])], chunks=[], swapped=False, resident=[], total_weight_bytes=35 * 1024**3)}
        solver._layer_stream_graphs = {}
        solver._layer_stream_moe = {}
        solver._layer_stream_tilings = {}
        solver._layer_stream_cost = {"dit": int(window_mb * 1024 * 1024)}
        solver._lm_component_name = None
        # Both cards are in `devices`, while only cuda:2 holds the component.
        return {"dit": ("cuda:2", None)}, list(devices)

    monkeypatch.setattr(of_mod, "measure_owed", lambda *a, **kw: None)
    monkeypatch.setattr(solver, "_prepare_devices", _fake_prepare_devices)
    monkeypatch.setattr(solver, "_compute_memory", _fake_compute_memory)
    for _strat in (
        "_try_single_gpu",
        "_try_single_gpu_lifecycle",
        "_try_component_placement",
        "_try_pipeline_parallel",
        "_try_block_scatter",
        "_try_weight_sharding",
        "_try_lazy_sequential",
        "_try_component_placement_lazy",
        "_try_zero3",
        "_try_op_level_tiling",
        "_try_cpu_execution",
        "_try_cpu_streaming",
    ):
        if hasattr(solver, _strat):
            monkeypatch.setattr(solver, _strat, lambda *a, **kw: None)
    monkeypatch.setattr(solver, "_try_layer_streaming", _fake_layer_streaming)
    monkeypatch.setattr(solver, "_estimate_times", lambda *a, **kw: {})
    monkeypatch.setattr(solver, "_detect_op_level_tiling_pairs", lambda *a, **kw: {})
    monkeypatch.setattr(solver, "_flow_topology", lambda *a, **kw: {})
    monkeypatch.setattr(solver, "_declared_vae_scale", lambda *a, **kw: None)
    monkeypatch.setattr(solver, "_read_lm_config", lambda *a, **kw: None)
    monkeypatch.setattr(solver, "_weight_sizes_by_component", lambda *a, **kw: {"dit": {"w": 35 * (1024 ** 3)}})
    monkeypatch.setattr(solver, "_assign_drain_cliffs", lambda *a, **kw: None)
    monkeypatch.setattr(solver, "_assign_attention_budgets", lambda *a, **kw: None)

    return solver.solve(_FakeContainer(), profile, mod.InputConfig(batch_size=1), mode=mode)


@pytest.mark.parametrize("branch", ["triton", "pytorch"])
def test_execution_door_accepts_streamed_window_on_assigned_32gb_card_ignoring_unused_16gb_card(branch, monkeypatch):
    """cuda:0 (16 GB, 15500 MB free) holds zero components; cuda:2 (32 GB, 31500 MB free)
    holds the streamed component with device_window_mb=29000. The door must stay open."""
    plan = _run_hetero_solve(
        branch, monkeypatch, cuda0_free_mb=15500.0, cuda2_free_mb=31500.0, window_mb=29000.0,
    )
    assert plan.strategy == "layer_streaming"
    assert plan.device_window_mb == pytest.approx(29000.0)
    assert plan.components["dit"].devices == ["cuda:2"]
    assert plan.execution_door is None, plan.execution_door


@pytest.mark.parametrize("branch", ["triton", "pytorch"])
def test_execution_door_still_refuses_when_assigned_card_has_less_than_device_window(branch, monkeypatch):
    """When cuda:2 itself has only 25000 MB free (< 29000 MB device_window_mb), the door
    must still refuse and name cuda:2."""
    plan = _run_hetero_solve(
        branch, monkeypatch, cuda0_free_mb=15500.0, cuda2_free_mb=25000.0, window_mb=29000.0,
    )
    assert plan.strategy == "layer_streaming"
    assert plan.device_window_mb == pytest.approx(29000.0)
    assert plan.components["dit"].devices == ["cuda:2"]
    assert plan.execution_door is not None
    assert plan.execution_door.startswith("cuda:2:"), plan.execution_door
