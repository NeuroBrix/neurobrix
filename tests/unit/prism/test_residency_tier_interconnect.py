"""Unit tests for P4 Lot 2 — Prism Interconnect-Aware 4-Tier Serving Residency Plan."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.core.state.residency_tier import (
    VALID_RESIDENCY_TIERS,
    ResidencyPlan,
    compute_residency_plan,
)
from neurobrix.pytorch.prism.cpu_config import CPUConfig as PyCPUConfig
from neurobrix.pytorch.prism.solver import (
    ComponentAllocation as PyComponentAllocation,
    ComponentMemory as PyComponentMemory,
    ExecutionPlan as PyExecutionPlan,
)
from neurobrix.pytorch.prism.structure import (
    ClusterNode as PyClusterNode,
    DeviceBrand as PyDeviceBrand,
    DeviceSpec as PyDeviceSpec,
    InterconnectGroup as PyInterconnectGroup,
    InterconnectTech as PyInterconnectTech,
    InterconnectTopology as PyInterconnectTopology,
    InterconnectType as PyInterconnectType,
    NodeLink as PyNodeLink,
    PrismProfile as PyPrismProfile,
)
from neurobrix.triton.prism.cpu_config import CPUConfig as TrCPUConfig
from neurobrix.triton.prism.solver import (
    ComponentAllocation as TrComponentAllocation,
    ComponentMemory as TrComponentMemory,
    ExecutionPlan as TrExecutionPlan,
)
from neurobrix.triton.prism.structure import (
    DeviceBrand as TrDeviceBrand,
    DeviceSpec as TrDeviceSpec,
    InterconnectGroup as TrInterconnectGroup,
    InterconnectTech as TrInterconnectTech,
    InterconnectTopology as TrInterconnectTopology,
    InterconnectType as TrInterconnectType,
    PrismProfile as TrPrismProfile,
)


def _make_device(spec_cls, brand_cls, index: int, memory_mb: int):
    return spec_cls(
        index=index,
        name=f"GPU-{index}",
        brand=brand_cls.NVIDIA,
        architecture="volta",
        compute_capability="7.0",
        memory_mb=memory_mb,
        supports_dtypes=["float32", "float16"],
    )


def _make_profile(
    branch: str,
    gpu_mbs: list[int],
    ram_mb: int = 65536,
    interconnect: str = "pcie",
    bw_gbps: float = 32.0,
    pcie_fallback_gbps: float = 32.0,
    host_measured: dict | None = None,
):
    if branch == "pytorch":
        spec_cls, brand_cls = PyDeviceSpec, PyDeviceBrand
        topo_cls, grp_cls = PyInterconnectTopology, PyInterconnectGroup
        tech_cls, type_cls = PyInterconnectTech, PyInterconnectType
        cpu_cls, prof_cls = PyCPUConfig, PyPrismProfile
    else:
        spec_cls, brand_cls = TrDeviceSpec, TrDeviceBrand
        topo_cls, grp_cls = TrInterconnectTopology, TrInterconnectGroup
        tech_cls, type_cls = TrInterconnectTech, TrInterconnectType
        cpu_cls, prof_cls = TrCPUConfig, TrPrismProfile

    devices = [_make_device(spec_cls, brand_cls, i, mb) for i, mb in enumerate(gpu_mbs)]
    groups = []
    if len(gpu_mbs) > 1 and interconnect == "nvlink":
        groups = [
            grp_cls(
                name="nvlink_0",
                members=list(range(len(gpu_mbs))),
                tech=tech_cls.NVLINK,
                type=type_cls.P2P_DIRECT,
                internal_bandwidth_gbps=bw_gbps,
            )
        ]
    elif len(gpu_mbs) > 1 and interconnect == "xgmi":
        groups = [
            grp_cls(
                name="xgmi_0",
                members=list(range(len(gpu_mbs))),
                tech=tech_cls.XGMI,
                type=type_cls.P2P_DIRECT,
                internal_bandwidth_gbps=bw_gbps,
            )
        ]

    topo = topo_cls(
        groups=groups,
        links=[],
        default_tech=tech_cls.PCIE,
        default_bandwidth_gbps=pcie_fallback_gbps,
    )
    cpu = cpu_cls(
        model="TestCPU",
        cores=16,
        threads=32,
        ram_mb=ram_mb,
        architecture="x86_64",
    )
    return prof_cls(
        id=f"test_{branch}_{interconnect}",
        vendor="nvidia",
        devices=devices,
        topology=topo,
        cpu=cpu,
        host_measured=host_measured,
    )


def _make_plan(
    branch: str,
    weight_mb: float,
    total_mb: float | None = None,
    devices: list[str] | None = None,
    strategy: str = "single_gpu",
    loading_mode: str = "eager",
):
    alloc_cls = PyComponentAllocation if branch == "pytorch" else TrComponentAllocation
    mem_cls = PyComponentMemory if branch == "pytorch" else TrComponentMemory
    plan_cls = PyExecutionPlan if branch == "pytorch" else TrExecutionPlan
    dev_list = devices or ["cuda:0"]
    tot = total_mb if total_mb is not None else weight_mb
    comp_alloc = alloc_cls(
        name="model",
        devices=dev_list,
        dtype="float16",
        memory_mb=tot,
        architecture="volta",
        vendor="nvidia",
        sharded=len(dev_list) > 1,
    )
    comp_mem = mem_cls(
        component_name="model",
        weight_bytes=int(weight_mb * 1024 * 1024),
        activation_bytes=int(max(0.0, tot - weight_mb) * 1024 * 1024),
        overhead_bytes=0,
    )
    return plan_cls(
        components={"model": comp_alloc},
        target_dtype="float16",
        total_memory_mb=tot,
        strategy=strategy,
        component_memory={"model": comp_mem},
        loading_mode=loading_mode,
    )


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
def test_single_gpu_fits_vram_resident(branch: str):
    profile = _make_profile(branch, gpu_mbs=[16384], ram_mb=65536)
    plan = _make_plan(branch, weight_mb=6000.0, total_mb=8000.0, strategy="single_gpu", loading_mode="eager")

    rp = compute_residency_plan(plan, profile)
    assert rp.residency_tier == "vram_resident"
    assert rp.peer_spill_allowed is False
    assert rp.interconnect_bandwidth_gbps == 0.0
    assert rp.prefetch_depth == 0
    assert rp.vram_budget_mb == pytest.approx(16384.0)
    assert rp.host_pinned_budget_mb > 0.0


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
def test_multi_gpu_nvlink_overflow_uses_remote_vram_spill(branch: str):
    profile = _make_profile(
        branch,
        gpu_mbs=[16384, 16384],
        ram_mb=65536,
        interconnect="nvlink",
        bw_gbps=300.0,
        pcie_fallback_gbps=32.0,
    )
    plan = _make_plan(
        branch,
        weight_mb=22000.0,
        total_mb=24000.0,
        devices=["cuda:0"],
        strategy="layer_streaming",
        loading_mode="lazy",
    )

    rp = compute_residency_plan(plan, profile)
    assert rp.residency_tier == "remote_vram_spill"
    assert rp.peer_spill_allowed is True
    assert rp.interconnect_bandwidth_gbps == pytest.approx(300.0)
    assert rp.prefetch_depth >= 1
    assert rp.vram_budget_mb == pytest.approx(32768.0)


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
def test_multi_gpu_xgmi_overflow_uses_remote_vram_spill(branch: str):
    profile = _make_profile(
        branch,
        gpu_mbs=[16384, 16384],
        ram_mb=65536,
        interconnect="xgmi",
        bw_gbps=200.0,
        pcie_fallback_gbps=32.0,
    )
    plan = _make_plan(
        branch,
        weight_mb=20000.0,
        total_mb=22000.0,
        devices=["cuda:0"],
        strategy="zero3",
        loading_mode="lazy",
    )

    rp = compute_residency_plan(plan, profile)
    assert rp.residency_tier == "remote_vram_spill"
    assert rp.peer_spill_allowed is True
    assert rp.interconnect_bandwidth_gbps == pytest.approx(200.0)
    assert rp.prefetch_depth >= 1


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
@pytest.mark.parametrize("pcie_gbps", [16.0, 32.0])
def test_multi_gpu_pcie_only_disallows_peer_spill_and_falls_to_host_pinned(branch: str, pcie_gbps: float):
    profile = _make_profile(
        branch,
        gpu_mbs=[16384, 16384],
        ram_mb=65536,
        interconnect="pcie",
        bw_gbps=pcie_gbps,
        pcie_fallback_gbps=pcie_gbps,
    )
    plan = _make_plan(
        branch,
        weight_mb=22000.0,
        total_mb=24000.0,
        devices=["cuda:0"],
        strategy="layer_streaming",
        loading_mode="lazy",
    )

    rp = compute_residency_plan(plan, profile)
    assert rp.peer_spill_allowed is False
    assert rp.interconnect_bandwidth_gbps == pytest.approx(pcie_gbps)
    assert rp.residency_tier == "host_pinned"
    assert rp.prefetch_depth >= 1


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
def test_weights_exceed_vram_fit_host_pinned(branch: str):
    profile = _make_profile(branch, gpu_mbs=[16384], ram_mb=65536)
    plan = _make_plan(
        branch,
        weight_mb=28000.0,
        total_mb=30000.0,
        devices=["cuda:0"],
        strategy="zero3",
        loading_mode="lazy",
    )

    rp = compute_residency_plan(plan, profile)
    assert rp.residency_tier == "host_pinned"
    assert rp.peer_spill_allowed is False
    assert rp.prefetch_depth >= 1
    assert rp.host_pinned_budget_mb >= 28000.0


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
def test_weights_exceed_vram_and_host_ram_disk_stream(branch: str):
    profile = _make_profile(branch, gpu_mbs=[16384], ram_mb=32768)
    plan = _make_plan(
        branch,
        weight_mb=40000.0,
        total_mb=42000.0,
        devices=["cuda:0"],
        strategy="lazy_sequential",
        loading_mode="lazy",
    )

    rp = compute_residency_plan(plan, profile)
    assert rp.residency_tier == "disk_stream"
    assert rp.peer_spill_allowed is False
    assert rp.prefetch_depth == 0


def test_multinode_qsfp_slow_link_blocks_peer_spill():
    devices = [
        _make_device(PyDeviceSpec, PyDeviceBrand, 0, 16384),
        _make_device(PyDeviceSpec, PyDeviceBrand, 1, 16384),
    ]
    cpu = PyCPUConfig(model="TestCPU", cores=16, threads=32, ram_mb=65536, architecture="x86_64")
    profile = PyPrismProfile(
        id="multinode_qsfp",
        vendor="nvidia",
        devices=devices,
        topology=PyInterconnectTopology(default_bandwidth_gbps=32.0),
        cpu=cpu,
        nodes=[
            PyClusterNode(id="node0", device_indices=[0], cpu=cpu),
            PyClusterNode(id="node1", device_indices=[1], cpu=cpu),
        ],
        node_links=[PyNodeLink(node_a="node0", node_b="node1", bandwidth_gbps=50.0, latency_us=5.0, tech="qsfp")],
    )
    plan = _make_plan("pytorch", weight_mb=22000.0, total_mb=24000.0, strategy="layer_streaming", loading_mode="lazy")

    rp = compute_residency_plan(plan, profile)
    assert rp.interconnect_bandwidth_gbps == pytest.approx(50.0)
    assert rp.peer_spill_allowed is False
    assert rp.residency_tier == "host_pinned"


def test_twin_symmetry_pytorch_vs_triton():
    scenarios = [
        ([16384], 65536, "pcie", 32.0, 6000.0, 8000.0, "single_gpu", "eager"),
        ([16384, 16384], 65536, "nvlink", 300.0, 22000.0, 24000.0, "layer_streaming", "lazy"),
        ([16384, 16384], 65536, "pcie", 32.0, 22000.0, 24000.0, "layer_streaming", "lazy"),
        ([16384], 32768, "pcie", 32.0, 40000.0, 42000.0, "lazy_sequential", "lazy"),
    ]
    for gpus, ram, ic, bw, w_mb, t_mb, strat, mode in scenarios:
        py_rp = compute_residency_plan(
            _make_plan("pytorch", w_mb, t_mb, strategy=strat, loading_mode=mode),
            _make_profile("pytorch", gpus, ram, ic, bw),
        )
        tr_rp = compute_residency_plan(
            _make_plan("triton", w_mb, t_mb, strategy=strat, loading_mode=mode),
            _make_profile("triton", gpus, ram, ic, bw),
        )
        assert py_rp == tr_rp
        assert py_rp.to_dict() == tr_rp.to_dict()


@pytest.mark.parametrize("branch,plan_cls", [("pytorch", PyExecutionPlan), ("triton", TrExecutionPlan)])
def test_execution_plan_serialization_roundtrip_and_backward_compat(tmp_path: Path, branch: str, plan_cls):
    profile = _make_profile(branch, gpu_mbs=[16384, 16384], ram_mb=65536, interconnect="nvlink", bw_gbps=300.0)
    plan = _make_plan(branch, weight_mb=22000.0, total_mb=24000.0, strategy="layer_streaming", loading_mode="lazy")
    plan.residency_plan = compute_residency_plan(plan, profile)

    plan_path = tmp_path / f"plan_{branch}.json"
    plan.save(plan_path, hardware_profile=profile.id)

    loaded = plan_cls.load(plan_path)
    assert loaded.residency_plan is not None
    assert loaded.residency_plan == plan.residency_plan

    # Backward compatibility: older cached plan JSON without residency_plan key
    raw = json.loads(plan_path.read_text())
    raw.pop("residency_plan", None)
    legacy_path = tmp_path / f"legacy_{branch}.json"
    legacy_path.write_text(json.dumps(raw))
    legacy_loaded = plan_cls.load(legacy_path)
    assert legacy_loaded.residency_plan is None


def test_inference_engine_status_and_release_cold_request():
    from neurobrix.serving.engine import InferenceEngine

    def _stub_engine(tier: str):
        eng = object.__new__(InferenceEngine)
        eng.model_name = "test-model"
        eng.hardware_id = "test-hw"
        eng.mode = "triton"
        eng.max_context = None
        eng._family = "llm"
        eng._is_loaded = False
        eng._device = "cuda:0"
        eng._session = None
        eng.one_name = lambda: "test-model"

        rp = ResidencyPlan(
            residency_tier=tier,
            vram_budget_mb=16384.0,
            host_pinned_budget_mb=52428.8,
            peer_spill_allowed=(tier == "remote_vram_spill"),
            interconnect_bandwidth_gbps=(300.0 if tier == "remote_vram_spill" else 0.0),
            prefetch_depth=(1 if tier in ("remote_vram_spill", "host_pinned") else 0),
            reason=f"test {tier}",
        )
        eng._plan = SimpleNamespace(strategy="test_strat", loading_mode="lazy", residency_plan=rp, components={})
        eng._warm_serving = eng._resolve_warm_serving(eng._plan)

        unloaded = []

        class _FakeCompExec:
            def __init__(self):
                self._weights_loaded = True
                self._devices_computed = True
                self._pinned_host_block = object()

            def unload_weights(self):
                unloaded.append("unloaded")

        comp = _FakeCompExec()
        eng._executor = SimpleNamespace(strategy=None, executors={"model": comp})
        return eng, comp, unloaded, rp

    for tier in VALID_RESIDENCY_TIERS:
        eng, comp, unloaded, rp = _stub_engine(tier)
        status = eng.get_status()
        assert status["residency_tier"] == tier
        assert status["residency_plan"] == rp.to_dict()

        eng._release_cold_request()
        if tier == "disk_stream":
            assert eng._warm_serving is False
            assert unloaded == ["unloaded"]
            assert comp._weights_loaded is False
        else:
            assert eng._warm_serving is True
            assert unloaded == []
            assert comp._weights_loaded is True


def test_residency_tier_module_is_zero_torch_and_zero_numpy():
    mod_path = Path(__file__).resolve().parents[3] / "src/neurobrix/core/state/residency_tier.py"
    tree = ast.parse(mod_path.read_text())
    forbidden = {"torch", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden
