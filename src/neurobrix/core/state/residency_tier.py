"""Interconnect-aware 4-tier serving residency policy for Prism.

Prism is the single decision-maker for serving weight/KV residency. Reading the
hardware profile's declared topology (`interconnect` vs `pcie_fallback`, plus
multi-node `node_links`), VRAM budgets, and host RAM budget (`host_measured` /
`cpu.ram_mb`), this module emits a pure, torch-free `ResidencyPlan`:

1. `vram_resident`     — weights + active KV fit eagerly in assigned GPU VRAM.
2. `remote_vram_spill` — local card overflows, multi-GPU peers share a fast
                         interconnect (>= peer spill threshold derived from
                         `pcie_fallback`, e.g. NVLink/xGMI >= 100 GB/s), and
                         peer VRAM can absorb the overflow.
3. `host_pinned`       — weights exceed usable VRAM (or peer spill is disabled
                         on PCIe-only / single-GPU) but fit within the host
                         pinned RAM budget; async prefetch double-buffering is
                         enabled (`prefetch_depth >= 1`).
4. `disk_stream`       — neither VRAM nor host RAM budget suffices; serving
                         releases cold request buffers after each request.

Pure Python standard library only (zero Torch, zero NumPy, zero model-name
hardcoding) so both `pytorch/prism` and `triton/prism` share one authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

VALID_RESIDENCY_TIERS: Tuple[str, ...] = (
    "vram_resident",
    "remote_vram_spill",
    "host_pinned",
    "disk_stream",
)
WARM_RESIDENCY_TIERS: Tuple[str, ...] = (
    "vram_resident",
    "remote_vram_spill",
    "host_pinned",
)
_FAST_PEER_FLOOR_GBPS: float = 100.0
_OFFLOAD_STRATEGY_MARKERS: Tuple[str, ...] = ("zero3", "offload", "stream", "lazy")


@dataclass(frozen=True)
class ResidencyPlan:
    """Structured 4-tier serving residency decision emitted by Prism."""
    residency_tier: str
    vram_budget_mb: float
    host_pinned_budget_mb: float
    peer_spill_allowed: bool
    interconnect_bandwidth_gbps: float
    prefetch_depth: int
    reason: str = ""

    def __post_init__(self) -> None:
        if self.residency_tier not in VALID_RESIDENCY_TIERS:
            raise ValueError(
                f"Invalid residency_tier {self.residency_tier!r}; "
                f"expected one of {VALID_RESIDENCY_TIERS}"
            )
        if self.prefetch_depth < 0:
            raise ValueError(f"prefetch_depth must be >= 0, got {self.prefetch_depth}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residency_tier": self.residency_tier,
            "vram_budget_mb": round(float(self.vram_budget_mb), 2),
            "host_pinned_budget_mb": round(float(self.host_pinned_budget_mb), 2),
            "peer_spill_allowed": bool(self.peer_spill_allowed),
            "interconnect_bandwidth_gbps": round(float(self.interconnect_bandwidth_gbps), 2),
            "prefetch_depth": int(self.prefetch_depth),
            "reason": str(self.reason),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ResidencyPlan"]:
        if data is None:
            return None
        return cls(
            residency_tier=str(data["residency_tier"]),
            vram_budget_mb=float(data.get("vram_budget_mb", 0.0)),
            host_pinned_budget_mb=float(data.get("host_pinned_budget_mb", 0.0)),
            peer_spill_allowed=bool(data.get("peer_spill_allowed", False)),
            interconnect_bandwidth_gbps=float(data.get("interconnect_bandwidth_gbps", 0.0)),
            prefetch_depth=int(data.get("prefetch_depth", 0)),
            reason=str(data.get("reason", "")),
        )


def _parse_device_index(device_str: str) -> Optional[int]:
    """Extract integer GPU index from a device string like 'cuda:0', or None for CPU."""
    text = str(device_str or "").strip().lower()
    if not text or text == "cpu" or ":" not in text:
        return None
    suffix = text.rsplit(":", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _collect_vram_budgets(
    plan: Any, profile: Any, devices: Optional[Sequence[Any]] = None
) -> Tuple[Dict[int, float], List[int], List[int]]:
    """Return (per_device_vram_mb, all_gpu_indices, assigned_gpu_indices)."""
    per_device: Dict[int, float] = {}
    for spec in getattr(profile, "devices", None) or ():
        idx = int(spec.index)
        per_device[idx] = float(getattr(spec, "memory_mb", 0.0) or 0.0)
    for dev in devices or ():
        spec = getattr(dev, "spec", None)
        idx = int(spec.index) if spec is not None else _parse_device_index(getattr(dev, "device_string", ""))
        if idx is None:
            continue
        budget = float(
            getattr(dev, "budget_mb", 0.0)
            or getattr(dev, "capacity_mb", 0.0)
            or (getattr(spec, "memory_mb", 0.0) if spec is not None else 0.0)
        )
        if budget > 0.0:
            per_device[idx] = budget

    all_gpus = sorted(per_device)
    assigned_set = set()
    for alloc in (getattr(plan, "components", None) or {}).values():
        dev_list = getattr(alloc, "devices", None) or [getattr(alloc, "device", "")]
        for d_str in dev_list:
            idx = _parse_device_index(d_str)
            if idx is not None and idx in per_device:
                assigned_set.add(idx)
    if not assigned_set and all_gpus:
        assigned_set.add(all_gpus[0])
    return per_device, all_gpus, sorted(assigned_set)


def _node_for_device(profile: Any, dev_idx: int) -> Optional[str]:
    """Return the ClusterNode id owning `dev_idx`, if the profile declares nodes."""
    for node in getattr(profile, "nodes", None) or ():
        if dev_idx in (getattr(node, "device_indices", None) or ()):
            return str(node.id)
    return None


def _pairwise_link_info(profile: Any, dev_a: int, dev_b: int) -> Tuple[float, bool]:
    """Return (bandwidth_gbps, is_direct_fast_type) for a device pair."""
    node_a = _node_for_device(profile, dev_a)
    node_b = _node_for_device(profile, dev_b)
    if node_a and node_b and node_a != node_b:
        for link in getattr(profile, "node_links", None) or ():
            ends = (str(link.node_a), str(link.node_b))
            if (node_a, node_b) == ends or (node_b, node_a) == ends:
                return float(link.bandwidth_gbps), True
        return 0.0, False

    topology = getattr(profile, "topology", None)
    if topology is None:
        return 0.0, False
    bw = float(topology.get_bandwidth(dev_a, dev_b))
    itype = topology.get_interconnect_type(dev_a, dev_b)
    itype_name = str(getattr(itype, "value", itype)).lower()
    return bw, itype_name != "cpu_bounce"


def _assess_interconnect(profile: Any, gpu_indices: Sequence[int]) -> Tuple[float, bool]:
    """Compute (min_pairwise_bandwidth_gbps, peer_spill_allowed) across GPU pool."""
    if len(gpu_indices) <= 1:
        return 0.0, False

    topology = getattr(profile, "topology", None)
    pcie_bw = float(getattr(topology, "default_bandwidth_gbps", 0.0) or 0.0)
    peer_threshold = max(_FAST_PEER_FLOOR_GBPS, 2.0 * pcie_bw) if pcie_bw > 0.0 else _FAST_PEER_FLOOR_GBPS

    min_bw = float("inf")
    all_fast_type = True
    for i in range(len(gpu_indices)):
        for j in range(i + 1, len(gpu_indices)):
            bw, fast_type = _pairwise_link_info(profile, gpu_indices[i], gpu_indices[j])
            if bw < min_bw:
                min_bw = bw
            if not fast_type:
                all_fast_type = False

    if min_bw == float("inf"):
        return 0.0, False
    peer_allowed = all_fast_type and (min_bw >= peer_threshold) and (min_bw > pcie_bw)
    return round(min_bw, 2), peer_allowed


def _host_pinned_budget_mb(plan: Any, profile: Any) -> float:
    """Derive usable host pinned RAM budget in MB from profile host facts and CPU RAM."""
    host_measured = getattr(profile, "host_measured", None)
    if isinstance(host_measured, dict):
        explicit = float(host_measured.get("host_pinned_budget_mb") or 0.0)
        if explicit > 0.0:
            return round(explicit, 2)
        available = float(host_measured.get("available_mb") or 0.0)
        if available > 0.0:
            return round(available * 0.85, 2)

    cpu = getattr(profile, "cpu", None)
    ram_mb = float(getattr(cpu, "ram_mb", 0.0) or 0.0) if cpu is not None else 0.0
    if ram_mb <= 0.0:
        ram_mb = float(getattr(plan, "cpu_ram_mb", 0.0) or 0.0)
    if ram_mb <= 0.0:
        return 0.0

    runtime_bases = getattr(cpu, "runtime_base_mb", None) if cpu is not None else None
    base_mb = float(max(runtime_bases.values())) if isinstance(runtime_bases, dict) and runtime_bases else 0.0
    return round(max(0.0, min(ram_mb * 0.80, ram_mb - base_mb)), 2)


def _plan_memory_demand_mb(plan: Any) -> Tuple[float, float]:
    """Return (required_vram_mb, weight_mb) for the plan."""
    comp_mem = getattr(plan, "component_memory", None) or {}
    weight_mb = sum(float(getattr(m, "weight_mb", 0.0) or 0.0) for m in comp_mem.values())
    kv_plan = getattr(plan, "kv_cache_plan", None)
    kv_mb = float(getattr(kv_plan, "memory_mb", 0.0) or 0.0) if kv_plan is not None else 0.0
    total_mb = float(getattr(plan, "total_memory_mb", 0.0) or 0.0)
    if weight_mb <= 0.0:
        weight_mb = total_mb
    required_vram_mb = max(total_mb, weight_mb + kv_mb)
    return required_vram_mb, weight_mb


def _compute_prefetch_depth(
    tier: str, overflow_mb: float, vram_budget_mb: float, host_budget_mb: float, bw_gbps: float
) -> int:
    """Scale async prefetch depth cleanly by tier and available memory headroom."""
    if tier in ("vram_resident", "disk_stream"):
        return 0
    denom = max(float(overflow_mb), 1.0)
    if tier == "remote_vram_spill":
        headroom = float(vram_budget_mb) / denom
        return min(4, 1 + int(headroom >= 2.0) + int(bw_gbps >= 200.0 and headroom >= 1.5))
    headroom = float(host_budget_mb) / denom
    return min(4, 1 + int(headroom >= 2.0) + int(headroom >= 4.0))


def compute_residency_plan(
    plan: Any, profile: Any, devices: Optional[Sequence[Any]] = None
) -> ResidencyPlan:
    """Compute Prism's interconnect-aware 4-tier ResidencyPlan for `plan` on `profile`."""
    per_dev_vram, all_gpus, assigned_gpus = _collect_vram_budgets(plan, profile, devices)
    vram_budget_mb = sum(per_dev_vram.values())
    assigned_vram_mb = sum(per_dev_vram[i] for i in assigned_gpus if i in per_dev_vram)
    bw_gbps, peer_allowed = _assess_interconnect(profile, all_gpus)
    host_budget_mb = _host_pinned_budget_mb(plan, profile)
    required_vram_mb, weight_mb = _plan_memory_demand_mb(plan)

    loading_mode = str(getattr(plan, "loading_mode", "lazy") or "lazy").lower()
    strategy = str(getattr(plan, "strategy", "") or "").lower()
    is_offload = (loading_mode != "eager") or any(m in strategy for m in _OFFLOAD_STRATEGY_MARKERS)

    if assigned_vram_mb > 0.0 and required_vram_mb <= assigned_vram_mb and not is_offload:
        return ResidencyPlan(
            residency_tier="vram_resident",
            vram_budget_mb=vram_budget_mb,
            host_pinned_budget_mb=host_budget_mb,
            peer_spill_allowed=peer_allowed,
            interconnect_bandwidth_gbps=bw_gbps,
            prefetch_depth=0,
            reason=f"weights+kv ({required_vram_mb:.0f} MB) fit assigned VRAM ({assigned_vram_mb:.0f} MB)",
        )

    overflow_mb = max(0.0, required_vram_mb - assigned_vram_mb)
    if peer_allowed and vram_budget_mb >= required_vram_mb and (vram_budget_mb > assigned_vram_mb or is_offload):
        depth = _compute_prefetch_depth("remote_vram_spill", overflow_mb, vram_budget_mb, host_budget_mb, bw_gbps)
        return ResidencyPlan(
            residency_tier="remote_vram_spill",
            vram_budget_mb=vram_budget_mb,
            host_pinned_budget_mb=host_budget_mb,
            peer_spill_allowed=True,
            interconnect_bandwidth_gbps=bw_gbps,
            prefetch_depth=depth,
            reason=f"local VRAM overflows by {overflow_mb:.0f} MB; peer VRAM ({vram_budget_mb:.0f} MB) on {bw_gbps:.0f} GB/s link absorbs spill",
        )

    if host_budget_mb > 0.0 and weight_mb <= host_budget_mb:
        depth = _compute_prefetch_depth("host_pinned", overflow_mb or weight_mb, vram_budget_mb, host_budget_mb, bw_gbps)
        return ResidencyPlan(
            residency_tier="host_pinned",
            vram_budget_mb=vram_budget_mb,
            host_pinned_budget_mb=host_budget_mb,
            peer_spill_allowed=peer_allowed,
            interconnect_bandwidth_gbps=bw_gbps,
            prefetch_depth=depth,
            reason=f"weights ({weight_mb:.0f} MB) exceed usable VRAM ({assigned_vram_mb:.0f} MB) but fit host pinned budget ({host_budget_mb:.0f} MB)",
        )

    return ResidencyPlan(
        residency_tier="disk_stream",
        vram_budget_mb=vram_budget_mb,
        host_pinned_budget_mb=host_budget_mb,
        peer_spill_allowed=peer_allowed,
        interconnect_bandwidth_gbps=bw_gbps,
        prefetch_depth=0,
        reason=f"weights ({weight_mb:.0f} MB) exceed both VRAM ({vram_budget_mb:.0f} MB) and host pinned budget ({host_budget_mb:.0f} MB)",
    )
