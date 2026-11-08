"""
NeuroBrix Prism - Loader
YAML Reader for Hardware Profiles with Multi-Interconnect Support

ZERO HARDCODE: All values from hardware profile YAML
"""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Tuple

from neurobrix.core.prism.structure import (
    PrismProfile,
    ClusterNode,
    NodeLink,
    DeviceSpec,
    DeviceBrand,
    InterconnectTopology,
    InterconnectGroup,
    InterconnectLink,
    InterconnectTech,
    InterconnectType,
)
from neurobrix.core.prism.cpu_config import CPUConfig

# Project Root Resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
#: The COMMITTED profiles, shipped with the package.
HARDWARE_DIR = PROJECT_ROOT / "config" / "hardware"
#: The machine's GENERATED profiles (`default`, `default-<tag>`), ONE place per machine beside the measured-facts store,
#: never inside a source tree, where each tree kept its own copy (the team's review, 2026-11-02).
from neurobrix.core.paths import engine_home as _engine_home
GENERATED_DIR = _engine_home() / "hardware"


def is_generated(hardware_id: str) -> bool:
    """A name the engine generates (`default`, `default-<tag>`): it lives in GENERATED_DIR, never in the package."""
    return hardware_id == "default" or str(hardware_id).startswith("default-")


def profile_path(hardware_id: str) -> Path:
    """THE file of a hardware profile: a generated one in GENERATED_DIR, a committed one in HARDWARE_DIR. A generated
    name found in the package is refused by name — two copies of one machine's facts, the one read decided by where
    the reader runs, is what this place exists to end."""
    if is_generated(hardware_id):
        stale = HARDWARE_DIR / f"{hardware_id}.yml"
        if stale.exists():
            raise RuntimeError(
                f"ZERO FALLBACK: a generated hardware profile sits in the package ({stale}); generated profiles live "
                f"in {GENERATED_DIR}, one per machine — move it there (or delete it: the engine regenerates it at the "
                f"next launch without --hardware)")
        return GENERATED_DIR / f"{hardware_id}.yml"
    return HARDWARE_DIR / f"{hardware_id}.yml"


def list_available_profiles() -> List[str]:
    """List all available hardware profile names: the committed ones and this machine's generated ones."""
    names = set()
    for d in (HARDWARE_DIR, GENERATED_DIR):
        if d.exists():
            # Filter out macOS hidden files (._*)
            names.update(p.stem for p in d.glob("*.yml") if not p.stem.startswith("._"))
    return sorted(names)


def _parse_brand(brand_str: str) -> DeviceBrand:
    """Parse brand string to enum."""
    brand_map = {
        "nvidia": DeviceBrand.NVIDIA,
        "amd": DeviceBrand.AMD,
        "intel": DeviceBrand.INTEL,
        "tenstorrent": DeviceBrand.TENSTORRENT,
        "apple": DeviceBrand.APPLE,
    }
    brand = brand_map.get(brand_str.lower())
    if brand is None:
        raise RuntimeError(f"ZERO FALLBACK: Unknown GPU brand '{brand_str}' in hardware profile")
    return brand


def _parse_interconnect_tech(tech_str: str) -> InterconnectTech:
    """Parse interconnect technology string to enum."""
    tech_map = {
        "nvlink": InterconnectTech.NVLINK,
        "xgmi": InterconnectTech.XGMI,
        "qsfp": InterconnectTech.QSFP,
        "ualink": InterconnectTech.UALINK,
        "cxl": InterconnectTech.CXL,
        "pcie": InterconnectTech.PCIE,
    }
    return tech_map.get(tech_str.lower(), InterconnectTech.PCIE)


def _parse_interconnect_type(type_str: str) -> InterconnectType:
    """Parse interconnect type string to enum."""
    type_map = {
        "p2p_direct": InterconnectType.P2P_DIRECT,
        "p2p_bridge": InterconnectType.P2P_BRIDGE,
        "shared_memory": InterconnectType.SHARED_MEMORY,
        "cpu_bounce": InterconnectType.CPU_BOUNCE,
        # Legacy mappings for backward compatibility
        "nvlink_p2p": InterconnectType.P2P_DIRECT,
        "full_mesh": InterconnectType.P2P_DIRECT,  # Full mesh = all-to-all P2P
        "pcie": InterconnectType.CPU_BOUNCE,
    }
    return type_map.get(type_str.lower(), InterconnectType.CPU_BOUNCE)


def _is_full_mesh(type_str: str) -> bool:
    """Check if type indicates full mesh connectivity."""
    return type_str.lower() == "full_mesh"


def _parse_interconnect_group(group_data: dict) -> InterconnectGroup:
    """Parse a single interconnect group from YAML."""
    return InterconnectGroup(
        name=group_data.get("name", "unnamed"),
        members=group_data.get("members", []),
        tech=_parse_interconnect_tech(group_data.get("tech", "pcie")),
        type=_parse_interconnect_type(group_data.get("type", "cpu_bounce")),
        internal_bandwidth_gbps=group_data.get("bandwidth_gbps", 32.0),
    )


def _parse_interconnect_link(link_data: dict) -> InterconnectLink:
    """Parse a single interconnect link from YAML."""
    return InterconnectLink(
        device_a=link_data["device_a"],
        device_b=link_data["device_b"],
        tech=_parse_interconnect_tech(link_data.get("tech", "pcie")),
        type=_parse_interconnect_type(link_data.get("type", "cpu_bounce")),
        bandwidth_gbps=link_data.get("bandwidth_gbps", 32.0),
        bidirectional=link_data.get("bidirectional", True),
    )


def _generate_full_mesh_links(
    members: List[int],
    tech: InterconnectTech,
    bandwidth_gbps: float,
) -> List[InterconnectLink]:
    """
    Generate all pairwise links for a full mesh group.

    Full mesh: Every GPU can communicate directly with every other GPU.
    For N GPUs, generates N*(N-1)/2 bidirectional links.

    Example: 4 GPUs [0,1,2,3] generates:
    0↔1, 0↔2, 0↔3, 1↔2, 1↔3, 2↔3 (6 links)
    """
    links = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            links.append(InterconnectLink(
                device_a=members[i],
                device_b=members[j],
                tech=tech,
                type=InterconnectType.P2P_DIRECT,
                bandwidth_gbps=bandwidth_gbps,
                bidirectional=True,
            ))
    return links


def _parse_interconnect(data: dict) -> InterconnectTopology:
    """
    Parse interconnect topology from YAML.

    Supports both simple and full topology formats.
    Also parses pcie_fallback for default bandwidth.
    """
    interconnect = data.get("interconnect", {})
    pcie_fallback = data.get("pcie_fallback", {})

    # Get PCIe fallback values
    pcie_bandwidth = pcie_fallback.get("bandwidth_gbps", 32.0)  # PCIe 4.0 x16 default

    # V2 format: full topology
    if "topology" in interconnect:
        topo_data = interconnect["topology"]
        groups = [_parse_interconnect_group(g) for g in topo_data.get("groups", [])]
        links = [_parse_interconnect_link(l) for l in topo_data.get("links", [])]
        return InterconnectTopology(
            groups=groups,
            links=links,
            default_tech=_parse_interconnect_tech(topo_data.get("default_tech", "pcie")),
            default_bandwidth_gbps=topo_data.get("default_bandwidth_gbps", pcie_bandwidth),
        )

    # V1 format: simple groups (backward compatibility)
    groups = []
    all_links = []

    for i, group_data in enumerate(interconnect.get("groups", [])):
        # V1 had members + type only
        if "name" not in group_data:
            group_data["name"] = f"group_{i}"

        # Get type before processing
        v1_type = group_data.get("type", "pcie")

        if "tech" not in group_data:
            # Infer tech from type
            if "nvlink" in v1_type.lower():
                group_data["tech"] = "nvlink"
            elif "xgmi" in v1_type.lower():
                group_data["tech"] = "xgmi"
            else:
                group_data["tech"] = "pcie"

        if "bandwidth_gbps" not in group_data:
            group_data["bandwidth_gbps"] = interconnect.get("bandwidth_gbps", 32.0)

        group = _parse_interconnect_group(group_data)
        groups.append(group)

        # For full_mesh groups, generate explicit pairwise links
        if _is_full_mesh(v1_type):
            mesh_links = _generate_full_mesh_links(
                members=group.members,
                tech=group.tech,
                bandwidth_gbps=group.internal_bandwidth_gbps,
            )
            all_links.extend(mesh_links)

    return InterconnectTopology(
        groups=groups,
        links=all_links,
        default_tech=InterconnectTech.PCIE,
        default_bandwidth_gbps=pcie_bandwidth,
    )


def _vendor_block(d_data: dict, block: str) -> Optional[dict]:
    """A device's `measured:` / the machine's `host_measured:` block read from the TRACKED vendor file of its
    architecture (config/vendors/<brand>/<architecture>.yml) when the hardware profile carries none — a machine whose
    profile is auto-generated (the Mac's `auto-apple-m4-pro-18g`, untracked) keeps its measured facts with their
    source under git all the same (point B, 2026-10-25). None when the vendor file has no such block or no vendor file
    describes the architecture: nothing is invented."""
    from neurobrix.core.config.loader import vendor_config_for_device, UnsupportedArchitectureError
    brand, arch = d_data.get("brand", "nvidia"), d_data.get("architecture")
    cc = d_data.get("compute_capability")
    if not arch and not cc and not d_data.get("model"):
        return None
    try:
        # the device's own vendor file, by the kernels' rule: the chip file (compute capability) before the family
        return (vendor_config_for_device(str(brand), arch, cc, model=d_data.get("model")) or {}).get(block)
    except UnsupportedArchitectureError:
        return None


def _over(under, over):
    """`over`'s facts over `under`'s, key by key, into nested tables (weight_reload_gbps: {engine: {stored: ...}})."""
    if not (isinstance(under, dict) and isinstance(over, dict)):
        return over if over is not None else under
    return {**under, **{k: _over(under.get(k), v) for k, v in over.items()}}


def _refuse_unified_without_largest_allocation(spec: DeviceSpec, path) -> None:
    """A unified device's profile carries `max_allocation_mb` (the largest single allocation its backend grants, asked
    of the device at generation: Metal's MTLDevice.maxBufferLength, Level Zero's maxMemAllocSize), or the profile is
    refused by name. Silence would read as "no limit below memory_mb" — true of a CUDA card, false of the unified
    devices this engine drives, whose single buffer is capped below their working set (13 639 of 18 186 MB on an
    M4 Pro). The Mac, 2026-11-08: a profile regenerated without the key let a 13 912.6 MiB arena through to
    Open-Sora-v2's Triton cell, which failed at weight loading. Keyed on the device's own answer
    (`DeviceSpec.has_unified_memory`: its `unified_memory` fact, else its architecture's), never a brand — a profile
    unified only through its architecture (an older hand-written `apple_silicon` one) is asked the same."""
    if not spec.has_unified_memory or spec.max_allocation_mb:
        return
    raise ValueError(
        f"ZERO FALLBACK: {path} describes the unified GPU '{spec.name}' without max_allocation_mb (its largest single "
        f"allocation: Metal's MTLDevice.maxBufferLength, Level Zero's maxMemAllocSize): Prism would plan arenas the "
        f"device refuses. Remove this generated profile and run again: it is regenerated from the device, which is "
        f"asked for the fact (on a Mac, `python -m nbxtensor install-metal` first if pyobjc-framework-Metal is "
        f"missing).")


def load_profile(hardware_id: str) -> PrismProfile:
    """
    Load hardware profile from YAML file.

    ZERO HARDCODE: All values from YAML.
    """
    path = profile_path(hardware_id)

    if not path.exists():
        available = list_available_profiles()
        if available:
            profiles_list = "\n".join(f"  - {p}" for p in available)
            msg = (
                f"Hardware profile '{hardware_id}' not found.\n"
                f"\n"
                f"Available profiles:\n"
                f"{profiles_list}"
            )
        else:
            msg = (
                f"Hardware profile '{hardware_id}' not found.\n"
                f"No profiles found in {HARDWARE_DIR}/ or {GENERATED_DIR}/"
            )
        raise FileNotFoundError(msg)

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Parse Devices
    devices = []
    for d_data in data.get("devices", []):
        devices.append(DeviceSpec(
            index=d_data.get("index", len(devices)),
            name=d_data.get("model", d_data.get("name", "Unknown GPU")),
            memory_mb=d_data["memory_mb"],
            compute_capability=d_data.get("compute_capability", "0.0"),
            supports_dtypes=d_data.get("supports_dtypes", ["float32"]),
            architecture=d_data.get("architecture", "unknown"),
            brand=_parse_brand(d_data.get("brand", "nvidia")),
            # None when the profile is silent, so DeviceSpec falls back to
            # the architecture mapping rather than assuming "discrete".
            unified_memory=d_data.get("unified_memory"),
            # None when the profile is silent: no single-allocation limit below the device's memory.
            max_allocation_mb=d_data.get("max_allocation_mb"),
            runtime_total_mb=d_data.get("runtime_total_mb"),
            runtime_context_mb=d_data.get("runtime_context_mb"),
            measured=d_data.get("measured") or _vendor_block(d_data, "measured"),
        ))
        _refuse_unified_without_largest_allocation(devices[-1], path)

    # Parse Interconnect Topology
    topology = _parse_interconnect(data)

    # Parse CPU section
    cpu_config = None
    cpu_data = data.get("cpu")
    if cpu_data:
        cpu_config = CPUConfig.from_yaml_dict(cpu_data)

    # Step 6: optional machines and links; absent = the one-node case (PrismProfile.__post_init__)
    nodes = [ClusterNode(id=str(n["id"]), device_indices=[int(i) for i in n["devices"]],
                         cpu=CPUConfig.from_yaml_dict(n["cpu"]) if n.get("cpu") else None)
             for n in (data.get("nodes") or [])]
    node_links = [NodeLink(node_a=str(l["between"][0]), node_b=str(l["between"][1]),
                           bandwidth_gbps=float(l["bandwidth_gbps"]), latency_us=float(l["latency_us"]),
                           tech=str(l.get("tech", "")))
                  for l in (data.get("node_links") or [])]

    from neurobrix.core.workspace import declare_io_workers
    declare_io_workers(cpu_config.io_workers if cpu_config else None)
    return PrismProfile(
        id=data["id"],
        vendor=data.get("vendor", "unknown"),
        devices=devices,
        topology=topology,
        cpu=cpu_config,
        preferred_dtype=data.get("preferred_dtype", None),
        # the vendor's block (the architecture's facts) under the profile's (this machine's, measured by the engine), key by
        # key, into its tables (weight_reload_gbps: {engine: {stored}}) — the Dell's 1.12 rule, 2026-11-03 22:54
        host_measured=_over(next((_vendor_block(d, "host_measured") for d in data.get("devices", [])
                                  if _vendor_block(d, "host_measured")), None), data.get("host_measured")) or None,
        nodes=nodes,
        node_links=node_links,
        refused_devices=list(data.get("refused_devices") or []),
    )
