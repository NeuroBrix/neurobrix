"""
NeuroBrix Prism - Loader
YAML Reader for Hardware Profiles with Multi-Interconnect Support

ZERO HARDCODE: All values from hardware profile YAML
"""

import yaml
from pathlib import Path
from typing import List, Optional, Tuple

from neurobrix.core.prism.structure import (
    PrismProfile,
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
HARDWARE_DIR = PROJECT_ROOT / "config" / "hardware"


def list_available_profiles() -> List[str]:
    """List all available hardware profile names."""
    if not HARDWARE_DIR.exists():
        return []
    # Filter out macOS hidden files (._*)
    return sorted([p.stem for p in HARDWARE_DIR.glob("*.yml") if not p.stem.startswith("._")])


def _parse_brand(brand_str: str) -> DeviceBrand:
    """Parse brand string to enum."""
    brand_map = {
        "nvidia": DeviceBrand.NVIDIA,
        "amd": DeviceBrand.AMD,
        "intel": DeviceBrand.INTEL,
        "tenstorrent": DeviceBrand.TENSTORRENT,
    }
    return brand_map.get(brand_str.lower(), DeviceBrand.NVIDIA)


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


def load_profile(hardware_id: str) -> PrismProfile:
    """
    Load hardware profile from YAML file.

    ZERO HARDCODE: All values from YAML.
    """
    path = HARDWARE_DIR / f"{hardware_id}.yml"

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
                f"No profiles found in {HARDWARE_DIR}/"
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
        ))

    # Parse Interconnect Topology
    topology = _parse_interconnect(data)

    # Parse CPU section
    cpu_config = None
    cpu_data = data.get("cpu")
    if cpu_data:
        cpu_config = CPUConfig.from_yaml_dict(cpu_data)

    return PrismProfile(
        id=data["id"],
        vendor=data.get("vendor", "unknown"),
        devices=devices,
        topology=topology,
        cpu=cpu_config,
        preferred_dtype=data.get("preferred_dtype", None),
    )
