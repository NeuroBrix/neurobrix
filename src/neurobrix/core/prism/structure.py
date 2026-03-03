"""
NeuroBrix Prism - Structure
Multi-Interconnect Topology Support for Heterogeneous Hardware

ZERO HARDCODE: All values from hardware profile YAML
VENDORLESS: nvidia/amd/intel → cuda/hip/xpu device strings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from neurobrix.core.prism.cpu_config import CPUConfig


# =============================================================================
# ENUMS
# =============================================================================

class DeviceBrand(Enum):
    """Vendorless GPU brand enumeration."""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    TENSTORRENT = "tenstorrent"

    def to_device_prefix(self) -> str:
        """Convert brand to PyTorch device prefix."""
        mapping = {
            DeviceBrand.NVIDIA: "cuda",
            DeviceBrand.AMD: "hip",
            DeviceBrand.INTEL: "xpu",
            DeviceBrand.TENSTORRENT: "tt",
        }
        return mapping[self]


class InterconnectTech(Enum):
    """Physical interconnect technology."""
    NVLINK = "nvlink"       # NVIDIA NVLink (all generations)
    XGMI = "xgmi"           # AMD Infinity Fabric
    QSFP = "qsfp"           # Standard high-speed optical
    UALINK = "ualink"       # Open standard (future)
    CXL = "cxl"             # Compute Express Link
    PCIE = "pcie"           # PCI Express


class InterconnectType(Enum):
    """Logical interconnect capability."""
    P2P_DIRECT = "p2p_direct"           # Direct peer-to-peer (NVLink, xGMI)
    P2P_BRIDGE = "p2p_bridge"           # Through bridge/switch
    SHARED_MEMORY = "shared_memory"     # Shared memory pool (NVSwitch)
    CPU_BOUNCE = "cpu_bounce"           # Through CPU (slow)


class AllocationStrategy(Enum):
    """
    Execution strategy for model allocation across hardware.

    Prism scores ALL strategies and selects the best viable one for the
    given hardware profile. Every strategy listed here is a first-class
    citizen — NeuroBrix is universal and must handle any hardware combination.

    === Granularity Hierarchy ===
    Single GPU → Component Placement → Pipeline Parallel (layer) → Block Scatter → Weight Sharding

    === Interconnect ===
    Scoring uses bandwidth_gbps (continuous value), NOT technology name.
    No _nvlink/_pcie suffixes — interconnect speed is a scoring factor, not a strategy variant.

    === Memory Modes ===
    Eager: weights stay in VRAM permanently (fast serving)
    Lazy: weights swap in/out per phase (models too large for combined VRAM)
    Offload: weights on CPU, stream to GPU for compute (very large models)
    """

    # === Single Device ===
    SINGLE_GPU = "single_gpu"                       # All components on 1 GPU
    SINGLE_GPU_LIFECYCLE = "single_gpu_lifecycle"   # Persistent + transient classification on 1 GPU

    # === Component Placement — whole-component distribution ===
    # Each component (text_encoder, transformer, vae) on its assigned GPU.
    # Activations transfer between GPUs at component boundaries.
    COMPONENT_PLACEMENT = "component_placement"
    COMPONENT_PLACEMENT_LAZY = "component_placement_lazy"  # + lazy weight swap between phases

    # === Pipeline Parallel — per-layer sequential fill ===
    # Layers distributed sequentially across GPUs (like Accelerate device_map="auto").
    # Layers 0..N on GPU0, N+1..M on GPU1, etc. Only N-1 boundary crossings for N GPUs.
    # Needed when a component exceeds single GPU but layers are sequential.
    PIPELINE_PARALLEL = "pipeline_parallel"

    # === Block Scatter — block-level best-fit distribution ===
    # Individual transformer blocks distributed across GPUs using best-fit-decreasing.
    # Blocks can land on any GPU (not necessarily sequential).
    # Needed when a single component exceeds any single GPU's VRAM.
    BLOCK_SCATTER = "block_scatter"

    # === Weight Sharding — weight-file-level round-robin ===
    # Single component's weight files distributed across GPUs.
    # CompiledSequence handles cross-device execution with automatic device alignment.
    WEIGHT_SHARDING = "weight_sharding"

    # === Sequential / Offload ===
    LAZY_SEQUENTIAL = "lazy_sequential"             # One component at a time on largest GPU
    ZERO3 = "zero3"                                 # CPU offload, stream to GPU for compute

    @property
    def is_eager(self) -> bool:
        """True if weights stay in VRAM permanently (fast serving)."""
        return self in _EAGER_STRATEGIES

    @property
    def is_multi_gpu(self) -> bool:
        """True if strategy distributes across multiple GPUs."""
        return self not in (
            AllocationStrategy.SINGLE_GPU,
            AllocationStrategy.SINGLE_GPU_LIFECYCLE,
            AllocationStrategy.LAZY_SEQUENTIAL,
        )

    @property
    def granularity(self) -> str:
        """Distribution granularity: 'single', 'component', 'layer', 'block', 'weight_file'."""
        if self in (AllocationStrategy.SINGLE_GPU, AllocationStrategy.SINGLE_GPU_LIFECYCLE,
                    AllocationStrategy.LAZY_SEQUENTIAL, AllocationStrategy.ZERO3):
            return "single"
        if self in (AllocationStrategy.COMPONENT_PLACEMENT,
                    AllocationStrategy.COMPONENT_PLACEMENT_LAZY):
            return "component"
        if self == AllocationStrategy.PIPELINE_PARALLEL:
            return "layer"
        if self == AllocationStrategy.BLOCK_SCATTER:
            return "block"
        if self == AllocationStrategy.WEIGHT_SHARDING:
            return "weight_file"
        return "unknown"

    def is_component_placement(self) -> bool:
        """Check if this is a component placement strategy."""
        return self in (
            AllocationStrategy.COMPONENT_PLACEMENT,
            AllocationStrategy.COMPONENT_PLACEMENT_LAZY,
        )

    def is_pipeline_parallel(self) -> bool:
        """Check if this is a pipeline parallel strategy (per-layer sequential fill)."""
        return self == AllocationStrategy.PIPELINE_PARALLEL

    def is_block_scatter(self) -> bool:
        """Check if this is a block scatter strategy."""
        return self == AllocationStrategy.BLOCK_SCATTER

    def is_weight_sharding(self) -> bool:
        """Check if this is a weight sharding strategy."""
        return self == AllocationStrategy.WEIGHT_SHARDING


# Eager strategies: weights stay in VRAM permanently
_EAGER_STRATEGIES = frozenset({
    AllocationStrategy.SINGLE_GPU,
    AllocationStrategy.SINGLE_GPU_LIFECYCLE,
    AllocationStrategy.COMPONENT_PLACEMENT,
    AllocationStrategy.PIPELINE_PARALLEL,
    AllocationStrategy.BLOCK_SCATTER,
    AllocationStrategy.WEIGHT_SHARDING,
})


# =============================================================================
# DEVICE SPECIFICATION
# =============================================================================

@dataclass
class DeviceSpec:
    """
    Single GPU device specification.

    ZERO HARDCODE: All values from hardware profile YAML.
    """
    index: int
    name: str
    memory_mb: int
    compute_capability: str
    supports_dtypes: List[str]  # ["float32", "float16", "bfloat16"]
    architecture: str = "unknown"  # volta, ampere, hopper, rdna3, etc.
    brand: DeviceBrand = DeviceBrand.NVIDIA

    def get_device_string(self) -> str:
        """
        Get PyTorch device string for this GPU.

        VENDORLESS: Converts brand enum to correct device prefix.
        - NVIDIA -> cuda:X
        - AMD -> hip:X
        - INTEL -> xpu:X
        - TENSTORRENT -> tt:X
        """
        return f"{self.brand.to_device_prefix()}:{self.index}"

    @property
    def memory_gb(self) -> float:
        """Memory in gigabytes."""
        return self.memory_mb / 1024


# =============================================================================
# INTERCONNECT TOPOLOGY
# =============================================================================

@dataclass
class InterconnectLink:
    """
    Single interconnect link between devices.

    ZERO HARDCODE: Bandwidth from hardware profile YAML.
    """
    device_a: int
    device_b: int
    tech: InterconnectTech
    type: InterconnectType
    bandwidth_gbps: float
    bidirectional: bool = True

    def connects(self, dev_a: int, dev_b: int) -> bool:
        """Check if this link connects two devices (order independent)."""
        return (self.device_a == dev_a and self.device_b == dev_b) or \
               (self.device_a == dev_b and self.device_b == dev_a)


@dataclass
class InterconnectGroup:
    """
    Group of GPUs sharing fast interconnect.

    Example: 4 GPUs on NVSwitch = one group
    """
    name: str
    members: List[int]  # Device indices
    tech: InterconnectTech
    type: InterconnectType
    internal_bandwidth_gbps: float

    def contains(self, device_index: int) -> bool:
        """Check if device is in this group."""
        return device_index in self.members

    def get_peer_devices(self, device_index: int) -> List[int]:
        """Get all peer devices in this group (excluding self)."""
        if device_index not in self.members:
            return []
        return [m for m in self.members if m != device_index]


@dataclass
class InterconnectTopology:
    """
    Complete interconnect topology for a hardware profile.

    Supports:
    - Multi-technology (NVLink + PCIe, xGMI + QSFP)
    - Groups (NVSwitch domains, xGMI hives)
    - Fallback paths (NVLink → PCIe if needed)
    """
    groups: List[InterconnectGroup] = field(default_factory=list)
    links: List[InterconnectLink] = field(default_factory=list)
    default_tech: InterconnectTech = InterconnectTech.PCIE
    default_bandwidth_gbps: float = 32.0  # PCIe 4.0 x16

    def get_link(self, dev_a: int, dev_b: int) -> Optional[InterconnectLink]:
        """Get best link between two devices."""
        for link in self.links:
            if link.connects(dev_a, dev_b):
                return link
        return None

    def get_bandwidth(self, dev_a: int, dev_b: int) -> float:
        """
        Get bandwidth between two devices.

        Priority:
        1. Direct link (NVLink, xGMI)
        2. Group internal bandwidth
        3. Default (PCIe) bandwidth
        """
        # Check direct links first
        link = self.get_link(dev_a, dev_b)
        if link:
            return link.bandwidth_gbps

        # Check groups
        for group in self.groups:
            if group.contains(dev_a) and group.contains(dev_b):
                return group.internal_bandwidth_gbps

        # Fallback to default
        return self.default_bandwidth_gbps

    def get_interconnect_type(self, dev_a: int, dev_b: int) -> InterconnectType:
        """Get interconnect type between two devices."""
        # Check direct links
        link = self.get_link(dev_a, dev_b)
        if link:
            return link.type

        # Check groups
        for group in self.groups:
            if group.contains(dev_a) and group.contains(dev_b):
                return group.type

        # Fallback
        return InterconnectType.CPU_BOUNCE

    def devices_have_fast_interconnect(self, devices: List[int]) -> bool:
        """Check if all devices share fast interconnect (same group or direct links)."""
        if len(devices) <= 1:
            return True

        # Check if all in same group
        for group in self.groups:
            if all(group.contains(d) for d in devices):
                return True

        # Check pairwise links (not CPU_BOUNCE)
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                itype = self.get_interconnect_type(devices[i], devices[j])
                if itype == InterconnectType.CPU_BOUNCE:
                    return False

        return True

    def get_fastest_group(self) -> Optional[InterconnectGroup]:
        """Get group with highest internal bandwidth."""
        if not self.groups:
            return None
        return max(self.groups, key=lambda g: g.internal_bandwidth_gbps)


# =============================================================================
# PRISM PROFILE
# =============================================================================

@dataclass
class PrismProfile:
    """
    Complete hardware profile for execution planning.

    ZERO HARDCODE: All values from hardware profile YAML.
    """
    id: str
    vendor: str
    devices: List[DeviceSpec]
    topology: InterconnectTopology = field(default_factory=InterconnectTopology)
    cpu: Optional["CPUConfig"] = None  # CPU config from hardware profile
    preferred_dtype: Optional[str] = None  # Override dtype for memory efficiency

    @property
    def cpu_memory_gb(self) -> float:
        """CPU RAM in GB. Backward compat for code reading cpu_memory_gb."""
        if self.cpu is None:
            return 0.0
        return self.cpu.ram_gb

    @property
    def total_vram_mb(self) -> int:
        """Total VRAM across all devices."""
        return sum(d.memory_mb for d in self.devices)

    @property
    def total_vram_gb(self) -> float:
        """Total VRAM in gigabytes."""
        return self.total_vram_mb / 1024

    @property
    def device_count(self) -> int:
        """Number of devices."""
        return len(self.devices)

    def get_device(self, index: int) -> Optional[DeviceSpec]:
        """Get device by index."""
        for d in self.devices:
            if d.index == index:
                return d
        return None

    def devices_support_dtype(self, dtype: str) -> bool:
        """Check if ALL devices support a dtype."""
        return all(dtype in d.supports_dtypes for d in self.devices)

    def get_supported_dtypes(self) -> Set[str]:
        """Get dtypes supported by ALL devices."""
        if not self.devices:
            return set()
        common = set(self.devices[0].supports_dtypes)
        for device in self.devices[1:]:
            common &= set(device.supports_dtypes)
        return common

    def get_device_strings(self) -> List[str]:
        """Get PyTorch device strings for all devices."""
        return [d.get_device_string() for d in self.devices]

    def has_fast_interconnect(self) -> bool:
        """Check if profile has fast interconnect (any technology exceeding PCIe bandwidth)."""
        # Technology-agnostic: any group with bandwidth > PCIe 4.0 x16 (64 Gbps) is "fast"
        FAST_THRESHOLD_GBPS = 64.0
        if self.topology.groups:
            return any(g.internal_bandwidth_gbps > FAST_THRESHOLD_GBPS for g in self.topology.groups)
        return False

    def get_min_pairwise_bandwidth(self, device_indices: Optional[List[int]] = None) -> float:
        """Get minimum bandwidth between any pair of devices (Gbps).

        If device_indices is None, considers all devices.
        Returns the bottleneck bandwidth — the slowest link in the set.
        """
        if device_indices is None:
            device_indices = [d.index for d in self.devices]
        if len(device_indices) <= 1:
            return float('inf')
        min_bw = float('inf')
        for i in range(len(device_indices)):
            for j in range(i + 1, len(device_indices)):
                bw = self.topology.get_bandwidth(device_indices[i], device_indices[j])
                min_bw = min(min_bw, bw)
        return min_bw


# =============================================================================
# ALLOCATION STRUCTURES
# =============================================================================
#
# The authoritative ComponentAllocation and ExecutionPlan are defined in
# solver.py — they use plain strings for strategy (matching Prism's scoring
# system) and are the active runtime types.
#
# Legacy re-exports below provide backward compatibility for code that
# imports from structure.py. These are thin wrappers around solver types.
# =============================================================================

@dataclass
class PipelineStage:
    """Single stage in pipeline parallelism."""
    device: str  # "cuda:0", "hip:1", etc.
    components: List[str]
    memory_mb: float


@dataclass
class TensorParallelIntent:
    """
    TP intent declaration for future DAG rewriter.

    Declares that a tensor SHOULD be split for tensor parallelism.
    Actual execution requires a TP-DAG-Rewriter (transforms mm → mm_slice + all_reduce).
    """
    tensor_name: str       # "transformer.blocks.0.attn.q.weight"
    split_dim: int         # 0 (row) or 1 (column)
    world_size: int        # Number of GPUs for this split


@dataclass
class Zero3Config:
    """Configuration for ZeRO-3 style CPU offloading."""
    partition_count: int
    cpu_offload: bool
    nvme_offload: bool
    nvme_path: Optional[str] = None
