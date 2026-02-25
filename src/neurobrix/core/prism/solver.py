"""
NeuroBrix Prism - Hardware Allocation Solver

Algorithm: Best-Fit-Decreasing with Activation-Aware Memory Estimation
- Sort components by total memory DESCENDING (largest first)
- Sort GPUs by capacity DESCENDING (32GB before 16GB)
- Best-Fit placement: each component goes to smallest GPU that fits
- NVLink/PCIe topology awareness for optimal spillover
- Per-GPU contextual dtype conversion costing
"""

import json
import re
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

from neurobrix.core.prism.structure import AllocationStrategy, DeviceSpec, PrismProfile
from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig
from neurobrix.core.prism.memory_estimator import compute_dtype_factor, get_dtype_bytes_per_element
from neurobrix.core.config import get_prism_defaults, get_dtype_bytes
from neurobrix.core.config.system import PRISM_DEFAULTS

if TYPE_CHECKING:
    from nbx import NBXContainer, ComponentData


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ComponentAllocation:
    """Allocation plan for a single component."""
    name: str
    devices: List[str]
    dtype: torch.dtype
    memory_mb: float
    architecture: str
    vendor: str
    sharded: bool = False
    shard_map: Dict[str, str] = field(default_factory=dict)
    strategy: str = "single_gpu"

    @property
    def device(self) -> str:
        return self.devices[0] if self.devices else "cpu"

    @property
    def is_cpu(self) -> bool:
        return self.devices == ["cpu"]

    @property
    def offload_to_cpu(self) -> bool:
        return "cpu" in self.devices

    def get_shard_device(self, shard_file: str) -> str:
        return self.shard_map.get(shard_file, self.device)


@dataclass
class ComponentMemory:
    """Memory requirement: Total = Weights + Activations + Overhead"""
    component_name: str
    weight_bytes: int
    activation_bytes: int
    overhead_bytes: int
    peak_op_uid: str = ""
    peak_step: int = 0
    activation_profiled: bool = False
    attention_per_block_bytes: int = 0
    attention_dominates: bool = False

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.activation_bytes + self.overhead_bytes

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def weight_mb(self) -> float:
        return self.weight_bytes / (1024 * 1024)

    @property
    def activation_mb(self) -> float:
        return self.activation_bytes / (1024 * 1024)

    @property
    def overhead_mb(self) -> float:
        return self.overhead_bytes / (1024 * 1024)


@dataclass
class KVCachePlan:
    """KV cache allocation computed by Prism. Runtime MUST use these values."""
    max_cache_len: int          # Max tokens cacheable (Prism-computed trade-off)
    num_layers: int
    num_kv_heads: int
    k_head_dim: int             # K head dimension (qk_nope + qk_rope for MLA, else head_dim)
    v_head_dim: int             # V head dimension (v_head_dim for MLA, else head_dim)
    dtype: torch.dtype          # Cache dtype (may differ from weight dtype)
    memory_bytes: int           # Total KV cache memory budget
    per_token_bytes: int        # Memory per cached token (for runtime info)

    @property
    def head_dim(self) -> int:
        """Backward compat for code that reads .head_dim"""
        return self.k_head_dim

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)


@dataclass
class DeviceState:
    """
    GPU Resource Entity with Contextual Sizing.

    The cost of storing a block depends on THIS GPU's capabilities:
    - On A100 (supports BF16): Block costs X MB
    - On V100 (no BF16): Block costs 2X MB (must convert to FP32)
    - On H100 (supports FP8): Block could cost 0.5X MB
    """
    device_string: str
    capacity_mb: float
    used_mb: float = 0.0
    components: List[str] = field(default_factory=list)
    spec: Optional[DeviceSpec] = None

    @property
    def free_mb(self) -> float:
        return self.capacity_mb - self.used_mb

    @property
    def utilization_pct(self) -> float:
        return (self.used_mb / self.capacity_mb) * 100 if self.capacity_mb > 0 else 0

    @property
    def supports_dtypes(self) -> List[str]:
        return self.spec.supports_dtypes if self.spec else ["float32", "float16"]

    def get_cost_multiplier(self, model_dtype: str) -> float:
        """Contextual cost: BF16→FP32 = 2x on V100, native = 1x"""
        supported = self.supports_dtypes
        if model_dtype == "bfloat16" and "bfloat16" not in supported:
            if "float16" in supported:
                return 1.0  # bf16 → fp16 (Prism validates safety)
            return 2.0  # bf16 → fp32
        if model_dtype == "float16" and "float16" not in supported:
            return 2.0
        # Future: FP8 could return 0.5
        return 1.0

    def get_real_block_size(self, base_size_mb: float, model_dtype: str) -> float:
        return base_size_mb * self.get_cost_multiplier(model_dtype)

    def can_fit(self, memory_mb: float) -> bool:
        return memory_mb <= self.free_mb

    def can_fit_block(self, base_size_mb: float, model_dtype: str) -> bool:
        return self.can_fit(self.get_real_block_size(base_size_mb, model_dtype))

    def allocate(self, component_name: str, memory_mb: float) -> None:
        if not self.can_fit(memory_mb):
            raise RuntimeError(f"ZERO FALLBACK: Cannot allocate {memory_mb:.0f}MB to {self.device_string}")
        self.used_mb += memory_mb
        self.components.append(component_name)

    def allocate_block(self, block_name: str, base_size_mb: float, model_dtype: str) -> float:
        real_size = self.get_real_block_size(base_size_mb, model_dtype)
        self.allocate(block_name, real_size)
        return real_size


@dataclass
class ExecutionPlan:
    """Complete execution plan with component allocations."""
    components: Dict[str, ComponentAllocation]
    target_dtype: torch.dtype
    total_memory_mb: float
    strategy: str
    component_memory: Dict[str, ComponentMemory]
    loading_mode: str = "lazy"
    kv_cache_plan: Optional[KVCachePlan] = None

    @property
    def primary_device(self) -> str:
        return next(iter(self.components.values())).device if self.components else "cpu"

    @property
    def dtype(self) -> str:
        return str(self.target_dtype).split('.')[-1]

    def get_allocation(self, component_name: str) -> Optional[ComponentAllocation]:
        return self.components.get(component_name)

    def get_device(self, component_name: str) -> str:
        alloc = self.components.get(component_name)
        return alloc.device if alloc else "cpu"

    def get_memory_breakdown(self, component_name: str) -> Optional[ComponentMemory]:
        return self.component_memory.get(component_name)

    def to_dict(self, hardware_profile: str = "unknown") -> Dict[str, Any]:
        dtype_str = lambda dt: str(dt).split('.')[-1]
        return {
            "version": "0.1.0",
            "created_at": datetime.now().isoformat(),
            "hardware_profile": hardware_profile,
            "strategy": self.strategy,
            "loading_mode": self.loading_mode,
            "target_dtype": dtype_str(self.target_dtype),
            "total_memory_mb": self.total_memory_mb,
            "components": {
                name: {
                    "devices": alloc.devices,
                    "primary_device": alloc.device,
                    "dtype": dtype_str(alloc.dtype),
                    "memory_mb": alloc.memory_mb,
                    "architecture": alloc.architecture,
                    "vendor": alloc.vendor,
                    "sharded": alloc.sharded,
                    "shard_map": alloc.shard_map,
                }
                for name, alloc in self.components.items()
            },
            "memory_breakdown": {
                name: {
                    "weight_mb": mem.weight_mb,
                    "activation_mb": mem.activation_mb,
                    "overhead_mb": mem.overhead_mb,
                    "total_mb": mem.total_mb,
                    "peak_op": mem.peak_op_uid,
                    "activation_profiled": mem.activation_profiled,
                }
                for name, mem in self.component_memory.items()
            },
            "kv_cache_plan": {
                "max_cache_len": self.kv_cache_plan.max_cache_len,
                "num_layers": self.kv_cache_plan.num_layers,
                "num_kv_heads": self.kv_cache_plan.num_kv_heads,
                "k_head_dim": self.kv_cache_plan.k_head_dim,
                "v_head_dim": self.kv_cache_plan.v_head_dim,
                "dtype": dtype_str(self.kv_cache_plan.dtype),
                "memory_mb": self.kv_cache_plan.memory_mb,
                "per_token_bytes": self.kv_cache_plan.per_token_bytes,
            } if self.kv_cache_plan else None,
        }

    def save(self, output_path: Path, hardware_profile: str = "unknown") -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(hardware_profile), f, indent=2)

    @classmethod
    def load(cls, plan_path: Path) -> "ExecutionPlan":
        with open(plan_path, 'r') as f:
            data = json.load(f)

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

        components = {
            name: ComponentAllocation(
                name=name,
                devices=comp["devices"],
                dtype=dtype_map.get(comp["dtype"], torch.float32),
                memory_mb=comp["memory_mb"],
                architecture=comp.get("architecture", ""),
                vendor=comp.get("vendor", ""),
                sharded=comp["sharded"],
                shard_map=comp.get("shard_map", {}),
            )
            for name, comp in data.get("components", {}).items()
        }

        component_memory = {
            name: ComponentMemory(
                component_name=name,
                weight_bytes=int(mem.get("weight_mb", 0) * 1024 * 1024),
                activation_bytes=int(mem.get("activation_mb", 0) * 1024 * 1024),
                overhead_bytes=int(mem.get("overhead_mb", 0) * 1024 * 1024),
                peak_op_uid=mem.get("peak_op", ""),
                activation_profiled=mem.get("activation_profiled", False),
            )
            for name, mem in data.get("memory_breakdown", {}).items()
        }

        # Deserialize KV cache plan
        kv_data = data.get("kv_cache_plan")
        kv_cache_plan = None
        if kv_data:
            kv_cache_plan = KVCachePlan(
                max_cache_len=kv_data["max_cache_len"],
                num_layers=kv_data["num_layers"],
                num_kv_heads=kv_data["num_kv_heads"],
                k_head_dim=kv_data.get("k_head_dim") or kv_data["head_dim"],
                v_head_dim=kv_data.get("v_head_dim") or kv_data["head_dim"],
                dtype=dtype_map.get(kv_data.get("dtype", "float16"), torch.float16),
                memory_bytes=int(kv_data.get("memory_mb", 0) * 1024 * 1024),
                per_token_bytes=kv_data.get("per_token_bytes", 0),
            )

        return cls(
            components=components,
            target_dtype=dtype_map.get(data.get("target_dtype", "float32"), torch.float32),
            total_memory_mb=data.get("total_memory_mb", 0.0),
            strategy=data.get("strategy", "unknown"),
            component_memory=component_memory,
            loading_mode=data.get("loading_mode", "lazy"),
            kv_cache_plan=kv_cache_plan,
        )


# Backwards compatibility
PipelineExecutionPlan = ExecutionPlan


# =============================================================================
# PRISM SOLVER - Enterprise Grade
# =============================================================================

class PrismSolver:
    """
    Enterprise Grade Hardware Allocation Solver.

    Intelligence Features:
    - Best-Fit-Decreasing with activation profiling
    - NVLink/PCIe topology-aware spillover
    - Per-GPU contextual dtype costing
    - Quadratic attention memory correction
    - LLM vs Diffusion model detection
    - Automatic strategy selection with fallbacks
    """

    def __init__(self, safety_margin: float = 0.95, overhead_factor: float = 0.05):
        prism_defaults = get_prism_defaults()
        self.safety_margin = prism_defaults.get("safety_margin", safety_margin)
        self.overhead_factor = overhead_factor
        self._dtype_bytes = get_dtype_bytes()

    # =========================================================================
    # TOPOLOGY INTELLIGENCE
    # =========================================================================

    def _get_device_index(self, device_string: str) -> int:
        """Extract device index: 'cuda:2' → 2"""
        return int(device_string.split(':')[1])

    def _get_spillover_device(
        self, current_dev: DeviceState, available_devices: List[DeviceState], profile: PrismProfile
    ) -> Optional[DeviceState]:
        """
        NVLink-aware spillover: prefer same interconnect group.
        Score = (same_group: bool, free_mb: float)
        """
        current_idx = self._get_device_index(current_dev.device_string)
        topology = profile.topology

        def score(dev: DeviceState) -> tuple:
            if dev.free_mb <= 0:
                return (False, -1.0)
            dev_idx = self._get_device_index(dev.device_string)
            same_group = topology.devices_have_fast_interconnect([current_idx, dev_idx])
            return (same_group, dev.free_mb)

        candidates = [d for d in available_devices if d.device_string != current_dev.device_string]
        if not candidates:
            return None

        best = max(candidates, key=score)
        return best if score(best)[1] > 0 else None


    def _find_pipeline_device(
        self, required_mb: float, devices: List[DeviceState],
        last_device: Optional[DeviceState], profile: PrismProfile
    ) -> Optional[DeviceState]:
        """
        Topology-aware device selection for pipeline.
        Priority: 1) Same NVLink group 2) Any device with capacity
        """
        if not devices:
            return None

        if last_device is None:
            return next((d for d in devices if d.free_mb >= required_mb), None)

        last_idx = self._get_device_index(last_device.device_string)
        topology = profile.topology

        same_group = []
        other = []

        for dev in devices:
            if dev.free_mb < required_mb:
                continue
            dev_idx = self._get_device_index(dev.device_string)
            if topology.devices_have_fast_interconnect([last_idx, dev_idx]):
                same_group.append(dev)
            else:
                other.append(dev)

        return same_group[0] if same_group else (other[0] if other else None)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def solve(
        self, container: "NBXContainer", profile: PrismProfile, input_config: Optional[InputConfig] = None,
        serve_mode: bool = False,
    ) -> ExecutionPlan:
        """
        Solve optimal allocation with Best-Fit-Decreasing.

        Strategies tried in order:
        1. single_gpu - All on largest GPU (lazy loading)
        2. pp_nvlink/pp_pcie - Pipeline parallel
        3. fgp_nvlink/fgp_pcie - Fine-grained pipeline (block sharding)
        4. tp - Tensor parallel
        """
        self._serve_mode = serve_mode

        neural_components = container.get_neural_components()
        if not neural_components:
            return self._empty_plan(profile)

        if input_config is None:
            batch = PRISM_DEFAULTS.get("default_batch_size", 2)
            input_config = InputConfig(batch_size=batch, height=1024, width=1024)

        neural_components = sorted(neural_components, key=lambda c: c.name)

        # Step 0: Determine model category from manifest family
        manifest = container.get_manifest() or {}
        family = manifest.get("family", "unknown")
        has_lm_config = bool(self._read_lm_config(container))

        if family == "llm":
            self._model_category = "llm"
        elif family == "image" and has_lm_config:
            self._model_category = "image_vq"
        else:
            self._model_category = "diffusion"

        self._needs_kv_cache = self._model_category in ("llm", "image_vq")

        # Step 1: Resolve target dtype
        target_dtype = self._resolve_dtype(container, profile)
        target_dtype_str = str(target_dtype).split('.')[-1]
        self._target_dtype_str = target_dtype_str

        # Step 2: Compute memory requirements
        component_memory = self._compute_memory(container, neural_components, input_config, target_dtype_str)
        sorted_components = sorted(component_memory.items(), key=lambda x: -x[1].total_bytes)

        total_mem = sum(m.total_mb for _, m in sorted_components)

        # Step 3: Prepare devices
        devices = self._prepare_devices(profile)
        if not devices:
            raise RuntimeError("ZERO FALLBACK: No GPU devices in hardware profile")

        total_vram = sum(d.capacity_mb for d in devices)

        # Step 4: Try ALL strategies and pick best by score
        strategies = [
            ("single_gpu", self._try_single_gpu),
            ("single_gpu_lifecycle", self._try_single_gpu_lifecycle),
            ("pp_nvlink" if profile.has_fast_interconnect() else "pp_pcie", self._try_pipeline),
            ("fgp_nvlink" if profile.has_fast_interconnect() else "fgp_pcie", self._try_fgp),
            ("tp", self._try_tp),
            ("pp_lazy_nvlink" if profile.has_fast_interconnect() else "pp_lazy_pcie", self._try_pp_lazy),
            ("lazy_sequential", self._try_lazy_sequential),
            ("zero3", self._try_zero3),
        ]

        shard_sizes = container.get_shard_sizes()
        candidates = self._evaluate_all_strategies(
            strategies, sorted_components, component_memory, devices, shard_sizes, profile, container
        )

        # FP32 fallback for BF16 models
        if not candidates:
            if self._try_fp32_fallback(container, profile):
                # Retrying with FP32 fallback
                target_dtype_str = "float32"
                component_memory = self._compute_memory(container, neural_components, input_config, target_dtype_str)
                sorted_components = sorted(component_memory.items(), key=lambda x: -x[1].total_bytes)
                devices = self._prepare_devices(profile)
                candidates = self._evaluate_all_strategies(
                    strategies, sorted_components, component_memory, devices, shard_sizes, profile, container
                )

        if not candidates:
            self._fail_error(sorted_components, devices)

        # Sort candidates by score descending
        ranked = sorted(candidates, key=lambda x: -x[0])

        # Strategies ranked by score

        # Pick best strategy, validating KV cache fits
        allocations = None
        chosen_strategy = None
        kv_cache_plan = None
        chosen_devices = None

        for best in ranked:
            score, strat_name, strat_allocs, strat_devices = best

            if self._needs_kv_cache:
                # KV cache is already included in LM activation_bytes — subtract to avoid double-counting
                kv_already_counted = self._estimate_kv_cache_bytes(container, target_dtype_str)

                # Strategy-aware capacity and allocation calculation
                if strat_name in ("single_gpu", "single_gpu_lifecycle"):
                    # Single GPU: capacity = largest GPU only
                    total_capacity = int(max(d.capacity_mb for d in strat_devices) * 1024 * 1024)
                else:
                    total_capacity = sum(int(d.capacity_mb * 1024 * 1024) for d in strat_devices)

                if "zero3" in strat_name:
                    total_allocated = sum(m.activation_bytes + m.overhead_bytes
                                         for m in component_memory.values())
                elif strat_name == "single_gpu_lifecycle":
                    persistent, _ = self._classify_lifecycle(container)
                    total_allocated = sum(
                        m.total_bytes for name, m in component_memory.items()
                        if name in persistent
                    )
                elif strat_name == "single_gpu":
                    # Lazy: peak = max one component. Eager: all resident.
                    total_model = sum(m.total_bytes for m in component_memory.values())
                    largest_cap_bytes = int(max(d.capacity_mb for d in strat_devices) * 0.90 * 1024 * 1024)
                    if total_model <= largest_cap_bytes:
                        total_allocated = total_model  # eager
                    else:
                        total_allocated = max(m.total_bytes for m in component_memory.values())  # lazy
                else:
                    total_allocated = sum(m.total_bytes for m in component_memory.values())

                # Remove KV cache double-count (already in LM activation_bytes)
                total_allocated = max(total_allocated - kv_already_counted, 0)
                remaining = max(total_capacity - total_allocated, 0)
                try:
                    kv_plan = self._compute_kv_cache_plan(container, target_dtype, remaining)
                except RuntimeError as e:
                    # Strategy rejected: KV cache doesn't fit
                    continue
                kv_cache_plan = kv_plan

            allocations = strat_allocs
            chosen_strategy = strat_name
            chosen_devices = strat_devices
            break

        if allocations is None:
            raise RuntimeError(
                "ZERO FALLBACK: No strategy can fit model + KV cache on available hardware.\n"
                "Consider using a GPU with more VRAM or a multi-GPU setup."
            )

        devices = chosen_devices
        assert chosen_strategy is not None, "Strategy must be selected"
        assert devices is not None, "Devices must be allocated"

        # Step 5: Resolve per-component dtypes
        component_dtypes = self._resolve_component_dtypes(neural_components, profile, container)

        # Step 7: Build plan
        plan = self._build_plan(allocations, component_memory, devices, component_dtypes, profile, chosen_strategy)
        plan.kv_cache_plan = kv_cache_plan

        # Step 8: Summary
        self._print_summary(devices, plan, profile)
        return plan

    def solve_smart(self, container, profile, input_config=None, serve_mode: bool = False):
        """Backward compatibility alias for solve()."""
        return self.solve(container, profile, input_config, serve_mode=serve_mode)

    # =========================================================================
    # MEMORY COMPUTATION - Full Intelligence
    # =========================================================================

    def _compute_memory(
        self, container: "NBXContainer", components: List["ComponentData"],
        input_config: InputConfig, target_dtype_str: str
    ) -> Dict[str, ComponentMemory]:
        """Compute Total = Weights + Activations + Overhead with full intelligence."""
        result = {}
        shard_sizes = container.get_shard_sizes()
        manifest = container.get_manifest()
        assert manifest is not None, "Container manifest must be present"

        # Family-driven flags (set by solve() before calling us)
        category = getattr(self, '_model_category', 'diffusion')
        needs_kv_cache = getattr(self, '_needs_kv_cache', False)

        # Identify which component is the language model (for KV cache budgeting)
        lm_component_name = None
        if needs_kv_cache:
            lm_config = self._read_lm_config(container)
            if lm_config:
                lm_component_name = lm_config.get("component_name")
            if not lm_component_name:
                comp_names = [c.name for c in components]
                for candidate in ("language_model", "model", "llm"):
                    if candidate in comp_names:
                        lm_component_name = candidate
                        break

        for comp in components:
            source_dtype = comp.get_dominant_dtype()
            dtype_mult = compute_dtype_factor(source_dtype, target_dtype_str)

            # Weight memory
            weight_bytes = sum(int(s * dtype_mult) for s in shard_sizes.get(comp.name, {}).values())

            # Activation memory
            activation_bytes = 0
            peak_op_uid = ""
            peak_step = 0
            activation_profiled = False
            attn_per_block = 0
            attn_dominates = False

            if comp.graph is not None:
                try:
                    profiler = ActivationProfiler(comp.graph)
                    profile = profiler.estimate_peak_memory(
                        input_config=input_config,
                        dtype_bytes=get_dtype_bytes_per_element(target_dtype_str),
                    )
                    activation_bytes = profile.peak_bytes
                    peak_op_uid = profile.peak_op_uid
                    peak_step = profile.peak_step
                    activation_profiled = True
                except Exception:
                    activation_bytes = int(weight_bytes * 0.5)
            else:
                if category == "diffusion":
                    activation_bytes = int(weight_bytes * 0.5)
                # llm/image_vq without graph: no activation estimate (KV cache added below)

            # Add KV cache to activation budget for the LM component
            if needs_kv_cache and lm_component_name and comp.name == lm_component_name:
                kv_estimate = self._estimate_kv_cache_bytes(container, target_dtype_str)
                activation_bytes += kv_estimate

            # Attention correction for transformer components
            if any(n in comp.name.lower() for n in ('transformer', 'dit', 'diffusion')):
                activation_bytes, attn_per_block, attn_dominates = self._apply_attention_correction(
                    comp, activation_bytes, manifest, target_dtype_str, activation_profiled
                )

            overhead_bytes = int((weight_bytes + activation_bytes) * self.overhead_factor)

            result[comp.name] = ComponentMemory(
                component_name=comp.name,
                weight_bytes=weight_bytes,
                activation_bytes=activation_bytes,
                overhead_bytes=overhead_bytes,
                peak_op_uid=peak_op_uid,
                peak_step=peak_step,
                activation_profiled=activation_profiled,
                attention_per_block_bytes=attn_per_block,
                attention_dominates=attn_dominates,
            )

        return result

    def _read_defaults(self, container) -> Optional[Dict]:
        """Read defaults.json from cache."""
        if not container.cache_path:
            return None
        p = container.cache_path / "runtime" / "defaults.json"
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)

    def _read_lm_config(self, container) -> Optional[Dict]:
        """Read lm_config from defaults.json. Returns None if not LLM."""
        defaults = self._read_defaults(container)
        if not defaults:
            return None
        return defaults.get("lm_config")

    def _estimate_kv_cache_bytes(self, container, target_dtype_str: str) -> int:
        """Estimate KV cache memory for budget planning.
        Serve mode: target full context window (max_position_embeddings).
        Run mode: target max_tokens + prompt_margin."""
        lm_config = self._read_lm_config(container)
        if not lm_config:
            return 0

        num_layers = lm_config.get("num_layers", 0)
        num_heads = lm_config.get("num_heads", 0)
        hidden_size = lm_config.get("hidden_size", 0)
        num_kv_heads = lm_config.get("num_kv_heads", num_heads)
        head_dim = lm_config.get("head_dim", hidden_size // max(num_heads, 1))
        k_head_dim = lm_config.get("k_head_dim", head_dim)
        v_head_dim = lm_config.get("v_head_dim", head_dim)

        defaults = self._read_defaults(container)
        max_tokens = defaults.get("max_tokens") if defaults else None
        max_pos = lm_config.get("max_position_embeddings", 0)
        prompt_margin = defaults.get("prompt_margin", 128) if defaults else 128

        # Strategy evaluation estimate: use max_tokens + margin (not full context window).
        # Actual KV cache is sized later by _compute_kv_cache_plan() which constrains to VRAM.
        # Using max_position_embeddings here would reject valid strategies (e.g., FGP for Qwen3-30B
        # where max_pos=262144 adds ~25GB vs max_tokens=32768 adding ~3.2GB).
        cache_len = (max_tokens + prompt_margin) if max_tokens else max_pos

        dtype_bytes = 2 if target_dtype_str in ("float16", "bfloat16") else 4
        per_token_bytes = num_layers * num_kv_heads * (k_head_dim + v_head_dim) * dtype_bytes
        return cache_len * per_token_bytes

    def _compute_kv_cache_plan(
        self, container, target_dtype: torch.dtype, remaining_vram_bytes: int
    ) -> Optional[KVCachePlan]:
        """
        Compute KV cache allocation from lm_config. ZERO FALLBACK.

        remaining_vram_bytes = total device capacity - weights - overhead
        Prism decides max_cache_len from the remaining budget.
        """
        lm_config = self._read_lm_config(container)
        if not lm_config:
            return None

        num_layers: int = lm_config.get("num_layers", 0)
        num_heads: int = lm_config.get("num_heads", 0)
        hidden_size: int = lm_config.get("hidden_size", 0)
        if not all([num_layers, num_heads, hidden_size]):
            raise RuntimeError(
                "ZERO FALLBACK: lm_config missing num_layers/num_heads/hidden_size.\n"
                "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

        num_kv_heads: int = lm_config.get("num_kv_heads", 0) or num_heads
        head_dim: int = lm_config.get("head_dim", 0) or (hidden_size // num_heads)
        k_head_dim: int = lm_config.get("k_head_dim", head_dim)
        v_head_dim_val: int = lm_config.get("v_head_dim", head_dim)
        max_pos: int = lm_config.get("max_position_embeddings") or 0

        dtype_bytes = 2 if target_dtype in (torch.float16, torch.bfloat16) else 4
        per_token_bytes: int = num_layers * num_kv_heads * (k_head_dim + v_head_dim_val) * dtype_bytes

        # Prism decides max_cache_len from remaining VRAM budget
        defaults = self._read_defaults(container)
        max_tokens = (defaults.get("max_tokens") if defaults else None) or 0
        max_affordable = remaining_vram_bytes // per_token_bytes if per_token_bytes > 0 else 0
        prompt_margin = (defaults.get("prompt_margin", 128) if defaults else 128)

        if getattr(self, '_serve_mode', False):
            # Serve mode: target full context window, VRAM is the constraint
            upper_bound = max_pos
            # Minimum: at least 2 full turns must fit
            min_tokens = max_tokens * 2 + prompt_margin if max_tokens else 64
        else:
            # Run mode: single-shot, only need max_tokens + margin
            upper_bound = (max_tokens + prompt_margin) if max_tokens else max_pos
            min_tokens = (max_tokens + prompt_margin) if max_tokens else 64

        max_cache_len = min(max_affordable, upper_bound)

        if max_cache_len < min_tokens:
            raise RuntimeError(
                f"ZERO FALLBACK: KV cache budget insuffisant.\n"
                f"Remaining VRAM: {remaining_vram_bytes / 1e6:.0f}MB, "
                f"need ≥{min_tokens * per_token_bytes / 1e6:.0f}MB for {min_tokens} tokens.\n"
                f"Consider using a GPU with more VRAM or a multi-GPU setup."
            )

        memory_bytes = max_cache_len * per_token_bytes

        return KVCachePlan(
            max_cache_len=max_cache_len,
            num_layers=int(num_layers),
            num_kv_heads=int(num_kv_heads),
            k_head_dim=int(k_head_dim),
            v_head_dim=int(v_head_dim_val),
            dtype=target_dtype,
            memory_bytes=memory_bytes,
            per_token_bytes=per_token_bytes,
        )

    def _apply_attention_correction(
        self, comp, activation_bytes: int, manifest: dict, dtype_str: str, profiled: bool
    ) -> Tuple[int, int, bool]:
        """Apply quadratic attention memory correction for transformers."""
        vae_blocks = manifest.get('components', {}).get('vae', {}).get('blocks')
        attrs = comp.attributes if hasattr(comp, 'attributes') else None

        if not vae_blocks or not attrs:
            return activation_bytes, 0, False

        h = attrs.get('state_extent_0')
        w = attrs.get('state_extent_1')
        head_dim = attrs.get('attention_head_dim')
        channels = attrs.get('state_channels')

        if not all([h, w, head_dim, channels]):
            return activation_bytes, 0, False

        scale_factor = 2 ** (vae_blocks - 1)
        patch = PRISM_DEFAULTS.get("default_patch_size", 2)
        trace_seq = (h // patch) * (w // patch)
        runtime_seq = trace_seq  # Same resolution for now

        num_blocks = manifest.get('components', {}).get('transformer', {}).get('blocks', 20)
        num_heads = max(1, channels * 2)
        batch = PRISM_DEFAULTS.get("default_batch_size", 2)
        dtype_bytes = get_dtype_bytes_per_element(dtype_str)
        hidden_dim = channels * num_heads

        # Attention = seq² × heads × batch + QKV
        attn_matrix = runtime_seq * runtime_seq * num_heads * batch * dtype_bytes
        qkv = 3 * runtime_seq * hidden_dim * batch * dtype_bytes
        attn_per_block = int(attn_matrix + qkv)
        block_accum = num_blocks ** 0.5
        peak_attn = int(attn_per_block * block_accum)

        attn_dominates = False
        if not profiled and peak_attn > activation_bytes:
            activation_bytes = peak_attn
            attn_dominates = True

        # Resolution and safety scaling
        raw_scale = runtime_seq / trace_seq if trace_seq > 0 else 1.0
        resolution_scale = max(1.0, raw_scale ** 0.7)
        safety = 1.2
        total_mult = resolution_scale * safety
        activation_bytes = int(activation_bytes * total_mult)

        return activation_bytes, attn_per_block, attn_dominates

    # =========================================================================
    # DEVICE PREPARATION
    # =========================================================================

    def _prepare_devices(self, profile: PrismProfile) -> List[DeviceState]:
        """Prepare GPUs sorted by capacity DESC."""
        devices = [
            DeviceState(
                device_string=dev.get_device_string(),
                capacity_mb=dev.memory_mb * self.safety_margin,
                spec=dev,
            )
            for dev in profile.devices
        ]
        devices.sort(key=lambda d: (-d.capacity_mb, d.device_string))
        return devices

    def _fresh_devices(self, devices: List[DeviceState]) -> List[DeviceState]:
        """Create fresh device copies for strategy testing."""
        return [
            DeviceState(
                device_string=d.device_string,
                capacity_mb=d.capacity_mb,
                used_mb=0.0,
                components=[],
                spec=d.spec,
            )
            for d in devices
        ]

    # =========================================================================
    # STRATEGIES
    # =========================================================================

    def _try_single_gpu(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """All components on largest GPU (lazy loading - peak = max component)."""
        if not devices:
            return None

        largest = devices[0]

        # Peak = max single component
        peaks = [(n, m.weight_mb + m.activation_mb) for n, m in sorted_comps]
        max_name, max_peak = max(peaks, key=lambda x: x[1])

        needs_kv = getattr(self, '_needs_kv_cache', False)
        overhead_pct = 0.0 if needs_kv else 0.05
        overhead = max_peak * overhead_pct
        total_required = max_peak + overhead

        if total_required > largest.capacity_mb:
            return None

        allocations = {}
        fresh = self._fresh_devices(devices)

        for comp_name, _ in sorted_comps:
            shard_map = {s: largest.device_string for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (largest.device_string, shard_map)
            fresh[0].components.append(comp_name)

        return allocations, fresh

    def _classify_lifecycle(self, container) -> Tuple[set, set]:
        """Classify components as persistent (loop-resident) or transient (used once).

        For autoregressive models:
        - Persistent: language_model + gen_head + gen_embed + gen_aligner (called every token)
        - Transient: everything else (vision_model, gen_vision_model, aligner)

        Source: topology.json flow.generation config.
        """
        topology_path = container.cache_path / "topology.json" if container.cache_path else None
        gen_config = {}
        if topology_path and topology_path.exists():
            with open(topology_path) as f:
                gen_config = json.load(f).get("flow", {}).get("generation", {})

        persistent = set()
        for key in ("lm_component", "head_component", "embed_component", "aligner_component"):
            comp = gen_config.get(key)
            if comp:
                persistent.add(comp)

        # Fallback by name convention
        if not persistent:
            manifest = container.get_manifest() or {}
            for name in manifest.get("components", {}):
                if name in ("language_model", "model", "llm") or \
                   (name.startswith("gen_") and name != "gen_vision_model"):
                    persistent.add(name)

        all_comps = set((container.get_manifest() or {}).get("components", {}).keys())
        return persistent, all_comps - persistent

    def _try_single_gpu_lifecycle(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Lifecycle-aware single GPU: persistent components loaded together, transient on-demand.

        Peak = sum(all persistent) + max(one transient at a time).
        Only for LLM/image_vq models — diffusion uses _try_single_gpu.
        """
        if not devices:
            return None
        category = getattr(self, '_model_category', 'diffusion')
        if category == "diffusion":
            return None

        largest = devices[0]
        persistent, transient = self._classify_lifecycle(container)

        persistent_mb = sum(
            comp_mem[n].weight_mb + comp_mem[n].activation_mb
            for n in persistent if n in comp_mem
        )
        transient_peaks = [
            comp_mem[n].weight_mb + comp_mem[n].activation_mb
            for n in transient if n in comp_mem
        ]
        max_transient = max(transient_peaks) if transient_peaks else 0

        peak = persistent_mb + max_transient

        if peak > largest.capacity_mb:
            return None

        allocations = {}
        fresh = self._fresh_devices(devices)
        for comp_name, _ in sorted_comps:
            shard_map = {s: largest.device_string for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (largest.device_string, shard_map)
            fresh[0].components.append(comp_name)

        return allocations, fresh

    def _try_pipeline(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Distribute components across GPUs (pipeline parallel)."""
        if len(devices) < 2:
            return None

        fresh = self._fresh_devices(devices)
        allocations = {}
        last_device = None

        for comp_name, mem in sorted_comps:
            required = mem.weight_mb + mem.activation_mb

            # Attention components need 20% headroom
            is_attn = any(n in comp_name.lower() for n in ('transformer', 'dit', 'diffusion'))
            if is_attn:
                min_required = max(required * 1.20, 20000)
                target = self._find_pipeline_device(min_required, fresh, last_device, profile)
                if target is None:
                    return None
            else:
                target = self._find_pipeline_device(required, fresh, last_device, profile)

            if target is None:
                return None

            target.allocate(comp_name, required)
            shard_map = {s: target.device_string for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (target.device_string, shard_map)
            last_device = target

        # Need at least 2 devices used
        used = set(a[0] for a in allocations.values())
        if len(used) < 2:
            return None

        return allocations, fresh

    def _try_fgp(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Fine-grained pipeline: spread transformer blocks across GPUs."""
        if len(devices) < 2:
            return None

        capacity_threshold = devices[0].capacity_mb * 0.80
        needs_fgp = [(n, m) for n, m in sorted_comps if m.weight_mb + m.activation_mb > capacity_threshold]

        if not needs_fgp:
            return None

        fgp_target = PRISM_DEFAULTS.get("fgp_utilization_target", 0.92)
        fresh = [
            DeviceState(
                device_string=d.device_string,
                capacity_mb=d.capacity_mb * fgp_target,
                used_mb=0.0, components=[], spec=d.spec
            )
            for d in devices
        ]

        allocations = {}
        current_dev_idx = 0

        for comp_name, mem in needs_fgp:
            blocks = self._parse_blocks(container, comp_name)
            if not blocks['blocks']:
                return None

            # Get model dtype
            model_dtype = None
            for comp in container.get_neural_components():
                if comp.name == comp_name:
                    model_dtype = comp.get_dominant_dtype()
                    break
            model_dtype = model_dtype or "bfloat16"

            n_blocks = len(blocks['blocks'])
            act_per_block = mem.activation_mb / n_blocks

            shard_map = {}

            # Allocate non-block weights
            if blocks['non_block_mb'] > 0:
                while current_dev_idx < len(fresh):
                    dev = fresh[current_dev_idx]
                    real_size = dev.get_real_block_size(blocks['non_block_mb'], model_dtype) * 1.05
                    if dev.can_fit(real_size):
                        dev.used_mb += real_size
                        dev.components.append(f"{comp_name}.non_block")
                        for key in blocks['non_block_keys']:
                            shard_map[key] = dev.device_string
                        break
                    current_dev_idx += 1
                if current_dev_idx >= len(fresh):
                    return None

            # Allocate blocks - best-fit across ALL GPUs (not monotonic)
            for block_num in sorted(blocks['blocks'].keys()):
                block_keys = blocks['blocks'][block_num]
                base_block = blocks['block_sizes'][block_num]
                real_block = fresh[0].get_real_block_size(base_block, model_dtype)
                total_block = (real_block + act_per_block) * 1.05

                # Find GPU with most free space that can fit this block
                candidates = [d for d in fresh if d.can_fit(total_block)]
                if not candidates:
                    return None

                # Prefer NVLink-connected GPUs, then most free space
                target = max(candidates, key=lambda d: d.free_mb)
                target.used_mb += total_block
                target.components.append(f"{comp_name}.block.{block_num}")
                for key in block_keys:
                    shard_map[key] = target.device_string

            devices_used = set(shard_map.values())
            allocations[comp_name] = (f"fgp:{','.join(sorted(devices_used))}", shard_map)

        # Allocate regular components
        regular = sorted(
            [(n, m) for n, m in sorted_comps if n not in allocations],
            key=lambda x: x[1].weight_mb + x[1].activation_mb
        )

        for comp_name, mem in regular:
            required = mem.weight_mb + mem.activation_mb
            target = next((d for d in fresh if d.can_fit(required)), None)
            if target is None:
                return None
            target.allocate(comp_name, required)
            shard_map = {s: target.device_string for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (target.device_string, shard_map)

        return allocations, fresh

    def _try_tp(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Tensor parallel: shard weights across GPUs.

        TP distributes weight columns/rows across GPUs. No DAG rewrite needed:
        - Prism generates tp_shard_plan (which weight slices go where)
        - WeightLoader loads slices directly to assigned GPUs
        - CompiledSequence handles multi-device execution with all_reduce at sync points
        - Graph metadata (shapes, parent_module) provides all info for sharding

        Weight files distributed via shard_map across TP GPUs.
        CompiledSequence multi-device path handles cross-device execution
        with automatic device alignment at op boundaries.
        """
        if len(devices) < 2:
            return None

        largest_cap = devices[0].capacity_mb
        needs_tp = [(n, m) for n, m in sorted_comps if m.weight_mb + m.activation_mb > largest_cap]

        if not needs_tp:
            return None

        fresh = self._fresh_devices(devices)
        allocations = {}
        tp_used = set()

        for comp_name, mem in needs_tp:
            available = [d for d in fresh if d.device_string not in tp_used]

            if not available:
                return None

            # Get model dtype for cost multiplier (bf16→fp32 = 2x on V100)
            model_dtype = None
            for comp in container.get_neural_components():
                if comp.name == comp_name:
                    model_dtype = comp.get_dominant_dtype()
                    break
            model_dtype = model_dtype or "bfloat16"

            # Determine if attention dominates (for smarter sharding)
            attn_per_block_mb = mem.attention_per_block_bytes / (1024 * 1024) if mem.attention_dominates else 0

            # Find minimum GPUs needed (dtype-aware)
            n_gpus = 2
            for n in range(2, len(available) + 1):
                cost_mult = available[0].get_cost_multiplier(model_dtype)
                weight_per_gpu = (mem.weight_mb * cost_mult) / n
                act_per_gpu = (attn_per_block_mb / n) if attn_per_block_mb > 0 else (mem.activation_mb / n)
                overhead = (weight_per_gpu + act_per_gpu) * 0.1 if act_per_gpu > 0 else 0
                per_gpu_total = weight_per_gpu + act_per_gpu + overhead

                if per_gpu_total <= available[n-1].capacity_mb:
                    n_gpus = n
                    break
            else:
                return None

            required = (mem.weight_mb * available[0].get_cost_multiplier(model_dtype)) + mem.activation_mb

            tp_gpus = available[:n_gpus]
            per_gpu = required / n_gpus

            for gpu in tp_gpus:
                gpu.allocate(comp_name, per_gpu)
                tp_used.add(gpu.device_string)

            # Distribute weight shards across TP GPUs (round-robin by size descending)
            comp_shards = sorted(
                shard_sizes.get(comp_name, {}).items(),
                key=lambda x: x[1], reverse=True,
            )
            gpu_loads = [0.0] * n_gpus
            shard_map = {}
            for shard_name, shard_bytes in comp_shards:
                # Assign to GPU with least load (greedy balancing)
                min_idx = gpu_loads.index(min(gpu_loads))
                shard_map[shard_name] = tp_gpus[min_idx].device_string
                gpu_loads[min_idx] += shard_bytes
            allocations[comp_name] = (f"tp:{','.join(g.device_string for g in tp_gpus)}", shard_map)

        # Allocate regular components
        remaining = [d for d in fresh if d.device_string not in tp_used] or [fresh[0]]
        regular = [(n, m) for n, m in sorted_comps if n not in allocations]

        for comp_name, mem in regular:
            required = mem.weight_mb + mem.activation_mb
            target = next((d for d in remaining if d.can_fit(required)), None)
            if target is None:
                return None
            target.allocate(comp_name, required)
            shard_map = {s: target.device_string for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (target.device_string, shard_map)

        return allocations, fresh

    def _try_lazy_sequential(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """
        Lazy sequential: load one component at a time.
        Peak memory = max(single component), not sum(all components).

        Each component is placed using the recursive cascade:
        _place_component() tries single_gpu → fgp → zero3 for the component.
        """
        if not devices:
            return None

        fresh = self._fresh_devices(devices)
        allocations = {}

        for comp_name, mem in sorted_comps:
            result = self._place_component(
                container, comp_name, mem, devices, shard_sizes, profile
            )
            if result is None:
                return None
            allocations[comp_name] = result

        return allocations, fresh

    # =========================================================================
    # RECURSIVE CASCADE — Component & Block Level
    # =========================================================================

    def _place_component(
        self,
        container: "NBXContainer",
        comp_name: str,
        mem: ComponentMemory,
        devices: List[DeviceState],
        shard_sizes: Dict,
        profile: "PrismProfile",
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Recursive cascade for a single component.
        Same logic as model-level strategies, applied to one component.

        Cascade order:
        1. single_gpu — component fits on largest GPU
        2. fgp — distribute blocks across GPUs
        3. zero3 — shard everything to CPU, stream to GPU for compute

        Returns (device_string, shard_map) or None if placement impossible.
        """
        model_dtype = self._get_component_dtype(container, comp_name)
        largest = max(devices, key=lambda d: d.capacity_mb)
        cost_mult = largest.get_cost_multiplier(model_dtype)
        real_weight = mem.weight_mb * cost_mult
        required = real_weight + mem.activation_mb

        # Strategy 1: single_gpu — component fits on largest GPU
        if required <= largest.capacity_mb * 0.92:
            shard_map = {s: largest.device_string for s in shard_sizes.get(comp_name, {})}
            return (largest.device_string, shard_map)

        # Strategy 2: fgp — distribute blocks across GPUs
        fgp_result = self._place_component_fgp(
            container, comp_name, mem, devices, model_dtype
        )
        if fgp_result is not None:
            return fgp_result

        # Strategy 3: zero3 — CPU offload, GPU for compute only
        if mem.activation_mb <= largest.capacity_mb * 0.92:
            shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
            return (f"zero3:{largest.device_string}", shard_map)

        return None

    def _place_component_fgp(
        self,
        container: "NBXContainer",
        comp_name: str,
        mem: ComponentMemory,
        devices: List[DeviceState],
        model_dtype: str,
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        FGP placement for a single component: distribute blocks across GPUs.

        If a single block is too large for any GPU, falls back to block-level
        cascade (_place_block_across_gpus) which splits the block's keys.

        Returns (device_string, shard_map) or None.
        """
        blocks = self._parse_blocks(container, comp_name)
        if not blocks['blocks']:
            return None

        fgp_target = PRISM_DEFAULTS.get("fgp_utilization_target", 0.92)
        block_devices = [
            DeviceState(
                device_string=d.device_string,
                capacity_mb=d.capacity_mb * fgp_target,
                used_mb=0.0, components=[], spec=d.spec
            )
            for d in self._fresh_devices(devices)
        ]

        # Check total FGP capacity
        cost_mult = block_devices[0].get_cost_multiplier(model_dtype)
        real_weight = mem.weight_mb * cost_mult
        total_capacity = sum(d.capacity_mb for d in block_devices)
        if real_weight + mem.activation_mb > total_capacity:
            return None

        n_blocks = len(blocks['blocks'])
        act_per_block = mem.activation_mb / n_blocks
        non_block_keys = blocks.get('non_block_keys', [])
        non_block_mb = blocks.get('non_block_mb', 0)

        # Reserve non-block space on largest-capacity GPU BEFORE blocks
        primary = max(block_devices, key=lambda d: d.capacity_mb)
        real_non_block = primary.get_real_block_size(non_block_mb, model_dtype) * 1.05
        primary.used_mb += real_non_block

        # Allocate blocks across GPUs (best-fit)
        shard_map: Dict[str, str] = {}
        for block_num in sorted(blocks['blocks'].keys()):
            block_keys = blocks['blocks'][block_num]
            base_block = blocks['block_sizes'][block_num]
            real_block = block_devices[0].get_real_block_size(base_block, model_dtype)
            total_block = (real_block + act_per_block) * 1.05

            # Try to fit entire block on one GPU
            candidates = [d for d in block_devices if d.can_fit(total_block)]
            if candidates:
                target = max(candidates, key=lambda d: d.free_mb)
                target.used_mb += total_block
                target.components.append(f"{comp_name}.block.{block_num}")
                for key in block_keys:
                    shard_map[key] = target.device_string
            else:
                # Block-level cascade: block too large for any single GPU
                # Split the block's keys across multiple GPUs by individual key size
                placed = self._place_block_across_gpus(
                    comp_name, block_num, block_keys,
                    blocks.get('key_sizes', {}), act_per_block,
                    block_devices, model_dtype
                )
                if not placed:
                    return None
                shard_map.update(placed)

        # Assign non-block keys to primary
        for key in non_block_keys:
            shard_map[key] = primary.device_string

        return (primary.device_string, shard_map)

    def _place_block_across_gpus(
        self,
        comp_name: str,
        block_num: int,
        block_keys: List[str],
        key_sizes: Dict[str, int],
        act_budget_mb: float,
        devices: List[DeviceState],
        model_dtype: str,
    ) -> Optional[Dict[str, str]]:
        """
        Block-level cascade: distribute a single block's keys across GPUs.

        When a block is too large for any single GPU (rare but possible for
        very large MoE blocks or oversized attention layers), split the block's
        individual weight keys across GPUs using best-fit.

        Args:
            comp_name: Component name (for logging)
            block_num: Block index
            block_keys: Weight keys belonging to this block
            key_sizes: Dict of key → size in bytes (from _parse_blocks)
            act_budget_mb: Activation budget per block in MB
            devices: Available GPU states
            model_dtype: Model dtype string

        Returns:
            Dict of key → device_string, or None if placement impossible.
        """
        n_keys = len(block_keys)
        act_per_key = act_budget_mb / max(n_keys, 1)
        shard_map: Dict[str, str] = {}

        # Sort keys by size descending (best-fit-decreasing)
        sorted_keys = sorted(
            block_keys,
            key=lambda k: key_sizes.get(k, 0),
            reverse=True,
        )

        for key in sorted_keys:
            key_bytes = key_sizes.get(key, 0)
            key_mb = key_bytes / (1024 * 1024)
            real_key = devices[0].get_real_block_size(key_mb, model_dtype)
            total_key = (real_key + act_per_key) * 1.05

            candidates = [d for d in devices if d.can_fit(total_key)]
            if not candidates:
                return None

            target = max(candidates, key=lambda d: d.free_mb)
            target.used_mb += total_key
            target.components.append(f"{comp_name}.block.{block_num}.key")
            shard_map[key] = target.device_string

        return shard_map

    def _get_component_dtype(self, container: "NBXContainer", comp_name: str) -> str:
        """Get the dominant dtype for a component."""
        for comp in container.get_neural_components():
            if comp.name == comp_name:
                return comp.get_dominant_dtype()
        return "bfloat16"

    # =========================================================================
    # SCORE-BASED STRATEGY EVALUATION
    # =========================================================================

    def _evaluate_all_strategies(
        self, strategies, sorted_components, component_memory, devices, shard_sizes, profile, container
    ) -> List[Tuple[float, str, Dict, List]]:
        """Try ALL strategies and return scored candidates."""
        candidates = []
        for strategy_name, strategy_fn in strategies:
            result = strategy_fn(sorted_components, component_memory, devices, shard_sizes, profile, container)
            if result is not None:
                allocations, used_devices = result
                score = self._score_strategy(strategy_name, allocations, profile)
                candidates.append((score, strategy_name, allocations, used_devices))
        return candidates

    def _score_strategy(self, strategy_name: str, allocations: Dict, profile: PrismProfile) -> float:
        """
        Score a strategy by estimated relative throughput (higher = better).

        Factors:
        - Base score by strategy type (single_gpu best, zero3 worst)
        - Transfer penalty weighted by interconnect bandwidth
        - Lazy loading overhead penalty
        - CPU streaming penalty for zero3
        """
        BASE_SCORES = {
            "single_gpu": 1000,
            "single_gpu_lifecycle": 900,
            "pp_nvlink": 800, "pp_pcie": 700,
            "fgp_nvlink": 750, "fgp_pcie": 650,
            "tp": 780,
            "pp_lazy_nvlink": 500, "pp_lazy_pcie": 400,
            "lazy_sequential": 300,
            "zero3": 100,
        }
        score = float(BASE_SCORES.get(strategy_name, 500))

        # Count unique devices used
        device_strings = set()
        for alloc in allocations.values():
            dev_str = alloc[0] if isinstance(alloc, tuple) else alloc
            # Strip zero3: prefix
            if dev_str.startswith("zero3:"):
                dev_str = dev_str[6:]
            if dev_str.startswith("cuda:"):
                device_strings.add(dev_str)

        n_devices = len(device_strings)

        # Transfer penalty: each cross-device hop costs, weighted by bandwidth
        if n_devices > 1 and profile.topology:
            topology = profile.topology
            device_indices = sorted(int(d.split(':')[-1]) for d in device_strings)
            for i in range(len(device_indices) - 1):
                bw = topology.get_bandwidth(device_indices[i], device_indices[i + 1])
                # NVLink ~300 Gbps -> small penalty; PCIe ~32 Gbps -> big penalty
                transfer_penalty = 50.0 * (32.0 / max(bw, 1.0))
                score -= transfer_penalty

        # Lazy loading penalty: each component load/unload adds latency
        if "lazy" in strategy_name:
            n_components = len(allocations)
            score -= 20.0 * n_components

        # Zero3: CPU->GPU streaming is always slowest
        if "zero3" in strategy_name:
            score -= 200.0

        return max(score, 1.0)

    # =========================================================================
    # NEW STRATEGIES: PP_LAZY, ZERO3
    # =========================================================================

    def _try_pp_lazy(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """
        Pipeline-Lazy hybrid: distribute component GROUPS across N GPUs.
        Each GPU lazily loads/unloads its assigned components.
        Peak per GPU = max(component in group), not sum(all in group).
        """
        if len(devices) < 2:
            return None

        fresh = self._fresh_devices(devices)
        n_gpus = len(fresh)

        # Group components into N bins (greedy best-fit decreasing)
        bins: List[List[Tuple[str, Any]]] = [[] for _ in range(n_gpus)]
        bin_peaks = [0.0] * n_gpus

        for comp_name, mem in sorted_comps:
            # Get model dtype for cost multiplier
            model_dtype = None
            for comp in container.get_neural_components():
                if comp.name == comp_name:
                    model_dtype = comp.get_dominant_dtype()
                    break
            model_dtype = model_dtype or "bfloat16"

            cost = fresh[0].get_cost_multiplier(model_dtype)
            required = mem.weight_mb * cost + mem.activation_mb

            # Find best GPU where peak stays within capacity (best-fit)
            best_gpu = None
            best_headroom = float('inf')
            for i in range(n_gpus):
                new_peak = max(bin_peaks[i], required)
                capacity = fresh[i].capacity_mb * 0.92
                if new_peak <= capacity:
                    headroom = capacity - new_peak
                    if headroom < best_headroom:
                        best_gpu = i
                        best_headroom = headroom

            if best_gpu is None:
                return None

            bins[best_gpu].append((comp_name, mem))
            bin_peaks[best_gpu] = max(bin_peaks[best_gpu], required)

        # Build allocations
        allocations = {}
        for gpu_idx, gpu_comps in enumerate(bins):
            for comp_name, mem in gpu_comps:
                shard_map = {s: fresh[gpu_idx].device_string for s in shard_sizes.get(comp_name, {})}
                allocations[comp_name] = (fresh[gpu_idx].device_string, shard_map)

        return allocations, fresh

    def _try_zero3(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """
        ZeRO-3 CPU offload: weights on CPU pinned memory, GPU for compute only.
        Peak GPU = activations only (weights streamed block-by-block).
        Succeeds if at least 1 GPU can hold the largest activation footprint.
        """
        if not devices:
            return None

        fresh = self._fresh_devices(devices)
        largest = max(fresh, key=lambda d: d.capacity_mb)
        allocations = {}

        for comp_name, mem in sorted_comps:
            # Zero3: only activations need GPU memory
            if mem.activation_mb > largest.capacity_mb * 0.92:
                return None

            shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (f"zero3:{largest.device_string}", shard_map)

        return allocations, fresh

    # Dtype string to bytes-per-element for safetensors header parsing
    _ST_DTYPE_BYTES = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
        "BOOL": 1,
    }

    def _parse_blocks(self, container: "NBXContainer", comp_name: str) -> Dict:
        """
        Parse weight keys to identify transformer block structure.

        Uses safetensors HEADER METADATA (shape + dtype) to compute sizes
        without loading actual tensors. This avoids loading 32GB of weights
        just to determine block structure.
        """
        cache_path = container.cache_path
        if not cache_path:
            return {'blocks': {}, 'block_sizes': {}, 'non_block_keys': [], 'non_block_mb': 0}

        weights_path = cache_path / 'components' / comp_name / 'weights'
        if not weights_path.exists():
            return {'blocks': {}, 'block_sizes': {}, 'non_block_keys': [], 'non_block_mb': 0}

        patterns = [r'^blocks?\.(\d+)\.', r'^layers?\.(\d+)\.',
                    r'\.blocks?\.(\d+)\.', r'\.layers?\.(\d+)\.',
                    r'^\w+_blocks?\.(\d+)\.', r'^\w+_layers?\.(\d+)\.']

        blocks = {}
        non_block_keys = []
        key_sizes = {}

        for shard in weights_path.glob('*.safetensors'):
            # Read header only — no tensor data loaded
            header = self._read_safetensors_header(shard)
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                # Compute size from shape × dtype_bytes
                shape = meta.get("shape", [])
                dtype_str = meta.get("dtype", "F32")
                dtype_bytes = self._ST_DTYPE_BYTES.get(dtype_str, 4)
                numel = 1
                for dim in shape:
                    numel *= dim
                key_sizes[key] = numel * dtype_bytes

                matched = False
                for pattern in patterns:
                    m = re.search(pattern, key)
                    if m:
                        blocks.setdefault(int(m.group(1)), []).append(key)
                        matched = True
                        break
                if not matched:
                    non_block_keys.append(key)

        block_sizes = {n: sum(key_sizes[k] for k in keys) / 1024**2 for n, keys in blocks.items()}
        non_block_mb = sum(key_sizes[k] for k in non_block_keys) / 1024**2

        return {
            'blocks': blocks,
            'block_sizes': block_sizes,
            'avg_block_mb': sum(block_sizes.values()) / len(block_sizes) if block_sizes else 0,
            'non_block_keys': non_block_keys,
            'non_block_mb': non_block_mb,
            'key_sizes': key_sizes,
        }

    @staticmethod
    def _read_safetensors_header(path: Path) -> Dict:
        """Read safetensors header without loading tensor data."""
        import struct
        with open(path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header_bytes = f.read(header_size)
        return json.loads(header_bytes)

    def _scan_bf16_fp16_safety(self, container: "NBXContainer") -> bool:
        """
        Scan bf16 weight files to verify all values fit in fp16 range (±65504).
        Returns True if bf16→fp16 is safe, False if fp32 fallback needed.

        Results are cached in two layers:
        1. Instance cache (in-memory, per solver invocation)
        2. Disk cache (~/.neurobrix/cache/<model>/bf16_safety.json) — persists across runs.
           Invalidated if any weight file's mtime is newer than the cache file.
        """
        if hasattr(self, '_bf16_fp16_safe_cache'):
            return self._bf16_fp16_safe_cache

        cache_path = container.cache_path
        if not cache_path:
            return False

        components_dir = cache_path / "components"
        if not components_dir.exists():
            return False

        # Check disk cache
        safety_cache_file = cache_path / "bf16_safety.json"
        if safety_cache_file.exists():
            # Validate: cache must be newer than all weight files
            cache_mtime = safety_cache_file.stat().st_mtime
            cache_valid = True
            for comp_dir in components_dir.iterdir():
                weights_path = comp_dir / "weights"
                if not weights_path.exists():
                    continue
                for shard in weights_path.glob("*.safetensors"):
                    if shard.stat().st_mtime > cache_mtime:
                        cache_valid = False
                        break
                if not cache_valid:
                    break

            if cache_valid:
                with open(safety_cache_file) as f:
                    cached = json.load(f)
                result = cached.get("bf16_fp16_safe", False)
                self._bf16_fp16_safe_cache = result
                return result

        # Full scan using safetensors header metadata + selective tensor loading
        from safetensors import safe_open

        FP16_MAX = 65504.0
        total_params = 0

        for comp_dir in components_dir.iterdir():
            weights_path = comp_dir / "weights"
            if not weights_path.exists():
                continue
            for shard in sorted(weights_path.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        if tensor.dtype == torch.bfloat16:
                            total_params += tensor.numel()
                            abs_max = tensor.abs().max().item()
                            if abs_max > FP16_MAX:
                                # Cache negative result to disk
                                with open(safety_cache_file, 'w') as cf:
                                    json.dump({"bf16_fp16_safe": False, "reason": f"{key}: abs_max={abs_max}"}, cf)
                                self._bf16_fp16_safe_cache = False
                                return False

        result = total_params > 0
        # Cache positive result to disk
        with open(safety_cache_file, 'w') as cf:
            json.dump({"bf16_fp16_safe": result, "bf16_params": total_params}, cf)
        self._bf16_fp16_safe_cache = result
        return result

    # =========================================================================
    # DTYPE RESOLUTION
    # =========================================================================

    def _resolve_dtype(self, container: "NBXContainer", profile: PrismProfile) -> torch.dtype:
        """Resolve target dtype. bf16 → fp16 if weights fit in range, else fp32."""
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

        # Find model's dominant dtype
        dtypes = {}
        for comp in container.get_neural_components():
            d = comp.get_dominant_dtype()
            dtypes[d] = dtypes.get(d, 0) + 1
        requested = max(dtypes.keys(), key=lambda d: dtypes[d]) if dtypes else "float16"

        # Priority: hardware preferred > model native > fallback
        if profile.preferred_dtype and profile.devices_support_dtype(profile.preferred_dtype):
            return dtype_map.get(profile.preferred_dtype, torch.float32)

        if profile.devices_support_dtype(requested):
            return dtype_map.get(requested, torch.float32)

        if requested == "bfloat16":
            if profile.devices_support_dtype("float16") and self._scan_bf16_fp16_safety(container):
                return torch.float16
            return torch.float32
        elif requested == "float16":
            return torch.float32

        return torch.float32

    def _resolve_component_dtypes(self, components, profile: PrismProfile, container: "Optional[NBXContainer]" = None) -> Dict[str, torch.dtype]:
        """Resolve dtype per component."""
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        result = {}

        for comp in components:
            native = comp.get_dominant_dtype()
            if profile.preferred_dtype and profile.devices_support_dtype(profile.preferred_dtype):
                resolved = dtype_map.get(profile.preferred_dtype, torch.float32)
            elif profile.devices_support_dtype(native):
                resolved = dtype_map.get(native, torch.float32)
            elif native == "bfloat16":
                if container is not None and profile.devices_support_dtype("float16") and self._scan_bf16_fp16_safety(container):
                    resolved = torch.float16
                else:
                    resolved = torch.float32
            elif native == "float16":
                resolved = torch.float32
            else:
                resolved = torch.float32
            result[comp.name] = resolved

        return result

    def _try_fp32_fallback(self, container: "NBXContainer", profile: PrismProfile) -> bool:
        """Check if FP32 fallback should be tried for BF16 models."""
        for comp in container.get_neural_components():
            if comp.get_dominant_dtype() == "bfloat16":
                return True
        return False

    # =========================================================================
    # PLAN BUILDING
    # =========================================================================

    def _build_plan(
        self, allocations, comp_mem, devices, comp_dtypes, profile: PrismProfile, strategy: str
    ) -> ExecutionPlan:
        """Build ExecutionPlan from allocation results."""
        arch = profile.devices[0].architecture if profile.devices else ""
        vendor = profile.devices[0].brand.value if profile.devices else ""

        components = {}
        total_mb = 0.0

        for comp_name, (device_str, shard_map) in allocations.items():
            mem = comp_mem[comp_name]

            if device_str.startswith("fgp:"):
                device_list = device_str[4:].split(",")
                comp_strategy = "fgp_nvlink" if profile.has_fast_interconnect() else "fgp_pcie"
            elif device_str.startswith("tp:"):
                device_list = device_str[3:].split(",")
                comp_strategy = "tp"
            elif device_str.startswith("zero3:"):
                # CPU offload with GPU compute — device is the GPU, weights on CPU
                gpu_device = device_str[6:]
                device_list = [gpu_device]
                comp_strategy = "zero3"
            else:
                device_list = [device_str]
                comp_strategy = "single_gpu"

            components[comp_name] = ComponentAllocation(
                name=comp_name,
                devices=device_list,
                dtype=comp_dtypes.get(comp_name, torch.float32),
                memory_mb=mem.total_mb,
                architecture=arch,
                vendor=vendor,
                sharded=len(device_list) > 1,
                shard_map=shard_map,
                strategy=comp_strategy,
            )
            total_mb += mem.total_mb

        # Determine loading mode — strategy-driven
        # Eager: weights stay in VRAM permanently (fast serving)
        # Lazy: weights swap in/out per execution phase (large models)
        # Source of truth: AllocationStrategy.is_eager in structure.py
        from neurobrix.core.prism.structure import AllocationStrategy
        _EAGER_VALUES = {s.value for s in AllocationStrategy if s.is_eager}
        _LAZY_VALUES = {s.value for s in AllocationStrategy if not s.is_eager}
        if strategy in _EAGER_VALUES:
            loading_mode = "eager"
        elif strategy in _LAZY_VALUES:
            loading_mode = "lazy"
        else:
            # Unknown strategy — check if model fits in VRAM
            total_gpu_mb = sum(d.capacity_mb for d in devices if d.device_string.startswith("cuda"))
            loading_mode = "eager" if total_mb <= total_gpu_mb * 0.90 else "lazy"

        # Determine primary dtype
        primary_dtype = torch.float16
        for dt in comp_dtypes.values():
            if dt == torch.float32:
                primary_dtype = torch.float32
                break

        return ExecutionPlan(
            components=components,
            target_dtype=primary_dtype,
            total_memory_mb=total_mb,
            strategy=strategy,
            component_memory=comp_mem,
            loading_mode=loading_mode,
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _empty_plan(self, profile: PrismProfile) -> ExecutionPlan:
        return ExecutionPlan({}, torch.float32, 0.0, "empty", {}, "lazy")

    def _fail_error(self, sorted_comps, devices):
        total_req = sum(m.total_mb for _, m in sorted_comps)
        total_avail = sum(d.capacity_mb for d in devices)
        comp_info = "\n".join(f"  {n}: {m.total_mb:.0f}MB (W={m.weight_mb:.0f}, A={m.activation_mb:.0f})" for n, m in sorted_comps)
        dev_info = "\n".join(f"  {d.device_string}: {d.capacity_mb:.0f}MB" for d in devices)
        raise RuntimeError(
            f"ZERO FALLBACK: No strategy can fit this model.\n\n"
            f"Strategies tried: single_gpu, pipeline, fgp, tp - ALL FAILED\n\n"
            f"Components:\n{comp_info}\n\n"
            f"Total required: {total_req:.0f}MB\n\n"
            f"GPUs:\n{dev_info}\n\n"
            f"Total available: {total_avail:.0f}MB\n\n"
            f"Solutions:\n  1. Use larger GPUs\n  2. Reduce resolution/batch\n  3. Use smaller model"
        )

    def _print_summary(self, devices: List[DeviceState], plan: ExecutionPlan, profile: PrismProfile):
        """Print allocation summary."""
        pass



# =============================================================================
# IMPORT PLANNING (for NBX Importer)
# =============================================================================

@dataclass
class ImportPlan:
    """Plan for ONNX export (always CPU - GPU provides no benefit for symbolic tracing)."""
    device: str
    can_load: bool
    reason: str


class PrismImportPlanner:
    """Hardware planning for NBX Import phase. ONNX export is symbolic - always uses CPU."""

    def plan_import(self, safetensors_size_bytes: int) -> ImportPlan:
        size_gb = safetensors_size_bytes / (1024**3)
        return ImportPlan("cpu", True, f"CPU export ({size_gb:.1f}GB weights)")

    def verify_before_load(self, plan: ImportPlan) -> bool:
        return True


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def solve(container: "NBXContainer", profile: PrismProfile, input_config: Optional[InputConfig] = None) -> ExecutionPlan:
    """Convenience wrapper for PrismSolver.solve()"""
    return PrismSolver().solve(container, profile, input_config)
