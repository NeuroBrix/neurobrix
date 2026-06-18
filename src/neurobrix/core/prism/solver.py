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
import logging
import os
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
    dtype: str
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
    max_cache_len: int          # Max tokens cacheable — ceiling (Prism VRAM budget)
    num_layers: int
    num_kv_heads: int
    k_head_dim: int             # K head dimension (qk_nope + qk_rope for MLA, else head_dim)
    v_head_dim: int             # V head dimension (v_head_dim for MLA, else head_dim)
    dtype: str                   # Cache dtype string (may differ from weight dtype)
    memory_bytes: int           # Total KV cache memory budget (at max_cache_len)
    per_token_bytes: int        # Memory per cached token (for runtime info)
    initial_cache_len: int = 0  # Initial buffer size (0 = allocate at max_cache_len)

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
    target_dtype: str
    total_memory_mb: float
    strategy: str
    component_memory: Dict[str, ComponentMemory]
    loading_mode: str = "lazy"
    kv_cache_plan: Optional[KVCachePlan] = None
    cpu_ram_mb: int = 0  # CPU RAM budget for offload strategies
    # Op-level tiling — per-component plan emitted when a single op's
    # output+workspace exceeds the assigned GPU's safe VRAM budget. Picked
    # up by RuntimeExecutor to wire op_uid interceptors on the component's
    # GraphExecutor. Empty dict when no op-level tiling is required.
    runtime_op_tiling: Dict = field(default_factory=dict)
    # Component-level spatial tiling — per-component plan emitted when a
    # spatial component (4D/5D input + scale config) would NOT fit a GPU
    # untiled (so it would otherwise be offloaded to host RAM) but DOES fit
    # when its decode is run on overlapping spatial tiles. Picked up by
    # RuntimeExecutor to instantiate a TilingEngine with the Prism-chosen
    # tile_size, keeping the component on GPU. Empty dict when no
    # component-level tiling is required (e.g. the video VAE at high
    # resolution; reuses the R31 TilingEngine brick). Each entry:
    # {"tile_size", "scale_factor", "overlap", "window_alignment"}.
    component_tiling: Dict = field(default_factory=dict)

    @property
    def primary_device(self) -> str:
        return next(iter(self.components.values())).device if self.components else "cpu"

    @property
    def dtype(self) -> str:
        return self.target_dtype

    def get_allocation(self, component_name: str) -> Optional[ComponentAllocation]:
        return self.components.get(component_name)

    def get_device(self, component_name: str) -> str:
        alloc = self.components.get(component_name)
        return alloc.device if alloc else "cpu"

    def get_memory_breakdown(self, component_name: str) -> Optional[ComponentMemory]:
        return self.component_memory.get(component_name)

    def to_dict(self, hardware_profile: str = "unknown") -> Dict[str, Any]:
        return {
            "version": "0.1.0",
            "created_at": datetime.now().isoformat(),
            "hardware_profile": hardware_profile,
            "strategy": self.strategy,
            "loading_mode": self.loading_mode,
            "target_dtype": self.target_dtype,
            "total_memory_mb": self.total_memory_mb,
            "components": {
                name: {
                    "devices": alloc.devices,
                    "primary_device": alloc.device,
                    "dtype": alloc.dtype,
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
                "dtype": self.kv_cache_plan.dtype,
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

        components = {
            name: ComponentAllocation(
                name=name,
                devices=comp["devices"],
                dtype=comp.get("dtype", "float32"),
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
                dtype=kv_data.get("dtype", "float16"),
                memory_bytes=int(kv_data.get("memory_mb", 0) * 1024 * 1024),
                per_token_bytes=kv_data.get("per_token_bytes", 0),
            )

        return cls(
            components=components,
            target_dtype=data.get("target_dtype", "float32"),
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

        In serve mode: tries hot budget first (all weights resident). If no strategy
        fits hot mode, falls back to cold budget and warns the user that the daemon
        will run in cold mode (weights swap per phase, not near-zero latency).

        Strategies tried in order:
        1. single_gpu - All on largest GPU
        2. component_placement - Whole components on different GPUs
        3. pipeline_parallel - Per-layer sequential fill (like Accelerate)
        4. block_scatter - Block-level best-fit distribution
        5. weight_sharding - Weight-file round-robin
        6. component_placement_lazy - Component placement with lazy swap
        7. lazy_sequential - One component at a time
        8. zero3 - CPU offload
        """
        self._serve_mode = serve_mode
        self._serve_cold_fallback = False  # Track if serve degraded to cold

        neural_components = container.get_neural_components()
        if not neural_components:
            return self._empty_plan(profile)

        if input_config is None:
            batch = PRISM_DEFAULTS.get("default_batch_size", 2)
            input_config = InputConfig(batch_size=batch, height=1024, width=1024)

        neural_components = sorted(neural_components, key=lambda c: c.name)

        # Step 0a: Detect op-level tiling needs per component (fused
        # upsample→conv pairs whose intermediate tensor would OOM). Done
        # AFTER strategies are tried — only if the cascade can't place the
        # component otherwise. For Sana 4Kpx VAE: upsample::3→conv::55 and
        # upsample::4→conv::62 (16 GB intermediate at 4Kpx).
        # The actual scan happens after _evaluate_all_strategies — see
        # _detect_op_level_tiling_pairs() below. This comment marks where
        # the data lives in the plan: plan.runtime_op_tiling.

        # Step 0: Determine model category from manifest family — DATA-DRIVEN
        # Reads execution.has_kv_cache from config/families/<family>.yml plus
        # topology.flow.generation.type to discriminate VQ-image autoregressive
        # paths from text autoregressive paths. Falls back to legacy detection
        # for the unknown case (defensive).
        manifest = container.get_manifest() or {}
        family = manifest.get("family", "unknown")
        has_lm_config = bool(self._read_lm_config(container))
        family_kv_cache = False
        try:
            from neurobrix.core.config import get_family_config
            family_kv_cache = bool(
                get_family_config(family).get("execution", {}).get("has_kv_cache", False)
            )
        except (FileNotFoundError, RuntimeError):
            pass

        # Inspect topology for autoregressive_image (VQ multimodal Janus pattern)
        gen_type = ""
        try:
            topology = container.get_topology() or {}
            gen_type = topology.get("flow", {}).get("generation", {}).get("type", "") or ""
        except Exception:
            gen_type = ""
        is_image_vq = (gen_type == "autoregressive_image") and has_lm_config

        if is_image_vq:
            self._model_category = "image_vq"
        elif family_kv_cache:
            self._model_category = "llm"
        else:
            self._model_category = "diffusion"

        self._needs_kv_cache = self._model_category in ("llm", "image_vq")

        # Step 1: Resolve target dtype (model-level — used for KV-cache sizing
        # and the genuine all-fp32 fallback below).
        target_dtype = self._resolve_dtype(container, profile)
        target_dtype_str = str(target_dtype).split('.')[-1]
        self._target_dtype_str = target_dtype_str

        # Step 1.5: Resolve PER-COMPONENT dtype for the memory estimate.
        # `_resolve_dtype` collapses the WHOLE model to float32 as soon as a
        # single component is fp32-forced (e.g. a VAE-class component on
        # V100). Estimating every component at that model-wide float32
        # over-counts each fp16 component 2x (CogVideoX-5b transformer:
        # 21 GB estimated vs 11 GB real fp16) and can wrongly offload a
        # component that fits. Estimate each component at the dtype it
        # ACTUALLY runs — the same per-component map the executor uses
        # downstream (`_resolve_component_dtypes`, Step 5).
        component_dtypes = self._resolve_component_dtypes(
            neural_components, profile, container)

        # Component-level spatial tiling accumulator. Populated by
        # _place_component when a spatial-overflow component is kept on GPU
        # via tiling instead of host offload; surfaced on the plan for the
        # executor to instantiate a TilingEngine. _input_config is stashed so
        # the tile sizer can read the runtime spatial extent.
        self._component_tiling = {}
        self._input_config = input_config

        # Step 2: Compute memory requirements (tiling-aware via profile)
        component_memory = self._compute_memory(
            container, neural_components, input_config, target_dtype_str,
            profile=profile, component_dtypes=component_dtypes,
        )
        sorted_components = sorted(component_memory.items(), key=lambda x: -x[1].total_bytes)

        # Step 3: Prepare devices
        devices = self._prepare_devices(profile)

        # Step 4: Try ALL strategies and pick best by score.
        # Doctrine R35 — Prism never refuses. The cascade always ends with
        # `cpu_execution` (everything on CPU, perf libre). This is the last-
        # resort entry of the cascade for any hardware that exposes CPU +
        # RAM, including profiles with zero GPUs (developer machines, CI
        # runners, dev boxes, future Mac Studio).
        if not devices:
            # CPU-only profile: skip the entire GPU cascade and jump
            # straight to cpu_execution.
            strategies = [
                ("cpu_execution", self._try_cpu_execution),
            ]
        elif len(devices) == 1:
            # Single-GPU shortcut: skip multi-GPU strategies (Apple Silicon,
            # single NVIDIA, single RDNA, ...). Cascade ends in
            # cpu_execution (R35).
            strategies = [
                ("single_gpu", self._try_single_gpu),
                ("single_gpu_lifecycle", self._try_single_gpu_lifecycle),
                ("lazy_sequential", self._try_lazy_sequential),
                ("zero3", self._try_zero3),
                ("cpu_execution", self._try_cpu_execution),
            ]
        else:
            strategies = [
                ("single_gpu", self._try_single_gpu),
                ("single_gpu_lifecycle", self._try_single_gpu_lifecycle),
                ("component_placement", self._try_component_placement),
                ("pipeline_parallel", self._try_pipeline_parallel),
                ("block_scatter", self._try_block_scatter),
                ("weight_sharding", self._try_weight_sharding),
                ("component_placement_lazy", self._try_component_placement_lazy),
                ("lazy_sequential", self._try_lazy_sequential),
                ("zero3", self._try_zero3),
                ("cpu_execution", self._try_cpu_execution),
            ]

        # NBX_FORCE_STRATEGY: deterministic single-strategy selection for
        # matrix validation and debugging. Filter the cascade down to the
        # requested strategy; zero fallback if it cannot fit (the caller
        # asked for X specifically, picking Y silently would hide bugs).
        forced = os.environ.get("NBX_FORCE_STRATEGY", "").strip()
        if forced:
            valid = {
                "single_gpu", "single_gpu_lifecycle",
                "component_placement", "pipeline_parallel",
                "block_scatter", "weight_sharding",
                "component_placement_lazy", "lazy_sequential",
                "zero3",
                "cpu_execution",
            }
            if forced not in valid:
                raise RuntimeError(
                    f"NBX_FORCE_STRATEGY='{forced}' is invalid. "
                    f"Valid values: {sorted(valid)}"
                )
            filtered = [(n, fn) for n, fn in strategies if n == forced]
            if not filtered:
                raise RuntimeError(
                    f"NBX_FORCE_STRATEGY='{forced}' not available for "
                    f"this device count (len(devices)={len(devices)}). "
                    f"The single-GPU shortcut excludes multi-GPU-only "
                    f"strategies — run on a multi-GPU profile or pick "
                    f"one of: "
                    f"{sorted(n for n, _ in strategies)}"
                )
            strategies = filtered
            logging.getLogger(__name__).info(
                f"[Prism] NBX_FORCE_STRATEGY={forced} — forcing strategy, "
                f"bypassing score cascade"
            )

        shard_sizes = container.get_shard_sizes()
        candidates = self._evaluate_all_strategies(
            strategies, sorted_components, component_memory, devices, shard_sizes, profile, container
        )

        # When NBX_FORCE_STRATEGY is set and the strategy cannot fit, the
        # rest of solve() would fall through to fp32 fallback or the final
        # _fail_error path. Neither is desirable when the operator has
        # explicitly requested one strategy — they need to see the plain
        # "this strategy does not fit" signal without Prism retrying
        # alternatives.
        if forced and not candidates:
            raise RuntimeError(
                f"ZERO FALLBACK: NBX_FORCE_STRATEGY={forced} cannot fit "
                f"the model on the given hardware profile. Remove the env "
                f"var to let Prism's cascade select an alternative, or "
                f"use a larger hardware profile."
            )

        # Serve mode fallback: if hot mode failed, retry with cold budget
        # User chose serve but hardware can't keep all weights resident →
        # degrade gracefully to cold mode instead of crashing
        if not candidates and serve_mode:
            logging.getLogger(__name__).warning(
                "[Prism] Serve mode: hot budget doesn't fit — falling back to cold mode.\n"
                "The daemon will load/unload weights per request (not near-zero latency).\n"
                "For hot serve, use a GPU with more VRAM or a multi-GPU setup."
            )
            self._serve_mode = False  # Retry with cold budgets
            self._serve_cold_fallback = True
            devices = self._prepare_devices(profile)
            candidates = self._evaluate_all_strategies(
                strategies, sorted_components, component_memory, devices, shard_sizes, profile, container
            )
            self._serve_mode = serve_mode  # Restore for KV cache sizing

        # FP32 fallback for BF16 models
        if not candidates:
            if self._try_fp32_fallback(container, profile):
                target_dtype_str = "float32"
                component_memory = self._compute_memory(
                    container, neural_components, input_config, target_dtype_str,
                    profile=profile,
                )
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
                if strat_name == "cpu_execution":
                    # CPU-only: capacity = host RAM budget. KV cache lives
                    # in host RAM alongside weights and activations.
                    # Use 0.7 × ram_mb (matches `_try_cpu_execution`'s
                    # budget formula, R34 generic).
                    if profile.cpu and profile.cpu.ram_mb > 0:
                        total_capacity = int(profile.cpu.ram_mb * 0.7 * 1024 * 1024)
                    else:
                        # No CPU stats — accept unconditionally (runtime
                        # will fail clean if RAM truly insufficient).
                        total_capacity = 2**62
                elif strat_name in ("single_gpu", "single_gpu_lifecycle"):
                    # Single GPU: capacity = largest GPU only
                    total_capacity = int(max(d.capacity_mb for d in strat_devices) * 1024 * 1024)
                else:
                    total_capacity = sum(int(d.capacity_mb * 1024 * 1024) for d in strat_devices)

                if strat_name == "zero3":
                    total_allocated = sum(m.activation_bytes + m.overhead_bytes
                                         for m in component_memory.values())
                elif strat_name == "single_gpu_lifecycle":
                    persistent, transient = self._classify_lifecycle(container)
                    # Persistent: all weights resident + peak activation
                    persistent_weights = sum(
                        m.weight_bytes for name, m in component_memory.items()
                        if name in persistent
                    )
                    persistent_max_act = max(
                        (m.activation_bytes for name, m in component_memory.items() if name in persistent),
                        default=0
                    )
                    # Transient: max one at a time (weight + activation)
                    max_transient = max(
                        (m.total_bytes for name, m in component_memory.items() if name in transient),
                        default=0
                    )
                    total_allocated = persistent_weights + persistent_max_act + max_transient
                elif strat_name == "single_gpu":
                    # Hot/cold budget must match _try_single_gpu's decision
                    serve_mode = getattr(self, '_serve_mode', False)
                    if serve_mode:
                        # Hot: all weights resident + peak activations
                        total_weights = sum(m.weight_bytes for m in component_memory.values())
                        max_act = max((m.activation_bytes for m in component_memory.values()), default=0)
                        total_allocated = total_weights + max_act
                    else:
                        # Cold: peak of one component at a time
                        total_allocated = max(m.total_bytes for m in component_memory.values())
                else:
                    total_allocated = sum(m.total_bytes for m in component_memory.values())

                # Remove KV cache double-count (already in LM activation_bytes)
                total_allocated = max(total_allocated - kv_already_counted, 0)
                remaining = max(total_capacity - total_allocated, 0)
                try:
                    kv_plan = self._compute_kv_cache_plan(container, target_dtype, remaining)
                except RuntimeError:
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

        # Component-level spatial tiling decided during placement (Strategy 3.5
        # in _place_component) — keep only entries whose final allocation
        # actually landed on a GPU (drop any stale flag from a rejected
        # strategy attempt where the component ended up elsewhere).
        _ct = getattr(self, "_component_tiling", {}) or {}
        if _ct:
            plan.component_tiling = {
                cn: spec for cn, spec in _ct.items()
                if cn in plan.components
                and str(plan.components[cn].device).startswith("cuda")
            }

        # Step 7b: Op-level tiling — detect upsample→conv fusion pairs whose
        # intermediate tensor would OOM the assigned GPU. Decision happens
        # AFTER strategies are picked (allocations known) so we know the per-
        # component VRAM budget. Plan stored in plan.runtime_op_tiling for
        # the runtime executor to wire as op_uid interceptors.
        plan.runtime_op_tiling = self._detect_op_level_tiling_pairs(
            container, neural_components, allocations, profile, input_config,
            target_dtype_str,
        )

        # Step 8: Summary
        self._print_summary(devices, plan, profile)
        return plan

    def _detect_op_level_tiling_pairs(
        self, container, components, allocations, profile, input_config,
        target_dtype_str: str,
    ):
        """For each component, scan the DAG for upsample→conv pairs whose
        intermediate tensor exceeds the assigned GPU's safe VRAM budget.

        Returns: Dict[comp_name, OpLevelTilingPlan]. Empty dict if no
        component needs op-level tiling.
        """
        from neurobrix.core.module.tiling_engine import OpLevelTilingPlan
        from neurobrix.core.prism.profiler import ActivationProfiler
        from neurobrix.core.prism.memory_estimator import (
            get_dtype_bytes_per_element, estimate_op_workspace_bytes,
        )

        result = {}
        # Per-GPU capacity in bytes — pick the smallest assigned device for
        # this component (op-level tiling only fires if the worst-case GPU
        # can't hold the op).
        device_caps = {}
        for dev in profile.devices:
            device_caps[dev.get_device_string()] = int(dev.memory_mb * 1024 * 1024)

        dtype_bytes = get_dtype_bytes_per_element(target_dtype_str)

        for comp in components:
            if comp.graph is None:
                continue
            alloc = allocations.get(comp.name)
            if alloc is None:
                continue
            assigned_dev_str = alloc[0] if isinstance(alloc, tuple) else alloc.device
            comp_vram = device_caps.get(assigned_dev_str, 0)
            if comp_vram == 0:
                continue

            profiler = ActivationProfiler(comp.graph)
            ap = profiler.estimate_peak_memory(
                input_config=input_config,
                dtype_bytes=dtype_bytes,
                vram_per_gpu_bytes=comp_vram,
                mode="compiled",
                safety=0.85,
            )
            # S5: residual chains are detected unconditionally — even
            # when no overflow_ops are reported on this hardware, the
            # chain wrapper is structurally beneficial (it lowers the
            # transient peak by 2-3× per chain, which is what the
            # estimator's zero_uids relies on). When chains are detected
            # we keep going to build the plan; otherwise the legacy
            # short-circuit applies.
            has_chains = bool(
                self._identify_residual_chain_specs(comp.graph)
            )
            if not ap.overflow_ops and not has_chains:
                continue

            # Look for upsample→conv adjacency in execution_order
            ops = comp.graph.get("ops", {})
            order = comp.graph.get("execution_order", [])
            # Graph-aware symbol map (name-driven video/time overrides) —
            # same map estimate_peak_memory used to produce ap.overflow_ops.
            comp_symbol_map = profiler.build_symbol_map(input_config)
            uid_to_step = {uid: i for i, uid in enumerate(order)}
            # Build consumer map: tensor_id -> list of (consumer_uid, step)
            consumers = {}
            for uid in order:
                op = ops.get(uid, {})
                for tid in op.get("input_tensor_ids", []):
                    consumers.setdefault(tid, []).append((uid, uid_to_step[uid]))

            plan = OpLevelTilingPlan(comp.name)
            seen_pair_convs = set()
            for uid in order:
                op = ops.get(uid, {})
                if "upsample" not in op.get("op_type", ""):
                    continue
                # Find the unique consumer
                out_tids = op.get("output_tensor_ids", [])
                if len(out_tids) != 1:
                    continue
                cons = consumers.get(out_tids[0], [])
                if len(cons) != 1:
                    continue  # not a single-consumer upsample → can't fuse
                conv_uid, conv_step = cons[0]
                conv_op = ops.get(conv_uid, {})
                if "convolution" not in conv_op.get("op_type", ""):
                    continue
                if conv_uid in seen_pair_convs:
                    continue
                # Only fuse if the conv is in overflow_ops (i.e. it would OOM)
                conv_overflow = any(o[0] == conv_uid for o in ap.overflow_ops)
                # OR if the upsample output itself is large enough to OOM
                up_in_tids = op.get("input_tensor_ids", [])
                up_in_shape = []
                for tid in up_in_tids[:1]:
                    meta = comp.graph["tensors"].get(tid, {})
                    up_in_shape.append(profiler._resolve_shape(
                        meta, comp_symbol_map
                    ))
                up_out_meta = comp.graph["tensors"].get(out_tids[0], {})
                up_out_shape = profiler._resolve_shape(
                    up_out_meta, comp_symbol_map
                )
                up_out_bytes = profiler._compute_size(up_out_shape, up_out_meta, dtype_bytes)
                # Threshold: upsample output that exceeds 25% of VRAM is
                # worth fusing (typical: 16 GB on 32 GB GPU).
                up_overflow = up_out_bytes > 0.25 * comp_vram
                if not (conv_overflow or up_overflow):
                    continue

                # Compute tile_factor analytically. Each band materializes:
                #   up_band         = up_out_bytes / tile_factor
                #   cudnn_workspace = workspace_full_bytes / tile_factor
                #   conv_band       = conv_out_bytes / tile_factor
                # And the conv output (full resolution) stays allocated:
                #   conv_out_full = conv_out_bytes
                # Target: per-band transient + conv_out_full ≤ safety × VRAM
                # → tile_factor ≥ (up_out + cudnn_ws + conv_out) /
                #                  (budget − conv_out_full)
                conv_in_shapes = []
                for tid in conv_op.get("input_tensor_ids", []):
                    meta = comp.graph["tensors"].get(tid, {})
                    conv_in_shapes.append(profiler._resolve_shape(
                        meta, comp_symbol_map
                    ))
                conv_out_shapes = []
                conv_out_bytes = 0
                for tid in conv_op.get("output_tensor_ids", []):
                    meta = comp.graph["tensors"].get(tid, {})
                    sh = profiler._resolve_shape(meta, comp_symbol_map)
                    conv_out_shapes.append(sh)
                    conv_out_bytes += profiler._compute_size(sh, meta, dtype_bytes)
                cudnn_ws_bytes = estimate_op_workspace_bytes(
                    "compiled", conv_op.get("op_type", ""),
                    conv_in_shapes, conv_out_shapes, dtype_bytes,
                    vram_per_gpu_bytes=comp_vram,
                )
                # Per-band budget: total VRAM × 0.65 (safety) minus the
                # conv output that stays full-resolution allocated.
                budget_per_band = int(0.65 * comp_vram) - conv_out_bytes
                if budget_per_band < (256 * 1024 * 1024):  # < 256 MB usable
                    budget_per_band = 256 * 1024 * 1024
                total_to_tile = up_out_bytes + cudnn_ws_bytes + conv_out_bytes
                import math
                tile_factor = max(2, math.ceil(total_to_tile / budget_per_band))
                # Round up to power-of-2 for halo alignment.
                pow2 = 2
                while pow2 < tile_factor and pow2 < 64:
                    pow2 *= 2
                tile_factor = pow2

                plan.add_upsample_conv_fusion(uid, conv_uid, tile_factor)
                seen_pair_convs.add(conv_uid)

            # Standalone conv overflow (no upsample to fuse) — tile spatially
            # so cuDNN workspace stays bounded per band. Skip convs already
            # picked up by a fusion pair above.
            for op_uid, op_type, out_b, ws_b, in_tids in ap.overflow_ops:
                if op_uid in seen_pair_convs:
                    continue
                if "convolution" not in op_type:
                    continue  # only conv tiling implemented for now
                # Skip tiling for depthwise convolutions: the cuDNN
                # workspace estimator overshoots them (depthwise
                # actually needs a tiny workspace — kh*kw per channel),
                # so they get tile-flagged here even though they don't
                # need to be tiled. Tiling these convs introduces a
                # numerical divergence on 16g specifically (where they
                # trigger; on 32g the budget threshold is higher and
                # they pass through native).
                # Detection: depthwise weight shape is [out_c, 1, kh,
                # kw] (in_channels_per_group == 1).
                # Diagnosed factually 2026-05-13 via per-op bit-diff:
                # transformer block depthwise convs (block.X.ffn.conv_
                # depth) were the first divergent ops on 16g.
                _op_data = ops.get(op_uid, {})
                _in_shapes = _op_data.get("input_shapes", [])
                if (len(_in_shapes) >= 2 and len(_in_shapes[1]) == 4
                        and int(_in_shapes[1][1]) == 1):
                    # Depthwise — skip tiling.
                    continue
                # The spatial band-streaming wrappers (tiled_conv2d_spatial /
                # _pair) are rank-4 ONLY — they tile a 2D conv [B, C, H, W].
                # Any other input rank crashes at dispatch:
                #   - 5D conv3d (CogVideoX VAE): 'too many values to unpack'
                #     in _tiled_conv2d_spatial_torch (waits on the 5D tiling
                #     chantier; runs native, fits at proof sizes, OOMs visibly).
                #   - 3D conv1d (Kokoro iSTFTNet vocoder noise_convs / the audio
                #     decoders generally): _pair reads stride[1] on a 1-element
                #     stride list -> IndexError. A 1D audio conv is never a
                #     2D-spatial overflow anyway — it runs native (small).
                # Only 2D convs (rank-4 input) get spatial tiling.
                if (_in_shapes and _in_shapes[0] is not None
                        and len(_in_shapes[0]) != 4):
                    continue
                # Diagnostic gate: NBX_S5_SKIP_CONV_TILE=1 skips all
                # remaining standalone convolution tiling.
                import os as _os_skipc
                if _os_skipc.environ.get("NBX_S5_SKIP_CONV_TILE") == "1":
                    continue
                # tile_factor: same formula as fusion path but without the
                # upsample term (no intermediate to absorb).
                budget_per_band = int(0.65 * comp_vram) - out_b
                if budget_per_band < (256 * 1024 * 1024):
                    budget_per_band = 256 * 1024 * 1024
                total_to_tile = ws_b + out_b
                import math
                tf = max(2, math.ceil(total_to_tile / budget_per_band))
                pow2 = 2
                while pow2 < tf and pow2 < 64:
                    pow2 *= 2
                plan.add_tiled_op(op_uid, op_type, pow2)

            # Custom rms_norm tiling — rms_norm normalizes along the last
            # (channel) dim, so each H row is independent. Tile by H bands
            # to bound the per-band materialized output. Fires on rms_norm
            # ops whose output size > 25% of VRAM (same threshold as fusion).
            # 20% of VRAM — strict less-than-or-equal would skip exactly-8GB
            # tensors on a 32GB GPU which is precisely Sana 4Kpx's pattern.
            ovf_threshold = 0.20 * comp_vram
            for uid in order:
                op = ops.get(uid, {})
                if op.get("op_type", "") != "custom::rms_norm":
                    continue
                out_tids = op.get("output_tensor_ids", [])
                if not out_tids:
                    continue
                out_meta = comp.graph["tensors"].get(out_tids[0], {})
                out_sh = profiler._resolve_shape(
                    out_meta, comp_symbol_map
                )
                out_b = profiler._compute_size(out_sh, out_meta, dtype_bytes)
                if out_b <= ovf_threshold:
                    continue
                # Per-band budget tightened from 0.65× to 0.20× of
                # comp_vram. The original 0.65× assumed the rms_norm
                # wrapper ran in isolation (input + output ≈ full
                # budget). With S5 chains active upstream, the
                # pre-rms_norm baseline carries ~5-7 GiB of chain
                # state into the rms_norm dispatch; we need each
                # band's transient (fp32 cast of x_band) to fit
                # alongside that residue. 0.20× → tile_factor at
                # least ceil(out_b / 0.20×comp_vram), e.g. 4 GiB
                # output on 16 GiB GPU → tile_factor=4 (band 1 GiB,
                # fp32 cast 2 GiB transient, fits with chain
                # residue). S5 follow-up: in-place rms_norm to
                # eliminate the full output buffer entirely; for
                # now the conservative tile_factor preserves
                # numerical correctness over the in-place attempt
                # that produced a black PNG.
                budget_per_band = int(0.10 * comp_vram)
                import math
                tf = max(4, math.ceil(out_b / budget_per_band))
                pow2 = 2
                while pow2 < tf and pow2 < 64:
                    pow2 *= 2
                # Cap at tile_factor=8: empirically 16 produces a
                # numerically wrong output on the Sana 4Kpx
                # rms_norm::27 site (post-chain wrapper, NHWC view of
                # non-contig storage). tile_factor=8 was validated
                # coherent on 32g and is the largest viable value
                # observed. S5 follow-up: identify whether the issue
                # is the band_h=256 boundary stride math or some other
                # PyTorch issue with non-contig views + small bands.
                pow2 = min(pow2, 8)
                plan.add_tiled_op(uid, "custom::rms_norm", pow2)

            # P-PRISM-NEVER-REFUSE v2 S5: residual chain detection.
            # Add specs of detected chains to the plan so the runtime's
            # `OpLevelTilingEngine.register_into_graph_executor` wires
            # band-streaming interceptors on every chain op. Detection
            # is structural (R34): only models that match the
            # fork→linear-N≥3→merge signature with large intermediates
            # get any chain spec; cheaper models pay nothing.
            chain_specs = self._identify_residual_chain_specs(comp.graph)
            for spec in chain_specs:
                # Tile factor: aim for band size ≤ 1 GiB at runtime.
                # bytes_fp32 / 2 = bf16 estimate, divide by 1 GiB.
                import math as _math
                bf16_bytes = spec.get("bytes_fp32", 0) // 2
                tf = max(2, _math.ceil(bf16_bytes / (1024 ** 3)))
                pow2 = 2
                while pow2 < tf and pow2 < 32:
                    pow2 *= 2
                spec_with_tf = dict(spec)
                spec_with_tf["tile_factor"] = pow2
                plan.add_residual_chain(spec_with_tf)

            if not plan.is_empty():
                result[comp.name] = plan

        return result

    def _identify_residual_chain_specs(self, graph: Dict):
        """Return raw residual-chain specs from a DAG dict. Wrapper around
        `OpLevelTilingEngine._detect_residual_chains` that accepts a raw
        graph dict so the solver does not depend on a live GraphExecutor.
        S5 2026-05-13."""
        from neurobrix.core.module.tiling_engine import OpLevelTilingEngine

        class _DagWrap:
            def __init__(self, d):
                self._dag = d

        return OpLevelTilingEngine._detect_residual_chains(_DagWrap(graph))

    def solve_smart(self, container, profile, input_config=None, serve_mode: bool = False):
        """Backward compatibility alias for solve()."""
        return self.solve(container, profile, input_config, serve_mode=serve_mode)

    # =========================================================================
    # MEMORY COMPUTATION - Full Intelligence
    # =========================================================================

    def _identify_inplace_add_candidates_static(
        self, graph: Dict, threshold_bytes: int = 1024 * 1024 * 1024,
    ):
        """DAG-static replica of `OpLevelTilingEngine._detect_inplace_add_candidates`
        (`tiling_engine.py:613`) — returns the same list of `(op_uid,
        reuse_input_idx)` for residual `aten::add` ops where one input has
        its last use at the add (so its buffer can be reused as output).
        Operates on `graph.json` `input_tensor_ids` (the post-import
        canonical form) instead of the runtime version's
        `attributes.args` walk; both yield the same set on a
        well-formed graph (input_tensor_ids is built from args at import
        time). Threshold default 1 GiB matches the runtime helper.
        Same threshold formula MUST match — divergence would mean the
        estimator predicts an in-place that the runtime won't actually
        engage. P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE 2026-05-11.
        """
        ops = graph.get("ops", {})
        order = graph.get("execution_order", [])
        output_ids = set(graph.get("output_tensor_ids", []))
        consumers: Dict[str, List[str]] = {}
        for uid in order:
            op = ops.get(uid, {})
            for tid in op.get("input_tensor_ids", []):
                consumers.setdefault(tid, []).append(uid)
        candidates: List[Tuple[str, int]] = []
        for uid in order:
            op = ops.get(uid, {})
            if op.get("op_type") != "aten::add":
                continue
            in_tids = op.get("input_tensor_ids", [])
            ishapes = op.get("input_shapes", [])
            oshapes = op.get("output_shapes", [])
            if len(in_tids) < 2 or len(ishapes) < 2 or not oshapes:
                continue
            sh_a = ishapes[0]
            sh_b = ishapes[1]
            sh_o = oshapes[0]
            if sh_a != sh_b or len(sh_a) != 4:
                continue
            elems = 1
            for d in sh_o:
                elems *= max(1, int(d))
            out_bytes = elems * 4  # fp32 conservative — matches runtime helper
            if out_bytes < threshold_bytes:
                continue
            a_consumers = consumers.get(in_tids[0], [])
            b_consumers = consumers.get(in_tids[1], [])
            a_last = (len(a_consumers) == 1
                      and a_consumers[0] == uid
                      and in_tids[0] not in output_ids)
            b_last = (len(b_consumers) == 1
                      and b_consumers[0] == uid
                      and in_tids[1] not in output_ids)
            if a_last:
                candidates.append((uid, 0))
            elif b_last:
                candidates.append((uid, 1))
        return candidates

    def _identify_residual_chain_proxy_uids(self, graph: Dict):
        """Identify the chain-intermediate op_uids that
        `OpLevelTilingEngine._detect_residual_chains` will band-stream at
        runtime. Each intermediate writes a band-sized buffer at any
        given moment, not the full-resolution buffer it would otherwise.
        The estimator approximates this by treating the chain
        intermediates as zero-alloc, which gives a slightly optimistic
        peak; the existing 3 GiB `_OOM_RESERVE_MB` in `_try_single_gpu`
        absorbs the residual band transient comfortably (band size ≪
        full intermediate at tile_factor ≥ 4 on 4 GiB-class tensors).
        P-PRISM-NEVER-REFUSE v2 S5 2026-05-13.

        The fork op (the producer of `T_base`) and the merge op (the
        final `aten::add`) are NOT in the returned set — both keep
        their full buffer at runtime. The wrapper sees them as
        legitimate live tensors throughout the band loop.

        Routes through `OpLevelTilingEngine._detect_residual_chains`
        via a lightweight DAG wrapper so the structural signature
        stays in one place (avoids the drift risk noted above for
        `_identify_pixel_shuffle_chain_proxy_uids`).
        """
        from neurobrix.core.module.tiling_engine import OpLevelTilingEngine

        class _DagWrap:
            def __init__(self, d):
                self._dag = d

        chains = OpLevelTilingEngine._detect_residual_chains(_DagWrap(graph))
        proxy_uids: set = set()
        for c in chains:
            for uid in c.get("chain_uids", []):
                proxy_uids.add(uid)
        return proxy_uids

    def _identify_pixel_shuffle_chain_proxy_uids(self, graph: Dict):
        """Identify expand/clone/view op_uids that are part of an
        `expand → clone → view → pixel_shuffle` chain that
        `OpLevelTilingEngine._detect_pixel_shuffle_broadcast_chains` will
        intercept at runtime. The expand returns a stride-0 view (no
        allocation), clone returns a `BroadcastClonePyroxy` sentinel,
        and view is a pass-through — none of these materialize a real
        tensor at runtime. P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE
        2026-05-11.

        Note: the pixel_shuffle output IS materialized at runtime (its
        broadcast-aware kernel allocates a real output), so the
        pixel_shuffle uid is NOT in the returned set.

        Replicates the detection pattern from
        `tiling_engine.py:_detect_pixel_shuffle_broadcast_chains:469`
        exactly: same chain structure (single-consumer adjacency
        expand→clone→view→pixel_shuffle), same broadcast-dim detection
        (in_shape[d]==1 → out_shape[d]>1, exactly one such d). Mismatch
        with the runtime detection → estimator zeroes ops that
        runtime won't intercept → over-optimistic peak.
        """
        proxy_uids = set()
        ops = graph.get("ops", {})
        order = graph.get("execution_order", [])
        # Build consumer map from input_tensor_ids (the runtime version
        # walks `attributes.args` for tensor refs; the post-import
        # graph.json's `input_tensor_ids` is the canonical equivalent).
        consumers: Dict[str, List[str]] = {}
        for uid in order:
            op = ops.get(uid, {})
            for tid in op.get("input_tensor_ids", []):
                consumers.setdefault(tid, []).append(uid)
        for uid in order:
            op = ops.get(uid, {})
            if op.get("op_type") != "aten::expand":
                continue
            in_shapes = op.get("input_shapes", [])
            out_shapes = op.get("output_shapes", [])
            if not in_shapes or not out_shapes:
                continue
            in_sh = in_shapes[0]
            out_sh = out_shapes[0]
            if len(in_sh) != len(out_sh):
                continue
            broadcast_dim = -1
            for d in range(len(in_sh)):
                if in_sh[d] == 1 and out_sh[d] > 1:
                    if broadcast_dim != -1:
                        broadcast_dim = -1
                        break
                    broadcast_dim = d
            if broadcast_dim == -1:
                continue
            expand_out_tids = op.get("output_tensor_ids", [])
            if not expand_out_tids:
                continue
            expand_consumers = consumers.get(expand_out_tids[0], [])
            if len(expand_consumers) != 1:
                continue
            clone_uid = expand_consumers[0]
            clone_op = ops.get(clone_uid, {})
            if clone_op.get("op_type") != "aten::clone":
                continue
            clone_out_tids = clone_op.get("output_tensor_ids", [])
            if not clone_out_tids:
                continue
            clone_consumers = consumers.get(clone_out_tids[0], [])
            if len(clone_consumers) != 1:
                continue
            view_uid = clone_consumers[0]
            view_op = ops.get(view_uid, {})
            if view_op.get("op_type") != "aten::view":
                continue
            view_out_tids = view_op.get("output_tensor_ids", [])
            if not view_out_tids:
                continue
            view_consumers = consumers.get(view_out_tids[0], [])
            if len(view_consumers) != 1:
                continue
            ps_uid = view_consumers[0]
            ps_op = ops.get(ps_uid, {})
            if ps_op.get("op_type") != "aten::pixel_shuffle":
                continue
            # Chain confirmed — expand/clone/view are zero-alloc at runtime.
            proxy_uids.add(uid)
            proxy_uids.add(clone_uid)
            proxy_uids.add(view_uid)
        return proxy_uids

    def _identify_fusion_upsample_uids(
        self, graph: Dict, overflow_ops, input_config: InputConfig,
        dtype_bytes: int, comp_vram_bytes: int,
    ):
        """Identify upsample op_uids that will be fused with their consumer
        conv by `_detect_op_level_tiling_pairs` at runtime — replicates the
        same eligibility logic exactly (`conv_overflow OR up_overflow`,
        same `0.25 * comp_vram` threshold) so the tiling-aware activation
        estimate and the runtime tiling plan agree on which upsamples are
        materialized as `FusionUpsampleProxy` (size=0).

        P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE: critical to keep the
        threshold formula synced with `_detect_op_level_tiling_pairs:819`.
        Mismatch → estimator and runtime disagree about which upsamples
        are "free" → estimate says "fits" but runtime won't fuse.
        """
        from neurobrix.core.prism.profiler import ActivationProfiler
        if comp_vram_bytes <= 0:
            return set()
        profiler = ActivationProfiler(graph)
        ops = graph.get("ops", {})
        order = graph.get("execution_order", [])
        consumers: Dict[str, List[str]] = {}
        for uid in order:
            op = ops.get(uid, {})
            for tid in op.get("input_tensor_ids", []):
                consumers.setdefault(tid, []).append(uid)
        overflow_conv_uids = set()
        if overflow_ops:
            overflow_conv_uids = {o[0] for o in overflow_ops}
        symbol_map = profiler.build_symbol_map(input_config)
        fusion_uids = set()
        for uid in order:
            op = ops.get(uid, {})
            if "upsample" not in op.get("op_type", ""):
                continue
            out_tids = op.get("output_tensor_ids", [])
            if len(out_tids) != 1:
                continue
            cons = consumers.get(out_tids[0], [])
            if len(cons) != 1:
                continue  # not single-consumer upsample — can't fuse
            conv_uid = cons[0]
            conv_op = ops.get(conv_uid, {})
            if "convolution" not in conv_op.get("op_type", ""):
                continue
            # Eligibility: conv would overflow OR upsample output > 25% VRAM.
            # Same formula as solver.py:819 _detect_op_level_tiling_pairs.
            conv_overflow = conv_uid in overflow_conv_uids
            up_out_meta = graph["tensors"].get(out_tids[0], {})
            up_out_shape = profiler._resolve_shape(up_out_meta, symbol_map)
            up_out_bytes = profiler._compute_size(
                up_out_shape, up_out_meta, dtype_bytes
            )
            up_overflow = up_out_bytes > 0.25 * comp_vram_bytes
            if conv_overflow or up_overflow:
                fusion_uids.add(uid)
        return fusion_uids

    def _compute_memory(
        self, container: "NBXContainer", components: List["ComponentData"],
        input_config: InputConfig, target_dtype_str: str,
        profile: Optional["PrismProfile"] = None,
        component_dtypes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ComponentMemory]:
        """Compute Total = Weights + Activations + Overhead with full intelligence.

        When `profile` is passed, the activation estimation becomes
        **tiling-aware** via a two-pass flow per component:

        1. First pass with `vram_per_gpu_bytes = smallest_GPU_in_profile`
           (worst-case budget): identify overflow_ops at the standard
           `safety=0.85` threshold.
        2. Detect which upsample → conv adjacencies would be intercepted
           as fusion pairs by `_detect_op_level_tiling_pairs` at runtime.
        3. Second pass with `fusion_upsample_uids` set: re-estimate peak
           with upsample outputs in fusion pairs zeroed (matches runtime
           FusionUpsampleProxy sentinel behavior).

        The result is a realistic peak_bytes that lets the strategy
        cascade accept configs where the runtime tiling makes the model
        fit. P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE 2026-05-10.
        Without `profile` (legacy callers): worst-case full-materialization
        behaviour preserved.

        Caveat: when the profile has mixed-size GPUs, picking the smallest
        as the tiling-aware budget is conservative — a component placed
        on the larger GPU may have its peak under-counted (no harm) but
        a component flagged for tiling may have its estimate biased toward
        more fusion than strictly needed. For POINT 9 scope (v100-16g and
        v100-16g-x2-01 — both uniform 16 GiB), this caveat does not apply.
        """
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
            # Per-component runtime dtype (the dtype this component ACTUALLY
            # runs at), not the model-wide worst-case. Falls back to the
            # model-level dtype for legacy callers / the fp32 fallback path.
            comp_dtype_str = (component_dtypes or {}).get(
                comp.name, target_dtype_str)
            dtype_mult = compute_dtype_factor(source_dtype, comp_dtype_str)

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
                    dtype_bytes = get_dtype_bytes_per_element(comp_dtype_str)
                    # Tiling-aware budget: smallest GPU in the profile.
                    # When profile is None (legacy callers), this stays 0
                    # and the two-pass logic short-circuits to worst-case.
                    smallest_gpu_bytes = 0
                    if profile is not None and getattr(profile, "devices", None):
                        smallest_gpu_bytes = min(
                            int(d.memory_mb * 1024 * 1024) for d in profile.devices
                        )
                    # First pass: worst-case + overflow_ops detection.
                    # `force_compute_dtype_for_fp` aligns the activation
                    # estimate with the runtime compute dtype — graph
                    # tensors traced as fp32 (PyTorch autocast off /
                    # fp32 capture path) actually flow through fp16
                    # kernels at runtime, halving their byte footprint.
                    # Without this override the estimator doubles the
                    # activation bill for any model where trace dtype
                    # != runtime compute dtype (Sana 4Kpx VAE: graph
                    # tensors fp32, runtime fp16 → 2× over-estimation).
                    ap = profiler.estimate_peak_memory(
                        input_config=input_config,
                        dtype_bytes=dtype_bytes,
                        vram_per_gpu_bytes=smallest_gpu_bytes or None,
                        force_compute_dtype_for_fp=True,
                    )
                    # Second pass (tiling-aware) only when first pass found
                    # overflow_ops AND we have a real budget to reason about.
                    if smallest_gpu_bytes > 0 and ap.overflow_ops:
                        fusion_uids = self._identify_fusion_upsample_uids(
                            comp.graph, ap.overflow_ops, input_config,
                            dtype_bytes, smallest_gpu_bytes,
                        )
                        # Also zero out F2a pixel_shuffle broadcast-aware
                        # chain ops (expand/clone/view return stride-0
                        # views or sentinel proxies at runtime — see
                        # tiling_engine.py:_detect_pixel_shuffle_broadcast_chains
                        # and the BroadcastClonePyroxy interceptor wiring).
                        f2a_uids = self._identify_pixel_shuffle_chain_proxy_uids(
                            comp.graph
                        )
                        # S5 gate: include chain intermediates in
                        # zero_alloc so Prism's cascade accepts
                        # single_gpu on 16g and routes VAE to GPU
                        # under the residual-chain wrapper.
                        chain_uids = self._identify_residual_chain_proxy_uids(
                            comp.graph
                        )
                        zero_uids = fusion_uids | f2a_uids | chain_uids
                        # In-place residual adds: detected statically by the
                        # same liveness logic the runtime engine uses. The
                        # add's output buffer = one input's buffer at runtime
                        # (no new allocation). The profiler aliases the output
                        # tid to the reused input tid, which extends that
                        # input's lifetime through downstream consumers and
                        # zero-allocates the output.
                        inplace_adds = self._identify_inplace_add_candidates_static(
                            comp.graph
                        )
                        if zero_uids or inplace_adds:
                            ap_tiled = profiler.estimate_peak_memory(
                                input_config=input_config,
                                dtype_bytes=dtype_bytes,
                                zero_alloc_uids=zero_uids,
                                inplace_adds=inplace_adds,
                                force_compute_dtype_for_fp=True,
                            )
                            activation_bytes = ap_tiled.peak_bytes
                            peak_op_uid = ap_tiled.peak_op_uid
                            peak_step = ap_tiled.peak_step
                        else:
                            activation_bytes = ap.peak_bytes
                            peak_op_uid = ap.peak_op_uid
                            peak_step = ap.peak_step
                    else:
                        # No overflow ops at this budget — also try in-place
                        # add detection alone (relevant for shapes where
                        # in-place adds dominate even without upsample fusion).
                        inplace_adds = self._identify_inplace_add_candidates_static(
                            comp.graph
                        )
                        if inplace_adds:
                            ap_tiled = profiler.estimate_peak_memory(
                                input_config=input_config,
                                dtype_bytes=dtype_bytes,
                                inplace_adds=inplace_adds,
                                force_compute_dtype_for_fp=True,
                            )
                            activation_bytes = ap_tiled.peak_bytes
                            peak_op_uid = ap_tiled.peak_op_uid
                            peak_step = ap_tiled.peak_step
                        else:
                            activation_bytes = ap.peak_bytes
                            peak_op_uid = ap.peak_op_uid
                            peak_step = ap.peak_step
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
        self, container, target_dtype: str, remaining_vram_bytes: int
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

        dtype_bytes = 2 if target_dtype in ("float16", "bfloat16") else 4
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

        # Initial buffer: start small, grow on demand up to max_cache_len
        # Run mode: allocate at max right away (single-shot, known size)
        # Serve mode: start at max_tokens + margin, grow as user generates longer
        if getattr(self, '_serve_mode', False) and max_tokens:
            initial_len = min(max_tokens + prompt_margin, max_cache_len)
        else:
            initial_len = 0  # 0 = allocate at max_cache_len (legacy behavior for run)

        return KVCachePlan(
            max_cache_len=max_cache_len,
            num_layers=int(num_layers),
            num_kv_heads=int(num_kv_heads),
            k_head_dim=int(k_head_dim),
            v_head_dim=int(v_head_dim_val),
            dtype=target_dtype,
            memory_bytes=memory_bytes,
            per_token_bytes=per_token_bytes,
            initial_cache_len=initial_len,
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
        """All components on largest GPU.

        Budget depends on serve_mode:
        - Hot (serve): sum(all_weights) + max(activations) — all weights resident
        - Cold (run): max(component_weights + component_activations) — one at a time
        """
        if not devices:
            return None

        largest = devices[0]
        serve_mode = getattr(self, '_serve_mode', False)

        if serve_mode:
            # Hot mode: all weights resident + peak activation of any single component
            total_weights_mb = sum(m.weight_mb for _, m in sorted_comps)
            max_activation_mb = max((m.activation_mb for _, m in sorted_comps), default=0)
            total_required = total_weights_mb + max_activation_mb
        else:
            # Cold mode: only one component in VRAM at a time
            peaks = [(n, m.weight_mb + m.activation_mb) for n, m in sorted_comps]
            _, max_peak = max(peaks, key=lambda x: x[1])
            total_required = max_peak

        needs_kv = getattr(self, '_needs_kv_cache', False)
        overhead_pct = 0.0 if needs_kv else 0.05
        total_required += total_required * overhead_pct

        # P-PRISM-NEVER-REFUSE v2 B.4: blanket driver/library overhead
        # reserve (matches `_place_component` Strategy 1 — see comment
        # there for the empirical justification). Prevents `single_gpu`
        # from accepting plans that fit the activation estimator but
        # then runtime-OOM at the conv::62 boundary of Sana 4Kpx on
        # 1× V100 16 GiB. The cascade can then fall through to
        # `lazy_sequential` (which routes VAE to CPU via Strategy 4
        # of `_place_component`) or `cpu_execution`.
        _OOM_RESERVE_MB = 3072
        effective_capacity = largest.capacity_mb - _OOM_RESERVE_MB
        if total_required > effective_capacity:
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

        Lifecycle classification per family:

        LLM / image_vq (autoregressive):
        - Persistent: language_model + gen_head + gen_embed + gen_aligner (every token)
        - Transient: vision_model, gen_vision_model, aligner (used once)

        Diffusion (iterative_process):
        - Persistent: transformer/dit (called N times in denoising loop)
        - Transient: text_encoder(s), VAE (used once per request)

        Audio (encoder-decoder / transducer):
        - Persistent: decoder, predictor, joiner (autoregressive / streaming)
        - Transient: encoder (used once per utterance)

        Source: topology.json flow config.
        """
        topology_path = container.cache_path / "topology.json" if container.cache_path else None
        flow_config = {}
        if topology_path and topology_path.exists():
            with open(topology_path) as f:
                topo = json.load(f)
                flow_config = topo.get("flow", {})

        manifest = container.get_manifest() or {}
        all_comps = set(manifest.get("components", {}).keys())
        category = getattr(self, '_model_category', 'diffusion')
        persistent = set()

        if category in ("llm", "image_vq"):
            # Autoregressive: generation loop components are persistent
            gen_config = flow_config.get("generation", {})
            for key in ("lm_component", "head_component", "embed_component", "aligner_component"):
                comp = gen_config.get(key)
                if comp:
                    persistent.add(comp)
            # Fallback by name convention
            if not persistent:
                for name in all_comps:
                    if name in ("language_model", "model", "llm") or \
                       (name.startswith("gen_") and name != "gen_vision_model"):
                        persistent.add(name)

        elif category == "diffusion":
            # Diffusion: loop components (transformer/dit) are persistent
            loop_comps = flow_config.get("loop", {}).get("components", [])
            persistent.update(loop_comps)
            # Fallback by name convention
            if not persistent:
                for name in all_comps:
                    if any(k in name.lower() for k in ("transformer", "dit", "unet", "diffusion")):
                        persistent.add(name)

        else:
            # Audio: decoder/predictor/joiner are persistent (autoregressive output)
            for name in all_comps:
                if any(k in name.lower() for k in ("decoder", "predictor", "joiner", "language_model")):
                    persistent.add(name)

        # If no classification succeeded, treat all as persistent (safe default)
        if not persistent:
            persistent = all_comps

        return persistent, all_comps - persistent

    def _try_single_gpu_lifecycle(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Lifecycle-aware single GPU: persistent components loaded together, transient on-demand.

        Peak = sum(all persistent weights + max persistent activation) + max(one transient at a time).
        Works for ALL families — _classify_lifecycle handles per-family classification.
        """
        if not devices:
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

    def _try_component_placement(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Distribute whole components across GPUs."""
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

    def _try_block_scatter(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Block scatter: spread transformer blocks across GPUs using best-fit-decreasing."""
        if len(devices) < 2:
            return None

        capacity_threshold = devices[0].capacity_mb * 0.80
        needs_fgp = [(n, m) for n, m in sorted_comps if m.weight_mb + m.activation_mb > capacity_threshold]

        if not needs_fgp:
            return None

        fgp_target = PRISM_DEFAULTS.get("fgp_utilization_target", 0.92)
        # Use full device capacity (already has 0.95 safety margin from _prepare_devices).
        # fgp_target is applied as per-block inflation, not capacity reduction.
        packing_overhead = 1.0 / fgp_target  # ~1.087x per-block inflation
        fresh = [
            DeviceState(
                device_string=d.device_string,
                capacity_mb=d.capacity_mb,
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
                    real_size = dev.get_real_block_size(blocks['non_block_mb'], model_dtype) * packing_overhead
                    if dev.can_fit(real_size):
                        dev.used_mb += real_size
                        dev.components.append(f"{comp_name}.non_block")
                        for key in blocks['non_block_keys']:
                            shard_map[key] = dev.device_string
                        break
                    current_dev_idx += 1
                if current_dev_idx >= len(fresh):
                    return None

            # Allocate blocks - best-fit across GPUs with topology preference
            last_dev = None
            for block_num in sorted(blocks['blocks'].keys()):
                block_keys = blocks['blocks'][block_num]
                base_block = blocks['block_sizes'][block_num]
                real_block = fresh[0].get_real_block_size(base_block, model_dtype)
                total_block = (real_block + act_per_block) * packing_overhead

                # Find GPUs that can fit this block
                fit_devs = [d for d in fresh if d.can_fit(total_block)]
                if not fit_devs:
                    return None

                # Topology-aware: prefer same group as last block (minimize cross-device transfers)
                target = None
                if last_dev is not None and profile.topology:
                    last_idx = self._get_device_index(last_dev.device_string)
                    same_group = []
                    other = []
                    for d in fit_devs:
                        d_idx = self._get_device_index(d.device_string)
                        if d_idx == last_idx or profile.topology.devices_have_fast_interconnect([last_idx, d_idx]):
                            same_group.append(d)
                        else:
                            other.append(d)
                    # Prefer: same device (free space) > same NVLink group > any device
                    if same_group:
                        target = max(same_group, key=lambda d: d.free_mb)
                    elif other:
                        target = max(other, key=lambda d: d.free_mb)

                if target is None:
                    target = max(fit_devs, key=lambda d: d.free_mb)

                target.used_mb += total_block
                target.components.append(f"{comp_name}.block.{block_num}")
                for key in block_keys:
                    shard_map[key] = target.device_string
                last_dev = target

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

    def _try_pipeline_parallel(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Pipeline parallel: per-layer sequential fill across GPUs.

        Like Accelerate device_map="auto": layers 0..N on GPU0, N+1..M on GPU1, etc.
        Only N-1 boundary crossings for N GPUs. Greedy sequential fill — each layer
        goes to the current GPU until it's full, then moves to the next.

        This is the optimal strategy for large LLMs across multiple GPUs with
        fast interconnect, producing minimal cross-device transfers.
        """
        if len(devices) < 2:
            return None

        # Find components that need splitting (too large for single GPU)
        capacity_threshold = devices[0].capacity_mb * 0.80
        needs_split = [(n, m) for n, m in sorted_comps if m.weight_mb + m.activation_mb > capacity_threshold]

        if not needs_split:
            return None

        fgp_target = PRISM_DEFAULTS.get("fgp_utilization_target", 0.92)
        # Use full device capacity (already has 0.95 safety margin from _prepare_devices).
        # fgp_target is applied as per-block inflation, not capacity reduction.
        packing_overhead = 1.0 / fgp_target  # ~1.087x per-block inflation
        fresh = [
            DeviceState(
                device_string=d.device_string,
                capacity_mb=d.capacity_mb,
                used_mb=0.0, components=[], spec=d.spec
            )
            for d in devices
        ]

        allocations = {}
        current_dev_idx = 0

        for comp_name, mem in needs_split:
            blocks = self._parse_blocks(container, comp_name)
            if not blocks['blocks']:
                return None

            model_dtype = None
            for comp in container.get_neural_components():
                if comp.name == comp_name:
                    model_dtype = comp.get_dominant_dtype()
                    break
            model_dtype = model_dtype or "bfloat16"

            n_blocks = len(blocks['blocks'])
            act_per_block = mem.activation_mb / n_blocks

            # _parse_blocks reports ON-DISK sizes (safetensors header dtype, e.g.
            # F32 for Wan = 65.6GB), but the runtime loads each weight at the
            # COMPUTE dtype (fp16 on V100 = 32.8GB). Scale the block/non-block
            # estimates to the compute size so the fit matches what is actually
            # loaded — weight_sharding already uses the compute-dtype mem.weight_mb
            # (31.27GB); pipeline_parallel must be consistent, otherwise a model
            # whose F32 shards exceed 2x a GPU is wrongly rejected ("cannot fit")
            # though its fp16 weights fit with headroom (Wan-I2V-14B: 32.8GB F32/2
            # = 16.4GB/GPU fits, but the F32 estimate 65.6/2=32.8 > 30.4 rejected).
            # Data-driven, no-op when disk dtype == compute dtype (scale ~1.0).
            _disk_mb = sum(blocks['block_sizes'].values()) + blocks['non_block_mb']
            _dtype_scale = (mem.weight_mb / _disk_mb) if _disk_mb > 0 else 1.0

            shard_map = {}

            # Allocate non-block weights on current device
            if blocks['non_block_mb'] > 0:
                while current_dev_idx < len(fresh):
                    dev = fresh[current_dev_idx]
                    real_size = dev.get_real_block_size(blocks['non_block_mb'] * _dtype_scale, model_dtype) * packing_overhead
                    if dev.can_fit(real_size):
                        dev.used_mb += real_size
                        dev.components.append(f"{comp_name}.non_block")
                        for key in blocks['non_block_keys']:
                            shard_map[key] = dev.device_string
                        break
                    current_dev_idx += 1
                if current_dev_idx >= len(fresh):
                    return None

            # BALANCED PIPELINE FILL: contiguous layer ranges, spread EVENLY so
            # each GPU keeps headroom for runtime activations + triton autotune
            # workspace (neither fully modeled in the trace-time activation peak).
            # A greedy "pack GPU0 to capacity" fill leaves the first GPU with no
            # room and OOMs/stalls under CFG batch=2 (Wan-I2V-14B: 27 GB weights on
            # cuda:2, ~2 GB free). Pick the minimum #GPUs whose EVEN weight share
            # leaves a headroom reserve, then fill each to its even share (the last
            # pipeline GPU takes the remainder). Still a pipeline: one cross-device
            # transfer per GPU boundary, not per op.
            total_block_mb = sum(blocks['block_sizes'].values()) * _dtype_scale
            # CFG doubles the activation batch; the capacity floor also reserves for
            # the triton autotune workspace, which the activation profile does not
            # model. Data-driven (peak activation) with a hardware-universal floor.
            act_reserve_mb = max(mem.activation_mb * 2.0, fresh[0].capacity_mb * 0.18)
            n_pp = len(fresh)
            for ng in range(1, len(fresh) + 1):
                if (total_block_mb / ng) * packing_overhead + act_reserve_mb <= fresh[ng - 1].capacity_mb:
                    n_pp = ng
                    break
            target_per_gpu = total_block_mb / n_pp if n_pp > 0 else total_block_mb

            gpu_weight_mb = 0.0
            for block_num in sorted(blocks['blocks'].keys()):
                block_keys = blocks['blocks'][block_num]
                base_block = blocks['block_sizes'][block_num] * _dtype_scale
                real_block = fresh[0].get_real_block_size(base_block, model_dtype)
                total_block = (real_block + act_per_block) * packing_overhead

                while current_dev_idx < len(fresh):
                    dev = fresh[current_dev_idx]
                    # Advance to the next GPU once this one has its even share
                    # (never on the last pipeline GPU, which holds the remainder).
                    over_share = (gpu_weight_mb + real_block > target_per_gpu
                                  and current_dev_idx < n_pp - 1)
                    if dev.can_fit(total_block) and not over_share:
                        dev.used_mb += total_block
                        gpu_weight_mb += real_block
                        dev.components.append(f"{comp_name}.block.{block_num}")
                        for key in block_keys:
                            shard_map[key] = dev.device_string
                        break
                    # GPU full or even share reached — move to next
                    current_dev_idx += 1
                    gpu_weight_mb = 0.0
                else:
                    # No GPU can fit this block
                    return None

            devices_used = set(shard_map.values())
            allocations[comp_name] = (f"fgp:{','.join(sorted(devices_used))}", shard_map)

        # Allocate regular components on remaining capacity
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

    def _try_weight_sharding(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """Weight sharding: distribute weight files across GPUs (round-robin).

        Weight files distributed via shard_map across GPUs.
        CompiledSequence multi-device path handles cross-device execution
        with automatic device alignment at op boundaries.
        """
        if len(devices) < 2:
            return None

        largest_cap = devices[0].capacity_mb
        needs_tp = [(n, m) for n, m in sorted_comps if m.weight_mb + m.activation_mb > largest_cap]

        import os as _os
        if _os.environ.get("NBX_DIAG_TRITON_PRELOOP") == "1":
            print(f"   [NBX-DIAG-PRISM] _try_weight_sharding largest_cap={largest_cap:.0f}MB "
                  f"ndev={len(devices)} devs={[d.device_string for d in devices]}", flush=True)
            for _n, _m in sorted_comps:
                print(f"   [NBX-DIAG-PRISM]   {_n}: weight={_m.weight_mb:.0f}MB "
                      f"act={_m.activation_mb:.0f}MB sum={_m.weight_mb+_m.activation_mb:.0f}MB "
                      f"needs_tp={_m.weight_mb+_m.activation_mb > largest_cap}", flush=True)

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

            # Prefer NVLink-connected GPUs for TP (minimize cross-device transfer cost)
            if n_gpus <= len(available) and profile.topology:
                # Try to find n_gpus devices that are all fast-interconnected
                best_group = None
                for start in range(len(available) - n_gpus + 1):
                    group = available[start:start + n_gpus]
                    indices = [self._get_device_index(g.device_string) for g in group]
                    if profile.topology.devices_have_fast_interconnect(indices):
                        best_group = group
                        break
                tp_gpus = best_group if best_group else available[:n_gpus]
            else:
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

        # Reset component-tiling flags for this placement pass so a rejected
        # attempt never leaves stale entries (the chosen pass repopulates).
        self._component_tiling = {}

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

    def _spatial_component_tiling(
        self, container: "NBXContainer", comp_name: str,
        mem: ComponentMemory, budget_bytes: int,
    ) -> Optional[Dict[str, Any]]:
        """Return a TilingEngine spec when a spatial component (4D/5D input +
        scale config) overflows `budget_bytes` untiled but fits when its
        decode runs on overlapping spatial tiles — else None.

        REUSES the R31 TilingEngine brick: here Prism only SIZES the tile so
        the per-tile activation fits, then keeps the component on GPU instead
        of routing it to host RAM (the CogVideoX-5b VAE at 480x720x49 frames
        peaks at ~82 GB untiled — 5 live full-res feature maps — and was being
        offloaded to cpu). The runtime instantiates a TilingEngine from this
        spec. Activation scales ~quadratically with the tiled spatial area, so
        tile_size = sqrt(latent_area * budget / full_activation).
        """
        import json, math
        cache_path = getattr(container, "_cache_path", None)
        ic = getattr(self, "_input_config", None)
        if cache_path is None or ic is None:
            return None
        gpath = cache_path / "components" / comp_name / "graph.json"
        ppath = cache_path / "components" / comp_name / "profile.json"
        if not gpath.exists() or not ppath.exists():
            return None
        try:
            graph = json.load(open(gpath))
            profile_j = json.load(open(ppath))
        except Exception:
            return None

        # Spatial input: 4D NCHW or 5D NCDHW; trace_size = H (index -2).
        tensors = graph.get("tensors", {})
        trace_size = None
        for iid in graph.get("input_tensor_ids", []):
            shape = tensors.get(str(iid), {}).get("shape", [])
            if isinstance(shape, list) and len(shape) in (4, 5):
                trace_size = shape[-2]
                break
        if not trace_size:
            return None

        # Scale factor: upscale (upscalers) or VAE compression ratio.
        config = profile_j.get("config", {})
        scale_factor = config.get("upscale")
        if scale_factor is None:
            # Spatial compression ratio = 2^(num_blocks-1). VAE configs spell
            # the block list as decoder_block_out_channels (Sana/DC-AE) OR the
            # generic block_out_channels (CogVideoX / most diffusers VAEs).
            db = (config.get("decoder_block_out_channels")
                  or config.get("block_out_channels"))
            if db:
                scale_factor = 2 ** (len(db) - 1)
        if not scale_factor:
            return None

        full_act = mem.activation_bytes
        if full_act <= budget_bytes or full_act <= 0:
            return None  # fits untiled — no tiling needed

        # Runtime latent spatial extent (the input space the tiles cover).
        vae_scale = getattr(ic, "vae_scale", None) or 8
        latent_h = max(1, (getattr(ic, "height", None) or 1024) // vae_scale)
        latent_w = max(1, (getattr(ic, "width", None) or 1024) // vae_scale)

        window_alignment = config.get("window_size", 1) or 1
        frac = budget_bytes / full_act
        tile_size = int(math.sqrt(latent_h * latent_w * frac))
        if window_alignment > 1:
            tile_size = (tile_size // window_alignment) * window_alignment
        tile_size = max(window_alignment if window_alignment > 1 else 8,
                        min(tile_size, latent_h, latent_w))

        overlap = max(4, tile_size // 8)
        if window_alignment > 1:
            overlap = ((overlap + window_alignment - 1) // window_alignment) * window_alignment

        tiled_act = full_act * (tile_size * tile_size) / float(latent_h * latent_w)
        if tiled_act > budget_bytes:
            return None  # even the clamped tile doesn't fit

        return {
            "tile_size": int(tile_size),
            "scale_factor": int(scale_factor),
            "overlap": int(overlap),
            "window_alignment": int(window_alignment),
            "trace_size": int(trace_size),
            "tiled_activation_bytes": int(tiled_act),
        }

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

        # Strategy 1: single_gpu — component fits on largest GPU.
        # Capacity discounted by a fixed driver/library overhead reserve
        # (cuDNN/cuBLAS workspaces, Triton kernel cache, autotune state,
        # PyTorch caching allocator fragmentation). Observed empirically:
        # 16.6 GiB peak runtime usage on 32 GiB GPU for ~13 GiB of live
        # NBX tensors → ~3 GiB driver-side overhead. Applying the reserve
        # at planning time prevents per-component placements that fit
        # the estimator but OOM at runtime, which then triggers Strategy
        # 4 (CPU placement) below. P-PRISM-NEVER-REFUSE v2 B.4 —
        # 2026-05-12.
        _OOM_RESERVE_MB = 3072  # 3 GiB blanket driver/library overhead
        effective_capacity = largest.capacity_mb - _OOM_RESERVE_MB
        if required <= effective_capacity * 0.92:
            shard_map = {s: largest.device_string for s in shard_sizes.get(comp_name, {})}
            return (largest.device_string, shard_map)

        # Strategy 2: fgp — distribute blocks across GPUs
        fgp_result = self._place_component_fgp(
            container, comp_name, mem, devices, model_dtype
        )
        if fgp_result is not None:
            return fgp_result

        # Strategy 3: zero3 — CPU offload of weights, GPU for compute only.
        # Works only when activations fit on the largest GPU (with the
        # same driver-overhead reserve as Strategy 1).
        if mem.activation_mb <= effective_capacity * 0.92:
            shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
            return (f"zero3:{largest.device_string}", shard_map)

        # Strategy 3.5: spatial component tiling — keep a spatial-overflow
        # component (4D/5D input + scale config) on GPU by tiling its decode
        # (reuses the R31 TilingEngine) instead of offloading compute to host
        # RAM. Fires ONLY when the untiled activation fits no GPU even with
        # zero3, but the per-tile activation does — strictly additive
        # (non-spatial components, and ones that already fit, never reach
        # here). The CogVideoX-5b VAE (82 GB untiled at 480x720x49) stays on
        # GPU as overlapping spatial tiles instead of cpu.
        # Budget the tile conservatively (0.40 of the GPU): the quadratic
        # area-scaling estimate undercounts the true per-tile live set (the
        # resnet keeps several full-res feature maps live), and the triton
        # arena's live watermark runs ~1.3x the compiled caching allocator
        # for the same tile. CogVideoX-5b at 0.85 sized tile=39 → 24 GB
        # compiled (fit) but 31 GB+ triton (OOM at conv::90); 0.40 sizes a
        # tile that fits both. tiled_activation re-checked below.
        tile_budget = int(effective_capacity * 0.40 * 1024 * 1024)
        tiling = self._spatial_component_tiling(
            container, comp_name, mem, tile_budget)
        if tiling is not None:
            tiled_total_mb = real_weight + tiling["tiled_activation_bytes"] / (1024 * 1024)
            if tiled_total_mb <= effective_capacity * 0.92:
                self._component_tiling[comp_name] = tiling
                shard_map = {s: largest.device_string
                             for s in shard_sizes.get(comp_name, {})}
                return (largest.device_string, shard_map)

        # Strategy 4: cpu — both weights AND compute on CPU.
        # Last-resort placement for a single component whose activations
        # don't fit any GPU even with zero3 weight offload. Used by
        # `lazy_sequential` to route the largest component (e.g. Sana 4Kpx
        # VAE on V100 16 GiB) to host RAM while smaller components stay
        # on GPU. Validates that the component fits in 70% of host RAM
        # (matches `_try_cpu_execution`'s budget formula).
        #
        # When `profile.cpu` is missing (some older or hand-written
        # profile YAMLs without a `cpu:` section), accept unconditionally
        # per Doctrine R35 — Prism never refuses on hardware that exposes
        # CPU at the OS level. The runtime will fail clean if RAM is
        # truly insufficient, with a clearer signal than an OOM-at-conv.
        # P-PRISM-NEVER-REFUSE v2 B.4 — Doctrine R35 cascade per-component.
        if profile.cpu and profile.cpu.ram_mb > 0:
            if mem.total_mb <= profile.cpu.ram_mb * 0.7:
                shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
                return ("cpu", shard_map)
            # RAM accounted and insufficient → genuinely can't place
            return None

        # No CPU stats → accept (R35 default; runtime validates real RAM)
        shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
        return ("cpu", shard_map)

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
        - Transfer penalty weighted by actual interconnect bandwidth from profile
        - Boundary count penalty (fewer boundaries = better)
        - Topology completeness (partial interconnection = lower score)
        - Serve mode: eager strategies get boost (user wants near-zero latency)
        - Zero3: penalty scales with PCIe bandwidth + CPU cores from profile
        - Lazy loading overhead penalty
        """
        BASE_SCORES = {
            "single_gpu": 1000,
            "single_gpu_lifecycle": 900,
            "pipeline_parallel": 850,
            "component_placement": 750,
            "block_scatter": 700,
            "weight_sharding": 680,
            "component_placement_lazy": 400,
            "lazy_sequential": 300,
            "zero3": 100,
            # R35 last-resort: CPU execution. Very low score so any
            # successful GPU strategy beats it; only chosen when nothing
            # else fits OR when the profile has no GPUs.
            "cpu_execution": 10,
        }
        score = float(BASE_SCORES.get(strategy_name, 500))

        # Count unique devices used
        device_strings = set()
        for alloc in allocations.values():
            dev_str = alloc[0] if isinstance(alloc, tuple) else alloc
            for prefix in ("zero3:", "fgp:", "tp:"):
                if dev_str.startswith(prefix):
                    dev_str = dev_str[len(prefix):]
                    break
            for d in dev_str.split(","):
                d = d.strip()
                if d.startswith("cuda:") or d.startswith("hip:") or d.startswith("xpu:"):
                    device_strings.add(d)

        n_devices = len(device_strings)

        # Bandwidth reference from profile (not hardcoded)
        ref_bw = profile.topology.default_bandwidth_gbps if profile.topology else 32.0

        # Bandwidth-weighted transfer penalty for multi-GPU strategies
        if n_devices > 1 and profile.topology:
            device_indices = sorted(int(d.split(':')[-1]) for d in device_strings)

            # Get minimum pairwise bandwidth (bottleneck)
            min_bw = profile.get_min_pairwise_bandwidth(device_indices)

            # Topology completeness
            all_connected = profile.topology.devices_have_fast_interconnect(device_indices)

            # Boundary count estimation by strategy type
            if strategy_name == "pipeline_parallel":
                n_boundaries = n_devices - 1  # Sequential: N-1 crossings
            elif strategy_name == "component_placement":
                n_boundaries = len(allocations)
            elif strategy_name == "block_scatter":
                # Count actual cross-device boundaries from allocation
                dev_sequence = []
                for alloc in allocations.values():
                    dev = alloc[0] if isinstance(alloc, tuple) else alloc
                    if dev.startswith("fgp:"):
                        dev_sequence.extend(dev[4:].split(","))
                n_boundaries = max(len(set(dev_sequence)) - 1, n_devices)
            elif strategy_name == "weight_sharding":
                n_boundaries = n_devices * 20  # Many copies per forward pass
            else:
                n_boundaries = n_devices

            # Transfer penalty: inversely proportional to actual bandwidth
            transfer_penalty = n_boundaries * 15.0 * (ref_bw / max(min_bw, 1.0))
            score -= transfer_penalty

            # Partial topology penalty
            if not all_connected:
                score -= 200.0

        # Serve mode: eager strategies are much more valuable (near-zero latency)
        serve_mode = getattr(self, '_serve_mode', False)
        if serve_mode:
            eager_values = {s.value for s in AllocationStrategy if s.is_eager}
            if strategy_name in eager_values:
                score += 100.0  # Boost eager strategies in serve mode
            else:
                score -= 50.0   # Penalize lazy strategies in serve mode

        # Zero3: penalty scales with actual PCIe bandwidth and CPU capability
        if strategy_name == "zero3":
            # Base penalty adjusted by PCIe bandwidth
            # Fast PCIe (Gen5 64GB/s) = less penalty, slow (Gen3 16GB/s) = more penalty
            pcie_factor = ref_bw / 32.0  # Normalize to Gen4 baseline
            base_zero3_penalty = 200.0 / max(pcie_factor, 0.5)  # Better PCIe = less penalty

            # CPU cores adjustment: more cores = faster CPU-side compute
            cpu_cores = profile.cpu.cores if profile.cpu else 4
            core_factor = min(cpu_cores / 16.0, 2.0)  # Normalize to 16-core baseline, cap at 2x
            base_zero3_penalty /= max(core_factor, 0.5)  # More cores = less penalty

            score -= base_zero3_penalty

        # Lazy loading penalty: each component load/unload adds latency
        if "lazy" in strategy_name:
            n_components = len(allocations)
            score -= 20.0 * n_components

        return max(score, 1.0)

    # =========================================================================
    # LAZY / OFFLOAD STRATEGIES
    # =========================================================================

    def _try_component_placement_lazy(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """
        Component placement with lazy weight swap between phases.
        Distribute component GROUPS across N GPUs.
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
        Succeeds if at least 1 GPU can hold the largest activation footprint
        AND CPU has enough RAM to hold all weights.
        """
        if not devices:
            return None

        fresh = self._fresh_devices(devices)
        largest = max(fresh, key=lambda d: d.capacity_mb)
        allocations = {}

        # Validate CPU RAM budget: weights must fit in 70% of RAM
        # (reserve 30% for OS, activations, PyTorch overhead)
        if profile.cpu and profile.cpu.ram_mb > 0:
            total_weight_mb = sum(mem.weight_mb for _, mem in sorted_comps)
            available_ram_mb = profile.cpu.ram_mb * 0.7
            if total_weight_mb > available_ram_mb:
                return None

        for comp_name, mem in sorted_comps:
            # Zero3: only activations need GPU memory
            if mem.activation_mb > largest.capacity_mb * 0.92:
                return None

            shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = (f"zero3:{largest.device_string}", shard_map)

        return allocations, fresh

    def _try_cpu_execution(
        self, sorted_comps, comp_mem, devices, shard_sizes, profile, container
    ) -> Optional[Tuple[Dict, List[DeviceState]]]:
        """CPU-only execution — Doctrine R35 last-resort cascade.

        Place EVERY component (weights + activations) on CPU. Compute
        routed to PyTorch ATen native CPU dispatcher (branch A) — sequential
        and compiled modes work out of the box because `sequential_dispatcher`
        and CompiledSequence are device-aware. Triton modes fall back to
        PyTorch CPU when no Triton-CPU runtime is integrated (TODO
        documented in `triton_cpu_coverage_gaps.md` once the branch B
        Triton-CPU integration lands).

        Activation budget: `sum(component peaks) <= cpu.ram_mb * 0.7`
        (reserve 30% for OS, Python interpreter, MKL/oneDNN workspaces,
        and intermediate activations not modelled by the estimator).
        For profiles without a `cpu` config or with `ram_mb == 0`, accept
        unconditionally — the doctrine says Prism never refuses, and the
        runtime will fail clean if RAM is genuinely insufficient. This
        keeps the strategy useful on profiles where CPU stats weren't
        captured.

        Wall-time is intentionally unbounded — the strategy may take
        minutes or hours for large diffusion models at high resolution.
        Doctrine R35: perf libre, disponibilité first.

        R34 model-agnostic: discrimination only by hardware-profile
        (`cpu.ram_mb`) and graph-derived component memory. No model
        names, no families. P-PRISM-NEVER-REFUSE v2 — 2026-05-12.
        """
        # Validate CPU RAM if a cpu config is present
        if profile.cpu and profile.cpu.ram_mb > 0:
            total_required_mb = sum(mem.total_mb for _, mem in sorted_comps)
            available_ram_mb = profile.cpu.ram_mb * 0.7
            if total_required_mb > available_ram_mb:
                # Genuinely insufficient RAM. Return None so the cascade
                # can emit a clear error message at _fail_error level —
                # explicitly mentions hardware (more RAM) instead of
                # silently failing inside the runtime.
                return None

        allocations: Dict[str, Tuple[str, Dict[str, str]]] = {}
        for comp_name, _mem in sorted_comps:
            shard_map = {s: "cpu" for s in shard_sizes.get(comp_name, {})}
            allocations[comp_name] = ("cpu", shard_map)

        # CPU-only "fresh devices" list. Used downstream by the executor
        # factory for context; an empty list is the historical signal
        # for "no GPU used". We preserve that signal.
        fresh = self._fresh_devices(devices)
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

    def _components_force_fp32(self, container: "NBXContainer",
                               profile: Optional["PrismProfile"] = None) -> set:
        """Per-component fp32 pin set. Composes two sources by additive union:

        (1) MANUAL (Ch6-era): the `requires_fp32_compute` registry flag.
            For architectures whose activation range structurally exceeds
            the fp16 representable range (e.g. deep transformer-SR stacks
            with no inter-block normalisation — SwinIR's RSTB cascade hits
            NaN by conv_after_body in pure fp16). Read via
            `registry_flags.get_component_flag`, env-overridable by
            `NBX_FORCE_FP32_COMPUTE`. No .nbx field added (R18 preserved).

        (2) AUTO (Ch8 P-LAYER7-AUTO-FP32-FAMILY-AWARE): family-aware
            structural detection of fp16-conv-overflow risk. Reads
            `config/families/<family>.yml dtype_policy.auto_fp32_on_overflow_risk`
            (R10 data-driven). Discriminator (R34 strict, no model-name
            branching): family ∈ {image, video} AND component graph
            torch_dtype matches require_graph_dtype AND hardware is
            fp16-preferring AND the component is conv-cascade-dominant
            (`conv2d_count >= min_conv2d_count` and `conv2d_count >=
            conv2d_to_sdpa_ratio_min × sdpa_count`). Selects VAE-class
            encoders/decoders, excludes patch-embed transformers and
            hybrid linear-attention DiTs. Bypassable for diagnosis via
            `NBX_DISABLE_AUTO_FP32=1` (manual still honored).

        Manual ⊕ auto by set union: manual entries always end up in the
        result (manual > auto by construction; no conflict surface). Both
        sources default-absent ⇒ empty set ⇒ ZERO behaviour change for
        any model not matching the criteria (anti-régression guarantee).
        """
        from neurobrix.core.runtime.registry_flags import get_component_flag
        try:
            manifest = container.get_manifest() or {}
            model_name = manifest.get("model_name")
            family = manifest.get("family")
        except Exception:
            model_name = None
            family = None

        forced = set()

        # (1) Manual flag (Ch6-era).
        try:
            for comp in container.get_neural_components():
                if get_component_flag(model_name, comp.name,
                                      "requires_fp32_compute", default=False,
                                      env_override="NBX_FORCE_FP32_COMPUTE"):
                    forced.add(comp.name)
        except Exception:
            pass

        # (2) Auto-detect (Ch8). Bypassable for diagnosis.
        if os.environ.get("NBX_DISABLE_AUTO_FP32", "0") == "1":
            return forced
        auto = self._auto_fp32_components(container, profile, family)
        return forced | auto

    def _auto_fp32_components(self, container: "NBXContainer",
                              profile: "Optional[PrismProfile]",
                              family: "Optional[str]") -> set:
        """Family-aware structural auto-fp32 detection (Ch8).

        Policy and thresholds live in `config/families/<family>.yml`
        under `dtype_policy.auto_fp32_on_overflow_risk`. Default-absent
        ⇒ disabled ⇒ empty set. See _components_force_fp32 docstring
        for the full rule.
        """
        if family is None:
            return set()
        try:
            from neurobrix.core.config.loader import get_family_config
            fcfg = get_family_config(family) or {}
        except Exception:
            return set()

        policy = ((fcfg.get("dtype_policy") or {})
                  .get("auto_fp32_on_overflow_risk") or {})
        if not policy.get("enabled"):
            return set()

        # Hardware gate: skip on bf16-capable hardware (bf16 exponent
        # range = fp32, no conv-storage saturation).
        if policy.get("skip_when_hw_supports_bf16", True) and profile is not None:
            try:
                if profile.devices_support_dtype("bfloat16"):
                    return set()
            except Exception:
                pass

        require_dtype = policy.get("require_graph_dtype", "float32")
        min_conv = int(policy.get("min_conv2d_count", 20))
        ratio = float(policy.get("conv2d_to_sdpa_ratio_min", 10))

        # Op-type markers (NeuroBrix canonical aten::op format,
        # src/neurobrix/CLAUDE.md §6).
        _CONV_OPS = {"aten::conv2d", "aten::convolution",
                     "aten::_convolution"}
        _SDPA_OPS = {"aten::scaled_dot_product_attention",
                     "aten::_scaled_dot_product_efficient_attention",
                     "aten::_scaled_dot_product_flash_attention",
                     "aten::_scaled_dot_product_attention_math"}

        auto = set()
        try:
            comps = container.get_neural_components()
        except Exception:
            return set()
        for comp in comps:
            graph = getattr(comp, "graph", None) or {}
            if graph.get("torch_dtype") != require_dtype:
                continue
            ops = graph.get("ops") or {}
            if not ops:
                continue
            conv = 0
            sdpa = 0
            for op_data in ops.values():
                ot = op_data.get("op_type", "")
                if ot in _CONV_OPS:
                    conv += 1
                elif ot in _SDPA_OPS:
                    sdpa += 1
            if conv >= min_conv and conv >= ratio * sdpa:
                auto.add(comp.name)
        return auto

    def _resolve_dtype(self, container: "NBXContainer", profile: PrismProfile) -> str:
        """Resolve target dtype. bf16 → fp16 if weights fit in range, else fp32."""
        # Model-side opt-in (manual `requires_fp32_compute`) + Ch8
        # auto-detect (family-aware fp16-conv-overflow). Checked FIRST
        # so the fp16 preference below cannot win for fp16-unsafe
        # architectures or for VAE-class components on V100.
        if self._components_force_fp32(container, profile):
            return "float32"

        # Find model's dominant dtype
        dtypes = {}
        for comp in container.get_neural_components():
            d = comp.get_dominant_dtype()
            dtypes[d] = dtypes.get(d, 0) + 1
        requested = max(dtypes.keys(), key=lambda d: dtypes[d]) if dtypes else "float16"

        # Priority: hardware preferred > model native > fallback
        if profile.preferred_dtype and profile.devices_support_dtype(profile.preferred_dtype):
            return profile.preferred_dtype

        if profile.devices_support_dtype(requested):
            return requested

        if requested == "bfloat16":
            if profile.devices_support_dtype("float16") and self._scan_bf16_fp16_safety(container):
                return "float16"
            return "float32"
        elif requested == "float16":
            return "float32"

        return "float32"

    def _resolve_component_dtypes(self, components, profile: PrismProfile, container: "Optional[NBXContainer]" = None) -> Dict[str, str]:
        """Resolve dtype per component. Returns dtype strings."""
        result = {}

        forced_fp32 = self._components_force_fp32(container, profile) if container is not None else set()

        for comp in components:
            if comp.name in forced_fp32:
                result[comp.name] = "float32"
                continue
            native = comp.get_dominant_dtype()
            if profile.preferred_dtype and profile.devices_support_dtype(profile.preferred_dtype):
                resolved = profile.preferred_dtype
            elif profile.devices_support_dtype(native):
                resolved = native
            elif native == "bfloat16":
                if container is not None and profile.devices_support_dtype("float16") and self._scan_bf16_fp16_safety(container):
                    resolved = "float16"
                else:
                    resolved = "float32"
            elif native == "float16":
                resolved = "float32"
            else:
                resolved = "float32"
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
        # GPU profile: use first device's arch/brand. CPU-only profile:
        # fall back to the CPU's architecture (x86_64, arm64, riscv) and
        # a generic "cpu" vendor. The factory's `_extract_architecture`
        # and `_extract_vendor` accept any non-empty string — the values
        # are used downstream for vendor-specific kernel selection
        # (which is a no-op on the CPU path that uses PyTorch ATen
        # native via MKL/oneDNN). R34 generic — no model-specific.
        if profile.devices:
            arch = profile.devices[0].architecture
            vendor = profile.devices[0].brand.value
        elif profile.cpu:
            arch = profile.cpu.architecture or "x86_64"
            vendor = "cpu"
        else:
            arch = "x86_64"
            vendor = "cpu"

        components = {}
        total_mb = 0.0

        for comp_name, (device_str, shard_map) in allocations.items():
            mem = comp_mem[comp_name]

            if device_str.startswith("fgp:"):
                device_list = device_str[4:].split(",")
                # fgp: prefix is used by both pipeline_parallel and block_scatter
                comp_strategy = strategy if strategy in ("pipeline_parallel", "block_scatter") else "block_scatter"
            elif device_str.startswith("tp:"):
                device_list = device_str[3:].split(",")
                comp_strategy = "weight_sharding"
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
                dtype=comp_dtypes.get(comp_name, "float32"),
                memory_mb=mem.total_mb,
                architecture=arch,
                vendor=vendor,
                sharded=len(device_list) > 1,
                shard_map=shard_map,
                strategy=comp_strategy,
            )
            total_mb += mem.total_mb

        # Determine loading mode — strategy + serve_mode driven
        # Eager: weights stay in VRAM permanently (fast serving, near-zero latency)
        # Lazy: weights swap in/out per execution phase (large models, cold run)
        #
        # Serve cold fallback: user asked for serve (hot) but VRAM can't hold all
        # weights → force lazy mode. The daemon still works, just not near-zero latency.
        _EAGER_VALUES = {s.value for s in AllocationStrategy if s.is_eager}
        _LAZY_VALUES = {s.value for s in AllocationStrategy if not s.is_eager}

        if getattr(self, '_serve_cold_fallback', False):
            # Serve mode degraded to cold: force lazy even if strategy is eager-capable
            loading_mode = "lazy"
        elif strategy in _EAGER_VALUES:
            loading_mode = "eager"
        elif strategy in _LAZY_VALUES:
            loading_mode = "lazy"
        else:
            total_gpu_mb = sum(d.capacity_mb for d in devices if d.device_string.startswith("cuda"))
            loading_mode = "eager" if total_mb <= total_gpu_mb * 0.90 else "lazy"

        # Determine primary dtype
        primary_dtype = "float16"
        for dt in comp_dtypes.values():
            if dt == "float32":
                primary_dtype = "float32"
                break

        return ExecutionPlan(
            components=components,
            target_dtype=primary_dtype,
            total_memory_mb=total_mb,
            strategy=strategy,
            component_memory=comp_mem,
            loading_mode=loading_mode,
            cpu_ram_mb=profile.cpu.ram_mb if profile.cpu else 0,
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _empty_plan(self, profile: PrismProfile) -> ExecutionPlan:
        return ExecutionPlan({}, "float32", 0.0, "empty", {}, "lazy")

    def _fail_error(self, sorted_comps, devices):
        total_req = sum(m.total_mb for _, m in sorted_comps)
        total_avail = sum(d.capacity_mb for d in devices)
        comp_info = "\n".join(f"  {n}: {m.total_mb:.0f}MB (W={m.weight_mb:.0f}, A={m.activation_mb:.0f})" for n, m in sorted_comps)
        dev_info = "\n".join(f"  {d.device_string}: {d.capacity_mb:.0f}MB" for d in devices)
        # The full cascade depends on the device count (single-GPU vs multi-GPU
        # profile). The error message lists what Prism actually tried so the
        # user can correlate the failure with the strategy set in scope.
        # Hardcoded list previously only mentioned 5 of the 9 strategies,
        # misleading the diagnosis at P-SANA-4KPX-RUNTIME POINT 9.
        if len(devices) == 1:
            tried_str = ("single_gpu, single_gpu_lifecycle, lazy_sequential, "
                         "zero3 - ALL FAILED")
        else:
            tried_str = (
                "single_gpu, single_gpu_lifecycle, pipeline_parallel, "
                "component_placement, block_scatter, weight_sharding, "
                "component_placement_lazy, lazy_sequential, zero3 - "
                "ALL FAILED"
            )
        raise RuntimeError(
            f"ZERO FALLBACK: No strategy can fit this model.\n\n"
            f"Strategies tried: {tried_str}\n\n"
            f"Components:\n{comp_info}\n\n"
            f"Total required: {total_req:.0f}MB\n\n"
            f"GPUs:\n{dev_info}\n\n"
            f"Total available: {total_avail:.0f}MB\n\n"
            f"Solutions:\n  1. Use larger GPUs\n  2. Reduce resolution/batch\n"
            f"  3. Use smaller model\n"
            f"  4. CPU offload for the overflowing component "
            f"(open chantier P-PRISM-NEVER-REFUSE / "
            f"P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT for the architectural "
            f"work this requires)"
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
