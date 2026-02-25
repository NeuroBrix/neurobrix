"""
Executor Factory - Dispatch to appropriate executor based on Prism strategy.

ZERO HARDCODE: All values from ComponentAllocation (Prism).
ZERO FALLBACK: Explicit error if any required info missing.
ZERO IR: Uses TensorDAG directly, no reconstruction.

Architecture (100% ATen/Triton):
- TensorDAG → GraphExecutor → Triton kernels

All strategies are EXECUTABLE:
- single_gpu: Single GPU execution
- PP (Pipeline Parallel): Component placement across GPUs
- FGP (Fine-Grained Pipeline): Block-level sharding
- TP (Tensor Parallel): Head-parallel attention across GPUs

Component Handlers:
- Each component has a handler (VAE, Transformer, TextEncoder)
- Handlers encapsulate component-specific behavior (scaling, tiling, etc.)
- DATA-DRIVEN: All values from profile.json + runtime.json
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

from .graph_executor import GraphExecutor
from neurobrix.nbx.cache import ensure_extracted

if TYPE_CHECKING:
    from neurobrix.core.prism.solver import ComponentAllocation
    from neurobrix.core.components.base import ComponentHandler


class TPNotImplementedError(NotImplementedError):
    """Raised when trying to execute TP intent without TP-DAG-Rewriter."""
    pass


class ExecutorFactory:
    """
    Factory to create appropriate executor based on Prism allocation.

    PASSIVE CONSUMER: All info comes from ComponentAllocation.
    ZERO IR: Loads TensorDAG and passes directly to executors.
    No defaults, no inference, no fallbacks.
    """

    # Strategy to executor mapping — ALL 11 AllocationStrategy enum values.
    # Every strategy uses GraphExecutor. Execution flow is handled by strategies/ module.
    STRATEGY_MAP = {
        # === Single Device ===
        "single_gpu": "graph",
        "single_gpu_lifecycle": "graph",

        # === Pipeline Parallel (component-level) ===
        "pp_nvlink": "graph",
        "pp_pcie": "graph",
        "pp_lazy_nvlink": "graph",
        "pp_lazy_pcie": "graph",

        # === Fine-Grained Pipeline (block-level) ===
        "fgp_nvlink": "graph",
        "fgp_pcie": "graph",

        # === Tensor Parallel (tensor-level) ===
        "tp": "graph",

        # === Sequential / Offload ===
        "lazy_sequential": "graph",
        "zero3": "graph",
    }

    @classmethod
    def create(
        cls,
        component: str,
        allocation: "ComponentAllocation",
        nbx_path: str,
        dag: Dict[str, Any],
        mode: str = "compiled",
        skip_weights: bool = True,  # LAZY LOADING: Skip weights during creation
    ) -> GraphExecutor:
        """
        Create GraphExecutor for component based on Prism allocation.

        ZERO HARDCODE: All parameters extracted from allocation.
        ZERO FALLBACK: Missing info raises explicit error.
        ZERO IR: Uses TensorDAG directly (already loaded by NBXContainer).

        NOTE: All strategies now use GraphExecutor. The execution flow
        (pipeline parallelism, tensor parallel, zero3 offload) is handled
        by the strategies module (core/runtime/strategies/).

        Args:
            component: Component name
            allocation: Prism ComponentAllocation (SINGLE SOURCE OF TRUTH)
            nbx_path: Path to .nbx file (for weights)
            dag: TensorDAG dict (loaded by NBXContainer)
            mode: Execution engine mode: "compiled" (pre-compiled sequence, default),
                  "native" (Python loop + ATen), "triton" (Python loop + Triton)
            skip_weights: If True (default), skip weight loading during creation.
                         Weights should be loaded lazily via executor.load_weights()
                         before first execution. This prevents OOM from loading all
                         component weights upfront.

        Returns:
            GraphExecutor instance

        Raises:
            RuntimeError: Missing required info in allocation
        """
        # Validate DAG format (ZERO LEGACY)
        cls._validate_dag(dag, component)

        # Extract strategy (REQUIRED)
        strategy = cls._extract_strategy(allocation, component)

        # Map to executor type
        executor_type = cls.STRATEGY_MAP.get(strategy)
        if executor_type is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown strategy '{strategy}' for '{component}'.\n"
                f"Supported: {list(cls.STRATEGY_MAP.keys())}"
            )

        # Extract hardware info from allocation (REQUIRED)
        architecture = cls._extract_architecture(allocation, component)
        vendor = cls._extract_vendor(allocation, component)
        device = cls._extract_device(allocation, component)
        dtype = cls._extract_dtype(allocation, component)

        # All strategies now use GraphExecutor
        # Execution flow (PP, TP, Zero3) handled by strategies module
        return cls._create_graph_executor(
            component=component,
            allocation=allocation,
            nbx_path=nbx_path,
            dag=dag,
            architecture=architecture,
            vendor=vendor,
            device=device,
            dtype=dtype,
            mode=mode,
            skip_weights=skip_weights,
        )

    @classmethod
    def _validate_dag(cls, dag: Dict[str, Any], component: str) -> None:
        """
        Validate TensorDAG format.

        ZERO RELOAD: DAG already loaded by NBXContainer, just validate.
        """
        if not dag:
            raise RuntimeError(
                f"ZERO FALLBACK: Empty graph for '{component}'.\n"
                f"Component must have graph.json in .nbx"
            )

        format_type = dag.get("format", "")
        version = dag.get("version", "")

        if format_type != "tensor_dag":
            raise RuntimeError(
                f"ZERO LEGACY: Expected format 'tensor_dag', got '{format_type}'.\n"
                f"Component: {component}\n"
                f"Re-import the model with: neurobrix import ..."
            )

        if version != "0.1":
            raise RuntimeError(
                f"Unsupported container version '{version}' for '{component}'.\n"
                f"Expected: 0.1\n"
                f"Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

    @classmethod
    def _extract_strategy(cls, allocation: "ComponentAllocation", component: str) -> str:
        """Extract strategy from allocation. ZERO FALLBACK."""
        # Check for explicit strategy field first (from SmartSolver)
        if hasattr(allocation, "strategy"):
            strategy_val = allocation.strategy
            # Handle enum values
            if hasattr(strategy_val, "value"):
                return strategy_val.value
            # Handle string values
            if isinstance(strategy_val, str) and strategy_val:
                return strategy_val

        # Handle solver.py ComponentAllocation legacy (uses sharded bool + devices)
        if hasattr(allocation, "sharded"):
            if allocation.sharded:
                return "pp_pcie"
            elif allocation.devices == ["cpu"]:
                return "zero3"
            else:
                return "single_gpu"

        raise RuntimeError(
            f"ZERO FALLBACK: ComponentAllocation for '{component}' missing 'strategy'.\n"
            f"Prism must provide strategy in allocation."
        )

    @classmethod
    def _extract_architecture(cls, allocation: "ComponentAllocation", component: str) -> str:
        """Extract architecture from allocation. ZERO FALLBACK."""
        if hasattr(allocation, "architecture") and allocation.architecture:
            return allocation.architecture

        raise RuntimeError(
            f"ZERO FALLBACK: No 'architecture' in allocation for '{component}'.\n"
            f"Prism must include architecture (volta, ampere, hopper, etc.)."
        )

    @classmethod
    def _extract_vendor(cls, allocation: "ComponentAllocation", component: str) -> str:
        """Extract vendor from allocation. ZERO FALLBACK."""
        if hasattr(allocation, "vendor") and allocation.vendor:
            return allocation.vendor

        raise RuntimeError(
            f"ZERO FALLBACK: No 'vendor' in allocation for '{component}'.\n"
            f"Prism must include vendor (nvidia, amd, intel)."
        )

    @classmethod
    def _extract_device(cls, allocation: "ComponentAllocation", component: str) -> str:
        """Extract primary device from allocation. ZERO FALLBACK.

        Handles strategy-prefixed device formats:
        - "cuda:0" → "cuda:0"
        - "tp:cuda:0,cuda:1" → "cuda:0"
        - "zero3:cuda:2" → "cuda:2"
        - "fgp:cuda:0,cuda:1" → "cuda:0"
        """
        if hasattr(allocation, "device"):
            device = allocation.device
            # Strip strategy prefix: "strategy:cuda:N..." → "cuda:N..."
            known_strategies = ("tp", "pp", "fgp", "zero3")
            for prefix in known_strategies:
                tag = prefix + ":"
                if device.startswith(tag):
                    device = device[len(tag):]
                    break
            # For multi-device strings, take primary: "cuda:0,cuda:1" → "cuda:0"
            if "," in device:
                device = device.split(",")[0]
            return device
        if hasattr(allocation, "devices") and allocation.devices:
            return allocation.devices[0]

        raise RuntimeError(
            f"ZERO FALLBACK: No 'device' or 'devices' in allocation for '{component}'.\n"
            f"Prism must provide device assignment."
        )

    @classmethod
    def _extract_dtype(cls, allocation: "ComponentAllocation", component: str):
        """Extract dtype from allocation. ZERO FALLBACK."""
        import torch

        if not hasattr(allocation, "dtype"):
            raise RuntimeError(
                f"ZERO FALLBACK: No 'dtype' in allocation for '{component}'.\n"
                f"Prism must provide dtype."
            )

        dtype_val = allocation.dtype

        # Already torch.dtype
        if isinstance(dtype_val, torch.dtype):
            return dtype_val

        # String to dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if dtype_val not in dtype_map:
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown dtype '{dtype_val}' for '{component}'.\n"
                f"Supported: {list(dtype_map.keys())}"
            )

        return dtype_map[dtype_val]

    @classmethod
    def _extract_family(cls, nbx_path: str) -> str:
        """
        Extract family from manifest.json.

        ZERO HARDCODE: Family comes from manifest, not assumed.
        ZERO FALLBACK: Missing family raises explicit error.

        Args:
            nbx_path: Path to .nbx file

        Returns:
            Family string (image, llm, audio, video)
        """
        cache_path = ensure_extracted(Path(nbx_path))
        manifest_path = cache_path / "manifest.json"

        if not manifest_path.exists():
            raise RuntimeError(
                f"ZERO FALLBACK: manifest.json not found at {manifest_path}.\n"
                f"Re-import the model with: neurobrix import ..."
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        family = manifest.get("family")
        if not family:
            raise RuntimeError(
                f"ZERO FALLBACK: 'family' missing in manifest.json.\n"
                f"Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

        return family

    @classmethod
    def _infer_component_type(cls, component: str, dag: Dict[str, Any]) -> str:
        """
        Infer component type from name and DAG metadata.

        Args:
            component: Component name (e.g., "vae", "transformer", "text_encoder")
            dag: Component's TensorDAG

        Returns:
            Component type string
        """
        name_lower = component.lower()

        # Direct name matches
        if "vae" in name_lower:
            return "vae"
        if "transformer" in name_lower:
            return "transformer"
        if "text_encoder" in name_lower or "encoder" in name_lower:
            return "text_encoder"
        if "unet" in name_lower:
            return "unet"

        # Check DAG metadata if available
        metadata = dag.get("metadata", {})
        class_name = metadata.get("class_name", "")
        class_lower = class_name.lower()

        if "autoencoder" in class_lower or "vae" in class_lower:
            return "vae"
        if "transformer" in class_lower or "dit" in class_lower:
            return "transformer"
        if "t5" in class_lower or "clip" in class_lower or "encoder" in class_lower:
            return "text_encoder"

        return "unknown"

    @classmethod
    def _create_component_handler(
        cls,
        component: str,
        component_type: str,
        cache_path: str
    ) -> Optional["ComponentHandler"]:
        """
        Create component handler for the given component.

        Args:
            component: Component name
            component_type: Inferred component type
            cache_path: Path to extracted NBX cache

        Returns:
            ComponentHandler instance or None if creation fails
        """
        from neurobrix.core.components.registry import get_component_handler
        handler = get_component_handler(component, component_type, cache_path)
        return handler

    @classmethod
    def _create_graph_executor(
        cls,
        component: str,
        allocation: "ComponentAllocation",
        nbx_path: str,
        dag: Dict[str, Any],
        architecture: str,
        vendor: str,
        device: str,
        dtype,
        mode: str = "compiled",
        skip_weights: bool = True,
    ) -> GraphExecutor:
        """Create GraphExecutor for single-GPU execution.

        LAZY LOADING: When skip_weights=True (default), weights are NOT loaded
        during creation. Call executor.load_weights() before first execution.
        This prevents OOM from loading all component weights upfront.

        Component handlers are created to encapsulate component-specific behavior
        (e.g., VAE scaling, transformer pos_embed scaling).
        """
        # ZERO HARDCODE: Extract family from manifest (not assumed)
        family = cls._extract_family(nbx_path)

        # Create executor with Prism-provided values
        executor = GraphExecutor(
            family=family,
            vendor=vendor,
            arch=architecture,
            device=device,
            dtype=dtype,
            mode=mode,
        )

        # Load DAG directly (NO reconstruction)
        executor.load_graph_from_dict(dag)
        executor._component_name = component

        # Create and attach component handler (DATA-DRIVEN)
        cache_path = str(ensure_extracted(Path(nbx_path)))
        component_type = cls._infer_component_type(component, dag)
        handler = cls._create_component_handler(component, component_type, cache_path)
        executor._component_handler = handler

        # Store weight loading params for lazy loading
        shard_map = getattr(allocation, "shard_map", {})
        executor._weight_loading_params = {
            "nbx_path": nbx_path,
            "component": component,
            "shard_map": shard_map,
        }
        executor._weights_loaded = False

        # LAZY LOADING: Only load weights if skip_weights=False
        if not skip_weights:
            if shard_map:
                executor.load_weights(nbx_path, component, shard_map)
            else:
                executor.load_weights(nbx_path, component)
            executor._weights_loaded = True

        return executor
