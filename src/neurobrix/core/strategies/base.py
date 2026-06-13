"""
Base Execution Strategy

Abstract base class for all execution strategies.
Strategies handle HOW components execute - device placement, tensor transfers, etc.
Prism decides WHAT strategy to use, strategies execute it.

ZERO SEMANTIC: Strategies know nothing about model domains (image, audio, llm).
They only know tensors, devices, and execution flow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import torch
from neurobrix.core.device_utils import device_sync

if TYPE_CHECKING:
    from neurobrix.core.runtime.graph_executor import GraphExecutor


@dataclass
class StrategyContext:
    """
    Shared context passed to strategy from executor.

    Contains everything strategy needs to execute components.
    Data-driven: all info comes from Prism plan and NBX container.
    """

    # From Prism execution plan
    strategy_name: str
    allocations: Dict[str, tuple]  # component_name -> (device, shard_map)

    # Component executors (GraphExecutor instances)
    component_executors: Dict[str, "GraphExecutor"] = field(default_factory=dict)

    # Variable resolver for inputs/outputs
    variable_resolver: Any = None

    # Topology for synthesis rules
    topology: Dict[str, Any] = field(default_factory=dict)

    # Runtime package
    runtime_package: Any = None

    # Loading mode from Prism plan: "lazy" (load/unload per component) or "eager" (keep all in memory)
    # Note: "persistent" is an alias for "eager" for backward compatibility
    loading_mode: str = "lazy"

    # Execution mode: "compiled" | "triton" | "triton_sequential". Drives
    # get_strategy()'s pytorch-vs-triton dispatch so the triton branch can
    # run NBXTensor-native strategies (zero torch) while compiled keeps the
    # torch path. Default "compiled" = byte-identical legacy behaviour.
    mode: str = "compiled"

    # Active component tracking (for lazy loading)
    _active_component: Optional[str] = None

    def get_device(self, component_name: str) -> str:
        """Get device for component from Prism allocation."""
        if component_name not in self.allocations:
            raise RuntimeError(
                f"ZERO FALLBACK: No allocation for component '{component_name}'. "
                f"Available: {list(self.allocations.keys())}"
            )
        alloc = self.allocations[component_name]
        # Handle both dict format (new) and tuple format (legacy)
        if isinstance(alloc, dict):
            device = alloc.get('device')
            assert device is not None, f"Allocation dict missing 'device' key for {component_name}"
            return device
        else:
            # Type narrowing: if not dict, must be tuple
            assert isinstance(alloc, tuple), f"Allocation must be dict or tuple, got {type(alloc)}"
            return alloc[0]

    def get_all_devices(self) -> List[str]:
        """Get all unique devices used in allocations."""
        devices = []
        for alloc in self.allocations.values():
            # Handle both dict format (new) and tuple format (legacy)
            if isinstance(alloc, dict):
                device = alloc.get('device')
                assert device is not None, "Allocation dict missing 'device' key"
                devices.append(device)
            else:
                # Type narrowing: if not dict, must be tuple
                assert isinstance(alloc, tuple), f"Allocation must be dict or tuple, got {type(alloc)}"
                devices.append(alloc[0])
        return list(set(devices))

    def is_multi_device(self) -> bool:
        """Check if components are on multiple devices."""
        return len(self.get_all_devices()) > 1


class ExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies.

    Strategies handle:
    1. Component execution on assigned device
    2. Tensor transfers between devices (for pipeline)
    3. Memory management (lazy loading, cleanup)

    Strategies do NOT handle:
    - Device placement decisions (that's Prism's job)
    - Model-specific logic (ZERO SEMANTIC)
    - Input/output binding (that's variable resolver's job)
    """

    def __init__(self, context: StrategyContext, strategy_name: str):
        """
        Initialize strategy with context.

        Args:
            context: Shared execution context
            strategy_name: Name of this strategy (for logging)
        """
        self.context = context
        self.strategy_name = strategy_name

    @abstractmethod
    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a component.

        Args:
            component_name: Name of component to execute
            phase: Execution phase (pre_loop, loop, post_loop)
            inputs: Optional input overrides

        Returns:
            Component outputs or None
        """
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare inputs for component execution.

        Handles device transfers if needed.

        Args:
            component_name: Target component
            inputs: Input tensors

        Returns:
            Inputs on correct device
        """
        pass

    @abstractmethod
    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle component outputs.

        May transfer to target device for next component.

        Args:
            component_name: Source component
            outputs: Output tensors
            target_device: Device for next component (if known)

        Returns:
            Outputs (possibly transferred)
        """
        pass

    def load_weights(self, component_name: str) -> None:
        """Load weights for component (lazy loading)."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for component '{component_name}'"
            )

        # Check if weights already loaded (by checking if _weights dict is populated)
        weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
        if not weights_loaded:
            device = self.context.get_device(component_name)
            # GraphExecutor.load_weights requires nbx_path and component parameters
            # These should be available from context.runtime_package
            if hasattr(self.context, 'runtime_package') and self.context.runtime_package:
                nbx_path = getattr(self.context.runtime_package, 'nbx_path', None)
                if nbx_path:
                    executor.load_weights(nbx_path, component_name)
                    self.context._active_component = component_name

    def unload_weights(self, component_name: str) -> None:
        """Unload weights for component (memory cleanup)."""
        executor = self.context.component_executors.get(component_name)
        if executor is not None:
            # Check if weights loaded
            weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
            if weights_loaded:
                executor.unload_weights()
                if self.context._active_component == component_name:
                    self.context._active_component = None

    def transfer_tensor(
        self,
        tensor: Any,
        target_device: str,
        async_transfer: bool = False,
    ) -> Any:
        """
        Transfer tensor to target device.

        Polymorphic on tensor type:
          - torch.Tensor → .to(device) on the native (PyTorch) path.
          - NBXTensor   → NBXTensor.to_cuda(device_idx) on the triton
            path (zero torch, uses DeviceAllocator under the hood).

        The two branches live here so strategies don't have to know
        which engine they're wrapping. Zero3Strategy on triton would
        otherwise try to call .pin_memory() / .to() on NBXTensor and
        crash — this single point of polymorphism keeps the triton
        code 100% torch-free while reusing the same strategy API.

        Args:
            tensor: Tensor to transfer (torch.Tensor or NBXTensor)
            target_device: Target device string (e.g., "cuda:0")
            async_transfer: Use non-blocking transfer (for NVLink). Only
                honored on the torch path today; NBXTensor.to_cuda
                queues on the default stream synchronously.

        Returns:
            Tensor on target device (same type as input).
        """
        # Duck-type detection: NBXTensor exposes to_cuda/to_cpu and
        # has _device attr. This avoids a hard import dependency on
        # kernels.nbx_tensor from a core/strategies path that may run
        # before triton kernels are loaded.
        if hasattr(tensor, 'to_cuda') and hasattr(tensor, '_device'):
            # Triton path — zero torch. Parse "cuda:N" → int device_idx.
            if target_device.startswith("cuda:"):
                dev_idx = int(target_device.split(":", 1)[1])
            elif target_device == "cuda":
                dev_idx = 0
            else:
                # CPU target — use NBXTensor.to_cpu for zero3-style
                # evictions (if ever called from strategy code).
                return tensor.to_cpu()
            return tensor.to_cuda(dev_idx)

        if isinstance(tensor, torch.Tensor):
            if str(tensor.device) == target_device:
                return tensor
            target = torch.device(target_device)
            if async_transfer:
                return tensor.to(target, non_blocking=True)
            else:
                return tensor.to(target)

        # Unknown tensor type — return unchanged.
        return tensor

    def transfer_dict(
        self,
        tensors: Dict[str, Any],
        target_device: str,
        async_transfer: bool = False,
    ) -> Dict[str, Any]:
        """
        Transfer all tensors in dict to target device.

        Args:
            tensors: Dict of tensors
            target_device: Target device
            async_transfer: Use non-blocking transfer

        Returns:
            Dict with tensors on target device
        """
        result = {}
        for key, value in tensors.items():
            # torch.Tensor OR NBXTensor (duck-typed: exposes to_cuda + _device).
            # transfer_tensor is already polymorphic over both; transfer_dict
            # previously matched only torch.Tensor, so in triton mode every
            # NBXTensor fell through to the pass-through branch and was NEVER
            # moved across devices — silently breaking component_placement /
            # pipeline_parallel multi-GPU on the triton path. For single-GPU
            # (target == current device) transfer_tensor is a no-op
            # (NBXTensor.to_cuda returns self), so this is regression-free.
            if isinstance(value, torch.Tensor) or (
                    hasattr(value, 'to_cuda') and hasattr(value, '_device')):
                result[key] = self.transfer_tensor(value, target_device, async_transfer)
            elif isinstance(value, dict):
                result[key] = self.transfer_dict(value, target_device, async_transfer)
            else:
                result[key] = value
        return result

    def synchronize_device(self, device: str) -> None:
        """Synchronize device (wait for async operations)."""
        device_sync(device)
