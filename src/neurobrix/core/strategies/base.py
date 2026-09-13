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
from neurobrix.core.runtime.tensor_compat import is_torch_tensor
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

    # Components Prism classified as TRANSIENT — used once per request rather
    # than re-entered every step. Empty for every strategy but the lifecycle
    # one, whose acceptance budget is `persistent weights + one transient at a
    # time`. An eager strategy releases these after use and keeps the rest;
    # without this list "eager" means "release nothing", and the budget the
    # plan was accepted under is not the budget it runs under.
    transient_components: frozenset = frozenset()

    # component -> [[first_op_uid, last_op_uid], ...] for a component Prism
    # decided to stream at layer granularity. Empty for every other plan.
    # These are the segments the budget was ACCEPTED under, so the strategy
    # executes them rather than deriving its own and hoping they match.
    layer_segments: Dict[str, Any] = field(default_factory=dict)

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


_ACCELERATOR_PREFIXES = frozenset({"cuda", "hip", "xpu", "mps", "tt"})


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

    #: Does this strategy decide FOR ITSELF where its weights live at each
    #: moment — pinning them, streaming them, releasing them — rather than
    #: letting the runtime load a component once and leave it resident?
    #:
    #: The runtime asks this to know whether a component must be driven
    #: through `execute_component` and offered `install_for_executor`. It
    #: used to ask "is this zero3", which is a NAME, and a name cannot be
    #: extended: the second strategy to manage its own residency had no way
    #: to say so. This is the same shape of question as the vendor prefixes
    #: this engine stopped hard-coding — ask the thing, not its label.
    #:
    #: False by default: a strategy that does not say it manages residency
    #: is loaded and left resident, which is what every strategy but zero3
    #: did before this existed.
    manages_weight_residency: bool = False

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

    # Attribute names under which a runtime package may carry its artefact
    # path. Measured 2026-09-09: the executor's package carries `root_path`,
    # NBXContainer carries `cache_path`, and `nbx_path` — the ONLY name five
    # strategies used to read — exists on neither.
    _ARTIFACT_PATH_ATTRS = ('nbx_path', 'root_path', 'cache_path', 'path')

    def resolve_artifact_path(self, component_name: str) -> str:
        """Artefact path for `component_name`, or raise saying why not.

        This is the single place a strategy asks where the weights live.
        It never returns None: a strategy that cannot find the artefact
        cannot load the weights, and a component with no weights is not a
        lighter component, it is a wrong answer.
        """
        package = getattr(self.context, 'runtime_package', None)
        if package is None:
            raise RuntimeError(
                f"ZERO FALLBACK: '{component_name}' needs its weights loaded "
                f"but the strategy context ({type(self.context).__name__}) "
                f"carries no runtime package to load them from. Loading "
                f"nothing is not a way to continue.")

        for attr in self._ARTIFACT_PATH_ATTRS:
            value = getattr(package, attr, None)
            if value:
                return str(value)

        raise RuntimeError(
            f"ZERO FALLBACK: '{component_name}' needs its weights loaded but "
            f"the runtime package ({type(package).__name__}) names no artefact "
            f"path — tried {', '.join(self._ARTIFACT_PATH_ATTRS)}. It carries "
            f"{sorted(a for a in dir(package) if 'path' in a.lower())}.")

    def load_weights(self, component_name: str) -> None:
        """Load weights for component (lazy loading)."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for component '{component_name}'"
            )

        # Already loaded — the runtime usually gets there first
        # (`_ensure_weights_loaded`), and re-loading would be waste, not a
        # correction. This is the ONLY case where doing nothing is right.
        weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
        if weights_loaded:
            return

        # THREE SILENT SKIPS USED TO LIVE HERE, nested:
        #
        #     if hasattr(self.context, 'runtime_package') and ...:
        #         nbx_path = getattr(..., 'nbx_path', None)
        #         if nbx_path:
        #             executor.load_weights(...)
        #
        # and the middle one read a name that exists on NOTHING. Measured
        # 2026-09-09: the executor's package carries `root_path`, the
        # container carries `cache_path`, and `nbx_path` is absent from both.
        # So this method loaded nothing, ever, and said nothing about it —
        # six strategies call it.
        #
        # It happened to be harmless only because the runtime loads the
        # weights first, which is luck, not design: the one situation this
        # method exists for — weights NOT loaded — was also the one it
        # silently declined to handle.
        nbx_path = self.resolve_artifact_path(component_name)

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
            # Triton path — zero torch. Parse "<prefix>:N" → int device_idx.
            # `to_cuda` is the GENERIC accelerator move in this layer, not an
            # NVIDIA one. Matching only "cuda:" sent every hip:/mps:/xpu:
            # target into the else branch below and moved the tensor to the
            # HOST — asked for a GPU, it delivered CPU, silently and in the
            # wrong direction.
            prefix, _, _idx = str(target_device).partition(":")
            if prefix in _ACCELERATOR_PREFIXES:
                dev_idx = int(_idx) if _idx.isdigit() else 0
            elif prefix == "cpu":
                # CPU target — use NBXTensor.to_cpu for zero3-style
                # evictions (if ever called from strategy code).
                return tensor.to_cpu()
            else:
                # An unrecognised device is not a CPU request. Falling back to
                # to_cpu() here is what made the bug above silent.
                raise ValueError(
                    f"transfer_tensor: {target_device!r} names neither the host "
                    f"nor a known accelerator "
                    f"({', '.join(sorted(_ACCELERATOR_PREFIXES))})")
            return tensor.to_cuda(dev_idx)

        if is_torch_tensor(tensor):
            import torch
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
            if is_torch_tensor(value) or (
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
