"""Component Placement Execution Strategy — triton path (NBXTensor, zero torch).

Behavioural duplicate of core/strategies/component_placement.py, rebased on
TritonStrategy. Whole components are distributed across GPUs; activations
move across devices at component boundaries through NBXTensor.to_cuda
(via the TritonStrategy transfer helpers). Two-modes doctrine: same
contract as the pytorch strategy, NBXTensor compute path.
"""

from typing import Dict, Any, Optional, Set

from ..base import StrategyContext
from .base import TritonStrategy


class ComponentPlacementStrategy(TritonStrategy):
    """Component placement across multiple GPUs (triton path)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "component_placement"):
        super().__init__(context, strategy_name)

        self.devices = context.get_all_devices()
        self.use_async = True

        self.component_devices: Dict[str, str] = {}
        for comp_name, alloc_info in context.allocations.items():
            if isinstance(alloc_info, dict):
                device = alloc_info.get('device')
                assert device is not None, f"Allocation dict missing 'device' key for {comp_name}"
            else:
                assert isinstance(alloc_info, tuple), f"Allocation must be dict or tuple, got {type(alloc_info)}"
                device = alloc_info[0]
            self.component_devices[comp_name] = device

        self.state_device: Optional[str] = None

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component on its assigned GPU."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(f"ZERO FALLBACK: No executor for '{component_name}'")

        target_device = self.component_devices.get(component_name)
        if target_device is None:
            raise RuntimeError(f"ZERO FALLBACK: No device assignment for '{component_name}'")

        self.load_weights(component_name)

        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        return executor.run(inputs or {})

    def prepare_inputs(self, component_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer inputs to component's device (NBXTensor)."""
        target_device = self.component_devices.get(component_name)
        if target_device is None:
            return inputs

        transferred = self.transfer_dict(inputs, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred


    def transfer_model_output(self, model_output: Any, state_tensor: Any) -> Any:
        """Move the model output (NBXTensor) to the state tensor's device for
        the diffusion loop's residual update. NBXTensor device identity is
        carried by `_device_idx` (no torch `.device`)."""
        src_idx = getattr(model_output, "_device_idx", 0)
        dst_idx = getattr(state_tensor, "_device_idx", 0)
        if src_idx == dst_idx:
            return model_output
        target_device = f"cuda:{dst_idx}"
        transferred = self.transfer_tensor(model_output, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred


class ComponentPlacementLazyStrategy(ComponentPlacementStrategy):
    """Component placement with lazy weight swapping between phases (triton)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "component_placement_lazy"):
        super().__init__(context, strategy_name)
        self._phase_loaded: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if component_name not in self._phase_loaded:
            self.load_weights(component_name)
            self._phase_loaded.add(component_name)
        return super().execute_component(component_name, phase, inputs)

