"""
Component Placement Execution Strategy

Whole components distributed across multiple GPUs.
Each component executes on its assigned GPU with tensor transfers between.

Two variants:
- ComponentPlacementStrategy (component_placement): Eager — weights stay in VRAM permanently.
- ComponentPlacementLazyStrategy (component_placement_lazy): Lazy — weights swap in/out
  between execution phases. Used when total model exceeds combined VRAM but each
  phase (pre_loop, loop, post_loop) fits.

Key features:
- Cross-device tensor transfers (automatic)
- Async transfers when available
- State tracking for efficient transfers
"""

from typing import Dict, Any, Optional, Set
import torch

from .base import ExecutionStrategy, StrategyContext


class ComponentPlacementStrategy(ExecutionStrategy):
    """
    Component placement across multiple GPUs.

    Each component (text_encoder, transformer, vae) is on a different GPU.
    Tensors are automatically transferred between devices at component boundaries.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "component_placement"):
        super().__init__(context, strategy_name)

        self.devices = context.get_all_devices()
        self.use_async = True  # Always use async when available

        # Map components to their devices
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
        """Transfer inputs to component's device."""
        target_device = self.component_devices.get(component_name)
        if target_device is None:
            return inputs

        transferred = self.transfer_dict(inputs, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle outputs - transfer to target device if specified."""
        source_device = self.component_devices.get(component_name)
        if target_device and target_device != source_device:
            outputs = self.transfer_dict(outputs, target_device, self.use_async)
            if self.use_async:
                self.synchronize_device(target_device)
        return outputs

    def transfer_model_output(self, model_output: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
        """Transfer model output to same device as state tensor."""
        if model_output.device == state_tensor.device:
            return model_output
        target_device = str(state_tensor.device)
        transferred = self.transfer_tensor(model_output, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred

    def get_component_device(self, component_name: str) -> str:
        """Get device for component."""
        device = self.component_devices.get(component_name)
        if device is None:
            raise RuntimeError(f"ZERO FALLBACK: No device for component '{component_name}'")
        return device

    def set_state_device(self, device: str) -> None:
        self.state_device = device

    def get_state_device(self) -> Optional[str]:
        return self.state_device

    def ensure_on_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        return self.transfer_tensor(tensor, device, self.use_async)


class ComponentPlacementLazyStrategy(ComponentPlacementStrategy):
    """
    Component placement with lazy weight swapping between phases.

    Weights are loaded before each execution phase and unloaded after.
    Allows models where total weights exceed combined VRAM, as long as
    each phase (pre_loop, loop, post_loop) fits.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "component_placement_lazy"):
        super().__init__(context, strategy_name)
        self._phase_loaded: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component with lazy load/unload around execution."""
        if component_name not in self._phase_loaded:
            self.load_weights(component_name)
            self._phase_loaded.add(component_name)
        return super().execute_component(component_name, phase, inputs)

    def end_phase(self, phase: str) -> None:
        """Signal end of execution phase — unload weights loaded during this phase."""
        for comp_name in list(self._phase_loaded):
            self.unload_weights(comp_name)
        self._phase_loaded.clear()
        torch.cuda.empty_cache()
