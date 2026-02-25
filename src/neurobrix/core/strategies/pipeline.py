"""
Pipeline Execution Strategy

Components distributed across multiple GPUs.
Each component executes on its assigned GPU with tensor transfers between.

Two variants:
- PipelineStrategy (pp_nvlink, pp_pcie): Eager — weights stay in VRAM permanently.
- PipelineLazyStrategy (pp_lazy_nvlink, pp_lazy_pcie): Lazy — weights swap in/out
  between execution phases. Used when total model exceeds combined VRAM but each
  phase (pre_loop, loop, post_loop) fits. Weights are loaded before each phase
  and unloaded after.

Key features:
- Cross-device tensor transfers (automatic)
- NVLink async transfers when available
- State tracking for efficient transfers
"""

from typing import Dict, Any, Optional, Set
import torch

from .base import ExecutionStrategy, StrategyContext


class PipelineStrategy(ExecutionStrategy):
    """
    Pipeline execution across multiple GPUs.

    Components are on different GPUs (assigned by Prism).
    Tensors are automatically transferred between devices as needed.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "pipeline"):
        super().__init__(context, strategy_name)

        # Track which devices we're using
        self.devices = context.get_all_devices()
        self.use_async = "nvlink" in strategy_name.lower()

        # Map components to their devices
        self.component_devices: Dict[str, str] = {}
        for comp_name, alloc_info in context.allocations.items():
            if isinstance(alloc_info, dict):
                device = alloc_info.get('device')
                assert device is not None, f"Allocation dict missing 'device' key for {comp_name}"
            else:
                # Type narrowing: if not dict, must be tuple
                assert isinstance(alloc_info, tuple), f"Allocation must be dict or tuple, got {type(alloc_info)}"
                device = alloc_info[0]
            self.component_devices[comp_name] = device

        # Track current state device (for scheduler integration)
        self.state_device: Optional[str] = None

        pass

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute component on its assigned GPU.

        Handles tensor transfers from previous component's device.
        """
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for '{component_name}'"
            )

        # Get device for this component
        target_device = self.component_devices.get(component_name)
        if target_device is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No device assignment for '{component_name}'"
            )

        # Load weights if not loaded
        self.load_weights(component_name)

        # Prepare inputs (transfer to target device)
        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        # Execute
        outputs = executor.run(inputs or {})

        return outputs

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transfer inputs to component's device.

        Handles cross-device transfers for pipeline execution.
        """
        target_device = self.component_devices.get(component_name)
        if target_device is None:
            return inputs

        # Transfer all tensors to target device
        transferred = self.transfer_dict(inputs, target_device, self.use_async)

        # Sync if using async transfers
        if self.use_async:
            self.synchronize_device(target_device)

        return transferred

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle outputs - transfer to target device if specified.

        For scheduler integration, outputs may need to go to state device.
        """
        source_device = self.component_devices.get(component_name)

        # If target device specified and different, transfer
        if target_device and target_device != source_device:
            outputs = self.transfer_dict(outputs, target_device, self.use_async)
            if self.use_async:
                self.synchronize_device(target_device)

        return outputs

    def transfer_model_output(
        self,
        model_output: torch.Tensor,
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transfer model output to same device as state tensor.

        Used for scheduler.step() which needs both on same device.

        Args:
            model_output: Output from transformer/model
            state_tensor: Current latent state

        Returns:
            model_output on same device as state_tensor
        """
        if model_output.device == state_tensor.device:
            return model_output

        target_device = str(state_tensor.device)
        source_device = str(model_output.device)

        transferred = self.transfer_tensor(model_output, target_device, self.use_async)

        if self.use_async:
            self.synchronize_device(target_device)

        return transferred

    def get_component_device(self, component_name: str) -> str:
        """Get device for component."""
        device = self.component_devices.get(component_name)
        if device is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No device for component '{component_name}'"
            )
        return device

    def set_state_device(self, device: str) -> None:
        """Set the device where state tensors live."""
        self.state_device = device

    def get_state_device(self) -> Optional[str]:
        """Get current state device."""
        return self.state_device

    def ensure_on_device(
        self,
        tensor: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """
        Ensure tensor is on specified device.

        Convenience method for explicit device transfers.
        """
        return self.transfer_tensor(tensor, device, self.use_async)


class PipelineLazyStrategy(PipelineStrategy):
    """
    Pipeline execution with lazy weight swapping between phases.

    Same component-to-device mapping as PipelineStrategy, but weights are
    loaded before each execution phase and unloaded after. This allows
    models where total weights exceed combined VRAM, as long as each
    execution phase (pre_loop, loop, post_loop) fits.

    Weight lifecycle:
      Phase start → load_weights(component) → execute → unload_weights(component)
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "pp_lazy_nvlink"):
        super().__init__(context, strategy_name)
        self._phase_loaded: Set[str] = set()

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component with lazy load/unload around execution."""
        # Load weights for this component if not already loaded in this phase
        if component_name not in self._phase_loaded:
            self.load_weights(component_name)
            self._phase_loaded.add(component_name)

        # Execute using parent's logic
        result = super().execute_component(component_name, phase, inputs)

        return result

    def end_phase(self, phase: str) -> None:
        """
        Signal end of execution phase — unload weights loaded during this phase.

        Called by the executor between pre_loop/loop/post_loop phases
        to free VRAM for the next phase's components.
        """
        for comp_name in list(self._phase_loaded):
            self.unload_weights(comp_name)
        self._phase_loaded.clear()
        torch.cuda.empty_cache()
