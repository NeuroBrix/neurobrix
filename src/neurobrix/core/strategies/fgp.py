"""
Fine-Grained Pipeline (FGP) Execution Strategy

Block-level distribution across multiple GPUs with NVLink transfers.

Key differences from Pipeline:
- Pipeline: Component-level distribution (text_encoder → GPU0, transformer → GPU1)
- FGP: Block-level distribution within a single component (blocks 0-7 → GPU0, 8-15 → GPU1)

FGP is needed when a single component is too large for any single GPU.
The component's blocks (layers) are distributed across GPUs, with tensor
transfers between blocks at GPU boundaries.

ARCHITECTURE:
- Prism provides shard_map: {"block.0.*": "cuda:0", "block.1.*": "cuda:1", ...}
- WeightLoader loads weights directly to their assigned GPUs
- GraphExecutor runs in FGP mode with device alignment per op
- This strategy handles:
  * Setting FGP mode on executors
  * Coordinating weight loading with shard maps
  * Managing tensor transfers between GPUs during execution
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import torch

from .base import ExecutionStrategy, StrategyContext


class FGPStrategy(ExecutionStrategy):
    """
    Fine-Grained Pipeline execution across multiple GPUs.

    Blocks within a component are distributed across GPUs.
    Tensor transfers happen at block boundaries when crossing GPU boundaries.

    Unlike Pipeline (component-level), FGP operates at block-level within
    a single component, enabling models where even one component exceeds
    single GPU memory.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "fgp_nvlink"):
        super().__init__(context, strategy_name)

        # Track devices and their assignments
        self.devices: List[str] = context.get_all_devices()
        self.use_async = "nvlink" in strategy_name.lower()

        # Parse FGP allocations from context
        # Format: ("fgp:cuda:0,cuda:1", {"block.0.weight": "cuda:0", ...})
        self.fgp_components: Dict[str, Dict[str, str]] = {}
        self.component_devices: Dict[str, List[str]] = {}

        for comp_name, alloc in context.allocations.items():
            if isinstance(alloc, dict):
                device_str = alloc.get('device', '')
                shard_map = alloc.get('shard_map', {})
            elif isinstance(alloc, tuple) and len(alloc) >= 2:
                device_str, shard_map = alloc[0], alloc[1]
            else:
                continue

            if isinstance(device_str, str) and device_str.startswith("fgp:"):
                # This is an FGP-distributed component
                device_list = device_str[4:].split(",")  # "fgp:cuda:0,cuda:1" -> ["cuda:0", "cuda:1"]
                self.fgp_components[comp_name] = shard_map
                self.component_devices[comp_name] = device_list
            else:
                # Regular single-device component
                if isinstance(device_str, str):
                    self.component_devices[comp_name] = [device_str]

        # Primary device for non-FGP operations
        self.primary_device = self.devices[0] if self.devices else "cuda:0"

        # Track loaded components
        self._loaded_components: Set[str] = set()

        pass

    def is_fgp_component(self, component_name: str) -> bool:
        """Check if component is FGP-distributed."""
        return component_name in self.fgp_components

    def get_shard_map(self, component_name: str) -> Dict[str, str]:
        """Get shard map for FGP component."""
        return self.fgp_components.get(component_name, {})

    def get_component_devices(self, component_name: str) -> List[str]:
        """Get list of devices for component."""
        return self.component_devices.get(component_name, [self.primary_device])

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute component in FGP mode.

        For FGP components:
        - Enable FGP mode on executor (device alignment per op)
        - Load weights with shard map (direct to GPUs)
        - Execute graph with runtime device alignment
        """
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for '{component_name}'"
            )

        is_fgp = self.is_fgp_component(component_name)
        devices = self.get_component_devices(component_name)
        device_str = ",".join(devices) if len(devices) > 1 else devices[0]

        # Load weights if not already loaded
        if component_name not in self._loaded_components:
            self._load_fgp_weights(component_name)
            self._loaded_components.add(component_name)

        # Set FGP mode on executor for proper device alignment
        # Note: _is_fgp_mode attribute may not exist on all executor types
        if is_fgp:
            if hasattr(executor, '_is_fgp_mode'):
                executor._is_fgp_mode = True  # type: ignore[attr-defined]
                pass

        # Prepare inputs
        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        # Execute
        outputs = executor.run(inputs or {})

        # Disable FGP mode after execution if needed
        # (keep enabled for iterative components that run multiple times)

        return outputs

    def _load_fgp_weights(self, component_name: str) -> None:
        """
        Load weights for FGP component using shard map.

        Delegates to executor's load_weights with shard_map parameter
        which triggers WeightLoader._load_component_fgp().
        """
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            return

        # Check if weights already loaded
        weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
        if weights_loaded:
            return

        shard_map = self.get_shard_map(component_name)
        devices = self.get_component_devices(component_name)

        # Get nbx_path from context
        nbx_path = None
        if hasattr(self.context, 'runtime_package') and self.context.runtime_package:
            nbx_path = getattr(self.context.runtime_package, 'nbx_path', None)

        if nbx_path:
            if shard_map:
                # The shard_map is passed through allocation and picked up by WeightLoader
                executor.load_weights(nbx_path, component_name, shard_map)
            else:
                executor.load_weights(nbx_path, component_name)

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare inputs for FGP execution.

        For FGP components, inputs go to the first device in the device list.
        The graph executor handles per-op device alignment.
        """
        devices = self.get_component_devices(component_name)
        target_device = devices[0] if devices else self.primary_device

        return self.transfer_dict(inputs, target_device, self.use_async)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle outputs from FGP component.

        Outputs may be on the last device in the FGP chain.
        Transfer to target_device if specified.
        """
        if target_device:
            # Find where outputs currently are
            source_device = None
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor):
                    source_device = str(val.device)
                    break

            if source_device and source_device != target_device:
                outputs = self.transfer_dict(outputs, target_device, self.use_async)
                if self.use_async:
                    self.synchronize_device(target_device)

        return outputs

    def unload_weights(self, component_name: str) -> None:
        """
        Unload weights for FGP component.

        Clears weights from all devices where they were loaded.
        """
        executor = self.context.component_executors.get(component_name)
        if executor is not None:
            # Check if weights loaded
            weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
            if weights_loaded:
                devices = self.get_component_devices(component_name)
                # Disable FGP mode before unload
                if hasattr(executor, '_is_fgp_mode'):
                    executor._is_fgp_mode = False  # type: ignore[attr-defined]

                executor.unload_weights()
                self._loaded_components.discard(component_name)

                # Synchronize and clear cache on all devices
                for device in devices:
                    self.synchronize_device(device)

                torch.cuda.empty_cache()

    def get_embedding_device(self, component_name: str) -> str:
        """
        Get device where embedding layer is located.

        For FGP components, embeddings are typically on the first device.
        This is important for LLM token embedding lookups.
        """
        devices = self.get_component_devices(component_name)
        return devices[0] if devices else self.primary_device

    def get_head_device(self, component_name: str) -> str:
        """
        Get device where output head is located.

        For FGP components, the output head is typically on the last device.
        This is important for LLM logit computation.
        """
        devices = self.get_component_devices(component_name)
        return devices[-1] if devices else self.primary_device

    def transfer_to_embedding_device(
        self,
        tensor: torch.Tensor,
        component_name: str,
    ) -> torch.Tensor:
        """Transfer tensor to embedding device for component."""
        target = self.get_embedding_device(component_name)
        return self.transfer_tensor(tensor, target, self.use_async)

    def transfer_to_head_device(
        self,
        tensor: torch.Tensor,
        component_name: str,
    ) -> torch.Tensor:
        """Transfer tensor to head device for component."""
        target = self.get_head_device(component_name)
        return self.transfer_tensor(tensor, target, self.use_async)

    def synchronize_all(self) -> None:
        """Synchronize all devices used in FGP execution."""
        for device in self.devices:
            self.synchronize_device(device)


class FGPNVLinkStrategy(FGPStrategy):
    """FGP with NVLink high-speed transfers (default)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "fgp_nvlink"):
        super().__init__(context, strategy_name)
        self.use_async = True


class FGPPCIeStrategy(FGPStrategy):
    """FGP with PCIe transfers (slower, for systems without NVLink)."""

    def __init__(self, context: StrategyContext, strategy_name: str = "fgp_pcie"):
        super().__init__(context, strategy_name)
        self.use_async = False
