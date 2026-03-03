"""
Zero3 Strategy - CPU Offload with GPU Compute

Weights live on CPU (pinned memory for fast DMA), compute on GPU.
CompiledSequence multi-device path handles per-op CPU→GPU transfers.

ZERO HARDCODE: All config from Prism plan and hardware profile.
VENDORLESS: Uses device strings from Prism (cuda/hip/xpu).
"""

import logging
import torch
from typing import Dict, Optional, Any, Set

from .base import ExecutionStrategy, StrategyContext

logger = logging.getLogger(__name__)


class Zero3Strategy(ExecutionStrategy):
    """
    CPU offload strategy with GPU compute.

    Weights are loaded to CPU via shard_map (Prism sets all keys → "cpu").
    Then pinned for fast DMA transfers during per-op execution.
    CompiledSequence multi-device path handles per-op CPU→GPU alignment.

    Pin memory decision is data-driven from cpu_ram_mb in ExecutionPlan.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "zero3"):
        super().__init__(context, strategy_name)
        self.exec_device = self._get_exec_device()
        self._use_pin_memory = True
        self._pinned_components: Set[str] = set()
        self._loaded_components: Set[str] = set()

    def _get_exec_device(self) -> str:
        """Get GPU execution device from allocations."""
        for alloc_info in self.context.allocations.values():
            if isinstance(alloc_info, dict):
                device_str = alloc_info.get('device', '')
            elif isinstance(alloc_info, tuple):
                device_str = alloc_info[0]
            else:
                continue

            if device_str and device_str.startswith(("cuda", "hip", "xpu")):
                return device_str

        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def execute_component(
        self,
        component_name: str,
        phase: str = "loop",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component with Zero3 CPU offload.

        Weights stay on CPU (pinned). Inputs move to exec_device.
        CompiledSequence multi-device path handles per-op weight transfer.
        """
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for '{component_name}'"
            )

        # Load weights if not loaded (they go to CPU via shard_map)
        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            self._loaded_components.add(component_name)

        # Pin CPU weights for fast DMA (once per component)
        if component_name not in self._pinned_components:
            self._pin_cpu_weights(component_name, executor)
            self._pinned_components.add(component_name)

        # Prepare inputs on exec device
        prepared = self.prepare_inputs(component_name, inputs or {})

        # Execute — CompiledSequence multi-device handles CPU→GPU per-op
        return executor.run(prepared)

    def _pin_cpu_weights(self, component_name: str, executor: Any) -> None:
        """Pin CPU weights for fast DMA transfers.

        Pinned memory enables non_blocking .to(device) at ~16GB/s
        vs ~8GB/s for regular CPU→GPU copies.

        Decision is data-driven from should_pin_memory() if cpu_ram_mb available.
        """
        weights = getattr(executor, '_weights', None)
        if not weights:
            return

        # Calculate total CPU weight size
        total_mb = sum(
            t.numel() * t.element_size()
            for t in weights.values()
            if isinstance(t, torch.Tensor) and t.device.type == "cpu"
        ) / (1024 * 1024)

        if total_mb == 0:
            return

        # Check if we should pin (data-driven from cpu_config)
        use_pin = True
        plan = getattr(self.context, '_plan', None)
        cpu_ram_mb = getattr(plan, 'cpu_ram_mb', 0) if plan else 0

        if cpu_ram_mb > 0:
            from neurobrix.core.prism.cpu_config import should_pin_memory, CPUConfig
            cpu = CPUConfig(
                model="runtime", cores=1, threads=1,
                ram_mb=cpu_ram_mb, architecture="",
            )
            use_pin = should_pin_memory(cpu, total_mb)

        if not use_pin:
            logger.info(
                f"[Zero3] {component_name}: skip pin_memory "
                f"(weights={total_mb:.0f}MB, ram={cpu_ram_mb}MB)"
            )
            return

        # Pin CPU weights in-place for fast DMA
        pinned_count = 0
        for name, tensor in list(weights.items()):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                if not tensor.is_pinned():
                    weights[name] = tensor.contiguous().pin_memory()
                    pinned_count += 1

        if pinned_count > 0:
            logger.info(
                f"[Zero3] {component_name}: pinned {pinned_count} weights "
                f"({total_mb:.0f}MB) for DMA to {self.exec_device}"
            )

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transfer inputs to GPU execution device."""
        return self.transfer_dict(inputs, self.exec_device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Outputs already on exec device."""
        if target_device and target_device != self.exec_device:
            return self.transfer_dict(outputs, target_device)
        return outputs

    def unload_weights(self, component_name: str) -> None:
        """Unload weights and clear pinned memory."""
        super().unload_weights(component_name)
        self._loaded_components.discard(component_name)
        self._pinned_components.discard(component_name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        """Release all resources."""
        self._loaded_components.clear()
        self._pinned_components.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
