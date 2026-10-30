"""Strategy steps the compiled and the Triton strategies ran alike, body for body — written once (Hocine 2026-10-29).

One mixin per strategy, inherited first by both of its classes. Torch-free at import (the Triton strategies inherit
them: R33); the tensors they move are each branch's own, through the strategy's own transfer methods.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class BlockScatterSteps:
    """What the compiled and the Triton `BlockScatterStrategy` ran alike (core/strategies/block_scatter.py, core/strategies/triton/block_scatter.py)."""

    def is_fgp_component(self, component_name: str) -> bool:
        return component_name in self.fgp_components

    def get_shard_map(self, component_name: str) -> Dict[str, str]:
        return self.fgp_components.get(component_name, {})

    def get_component_devices(self, component_name: str) -> List[str]:
        return self.component_devices.get(component_name, [self.primary_device])

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(f"ZERO FALLBACK: No executor for '{component_name}'")

        is_fgp = self.is_fgp_component(component_name)

        if component_name not in self._loaded_components:
            self._load_fgp_weights(component_name)
            self._loaded_components.add(component_name)

        if is_fgp and hasattr(executor, '_is_fgp_mode'):
            executor._is_fgp_mode = True

        if inputs:
            inputs = self.prepare_inputs(component_name, inputs)

        return executor.run(inputs or {})

    def _load_fgp_weights(self, component_name: str) -> None:
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            return

        weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
        if weights_loaded:
            return

        shard_map = self.get_shard_map(component_name)
        # Was: read `nbx_path` off the package, and if absent skip the
        # load without a word. That name exists on no package this
        # engine builds, so the load never happened and never said so.
        nbx_path = self.resolve_artifact_path(component_name)

        if shard_map:
            executor.load_weights(nbx_path, component_name, shard_map)
        else:
            executor.load_weights(nbx_path, component_name)

    def prepare_inputs(self, component_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        devices = self.get_component_devices(component_name)
        target_device = devices[0] if devices else self.primary_device
        return self.transfer_dict(inputs, target_device, self.use_async)


class ComponentPlacementLazySteps:
    """What the compiled and the Triton `ComponentPlacementLazyStrategy` ran alike (core/strategies/component_placement.py, core/strategies/triton/component_placement.py)."""

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


class ComponentPlacementSteps:
    """What the compiled and the Triton `ComponentPlacementStrategy` ran alike (core/strategies/component_placement.py, core/strategies/triton/component_placement.py)."""

    def prepare_inputs(self, component_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer inputs to component's device (NBXTensor)."""
        target_device = self.component_devices.get(component_name)
        if target_device is None:
            return inputs

        transferred = self.transfer_dict(inputs, target_device, self.use_async)
        if self.use_async:
            self.synchronize_device(target_device)
        return transferred


class CPUExecutionSteps:
    """What the compiled and the Triton `CPUExecutionStrategy` ran alike (core/strategies/cpu_execution.py, core/strategies/triton/cpu_execution.py)."""

    def unload_weights(self, component_name: str) -> None:
        if self._eager:
            return
        super().unload_weights(component_name)

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.transfer_dict(inputs, self.device)


class LazySequentialSteps:
    """What the compiled and the Triton `LazySequentialStrategy` ran alike (core/strategies/lazy_sequential.py, core/strategies/triton/lazy_sequential.py)."""

    def _get_component_strategy(self, component_name: str) -> str:
        alloc = self.context.allocations.get(component_name)
        if isinstance(alloc, dict):
            return alloc.get('strategy', 'single_gpu')
        return 'single_gpu'

    def _get_component_device(self, component_name: str) -> str:
        alloc = self.context.allocations.get(component_name)
        if isinstance(alloc, dict):
            return alloc.get('device', 'cpu')
        elif isinstance(alloc, tuple):
            return alloc[0]
        return 'cpu'

    def uninstall_for_executor(self, component_name: str,
                               executor=None) -> None:
        """Lifecycle mirror of install_for_executor (see the compiled
        LazySequential twin for the request-2 stale-install class)."""
        brick = getattr(self, "_zero3_brick", None)
        if brick is not None:
            brick.uninstall_for_executor(component_name, executor)

    def unload_weights(self, component_name: str) -> None:
        self.uninstall_for_executor(component_name)
        super().unload_weights(component_name)
        self._loaded_components.discard(component_name)
        self._pinned_components.discard(component_name)


class PipelineParallelSteps:
    """What the compiled and the Triton `PipelineParallelStrategy` ran alike (core/strategies/pipeline_parallel.py, core/strategies/triton/pipeline_parallel.py)."""

    def is_pp_component(self, component_name: str) -> bool:
        return component_name in self.pp_components

    def get_shard_map(self, component_name: str) -> Dict[str, str]:
        return self.pp_components.get(component_name, {})

    def get_component_devices(self, component_name: str) -> List[str]:
        return self.component_devices.get(component_name, [self.primary_device])

    def _load_pp_weights(self, component_name: str) -> None:
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            return

        weights_loaded = hasattr(executor, '_weights') and bool(executor._weights)
        if weights_loaded:
            return

        shard_map = self.get_shard_map(component_name)
        # Was: read `nbx_path` off the package, and if absent skip the
        # load without a word. That name exists on no package this
        # engine builds, so the load never happened and never said so.
        nbx_path = self.resolve_artifact_path(component_name)

        if shard_map:
            executor.load_weights(nbx_path, component_name, shard_map)
        else:
            executor.load_weights(nbx_path, component_name)

    def prepare_inputs(self, component_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        devices = self.get_component_devices(component_name)
        target_device = devices[0] if devices else self.primary_device
        return self.transfer_dict(inputs, target_device, self.use_async)


class SingleGPUSteps:
    """What the compiled and the Triton `SingleGPUStrategy` ran alike (core/strategies/single_gpu.py, core/strategies/triton/single_gpu.py)."""

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ensure inputs are on the assigned GPU (NBXTensor transfer)."""
        return self.transfer_dict(inputs, self.device)


class WeightShardingSteps:
    """What the compiled and the Triton `WeightShardingStrategy` ran alike (core/strategies/weight_sharding.py, core/strategies/triton/weight_sharding.py)."""

    def execute_component(
        self,
        component_name: str,
        phase: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(f"ZERO FALLBACK: No executor for component '{component_name}'")

        if component_name in self.sharded_components:
            primary_device = self.sharded_components[component_name][0]
        elif component_name in self.regular_components:
            primary_device = self.regular_components[component_name]
        else:
            primary_device = str(executor.device)

        if inputs:
            inputs = self.transfer_dict(inputs, primary_device, async_transfer=False)

        return executor.run(inputs or {})

    def prepare_inputs(self, component_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if component_name in self.sharded_components:
            target = self.sharded_components[component_name][0]
        elif component_name in self.regular_components:
            target = self.regular_components[component_name]
        else:
            return inputs
        return self.transfer_dict(inputs, target, async_transfer=False)

    def cleanup(self) -> None:
        self._loaded_components.clear()
