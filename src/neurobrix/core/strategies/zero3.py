"""
Zero3 Strategy — CPU Offload with GPU Compute

Last-resort Prism cascade when no other strategy fits the model in
aggregate VRAM. Weights live on CPU (pinned for fast DMA); compute
happens on a single GPU. Correctness contract: every op runs with its
args (weights included) on the GPU execution device.

This file's correctness fix:

    compute_op_devices() only marks the FIRST weighted op as
    needs_transfer=True (Phase 4's activation-device transition scan
    was designed for FGP, not zero3). Every subsequent weighted op
    stays on the fast path and crashes with "mat2 on cpu".

    Two one-shot flag sweeps patch that at install time:
        mark_cpu_weighted_ops_for_transfer(exec_dev): every weighted op
            whose weight is CPU gets needs_transfer=True. The slow path
            in _run_inner_multi_device moves each op's args to the GPU
            target per-op — working set = one op's weight, VRAM stays
            bounded.
        override_weightless_op_devices(exec_dev): creation ops (arange,
            scalar_tensor, full, attn-mask casts) inherit device from
            the CPU-weighted activation chain, so they'd allocate on
            CPU. Force them to exec_dev directly.

    Both sweeps run once at the first pre-op callback tick because
    that's the earliest point where CompiledSequence has
    compiled+bound+computed devices.

Install model:

    Hooks installed at weight-load time so flow handlers that bypass
    strategy.execute_component (most notably GraphLMSession.prefill for
    autoregressive LLMs) get correct behaviour. GraphExecutor.run picks
    up executor._persistent_pre_op_callback transparently.

Deferred: block-wise prefetch pipelining. The CompiledSequence APIs it
needs (get_op_blocks, rebind_partial, recompute_op_devices_for_slots)
and the pre_op_callback / post_run_hook plumbing are landed in this
commit — a correct pipelining implementation requires solving a VRAM
retention issue where evicted GPU tensors are not reclaimed even after
torch.cuda.synchronize + gc.collect + empty_cache. Tracked separately.

Portability note:
    Device strings hardcoded to "cuda:X" here, coherent with NBXTensor's
    current cuda hardcode. ROCm/MPS portability of zero3 will be handled
    in a dedicated coherent pass across NBXTensor and all strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

import torch

from neurobrix.core.device_utils import device_empty_cache

from .base import ExecutionStrategy, StrategyContext

logger = logging.getLogger(__name__)


class Zero3Strategy(ExecutionStrategy):
    """CPU offload with per-op GPU compute (slow-path correctness mode).

    Weights stay on CPU (pinned for fast DMA). At each op, the slow
    path in CompiledSequence._run_inner_multi_device transfers args to
    the GPU target, runs the op, and lets the temporaries fall out of
    scope — VRAM stays bounded to the working set of the current op.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "zero3"):
        super().__init__(context, strategy_name)
        self.exec_device = self._get_exec_device()
        self._loaded_components: Set[str] = set()
        self._pinned_components: Set[str] = set()
        # component_name → executor on which hooks are installed. Used
        # by unload_weights to uninstall cleanly.
        self._installed: Dict[str, Any] = {}
        # component_name → has the one-shot flag sweep run? Flips True
        # on the first callback tick, stays True until unload_weights.
        self._primed: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Device / setup helpers
    # ------------------------------------------------------------------

    def _get_exec_device(self) -> str:
        """Pick the GPU execution device from Prism allocations.

        Zero3's shard_map pins all weights to "cpu" in Prism's plan, so
        the allocation dict cannot be trusted as the compute device. We
        iterate looking for any non-cpu entry; Prism encodes the target
        as "zero3:cuda:N" in zero3 allocations, so we strip that prefix.
        Failing that, fall back to the runtime's best available
        accelerator. CPU-only is returned as last resort — the strategy
        is useless there but we don't crash early.
        """
        for alloc_info in self.context.allocations.values():
            device_str = ""
            if isinstance(alloc_info, dict):
                device_str = alloc_info.get('device', '') or ''
            elif isinstance(alloc_info, tuple) and alloc_info:
                device_str = alloc_info[0]
            if device_str.startswith("zero3:"):
                device_str = device_str.split(":", 1)[1]
            if device_str and device_str.startswith(("cuda", "hip", "xpu", "mps")):
                return device_str
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps:0"
        return "cpu"

    def _pin_cpu_weights(self, component_name: str, executor: Any) -> None:
        """Pin CPU weights in-place for non_blocking DMA.

        Pinned memory doubles effective PCIe throughput. Decision is
        data-driven from cpu_ram_mb in the Prism plan via
        should_pin_memory(); if CPU RAM is too small we skip and accept
        the slower DMA rather than risking an OOM on the CPU side.
        """
        weights = getattr(executor, '_weights', None)
        if not weights:
            return
        total_mb = sum(
            t.numel() * t.element_size()
            for t in weights.values()
            if isinstance(t, torch.Tensor) and t.device.type == "cpu"
        ) / (1024 * 1024)
        if total_mb == 0:
            return

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

        pinned = 0
        for name, tensor in list(weights.items()):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                if not tensor.is_pinned():
                    weights[name] = tensor.contiguous().pin_memory()
                    pinned += 1
        if pinned:
            logger.info(
                f"[Zero3] {component_name}: pinned {pinned} weights "
                f"({total_mb:.0f}MB) for DMA to {self.exec_device}"
            )

    # ------------------------------------------------------------------
    # Install: attach hooks to executor so ANY caller triggers the
    # one-shot priming, regardless of whether the flow went through
    # strategy.execute_component or bypassed it (autoregressive prefill).
    # ------------------------------------------------------------------

    def install_for_executor(self, component_name: str, executor: Any) -> None:
        """Public entry point invoked post-weight-load.

        Called by RuntimeExecutor._ensure_weights_loaded so that flow
        handlers which bypass strategy.execute_component (most notably
        GraphLMSession.prefill calling executor.run directly) still
        benefit from the correctness fix. Idempotent.
        """
        if component_name in self._installed:
            return
        self._pin_cpu_weights(component_name, executor)
        self._pinned_components.add(component_name)
        if not self.exec_device.startswith("cuda"):
            # Non-CUDA: nothing to install; the slow path in
            # CompiledSequence already handles CPU-only execution.
            return
        self._install(component_name, executor)

    def _install(self, component_name: str, executor: Any) -> None:
        """Attach a priming pre-op callback to the executor.

        The callback fires on every op but returns immediately after
        the first tick sets the priming flag. One-shot cost: two flag
        sweeps over self._ops at first run.
        """
        strategy = self
        self._primed[component_name] = False

        def pre_op_cb(op_idx: int, op: Any) -> None:
            if strategy._primed.get(component_name, False):
                return
            compiled_seq = getattr(executor, '_compiled_seq', None)
            if compiled_seq is None:
                return
            exec_dev_t = torch.device(strategy.exec_device)
            # Order matters: weighted ops need their device set from
            # arena contents, which still reflects the CPU binding at
            # install time. override_weightless then forces creation
            # ops onto exec_dev.
            n_flipped = compiled_seq.mark_cpu_weighted_ops_for_transfer(exec_dev_t)
            compiled_seq.override_weightless_op_devices(exec_dev_t)
            logger.info(
                f"[Zero3] {component_name}: primed slow-path for "
                f"{n_flipped} CPU-weighted ops on {strategy.exec_device}"
            )
            strategy._primed[component_name] = True

        executor._persistent_pre_op_callback = pre_op_cb
        self._installed[component_name] = executor

    def _uninstall(self, component_name: str) -> None:
        """Reverse of _install — called on unload_weights."""
        executor = self._installed.pop(component_name, None)
        if executor is None:
            return
        executor._persistent_pre_op_callback = None
        executor._post_run_hook = None
        self._primed.pop(component_name, None)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def execute_component(
        self,
        component_name: str,
        phase: str = "loop",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run a component under zero3 via the slow-path correctness mode."""
        executor = self.context.component_executors.get(component_name)
        if executor is None:
            raise RuntimeError(
                f"ZERO FALLBACK: No executor for component '{component_name}'"
            )

        if component_name not in self._loaded_components:
            self.load_weights(component_name)
            self._loaded_components.add(component_name)
        if component_name not in self._installed:
            self.install_for_executor(component_name, executor)

        prepared = self.prepare_inputs(component_name, inputs or {})
        return executor.run(prepared)

    def load_weights(self, component_name: str) -> None:
        """Load weights then install hooks immediately.

        Overrides base so the post-load install fires regardless of
        whether the flow reaches execute_component (diffusion) or
        bypasses it (autoregressive LLM prefill). Idempotent.
        """
        super().load_weights(component_name)
        executor = self.context.component_executors.get(component_name)
        if executor is not None:
            self.install_for_executor(component_name, executor)

    # ------------------------------------------------------------------
    # Inputs / outputs / cleanup
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Move model inputs to the GPU execution device."""
        return self.transfer_dict(inputs, self.exec_device)

    def handle_outputs(
        self,
        component_name: str,
        outputs: Dict[str, Any],
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward outputs; retarget only if the next component differs."""
        if target_device and target_device != self.exec_device:
            return self.transfer_dict(outputs, target_device)
        return outputs

    def unload_weights(self, component_name: str) -> None:
        """Uninstall hooks, then fall through to the base implementation."""
        self._uninstall(component_name)
        super().unload_weights(component_name)
        self._loaded_components.discard(component_name)
        self._pinned_components.discard(component_name)
        device_empty_cache(self.exec_device)

    def cleanup(self) -> None:
        """Release all zero3 resources — called when the strategy exits."""
        for name in list(self._installed.keys()):
            self._uninstall(name)
        self._loaded_components.clear()
        self._pinned_components.clear()
        self._primed.clear()
        device_empty_cache(self.exec_device)
