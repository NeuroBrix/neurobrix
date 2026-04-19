"""
Zero3 Strategy — CPU Offload with GPU Compute + Block-Wise Pipelining

Last-resort Prism cascade when no other strategy fits the model in
aggregate VRAM. Weights live on CPU (pinned for fast DMA); compute
happens on a single GPU. Correctness contract: every op runs with its
args (weights included) on the GPU execution device.

Execution model: **block-wise ratchet pipelining** (no longer a
per-op slow-path). At any time at most two transformer blocks have
a GPU copy resident in the arena — the one currently executing
(block N) and the one being prefetched (N+1). Block N-1 is evicted
immediately after compute on block N begins, reclaiming its VRAM
before the next expand.

Async overlap is driven by a dedicated transfer stream:

    H2D(block N+1) on transfer_stream  ──┐
                                         │ event_record(block_N+1_ready)
                                         ▼
    compute(block N) on default stream   ▲
                                         │ stream_wait_event(block_N+1_ready)
                                         │ before first op of block N+1
                                         │
    evict(block N-1) on host             ▼

Eviction flow per block:
    1. materialize_slots_depending_on(weight_slot_ids)
         → copy any intermediate that aliases a block-N-1 weight (via
           `_base`/storage) into fresh storage. Safe because the
           moe_fusion.py Pass 2 output-sweep already prunes orphaned
           aten::t views at compile time, so on clean graphs this is
           a cheap no-op (0 slots materialized).
    2. rebind_partial({tid: cpu_original for tid in block_N-1})
       + recompute_op_devices_for_slots — arena slots point at CPU
       tensors again so subsequent pure-CPU ops from the block stay
       correct.
    3. Drop Python refs in gpu_cache[N-1] → NBXTensor.__del__
       → DeviceAllocator.free_cuda → PCIe-visible memory returned
       to the driver.

Priming (one-shot, first tick):
    * mark_cpu_weighted_ops_for_transfer(exec_dev): every weighted op
      whose weight is still CPU (never promoted) gets
      needs_transfer=True. Used only for weights outside any
      block — embeddings, final norm, lm_head. All block-weighted
      ops are handled by the ratchet, which rebinds them to GPU
      tensors before they execute.
    * override_weightless_op_devices(exec_dev): creation ops (arange,
      scalar_tensor, full) inherit device from the CPU-weighted chain
      so they'd allocate on CPU. Force them to exec_dev directly.

Multi-pass support (autoregressive decode):
    `_post_run_hook` fires after each forward pass. It evicts every
    block still in gpu_cache so the next pass starts from the same
    state as the first, avoiding VRAM growth across the decode loop.

Install model:
    Hooks installed at weight-load time so flow handlers that bypass
    strategy.execute_component (most notably GraphLMSession.prefill
    for autoregressive LLMs) get correct behaviour. GraphExecutor.run
    picks up executor._persistent_pre_op_callback and
    executor._post_run_hook transparently.

Polymorphism:
    Both native (CompiledSequence + torch.Tensor) and triton
    (TritonSequence + NBXTensor) paths are supported through a single
    ratchet — each tensor flavor exposes the same priming API
    (rebind_partial, recompute_op_devices_for_slots, get_op_blocks,
    materialize_slots_depending_on, override_weightless_op_devices,
    mark_cpu_weighted_ops_for_transfer). The transport primitives
    (to_cuda / to_cuda_async) live on the tensor, not on the strategy.

Portability:
    Stream + event API is backed by DeviceAllocator.create_stream /
    record_event / stream_wait_event via ctypes, routed through the
    CUDA or HIP runtime depending on the active backend. MPS /
    CPU-only execute devices fall back to synchronous prefetch (no
    stream) so the ratchet stays correct even without overlap.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set

import torch

from neurobrix.core.device_utils import device_empty_cache

from .base import ExecutionStrategy, StrategyContext

logger = logging.getLogger(__name__)


class Zero3Strategy(ExecutionStrategy):
    """CPU offload with block-wise GPU ratchet pipelining.

    Weights live on CPU (pinned for fast DMA). At most two blocks are
    resident on GPU at any time (current + prefetched). Async H2D on a
    dedicated transfer stream overlaps with compute on the default
    stream; per-block eviction returns VRAM to the driver immediately
    after the block's last op finishes.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "zero3"):
        super().__init__(context, strategy_name)
        self.exec_device = self._get_exec_device()
        self._loaded_components: Set[str] = set()
        self._pinned_components: Set[str] = set()
        self._installed: Dict[str, Any] = {}
        # component_name → ratchet state. Rebuilt lazily on first
        # callback tick since the compiled sequence isn't available
        # until the executor's bootstrap finishes.
        self._ratchet: Dict[str, Dict[str, Any]] = {}

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

    def _exec_dev_idx(self) -> int:
        """Parse the numeric device index out of self.exec_device."""
        if ":" in self.exec_device:
            try:
                return int(self.exec_device.split(":", 1)[1])
            except ValueError:
                return 0
        return 0

    def _pin_cpu_weights(self, component_name: str, executor: Any) -> None:
        """Pin CPU weights in-place for non_blocking DMA.

        Polymorphic on tensor type so both native (torch.Tensor) and
        triton (NBXTensor) paths get pinned host memory — native uses
        tensor.pin_memory(), triton uses NBXTensor.pin_host() which
        wraps cudaMallocHost via DeviceAllocator. Zero torch imports
        on the triton branch.

        In the triton path, the weight_loader already allocates CPU
        shards as pinned NBXTensors (see triton/weight_loader.py
        _load_to_pinned_cpu). This method becomes a no-op for those —
        pin_host() on an already-pinned tensor returns self. Kept for
        robustness: any CPU weight that slips through unpinned (e.g.
        from a custom loader, a test harness, or a future zero3
        variant) gets promoted here.

        Pinned memory doubles effective PCIe throughput AND is required
        for cudaMemcpyAsync H2D overlap on a non-default stream. With
        unpinned sources, the driver silently inserts a staging copy
        that blocks the transfer stream.
        """
        weights = getattr(executor, '_weights', None)
        if not weights:
            return

        total_mb = 0.0
        for t in weights.values():
            if isinstance(t, torch.Tensor) and t.device.type == "cpu":
                total_mb += t.numel() * t.element_size()
            elif hasattr(t, '_device') and getattr(t, '_device', None) == 'cpu':
                total_mb += t.numel() * t.element_size()
        total_mb /= (1024 * 1024)
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
            elif hasattr(tensor, '_device') and getattr(tensor, '_device', None) == 'cpu':
                if not getattr(tensor, '_pinned', False):
                    weights[name] = tensor.pin_host()
                    pinned += 1
        if pinned:
            logger.info(
                f"[Zero3] {component_name}: pinned {pinned} weights "
                f"({total_mb:.0f}MB) for DMA to {self.exec_device}"
            )

    # ------------------------------------------------------------------
    # Public hook API
    # ------------------------------------------------------------------

    def install_for_executor(self, component_name: str, executor: Any) -> None:
        """Install pinning, priming, and the ratchet callback.

        Called by RuntimeExecutor._ensure_weights_loaded so that flow
        handlers which bypass strategy.execute_component (most notably
        GraphLMSession.prefill calling executor.run directly) still
        benefit from the ratchet. Idempotent.
        """
        if component_name in self._installed:
            return
        self._pin_cpu_weights(component_name, executor)
        self._pinned_components.add(component_name)
        if not self.exec_device.startswith("cuda"):
            return
        self._install(component_name, executor)

    # ------------------------------------------------------------------
    # Ratchet state + install
    # ------------------------------------------------------------------

    def _is_triton_path(self, executor: Any) -> bool:
        """Triton path if the executor carries a _triton_seq attribute."""
        return getattr(executor, '_triton_seq', None) is not None

    def _seq_handle(self, executor: Any):
        """Return (seq, is_triton) or (None, None) if neither is ready."""
        tseq = getattr(executor, '_triton_seq', None)
        if tseq is not None:
            return tseq, True
        cseq = getattr(executor, '_compiled_seq', None)
        if cseq is not None:
            return cseq, False
        return None, None

    def _snapshot_cpu_originals(
        self, executor: Any, blocks: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Map tensor_id → current CPU NBXTensor / torch.Tensor.

        Captured once before the ratchet starts. These refs are what
        rebind_partial restores to when a block is evicted — without
        them the arena would have stale pointers to freed GPU memory.
        """
        weights = getattr(executor, '_weights', {}) or {}
        originals: Dict[str, Any] = {}
        for entry in blocks.values():
            for tid in entry.get('weight_tensor_ids', []):
                # Weight keys in executor._weights are stripped of the
                # "param::" prefix by the weight loader. Probe both.
                key_variants = (tid, tid[7:] if tid.startswith("param::") else tid)
                for k in key_variants:
                    if k in weights:
                        originals[tid] = weights[k]
                        break
        return originals

    def _build_op_to_block(
        self, blocks: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """Invert get_op_blocks() output into op_idx → block_idx."""
        op_to_block: Dict[int, int] = {}
        for bidx, entry in blocks.items():
            for op_idx in range(entry['first_op'], entry['last_op'] + 1):
                op_to_block[op_idx] = bidx
        return op_to_block

    def _weight_slot_ids_for_block(
        self, seq: Any, bidx: int, blocks: Dict[int, Dict[str, Any]]
    ) -> List[int]:
        """Translate block weight tensor_ids into arena slot indices."""
        entry = blocks.get(bidx)
        if entry is None:
            return []
        tid_to_slot = getattr(seq, '_tid_to_slot', {})
        slots: List[int] = []
        for tid in entry['weight_tensor_ids']:
            slot = tid_to_slot.get(tid)
            if slot is not None:
                slots.append(slot)
        return slots

    def _prefetch_block(
        self, state: Dict[str, Any], bidx: int, is_triton: bool
    ) -> None:
        """Async H2D for block `bidx` on the transfer stream.

        Populates state['gpu_cache'][bidx] with freshly allocated GPU
        tensors and records a readiness event. The default stream must
        call stream_wait_event on that event before consuming any of
        these tensors.
        """
        if bidx in state['gpu_cache']:
            return
        entry = state['blocks'].get(bidx)
        if entry is None:
            return
        dev_idx = state['exec_dev_idx']
        moved: Dict[str, Any] = {}
        stream = state['transfer_stream']
        if is_triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
            DA.set_device(dev_idx)
        else:
            torch.cuda.set_device(dev_idx)
        for tid in entry['weight_tensor_ids']:
            cpu_t = state['cpu_originals'].get(tid)
            if cpu_t is None:
                continue
            if is_triton:
                if getattr(cpu_t, '_device', 'cuda') != 'cpu':
                    continue
                if stream:
                    moved[tid] = cpu_t.to_cuda_async(dev_idx, stream=stream)
                else:
                    moved[tid] = cpu_t.to_cuda(dev_idx)
            else:
                if not (isinstance(cpu_t, torch.Tensor) and cpu_t.device.type == 'cpu'):
                    continue
                target = torch.device(f"cuda:{dev_idx}")
                if cpu_t.is_pinned() and state['transfer_stream_torch'] is not None:
                    with torch.cuda.stream(state['transfer_stream_torch']):
                        moved[tid] = cpu_t.to(target, non_blocking=True)
                else:
                    moved[tid] = cpu_t.to(target, non_blocking=True)
        state['gpu_cache'][bidx] = moved
        # Record readiness so the compute stream can wait on this block
        # before consuming any of its weights.
        if is_triton and stream:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
            ev = DA.create_event()
            DA.record_event(ev, stream=stream)
            state['block_events'][bidx] = ev
        elif not is_triton and state['transfer_stream_torch'] is not None:
            ev = torch.cuda.Event()
            ev.record(state['transfer_stream_torch'])
            state['block_events'][bidx] = ev

    def _install_block_on_arena(
        self, state: Dict[str, Any], bidx: int
    ) -> None:
        """Rebind arena slots of block bidx to its GPU-resident copy."""
        moved = state['gpu_cache'].get(bidx)
        if not moved:
            return
        seq = state['seq']
        modified = seq.rebind_partial(moved)
        seq.recompute_op_devices_for_slots(modified)

    def _wait_for_block(
        self, state: Dict[str, Any], bidx: int, is_triton: bool
    ) -> None:
        """Make the compute stream wait until block bidx is fully H2D'd.

        Drains the readiness event so subsequent ops on the default
        stream see committed data. Fast-path no-op if the event never
        got recorded (synchronous prefetch path).
        """
        ev = state['block_events'].pop(bidx, None)
        if ev is None:
            return
        if is_triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
            # stream=0 is the default (compute) stream.
            DA.stream_wait_event(0, ev, flags=0)
            DA.destroy_event(ev)
        else:
            # torch.cuda.Event.wait posts to the current stream.
            ev.wait()

    def _evict_block(
        self, state: Dict[str, Any], bidx: int, is_triton: bool
    ) -> None:
        """Release GPU memory of block bidx and restore CPU bindings.

        Order matters:
          1. Materialize any arena slot that aliases a block-bidx
             weight (rare — should be 0 on clean graphs, see
             moe_fusion.py Pass 2 comment).
          2. Rebind arena slots to the CPU originals. Subsequent
             fast-path reads on those slots see CPU tensors which
             route through the slow path automatically.
          3. Drop the Python refs in gpu_cache. NBXTensor.__del__
             (triton) or Python GC (native) calls into the allocator
             to free the GPU memory.
          4. Sync the device so outstanding reads on the freed
             tensors finish before we return.
        """
        if bidx not in state['gpu_cache']:
            return
        seq = state['seq']
        blocks = state['blocks']
        weight_slot_ids = self._weight_slot_ids_for_block(seq, bidx, blocks)

        # Step 1: break aliases
        seq.materialize_slots_depending_on(weight_slot_ids)

        # Step 2: restore CPU bindings
        restore_map: Dict[str, Any] = {}
        entry = blocks.get(bidx, {})
        for tid in entry.get('weight_tensor_ids', []):
            cpu_t = state['cpu_originals'].get(tid)
            if cpu_t is not None:
                restore_map[tid] = cpu_t
        if restore_map:
            modified = seq.rebind_partial(restore_map)
            seq.recompute_op_devices_for_slots(modified)

        # Step 3: drop GPU refs
        state['gpu_cache'].pop(bidx, None)

        # Step 4: sync so the next malloc sees the freed region
        if is_triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
            DA.sync_device()
        else:
            torch.cuda.synchronize()

    def _reset_ratchet(self, component_name: str) -> None:
        """Drop every GPU-cached block. Fired by _post_run_hook so
        multi-pass decode stays flat-VRAM."""
        state = self._ratchet.get(component_name)
        if state is None:
            return
        is_triton = state['is_triton']
        # Peak VRAM observation (criterion C diagnostics). Log once per
        # pass end when NBX_ZERO3_VRAM_LOG is set. Triton path reads
        # from DeviceAllocator (exact NBX accounting); native path reads
        # from torch.cuda.max_memory_allocated (device-wide — includes
        # any torch-cached allocations).
        if os.environ.get("NBX_ZERO3_VRAM_LOG"):
            dev_idx = state['exec_dev_idx']
            if is_triton:
                from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
                peak_mb = DA.peak_memory_allocated(dev_idx) / 1e6
                live_mb = DA.memory_allocated(dev_idx) / 1e6
                print(
                    f"[Zero3] {component_name} pass_end: "
                    f"peak_vram={peak_mb:.1f}MB live_vram={live_mb:.1f}MB "
                    f"blocks_cached={len(state['gpu_cache'])}"
                )
                DA.reset_peak_memory(dev_idx)
            else:
                peak_mb = torch.cuda.max_memory_allocated(dev_idx) / 1e6
                live_mb = torch.cuda.memory_allocated(dev_idx) / 1e6
                print(
                    f"[Zero3] {component_name} pass_end: "
                    f"peak_vram={peak_mb:.1f}MB live_vram={live_mb:.1f}MB "
                    f"blocks_cached={len(state['gpu_cache'])}"
                )
                torch.cuda.reset_peak_memory_stats(dev_idx)
        for bidx in list(state['gpu_cache'].keys()):
            self._evict_block(state, bidx, is_triton)
        # Also drop any unwaited readiness events.
        if is_triton:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
            for ev in state['block_events'].values():
                DA.destroy_event(ev)
        state['block_events'].clear()
        state['current_block'] = -1

    def _build_ratchet_state(
        self, component_name: str, executor: Any
    ) -> Optional[Dict[str, Any]]:
        """Construct per-component ratchet state. Returns None if the
        executor doesn't have block structure (e.g. image/audio VAE).

        Called once at the first pre_op_cb tick when the compiled
        sequence is guaranteed to be built and device-annotated.
        """
        seq, is_triton = self._seq_handle(executor)
        if seq is None:
            return None
        try:
            blocks = seq.get_op_blocks()
        except Exception as e:
            logger.warning(
                f"[Zero3] {component_name}: get_op_blocks failed ({e}); "
                f"falling back to unpipelined slow path"
            )
            return None
        # Filter out block -1 (embeddings / final norm / lm_head) — those
        # weights are small and stay resident across the whole pass.
        real_blocks = {bidx: entry for bidx, entry in blocks.items() if bidx >= 0}
        if not real_blocks:
            # No transformer blocks — nothing to pipeline.
            return None

        op_to_block = self._build_op_to_block(real_blocks)
        cpu_originals = self._snapshot_cpu_originals(executor, real_blocks)
        # If no block weight is CPU, there's nothing to promote — e.g.
        # Prism chose single_gpu but zero3 is still selected as strategy
        # for some reason. Skip pipelining; fall back to priming only.
        has_cpu_block_weights = any(
            (hasattr(t, '_device') and getattr(t, '_device', None) == 'cpu')
            or (isinstance(t, torch.Tensor) and t.device.type == 'cpu')
            for t in cpu_originals.values()
        )
        if not has_cpu_block_weights:
            return None

        # Stream setup. Transfer stream carries the async H2D; the
        # compute stream is the runtime default (stream=0).
        dev_idx = self._exec_dev_idx()
        transfer_stream = 0
        transfer_stream_torch = None
        if is_triton:
            try:
                from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
                DA.set_device(dev_idx)
                transfer_stream = DA.create_stream()
            except Exception as e:
                logger.warning(
                    f"[Zero3] {component_name}: transfer_stream create "
                    f"failed ({e}); falling back to synchronous prefetch"
                )
                transfer_stream = 0
        else:
            try:
                transfer_stream_torch = torch.cuda.Stream(device=dev_idx)
            except Exception:
                transfer_stream_torch = None

        state: Dict[str, Any] = {
            'seq': seq,
            'is_triton': is_triton,
            'exec_dev_idx': dev_idx,
            'blocks': real_blocks,
            'op_to_block': op_to_block,
            'cpu_originals': cpu_originals,
            'gpu_cache': {},
            'block_events': {},
            'current_block': -1,
            'transfer_stream': transfer_stream,
            'transfer_stream_torch': transfer_stream_torch,
            'primed': False,
            'executor': executor,
        }
        self._ratchet[component_name] = state
        logger.info(
            f"[Zero3] {component_name}: ratchet built — "
            f"{len(real_blocks)} blocks, window=2, "
            f"path={'triton' if is_triton else 'native'}, "
            f"stream={'async' if (transfer_stream or transfer_stream_torch) else 'sync'}"
        )
        return state

    def _ratchet_prime(self, state: Dict[str, Any], component_name: str) -> None:
        """One-shot priming sweep over the compiled op list.

        Flips CPU-weighted ops to slow-path transfer (for weights that
        NEVER get pipelined — block -1 weights) and forces weightless
        op devices to the exec GPU so creation ops allocate on the
        right device.
        """
        if state['primed']:
            return
        seq = state['seq']
        if state['is_triton']:
            dev = state['exec_dev_idx']
            n_flipped = seq.mark_cpu_weighted_ops_for_transfer(dev)
            seq.override_weightless_op_devices(dev)
            path_label = f"triton/cuda:{dev}"
        else:
            exec_dev_t = torch.device(self.exec_device)
            n_flipped = seq.mark_cpu_weighted_ops_for_transfer(exec_dev_t)
            seq.override_weightless_op_devices(exec_dev_t)
            path_label = f"compiled/{self.exec_device}"
        logger.info(
            f"[Zero3] {component_name}: primed {n_flipped} "
            f"non-pipelined CPU-weighted ops on {path_label}"
        )
        state['primed'] = True

    def _ratchet_on_op(
        self, state: Dict[str, Any], op_idx: int
    ) -> None:
        """Per-op ratchet step. Fast-path: same block as previous op,
        nothing to do. Slow path: transition into a new block,
        prefetch/evict/rebind as needed.
        """
        bidx = state['op_to_block'].get(op_idx)
        if bidx is None:
            # Op belongs to a non-block weight group (embed, head, norm).
            # Those weights are handled by the priming sweep — no action.
            return
        if bidx == state['current_block']:
            return

        # Transition into block `bidx`.
        is_triton = state['is_triton']

        # Step 1: make sure this block is prefetched. If we reached it
        # before the H2D fired (cold start, first block of a pass, or
        # a non-sequential op order), kick off a synchronous fetch.
        if bidx not in state['gpu_cache']:
            self._prefetch_block(state, bidx, is_triton)
        # Drain the readiness event (no-op for synchronous prefetch).
        self._wait_for_block(state, bidx, is_triton)
        # Bind arena to this block's GPU copy.
        self._install_block_on_arena(state, bidx)

        # Step 2: evict the block we just left (current_block), if any.
        prev = state['current_block']
        if prev >= 0 and prev != bidx and prev in state['gpu_cache']:
            self._evict_block(state, prev, is_triton)

        # Step 3: kick off prefetch of block bidx+1 for overlap.
        next_bidx = bidx + 1
        if next_bidx in state['blocks'] and next_bidx not in state['gpu_cache']:
            self._prefetch_block(state, next_bidx, is_triton)

        state['current_block'] = bidx

    def _install(self, component_name: str, executor: Any) -> None:
        """Attach ratchet + priming callbacks to the executor."""
        strategy = self

        def pre_op_cb(op_idx: int, op: Any) -> None:
            state = strategy._ratchet.get(component_name)
            if state is None:
                state = strategy._build_ratchet_state(component_name, executor)
                if state is None:
                    # Executor not ready or no block structure — nothing
                    # to pipeline. Return and try again next op.
                    return
            if not state['primed']:
                strategy._ratchet_prime(state, component_name)
            strategy._ratchet_on_op(state, op_idx)

        def post_run_hook() -> None:
            strategy._reset_ratchet(component_name)

        executor._persistent_pre_op_callback = pre_op_cb
        executor._post_run_hook = post_run_hook
        self._installed[component_name] = executor

    def _uninstall(self, component_name: str) -> None:
        """Reverse of _install. Called on unload_weights / cleanup."""
        executor = self._installed.pop(component_name, None)
        if executor is not None:
            executor._persistent_pre_op_callback = None
            executor._post_run_hook = None
        state = self._ratchet.pop(component_name, None)
        if state is not None:
            # Evict anything still resident, then tear down the stream.
            is_triton = state['is_triton']
            for bidx in list(state['gpu_cache'].keys()):
                self._evict_block(state, bidx, is_triton)
            if is_triton:
                from neurobrix.kernels.nbx_tensor import DeviceAllocator as DA
                for ev in state['block_events'].values():
                    DA.destroy_event(ev)
                ts = state.get('transfer_stream') or 0
                if ts:
                    DA.destroy_stream(ts)

    # ------------------------------------------------------------------
    # Strategy entry points
    # ------------------------------------------------------------------

    def execute_component(
        self,
        component_name: str,
        phase: str = "loop",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run a component under zero3 with ratchet pipelining."""
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
        self._ratchet.clear()
        device_empty_cache(self.exec_device)
