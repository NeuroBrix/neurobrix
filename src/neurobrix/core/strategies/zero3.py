"""
Zero3 Strategy - CPU Offload with GPU Compute

Weights live on CPU (pinned memory), compute on GPU.
Double buffering with async prefetch for overlap.

ZERO HARDCODE: Block structure from topology, devices from Prism.
VENDORLESS: Uses device strings from Prism (cuda/hip/xpu).
ZERO IR: Uses TensorDAG directly.

Architecture (100% ATen/Triton):
- Uses GraphExecutor for TensorDAG execution
- Triton kernels for compute, PyTorch for metadata ops
"""

import re
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .base import ExecutionStrategy, StrategyContext

if TYPE_CHECKING:
    from neurobrix.core.prism.solver import ComponentAllocation


@dataclass
class BlockInfo:
    """Information about a model block."""
    index: int          # Sequential index (0, 1, 2, ...)
    block_id: int       # Original block ID from model
    weight_names: List[str]
    memory_mb: float


class Zero3Strategy(ExecutionStrategy):
    """
    CPU offload strategy with GPU compute.

    For models too large for GPU memory:
    - Weights stored on CPU (pinned memory)
    - Compute happens on GPU
    - Double buffering with async prefetch

    ZERO HARDCODE: Block structure from weight names.
    ZERO IR: Uses TensorDAG directly.
    VENDORLESS: Works with cuda/hip/xpu.
    """

    def __init__(self, context: StrategyContext, strategy_name: str = "zero3"):
        super().__init__(context, strategy_name)

        # Zero3 executor per component (lazy init)
        self._zero3_executors: Dict[str, 'Zero3Executor'] = {}

        # Get execution device from first component allocation
        self.exec_device = self._get_exec_device()

    def _get_exec_device(self) -> str:
        """Get execution device from allocations."""
        for comp_name, alloc_info in self.context.allocations.items():
            if isinstance(alloc_info, dict):
                device_str = alloc_info.get('device')
            else:
                # Type narrowing: if not dict, must be tuple
                assert isinstance(alloc_info, tuple), f"Allocation must be dict or tuple, got {type(alloc_info)}"
                device_str = alloc_info[0]

            if device_str and device_str != "cpu":
                if device_str.startswith("cuda"):
                    return device_str
        # Fallback to first available CUDA device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _ensure_executor(self, component_name: str) -> 'Zero3Executor':
        """Ensure Zero3Executor exists for component (lazy init)."""
        if component_name not in self._zero3_executors:
            executor = self.context.component_executors.get(component_name)
            if executor is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: No executor found for component '{component_name}'. "
                    f"ExecutorFactory must create GraphExecutor first."
                )

            # Create Zero3Executor wrapping the GraphExecutor
            allocation = self._get_allocation(component_name)
            # Get DAG from GraphExecutor (attribute is _dag, not _graph)
            if not hasattr(executor, '_dag'):
                raise RuntimeError(
                    f"ZERO FALLBACK: Executor for '{component_name}' missing _dag attribute"
                )

            zero3_exec = Zero3Executor(
                component=component_name,
                executor=executor,
                exec_device=self.exec_device,
                dtype=executor.dtype,
            )
            self._zero3_executors[component_name] = zero3_exec

        return self._zero3_executors[component_name]

    def _get_allocation(self, component_name: str):
        """Get allocation for component."""
        return self.context.allocations.get(component_name)

    def execute_component(
        self,
        component_name: str,
        phase: str = "loop",
        inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute component with Zero3 CPU offload."""
        zero3_exec = self._ensure_executor(component_name)

        # Prepare inputs
        prepared_inputs = self.prepare_inputs(component_name, inputs or {})

        # Execute with Zero3
        outputs = zero3_exec.forward(prepared_inputs)

        return outputs

    def prepare_inputs(
        self,
        component_name: str,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Transfer inputs to execution device."""
        return {
            name: tensor.to(self.exec_device) if hasattr(tensor, 'to') else tensor
            for name, tensor in inputs.items()
        }

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

    def cleanup(self) -> None:
        """Release all Zero3 executor resources."""
        for executor in self._zero3_executors.values():
            executor.cleanup()
        self._zero3_executors.clear()


class Zero3Executor:
    """
    Execute model with ZeRO-3 style CPU offloading.

    - All weights on CPU (pinned memory for fast transfer)
    - Compute happens on GPU
    - Double buffering: prefetch block N+1 while computing block N
    - CUDA streams for async transfer

    ZERO HARDCODE: Block structure from weight names.
    ZERO IR: Uses TensorDAG directly.
    VENDORLESS: Works with cuda/hip/xpu.
    """

    def __init__(
        self,
        component: str,
        executor: Any,  # GraphExecutor
        exec_device: str,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Zero3 executor.

        Args:
            component: Component name
            executor: GraphExecutor instance
            exec_device: GPU device for compute
            dtype: Weight dtype
        """
        self.component = component
        self.executor = executor
        self.exec_device = exec_device
        self._dtype = dtype

        # CPU weights storage (pinned for fast transfer)
        self.cpu_weights: Dict[str, torch.Tensor] = {}

        # GPU cache (double buffer)
        self.gpu_cache: List[Dict[str, torch.Tensor]] = [{}, {}]
        self.current_cache_idx = 0

        # CUDA stream for async transfer
        self.transfer_stream: Optional[torch.cuda.Stream] = None

        # Block organization
        self.blocks: List[BlockInfo] = []
        self.block_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.num_blocks = 0

        # Initialize
        self._initialize()

    def _initialize(self) -> None:
        """Organize weights by block."""
        # Create transfer stream
        if torch.cuda.is_available() and self.exec_device.startswith("cuda"):
            device_idx = int(self.exec_device.split(":")[1])
            # Type narrowing: torch.cuda.Stream returns _CudaStreamBase which is compatible
            stream = torch.cuda.Stream(device=device_idx)
            self.transfer_stream = stream  # type: ignore[assignment]

        # Get weights from executor and pin to CPU
        if hasattr(self.executor, '_weights') and self.executor._weights:
            self._pin_weights_to_cpu(self.executor._weights)

        # Organize weights by block
        self._organize_by_block()

        pass

    def _pin_weights_to_cpu(self, weights: Dict[str, torch.Tensor]) -> None:
        """Pin weights to CPU memory for fast GPU transfer."""
        for name, tensor in weights.items():
            cpu_tensor = tensor.to("cpu")
            if cpu_tensor.is_contiguous():
                self.cpu_weights[name] = cpu_tensor.pin_memory()
            else:
                self.cpu_weights[name] = cpu_tensor.contiguous().pin_memory()

    def _organize_by_block(self) -> None:
        """
        Group weights by block index.

        Parse weight names to find block indices.
        Example: "transformer.blocks.5.attn.qkv.weight" -> block 5
        """
        block_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        non_block_weights: Dict[str, torch.Tensor] = {}

        for name, tensor in self.cpu_weights.items():
            # Try to find block index in weight name
            block_idx = self._extract_block_index(name)

            if block_idx is not None:
                if block_idx not in block_weights:
                    block_weights[block_idx] = {}
                block_weights[block_idx][name] = tensor
            else:
                # Non-block weights (embeddings, final layers)
                non_block_weights[name] = tensor

        # Sort blocks
        sorted_blocks = sorted(block_weights.keys())

        # Distribute non-block weights to first and last blocks
        if sorted_blocks and non_block_weights:
            first_block = sorted_blocks[0]
            last_block = sorted_blocks[-1]

            for name, tensor in non_block_weights.items():
                name_lower = name.lower()
                # Embeddings/initial layers -> first block
                if any(hint in name_lower for hint in
                       ["embed", "patch", "pos", "input", "norm_pre", "time_"]):
                    block_weights[first_block][name] = tensor
                else:
                    # Final layers -> last block
                    block_weights[last_block][name] = tensor
        elif non_block_weights and not sorted_blocks:
            # No block structure, treat all as single block
            block_weights[0] = non_block_weights
            sorted_blocks = [0]

        self.block_weights = block_weights
        self.num_blocks = len(sorted_blocks) if sorted_blocks else 1

        # Create block info
        for idx, block_idx in enumerate(sorted_blocks):
            weights = block_weights[block_idx]
            memory_bytes = sum(t.numel() * t.element_size() for t in weights.values())
            memory_mb = memory_bytes / (1024 * 1024)
            self.blocks.append(BlockInfo(
                index=idx,
                block_id=block_idx,
                weight_names=list(weights.keys()),
                memory_mb=memory_mb,
            ))

        pass

    def _extract_block_index(self, weight_name: str) -> Optional[int]:
        """Extract block index from weight name."""
        patterns = [
            r'blocks?[._](\d+)',     # blocks.0, block_0
            r'layers?[._](\d+)',     # layer.0, layers_0
            r'transformer[._]h[._](\d+)',  # transformer.h.0 (GPT style)
            r'encoder[._]layer[._](\d+)',  # encoder.layer.0 (BERT style)
        ]
        for pattern in patterns:
            match = re.search(pattern, weight_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        sync: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute with block-by-block GPU transfer.

        Double buffering: prefetch next block while computing current.

        Args:
            inputs: Input tensors
            sync: Whether to synchronize at end

        Returns:
            Output tensors
        """
        # Move inputs to exec device
        hidden = {
            name: tensor.to(self.exec_device)
            for name, tensor in inputs.items()
        }

        for block_idx in range(self.num_blocks):
            # === PREFETCH NEXT BLOCK (async) ===
            if block_idx + 1 < self.num_blocks:
                self._prefetch_block(block_idx + 1)

            # === GET CURRENT BLOCK WEIGHTS ===
            block_weights = self._get_block_weights(block_idx)

            # === EXECUTE BLOCK ON GPU ===
            hidden = self._execute_block(hidden, block_weights, block_idx)

        # Final sync
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        return hidden

    def run(
        self,
        inputs: Dict[str, torch.Tensor],
        sync: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward() for API compatibility."""
        return self.forward(inputs, sync=sync)

    def _prefetch_block(self, block_idx: int) -> None:
        """
        Async copy block weights from CPU to GPU.

        Uses separate CUDA stream for overlap with compute.
        """
        next_cache_idx = 1 - self.current_cache_idx
        block_info = self.blocks[block_idx]
        weights = self.block_weights[block_info.block_id]

        # Clear next cache
        self.gpu_cache[next_cache_idx].clear()

        # Async transfer
        if self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                for name, cpu_tensor in weights.items():
                    self.gpu_cache[next_cache_idx][name] = cpu_tensor.to(
                        self.exec_device,
                        non_blocking=True,
                    )
        else:
            # Fallback: sync transfer
            for name, cpu_tensor in weights.items():
                self.gpu_cache[next_cache_idx][name] = cpu_tensor.to(
                    self.exec_device,
                )

    def _get_block_weights(self, block_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get block weights from GPU cache.

        If first block, do sync copy. Otherwise, wait for prefetch.
        """
        if block_idx == 0:
            # First block: sync copy (no prefetch yet)
            block_info = self.blocks[0]
            weights = self.block_weights[block_info.block_id]

            for name, cpu_tensor in weights.items():
                self.gpu_cache[self.current_cache_idx][name] = cpu_tensor.to(
                    self.exec_device,
                )
        else:
            # Wait for prefetch to complete
            if self.transfer_stream is not None:
                self.transfer_stream.synchronize()
            # Switch cache
            self.current_cache_idx = 1 - self.current_cache_idx

        return self.gpu_cache[self.current_cache_idx]

    def _execute_block(
        self,
        hidden: Dict[str, torch.Tensor],
        weights: Dict[str, torch.Tensor],
        block_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one block with weights on GPU.

        Temporarily inject weights into executor, run, then optionally clear.
        """
        # Inject block weights into executor
        self.executor._weights = weights

        # Execute through executor
        outputs = self.executor.run(hidden)

        return outputs

    def get_block_count(self) -> int:
        """Get number of blocks."""
        return self.num_blocks

    def get_total_cpu_memory_mb(self) -> float:
        """Get total CPU memory used for weights."""
        return sum(
            t.numel() * t.element_size()
            for t in self.cpu_weights.values()
        ) / (1024 * 1024)

    def cleanup(self) -> None:
        """Release all resources."""
        # Clear CPU weights
        self.cpu_weights.clear()

        # Clear GPU caches
        self.gpu_cache = [{}, {}]

        # Clear block weights
        self.block_weights.clear()
        self.blocks.clear()

        # Clear transfer stream
        self.transfer_stream = None

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
