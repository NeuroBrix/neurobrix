"""Triton Executor — complete inference pipeline, zero torch dependency.

Orchestrates: weight loading → graph compilation → execution → output gathering.
Replaces GraphExecutor for --triton mode.
"""

import json
import os
import time
from typing import Dict, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype

from .sequence import TritonSequence
from .weight_loader import load_safetensors
from .constants import load_constants_from_graph


class TritonExecutor:
    """Triton-mode graph executor. Zero torch dependency.

    Usage:
        executor = TritonExecutor(graph_json, weights_dir, device_idx=2)
        executor.load()
        outputs = executor.run({"input::input_ids": nbx_tensor})
    """

    def __init__(self, graph_path: str, weights_dir: str, device_idx: int = 0,
                 dtype: str = "float16"):
        self.graph_path = graph_path
        self.weights_dir = weights_dir
        self.device_idx = device_idx
        self.target_dtype = parse_dtype(dtype)
        self._sequence: Optional[TritonSequence] = None
        self._loaded = False

    def load(self):
        """Load graph, compile, bind weights."""
        # 1. Read graph.json
        with open(self.graph_path, 'r') as f:
            dag = json.load(f)

        # 2. Compile
        self._sequence = TritonSequence(dag, device_idx=self.device_idx)
        self._sequence.compile()

        # 3. Load weights from safetensors
        weights = {}
        if os.path.isdir(self.weights_dir):
            for fname in sorted(os.listdir(self.weights_dir)):
                if fname.endswith('.safetensors'):
                    path = os.path.join(self.weights_dir, fname)
                    shard = load_safetensors(path, device_idx=self.device_idx)
                    weights.update(shard)

        # 4. Load constants from graph
        tensors = dag.get("tensors", {})
        constants = load_constants_from_graph(tensors, device_idx=self.device_idx)
        weights.update(constants)

        # 5. Bind weights
        self._sequence.bind_weights(weights)
        self._loaded = True

    def run(self, inputs: Dict[str, NBXTensor]) -> Dict[str, NBXTensor]:
        """Execute one forward pass.

        Args:
            inputs: Dict mapping input tensor IDs → NBXTensor

        Returns:
            Dict mapping output tensor IDs → NBXTensor
        """
        if not self._loaded:
            raise RuntimeError("Call load() before run()")

        # Bind inputs
        self._sequence.bind_inputs(inputs)

        # Bind symbols from input shapes
        self._sequence.bind_symbols(inputs)

        # Narrow seq-dependent constants (RoPE cos/sin) to actual seq_len
        self._sequence.update_seq_dependent_constants()

        # Decode steps (seq_len==1) reuse same-size intermediates every step.
        # Skip kill_slots to avoid cudaFree + sync overhead.
        is_decode = any(
            t.shape[1] == 1 for t in inputs.values()
            if hasattr(t, 'shape') and len(t.shape) >= 2)

        # Execute
        DeviceAllocator.set_device(self.device_idx)
        self._sequence.run(skip_kills=is_decode)

        # Gather outputs
        return self._sequence.gather_outputs()

    @property
    def num_ops(self) -> int:
        return self._sequence.num_ops if self._sequence else 0
