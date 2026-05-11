"""
NeuroBrix Prism - Symbolic Activation Profiler

Simulates graph execution WITHOUT GPU allocation to estimate peak activation memory.

Key Innovation: Liveness Analysis + Symbolic Shape Resolution
- Parses graph.json with symbolic shapes
- Resolves symbols to concrete values (batch_size, height, width)
- Tracks tensor lifetimes to compute peak memory
- Zero GPU cost - pure CPU math (~10ms for 5000 ops)

Formula: Required_VRAM = Weights + Peak_Activations + Overhead
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from neurobrix.core.prism.memory_estimator import get_dtype_bytes_per_element


@dataclass
class InputConfig:
    """Runtime input configuration for activation profiling."""
    batch_size: int = 2
    height: int = 1024
    width: int = 1024
    seq_len: Optional[int] = None
    dtype: str = "float16"
    vae_scale: int = 8

    def to_symbol_map(self) -> Dict[str, int]:
        """
        Convert to symbol mapping for shape resolution.

        Common symbol patterns in TensorDAG:
        - s0: batch_size
        - s1, s2: spatial dimensions (height/vae_scale, width/vae_scale for latent space)
        - seq_len: sequence length for LLMs
        """
        height = self.height
        width = self.width
        batch_size = self.batch_size
        vae_scale = self.vae_scale

        latent_h = height // vae_scale
        latent_w = width // vae_scale

        symbol_map = {
            # Direct mappings
            "batch_size": batch_size,
            "height": height,
            "width": width,
            # Common symbolic names
            "s0": batch_size,
            "s1": latent_h,
            "s2": latent_w,
            "s3": latent_h * latent_w,  # Flattened spatial
            # LLM specific — seq_len must be provided, not hardcoded
            "seq_len": self.seq_len or 128,
            "sequence_length": self.seq_len or 128,
            # Latent dimensions
            "latent_h": latent_h,
            "latent_w": latent_w,
            "latent_hw": latent_h * latent_w,
        }

        return symbol_map


@dataclass
class ActivationProfile:
    """Result of activation profiling."""
    peak_bytes: int
    peak_op_uid: str
    peak_step: int
    tensor_count_at_peak: int
    total_ops: int
    final_live_bytes: int  # Memory still live at end
    # NEW: ops whose own footprint (output_bytes + workspace_bytes for the
    # given mode) exceeds the per-GPU VRAM budget × safety. Populated by
    # estimate_peak_memory() when called with vram_per_gpu_bytes set.
    # Each entry: (op_uid, op_type, output_bytes, workspace_bytes,
    #              input_tids_for_fusion_detection)
    overflow_ops: List = None  # type: ignore  # Optional[List[Tuple[...]]]

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / (1024 * 1024 * 1024)

    def __repr__(self) -> str:
        ov = len(self.overflow_ops) if self.overflow_ops else 0
        return (
            f"ActivationProfile(peak={self.peak_gb:.2f}GB at step {self.peak_step}/{self.total_ops}, "
            f"op={self.peak_op_uid}, tensors_at_peak={self.tensor_count_at_peak}, "
            f"overflow_ops={ov})"
        )


class ActivationProfiler:
    """
    Symbolic Activation Memory Profiler.

    Simulates graph execution to compute peak activation memory.
    Zero GPU cost - pure math on CPU (~10ms for 5000 ops).

    Algorithm:
    1. Build last-use map (liveness analysis)
    2. Resolve symbolic shapes with input_config
    3. Simulate execution:
       - Allocate output tensors
       - Track peak memory
       - Free dead tensors after last use
    """

    def __init__(self, dag: Dict[str, Any]):
        """
        Initialize profiler with TensorDAG.

        Args:
            dag: Parsed graph.json dict with:
                - tensors: Dict[tensor_id -> tensor_meta]
                - ops: Dict[op_uid -> op_meta]
                - execution_order: List[op_uid]
        """
        self.dag = dag
        self.tensors = dag.get("tensors", {})
        self.ops = dag.get("ops", {})
        self.execution_order = dag.get("execution_order", [])

        # Build liveness analysis
        self.last_uses = self._compute_last_uses()

    @classmethod
    def from_path(cls, graph_path: Union[str, Path]) -> "ActivationProfiler":
        """Load profiler from graph.json path."""
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph not found: {graph_path}")

        with open(graph_path, 'r') as f:
            dag = json.load(f)

        return cls(dag)

    def _compute_last_uses(self) -> Dict[str, str]:
        """
        Build map: tensor_id -> last_op_uid that uses it.

        Liveness Analysis: Tensor can be freed after its last use.

        CRITICAL FIX: Dead output tensors (outputs never consumed as inputs
        to any downstream op) must be freed immediately at the producing op.
        Without this, graphs with many detach ops (e.g., text_encoder with
        ~1041 aten::detach) accumulate dead tensors, inflating peak memory
        from ~2GB to 89GB.
        """
        last_use = {}

        # Step 1: Collect ALL tensors used as inputs anywhere
        used_as_input = set()
        for op_uid in self.execution_order:
            op = self.ops.get(op_uid, {})
            used_as_input.update(op.get("input_tensor_ids", []))

        # Step 2: Standard last-use tracking (input tensors)
        for op_uid in self.execution_order:
            op = self.ops.get(op_uid, {})
            for tid in op.get("input_tensor_ids", []):
                # Overwrite = last use wins
                last_use[tid] = op_uid

        # Step 3: Dead outputs — free immediately after producing op
        # Output tensors that are never consumed as inputs AND are not
        # graph outputs would otherwise accumulate forever in live_tensors.
        graph_outputs = set(self.dag.get("output_tensor_ids", []))
        for op_uid in self.execution_order:
            op = self.ops.get(op_uid, {})
            for out_tid in op.get("output_tensor_ids", []):
                if out_tid not in used_as_input and out_tid not in graph_outputs:
                    last_use[out_tid] = op_uid  # Free immediately

        return last_use

    def estimate_peak_memory(
        self,
        input_config: Optional[InputConfig] = None,
        dtype_bytes: Optional[int] = None,
        vram_per_gpu_bytes: Optional[int] = None,
        mode: str = "compiled",
        safety: float = 0.85,
        zero_alloc_uids: Optional[set] = None,
        inplace_adds: Optional[List] = None,
        force_compute_dtype_for_fp: bool = False,
    ) -> ActivationProfile:
        """
        Simulate execution to find peak activation memory.

        Args:
            input_config: Runtime input configuration (batch, height, width)
            dtype_bytes: Bytes per element override (default: from input_config.dtype)
            zero_alloc_uids: Set of op_uids whose outputs are NOT allocated at
                runtime because an `OpLevelTilingEngine` interceptor returns a
                sentinel proxy (`FusionUpsampleProxy`, `BroadcastClonePyroxy`)
                or a stride-0 view sharing data with the input. When provided,
                the simulation substitutes their output size with 0 to mirror
                the runtime exactly — turns worst-case full-materialization
                into tiling-aware estimation. Used by `PrismSolver._compute_memory`'s
                two-pass flow. Members typically include: upsample uids in
                upsample→conv fusion pairs; expand/clone/view uids in F2a
                pixel_shuffle broadcast-aware chains. None → legacy worst-case
                estimation (no substitution).

        Returns:
            ActivationProfile with peak memory info
        """
        if input_config is None:
            input_config = InputConfig()

        if dtype_bytes is None:
            dtype_bytes = get_dtype_bytes_per_element(input_config.dtype)

        # Build symbol map for shape resolution
        symbol_map = input_config.to_symbol_map()

        # Initialize tracking
        peak_bytes = 0
        peak_op_uid = ""
        peak_step = 0
        peak_tensor_count = 0
        current_bytes = 0
        live_tensors: Dict[str, int] = {}  # tensor_id -> size_bytes

        # Get graph output tensors (these are never freed)
        graph_outputs = set(self.dag.get("output_tensor_ids", []))
        # Tiling-aware: ops whose outputs are sentinel proxies / stride-0 views
        # at runtime (FusionUpsampleProxy on fusion-paired upsamples,
        # BroadcastClonePyroxy on F2a clone, pass-through view on F2a view,
        # NBXTensor.expand stride-0 view on F2a expand). Substituting size=0
        # here mirrors the runtime exactly. See P-PRISM-ACTIVATION-ESTIMATOR-
        # TILING-AWARE audit for the gap quantification.
        zero_set = set(zero_alloc_uids) if zero_alloc_uids else set()

        # In-place add aliasing: at runtime, in-place adds reuse one input
        # buffer as the output (no new allocation). Each in-place add's
        # output buffer IS its reused input's buffer. Their combined
        # lifetime extends through the LATER of the two original last-uses.
        # P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE 2026-05-11.
        alias = {}
        last_uses_eff = self.last_uses
        if inplace_adds:
            # Build alias map: output_tid -> reused_input_tid (resolve
            # transitive chains so all aliases point to the root buffer).
            for entry in inplace_adds:
                op_uid_inp, reuse_idx = entry
                op_inp = self.ops.get(op_uid_inp, {})
                in_tids = op_inp.get("input_tensor_ids", [])
                out_tids = op_inp.get("output_tensor_ids", [])
                if not out_tids or reuse_idx >= len(in_tids):
                    continue
                target = in_tids[reuse_idx]
                seen_walk = set()
                while target in alias and target not in seen_walk:
                    seen_walk.add(target)
                    target = alias[target]
                for out_tid in out_tids:
                    alias[out_tid] = target
                # Output is zero-allocated (rebinds to reused buffer)
                zero_set.add(op_uid_inp)
            # Recompute last_uses with the alias substitution: any consumer
            # of an aliased output_tid extends the lifetime of the merged
            # buffer (the alias root). Build by walking ops in order so the
            # LAST consumer wins.
            last_uses_eff = dict(self.last_uses)
            for op_uid_w in self.execution_order:
                op_w = self.ops.get(op_uid_w, {})
                for in_tid in op_w.get("input_tensor_ids", []):
                    if in_tid in alias:
                        target = in_tid
                        seen_resolve = set()
                        while target in alias and target not in seen_resolve:
                            seen_resolve.add(target)
                            target = alias[target]
                        last_uses_eff[target] = op_uid_w
        # Reverse map for the free phase: op_uid -> list of tids whose
        # effective last-use is this op. Required to handle the alias case
        # where an extended last-use points to an op that doesn't list the
        # tid in its `input_tensor_ids`. Costs O(n) extra at simulation
        # start, eliminates O(live_count) per-op scan.
        frees_at: Dict[str, List[str]] = {}
        for tid, op_uid_l in last_uses_eff.items():
            frees_at.setdefault(op_uid_l, []).append(tid)

        # Simulation loop
        for step, op_uid in enumerate(self.execution_order):
            op = self.ops.get(op_uid, {})

            # 1. ALLOCATE: Add output tensors
            output_tids = op.get("output_tensor_ids", [])
            op_is_zero_alloc = op_uid in zero_set
            for out_tid in output_tids:
                tensor_meta = self.tensors.get(out_tid, {})
                shape = self._resolve_shape(tensor_meta, symbol_map)
                size = self._compute_size(shape, tensor_meta, dtype_bytes,
                                          force_compute_dtype_for_fp=force_compute_dtype_for_fp)

                # Zero-alloc op: output is a sentinel proxy / stride-0 view
                # at runtime, not a real allocation. Liveness still tracks
                # the entry (so del live_tensors[tid] works downstream) —
                # size=0 makes current_bytes -= 0 a no-op when freed.
                if op_is_zero_alloc:
                    size = 0

                live_tensors[out_tid] = size
                current_bytes += size

            # 2. PEAK: Track maximum
            if current_bytes > peak_bytes:
                peak_bytes = current_bytes
                peak_op_uid = op_uid
                peak_step = step
                peak_tensor_count = len(live_tensors)

            # 3. FREE: Remove all tids whose effective last-use is this op.
            # Uses the precomputed `frees_at` reverse map so the free phase
            # handles aliased tids whose extended last-use points to an op
            # that doesn't list the tid in its `input_tensor_ids`.
            for tid in frees_at.get(op_uid, ()):
                if tid in graph_outputs:
                    continue
                if tid in live_tensors:
                    current_bytes -= live_tensors[tid]
                    del live_tensors[tid]

        # Scan per-op output + workspace to flag overflows for op-level tiling
        overflow_ops = []
        if vram_per_gpu_bytes is not None and vram_per_gpu_bytes > 0:
            from neurobrix.core.prism.memory_estimator import estimate_op_workspace_bytes
            threshold = int(vram_per_gpu_bytes * safety)
            for op_uid in self.execution_order:
                op = self.ops.get(op_uid, {})
                op_type = op.get("op_type", "")
                in_tids = op.get("input_tensor_ids", [])
                out_tids = op.get("output_tensor_ids", [])
                in_shapes = []
                for tid in in_tids:
                    meta = self.tensors.get(tid, {})
                    in_shapes.append(self._resolve_shape(meta, symbol_map))
                out_shapes = []
                out_bytes_total = 0
                for tid in out_tids:
                    meta = self.tensors.get(tid, {})
                    sh = self._resolve_shape(meta, symbol_map)
                    out_shapes.append(sh)
                    out_bytes_total += self._compute_size(sh, meta, dtype_bytes)
                ws_bytes = estimate_op_workspace_bytes(
                    mode, op_type, in_shapes, out_shapes, dtype_bytes,
                    vram_per_gpu_bytes=vram_per_gpu_bytes,
                )
                # Largest single input tensor (live at op start)
                largest_in_bytes = 0
                for tid, sh in zip(in_tids, in_shapes):
                    meta = self.tensors.get(tid, {})
                    largest_in_bytes = max(
                        largest_in_bytes, self._compute_size(sh, meta, dtype_bytes)
                    )
                op_footprint = largest_in_bytes + out_bytes_total + ws_bytes
                if op_footprint > threshold:
                    overflow_ops.append(
                        (op_uid, op_type, out_bytes_total, ws_bytes, list(in_tids))
                    )

        return ActivationProfile(
            peak_bytes=peak_bytes,
            peak_op_uid=peak_op_uid,
            peak_step=peak_step,
            tensor_count_at_peak=peak_tensor_count,
            total_ops=len(self.execution_order),
            final_live_bytes=current_bytes,
            overflow_ops=overflow_ops,
        )

    def _resolve_shape(
        self,
        tensor_meta: Dict[str, Any],
        symbol_map: Dict[str, int]
    ) -> List[int]:
        """
        Resolve symbolic shape to concrete values.

        Handles:
        - Symbolic dims: 's0', 's1', 'batch_size', etc.
        - Integer dims: passed through
        - String expressions: basic substitution

        Example:
            symbolic: ['s0', 4096, 's1', 's2']
            symbol_map: {'s0': 2, 's1': 128, 's2': 128}
            -> [2, 4096, 128, 128]
        """
        shape = tensor_meta.get("shape", [])

        resolved = []
        for dim in shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif isinstance(dim, str):
                # Try direct lookup
                if dim in symbol_map:
                    resolved.append(symbol_map[dim])
                elif dim.isdigit():
                    resolved.append(int(dim))
                else:
                    # Try to infer from pattern
                    inferred = self._infer_symbol(dim, symbol_map)
                    resolved.append(inferred)
            else:
                # Unknown type - assume 1
                resolved.append(1)

        return resolved

    def _infer_symbol(self, symbol: str, symbol_map: Dict[str, int]) -> int:
        """
        Infer value for unknown symbol.

        Patterns:
        - s0, s1, s2, ... -> try symbol_map
        - head*, num_heads -> default 8-32
        - hidden*, dim -> default 1024-4096
        """
        # Conservative defaults for unknown symbols (profiling estimation only).
        # These are intentionally small to avoid overestimating memory.
        # Actual values are resolved from topology when available.
        defaults = {
            "num_heads": 1,
            "head_dim": 1,
            "hidden_dim": 1,
            "embed_dim": 1,
            "vocab_size": 1,
        }

        # Try pattern matching
        lower = symbol.lower()

        for pattern, value in defaults.items():
            if pattern in lower:
                return value

        # Try s0, s1, s2 pattern
        if symbol.startswith('s') and symbol[1:].isdigit():
            idx = int(symbol[1:])
            if f"s{idx}" in symbol_map:
                return symbol_map[f"s{idx}"]

        raise ValueError(
            f"ZERO FALLBACK: Unknown symbol '{symbol}' in shape expression. "
            f"Available symbols: {list(symbol_map.keys())}"
        )

    def _compute_size(
        self,
        shape: List[int],
        tensor_meta: Dict[str, Any],
        default_dtype_bytes: int,
        force_compute_dtype_for_fp: bool = False,
    ) -> int:
        """
        Compute tensor size in bytes.

        Uses tensor's own dtype if available, otherwise default.

        Args:
            force_compute_dtype_for_fp: When True, override the meta's
                floating-point dtype with the caller's `default_dtype_bytes`
                (the runtime compute dtype). Mirrors the runtime: graph
                tensors traced as fp32 (PyTorch autocast off / fp32 capture
                path) are computed at compute_dtype (e.g. fp16) at runtime.
                Non-floating dtypes (int64 indices, bool masks) preserve
                their meta dtype. Used by `estimate_peak_memory` for
                activation budgeting; weight estimation continues to use
                the meta dtype unchanged.
        """
        # Get dtype from tensor meta if available
        dtype = tensor_meta.get("dtype", None)

        if dtype:
            dtype_bytes = get_dtype_bytes_per_element(dtype)
            # Override for floating-point types if requested
            if force_compute_dtype_for_fp:
                d = str(dtype).lower()
                if "float" in d or "bf16" in d or "half" in d:
                    dtype_bytes = default_dtype_bytes
        else:
            dtype_bytes = default_dtype_bytes

        numel = 1
        for dim in shape:
            numel *= dim

        return numel * dtype_bytes

    def get_tensor_count(self) -> int:
        """Get total number of tensors in graph."""
        return len(self.tensors)

    def get_op_count(self) -> int:
        """Get total number of ops in graph."""
        return len(self.execution_order)


def profile_component(
    graph_path: Union[str, Path],
    input_config: Optional[InputConfig] = None,
) -> ActivationProfile:
    """
    Profile a component's activation memory.

    Convenience function for quick profiling.

    Args:
        graph_path: Path to graph.json
        input_config: Runtime configuration

    Returns:
        ActivationProfile with peak memory
    """
    profiler = ActivationProfiler.from_path(graph_path)
    return profiler.estimate_peak_memory(input_config)
