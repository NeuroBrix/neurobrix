"""
Graph Executor - Execute TensorDAG directly.

ZERO IR: The TensorDAG JSON is the ONLY truth.
ZERO RECONSTRUCTION: No intermediate data structures.

Architecture (decomposed):
- ExecutionContext: Shared state for a single run()
- TensorResolver: Tensor resolution from DAG to live tensors
- Op dispatch delegated to kernels/ infrastructure:
  - kernels/dispatch.py: Triton kernel dispatch (--triton mode)
  - kernels/metadata_ops.py: execute_metadata_op() for PyTorch native
  - core/runtime/graph/sequential_dispatcher.py: NativeATenDispatcher for ATen ops

Execution Modes:
- "compiled": Pre-compiled PyTorch execution sequence (DEFAULT, zero-overhead)
- "sequential": Pure PyTorch ATen ops, op-by-op eager (debug/compatibility)
- "triton": Triton-pure compiled (NeuroBrix arena + closures + fused kernels)
- "triton_sequential": Triton-pure op-by-op (debug Triton kernels)
See CLAUDE.md "Execution Modes" section for the full performance contract
between the four modes. The string value "native" is deprecated and
remapped to "sequential".

The executor mechanically plays the DAG:
1. Read op_data DIRECTLY from dag["ops"][op_uid]
2. Gather inputs via TensorResolver
3. Dispatch to kernel based on op_type
4. Store outputs via ExecutionContext
"""

import torch
import json
import time

import os
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Union, TYPE_CHECKING

from neurobrix.core.device_utils import device_empty_cache
from neurobrix.core.dtype.config import get_torch_dtype
from neurobrix.kernels.classification import OpExecution, get_execution_type

if TYPE_CHECKING:
    from neurobrix.core.components.base import ComponentHandler

from .graph.execution_context import ExecutionContext
from .graph.tensor_resolver import TensorResolver
from .graph.memory_pool import MemoryPool
from .graph.compiled_sequence import CompiledSequence
from neurobrix.core.memory import MemoryManager


class GraphStructureError(Exception):
    """Error in graph structure or format."""
    pass


class ExecutionStats:
    """Statistics from graph execution."""

    def __init__(
        self,
        total_ops: int = 0,
        triton_ops: int = 0,
        metadata_ops: int = 0,
        total_time_ms: float = 0.0,
    ):
        self.total_ops = total_ops
        self.triton_ops = triton_ops
        self.metadata_ops = metadata_ops
        self.total_time_ms = total_time_ms

    def __str__(self) -> str:
        return (
            f"Executed {self.total_ops} ops: "
            f"{self.triton_ops} triton, {self.metadata_ops} metadata "
            f"in {self.total_time_ms:.2f}ms"
        )


class GraphExecutor:
    """
    Execute TensorDAG directly.

    NO intermediate representation.
    NO reconstruction.
    The DAG JSON is the ONLY truth.

    Delegates to:
    - TensorResolver for tensor resolution
    - ExecutionContext for state management
    - kernels/ for op dispatch (EXISTING infrastructure)

    Usage:
        executor = GraphExecutor(family="image", vendor="nvidia", arch="volta")
        executor.load_graph("path/to/graph.json")
        executor.load_weights("path/to/model.nbx", "transformer", shard_map)
        outputs = executor.run(inputs)
    """

    def __init__(
        self,
        family: str,
        vendor: str,
        arch: str,
        device: str,
        dtype: torch.dtype = torch.float32,
        mode: str = "compiled",
    ):
        """
        Initialize GraphExecutor.

        Args:
            family: Model family (image, llm, audio, video)
            vendor: Hardware vendor (nvidia, amd, intel)
            arch: Architecture (volta, ampere, hopper)
            device: Device string (cuda:0, cuda:1, etc.)
            dtype: Execution dtype
            mode: Execution engine mode:
                  - "compiled": Pre-compiled PyTorch sequence (DEFAULT)
                  - "sequential": PyTorch ATen op-by-op eager (debug)
                  - "triton": Triton-pure compiled (NeuroBrix arena+closures)
                  - "triton_sequential": Triton-pure op-by-op (debug)
        """
        self.family = family
        self.vendor = vendor
        self.arch = arch
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self._sequential_dispatcher = None  # For ATen op dispatch (sequential mode)
        self._compiled_seq: Optional[CompiledSequence] = None  # For compiled mode (zero-overhead)
        self._compiled_exec_count = 0  # Track executions for verbose control

        self._dag = None
        self._weights = {}
        self._last_stats = None
        self._last_symbols = {}  # For CFG batch inference
        self._component_name = "unknown"

        # Symbolic shape resolver
        self._shape_resolver = None
        self._symbolic_shapes_enabled = False

        # Execution context and resolver (created per-run)
        self._ctx: Optional[ExecutionContext] = None
        self._resolver: Optional[TensorResolver] = None

        # Runtime resolution for pos_embed scaling
        self._runtime_height = None
        self._runtime_width = None

        # Component handler for component-specific behavior (DATA-DRIVEN)
        # Set by ExecutorFactory after creation
        self._component_handler: Optional["ComponentHandler"] = None

        # Persistent tensor IDs - prevent GC for specific tensors (e.g., hidden_states)
        self._persistent_tensor_ids: set = set()

        # Pre-compiled dispatch table: op_uid -> OpExecution type
        # Computed once in _init_from_dag(), avoids per-op get_execution_type() calls
        # _exec_type_map removed — old triton dispatch path eliminated

        # Memory pooling - DISABLED (PyTorch's internal caching allocator is optimal)
        self._use_memory_pool = False
        self._memory_pool: Optional[MemoryPool] = None

        # Op interceptors for KV cache injection (LLM execution)
        # Maps op_type (e.g., "aten::scaled_dot_product_attention") to interceptor callable
        # When empty (diffusion models), zero overhead — dict.get() on empty dict is ~30ns
        self._op_interceptors: Dict[str, Callable] = {}
        self._interceptors_dirty = False  # Track if interceptors changed after compilation

        # Weight loading params for lazy loading (set by ExecutorFactory)
        self._weight_loading_params: Optional[Dict[str, Any]] = None
        self._weights_loaded: bool = False

        # Persistent mode: when True, cleanup() and unload_weights() preserve weights in VRAM.
        # Set by serving layer for warm-compatible strategies (load once, serve many).
        self._persistent: bool = False

        # Per-call pre-op callback forwarded to CompiledSequence.run.
        # Set by run() and cleared in its finally block — used exclusively
        # by zero3 block-wise pipelining. None on the fast path.
        self._pre_op_callback: Optional[Callable[[int, Any], None]] = None

        # Persistent hooks installed by ExecutionStrategy (zero3) at
        # weight-load time. Survive across run() calls and are picked up
        # transparently, which is essential for flow handlers that
        # bypass strategy.execute_component and call executor.run
        # directly (e.g. GraphLMSession.prefill for autoregressive LLMs).
        # The pre-op hook takes priority over the per-call callback only
        # when the per-call one is None — explicit per-call wins.
        self._persistent_pre_op_callback: Optional[Callable[[int, Any], None]] = None
        self._post_run_hook: Optional[Callable[[], None]] = None

    # =========================================================================
    # Op Interceptor Registration (Phase 2.1: KV Cache Support)
    # =========================================================================

    def register_op_interceptor(self, op_type: str, interceptor: Callable) -> None:
        """
        Register an interceptor for a specific op type.

        Used for KV cache injection in LLM execution. The interceptor is called
        instead of the native op with the same arguments.

        Args:
            op_type: ATen op type (e.g., "aten::scaled_dot_product_attention")
            interceptor: Callable that receives (q, k, v, attn_mask, dropout_p, is_causal, scale, layer_idx)

        Note: If called after compilation, requires re-compilation for changes to take effect.
              Use register_op_interceptors() to batch-register and avoid repeated recompilation.
        """
        self._op_interceptors[op_type] = interceptor
        self._interceptors_dirty = True
        # Re-compile if already compiled to pick up new interceptor
        if self._compiled_seq is not None:
            self._compile_execution_sequence()
            self._interceptors_dirty = False

    def register_op_interceptors(self, interceptors: Dict[str, Callable]) -> None:
        """
        Batch-register multiple op interceptors.

        If compiled sequence exists, hot-swaps interceptor functions
        without recompilation. Otherwise marks dirty for next compile.

        Args:
            interceptors: Dict mapping op_type to interceptor callable
        """
        for op_type, interceptor in interceptors.items():
            self._op_interceptors[op_type] = interceptor
        if self._compiled_seq is not None:
            # Hot-swap: patch func references in existing compiled ops
            self._compiled_seq.update_op_interceptors(interceptors)
            self._interceptors_dirty = False
        else:
            self._interceptors_dirty = True

    def unregister_op_interceptor(self, op_type: str) -> None:
        """
        Remove an interceptor for a specific op type.

        Args:
            op_type: ATen op type to unregister
        """
        if op_type in self._op_interceptors:
            del self._op_interceptors[op_type]
            self._interceptors_dirty = True
            # Re-compile if already compiled
            if self._compiled_seq is not None:
                self._compile_execution_sequence()
                self._interceptors_dirty = False

    def register_op_uid_interceptors(self, interceptors: Dict[str, Callable]) -> None:
        """
        Register fine-grained per-op_uid interceptors on BOTH the compiled
        and triton sequences (R30: native + triton + triton_sequential).

        Targets specific op INSTANCES (e.g. only "aten.convolution::62") rather
        than every op of a given type. Used by the op-level tiling engine to
        intercept exactly the ops Prism flagged as VRAM overflows while
        leaving sibling ops of the same type on the native path.

        Hot-swaps if already compiled, otherwise marks dirty for next compile.
        Op_uid interceptors take priority over op_type interceptors.
        """
        if not hasattr(self, '_op_uid_interceptors'):
            self._op_uid_interceptors = {}
        for op_uid, interceptor in interceptors.items():
            self._op_uid_interceptors[op_uid] = interceptor
        if self._compiled_seq is not None:
            self._compiled_seq.update_op_uid_interceptors(interceptors)
            self._interceptors_dirty = False
        else:
            self._interceptors_dirty = True
        # Triton path mirror — register on _triton_seq if it exists, otherwise
        # stash for _ensure_triton_compiled to pick up.
        if hasattr(self, '_triton_seq') and self._triton_seq is not None:
            self._triton_seq.update_op_uid_interceptors(interceptors)
        else:
            if not hasattr(self, '_pending_triton_uid_interceptors'):
                self._pending_triton_uid_interceptors = {}
            self._pending_triton_uid_interceptors.update(interceptors)

    def _execute_intercepted_op(
        self,
        op_type: str,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute an op through its registered interceptor.

        Called by CompiledSequence when an op has a registered interceptor.
        The interceptor receives the same args/kwargs as the native op.

        Args:
            op_type: ATen op type (e.g., "aten::scaled_dot_product_attention")
            args: Positional arguments for the op
            kwargs: Keyword arguments for the op

        Returns:
            Result from the interceptor
        """
        interceptor = self._op_interceptors.get(op_type)
        if interceptor is None:
            raise RuntimeError(f"No interceptor registered for {op_type}")
        return interceptor(*args, **kwargs)

    def execute(self, inputs: Dict[str, Any],
                skip_kills: Optional[bool] = None) -> Dict[str, Any]:
        """
        Execute the graph with given inputs.

        Alias for run() to satisfy interface requirements.

        Args:
            inputs: Input tensors
            skip_kills: Optional explicit control of triton kill_slots.
                Defaults to None (run() treats None as False).

        Returns:
            Output tensors
        """
        return self.run(inputs, skip_kills=skip_kills)

    def _setup_executors(self) -> None:
        """
        Setup component executors.

        For GraphExecutor, this is a no-op as the executor is self-contained.
        Required by interface for compatibility with RuntimeExecutor.
        """
        pass

    def _setup_modules(self) -> None:
        """
        Setup auxiliary modules.

        For GraphExecutor, this is a no-op as modules are handled externally.
        Required by interface for compatibility with RuntimeExecutor.
        """
        pass

    def set_moe_config(self, norm_topk_prob: bool = True) -> None:
        """Set MoE routing configuration from lm_config.

        TIMING FIX: MoE fusion runs in __init__ (before this is called) and
        bakes norm_topk_prob into fused op attributes. We patch the DAG so
        that CompiledSequence reads the correct value when it compiles later.
        """
        self._moe_norm_topk_prob = norm_topk_prob

        # Patch fused op attributes in the DAG (fusion already ran with default=True)
        if self._dag:
            for _uid, op in self._dag.get("ops", {}).items():
                if op.get("op_type") == "custom::moe_fused":
                    op.setdefault("attributes", {})["norm_topk_prob"] = norm_topk_prob

    def set_runtime_resolution(self, height: int, width: int) -> None:
        """
        Set runtime resolution for positional embedding scaling.

        Called by RuntimeExecutor when resolution differs from trace-time.
        Stores resolution; scaling happens after weights are loaded.

        Args:
            height: Runtime pixel height
            width: Runtime pixel width
        """
        self._runtime_height = height
        self._runtime_width = width
        # Note: Scaling happens in load_weights() after weights are loaded

    # =========================================================================
    # Computable Buffer Methods (Position Embeddings)
    # =========================================================================

    def _compute_computable_buffers(self) -> None:
        """
        Compute all registered computable buffers at runtime resolution.

        This is called from load_weights() when:
        1. We have computable buffer specs registered from graph.json
        2. We have runtime resolution set (different from trace-time)

        For each computable buffer, calls the appropriate computation method
        based on the computation_spec stored in graph.json.
        """
        if not hasattr(self, "_computable_specs") or not self._computable_specs:
            return

        if not hasattr(self, "_maybe_to_nbx"):
            # Bind helper once. In triton modes a computed-at-runtime buffer
            # (e.g. sincos 2D pos embed, interpolated pos embed) must enter
            # the runtime as an NBXTensor; otherwise downstream ops crash
            # with `'Tensor' object has no attribute '_dtype'` (PixArt path).
            def _maybe_to_nbx(t):
                if self.mode not in ("triton", "triton_sequential"):
                    return t
                if not isinstance(t, torch.Tensor):
                    return t
                from neurobrix.kernels.nbx_tensor import (
                    NBXTensor, DeviceAllocator)
                import numpy as np
                dev_idx = (int(self.device.split(':')[1])
                           if isinstance(self.device, str) and ':' in self.device
                           else 0)
                DeviceAllocator.set_device(dev_idx)
                arr = np.ascontiguousarray(t.detach().cpu().numpy())
                return NBXTensor.from_numpy(arr)
            self._maybe_to_nbx = _maybe_to_nbx

        if self._runtime_height is None or self._runtime_width is None:
            # No runtime resolution set - use traced constant (should already be loaded)
            return

        computed_count = 0
        for weight_name, spec in self._computable_specs.items():
            method = spec.get("method")
            params = spec.get("params", {})

            if method == "sincos_2d_pos_embed":
                # Get vae_scale from spec params (may be pre-computed during import)
                vae_scale = params.get("vae_scale")
                if vae_scale is None and self._component_handler is not None:
                    # DATA-DRIVEN: Get from handler which reads from profile.json
                    vae_scale = self._component_handler.get_latent_scale()
                if vae_scale is None:
                    # ZERO FALLBACK: Crash explicitly - don't guess
                    raise RuntimeError(
                        "ZERO FALLBACK: Cannot determine vae_scale for sincos_2d_pos_embed. "
                        f"Neither 'vae_scale' in spec params nor component_handler available. "
                        "Ensure profile.json has 'block_out_channels' or 'vae_scale_factor'."
                    )

                # Get patch_size from spec (Sana=1, PixArt=2)
                patch_size = params.get("patch_size", 1)

                # Compute sincos positional embeddings at runtime resolution
                tensor = self._compute_sincos_2d_pos_embed(
                    runtime_height=self._runtime_height,
                    runtime_width=self._runtime_width,
                    embed_dim=params.get("embed_dim"),
                    base_size=params.get("base_size"),
                    interpolation_scale=params.get("interpolation_scale"),
                    vae_scale=vae_scale,
                    patch_size=patch_size,
                )
                self._weights[weight_name] = self._maybe_to_nbx(tensor)
                computed_count += 1
            elif method == "interpolate_learned_pos_embed":
                # Bilinearly interpolate learned positional embeddings
                tensor = self._interpolate_learned_pos_embed(
                    weight_name=weight_name,
                    runtime_height=self._runtime_height,
                    runtime_width=self._runtime_width,
                    traced_grid_h=params.get("traced_grid_h"),
                    traced_grid_w=params.get("traced_grid_w"),
                    embed_dim=params.get("embed_dim"),
                    traced_data=spec.get("traced_data"),  # Base64-encoded traced tensor
                    patch_size=params.get("patch_size", 2),  # Transformer patch size
                )
                self._weights[weight_name] = self._maybe_to_nbx(tensor)
                computed_count += 1

    def _compute_sincos_2d_pos_embed(
        self,
        runtime_height: int,
        runtime_width: int,
        embed_dim: int,
        base_size: int,
        interpolation_scale: float,
        vae_scale: int,
        patch_size: int,
    ) -> torch.Tensor:
        """
        Compute 2D sincos positional embeddings at runtime resolution.

        This replicates the exact computation used by diffusers SanaTransformer2DModel.
        Formula: grid_pos = arange(grid_size) / (grid_size / base_size) / interpolation_scale

        Args:
            runtime_height: Runtime pixel height
            runtime_width: Runtime pixel width
            embed_dim: Embedding dimension (e.g., 2240 for Sana)
            base_size: Base grid size from config (e.g., 128 for Sana 4K)
            interpolation_scale: Interpolation scale from config (e.g., 2.0 for Sana 4K)
            vae_scale: VAE scale factor (8 for PixArt/SD, 32 for Sana)
            patch_size: Transformer patch size (1 for Sana, 2 for PixArt)

        Returns:
            Positional embeddings tensor [1, seq_len, embed_dim]
        """
        import numpy as np

        # Calculate runtime grid size (latent dimensions / patch_size)
        # For Sana: 4096 / 32 / 1 = 128x128 grid
        # For PixArt: 1024 / 8 / 2 = 64x64 grid
        grid_h = runtime_height // vae_scale // patch_size
        grid_w = runtime_width // vae_scale // patch_size

        # Create grid positions with proper scaling (matches diffusers exactly)
        # Formula: grid_pos = arange(grid_size) / (grid_size / base_size) / interpolation_scale
        grid_h_positions = np.arange(grid_h, dtype=np.float64) / (grid_h / base_size) / interpolation_scale
        grid_w_positions = np.arange(grid_w, dtype=np.float64) / (grid_w / base_size) / interpolation_scale

        # Create 2D grid using meshgrid (x, y indexing)
        grid_w_coords, grid_h_coords = np.meshgrid(grid_w_positions, grid_h_positions)

        # Compute sincos embeddings
        # Half of embed_dim for height, half for width
        half_dim = embed_dim // 2

        # Frequency bands (same as diffusers get_1d_sincos_pos_embed_from_grid)
        omega = np.arange(half_dim // 2, dtype=np.float64)
        omega /= (half_dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)

        # Width embeddings: [grid_h * grid_w, half_dim]
        # NOTE: Diffusers computes W first, H second (despite misleading variable names in their code)
        w_flat = grid_w_coords.flatten()[:, np.newaxis]  # [seq, 1]
        w_emb = w_flat * omega[np.newaxis, :]  # [seq, half_dim/2]
        w_emb = np.concatenate([np.sin(w_emb), np.cos(w_emb)], axis=1)  # [seq, half_dim]

        # Height embeddings: [grid_h * grid_w, half_dim]
        h_flat = grid_h_coords.flatten()[:, np.newaxis]  # [seq, 1]
        h_emb = h_flat * omega[np.newaxis, :]  # [seq, half_dim/2]
        h_emb = np.concatenate([np.sin(h_emb), np.cos(h_emb)], axis=1)  # [seq, half_dim]

        # Concatenate: [W_emb, H_emb] to match diffusers exactly
        pos_embed = np.concatenate([w_emb, h_emb], axis=1)

        # Add batch dimension: [1, seq, embed_dim]
        pos_embed = pos_embed[np.newaxis, :, :]

        # Convert to torch tensor with target dtype
        pos_embed_tensor = torch.from_numpy(pos_embed).to(dtype=get_torch_dtype(self.dtype), device=self.device)

        return pos_embed_tensor

    def _interpolate_learned_pos_embed(
        self,
        weight_name: str,
        runtime_height: int,
        runtime_width: int,
        traced_grid_h: int,
        traced_grid_w: int,
        embed_dim: int,
        traced_data: Optional[str] = None,
        patch_size: int = 2,
    ) -> torch.Tensor:
        """
        Bilinearly interpolate learned positional embeddings at runtime resolution.

        This is used for models like PixArt that use learned (not sincos) pos_embed.
        The traced embedding is reshaped to 2D, bilinearly interpolated, then reshaped back.

        Args:
            weight_name: Weight name for loading traced tensor
            runtime_height: Runtime pixel height
            runtime_width: Runtime pixel width
            traced_grid_h: Traced grid height (from trace time)
            traced_grid_w: Traced grid width (from trace time)
            embed_dim: Embedding dimension
            traced_data: Base64-encoded traced tensor (if stored in spec)
            patch_size: Transformer patch size (default 2 for PixArt)

        Returns:
            Interpolated positional embeddings tensor [1, seq_len, embed_dim]
        """
        import torch.nn.functional as F

        # DATA-DRIVEN: Get vae_scale from handler
        vae_scale = None
        if self._component_handler is not None:
            vae_scale = self._component_handler.get_latent_scale()
        if vae_scale is None:
            # ZERO FALLBACK: Crash explicitly - don't guess
            raise RuntimeError(
                "ZERO FALLBACK: Cannot determine vae_scale for interpolate_learned_pos_embed. "
                "Ensure component_handler is set with valid profile.json containing "
                "'block_out_channels' or 'vae_scale_factor'."
            )

        # Calculate runtime grid size accounting for patch_size
        runtime_grid_h = runtime_height // vae_scale // patch_size
        runtime_grid_w = runtime_width // vae_scale // patch_size

        # Check if interpolation is needed
        traced_seq = traced_grid_h * traced_grid_w
        runtime_seq = runtime_grid_h * runtime_grid_w

        if traced_seq == runtime_seq:
            # No interpolation needed - load traced tensor directly
            if weight_name in self._weights:
                return self._weights[weight_name]
            # Buffer not in safetensors (e.g., pos_embed.pos_embed is a computed buffer).
            # Decode from traced_data stored in graph.json.
            if traced_data is not None:
                import base64
                import numpy as np
                traced_bytes = base64.b64decode(traced_data)
                traced_np = np.frombuffer(traced_bytes, dtype=np.float32).reshape(1, traced_seq, embed_dim)
                if np.any(traced_np != 0):
                    return torch.from_numpy(traced_np.copy()).to(dtype=get_torch_dtype(self.dtype), device=self.device)
                # traced_data is all zeros (VMM pool artifact) — recompute from sincos
                return self._compute_sincos_2d_pos_embed(
                    runtime_height=runtime_height,
                    runtime_width=runtime_width,
                    embed_dim=embed_dim,
                    base_size=traced_grid_h,  # base_size = grid_size at trace resolution
                    interpolation_scale=1.0,
                    vae_scale=vae_scale,
                    patch_size=patch_size,
                )

        # Get the traced positional embedding
        traced_embed = None

        # Try to load from traced_data (base64 encoded)
        if traced_data is not None:
            import base64
            import numpy as np
            traced_bytes = base64.b64decode(traced_data)
            traced_np = np.frombuffer(traced_bytes, dtype=np.float32).reshape(1, traced_seq, embed_dim)
            if not np.any(traced_np != 0):
                traced_np = None  # VMM pool artifact — skip zero data
            else:
                traced_embed = torch.from_numpy(traced_np).to(dtype=get_torch_dtype(self.dtype), device=self.device)

        # Try to load from weights file
        if traced_embed is None and weight_name in self._weights:
            traced_embed = self._weights[weight_name]

        if traced_embed is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Cannot interpolate {weight_name} - "
                f"no traced tensor found. Need traced_data in spec or loaded weight."
            )

        # Reshape to 2D grid: [1, seq, dim] -> [1, dim, h, w]
        pos_2d = traced_embed.squeeze(0).transpose(0, 1).reshape(1, embed_dim, traced_grid_h, traced_grid_w)

        # Bilinear interpolation
        pos_2d_scaled = F.interpolate(
            pos_2d.float(),
            size=(runtime_grid_h, runtime_grid_w),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back to [1, seq, dim]
        scaled_embed = pos_2d_scaled.reshape(embed_dim, runtime_seq).transpose(0, 1).unsqueeze(0)
        scaled_embed = scaled_embed.to(dtype=get_torch_dtype(self.dtype))

        return scaled_embed

    # =========================================================================
    # Graph Loading
    # =========================================================================

    def load_graph(self, graph_path: Union[str, Path]) -> None:
        """
        Load TensorDAG as-is. NO transformation.

        Supports symbolic shapes.

        Args:
            graph_path: Path to graph.json (TensorDAG format)

        Raises:
            GraphStructureError: If format is not tensor_dag v2.x or v3.x
        """
        with open(graph_path) as f:
            self._dag = json.load(f)

        self._init_from_dag()

    def _init_from_dag(self) -> None:
        """Initialize executor from loaded DAG."""
        assert self._dag is not None
        # Validate format
        format_type = self._dag.get("format", "")
        version = self._dag.get("version", "")

        if format_type != "tensor_dag":
            raise GraphStructureError(
                f"Expected format 'tensor_dag', got '{format_type}'.\n"
                f"Model graph invalid. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

        if version != "0.1":
            raise GraphStructureError(
                f"Unsupported container version '{version}'.\n"
                f"Expected: 0.1\n"
                f"Model graph invalid. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )
        # tensor_dag format uses symbolic shapes
        self._symbolic_shapes_enabled = True
        from .shape_resolver import SymbolicShapeResolver
        symbolic_context = self._dag.get("symbolic_context", {})
        self._shape_resolver = SymbolicShapeResolver(symbolic_context)

        self._component_name = self._dag.get("component_name")

        # Extract graph dtype for DtypeEngine
        from neurobrix.core.dtype.config import parse_dtype as _cfg_parse_dtype
        graph_dtype_str = self._dag.get("torch_dtype", "")
        self._graph_dtype = _cfg_parse_dtype(graph_dtype_str) if graph_dtype_str else None

        # DtypeEngine: single entry point for all dtype decisions.
        # AMP (Automatic Mixed Precision) is DISABLED for diffusion components
        # (transformer, VAE) in image/video families. These pipelines are designed
        # for pure fp16/bf16 — AMP's fp32↔fp16 oscillation at every layer creates
        # spatially-structured quantization noise that CFG amplifies into grid lines.
        # Text encoders KEEP AMP: T5/Gemma have residual accumulation that overflows fp16.
        # LLM family keeps AMP everywhere: RMSNorm pow→mean→rsqrt overflows without it.
        if self.mode not in ("triton", "triton_sequential"):
            from neurobrix.core.dtype.engine import DtypeEngine
            amp_enabled = self._should_enable_amp()
            self._dtype_engine = DtypeEngine(get_torch_dtype(self.dtype), graph_dtype=self._graph_dtype,
                                             amp_enabled=amp_enabled)
        else:
            self._dtype_engine = None  # Triton uses TritonDtypeEngine in sequence.py

        # DAG loaded silently - stats available via _dag

        # Load constants embedded in graph (buffers like pos_embed)
        self._load_constants_from_graph()

        # Fix SDPA double-scaling from PyTorch's decomposition + pattern reassembly
        self._normalize_sdpa_scaling()

        # MoE Fusion Pass: detect and fuse MoE expert subgraphs BEFORE any execution.
        # CRITICAL: This runs for ALL modes (compiled, native, triton).
        # Without fusion, MoE models use hardcoded trace-time routing indices → garbage output.
        from .graph.moe_fusion import detect_and_fuse_moe
        self._dag = detect_and_fuse_moe(
            self._dag, self.family,
            norm_topk_prob=getattr(self, '_moe_norm_topk_prob', True)
        )

        # Pre-compile dispatch table for Triton mode
        # Only needed for Triton mode — native mode bypasses classification entirely
        # _precompile_dispatch_table removed — old triton dispatch path eliminated

        # DtypeEngine handles mixed precision at the op level.
        # No DAG rewriting needed — graph annotations stay as-is (from trace).
        # Weights loaded at Prism dtype. DtypeEngine wraps each op at compile time:
        # - Stability ops (softmax, rsqrt, layer_norm): upcast to fp32, recast after
        # - Compute ops (mm, bmm, conv): ensure inputs in compute_dtype
        # - All others: passthrough

        # COMPILED MODE: Lazy compile on first execution.
        # Interceptors (KV cache) and persistent tensors (hidden states) are registered
        # after __init__. Compiling here wastes time — it gets recompiled on interceptor
        # registration. Instead, _execute_compiled_graph() handles: _compiled_seq is None → compile.

    def _should_enable_amp(self) -> bool:
        """Determine whether AMP should be enabled for this component.

        AMP is required for precision: RMSNorm's pow→mean→rsqrt chain,
        group_norm accumulation, and softmax all need fp32 compute.

        MPS: AMP stays ON (same rules as CUDA). Mixed-dtype safety at op
        boundaries is handled by FP16 ops (_make_lower_precision_wrapper)
        which cast inputs to compute_dtype. fp32 only flows through
        single-input ops (pow, mean, rsqrt) where no dtype mismatch occurs.
        """
        return True

    def _normalize_sdpa_scaling(self) -> None:
        """
        Fix SDPA double-scaling caused by PyTorch's SDPA math decomposition.

        PyTorch decomposes F.scaled_dot_product_attention(Q, K, V, scale=s) into:
          Q_scaled = Q * sqrt(s)    # Pre-scale Q for numerical stability
          K_scaled = K * sqrt(s)    # Pre-scale K
          scores = Q_scaled @ K_scaled^T  # Already scaled by s
          output = softmax(scores) @ V

        When the tracer's pattern reassembly reconstructs SDPA, it may keep the
        pre-scaling mul ops AND set scale=s in the SDPA attributes. This causes
        double-scaling: total = sqrt(s)^2 * s = s^2 instead of s.

        Detection: For each SDPA op with scale=s, trace Q and K inputs back through
        passthrough ops (expand, clone, _to_copy, transpose). If both arrive at mul
        ops with scalars whose product ≈ s, the pre-scaling is redundant.

        Fix: Set the pre-scaling mul scalars to 1.0 (neutralize them).
        """
        assert self._dag is not None
        dag_ops = self._dag.get("ops", {})

        # Build tensor_id → producer op_uid mapping
        tensor_to_producer: Dict[str, str] = {}
        for uid, op in dag_ops.items():
            for tid in op.get("output_tensor_ids", []):
                tensor_to_producer[tid] = uid

        # Ops that pass values through without changing them (for tracing)
        _PASSTHROUGH = frozenset({
            "aten::expand", "aten::clone", "aten::contiguous",
            "aten::_to_copy", "aten::view", "aten::_unsafe_view",
            "aten::unsqueeze", "aten::transpose",
        })

        def _trace_to_mul(tensor_id: str) -> Optional[str]:
            """Trace back through passthrough ops to find a producer mul op."""
            tid = tensor_id
            for _ in range(8):  # Max depth
                producer_uid = tensor_to_producer.get(tid)
                if producer_uid is None:
                    return None
                producer_op = dag_ops[producer_uid]
                if producer_op["op_type"] == "aten::mul":
                    return producer_uid
                if producer_op["op_type"] in _PASSTHROUGH:
                    in_tids = producer_op.get("input_tensor_ids", [])
                    if in_tids:
                        tid = in_tids[0]
                    else:
                        return None
                else:
                    return None
            return None

        def _get_mul_scalar(op: Dict[str, Any]) -> Optional[float]:
            """Extract the scalar value from a mul op's attributes."""
            for arg in op.get("attributes", {}).get("args", []):
                if arg.get("type") == "scalar" and isinstance(arg.get("value"), (int, float)):
                    return float(arg["value"])
            return None

        neutralized = 0
        for uid, op in dag_ops.items():
            if op["op_type"] != "aten::scaled_dot_product_attention":
                continue

            attrs = op.get("attributes", {})
            scale = attrs.get("kwargs", {}).get("scale") or attrs.get("scale")
            if not isinstance(scale, (int, float)) or scale <= 0:
                continue

            in_tids = op.get("input_tensor_ids", [])
            if len(in_tids) < 3:
                continue

            # Trace Q (input[0]) and K (input[1]) back to their producer mul ops
            q_mul_uid = _trace_to_mul(in_tids[0])
            k_mul_uid = _trace_to_mul(in_tids[1])
            if q_mul_uid is None or k_mul_uid is None:
                continue

            q_scalar = _get_mul_scalar(dag_ops[q_mul_uid])
            k_scalar = _get_mul_scalar(dag_ops[k_mul_uid])
            if q_scalar is None or k_scalar is None:
                continue

            # Check if pre-scaling product ≈ SDPA scale (double-scaling pattern)
            product = q_scalar * k_scalar
            if abs(product - scale) / max(abs(scale), 1e-10) < 0.01:
                # Neutralize pre-scaling by setting mul scalars to 1.0
                for arg in dag_ops[q_mul_uid]["attributes"]["args"]:
                    if arg.get("type") == "scalar":
                        arg["value"] = 1.0
                for arg in dag_ops[k_mul_uid]["attributes"]["args"]:
                    if arg.get("type") == "scalar":
                        arg["value"] = 1.0
                neutralized += 1

        if neutralized > 0:
            import logging
            logging.getLogger(__name__).info(
                f"[GraphExecutor] Normalized SDPA scaling: neutralized {neutralized * 2} "
                f"pre-scaling mul ops across {neutralized} attention layers"
            )

    def _compile_execution_sequence(self) -> None:
        """
        Pre-compile the DAG into a CompiledSequence.

        CompiledSequence eliminates ALL Python overhead in the execution loop by:
        1. Pre-resolving all tensor lookups to integer indices
        2. Pre-binding function references (no string dispatch)
        3. Using list-based arena with __slots__ instead of dict-based tensor_store
        4. Pre-generating resolver CLOSURES that eliminate isinstance() at runtime

        Performance gains vs legacy:
        - Legacy: isinstance() check for EVERY arg at runtime (~100ns each)
        - CompiledSequence: Pre-compiled closures, zero isinstance() in hot loop

        NOTE: CompiledSequence is 100% AUTONOMOUS - no dependency on NativeATenDispatcher.
        All op resolution is handled by CompiledOpResolver internally.
        """
        # MoE fusion already applied in _init_from_dag() (runs for ALL modes).
        # Create and compile the CompiledSequence (100% autonomous)
        self._compiled_seq = CompiledSequence(
            dag=self._dag,
            device=torch.device(self.device),
            dtype=get_torch_dtype(self.dtype),
            amp_enabled=self._dtype_engine.amp_enabled,
            use_triton=(self.mode == "triton"),
        )

        # Register any op interceptors BEFORE compilation (Phase 2.2: KV cache support)
        for op_type, interceptor in self._op_interceptors.items():
            self._compiled_seq.register_op_interceptor(op_type, interceptor)
        # Register fine-grained per-op_uid interceptors (op-level tiling)
        if hasattr(self, '_op_uid_interceptors'):
            for op_uid, interceptor in self._op_uid_interceptors.items():
                self._compiled_seq.register_op_uid_interceptor(op_uid, interceptor)

        self._compiled_seq.compile()
        self._interceptors_dirty = False

        # Apply persistent tensor protections registered before compilation (lazy compile)
        for tid in self._persistent_tensor_ids:
            self._compiled_seq.protect_tensor(tid)

    def load_graph_from_dict(self, dag: Dict[str, Any]) -> None:
        """
        Load TensorDAG from dict (already parsed).

        Supports symbolic shapes.
        Useful when DAG is already loaded by caller.
        """
        self._dag = dag
        self._init_from_dag()

    # =========================================================================
    # Weight Loading
    # =========================================================================

    def load_weights(
        self,
        nbx_path: str,
        component: str,
        shard_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Load weights from cache (extracted from NBX on first run).

        ARCHITECTURE (ZERO HARDCODE):
        - Weights stored as fp32 in cache (preserves precision)
        - self.dtype comes from Prism (embedded from hardware profile supports_dtypes)
        - Weights converted to self.dtype during load

        Uses shard_map from Prism for multi-GPU placement.
        Same lifecycle for both native and triton — only tensor FORMAT differs.
        """
        if self.mode in ("triton", "triton_sequential"):
            self._load_weights_triton(nbx_path, component, shard_map)
        else:
            self._load_weights_native(nbx_path, component, shard_map)

        # Weights changed — force rebind on next _execute_compiled_graph
        self._weights_bound = False

        # Reload graph-embedded constants (RoPE cos/sin, inv_freq, etc.)
        # These are NOT in safetensors — they exist only in graph.json as base64 data.
        # Must reload on every weight load since cleanup() clears _weights entirely.
        self._load_constants_from_graph()

        # Compute runtime-resolved buffers (sincos 2D pos_embed, interpolated
        # pos_embed). These are marked `is_computable=True` in graph.json —
        # the safetensors shards do not contain them. Mirrored into both
        # modes: previously only _load_weights_triton called this, which
        # left `aten.add::0` (pos_embed add) with an undefined input on the
        # native path for PixArt/Sana/any DiT with sincos pos_embed. The
        # legacy component-handler `prepare_weights` fallback is kept for
        # models that use learned positional embeddings scaled at load
        # time (no computable_spec in graph).
        if hasattr(self, "_computable_specs") and self._computable_specs:
            self._compute_computable_buffers()
        elif self._component_handler is not None:
            if self._runtime_height is not None and self._runtime_width is not None:
                self._weights = self._component_handler.prepare_weights(
                    self._weights,
                    self._runtime_height,
                    self._runtime_width,
                )

    def _load_weights_native(self, nbx_path, component, shard_map):
        """Load weights as torch.Tensor (native mode)."""
        from neurobrix.core.io import WeightLoader
        torch_dtype = get_torch_dtype(self.dtype)
        with WeightLoader(nbx_path) as loader:
            if shard_map:
                self._weights = loader.load_component_with_shard_map(
                    component, shard_map, torch_dtype)
            else:
                self._weights = loader.load_component(
                    component, self.device, torch_dtype)

    def _load_weights_triton(self, nbx_path, component, shard_map):
        """Load weights as NBXTensor (triton mode). Zero torch."""
        from neurobrix.triton.weight_loader import load_component_weights
        from neurobrix.kernels.nbx_tensor import parse_dtype, DeviceAllocator
        from neurobrix.kernels import wrappers as _w

        device_idx = int(self.device.split(':')[1]) if ':' in str(self.device) else 0
        compute_dtype = parse_dtype(self.dtype)

        DeviceAllocator.set_device(device_idx)
        DeviceAllocator.ensure_triton_device(device_idx)

        # Bind-time fp16→fp32 weight upcast is always OFF. The in-kernel
        # PROMOTE_B path in matmul_kernel / addmm_kernel (landed Apr 2026)
        # handles fp32 activation × fp16 weight on pre-Ampere without any
        # full-weight fp32 copy — the kernel casts each b tile inline,
        # fused with the load. Keeping weights fp16 in memory halves the
        # VRAM footprint (measured: TinyLlama peak 4.74 → 2.54 GB on V100)
        # and is also faster end-to-end (TinyLlama 9.94 s → 5.04 s),
        # because the bind-time upcast cost itself was never cheap.
        self._weights = load_component_weights(
            nbx_path, component, device_idx, compute_dtype,
            shard_map=shard_map)

        # TEMP DIAG: probe VRAM right after triton weight load commits the
        # component's arena. Gate env NBX_UNLOAD_DIAG=1.
        import os as _os_d
        if _os_d.environ.get("NBX_UNLOAD_DIAG") == "1":
            import ctypes as _ct
            try:
                try:
                    _rt = _ct.CDLL("libcudart.so")
                except OSError:
                    _rt = None
                    for _v in (12, 11, 10):
                        try:
                            _rt = _ct.CDLL(f"libcudart.so.{_v}")
                            break
                        except OSError:
                            continue
                if _rt is not None:
                    _free = _ct.c_size_t(0); _total = _ct.c_size_t(0)
                    _rt.cudaMemGetInfo(_ct.byref(_free), _ct.byref(_total))
                    _used_gb = (_total.value - _free.value) / 1e9
                    _free_gb = _free.value / 1e9
                    print(f"[VRAM_PROBE after_load_weights_triton[{component}]] "
                          f"used={_used_gb:.2f}GB free={_free_gb:.2f}GB "
                          f"cuda:{device_idx}", flush=True)
            except Exception as _e:
                print(f"[VRAM_PROBE after_load_weights_triton[{component}]] "
                      f"failed: {_e}", flush=True)

        # Computable buffers + handler prep now live in load_weights() so
        # both native and triton paths run them. See shared post-dispatch
        # block above.

    def _reconcile_weight_keys(self) -> None:
        """Reconcile weight dict keys with graph param names.

        Handles prefix hierarchy mismatches between traced model structure
        and HF safetensors structure. For example:
          Trace:  model.language_model.block.0.attn.key.weight
          Import: language_model.model.block.0.attn.key.weight

        Both are neurotaxed but the model wrapper hierarchy differs.
        This method remaps self._weights keys to match graph param:: IDs.
        Called once per component load — zero overhead during execution.

        Two-pass strategy:
        1. Unique suffix matching (handles most keys)
        2. Prefix transformation (handles ambiguous suffixes in multimodal models)
        """
        if not self._dag or not self._weights:
            return

        tensors = self._dag.get("tensors", {})
        graph_params = set()
        for tid in tensors:
            if tid.startswith("param::"):
                graph_params.add(tid[7:])
            elif tid.startswith("buffer::"):
                graph_params.add(tid[8:])

        weight_keys = set(self._weights.keys())

        # Fast path: all keys match directly — no reconciliation needed
        if weight_keys <= graph_params:
            return

        # Build suffix index from graph params (unique suffixes only)
        suffix_to_param: dict = {}
        suffix_ambiguous: set = set()
        for param in graph_params:
            parts = param.split('.')
            for i in range(len(parts)):
                suffix = '.'.join(parts[i:])
                if suffix in suffix_ambiguous:
                    continue
                if suffix in suffix_to_param:
                    del suffix_to_param[suffix]
                    suffix_ambiguous.add(suffix)
                else:
                    suffix_to_param[suffix] = param

        # Pass 1: Match by unique suffix
        remapped: dict = {}
        unmatched_keys: list = []
        reconciled = 0
        for wk in weight_keys:
            if wk in graph_params:
                remapped[wk] = self._weights[wk]
                continue

            # Strip LoRA wrapper tokens before suffix matching
            # PEFT wraps layers: base_model.model.X.base_layer.weight → X.weight
            wk_clean = wk.replace('.base_layer.', '.').replace('base_model.model.', '')
            if wk_clean != wk and wk_clean in graph_params:
                remapped[wk_clean] = self._weights[wk]
                reconciled += 1
                continue

            parts = wk_clean.split('.')
            matched_param = None
            for i in range(len(parts)):
                suffix = '.'.join(parts[i:])
                if suffix in suffix_to_param:
                    matched_param = suffix_to_param[suffix]
                    break

            if matched_param:
                remapped[matched_param] = self._weights[wk]
                reconciled += 1
            else:
                unmatched_keys.append(wk)

        # Pass 2: Prefix transformations for ambiguous suffixes
        # Handles multimodal models where sub-models share block structure
        if unmatched_keys:
            remaining_params = graph_params - set(remapped.keys())
            for wk in unmatched_keys:
                parts = wk.split('.')
                candidates = []
                if len(parts) >= 2:
                    # Try A.B.rest → B.A.rest (prefix swap)
                    swapped = f"{parts[1]}.{parts[0]}.{'.'.join(parts[2:])}"
                    if swapped in remaining_params:
                        candidates.append(swapped)
                    # Try A.B.rest → model.A.rest (prepend model, skip B)
                    prepended = f"model.{parts[0]}.{'.'.join(parts[2:])}"
                    if prepended in remaining_params:
                        candidates.append(prepended)
                if len(parts) >= 1:
                    # Try A.rest → model.A.rest (prepend model)
                    prepended_simple = f"model.{wk}"
                    if prepended_simple in remaining_params:
                        candidates.append(prepended_simple)
                    # Try A.rest → rest (strip first segment)
                    stripped = '.'.join(parts[1:])
                    if stripped in remaining_params:
                        candidates.append(stripped)

                if len(candidates) == 1:
                    remapped[candidates[0]] = self._weights[wk]
                    remaining_params.discard(candidates[0])
                    reconciled += 1
                elif len(candidates) > 1:
                    # Multiple candidates — use the one with most segment overlap
                    best = max(candidates, key=lambda c: sum(
                        1 for s in c.split('.') if s in parts
                    ))
                    remapped[best] = self._weights[wk]
                    remaining_params.discard(best)
                    reconciled += 1
                else:
                    remapped[wk] = self._weights[wk]

        if reconciled > 0:
            self._weights = remapped

    def _load_constants_from_graph(self) -> None:
        """
        Load constant tensors embedded in graph.json.

        Constants are identified by: constant=True, constant_data=<base64>
        These are non-trainable tensors (buffers) like positional embeddings.

        COMPUTABLE BUFFERS:
        If a tensor has is_computable=True, it needs runtime computation instead
        of loading the traced constant. This is used for sincos pos_embed that
        needs to be recomputed at runtime resolution (not the trace resolution).

        ZERO HARDCODE: dtype from graph tensor metadata (SOURCE OF TRUTH).
        ZERO FALLBACK: Error if constant marked but no data (unless computable).
        """
        import base64
        import io

        assert self._dag is not None
        tensors = self._dag.get("tensors", {})
        const_count = 0
        computable_count = 0

        for tid, tdata in tensors.items():
            # Check for computable buffers FIRST
            if tdata.get("is_computable"):
                weight_name = tdata.get("weight_name")
                computation_spec = tdata.get("computation_spec")

                if weight_name and computation_spec:
                    # Store the computation spec for later use in load_weights
                    # Actual computation happens in _compute_buffer_at_runtime
                    if not hasattr(self, "_computable_specs"):
                        self._computable_specs = {}
                    self._computable_specs[weight_name] = computation_spec
                    computable_count += 1
                continue  # Skip loading constant_data for computable buffers

            if not tdata.get("constant"):
                continue

            b64_data = tdata.get("constant_data")
            if not b64_data:
                raise RuntimeError(
                    f"ZERO FALLBACK: Constant tensor '{tid}' has no constant_data.\n"
                    f"weight_name: {tdata.get('weight_name')}\n"
                    f"Graph may be corrupted or was traced with old format."
                )

            weight_name = tdata.get("weight_name")
            if not weight_name:
                continue

            if self.mode in ("triton", "triton_sequential"):
                self._load_constant_triton(b64_data, weight_name, tdata)
            else:
                self._load_constant_native(b64_data, weight_name)
            const_count += 1

    def _load_constant_native(self, b64_data: str, weight_name: str):
        """Load base64 constant via torch (native mode)."""
        import base64, io
        buffer = io.BytesIO(base64.b64decode(b64_data))
        tensor = torch.load(buffer, map_location='cpu', weights_only=True)
        tensor = tensor.to(self.device)
        if tensor.is_floating_point() and tensor.dtype != get_torch_dtype(self.dtype):
            tensor = self._dtype_engine.convert_constant(tensor)
        self._weights[weight_name] = tensor

    def _load_constant_triton(self, b64_data: str, weight_name: str,
                               tdata: dict):
        """Load base64 constant via ZIP parse + numpy (triton mode). Zero torch."""
        import base64, io, zipfile
        import numpy as np
        from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype

        shape = tuple(tdata.get("shape", []))
        dtype_str = tdata.get("dtype", "float32").replace("torch.", "")
        device_idx = int(self.device.split(':')[1]) if ':' in str(self.device) else 0
        compute_dtype = parse_dtype(self.dtype)

        # Extract raw bytes from torch.save ZIP archive
        raw_zip = base64.b64decode(b64_data)
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            data_files = [n for n in zf.namelist()
                          if n.startswith("archive/data/")]
            if not data_files:
                return
            tensor_bytes = zf.read(data_files[0])

        # Shape vs bytes reconciliation. Some graphs declare the FULL shape
        # of a seq-dependent buffer (e.g. DeepSeek RoPE cos/sin_cached:
        # declared (max_position, head_dim) but bytes stored at (trace_seq_len,
        # head_dim)). The native path gets the trace shape for free via
        # torch.load's pickled metadata; we have to derive it from bytes.
        # Find the dim to shrink so that the shape matches byte count.
        dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4,
                       "float64": 8, "int32": 4, "int64": 8,
                       "int8": 1, "uint8": 1, "bool": 1}.get(dtype_str, 4)
        declared_elems = 1
        for d in shape:
            declared_elems *= d
        actual_elems = len(tensor_bytes) // dtype_bytes
        if shape and declared_elems != actual_elems and actual_elems > 0:
            # Look up trace_seq_len from symbolic_context; if any dim, replaced
            # by trace_seq_len, makes the element count match, use that.
            dag = self._dag
            if dag is None:
                raise RuntimeError(
                    f"Constant '{weight_name}': declared shape {shape} "
                    f"({declared_elems} elems) but bytes hold {actual_elems} "
                    f"elems and no symbolic_context to reconcile.")
            sym_ctx = dag.get("symbolic_context", {})
            trace_seq_len = None
            for sid, sinfo in sym_ctx.get("symbols", {}).items():
                if sinfo.get("name") == "seq_len":
                    trace_seq_len = sinfo.get("trace_value")
                    break
            fixed = None
            # Strategy A: a single dim swapped with trace_seq_len reconciles
            # (DeepSeek per-block RoPE cache pattern).
            if trace_seq_len:
                for axis in range(len(shape)):
                    other = declared_elems // shape[axis] if shape[axis] else 0
                    if other and other * trace_seq_len == actual_elems:
                        fixed = list(shape)
                        fixed[axis] = trace_seq_len
                        break
            # Strategy B: 1-D constant where the bytes simply hold a few more
            # (or fewer) elements than the declared shape — happens on TTS
            # models like Kokoro whose graphs do not carry a seq_len symbol
            # but where the tracer wrote a slightly off-by-one length on a
            # 1-D table. Trust the bytes (mirrors what torch.load does on the
            # native path; it ignores the JSON shape and uses the pickled
            # tensor's metadata).
            if fixed is None and len(shape) == 1:
                fixed = [actual_elems]
            if fixed is None:
                raise RuntimeError(
                    f"Constant '{weight_name}' declared shape {shape} "
                    f"({declared_elems} elems) but bytes hold {actual_elems} "
                    f"elems; no single dim swap with trace_seq_len="
                    f"{trace_seq_len} reconciles.")
            shape = tuple(fixed)

        # Parse raw bytes based on dtype
        if dtype_str == "bfloat16":
            raw_u16 = np.frombuffer(tensor_bytes, dtype=np.uint16).reshape(shape)
            if compute_dtype == NBXDtype.float16:
                fp32 = np.zeros(raw_u16.shape, dtype=np.float32)
                fp32.view(np.uint32).flat[:] = raw_u16.flat[:].astype(np.uint32) << 16
                arr = np.ascontiguousarray(fp32.astype(np.float16))
            else:
                arr = np.ascontiguousarray(raw_u16)
        else:
            np_dt = {"float16": np.float16, "float32": np.float32,
                     "float64": np.float64, "int32": np.int32,
                     "int64": np.int64, "int8": np.int8,
                     "uint8": np.uint8, "bool": np.bool_,
                     }.get(dtype_str, np.float32)
            arr = np.frombuffer(tensor_bytes, dtype=np_dt).reshape(shape)
            arr = np.ascontiguousarray(arr)

        DeviceAllocator.set_device(device_idx)
        self._weights[weight_name] = NBXTensor.from_numpy(arr)

    # =========================================================================
    # Execution: run() and helpers
    # =========================================================================

    def run(
        self,
        inputs: Dict[str, Any],
        pre_op_callback: Optional[Callable[[int, Any], None]] = None,
        skip_kills: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute DAG mechanically.

        For triton mode with NBXTensor inputs, bypasses torch.inference_mode
        and handles inputs directly. For native/compiled mode, uses torch path.

        Args:
            inputs: Named input tensors (model inputs). Can be torch.Tensor or NBXTensor.
            pre_op_callback: Optional (op_idx, op) -> None hook forwarded
                to CompiledSequence.run() for the compiled path. Used by
                zero3 block-wise pipelining to swap block weights between
                CPU and GPU at block boundaries. Only the compiled+
                multi-device path honors it; triton and native paths
                ignore it (zero3 is PyTorch-only today).
            skip_kills: Optional explicit control of triton kill_slots
                processing. None (default) → kill_slots fire (safe,
                required for diffusion / prefill / any graph with distinct
                per-op output slots). True → kill_slots skipped (only
                legitimate for autoregressive decode steps that reuse
                the same arena slots every step — setting True anywhere
                else retains all intermediate tensors in arena and
                blows up VRAM). Caller flow handlers pass True
                explicitly; the triton path itself never infers it from
                input shapes. See PR "triton: explicit skip_kills flag"
                for the history — the prior shape-based heuristic
                mis-fired on diffusion `aspect_ratio` (B, 1) inputs,
                which are shape-indistinguishable from LLM decode
                `input_ids` (B, 1), so only the flow-handler-provided
                value is reliable.

        Returns:
            Output tensors
        """
        # Explicit per-call callback wins over the persistent one
        # (installed by zero3). _execute_compiled_graph consumes via
        # self._pre_op_callback — reset in finally so the instance
        # state is clean between calls.
        self._pre_op_callback = pre_op_callback or self._persistent_pre_op_callback

        try:
            # TRITON MODES: direct NBXTensor path — no torch wrapping
            if self.mode in ("triton", "triton_sequential"):
                return self._run_triton(inputs, skip_kills=skip_kills)

            # NATIVE/COMPILED MODE: torch path
            with torch.inference_mode():
                self._ctx = self._prepare_execution(inputs)
                self._resolver = TensorResolver(self._ctx)
                stats = self._execute_all_ops()
                outputs = self._gather_outputs()
                self._last_stats = stats
                return outputs
        finally:
            self._pre_op_callback = None
            # Post-run hook runs unconditionally (even on exception) so
            # zero3 can evict the current block back to CPU and leave
            # the arena in a clean state for the next call.
            if self._post_run_hook is not None:
                self._post_run_hook()

    def _run_triton(self, inputs: Dict[str, Any],
                    skip_kills: Optional[bool] = None) -> Dict[str, Any]:
        """Execute via triton path with NBXTensor inputs. Zero torch.

        Supports both compiled (--triton) and sequential (--triton-sequential).
        Compiled: TritonSequence (arena + closures, fast).
        Sequential: TritonSequentialDispatcher (dict lookup per op, debuggable).

        Args:
            inputs: Named input tensors.
            skip_kills: Forwarded to the underlying compiled-sequence
                hot loop. None → False (default safe — kill_slots fire).
                See `run()` for why this is an explicit flag.
        """
        import time as _time
        from neurobrix.kernels.nbx_tensor import NBXTensor, parse_dtype, DeviceAllocator

        start = _time.perf_counter()
        if isinstance(self.device, str):
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
        elif hasattr(self.device, 'index') and isinstance(self.device.index, int):
            device_idx = self.device.index
        else:
            device_idx = 0

        DeviceAllocator.set_device(device_idx)
        DeviceAllocator.ensure_triton_device(device_idx)

        # Build input map: name → NBXTensor (keyed by tensor ID).
        # InputSynthesizer.synthesize_missing_inputs writes dotted inputs
        # (e.g. "added_cond_kwargs.resolution") as nested dicts via
        # `_set_nested` (resolution/input_synthesizer.py:130-137). Native
        # reads them back via tensor_resolver.py:82-101 with a two-strategy
        # cascade (flat lookup, then dotted-path navigation). The triton
        # path was missing the dotted-path branch: the flat `.get()` alone
        # returned None → input slot unset → downstream op received
        # undefined input → NOP propagation cascaded through the micro-
        # conditioning chain and crashed (PixArt-Alpha `aten.cat::2` None,
        # PixArt-Sigma later `aten.clone::28` error 700 in a subsequent
        # op). Universal for any model with synthesis rules (PixArt,
        # Sana, Flex, SANA-Video, future diffusion).
        input_map = {}
        tensors_meta = self._dag.get("tensors", {})
        for tid in tensors_meta:
            if not tid.startswith("input::"):
                continue
            input_name = tid[7:]
            value = inputs.get(input_name)
            if value is None and "." in input_name:
                cur = inputs
                for part in input_name.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        cur = None
                        break
                value = cur
            if value is None:
                continue
            if isinstance(value, NBXTensor):
                input_map[tid] = value
            elif hasattr(value, 'data_ptr'):
                # Cross-device safety. InputSynthesizer falls back to
                # `plan.primary_device_index` (default 0) when it can't
                # read the device off an existing tensor — on a multi-GPU
                # profile where the executor runs on cuda:N ≠ 0, the
                # synthesised tensor lands on cuda:0. Triton rejects
                # cross-device pointers with the exact same error message
                # as genuine CPU pointers ("Pointer argument ... cannot
                # be accessed from Triton (cpu tensor?)"). Native's
                # `_execute_compiled_graph` entry does `.to(self.device)`
                # for every torch input — triton must do the same here.
                src_idx = getattr(value.device, 'index', None)
                if (isinstance(src_idx, int) and src_idx != device_idx) \
                        or str(value.device) == "cpu":
                    value = value.to(f"cuda:{device_idx}")
                didx = value.device.index if isinstance(getattr(value.device, 'index', None), int) else device_idx
                input_map[tid] = NBXTensor.from_raw(
                    value.data_ptr(), tuple(value.shape),
                    parse_dtype(str(value.dtype)), 'cuda',
                    owns_data=False, device_idx=didx, base=value)

        if self.mode == "triton_sequential":
            raw_outputs, num_ops = self._run_triton_sequential(
                input_map, device_idx, skip_kills=skip_kills)
        else:
            raw_outputs, num_ops = self._run_triton_compiled(
                input_map, device_idx, skip_kills=skip_kills)

        # Map tensor IDs to semantic output names
        outputs = {}
        for tid, tensor in raw_outputs.items():
            info = tensors_meta.get(tid, {})
            name = info.get("output_name", tid)
            outputs[name] = tensor

        elapsed = (_time.perf_counter() - start) * 1000
        self._last_stats = ExecutionStats(
            total_ops=num_ops, triton_ops=num_ops, total_time_ms=elapsed)
        return outputs

    def _run_triton_compiled(self, input_map: dict, device_idx: int,
                             skip_kills: Optional[bool] = None):
        """Execute via TritonSequence (arena + closures).

        `skip_kills` is an explicit flag threaded from the caller's
        flow handler. Default False (safe). The prior shape-based
        heuristic (`any(t.shape[1] == 1 for t in inputs)`) was removed
        because it cannot distinguish an LLM decode step
        (`input_ids` shape `(B, 1)`, 2D, seq_len=1) from a diffusion
        model's micro-conditioning input (`aspect_ratio` shape
        `(B, 1)`, 2D, a per-item scalar), so it fired `skip_kills=True`
        on PixArt batched-CFG runs and left every intermediate alive
        in the arena for the full 2495-op transformer forward pass →
        ~31 GB peak on V100 32 GB and OOM. Autoregressive flow handlers
        that genuinely benefit from skip_kills pass it explicitly
        (see `triton/flow/autoregressive.py`), diffusion flows leave
        it at the default.
        """
        self._ensure_triton_compiled(device_idx)
        self._triton_seq.bind_inputs(input_map)
        self._triton_seq.bind_symbols(input_map)
        self._triton_seq.update_seq_dependent_constants()

        effective_skip_kills = bool(skip_kills) if skip_kills is not None else False

        # Thread pre_op_callback from the executor into TritonSequence.
        # Explicit per-call wins over the persistent hook (installed by
        # strategies like zero3). This mirrors the behaviour of
        # _execute_compiled_graph in native mode.
        cb = self._pre_op_callback or self._persistent_pre_op_callback
        self._triton_seq.run(skip_kills=effective_skip_kills, pre_op_callback=cb)
        return self._triton_seq.gather_outputs(), self._triton_seq.num_ops

    def _run_triton_sequential(self, input_map: dict, device_idx: int,
                               skip_kills: Optional[bool] = None):
        """Execute via TritonSequentialDispatcher (op-by-op, debuggable).

        `skip_kills` is accepted for signature parity with
        `_run_triton_compiled`; the sequential dispatcher does not use
        arena slots and therefore has no kill_slots concept — the flag
        is effectively ignored here.
        """
        del skip_kills  # unused, kept for signature parity
        from neurobrix.triton.sequential import TritonSequentialDispatcher
        from neurobrix.triton.symbols import SymbolResolver
        from neurobrix.kernels.nbx_tensor import NBXTensor, parse_dtype

        dispatcher = TritonSequentialDispatcher(
            device_idx=device_idx, compute_dtype=parse_dtype(self.dtype))

        tensors = self._dag.get("tensors", {})
        ops_meta = self._dag.get("ops", {})
        exec_order = self._dag.get("execution_order", [])

        # Symbol resolver for symbolic shapes
        sym_ctx = self._dag.get("symbolic_context", {})
        sym_resolver = SymbolResolver(sym_ctx) if sym_ctx else None
        if sym_resolver:
            sym_resolver.bind_from_inputs(input_map,
                                          self._dag.get("input_tensor_ids", []),
                                          tensors)

        # Apply symbolic promotion (shared with compiled mode) so that
        # ops like ones([23,23]) become ones([s1,s1]) → ones([actual_seq_len, actual_seq_len])
        if not hasattr(self, '_seq_promotion_done'):
            from neurobrix.triton.promotion import promote_seq_len_scalars
            promote_seq_len_scalars(self._dag, tensors, ops_meta)
            self._seq_promotion_done = True

        # Tensor store: maps tensor_id → NBXTensor
        store: Dict[str, Any] = {}

        # Weights already loaded as NBXTensor by load_weights() via lifecycle
        for tid, tdata in tensors.items():
            if tid.startswith("param::") or tid.startswith("buffer::"):
                wname = tdata.get("weight_name", "")
                if wname in self._weights:
                    store[tid] = self._weights[wname]

        # Load inputs into store
        for tid, tensor in input_map.items():
            store[tid] = tensor

        # Liveness analysis: mirror TritonSequence._compute_liveness
        # (triton/sequence.py:1394-1421) so the sequential dispatcher
        # reclaims intermediate VRAM after an op's last consumer.
        # Without this, `store` grows monotonically across the whole
        # component forward pass — fine for small LLMs (TinyLlama 4-D
        # intermediates, ≤ 5 GB), fatal for diffusion VAEs (Sana
        # DC-AE 32× upsamples to (1, C, 1024, 1024) tensors that
        # cumulate into > 32 GB if never freed). `dead_at_op[op_idx]`
        # lists the tids whose last use is this op; after dispatch we
        # `store.pop` them, which drops their Python refcount. Unlike
        # the compiled path, the sequential dispatcher has no in-flight
        # kernel overlap — every op call is synchronous from Python's
        # point of view — so `NBXTensor.__del__` → `free_cuda` at
        # refcount-0 is safe without an explicit sync_device().
        protected: set = set()
        for tid in tensors:
            if tid.startswith("param::") or tid.startswith("buffer::") \
                    or tid.startswith("input::"):
                protected.add(tid)
        for tid in self._dag.get("output_tensor_ids", []):
            protected.add(tid)

        def _collect_arg_tids(arg, out_set):
            if not isinstance(arg, dict):
                return
            atype = arg.get("type")
            if atype in ("tensor", "tensor_ref"):
                t = arg.get("tensor_id")
                if t:
                    out_set.add(t)
            elif atype == "tensor_tuple":
                for t in arg.get("tensor_ids", []):
                    out_set.add(t)
            elif atype == "list":
                for item in arg.get("value", []):
                    _collect_arg_tids(item, out_set)

        last_use: Dict[str, int] = {}
        for op_idx, op_uid in enumerate(exec_order):
            op_data = ops_meta.get(op_uid)
            if op_data is None:
                continue
            seen: set = set()
            attrs = op_data.get("attributes", {})
            for arg in attrs.get("args", []):
                _collect_arg_tids(arg, seen)
            for arg in attrs.get("kwargs", {}).values():
                _collect_arg_tids(arg, seen)
            for t in seen:
                last_use[t] = op_idx
        dead_at_op: Dict[int, list] = {}
        for tid, li in last_use.items():
            if tid in protected:
                continue
            dead_at_op.setdefault(li, []).append(tid)

        # Execute ops sequentially. Thread pre_op_callback here for
        # the same reason as _run_triton_compiled — zero3 (and any other
        # strategy that installs a persistent hook) needs to see every op
        # before dispatch so it can promote the block weights the op
        # needs to GPU. Without this the sequential path never fires the
        # hook and zero3-weighted ops run against CPU pointers.
        cb = self._pre_op_callback or self._persistent_pre_op_callback
        num_ops = 0
        for op_idx, op_uid in enumerate(exec_order):
            op_data = ops_meta.get(op_uid)
            if op_data is None:
                continue

            if cb is not None:
                try:
                    cb(op_idx, op_data)
                except Exception:
                    # Callback must never break dispatch — match the
                    # TritonSequence.run() contract where cb failures are
                    # isolated from the compute loop.
                    pass

            op_type = op_data.get("op_type", "")
            attrs = op_data.get("attributes", {})
            output_tids = op_data.get("output_tensor_ids", [])

            # MoE fused op — reads weights from store by tensor_id
            if op_type == "custom::moe_fused":
                result = self._execute_moe_fused_triton_sequential(attrs, store)
                if isinstance(result, tuple):
                    for i, tid in enumerate(output_tids):
                        store[tid] = result[i] if i < len(result) else None
                elif output_tids:
                    store[output_tids[0]] = result
                for dead_tid in dead_at_op.get(op_idx, ()):
                    store.pop(dead_tid, None)
                num_ops += 1
                continue

            # Resolve args from store
            raw_args = attrs.get("args", [])
            resolved_args = []
            for arg in raw_args:
                resolved_args.append(
                    self._resolve_sequential_arg(arg, store, sym_resolver, dispatcher))

            # Op_uid interceptor priority (matches CompiledSequence and
            # TritonSequence: op_uid > op_type > native dispatch). Without
            # this branch, triton_sequential silently bypasses Prism's
            # `register_op_uid_interceptors` plumbing — Sana 4Kpx VAE
            # conv::54 would arrive at the conv2d_wrapper with the raw
            # 36 GiB request shape (only the wrapper-internal 4 GiB
            # band-streaming would protect), retaining ~29 GB live and
            # OOMing on a 32 GB V100. Interceptors set
            # `self_manages_dtype = True` (see fused_upsample_conv.py)
            # so no dtype-engine AMP wrap needed here. R30 (Mode
            # Universality) parity, P-SANA-4KPX-RUNTIME 2026-05-05
            # bisection.
            if hasattr(self, '_op_uid_interceptors') and op_uid in self._op_uid_interceptors:
                resolved_kwargs = dispatcher.resolve_kwargs(attrs)
                result = self._op_uid_interceptors[op_uid](
                    *resolved_args, **resolved_kwargs)
            else:
                # Dispatch
                result = dispatcher.dispatch(op_type, resolved_args, attrs)

            # Store outputs
            if isinstance(result, tuple):
                for i, tid in enumerate(output_tids):
                    store[tid] = result[i] if i < len(result) else None
            else:
                if output_tids:
                    store[output_tids[0]] = result

            # Evict tids whose last-use was this op. See the liveness
            # comment before the loop for the motivation.
            for dead_tid in dead_at_op.get(op_idx, ()):
                store.pop(dead_tid, None)

            num_ops += 1

        # Gather outputs
        output_tids = self._dag.get("output_tensor_ids", [])
        outputs = {}
        for tid in output_tids:
            if tid in store and store[tid] is not None:
                outputs[tid] = store[tid]

        return outputs, num_ops

    def _resolve_sequential_arg(self, arg, store, sym_resolver, dispatcher):
        """Resolve a single arg for sequential mode."""
        if isinstance(arg, dict):
            atype = arg.get("type")
            if atype in ("tensor", "tensor_ref"):
                tid = arg.get("tensor_id")
                return store.get(tid)
            if atype == "tensor_tuple":
                return [store.get(tid) for tid in arg.get("tensor_ids", [])]
            if atype == "symbol":
                sid = arg.get("symbol_id") or arg.get("id")
                if sym_resolver and sid:
                    v = sym_resolver.get(sid)
                    if v > 0:
                        return v + arg.get("offset", 0)
                return arg.get("trace_value", arg.get("trace", 0))
            if atype in ("mul", "add", "sub", "floordiv", "mod", "neg", "product"):
                if sym_resolver:
                    try:
                        return sym_resolver.resolve(arg)
                    except Exception:
                        pass
                return arg.get("trace", arg.get("trace_value", 0))
            if atype == "list":
                items = arg.get("value", [])
                return [self._resolve_sequential_arg(item, store, sym_resolver, dispatcher)
                        for item in items]
            return dispatcher.resolve_attr(arg)
        if isinstance(arg, (list, tuple)):
            return [self._resolve_sequential_arg(item, store, sym_resolver, dispatcher)
                    for item in arg]
        return arg

    def _execute_moe_fused_triton_sequential(self, attrs: dict, store: dict):
        """Execute MoE fused op in triton sequential mode — delegates to triton/moe.py."""
        from neurobrix.triton.moe import execute_moe_fused

        gate_weight_ids = attrs["expert_gate_weight_ids"]
        up_weight_ids = attrs["expert_up_weight_ids"]
        down_weight_ids = attrs["expert_down_weight_ids"]

        cache_key = f"triton_seq_{attrs['gate_scores_tid']}"
        return execute_moe_fused(
            gate_scores=store.get(attrs["gate_scores_tid"]),
            hidden_states=store.get(attrs["hidden_states_tid"]),
            gate_weights=[store.get(wid) for wid in gate_weight_ids],
            up_weights=[store.get(wid) for wid in up_weight_ids],
            down_weights=[store.get(wid) for wid in down_weight_ids],
            top_k=attrs["top_k"],
            num_experts=attrs["num_experts"],
            norm_topk_prob=attrs.get("norm_topk_prob", True),
            cache_key=cache_key,
        )

    def register_triton_interceptors(self, interceptors: Dict[str, Any]):
        """Store interceptors to apply when triton sequence is compiled."""
        if not hasattr(self, '_pending_triton_interceptors'):
            self._pending_triton_interceptors = {}
        self._pending_triton_interceptors.update(interceptors)
        # Apply immediately if already compiled
        if hasattr(self, '_triton_seq') and self._triton_seq is not None:
            for op_type, func in interceptors.items():
                self._triton_seq.register_op_interceptor(op_type, func)

    def _ensure_triton_compiled(self, device_idx: int):
        """Compile triton sequence and bind weights (once).

        Weights are already in self._weights as NBXTensor (loaded by
        load_weights → _load_weights_triton). We just bind them.
        """
        if hasattr(self, '_triton_seq') and self._triton_seq is not None:
            return

        from neurobrix.triton.sequence import TritonSequence
        from neurobrix.kernels.nbx_tensor import parse_dtype, DeviceAllocator

        DeviceAllocator.set_device(device_idx)
        DeviceAllocator.ensure_triton_device(device_idx)

        # Persistent tids (e.g., hidden_states for image-AR LLMs where the
        # graph output is text logits) must be added to the dag's declared
        # outputs BEFORE compile(). TritonSequence's liveness analysis uses
        # `dag.output_tensor_ids` to protect slots from being freed; without
        # this, the hidden_tid's slot is reclaimed after its last use and
        # `gather_outputs([hidden_tid])` returns nothing. Native's
        # CompiledSequence uses a separate _persistent_tensor_ids set +
        # protect_tensor() hook; triton gets there via the dag outputs path.
        dag_for_triton = self._dag
        if self._persistent_tensor_ids:
            dag_for_triton = dict(self._dag)
            existing = list(self._dag.get("output_tensor_ids", []))
            extras = [t for t in self._persistent_tensor_ids if t not in existing]
            if extras:
                dag_for_triton["output_tensor_ids"] = existing + extras

        self._triton_seq = TritonSequence(
            dag_for_triton, device_idx=device_idx,
            compute_dtype=parse_dtype(self.dtype))

        # Phase 1 — read per-component opt-in flag from
        # forge/config/model_registry.yml (no Forge re-build required;
        # runtime-direct read keeps the .nbx contract immutable per R18).
        # Defaults to False (conservative). Annotated only for models whose
        # activation ranges have been validated fp16-safe via
        # neurobrix.tools.measure_activation_ranges.
        from neurobrix.core.runtime.registry_flags import get_component_flag
        _model_name = None
        try:
            _cache_path = getattr(self._pkg, 'cache_path', None) if hasattr(self, '_pkg') else None
            if _cache_path is not None:
                _model_name = Path(_cache_path).name
        except Exception:
            pass
        _fp16_safe = get_component_flag(
            _model_name, self._component_name,
            "activations_fp16_safe", default=False,
            env_override="NBX_ACTIVATIONS_FP16_SAFE",
        )
        self._triton_seq.set_activations_fp16_safe(bool(_fp16_safe))

        self._triton_seq.compile()

        # Weights already loaded as NBXTensor by load_weights()
        self._triton_seq.bind_weights(self._weights)

        # Apply pending interceptors (registered before compilation)
        if hasattr(self, '_pending_triton_interceptors'):
            for op_type, func in self._pending_triton_interceptors.items():
                self._triton_seq.register_op_interceptor(op_type, func)
        # Apply pending per-op_uid interceptors (op-level tiling, R30 mirror)
        if hasattr(self, '_pending_triton_uid_interceptors'):
            for op_uid, func in self._pending_triton_uid_interceptors.items():
                self._triton_seq.register_op_uid_interceptor(op_uid, func)

    def _prepare_execution(self, inputs: Dict[str, Any]) -> ExecutionContext:
        """
        Prepare execution context for a run.

        Sets up:
        - Clear runtime state
        - Store inputs with dtype conversion
        - Bind symbolic shapes

        Args:
            inputs: Named input tensors

        Returns:
            ExecutionContext ready for execution
        """
        # Create context with immutable dag and settings
        assert self._dag is not None
        assert isinstance(self._component_name, str)
        ctx = ExecutionContext(
            dag=self._dag,
            device=self.device,
            dtype=get_torch_dtype(self.dtype),
            weights=self._weights,
            shape_resolver=self._shape_resolver,
            symbolic_shapes_enabled=self._symbolic_shapes_enabled,
            component_name=self._component_name,
        )

        # Store inputs by input_name (semantic)
        tensor_inputs = {}  # For symbolic shape binding

        # SPRINT 0 - R0.2: Single TensorResolver instance for dtype checking
        # Avoid creating new TensorResolver for each input (was O(N) allocations)
        dtype_resolver = TensorResolver(ctx)

        for input_name, val in inputs.items():
            if isinstance(val, torch.Tensor):
                # Convert to expected dtype from graph
                expected_dtype = dtype_resolver.get_input_dtype(input_name)
                if expected_dtype is not None and val.dtype != expected_dtype:
                    val = val.to(expected_dtype)
                ctx.inputs[input_name] = val.to(self.device)
            elif isinstance(val, dict):
                # Handle dictionary inputs (e.g. added_cond_kwargs)
                ctx.inputs[input_name] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in val.items()
                }
            else:
                ctx.inputs[input_name] = val

            # Collect tensors for symbolic shape binding
            if isinstance(val, torch.Tensor):
                tensor_inputs[input_name] = val
            elif isinstance(val, dict):
                tensor_inputs[input_name] = val

        # Bind symbolic shapes from actual inputs
        if self._symbolic_shapes_enabled and self._shape_resolver:
            assert self._dag is not None
            self._shape_resolver.bind_from_inputs(
                tensor_inputs,
                self._dag.get("tensors", {})
            )
            bound = self._shape_resolver.get_bound_symbols()

            # Register combined seq_len symbols (sums of pairs).
            # Compiled ops may reference _sum_sA_sB synthetic symbols
            # created by _promote_seq_len_scalars_to_symbolic.
            symbols = self._dag.get("symbolic_context", {}).get("symbols", {})
            seq_syms = []
            for sid, info in symbols.items():
                tv = info.get("trace_value")
                if info.get("name") == "seq_len" and tv is not None:
                    seq_syms.append((sid, int(tv)))
            for i, (sid_a, tv_a) in enumerate(seq_syms):
                for sid_b, tv_b in seq_syms[i+1:]:
                    if tv_a != tv_b:
                        sum_id = f"_sum_{sid_a}_{sid_b}"
                        val_a = bound.get(sid_a, tv_a)
                        val_b = bound.get(sid_b, tv_b)
                        self._shape_resolver._runtime_values[sum_id] = val_a + val_b

            if bound:
                self._last_symbols = bound  # Store for CFG batch inference

            # Sequential mode: adapt seq-dependent constant weights.
            # constant_T_* tensors (RoPE cos/sin) have trace_seq_len in their shape.
            # Slice to match actual runtime seq_len to prevent index out of bounds.
            # Compiled mode does this in update_seq_dependent_constants().
            if self.mode in ("sequential", "triton_sequential"):
                # Sequential modes: patch ops that reference trace_seq_len in shape args
                self._patch_seq_len_in_ops(bound)

        return ctx

    def _patch_seq_len_in_ops(self, bound_symbols: Dict[str, int]) -> None:
        """Replace trace-time seq_len values with runtime values in op attributes.

        The compiled path does this via _promote_seq_len_scalars_to_symbolic + ExprArg.
        For sequential mode, we patch from SAVED ORIGINALS each call (AR-safe).

        Targets: ones/zeros/full/new_zeros/new_ones (shape args),
                 expand (size arg), slice (end arg), arange (end arg).
        """
        import copy

        assert self._dag is not None
        symbolic_context = self._dag.get("symbolic_context", {})
        symbols = symbolic_context.get("symbols", {})

        # Build trace_val → runtime_val map for seq_len symbols
        seq_trace_vals = set()
        runtime_map = {}  # trace_value → runtime_value
        for sym_id, sym_info in symbols.items():
            if sym_info.get("name") != "seq_len":
                continue
            trace_val = sym_info.get("trace_value")
            runtime_val = bound_symbols.get(sym_id)
            if trace_val is not None and runtime_val is not None:
                seq_trace_vals.add(trace_val)
                runtime_map[trace_val] = runtime_val

        if not seq_trace_vals:
            return

        # Save original args on first call (AR loop will re-patch each step)
        if not hasattr(self, '_original_op_args'):
            self._original_op_args = {}
            ops = self._dag.get("ops", {})
            for op_uid, op_data in ops.items():
                attrs = op_data.get("attributes", {})
                args = attrs.get("args", [])
                if args:
                    self._original_op_args[op_uid] = copy.deepcopy(args)

        # Restore originals then apply current runtime values
        ops = self._dag.get("ops", {})
        for op_uid, op_data in ops.items():
            if op_uid not in self._original_op_args:
                continue

            op_type = op_data.get("op_type", "")
            attrs = op_data.get("attributes", {})

            # Restore from saved original
            original_args = copy.deepcopy(self._original_op_args[op_uid])
            attrs["args"] = original_args
            args = original_args
            bare = op_type.replace("aten::", "")

            def _replace_in_list(vals):
                for i, v in enumerate(vals):
                    if isinstance(v, int) and v in runtime_map:
                        vals[i] = runtime_map[v]

            # ones/zeros/full/new_zeros/new_ones
            if bare in ("ones", "zeros", "full", "new_zeros", "new_ones"):
                shape_idx = 1 if bare.startswith("new_") else 0
                if len(args) > shape_idx:
                    arg = args[shape_idx]
                    if isinstance(arg, dict) and arg.get("type") == "list":
                        _replace_in_list(arg.get("value", []))

            # expand
            elif bare == "expand" and len(args) >= 2:
                arg = args[1]
                if isinstance(arg, dict) and arg.get("type") == "list":
                    _replace_in_list(arg.get("value", []))

            # slice
            elif bare == "slice" and len(args) >= 4:
                arg = args[3]
                if isinstance(arg, dict) and arg.get("type") == "scalar":
                    v = arg.get("value")
                    if isinstance(v, int) and v in runtime_map:
                        arg["value"] = runtime_map[v]

            # arange
            elif bare == "arange":
                for idx in range(min(len(args), 2)):
                    arg = args[idx]
                    if isinstance(arg, dict):
                        for key in ("value", "trace_value"):
                            v = arg.get(key)
                            if isinstance(v, int) and v in runtime_map:
                                arg[key] = runtime_map[v]

    def _adapt_seq_dependent_weights(self, bound_symbols: Dict[str, int]) -> None:
        """Slice constant_T_* weights to match runtime seq_len.

        RoPE cos/sin tables are captured at trace_seq_len. If runtime seq_len
        differs, we must slice (shorter) or the pre-computed tables (longer).
        This is the sequential equivalent of CompiledSequence.update_seq_dependent_constants().
        """
        assert self._dag is not None
        symbolic_context = self._dag.get("symbolic_context", {})
        symbols = symbolic_context.get("symbols", {})

        # Find seq_len symbol and its trace/runtime values
        for sym_id, sym_info in symbols.items():
            if sym_info.get("name") != "seq_len":
                continue
            trace_val = sym_info.get("trace_value")
            runtime_val = bound_symbols.get(sym_id)
            if trace_val is None or runtime_val is None or runtime_val == trace_val:
                continue

            # Scan weights for constant_T_* tensors with trace_seq_len in shape
            tensors_meta = self._dag.get("tensors", {})
            for wname, weight in list(self._weights.items()):
                if not wname.startswith("constant_T_"):
                    continue
                for axis, dim in enumerate(weight.shape):
                    if dim == trace_val:
                        if runtime_val < trace_val:
                            # Slice down
                            slices = [slice(None)] * weight.ndim
                            slices[axis] = slice(0, runtime_val)
                            self._weights[wname] = weight[tuple(slices)]
                        # If runtime > trace, the table is too small — keep full table
                        # (downstream ops will handle via modular indexing)
                        break

    def _execute_all_ops(self) -> ExecutionStats:
        """
        Execute all ops in execution order.

        Execution modes:
        - "compiled": Pre-compiled PyTorch execution sequence (DEFAULT)
        - "sequential": Pure PyTorch ATen op-by-op eager (debug)

        Note: "triton" and "triton_sequential" modes are handled by
        run() → _run_triton() and never reach this method.

        Returns:
            ExecutionStats with timing and counts
        """
        # COMPILED MODE (default): uses CompiledSequence with PyTorch ops
        if self.mode == "compiled":
            return self._execute_compiled_graph()

        # Compute last use of each tensor for memory management
        last_use_tid = self._compute_last_use()

        assert self._ctx is not None
        execution_order = self._ctx.execution_order
        ops = self._ctx.ops_metadata

        triton_count = 0
        metadata_count = 0
        total_ops = len(execution_order)

        # Sequential parity instrumentation — mirrors NBX_LIVE_DUMP_EVERY in
        # triton/sequence.py. Every N ops, log torch.cuda.memory_allocated()
        # so a sequential vs triton_sequential bisection has comparable per-op
        # live-watermark traces. Cheap (env read once + memory_allocated call
        # every N ops). Disabled by default (env var unset).
        import os as _os_nat
        _native_per = int(_os_nat.environ.get("NBX_NATIVE_LIVE_DUMP_EVERY", "0") or "0")

        for i, op_uid in enumerate(execution_order):
            # Read op_data DIRECTLY - no transformation
            op_data = ops[op_uid]
            op_type = op_data["op_type"]

            try:
                # Execute op on default stream
                exec_type = self._dispatch_op(op_uid, op_data, op_type)

                # Update counts
                if exec_type == OpExecution.METADATA:
                    metadata_count += 1
                else:
                    triton_count += 1

                # NAN/INF guard for TRITON ops only
                self._check_nan_inf(op_uid, op_type, exec_type)

                # Clear op_outputs entry after nan/inf check (memory management)
                if op_uid in self._ctx.op_outputs:
                    del self._ctx.op_outputs[op_uid]

            except Exception as e:
                self._handle_op_error(i, total_ops, op_uid, op_type, op_data, e)

            # Memory management: Clean up finished tensors
            self._cleanup_finished_tensors(i, op_data, last_use_tid)

            if _native_per > 0 and i % _native_per == 0:
                try:
                    _live = torch.cuda.memory_allocated() / 1024 / 1024
                    _peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                    print(f"[NATIVE_LIVE_TRACK op_idx={i} {op_uid}] "
                          f"live={_live:.0f}MB peak={_peak:.0f}MB",
                          flush=True)
                except Exception:
                    pass

        return ExecutionStats(
            total_ops=total_ops,
            triton_ops=triton_count,
            metadata_ops=metadata_count,
            total_time_ms=0,  # Computed in run()
        )

    def _execute_compiled_graph(self) -> ExecutionStats:
        """
        Execute via CompiledSequence (Zero-Overhead Mode).

        CompiledSequence uses pre-compiled closures for argument resolution:
        - Zero isinstance() checks in hot loop
        - Zero dict lookups (all slots are integers)
        - Zero string operations
        - Arena-based tensor storage with __slots__

        Performance gains vs standard native:
        - ~15-25ms per step → <1ms per step (95%+ reduction in Python overhead)
        - GPU-Util: 60-80% → 90-95% (GPU stays busy)

        THIS IS THE ZERO-OVERHEAD HOT PATH:
        1. Bind weights (once at load — cached after first call)
        2. Bind inputs (per inference)
        3. Run compiled closures (zero overhead)
        4. Gather outputs
        """
        start_time = time.perf_counter()

        if self._compiled_seq is None or self._interceptors_dirty:
            self._compile_execution_sequence()
            self._interceptors_dirty = False
            self._weights_bound = False  # Force rebind after recompile

        assert self._dag is not None
        tensors = self._dag.get("tensors", {})

        # Reconcile weight key names with graph param names.
        self._reconcile_weight_keys()

        # Bind weights to arena slots (uses tensor IDs from DAG)
        weight_map = {}
        for tid in tensors:
            if tid.startswith("param::"):
                weight_name = tid[7:]
                if weight_name in self._weights:
                    weight_map[tid] = self._weights[weight_name]
            elif tid.startswith("buffer::"):
                buffer_name = tid[8:]
                if buffer_name in self._weights:
                    weight_map[tid] = self._weights[buffer_name]

        assert self._compiled_seq is not None
        self._compiled_seq.bind_weights(weight_map)

        # FGP: Derive per-op device from weight tensor placement
        # Only run once — device layout is static after load_weights.
        if not getattr(self, '_devices_computed', False):
            self._compiled_seq.compute_op_devices()
            self._devices_computed = True

        # Bind inputs to arena slots
        # Input tensor IDs use prefix: input::input_name
        # The ctx.inputs dict uses just the input_name as key
        # For nested dicts like added_cond_kwargs.resolution, navigate the path
        #
        # CRITICAL: Apply same dtype conversion as sequential mode (_prepare_execution)
        # Prism dtype (self.dtype) is source of truth for floating-point inputs
        input_map = {}
        assert self._ctx is not None
        for tid, tdata in tensors.items():
            if tid.startswith("input::"):
                input_name = tid[7:]  # Strip "input::"
                value = None

                # Check flat key first (handles dotted names like ref_dict.prompt_token),
                # then try nested dict navigation for actual nested inputs
                if input_name in self._ctx.inputs:
                    value = self._ctx.inputs[input_name]
                elif "." in input_name:
                    # Handle dotted paths for nested dicts (e.g., "added_cond_kwargs.resolution")
                    parts = input_name.split(".")
                    value = self._ctx.inputs
                    found = True
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            found = False
                            break
                    if not found:
                        value = None

                # Apply Prism dtype + device conversion (same logic as sequential mode)
                if isinstance(value, torch.Tensor):
                    # For floating-point tensors, convert to Prism dtype
                    torch_dtype = get_torch_dtype(self.dtype)
                    if value.dtype.is_floating_point and value.dtype != torch_dtype:
                        value = value.to(torch_dtype)
                    # Move input to executor's device (critical for FGP/multi-device)
                    if str(value.device) != str(self.device):
                        value = value.to(self.device)
                    input_map[tid] = value

        assert self._compiled_seq is not None
        self._compiled_seq.bind_inputs(input_map)

        # Bind symbolic shape resolver to CompiledSequence
        # This enables dynamic symbol resolution for ops like view([s0, dim])
        if self._symbolic_shapes_enabled and self._shape_resolver:
            self._compiled_seq.bind_symbols(self._shape_resolver)
            # Slice RoPE constants to match runtime seq_len
            self._compiled_seq.update_seq_dependent_constants()

        # ZERO-OVERHEAD HOT LOOP
        # The run() method uses pre-compiled closures:
        # - args_resolver: lambda arena: [arena[s1], arena[s2], ...]
        # - kwargs_resolver: lambda arena: {"k1": arena[s3], ...}
        # - func: direct function reference (no string dispatch)
        # - output_slots: tuple of integer indices
        assert self._compiled_seq is not None
        num_ops = self._compiled_seq.num_ops

        self._compiled_seq.run(pre_op_callback=self._pre_op_callback)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._compiled_exec_count += 1

        # Store outputs in context for _gather_outputs()
        assert self._ctx is not None
        graph_output_ids = self._ctx.output_tensor_ids
        outputs = self._compiled_seq.gather_outputs(graph_output_ids)
        for tid, tensor in outputs.items():
            self._ctx.tensor_store[tid] = tensor

        # Copy persistent tensors (e.g., hidden states for LLM) to tensor_store
        if self._persistent_tensor_ids:
            persistent_outputs = self._compiled_seq.gather_outputs(list(self._persistent_tensor_ids))
            for tid, tensor in persistent_outputs.items():
                self._ctx.tensor_store[tid] = tensor

        return ExecutionStats(
            total_ops=num_ops,
            triton_ops=0,
            metadata_ops=num_ops,
            total_time_ms=elapsed_ms,
        )

    def _compute_last_use(self) -> Dict[str, int]:
        """
        Compute last use index for each tensor.

        When stride optimization is enabled, chain source tensors must live
        as long as their longest-lived view. This is because views created
        via as_strided() share the underlying storage with the chain source.

        Returns:
            Dict mapping tensor_id -> last execution_order index where used
        """
        last_use_tid = {}
        assert self._ctx is not None
        execution_order = self._ctx.execution_order
        ops = self._ctx.ops_metadata

        # Pass 1: Standard last use computation
        for i, op_uid in enumerate(execution_order):
            op_data = ops[op_uid]
            for tid in op_data.get("input_tensor_ids", []):
                last_use_tid[tid] = i

        # Pass 2: Extend chain source lifetime to cover all derived views
        # When stride skip is enabled, views share storage with chain source
        stride_skip_enabled = os.environ.get("NBX_STRIDE_SKIP", "0") == "1"
        if stride_skip_enabled and hasattr(self, '_stride_info') and self._stride_info is not None:  # type: ignore[attr-defined]
            # Build reverse mapping: source_tensor_id -> list of view tensor ids
            source_to_views: Dict[str, List[str]] = {}
            for view_tid, stride_info in self._stride_info.items():  # type: ignore[attr-defined]
                if stride_info.source_tensor_id:
                    source_tid = stride_info.source_tensor_id
                    if source_tid not in source_to_views:
                        source_to_views[source_tid] = []
                    source_to_views[source_tid].append(view_tid)

            # For each chain source, extend its lifetime to max of all views
            for source_tid, view_tids in source_to_views.items():
                max_view_last_use = 0
                for view_tid in view_tids:
                    view_last = last_use_tid.get(view_tid, 0)
                    max_view_last_use = max(max_view_last_use, view_last)

                # Extend source tensor's last use if views live longer
                current_last = last_use_tid.get(source_tid, 0)
                if max_view_last_use > current_last:
                    last_use_tid[source_tid] = max_view_last_use

        return last_use_tid

    def _dispatch_op(
        self,
        op_uid: str,
        op_data: Dict[str, Any],
        op_type: str
    ) -> OpExecution:
        """
        Dispatch op to appropriate handler.

        Args:
            op_uid: Operation UID
            op_data: Operation data from DAG
            op_type: Operation type string

        Returns:
            OpExecution type used
        """
        # SEQUENTIAL MODE: Run ALL ops through PyTorch ATen op-by-op
        # Return METADATA to skip nan/inf check - PyTorch handles this correctly
        # (e.g., T5 GELU pow(x,3) produces transient inf that gets masked by where())
        if self.mode == "sequential":
            self._execute_native_op(op_uid, op_data, op_type)
            return OpExecution.METADATA

        # All other modes (compiled, triton) go through their own execution paths
        # and never reach _dispatch_op. If we get here with an unexpected mode, crash.
        raise RuntimeError(
            f"ZERO FALLBACK: _dispatch_op called with unexpected mode '{self.mode}'. "
            f"Expected 'sequential' (the eager-mode dispatcher sets mode='sequential' "
            f"temporarily; the historical name 'native' is deprecated)."
        )

    def _check_nan_inf(self, op_uid: str, op_type: str, exec_type: OpExecution) -> None:
        """
        Check for NaN/Inf in TRITON op outputs.

        NOTE: Only checks TRITON ops. METADATA ops may produce transient inf
        values that get masked by subsequent operations.
        """
        if exec_type == OpExecution.METADATA:
            return

        assert self._ctx is not None
        if op_uid not in self._ctx.op_outputs:
            return

        for idx, out in enumerate(self._ctx.op_outputs[op_uid]):
            if isinstance(out, torch.Tensor) and out.dtype in (torch.float32, torch.float16, torch.bfloat16):
                has_nan = torch.isnan(out).any()
                has_pos_inf = torch.isinf(out).any() and (out == float('inf')).any()

                if has_nan or has_pos_inf:
                    msg = "NaN" if has_nan else "Pos-Inf"
                    if has_nan and has_pos_inf:
                        msg = "NaN and Pos-Inf"
                    raise RuntimeError(
                        f"{msg} detected in output {idx} of op '{op_uid}' ({op_type})."
                    )

    def _handle_op_error(
        self,
        op_idx: int,
        total_ops: int,
        op_uid: str,
        op_type: str,
        op_data: Dict[str, Any],
        error: Exception
    ) -> None:
        """Handle and re-raise op execution errors with context."""
        raise error

    def _cleanup_finished_tensors(
        self,
        op_idx: int,
        op_data: Dict[str, Any],
        last_use_tid: Dict[str, int]
    ) -> None:
        """
        Clean up tensors that are no longer needed.

        When memory pooling is enabled (NBX_MEMORY_POOL=1), instead of deleting
        tensors, we release them to the pool for potential reuse.

        Args:
            op_idx: Current operation index
            op_data: Current operation data
            last_use_tid: Dict mapping tensor_id -> last use index
        """
        # Check inputs of current op - if this was their last use, delete them
        assert self._ctx is not None
        for tid in op_data.get("input_tensor_ids", []):
            if tid in last_use_tid and last_use_tid[tid] <= op_idx:
                # Check if this is NOT a model output and NOT persistent
                if tid not in self._ctx.output_tensor_ids and tid not in self._persistent_tensor_ids:
                    # Phase 2+3: If memory pooling enabled, release to pool instead of deleting
                    if self._use_memory_pool and self._memory_pool is not None:
                        tensor = self._ctx.get_tensor(tid)
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            # Only pool contiguous tensors that own their storage
                            if tensor.is_contiguous() and tensor.storage_offset() == 0:
                                self._memory_pool.release(tensor)
                    self._ctx.delete_tensor(tid)

    # =========================================================================
    # Op Dispatch (delegates to kernels/ infrastructure)
    # =========================================================================

    def _execute_metadata_op(self, op_uid: str, op_data: Dict[str, Any]) -> None:
        """
        Execute a metadata op from its raw DAG data.

        Metadata ops are O(1) - they only modify tensor views/strides.
        PyTorch is optimal for these.

        Delegates to: kernels/metadata_ops.py
        """
        from neurobrix.kernels.metadata_ops import execute_metadata_op

        op_type = op_data["op_type"]
        attrs = op_data.get("attributes", {})

        # Resolve inputs via TensorResolver
        assert self._resolver is not None
        inputs = self._resolver.resolve_args(op_uid, attrs, op_type, op_data)

        # Build metadata attrs - INCLUDE args/kwargs for handlers
        metadata_attrs = dict(attrs)
        input_shapes = op_data.get("input_shapes", [])
        output_shapes = op_data.get("output_shapes", [])
        if input_shapes:
            metadata_attrs["input_shapes"] = input_shapes
        if output_shapes:
            metadata_attrs["_output_shape"] = output_shapes[0]
        metadata_attrs["_device"] = self.device

        # DtypeEngine intercept: _to_copy needs Prism override + complex protection
        if op_type == "aten::_to_copy" and inputs:
            to_copy_attrs = dict(attrs)
            if "output_dtypes" in op_data:
                to_copy_attrs["output_dtypes"] = op_data["output_dtypes"]
            to_copy_fn = self._dtype_engine.compile_op(op_type, None, to_copy_attrs)
            result = to_copy_fn(inputs[0])
        else:
            result = execute_metadata_op(op_type, inputs, metadata_attrs)

        # Store outputs
        self._store_op_outputs(op_uid, op_data, result)


    def _execute_native_op(self, op_uid: str, op_data: Dict[str, Any], op_type: str) -> None:
        """
        Execute compute op via native PyTorch ATen kernels (bypass Triton).

        Uses DtypeEngine AMP rules for numerical stability (fp32 upcast for
        pow/rsqrt/softmax, overflow protection for fp16 add/sub, etc.).

        Delegates to: core/runtime/sequential_dispatcher.py
        """
        from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher

        if self._sequential_dispatcher is None:
            self._sequential_dispatcher = NativeATenDispatcher(device=self.device, compute_dtype=get_torch_dtype(self.dtype))

        # FUSED MoE OP: Execute via custom handler (bypasses TensorResolver)
        # Must be checked BEFORE resolve_args — fused op's args list contains 192+
        # expert weight tensor IDs that the standard resolver can't handle.
        if op_type == "custom::moe_fused":
            result = self._execute_moe_fused_native(op_data)
            self._store_op_outputs(op_uid, op_data, result)
            # Defragment CUDA memory after MoE expert loop — the 64-expert iteration
            # creates many small allocations that fragment the caching allocator.
            device_empty_cache(self.device)
            return

        attrs = op_data.get("attributes", {})
        attrs = dict(attrs)  # Copy to avoid mutating source
        attrs["input_shapes"] = op_data.get("input_shapes", [])
        attrs["output_shapes"] = op_data.get("output_shapes", [])

        # Resolve inputs via TensorResolver
        assert self._resolver is not None
        resolved_inputs = self._resolver.resolve_args(op_uid, attrs, op_type, op_data)
        normalized_inputs = self._resolver.normalize_inputs(resolved_inputs)

        # AMP: Cast inputs per DtypeEngine rules (fp32 for pow/rsqrt/softmax, etc.)
        normalized_inputs = self._dtype_engine.amp_cast_inputs(op_type, normalized_inputs)

        # Check for op interceptors. Priority order matches CompiledSequence
        # and TritonSequence: op_uid (fine-grained, op-level tiling Prism)
        # > op_type (broad, KV cache SDPA) > native op resolver. Without the
        # op_uid branch, sequential mode silently bypasses Prism's
        # `register_op_uid_interceptors` plumbing — Sana 4Kpx VAE conv::54
        # then arrives at cuDNN with its raw 36 GiB request and OOMs on a
        # 32 GB V100, while compiled mode (which DOES consume the op_uid
        # interceptors) tiles it into ~600 MB bands. R30 (Mode Universality)
        # parity, P-SANA-4KPX-RUNTIME 2026-05-05 bisection.
        if hasattr(self, '_op_uid_interceptors') and op_uid in self._op_uid_interceptors:
            resolved_kwargs = self._resolver.resolve_kwargs(op_uid, attrs, op_type, op_data)
            result = self._op_uid_interceptors[op_uid](*normalized_inputs, **resolved_kwargs)
        elif op_type in self._op_interceptors:
            # Resolve kwargs for interceptor
            resolved_kwargs = self._resolver.resolve_kwargs(op_uid, attrs, op_type, op_data)
            result = self._op_interceptors[op_type](*normalized_inputs, **resolved_kwargs)
        elif op_type == "aten::_to_copy":
            # DtypeEngine handles _to_copy with Prism override and complex protection
            # Include output_dtypes from op_data (graph captures dtype conversions there)
            to_copy_attrs = dict(attrs)
            if "output_dtypes" in op_data:
                to_copy_attrs["output_dtypes"] = op_data["output_dtypes"]
            to_copy_fn = self._dtype_engine.compile_op(op_type, None, to_copy_attrs)
            result = to_copy_fn(normalized_inputs[0]) if normalized_inputs else None
        else:
            # Execute via PyTorch native
            result = self._sequential_dispatcher.dispatch(op_type, normalized_inputs, attrs)

        # AMP: Post-process result (overflow protection, inf clamping)
        result = self._dtype_engine.amp_cast_result(op_type, result)

        # Store outputs
        self._store_op_outputs(op_uid, op_data, result)

    def _execute_moe_fused_native(self, op_data: Dict[str, Any]) -> torch.Tensor:
        """
        Execute fused MoE op in native mode.

        Same logic as compiled_sequence.py's moe_fused_dispatch, but reads
        tensors from _ctx.tensor_store instead of arena slots.

        The fused op replaces ~893 individual ops per MoE layer with dynamic
        routing + expert FFN + scatter-add. Without this, native mode uses
        hardcoded trace-time routing indices → garbage output.
        """
        import torch.nn.functional as F

        assert self._ctx is not None
        attrs = op_data.get("attributes", {})

        gate_scores_tid = attrs["gate_scores_tid"]
        hidden_states_tid = attrs["hidden_states_tid"]
        gate_weight_ids = attrs["expert_gate_weight_ids"]
        up_weight_ids = attrs["expert_up_weight_ids"]
        down_weight_ids = attrs["expert_down_weight_ids"]
        top_k = attrs["top_k"]
        num_experts = attrs["num_experts"]
        norm_topk_prob = attrs.get("norm_topk_prob", True)

        # Note: _execute_moe_fused_native is the sequential fallback path.
        # Compiled mode uses moe_fused_dispatch() in compiled_sequence.py.
        def _dump_n(_label, _tensor): pass

        # Read tensors from store (activations are already resolved by prior ops)
        store = self._ctx.tensor_store
        hidden_states = store.get(hidden_states_tid)
        gate_scores = store.get(gate_scores_tid)

        if hidden_states is None:
            raise RuntimeError(
                f"MoE fused native: hidden_states is None (tid={hidden_states_tid})"
            )

        # Pre-resolve all expert weight tensors into store (they may not have been
        # resolved yet since the original ops that referenced them were removed by fusion)
        tensors_meta = self._ctx.tensors_metadata
        weights = self._ctx.weights
        for wid_list in (gate_weight_ids, up_weight_ids, down_weight_ids):
            for wid in wid_list:
                if wid not in store:
                    wname = tensors_meta.get(wid, {}).get("weight_name")
                    if wname and wname in weights:
                        store[wid] = weights[wname]

        # Handle 3D tensors [batch, seq, dim] → flatten to 2D [batch*seq, dim]
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        if gate_scores.dim() == 3:
            gate_scores = gate_scores.reshape(-1, gate_scores.size(-1))

        # Resolve weight dtype from first expert
        w_dtype = store.get(gate_weight_ids[0]).dtype
        if hidden_states.dtype != w_dtype:
            hidden_states = hidden_states.to(w_dtype)

        _dump_n("gate_scores_in", gate_scores)
        _dump_n("hidden_states_in", hidden_states)

        # ROUTING IN FP32 (same as compiled path)
        gate_scores = gate_scores.float()
        _compute_dev = hidden_states.device
        if gate_scores.device != _compute_dev:
            gate_scores = gate_scores.to(_compute_dev)

        # Dynamic routing (replaces hardcoded topk+sort+bincount+slice)
        scores, indices = gate_scores.topk(top_k, dim=-1)
        _dump_n("topk_scores", scores)
        _dump_n("topk_indices", indices)
        if norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
            _dump_n("scores_norm", scores)

        flat_indices = indices.flatten()
        _dump_n("flat_scores", scores.flatten())
        sorted_expert_ids, perm = flat_indices.sort()
        token_ids = perm // top_k

        counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
        boundaries = torch.cumsum(counts, dim=0)
        boundaries_cpu = boundaries.tolist()

        output = torch.zeros_like(hidden_states)
        start = 0

        for expert_id in range(num_experts):
            end = boundaries_cpu[expert_id]
            if start == end:
                start = end
                continue

            expert_token_ids = token_ids[start:end]
            expert_input = hidden_states[expert_token_ids]

            # Load expert weights from tensor store
            gate_w = store.get(gate_weight_ids[expert_id])
            up_w = store.get(up_weight_ids[expert_id])
            down_w = store.get(down_weight_ids[expert_id])

            # Multi-device alignment
            _dev = hidden_states.device
            if gate_w.device != _dev:
                gate_w = gate_w.to(_dev)
            if up_w.device != _dev:
                up_w = up_w.to(_dev)
            if down_w.device != _dev:
                down_w = down_w.to(_dev)
            if expert_input.device != _dev:
                expert_input = expert_input.to(_dev)
            if expert_token_ids.device != _dev:
                expert_token_ids = expert_token_ids.to(_dev)

            # SwiGLU FFN
            gate = F.silu(expert_input @ gate_w.t())
            up = expert_input @ up_w.t()
            expert_out = (gate * up) @ down_w.t()

            # Weighted scatter-add
            expert_scores = scores.flatten()[perm[start:end]].unsqueeze(-1).to(w_dtype)
            output.index_add_(0, expert_token_ids, expert_out * expert_scores)

            start = end

        _dump_n("result_final", output)
        # Restore original shape if input was 3D
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape)

        return output

    def _store_op_outputs(
        self,
        op_uid: str,
        op_data: Dict[str, Any],
        result: Any
    ) -> None:
        """
        Store operation outputs in context.

        Args:
            op_uid: Operation UID
            op_data: Operation data from DAG
            result: Result from op execution
        """
        output_tensor_ids = op_data.get("output_tensor_ids", [])

        if isinstance(result, torch.Tensor):
            current_outputs = [result]
        elif isinstance(result, (tuple, list)):
            current_outputs = list(result)
        else:
            current_outputs = [result]

        if len(current_outputs) < len(output_tensor_ids):
            op_type = op_data.get("op_type", "unknown")
            raise RuntimeError(
                f"Operator '{op_type}' ({op_uid}) returned {len(current_outputs)} tensors, "
                f"but graph expects {len(output_tensor_ids)}."
            )

        assert self._ctx is not None
        for idx, tid in enumerate(output_tensor_ids):
            self._ctx.store_tensor(tid, current_outputs[idx])

        # Also store in op_outputs for stats and legacy code
        self._ctx.op_outputs[op_uid] = current_outputs

    # =========================================================================
    # Output Gathering
    # =========================================================================

    def _gather_outputs(self) -> Dict[str, torch.Tensor]:
        """
        Gather final outputs using semantic resolution.

        Uses tensor IDs from graph.output_tensor_ids, resolves via TensorResolver.
        Prefers output_name if available for semantic keys.
        """
        assert self._ctx is not None
        assert self._resolver is not None
        output_ids = self._ctx.output_tensor_ids
        tensors_info = self._ctx.tensors_metadata
        outputs = {}

        for tid in output_ids:
            try:
                tensor = self._resolver.resolve(tid)
                # Use output_name as key if available, otherwise tensor_id
                tensor_info = tensors_info.get(tid, {})
                output_name = tensor_info.get("output_name")
                key = output_name if output_name else tid
                outputs[key] = tensor
            except RuntimeError as e:
                raise RuntimeError(f"Failed to gather output '{tid}': {e}")

        return outputs

    def enable_hidden_states_capture(self) -> Optional[str]:
        """
        Enable capture of hidden states for LLM models.

        Must be called BEFORE execute() to mark the hidden_states tensor as persistent.
        This prevents GC from deleting it during execution.

        For models where graph output IS hidden_states (no lm_head in graph,
        e.g. DeepSeek MoE), the graph output tensors are already persistent
        and no extra protection is needed — returns None gracefully.

        For models where graph includes lm_head (output is logits), finds
        the last mm op and protects its first input (hidden_states before projection).

        Returns:
            The tensor ID of the hidden states, or None if graph output is already hidden_states
        """
        if self._dag is None:
            return None

        # Check if graph output is already hidden_states (no lm_head in graph).
        # Graph outputs are always persistent, so no capture needed.
        output_tids = self._dag.get("output_tensor_ids", [])
        if output_tids:
            # Graph outputs are kept by default — no need to protect anything extra.
            # The caller (autoregressive flow) will use get_hidden_states() which
            # handles both cases: lm_head-in-graph and no-lm_head.
            pass

        # Find the last mm op in execution order (for lm_head-in-graph case)
        execution_order = self._dag.get("execution_order", [])
        for op_uid in reversed(execution_order):
            op_data = self._dag["ops"].get(op_uid, {})
            if op_data.get("op_type") in ("aten::mm", "mm"):
                input_ids = op_data.get("input_tensor_ids", [])
                if input_ids:
                    hidden_tid = input_ids[0]  # First input is hidden states
                    self._persistent_tensor_ids.add(hidden_tid)
                    # Forward to compiled sequence so it survives liveness GC
                    if self._compiled_seq is not None:
                        self._compiled_seq.protect_tensor(hidden_tid)
                    return hidden_tid
        return None

    def get_hidden_states(
        self,
        expected_hidden_dim: int = 4096,
        expected_batch_size: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract hidden states from the last execution (for LLM models).

        Two strategies (family-aware):
        1. Graph output IS hidden_states (no lm_head in graph, e.g. DeepSeek MoE):
           → Use graph output tensor directly if last dim == expected_hidden_dim
        2. Graph includes lm_head (output is logits, e.g. Janus):
           → Find last mm, get its first input (hidden_states before projection)

        Args:
            expected_hidden_dim: Expected hidden dimension (e.g., 4096 for 7B, 2048 for MoE)
            expected_batch_size: Expected batch size (e.g., 2 for CFG). If None, inferred.

        Returns:
            Hidden states tensor [batch, seq_len, hidden_dim] or None
        """
        # Triton mode: tensor_store is not populated (execution lives in
        # the TritonSequence arena, not self._ctx). Read the persistent
        # tid's slot directly from the triton sequence's arena.
        if self.mode in ("triton", "triton_sequential"):
            return self._get_hidden_states_triton(
                expected_hidden_dim, expected_batch_size)

        if self._ctx is None:
            return None

        # Strategy 1: Check graph output — if last dim matches hidden_dim,
        # the graph outputs hidden_states directly (no lm_head in graph).
        assert self._dag is not None
        output_tids = self._dag.get("output_tensor_ids", [])
        for tid in output_tids:
            tensor = self._ctx.tensor_store.get(tid)
            if tensor is not None and tensor.shape[-1] == expected_hidden_dim:
                result = self._reshape_hidden(tensor, expected_hidden_dim, expected_batch_size)
                if result is not None:
                    return result

        # Strategy 2: Find last mm op (lm_head projection), get its input.
        assert self._dag is not None
        execution_order = self._dag.get("execution_order", [])
        last_mm_uid = None
        for op_uid in reversed(execution_order):
            op_data = self._dag["ops"].get(op_uid, {})
            if op_data.get("op_type") in ("aten::mm", "mm"):
                last_mm_uid = op_uid
                break

        if not last_mm_uid:
            return None

        assert self._dag is not None
        op_data = self._dag["ops"][last_mm_uid]
        input_ids = op_data.get("input_tensor_ids", [])
        if len(input_ids) < 1:
            return None

        hidden_tid = input_ids[0]
        hidden_tensor = self._ctx.tensor_store.get(hidden_tid)
        if hidden_tensor is None:
            return None

        return self._reshape_hidden(hidden_tensor, expected_hidden_dim, expected_batch_size)

    def _get_hidden_states_triton(
        self,
        expected_hidden_dim: int,
        expected_batch_size: Optional[int],
    ) -> Optional[Any]:
        """Triton-mode hidden-states retrieval.

        Mirrors the two-strategy cascade of the native `get_hidden_states`
        but reads from the TritonSequence arena (the triton analog of
        ExecutionContext.tensor_store). Returns an NBXTensor — callers on
        the triton path are already NBX-native (TritonLMSession).
        """
        if not hasattr(self, "_triton_seq") or self._triton_seq is None:
            return None
        assert self._dag is not None

        # Strategy 1: declared graph output whose last dim matches hidden_dim
        output_tids = self._dag.get("output_tensor_ids", [])
        arena_outs = self._triton_seq.gather_outputs(list(output_tids))
        for tid, tensor in arena_outs.items():
            if tensor is not None and tensor.shape[-1] == expected_hidden_dim:
                return tensor

        # Strategy 2: pre-lm_head input (image-AR LLMs: graph output is
        # text-vocab logits, hidden states are the first input to the
        # last aten::mm). Requires that enable_hidden_states_capture()
        # added hidden_tid to _persistent_tensor_ids BEFORE triton compile
        # — _ensure_triton_compiled threads that into the dag's declared
        # outputs so the arena slot stays alive through liveness GC.
        execution_order = self._dag.get("execution_order", [])
        last_mm_uid = None
        for op_uid in reversed(execution_order):
            op_data = self._dag["ops"].get(op_uid, {})
            if op_data.get("op_type") in ("aten::mm", "mm"):
                last_mm_uid = op_uid
                break
        if not last_mm_uid:
            return None
        op_data = self._dag["ops"][last_mm_uid]
        input_ids = op_data.get("input_tensor_ids", [])
        if not input_ids:
            return None
        hidden_tid = input_ids[0]
        gathered = self._triton_seq.gather_outputs([hidden_tid])
        hidden = gathered.get(hidden_tid)
        if hidden is None:
            return None
        # Reshape: triton may hand back the flattened (1408, 4096) form
        # (per Janus's lm_head view). Unflatten to (batch, seq, hidden).
        if hidden.ndim == 2:
            total, hdim = hidden.shape
            if hdim == expected_hidden_dim:
                bsz = expected_batch_size or 1
                if bsz > 1 and total % bsz == 0:
                    return hidden.view(bsz, total // bsz, hdim)
                return hidden.unsqueeze(0)
        return hidden

    def _reshape_hidden(
        self,
        tensor: torch.Tensor,
        expected_hidden_dim: int,
        expected_batch_size: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Reshape hidden states tensor to [batch, seq, hidden_dim]."""
        if tensor.dim() == 2:
            total_tokens, hidden_dim = tensor.shape
            if hidden_dim == expected_hidden_dim:
                if expected_batch_size is not None and expected_batch_size > 1:
                    batch_size = expected_batch_size
                else:
                    batch_size = self._last_symbols.get("s0", 1) if hasattr(self, "_last_symbols") else 1

                if batch_size > 1 and total_tokens % batch_size == 0:
                    seq_len = total_tokens // batch_size
                    return tensor.view(batch_size, seq_len, hidden_dim)
                else:
                    return tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.shape[-1] == expected_hidden_dim:
                return tensor
        return None

    # =========================================================================
    # Properties and Utilities
    # =========================================================================

    @property
    def stats(self) -> Optional[ExecutionStats]:
        """Get stats from last execution."""
        return self._last_stats

    @property
    def op_count(self) -> int:
        """Total number of operations in graph."""
        if self._dag is None:
            return 0
        return len(self._dag.get("ops", {}))

    @property
    def dag(self) -> Optional[Dict[str, Any]]:
        """Get raw DAG dict (read-only access)."""
        return self._dag

    def get_op_summary(self) -> Dict[str, int]:
        """Get count of ops by execution type."""
        summary = {"triton": 0, "hybrid": 0, "metadata": 0, "unknown": 0}

        if self._dag is None:
            return summary

        for op_data in self._dag.get("ops", {}).values():
            op_type = op_data.get("op_type", "")
            try:
                exec_type = get_execution_type(op_type)
                summary[exec_type.name.lower()] += 1
            except KeyError:
                summary["unknown"] += 1

        return summary

    def get_embed_tokens(self) -> "Optional[torch.Tensor]":
        """Get embedding weight tensor for token embedding (LLMs).

        NeuroTax standard names: token_embed.weight, embed.weight
        Some models prefix with 'model.': model.token_embed.weight
        """
        # Try exact NeuroTax names first
        for pattern in ("token_embed.weight", "model.token_embed.weight",
                        "embed.weight", "weight"):
            if pattern in self._weights:
                return self._weights[pattern]
        # Fallback: find any 2D weight with 'embed' in key (NeuroTax standard)
        for key, weight in self._weights.items():
            if "embed" in key and weight.dim() == 2:
                return weight
        return None

    get_embed_weight = get_embed_tokens

    def unload_weights(self) -> None:
        """
        Unload all weights from GPU to free memory.

        Call this when the component is no longer needed (e.g., before VAE decode).
        Respects _persistent flag: when True, weights stay in VRAM (serving mode).

        CRITICAL: Must clear the compiled sequence arena BEFORE clearing weights
        dict, because the arena holds direct references to weight tensors in its
        slots. Without this, tensor refs survive in arena → GPU memory never
        freed. Applies to BOTH the native `_compiled_seq._arena` (torch.Tensor
        slots) and the triton `_triton_seq._arena` (NBXTensor slots). The
        `_triton_seq` attribute is created lazily in `_ensure_triton_compiled`,
        so guard with `hasattr`.
        """
        if self._persistent:
            return

        # Step 1a: Clear native compiled sequence arena (torch.Tensor refs).
        if self._compiled_seq is not None:
            assert hasattr(self._compiled_seq, '_arena')
            self._compiled_seq._arena.clear_all()  # type: ignore[attr-defined]
            self._compiled_seq = None

        # Step 1b: Clear triton sequence arena (NBXTensor refs).
        # Sync before releasing NBXTensor Python refs so __del__ → free_cuda
        # can't race an async kernel still reading the memory (UAF).
        if hasattr(self, '_triton_seq') and self._triton_seq is not None:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
            _DA.sync_device()
            self._triton_seq._arena.clear_all()  # type: ignore[attr-defined]
            self._triton_seq = None

        # Step 2: Use centralized memory manager
        MemoryManager.cleanup_context(self._ctx)
        MemoryManager.unload_weights(self._weights)
        self._weights_loaded = False
        self._weights_bound = False

    def cleanup(self) -> None:
        """
        Release per-request resources.

        Respects _persistent flag: when True, preserves weights in VRAM
        and keeps compiled sequence for reuse. Only clears per-request
        intermediates (arena slots, execution context).

        Mirrors the unload_weights arena-clear for BOTH native
        `_compiled_seq._arena` (torch.Tensor slots) and triton
        `_triton_seq._arena` (NBXTensor slots). Unlike unload_weights,
        cleanup preserves the sequence objects themselves so subsequent
        requests can rebind fresh weights into the same compiled slot
        mapping without recompiling the op graph.
        """
        if self._persistent:
            # Persistent mode: clear per-request state only, keep weights
            # + compiled sequence (both native and triton).
            if self._compiled_seq is not None:
                self._compiled_seq._arena.clear_all()  # type: ignore[attr-defined]
            if hasattr(self, '_triton_seq') and self._triton_seq is not None:
                from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
                _DA.sync_device()
                self._triton_seq._arena.clear_all()  # type: ignore[attr-defined]
            MemoryManager.cleanup_context(self._ctx)
            return

        # Standard mode: unload weights but KEEP compiled sequence.
        # CompiledSequence holds the compiled op graph (MoE fusion, SymPromotion,
        # DtypeEngine rules) — recompiling is expensive and produces identical results.
        # Only the arena (tensor data) and weights need clearing.
        if self._compiled_seq is not None:
            assert hasattr(self._compiled_seq, '_arena')
            self._compiled_seq._arena.clear_all()  # type: ignore[attr-defined]
            # Keep _compiled_seq alive — R2 rebinds fresh weights into same slots
            # Keep _persistent_tensor_ids — slot mappings remain valid in same sequence

        if hasattr(self, '_triton_seq') and self._triton_seq is not None:
            from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
            _DA.sync_device()
            self._triton_seq._arena.clear_all()  # type: ignore[attr-defined]
            # Keep _triton_seq alive — R2 rebinds fresh weights into same slots

        MemoryManager.cleanup_context(self._ctx)
        MemoryManager.unload_weights(self._weights)
        self._weights_loaded = False  # Allow re-loading on next use
