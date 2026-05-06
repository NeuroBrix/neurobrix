"""
TilingEngine — Universal Tiled Execution for Spatial Components.

Splits large spatial inputs into overlapping tiles, executes each through
the component's CompiledSequence at trace size, then blends results using
accumulate-and-divide averaging.

ZERO SEMANTIC: No domain knowledge. Handles any 4D spatial tensor [B, C, H, W].
ZERO HARDCODE: All parameters derived from graph.json + profile.json.
ZERO FALLBACK: Crashes if config is incomplete.

Supports:
- VAE decoders (Sana 4K DC-AE: window attention seam artifacts)
- Upscalers (Swin2SR, Real-ESRGAN: trained at 64x64, must handle arbitrary)
- Any spatial component traced at fixed resolution

Algorithm: Accumulate-and-divide (SwinIR/Swin2SR pattern)
- Each tile contributes weight=1 to output accumulator
- Overlapping pixels get averaged (sum / count)
- Proven across window-attention models, numerically stable
"""

import torch
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TilingEngine:
    """
    Universal tiled execution engine for spatial components.

    Tiles large inputs into overlapping patches at trace size,
    executes each tile, then blends via accumulate-and-divide.

    All parameters are derived from the component's graph and profile.
    """

    def __init__(
        self,
        trace_size: int,
        overlap: int,
        scale_factor: int,
        window_alignment: int = 1,
        tile_size: Optional[int] = None,
    ):
        """
        Initialize tiling engine.

        Args:
            trace_size: Spatial size the graph was traced at (e.g., 128 for 128x128)
            overlap: Overlap between tiles in input space (e.g., 16)
            scale_factor: Output/input spatial ratio (e.g., 32 for DC-AE, 4 for Swin2SR x4)
            window_alignment: Tile positions must align to this (e.g., 8 for window_size=8)
            tile_size: Actual tile size for execution. When graph has symbolic spatial
                      dims, this can be smaller than trace_size. Defaults to trace_size.
        """
        self.trace_size = trace_size
        self.tile_size = tile_size if tile_size is not None else trace_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.window_alignment = window_alignment

        # Stride = tile_size - overlap
        self.stride = self.tile_size - overlap

        # Align stride to window_alignment
        if window_alignment > 1:
            self.stride = (self.stride // window_alignment) * window_alignment
            if self.stride == 0:
                self.stride = window_alignment

        # Recalculate effective overlap from aligned stride
        self.overlap = self.tile_size - self.stride

    @classmethod
    def from_component_config(
        cls,
        graph_path: Path,
        profile_path: Path,
    ) -> Optional["TilingEngine"]:
        """
        Create TilingEngine from component's graph and profile.

        DATA-DRIVEN: Reads trace size from graph input shape, scale factor
        and window alignment from profile config.

        Returns None if the component has no tiling-relevant config
        (no 4D spatial input, no upscale/decoder config).

        Args:
            graph_path: Path to component's graph.json
            profile_path: Path to component's profile.json

        Returns:
            TilingEngine instance, or None if tiling not applicable
        """
        if not graph_path.exists() or not profile_path.exists():
            return None

        # --- Step 1: Read trace size from graph input shape ---
        with open(graph_path) as f:
            graph = json.load(f)

        input_ids = graph.get("input_tensor_ids", [])
        tensors = graph.get("tensors", {})

        # NBX SYMBOLIC SHAPES — MASTER CONTRACT
        # ────────────────────────────────────────────────────────────────
        # Forge tracer marks dims that the runtime can rebind as symbolic
        # in `tensor["symbolic_shape"]["dims"]`. CompiledSequence's symbol
        # binding pass adapts every consumer op to the actual runtime
        # shape, so a graph with symbolic spatial dims is spatial-adaptive
        # by construction and does NOT need TilingEngine.
        #
        # TilingEngine is for graphs with FULLY-CONCRETE spatial shapes
        # (typical: pure tile-only models like Swin2SR upscalers, or any
        # graph that hardcodes patch grids / spatial buffers).
        #
        # Reading only `tensor["shape"]` (the concrete trace value) and
        # ignoring `tensor["symbolic_shape"]` is THE pattern that produced
        # the Sana 4Kpx triton crash — TilingEngine activated on a DC-AE
        # VAE whose H/W are symbolic, fed an NBXTensor into the torch-only
        # accumulator path, and TypeError'd. The fix is upstream: never
        # instantiate TilingEngine for spatial-adaptive graphs.
        #
        # See docs/architecture/symbolic-shapes-contract.md for the full
        # contract and why every site that inspects an NBX graph shape
        # must consult symbolic_shape before drawing conclusions from the
        # concrete trace value.
        for input_id in input_ids:
            tensor = tensors.get(str(input_id), {})
            symbolic_dims = tensor.get("symbolic_shape", {}).get("dims", [])
            if not symbolic_dims:
                continue
            # Spatial dims start at index 2 (skip batch and channels).
            # 4D NCHW: indices 2 (H), 3 (W). 5D NCDHW: 2 (D), 3 (H), 4 (W).
            for idx in range(2, len(symbolic_dims)):
                dim_spec = symbolic_dims[idx]
                if isinstance(dim_spec, dict) and dim_spec.get("type") == "symbol":
                    return None  # spatial-adaptive graph — no tiling

        trace_size = None
        for input_id in input_ids:
            tensor = tensors.get(str(input_id), {})
            shape = tensor.get("shape", [])
            # Spatial input is [B, C, H, W] — get H
            if len(shape) == 4:
                trace_size = shape[2]
                break

        if trace_size is None:
            # No 4D spatial input — tiling not applicable
            return None

        # --- Step 2: Read scale factor and window config from profile ---
        with open(profile_path) as f:
            profile = json.load(f)

        config = profile.get("config", {})

        # Scale factor: upscale (upscalers) or compression ratio (VAE decoders)
        scale_factor = config.get("upscale")

        if scale_factor is None:
            # Try VAE compression ratio from decoder blocks
            decoder_blocks = config.get("decoder_block_out_channels")
            if decoder_blocks:
                scale_factor = 2 ** (len(decoder_blocks) - 1)

        if scale_factor is None:
            # No scale/compression info — tiling not applicable
            return None

        # Window alignment (for window-attention models)
        window_alignment = config.get("window_size", 1)

        # Overlap: data-driven from config, fallback to trace_size // 8
        # Minimum 4 to ensure some blending
        overlap = max(4, trace_size // 8)

        # Align overlap to window_alignment
        if window_alignment > 1:
            overlap = ((overlap + window_alignment - 1) // window_alignment) * window_alignment

        logger.info(
            f"[TilingEngine] Data-driven init: trace_size={trace_size}, "
            f"scale_factor={scale_factor}, overlap={overlap}, "
            f"window_alignment={window_alignment}"
        )

        return cls(
            trace_size=trace_size,
            overlap=overlap,
            scale_factor=scale_factor,
            window_alignment=window_alignment,
        )

    def should_tile(self, input_tensor: torch.Tensor) -> bool:
        """
        Check if input requires tiling.

        Returns True when input spatial dimensions exceed tile size.
        When tile_size < trace_size (symbolic spatial graphs), this enables
        tiling even when input matches trace size — producing overlapping
        tiles that eliminate grid artifacts via accumulate-and-divide blending.

        Args:
            input_tensor: Input tensor [B, C, H, W]

        Returns:
            True if input is larger than tile size on either axis
        """
        if input_tensor.dim() != 4:
            return False

        _, _, height, width = input_tensor.shape
        return height > self.tile_size or width > self.tile_size

    def tiled_execute(
        self,
        input_tensor: torch.Tensor,
        execute_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute component on tiled input with accumulate-and-divide blending.

        Algorithm (SwinIR/Swin2SR pattern):
        1. Compute tile positions with overlap (stride < trace_size)
        2. For each position, extract tile (pad if at edge)
        3. Execute tile through component
        4. Accumulate result into output buffer, increment weight
        5. Divide output by weight → averaged where tiles overlap

        Args:
            input_tensor: Input tensor [B, C, H, W]
            execute_fn: Function to execute a single tile: fn(tile) -> result

        Returns:
            Blended output tensor [B, C_out, H*scale, W*scale]
        """
        batch_size, _, height, width = input_tensor.shape
        sf = self.scale_factor
        ts = self.tile_size

        # Compute tile positions
        h_positions, w_positions = self._compute_tile_positions(height, width)

        # Execute first tile to determine output channels
        first_h, first_w = h_positions[0], w_positions[0]
        first_tile = self._extract_tile(input_tensor, first_h, first_w)
        first_result = execute_fn(first_tile)
        out_channels = first_result.shape[1]

        # Output accumulators
        out_h = height * sf
        out_w = width * sf
        output = torch.zeros(
            batch_size, out_channels, out_h, out_w,
            device=input_tensor.device, dtype=first_result.dtype,
        )
        weight = torch.zeros(
            batch_size, 1, out_h, out_w,
            device=input_tensor.device, dtype=first_result.dtype,
        )

        # Accumulate first tile
        rh = ts * sf
        rw = ts * sf
        oh = first_h * sf
        ow = first_w * sf
        # Clamp output region to actual output size
        actual_rh = min(rh, out_h - oh)
        actual_rw = min(rw, out_w - ow)
        output[:, :, oh:oh + actual_rh, ow:ow + actual_rw] += first_result[:, :, :actual_rh, :actual_rw]
        weight[:, :, oh:oh + actual_rh, ow:ow + actual_rw] += 1

        # Process remaining tiles
        tile_count = 1
        for y in h_positions:
            for x in w_positions:
                if y == first_h and x == first_w:
                    continue  # Already processed

                tile = self._extract_tile(input_tensor, y, x)
                result = execute_fn(tile)

                oy = y * sf
                ox = x * sf
                actual_rh = min(rh, out_h - oy)
                actual_rw = min(rw, out_w - ox)
                output[:, :, oy:oy + actual_rh, ox:ox + actual_rw] += result[:, :, :actual_rh, :actual_rw]
                weight[:, :, oy:oy + actual_rh, ox:ox + actual_rw] += 1
                tile_count += 1

        logger.debug(f"[TilingEngine] Processed {tile_count} tiles, output shape {output.shape}")

        # Divide by weight — where tiles overlap, average the predictions
        return output / weight

    def _compute_tile_positions(self, height: int, width: int) -> Tuple[List[int], List[int]]:
        """
        Compute tile start positions with overlap.

        Creates a grid of positions where stride < trace_size,
        ensuring the last tile reaches the boundary.

        Args:
            height: Input height in input space
            width: Input width in input space

        Returns:
            (h_positions, w_positions) lists of start coordinates
        """
        ts = self.tile_size
        stride = self.stride

        # Height positions
        h_positions = list(range(0, max(1, height - ts + 1), stride))
        # Ensure last tile reaches boundary
        last_h = max(0, height - ts)
        if not h_positions or h_positions[-1] != last_h:
            h_positions.append(last_h)

        # Width positions
        w_positions = list(range(0, max(1, width - ts + 1), stride))
        last_w = max(0, width - ts)
        if not w_positions or w_positions[-1] != last_w:
            w_positions.append(last_w)

        return h_positions, w_positions

    def _extract_tile(self, input_tensor: torch.Tensor, y: int, x: int) -> torch.Tensor:
        """
        Extract a tile from input tensor, padding if at edge.

        Tiles are tile_size x tile_size. When graph has symbolic spatial dims,
        tile_size can be smaller than trace_size. Otherwise tiles match trace_size.

        Args:
            input_tensor: Input tensor [B, C, H, W]
            y: Start row
            x: Start column

        Returns:
            Tile tensor [B, C, tile_size, tile_size]
        """
        ts = self.tile_size
        _, _, height, width = input_tensor.shape

        y_end = min(y + ts, height)
        x_end = min(x + ts, width)
        tile = input_tensor[:, :, y:y_end, x:x_end]

        # Pad to full trace_size if at edge
        actual_h = y_end - y
        actual_w = x_end - x
        if actual_h < ts or actual_w < ts:
            pad_h = ts - actual_h
            pad_w = ts - actual_w
            tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='replicate')

        return tile

    def __repr__(self) -> str:
        return (
            f"TilingEngine("
            f"trace_size={self.trace_size}, "
            f"tile_size={self.tile_size}, "
            f"stride={self.stride}, "
            f"overlap={self.overlap}, "
            f"scale_factor={self.scale_factor}, "
            f"window_alignment={self.window_alignment})"
        )


# ============================================================================
# Op-level tiling — intercepts specific ops by op_uid to stream-execute them
# without ever materializing the full intermediate tensor that would OOM.
#
# Pattern: Prism's ActivationProfiler flags ops whose output + workspace
# exceed the per-GPU VRAM budget. For Sana 4Kpx VAE the dominant overflow
# is the upsample_nearest2d::4 → convolution::62 pair (16 GB intermediate
# fp32 tensor consumed exactly once by the conv). Same pattern at
# upsample::3 → conv::55. Op-level tiling fuses each pair via
# fused_upsample_conv2d which streams band-by-band without ever
# materializing the upsampled tensor.
#
# Lives in tiling_engine.py per R31 (single architectural location for
# tiling logic). Plumbing ride on graph_executor's existing interceptor
# mechanism (Phase 2.1, originally for KV cache injection) extended with
# fine-grained per-op_uid matching.
# ============================================================================


class OpLevelTilingPlan:
    """Compact spec emitted by Prism describing which ops in which
    component must be intercepted with what tiling strategy.

    A "fusion pair" is the most common case: an upsample whose output is
    too big for VRAM, immediately consumed by a conv. The fusion absorbs
    the upsample into the conv; the upsample interceptor returns a
    FusionUpsampleProxy and the conv interceptor reads it.
    """

    __slots__ = ("component_name", "fusion_pairs", "tiled_ops", "inplace_adds")

    def __init__(self, component_name: str):
        self.component_name = component_name
        # Each entry: (upsample_op_uid, conv_op_uid, tile_factor)
        self.fusion_pairs: List[Tuple[str, str, int]] = []
        # Each entry: (op_uid, op_type, tile_factor) — single-op tiling
        # without fusion (e.g. a standalone conv whose workspace OOMs).
        self.tiled_ops: List[Tuple[str, str, int]] = []
        # Each entry: (op_uid, reuse_input_index) — residual aten::add
        # ops where Prism's liveness analysis proved one input has its
        # last use at this op, so its buffer can be reused as output
        # (saves a 3rd allocation = output_size bytes per call). Detected
        # at register-time from the DAG; populated by
        # `_detect_inplace_add_candidates`. P-SANA-4KPX-RUNTIME 2026-05-05
        # multi-branch fusion fix vector B.
        self.inplace_adds: List[Tuple[str, int]] = []

    def add_upsample_conv_fusion(self, upsample_uid: str, conv_uid: str,
                                  tile_factor: int) -> None:
        self.fusion_pairs.append((upsample_uid, conv_uid, max(1, int(tile_factor))))

    def add_tiled_op(self, op_uid: str, op_type: str, tile_factor: int) -> None:
        self.tiled_ops.append((op_uid, op_type, max(1, int(tile_factor))))

    def add_inplace_add(self, op_uid: str, reuse_input_index: int) -> None:
        self.inplace_adds.append((op_uid, int(reuse_input_index)))

    def is_empty(self) -> bool:
        return (not self.fusion_pairs and not self.tiled_ops
                and not self.inplace_adds)

    def __repr__(self) -> str:
        return (
            f"OpLevelTilingPlan(comp={self.component_name}, "
            f"fusion_pairs={len(self.fusion_pairs)}, "
            f"tiled_ops={len(self.tiled_ops)})"
        )


class OpLevelTilingEngine:
    """Builds and registers per-op_uid interceptors on a GraphExecutor.

    Constructed by Prism via from_op_level_constraint() once the strategy
    cascade has decided that op-level tiling is necessary on a given
    component. Wires interceptors via graph_executor.register_op_uid_interceptors
    which is hot-swap-safe (no recompilation of the CompiledSequence).
    """

    def __init__(self, plan: OpLevelTilingPlan):
        self.plan = plan

    @classmethod
    def from_op_level_constraint(cls, plan: OpLevelTilingPlan) -> Optional["OpLevelTilingEngine"]:
        """Factory — returns None if the plan is empty (no overflow ops)."""
        if plan.is_empty():
            return None
        return cls(plan)

    @staticmethod
    def _detect_pixel_shuffle_broadcast_chains(graph_executor):
        """Detect the expand→clone→view→pixel_shuffle pattern in the DAG.

        Each detected chain is a 4-tuple: (expand_uid, clone_uid, view_uid,
        pixel_shuffle_uid). Returns a list of dicts with the uids and
        broadcast/shape metadata for register_into_graph_executor to wire
        proxy interceptors on clone, view, and pixel_shuffle.

        Pattern criteria (all must hold):
          1. expand op produces a tensor with at least one stride-0 dim
             (= broadcast view, identifiable from input_shape vs
             output_shape comparison: the dim that grew from size N>=1 to
             size > N is the broadcast dim).
          2. expand's only consumer is an aten.clone op.
          3. clone's only consumer is an aten.view op.
          4. view's only consumer is an aten.pixel_shuffle op.
          5. The size of the broadcast dim must equal the pixel_shuffle
             upscale_factor (typical DC-AE pattern: bcast=r=2).

        On Sana 4Kpx VAE this matches 1 chain (exec[701-704]:
        expand::41 → clone::5 → view::95 → pixel_shuffle::4).
        Universal: any model with this decomposition pattern benefits.
        """
        dag = getattr(graph_executor, '_dag', None)
        if dag is None:
            return []
        ops = dag.get("ops", {})
        if isinstance(ops, list):
            ops_by_uid = {o["op_uid"]: o for o in ops}
        else:
            ops_by_uid = ops
        order = dag.get("execution_order", [])

        # Build consumer map (tid -> list of op_uid that reads it)
        def _collect_arg_tids(arg, out_set):
            if isinstance(arg, dict):
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

        consumers: Dict[str, List[str]] = {}
        for op_uid in order:
            op = ops_by_uid.get(op_uid)
            if op is None:
                continue
            seen: set = set()
            attrs = op.get("attributes", {})
            for arg in attrs.get("args", []):
                _collect_arg_tids(arg, seen)
            for arg in attrs.get("kwargs", {}).values():
                _collect_arg_tids(arg, seen)
            for tid in seen:
                consumers.setdefault(tid, []).append(op_uid)

        chains: List[Dict[str, Any]] = []
        for op_uid in order:
            op = ops_by_uid.get(op_uid)
            if op is None or op.get("op_type") != "aten::expand":
                continue
            expand_in_shapes = op.get("input_shapes", [])
            expand_out_shapes = op.get("output_shapes", [])
            if not expand_in_shapes or not expand_out_shapes:
                continue
            in_sh = expand_in_shapes[0]
            out_sh = expand_out_shapes[0]
            if len(in_sh) != len(out_sh):
                continue
            # Find the dim that was broadcast (in_sh[d]=1, out_sh[d]>1)
            broadcast_dim = -1
            broadcast_factor = 1
            for d in range(len(in_sh)):
                if in_sh[d] == 1 and out_sh[d] > 1:
                    if broadcast_dim != -1:
                        # Multiple broadcast dims — skip (rare, not our pattern)
                        broadcast_dim = -1
                        break
                    broadcast_dim = d
                    broadcast_factor = out_sh[d]
            if broadcast_dim == -1:
                continue
            expand_out_tid = op.get("output_tensor_ids", [None])[0]
            if expand_out_tid is None:
                continue
            expand_consumers = consumers.get(expand_out_tid, [])
            if len(expand_consumers) != 1:
                continue
            clone_uid = expand_consumers[0]
            clone_op = ops_by_uid.get(clone_uid)
            if clone_op is None or clone_op.get("op_type") != "aten::clone":
                continue
            clone_out_tid = clone_op.get("output_tensor_ids", [None])[0]
            if clone_out_tid is None:
                continue
            clone_consumers = consumers.get(clone_out_tid, [])
            if len(clone_consumers) != 1:
                continue
            view_uid = clone_consumers[0]
            view_op = ops_by_uid.get(view_uid)
            if view_op is None or view_op.get("op_type") != "aten::view":
                continue
            view_out_tid = view_op.get("output_tensor_ids", [None])[0]
            if view_out_tid is None:
                continue
            view_consumers = consumers.get(view_out_tid, [])
            if len(view_consumers) != 1:
                continue
            ps_uid = view_consumers[0]
            ps_op = ops_by_uid.get(ps_uid)
            if ps_op is None or ps_op.get("op_type") != "aten::pixel_shuffle":
                continue
            # Extract upscale_factor from pixel_shuffle args (second arg)
            ps_attrs = ps_op.get("attributes", {})
            ps_args = ps_attrs.get("args", [])
            upscale_factor = None
            if len(ps_args) >= 2:
                arg2 = ps_args[1]
                if isinstance(arg2, dict):
                    upscale_factor = arg2.get("value")
            if upscale_factor is None:
                continue
            # Optional sanity: bcast == upscale_factor in the typical pattern.
            # Don't enforce strictly — log if mismatch but still register.
            chains.append({
                "expand_uid": op_uid,
                "clone_uid": clone_uid,
                "view_uid": view_uid,
                "pixel_shuffle_uid": ps_uid,
                "bcast_dim": int(broadcast_dim),
                "bcast_factor": int(broadcast_factor),
                "post_clone_shape": tuple(clone_op.get("output_shapes", [[]])[0]),
                "post_view_shape": tuple(view_op.get("output_shapes", [[]])[0]),
                "upscale_factor": int(upscale_factor),
            })
        return chains

    @staticmethod
    def _detect_inplace_add_candidates(
        graph_executor, threshold_bytes: int = 1024 * 1024 * 1024,
    ) -> "List[Tuple[str, int]]":
        """Scan the GraphExecutor's DAG for residual aten::add ops where
        liveness analysis proves at least one input has its last use at
        this op. Returns a list of `(op_uid, reuse_input_index)` for adds
        whose output exceeds `threshold_bytes` (default 1 GiB).

        Detection is purely DAG-static (no Prism profile / no runtime
        info): builds a consumer map and checks whether each input's only
        downstream consumer is the add itself AND the input is not a
        graph output. This is the same liveness logic used by triton
        sequence's `_compute_liveness` and `_run_triton_sequential`'s
        `dead_at_op`.

        Universal — applies to any model whose decoder has multi-branch
        residual merges (DC-AE, ResNet residual blocks at high spatial
        resolution, etc.). On Sana 4Kpx VAE, returns 26 candidates
        (8 above 4 GiB at the 4096×4096 res, 4 above 4 GiB at 2048×2048).
        """
        dag = getattr(graph_executor, '_dag', None)
        if dag is None:
            return []
        ops = dag.get("ops", {})
        if isinstance(ops, list):
            ops_by_uid = {o["op_uid"]: o for o in ops}
        else:
            ops_by_uid = ops
        order = dag.get("execution_order", [])
        output_ids = set(dag.get("output_tensor_ids", []))

        # Build consumer map: tensor_id -> [op_uid that reads it]
        def _collect_arg_tids(arg, out_set):
            if isinstance(arg, dict):
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

        consumers: Dict[str, List[str]] = {}
        for op_uid in order:
            op = ops_by_uid.get(op_uid)
            if op is None:
                continue
            attrs = op.get("attributes", {})
            seen: set = set()
            for arg in attrs.get("args", []):
                _collect_arg_tids(arg, seen)
            for arg in attrs.get("kwargs", {}).values():
                _collect_arg_tids(arg, seen)
            for tid in seen:
                consumers.setdefault(tid, []).append(op_uid)

        candidates: List[Tuple[str, int]] = []
        for op_uid in order:
            op = ops_by_uid.get(op_uid)
            if op is None or op.get("op_type") != "aten::add":
                continue
            inp_tids = op.get("input_tensor_ids", [])
            ishapes = op.get("input_shapes", [])
            oshapes = op.get("output_shapes", [])
            if len(inp_tids) < 2 or len(ishapes) < 2 or not oshapes:
                continue
            sh_a = ishapes[0]
            sh_b = ishapes[1]
            sh_o = oshapes[0]
            # Residual pattern: identical 4D input shapes.
            if sh_a != sh_b or len(sh_a) != 4:
                continue
            # Output size (use fp32 as conservative estimate; runtime
            # dtype may be smaller but we'd rather under-trigger than
            # mis-trigger). The threshold is intentionally generous.
            elems = 1
            for d in sh_o:
                elems *= max(1, int(d))
            out_bytes = elems * 4  # fp32
            if out_bytes < threshold_bytes:
                continue
            # Liveness check: is at least one input's last use this op?
            a_consumers = consumers.get(inp_tids[0], [])
            b_consumers = consumers.get(inp_tids[1], [])
            a_last = (len(a_consumers) == 1
                      and a_consumers[0] == op_uid
                      and inp_tids[0] not in output_ids)
            b_last = (len(b_consumers) == 1
                      and b_consumers[0] == op_uid
                      and inp_tids[1] not in output_ids)
            if a_last:
                candidates.append((op_uid, 0))
            elif b_last:
                candidates.append((op_uid, 1))
        return candidates

    def register_into_graph_executor(self, graph_executor) -> int:
        """Wire all op_uid interceptors. Returns the count registered.

        Called by RuntimeExecutor right after the component's GraphExecutor
        is created, BEFORE the first execute() so interceptors are in
        place when CompiledSequence is compiled.
        """
        from neurobrix.kernels.ops.fused_upsample_conv import (
            make_upsample_proxy_interceptor,
            fused_upsample_conv2d,
        )

        # Auto-detect in-place residual add candidates from the DAG. Only
        # populated for components where Prism already triggered op-level
        # tiling (high-VRAM-pressure models — Sana 4Kpx and similar);
        # cheaper models bypass this entire engine. The detection is
        # universal (any multi-branch residual pattern; not Sana-specific).
        if not self.plan.inplace_adds:
            self.plan.inplace_adds = OpLevelTilingEngine._detect_inplace_add_candidates(
                graph_executor)

        interceptors: Dict[str, Callable] = {}

        # Upsample → conv fusion pairs
        for upsample_uid, conv_uid, tile_factor in self.plan.fusion_pairs:
            # Upsample interceptor: returns a FusionUpsampleProxy (no compute)
            interceptors[upsample_uid] = make_upsample_proxy_interceptor()

            # Conv interceptor: detects proxy, runs band-streaming fused kernel
            tf = tile_factor

            def make_conv_interceptor(_tile_factor):
                def _conv(input_or_proxy, weight, bias=None,
                          stride=1, padding=0, dilation=1,
                          transposed=False, output_padding=0, groups=1, *args, **kwargs):
                    return fused_upsample_conv2d(
                        input_or_proxy, weight, bias,
                        stride=stride, padding=padding, dilation=dilation,
                        transposed=transposed, output_padding=output_padding,
                        groups=groups, tile_factor=_tile_factor,
                    )
                return _conv

            interceptors[conv_uid] = make_conv_interceptor(tf)

        # Standalone tiled ops (no fusion — input already materialized but
        # the op's workspace alone would OOM)
        from neurobrix.kernels.ops.fused_upsample_conv import (
            tiled_conv2d_spatial, tiled_rms_norm_spatial,
        )
        for op_uid, op_type, tile_factor in self.plan.tiled_ops:
            cl = op_type.split("::")[-1]
            if cl in ("convolution", "conv2d", "_convolution"):
                tf = tile_factor

                def make_tiled_conv(_tile_factor):
                    def _tiled(input_tensor, weight, bias=None,
                               stride=1, padding=0, dilation=1,
                               transposed=False, output_padding=0, groups=1,
                               *args, **kwargs):
                        return tiled_conv2d_spatial(
                            input_tensor, weight, bias,
                            stride=stride, padding=padding, dilation=dilation,
                            transposed=transposed, output_padding=output_padding,
                            groups=groups, tile_factor=_tile_factor,
                        )
                    return _tiled

                interceptors[op_uid] = make_tiled_conv(tf)
            elif cl == "rms_norm":
                tf = tile_factor

                def make_tiled_rms(_tile_factor):
                    def _tiled(x, weight=None, eps=1e-6, *args, **kwargs):
                        return tiled_rms_norm_spatial(
                            x, weight, eps, tile_factor=_tile_factor,
                        )
                    return _tiled
                interceptors[op_uid] = make_tiled_rms(tf)
            else:
                logger.debug(
                    f"[OpLevelTilingEngine] No tiled implementation for "
                    f"op_type={op_type} (uid={op_uid}); skipping."
                )

        # In-place residual adds (multi-branch fusion fix vector B).
        # See `_detect_inplace_add_candidates` for the safety contract.
        # Dual-backend: routes to NBX in-place kernel for triton modes,
        # to torch's tensor.add_() for compiled/sequential modes (so
        # compiled mode keeps its perf path AND benefits from in-place
        # semantics — torch CachingAllocator already handles the alloc
        # pattern efficiently but in-place still saves a transient).
        # NBX_DISABLE_INPLACE_ADD=1 skips registration to bisect whether
        # in-place add is the source of an unidentified leak.
        import os as _os_iadd
        if _os_iadd.environ.get("NBX_DISABLE_INPLACE_ADD", "0") == "1":
            self.plan.inplace_adds = []
        if self.plan.inplace_adds:
            from neurobrix.kernels.wrappers import add_inplace_nbx
            from neurobrix.kernels.nbx_tensor import NBXTensor

            def make_inplace_add_interceptor(_reuse_index):
                def _inplace_add(a, b, alpha=1.0, *args, **kwargs):
                    target = a if _reuse_index == 0 else b
                    other = b if _reuse_index == 0 else a
                    if isinstance(target, NBXTensor):
                        return add_inplace_nbx(target, other,
                                               alpha=float(alpha))
                    # Torch path (compiled / sequential modes).
                    import torch
                    if isinstance(target, torch.Tensor):
                        # Cast `other` to target's dtype if needed (mirror
                        # the NBX wrapper's behaviour). If target is the
                        # narrower precision, fall back to non-in-place
                        # `torch.add` so the upcast happens in the new
                        # buffer instead of corrupting target.
                        if hasattr(other, 'dtype') and other.dtype != target.dtype:
                            try:
                                # Wider-or-equal dtype check via a tiny
                                # promotion: if torch's promotion picks
                                # target's dtype, in-place is safe.
                                promo = torch.promote_types(target.dtype,
                                                            other.dtype)
                                if promo == target.dtype:
                                    other = other.to(target.dtype)
                                else:
                                    return torch.add(target, other,
                                                     alpha=float(alpha))
                            except Exception:
                                return torch.add(target, other,
                                                 alpha=float(alpha))
                        target.add_(other, alpha=float(alpha))
                        return target
                    # Unknown backend — fall back to plain `+`.
                    return target + other * float(alpha)
                _inplace_add.self_manages_dtype = True
                return _inplace_add

            for op_uid, reuse_index in self.plan.inplace_adds:
                interceptors[op_uid] = make_inplace_add_interceptor(reuse_index)

        # Pixel-shuffle broadcast-aware fusion (P-SANA-4KPX-RUNTIME Fix F2,
        # Approche C). Detect `expand -> clone -> view -> pixel_shuffle`
        # statically. The expand output is already an NBX stride-0 view
        # (NBXTensor.expand line 1581: stride is 0 on the broadcast dim,
        # data_ptr shared with the pre-expand tensor). The clone interceptor
        # wraps that view directly into a BroadcastClonePyroxy — no _base
        # walk, no DAG runtime lookup. The view interceptor is a pass-through.
        # The pixel_shuffle interceptor reads via a stride-aware kernel that
        # exploits stride_v_b == 0 to alias broadcast indices. Universal:
        # any DC-AE-like decomposition benefits.
        chain_specs = OpLevelTilingEngine._detect_pixel_shuffle_broadcast_chains(
            graph_executor)
        if chain_specs:
            from neurobrix.kernels.ops.fused_upsample_conv import (
                BroadcastClonePyroxy,
            )
            from neurobrix.kernels.nbx_tensor import NBXTensor

            def make_clone_proxy_interceptor(_bcast_dim, _bcast_factor,
                                             _post_clone_shape,
                                             _post_view_shape):
                def _clone_proxy(input_tensor, *_args, **_kwargs):
                    # NBX path: input_tensor IS the expand stride-0 view —
                    # wrap it directly. No materialization, no _base walk.
                    if isinstance(input_tensor, NBXTensor):
                        return BroadcastClonePyroxy(
                            input_tensor, _bcast_dim, _bcast_factor,
                            _post_clone_shape, _post_view_shape)
                    # Torch path (compiled / sequential): mirror the original
                    # graph's `clone(..., memory_format=torch.contiguous_format)`
                    # semantic explicitly so the downstream view always sees
                    # a strictly contiguous tensor. torch CachingAllocator
                    # handles the materialization without leaking.
                    import torch as _torch
                    return input_tensor.clone(
                        memory_format=_torch.contiguous_format)
                setattr(_clone_proxy, "self_manages_dtype", True)
                return _clone_proxy

            def make_view_proxy_interceptor():
                def _view_proxy(input_tensor, *args, **_kwargs):
                    # NBX path: forward the proxy unchanged. The pixel_shuffle
                    # interceptor reads bcast_view + bcast_dim/factor from the
                    # proxy, so the intermediate view is logically transparent.
                    if isinstance(input_tensor, BroadcastClonePyroxy):
                        return input_tensor
                    # Torch path: use .reshape() instead of .view() — semantically
                    # equivalent on contiguous tensors, but tolerates the case
                    # where the upstream `clone` was a no-op interceptor or
                    # otherwise left the input non-contiguous. PyTorch's own
                    # error message ("Use .reshape(...) instead.") is exactly
                    # this advice.
                    if args:
                        first = args[0]
                        if isinstance(first, (list, tuple)):
                            return input_tensor.reshape(*first)
                        return input_tensor.reshape(*args)
                    return input_tensor.reshape(input_tensor.shape)
                setattr(_view_proxy, "self_manages_dtype", True)
                return _view_proxy

            def make_pixel_shuffle_proxy_interceptor():
                from neurobrix.kernels.wrappers import (
                    pixel_shuffle_wrapper as nbx_pixel_shuffle,
                )

                def _pixel_shuffle_proxy(input_tensor, upscale_factor,
                                         *_args, **_kwargs):
                    # NBX path (proxy or plain NBXTensor): the wrapper
                    # auto-routes proxy -> _pixel_shuffle_broadcast_aware,
                    # plain NBXTensor -> standard kernel.
                    if isinstance(input_tensor, (BroadcastClonePyroxy, NBXTensor)):
                        return nbx_pixel_shuffle(input_tensor, int(upscale_factor))
                    # Torch path: native pixel_shuffle (compiled mode).
                    import torch.nn.functional as F
                    return F.pixel_shuffle(input_tensor, int(upscale_factor))
                setattr(_pixel_shuffle_proxy, "self_manages_dtype", True)
                return _pixel_shuffle_proxy

            for spec in chain_specs:
                clone_uid = spec['clone_uid']
                view_uid = spec['view_uid']
                ps_uid = spec['pixel_shuffle_uid']
                bcast_dim = spec['bcast_dim']
                bcast_factor = spec['bcast_factor']
                post_clone_shape = spec['post_clone_shape']
                post_view_shape = spec['post_view_shape']
                interceptors[clone_uid] = make_clone_proxy_interceptor(
                    bcast_dim, bcast_factor,
                    post_clone_shape, post_view_shape)
                interceptors[view_uid] = make_view_proxy_interceptor()
                interceptors[ps_uid] = make_pixel_shuffle_proxy_interceptor()
                logger.info(
                    f"[OpLevelTilingEngine] F2 pixel_shuffle broadcast-aware "
                    f"chain wired: {spec['expand_uid']} -> {clone_uid} -> "
                    f"{view_uid} -> {ps_uid} (bcast_dim={bcast_dim} "
                    f"bcast={bcast_factor} r={spec['upscale_factor']})")

        if not interceptors:
            return 0
        graph_executor.register_op_uid_interceptors(interceptors)
        logger.info(
            f"[OpLevelTilingEngine] Registered {len(interceptors)} op_uid "
            f"interceptors on component '{self.plan.component_name}': "
            f"fusion_pairs={len(self.plan.fusion_pairs)} "
            f"tiled_ops={len(self.plan.tiled_ops)} "
            f"inplace_adds={len(self.plan.inplace_adds)}"
        )
        return len(interceptors)

    def __repr__(self) -> str:
        return f"OpLevelTilingEngine({self.plan})"
