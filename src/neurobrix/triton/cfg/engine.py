"""
TritonCFGEngine — Classifier-Free Guidance for triton mode.

Ported from core/cfg/engine.py. Complete separation:
native mode uses torch tensors, triton mode uses NBXTensor throughout.

No torch imports in this file.

Interface:
    engine = TritonCFGEngine.from_topology(ctx)
    if engine.is_enabled:
        output = engine.execute_with_cfg(comp_name, state, timestep)
"""

import math
import numpy as np
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype
from neurobrix.core.runtime.debug import DEBUG


def _ensure_nbx(tensor) -> NBXTensor:
    """Convert torch.Tensor → NBXTensor at the boundary.

    CPU tensors: numpy → from_numpy (H2D via cudaMemcpy, zero torch).
    CUDA tensors: from_raw (pointer wrap, zero copy).
    """
    if isinstance(tensor, NBXTensor) or tensor is None:
        return tensor
    if hasattr(tensor, 'data_ptr'):
        if hasattr(tensor, 'is_cuda') and not tensor.is_cuda:
            # CPU → CUDA via numpy path (zero torch)
            DeviceAllocator.set_device(DeviceAllocator.get_device())
            arr = tensor.detach().numpy()
            return NBXTensor.from_numpy(arr)
        didx = tensor.device.index if hasattr(tensor.device, 'index') and tensor.device.index is not None else 0
        return NBXTensor.from_raw(
            tensor.data_ptr(), tuple(tensor.shape),
            parse_dtype(str(tensor.dtype)), 'cuda',
            owns_data=False, device_idx=didx, base=tensor)
    return tensor

if TYPE_CHECKING:
    from neurobrix.core.flow.base import FlowContext


class CFGMode(Enum):
    """CFG execution mode."""
    DISABLED = auto()     # No CFG (scale <= threshold)
    BATCHED = auto()      # Batch [cond, uncond] together (default)
    SEQUENTIAL = auto()   # Run cond/uncond separately (TP strategy)


class TritonCFGEngine:
    """
    Unified Classifier-Free Guidance engine — triton mode.

    Handles:
    - Detection: guidance_scale vs threshold from config cascade
    - Batching: [uncond, cond] concatenation for batch=2 execution
    - Sequential: 2x batch=1 for TP strategies
    - Application: uncond + scale * (cond - uncond)
    - Variable management: save/restore encoder states

    ZERO TORCH: All tensor ops via NBXTensor.
    """

    def __init__(
        self,
        ctx: 'FlowContext',
        execute_component_fn: Callable[[str, str], Any],
        extract_primary_output_fn: Callable[[str, Any], Any],
        guidance_scale: float = 7.5,
        mode: CFGMode = CFGMode.BATCHED,
    ):
        self._ctx = ctx
        self._execute_component = execute_component_fn
        self._extract_primary_output = extract_primary_output_fn
        self.guidance_scale = guidance_scale
        self.mode = mode

    @property
    def is_enabled(self) -> bool:
        """Check if CFG is enabled (guidance_scale above threshold)."""
        return self.mode != CFGMode.DISABLED

    @classmethod
    def from_topology(
        cls,
        ctx: 'FlowContext',
        execute_component_fn: Callable[[str, str], Any],
        extract_primary_output_fn: Callable[[str, Any], Any],
    ) -> 'TritonCFGEngine':
        """
        Create TritonCFGEngine from topology config cascade.

        Priority for guidance_scale:
        1. CLI global.guidance_scale
        2. topology.extracted_values._global.guidance_scale
        3. Family config default

        Detects guidance-embedding models (Flux) and disables batch-2 CFG.
        """
        from neurobrix.core.config import get_family_defaults, get_family_config

        family = ctx.pkg.manifest.get("family")
        if family is None:
            raise RuntimeError(
                "ZERO FALLBACK: 'family' missing in manifest.\n"
                "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )
        family_defaults = get_family_defaults(family)
        family_config = get_family_config(family)
        diffusion_defaults = family_config.get("diffusion", {}).get("defaults", {})

        # Resolve guidance_scale from cascade
        try:
            guidance_scale = ctx.variable_resolver.get("global.guidance_scale")
        except KeyError:
            guidance_scale = None

        if guidance_scale is None:
            extracted_values = ctx.pkg.topology.get("extracted_values", {})
            model_globals = extracted_values.get("_global", {})
            guidance_scale = model_globals.get("guidance_scale")

        if guidance_scale is None:
            guidance_scale = ctx.pkg.defaults["guidance_scale"]

        # Get threshold
        cfg_threshold = family_defaults.get("cfg_threshold")
        if cfg_threshold is None:
            cfg_threshold = diffusion_defaults.get("cfg_threshold", 1.0)

        do_cfg = guidance_scale > cfg_threshold

        # Detect guidance-embedding models (Flux-style)
        if do_cfg:
            loop_info = ctx.pkg.topology.get("flow", {}).get("loop", {})
            loop_components = loop_info.get("components", [])
            components_data = ctx.pkg.topology.get("components", {})
            for comp_name in loop_components:
                comp_inputs = components_data.get(comp_name, {}).get("interface", {}).get("inputs", [])
                if "guidance" in comp_inputs:
                    # Model uses guidance embedding — inject as NBXTensor, skip batch-2 CFG
                    guidance_arr = np.array([guidance_scale], dtype=np.float32)
                    guidance_tensor = NBXTensor.from_numpy(guidance_arr)
                    ctx.variable_resolver.set(
                        "global.guidance",
                        guidance_tensor
                    )
                    # Synthesize position IDs if needed
                    comp_shapes = components_data.get(comp_name, {}).get("shapes", {})
                    _synthesize_position_ids(ctx, comp_inputs, comp_shapes)

                    do_cfg = False
                    break

        mode = CFGMode.DISABLED if not do_cfg else CFGMode.BATCHED

        engine = cls(
            ctx=ctx,
            execute_component_fn=execute_component_fn,
            extract_primary_output_fn=extract_primary_output_fn,
            guidance_scale=guidance_scale,
            mode=mode,
        )

        return engine

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def execute_component_with_cfg(
        self,
        comp_name: str,
        current_state,
        timestep,
        guidance_scale: float,
        encoder_dtype: Any,
    ) -> Dict[str, NBXTensor]:
        """Execute component with CFG (batched or sequential)."""
        # Convert torch.Tensor → NBXTensor at boundary
        current_state = _ensure_nbx(current_state)
        timestep = _ensure_nbx(timestep)

        if self._should_use_sequential_cfg(comp_name):
            return self._execute_sequential_cfg(
                comp_name, current_state, timestep, guidance_scale
            )
        return self._execute_batched_cfg(
            comp_name, current_state, timestep, guidance_scale, encoder_dtype
        )

    # =========================================================================
    # BATCHED CFG
    # =========================================================================

    def _execute_batched_cfg(
        self,
        comp_name: str,
        current_state: NBXTensor,
        timestep: NBXTensor,
        guidance_scale: float,
        encoder_dtype: Any,
    ) -> Dict[str, NBXTensor]:
        """Batched CFG: [uncond, cond] in single batch=2 forward pass."""
        encoder_key = self._get_encoder_hidden_states_key()
        if encoder_key is None:
            raise RuntimeError(
                "ZERO FALLBACK: Cannot determine encoder_hidden_states variable.\n"
                "Check topology connections from pre_loop to loop components."
            )

        encoder_comp = encoder_key.split(".")[0]
        negative_key = f"{encoder_comp}.negative_hidden_state"

        pos_hidden = _ensure_nbx(self._ctx.variable_resolver.get(encoder_key))
        neg_hidden = _ensure_nbx(self._ctx.variable_resolver.get(negative_key))

        if neg_hidden is None:
            raise RuntimeError(
                "ZERO FALLBACK: CFG enabled but negative embeddings not found.\n"
                f"Expected '{negative_key}' — ensure pre_loop ran with CFG negative encoding."
            )

        state_key = self._ctx.pkg.topology.get("flow", {}).get("loop", {}).get("state_variable")
        if state_key is None:
            raise RuntimeError("ZERO FALLBACK: 'state_variable' not defined in topology.flow.loop.")

        # Get mask
        suffix = ""
        if encoder_comp != "text_encoder" and encoder_comp.startswith("text_encoder_"):
            suffix = "_" + encoder_comp.split("text_encoder_")[1]
        mask_var = f"global.attention_mask{suffix}"
        pos_mask = _ensure_nbx(self._ctx.variable_resolver.get(mask_var))
        if pos_mask is None:
            pos_mask = _ensure_nbx(self._ctx.variable_resolver.get("global.attention_mask"))

        if pos_hidden.shape[1] != neg_hidden.shape[1]:
            raise RuntimeError(
                f"ZERO FALLBACK: pos/neg seq_len mismatch: pos={pos_hidden.shape[1]}, neg={neg_hidden.shape[1]}."
            )

        # Save originals
        orig_hidden = _ensure_nbx(self._ctx.variable_resolver.get(encoder_key))
        orig_state = current_state
        orig_timestep = _ensure_nbx(self._ctx.variable_resolver.loop_state.get(self._ctx.loop_id))

        # Batch inputs: [uncond, cond] — align dtypes before cat
        if hasattr(neg_hidden, '_dtype') and hasattr(pos_hidden, '_dtype') and neg_hidden._dtype != pos_hidden._dtype:
            neg_hidden = neg_hidden.to(pos_hidden._dtype)
        batched_hidden = NBXTensor.cat([neg_hidden, pos_hidden], dim=0)
        if hasattr(current_state, '_dtype'):
            batched_state = NBXTensor.cat([current_state, current_state], dim=0)
        else:
            batched_state = current_state  # scalar fallback

        # Batch mask
        batched_mask = None
        if pos_mask is not None and pos_mask.shape[-1] == pos_hidden.shape[1]:
            neg_mask = _ensure_nbx(self._ctx.variable_resolver.get(f"{encoder_comp}.negative_attention_mask", None))
            if neg_mask is None or neg_mask.shape[-1] != neg_hidden.shape[1]:
                # Create ones tensor via NBXTensor
                ones_shape = (neg_hidden.shape[0], neg_hidden.shape[1])
                neg_mask = _create_ones_tensor(ones_shape, pos_mask._dtype, pos_mask._device)
            elif neg_mask._dtype != pos_mask._dtype:
                neg_mask = neg_mask.to(pos_mask._dtype)
            batched_mask = NBXTensor.cat([neg_mask, pos_mask], dim=0)

        # Cast to encoder dtype
        if isinstance(encoder_dtype, NBXDtype):
            if batched_state._dtype != encoder_dtype:
                batched_state = batched_state.to(encoder_dtype)
        else:
            # encoder_dtype may be a torch dtype or string — convert via parse
            encoder_dtype_str = str(encoder_dtype).replace("torch.", "")
            from neurobrix.kernels.nbx_tensor import parse_dtype
            target_dt = parse_dtype(encoder_dtype_str)
            if batched_state._dtype != target_dt:
                batched_state = batched_state.to(target_dt)

        # Batch timestep
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        batched_timestep = timestep.expand(2)
        if batched_timestep._dtype != batched_state._dtype:
            batched_timestep = batched_timestep.to(batched_state._dtype)

        # Set batched inputs
        self._ctx.variable_resolver.set(encoder_key, batched_hidden)
        self._ctx.variable_resolver.set(state_key, batched_state)
        if batched_mask is not None:
            self._ctx.variable_resolver.set("global.encoder_attention_mask", batched_mask)
        self._ctx.variable_resolver.loop_state[self._ctx.loop_id] = batched_timestep

        # Execute once with batch=2
        output = self._execute_component(comp_name, "loop_cfg_batched")
        noise_pred_batched = self._extract_primary_output(comp_name, output)

        # CFG in float32 for stability
        if noise_pred_batched._dtype != NBXDtype.float32:
            noise_pred_batched = noise_pred_batched.to(NBXDtype.float32)

        # Split: chunk(2, dim=0) → narrow(0, 0, half) and narrow(0, half, half)
        half = noise_pred_batched.shape[0] // 2
        noise_pred_uncond = noise_pred_batched.narrow(0, 0, half)
        noise_pred_cond = noise_pred_batched.narrow(0, half, half)

        # CFG formula: uncond + scale * (cond - uncond)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if DEBUG:
            print(f"[CFG] guidance={guidance_scale}: uncond mean (triton), "
                  f"cond mean (triton), final mean (triton)")

        # Restore originals
        self._ctx.variable_resolver.set(encoder_key, orig_hidden)
        self._ctx.variable_resolver.set(state_key, orig_state)
        if batched_mask is not None:
            self._ctx.variable_resolver.set("global.encoder_attention_mask", pos_mask)
        self._ctx.variable_resolver.loop_state[self._ctx.loop_id] = orig_timestep

        # Persist the post-CFG noise_pred as the component's effective
        # output. Without this, RuntimeExecutor.store_component_outputs
        # (core/runtime/resolution/output_extractor.py:115) has already
        # written the RAW batch=2 transformer forward-pass output into
        # `variable_resolver.resolved[f"{comp_name}.output_0"]` during
        # the batched `_execute_component` call above, and that raw
        # batched tensor will be what any downstream connection
        # (typically `transformer.output_0 -> vae.z` in diffusion
        # topologies) resolves to — propagating the CFG batch all the
        # way to the VAE decoder and yielding `(2*B, C, H, W)` images
        # (Sana: incoherent; PixArt-Alpha/Sigma: silently doubled,
        # saved by uncond/cond convergence at low timesteps but still
        # wrong). Overwriting with `noise_pred` puts the effective
        # batch=1 guidance output in place for post-loop resolution.
        self._ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = noise_pred

        return {"output_0": noise_pred}

    # =========================================================================
    # SEQUENTIAL CFG (TP strategy)
    # =========================================================================

    def _execute_sequential_cfg(
        self,
        comp_name: str,
        current_state: NBXTensor,
        timestep: NBXTensor,
        guidance_scale: float,
    ) -> Dict[str, NBXTensor]:
        """Sequential CFG: 2x batch=1 passes (for TP strategy)."""
        encoder_key = self._get_encoder_hidden_states_key()
        if encoder_key is None:
            raise RuntimeError(
                "ZERO FALLBACK: Cannot determine encoder_hidden_states variable."
            )

        encoder_comp = encoder_key.split(".")[0]
        negative_key = f"{encoder_comp}.negative_hidden_state"
        state_key = self._ctx.pkg.topology.get("flow", {}).get("loop", {}).get("state_variable")
        if state_key is None:
            raise RuntimeError("ZERO FALLBACK: 'state_variable' not defined in topology.flow.loop.")

        pos_hidden = _ensure_nbx(self._ctx.variable_resolver.get(encoder_key))
        neg_hidden = _ensure_nbx(self._ctx.variable_resolver.get(negative_key))
        if neg_hidden is None:
            raise RuntimeError(
                f"ZERO FALLBACK: CFG negative embeddings not found: '{negative_key}'."
            )

        pos_mask = _ensure_nbx(self._ctx.variable_resolver.get("global.attention_mask"))
        neg_mask = _ensure_nbx(self._ctx.variable_resolver.get(f"{encoder_comp}.negative_attention_mask", None))
        if neg_mask is None or neg_mask.shape[-1] != neg_hidden.shape[1]:
            ones_shape = (neg_hidden.shape[0], neg_hidden.shape[1])
            neg_mask = _create_ones_tensor(ones_shape, pos_mask._dtype, pos_mask._device)
        elif neg_mask._dtype != pos_mask._dtype:
            neg_mask = neg_mask.to(pos_mask._dtype)

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        # Pass 1: unconditional
        self._ctx.variable_resolver.set(encoder_key, neg_hidden)
        self._ctx.variable_resolver.set(state_key, current_state)
        self._ctx.variable_resolver.set("global.encoder_attention_mask", neg_mask)
        self._ctx.variable_resolver.loop_state[self._ctx.loop_id] = timestep

        output_uncond = self._execute_component(comp_name, "cfg_uncond")
        noise_pred_uncond = self._extract_primary_output(comp_name, output_uncond)

        # Pass 2: conditional
        self._ctx.variable_resolver.set(encoder_key, pos_hidden)
        self._ctx.variable_resolver.set("global.encoder_attention_mask", pos_mask)

        output_cond = self._execute_component(comp_name, "cfg_cond")
        noise_pred_cond = self._extract_primary_output(comp_name, output_cond)

        # Apply CFG
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if DEBUG:
            print(f"[CFG] Sequential guidance={guidance_scale}: "
                  f"uncond (triton), cond (triton)")

        self._ctx.variable_resolver.set(encoder_key, pos_hidden)

        # Persist the post-CFG noise_pred as the component's effective
        # output for the same reason as in `_execute_batched_cfg` above:
        # `RuntimeExecutor.store_component_outputs` overwrote
        # `{comp_name}.output_0` on each of the two sequential passes
        # (last write wins → noise_pred_cond is what's there now),
        # which would leak the unguided conditional output to any
        # downstream connection resolving `transformer.output_0`
        # (typically the VAE). Overwrite with the CFG-applied
        # `noise_pred` so post-loop consumers see the effective output.
        self._ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = noise_pred

        return {"output_0": noise_pred}

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _should_use_sequential_cfg(self, comp_name: str) -> bool:
        """Check if component needs sequential CFG (TP strategy)."""
        strategy = self._ctx.strategy
        if strategy and hasattr(strategy, 'tp_components'):
            tp_components = getattr(strategy, 'tp_components', set())
            return comp_name in tp_components
        return False

    def _get_encoder_hidden_states_key(self) -> Optional[str]:
        """Get encoder hidden states variable key from topology connections."""
        topology = self._ctx.pkg.topology
        pre_loop = topology.get("flow", {}).get("pre_loop", [])
        loop_components = topology.get("flow", {}).get("loop", {}).get("components", [])

        if not pre_loop or not loop_components:
            return None

        connections = topology.get("connections", [])
        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")
            if "." in from_port and "." in to_port:
                from_comp = from_port.split(".")[0]
                to_comp, to_input = to_port.split(".", 1)
                if from_comp in pre_loop and to_comp in loop_components:
                    if "hidden_state" in to_input.lower():
                        return from_port
        return None

    @staticmethod
    def apply_guidance(
        cond_output: NBXTensor,
        uncond_output: NBXTensor,
        guidance_scale: float,
    ) -> NBXTensor:
        """Pure CFG formula: uncond + scale * (cond - uncond)."""
        return uncond_output + guidance_scale * (cond_output - uncond_output)


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _create_ones_tensor(shape: tuple, dtype: NBXDtype, device: str) -> NBXTensor:
    """Create a tensor filled with ones via dispatch fill kernel."""
    from neurobrix.kernels.dispatch import _create_ones
    return _create_ones(shape, dtype=dtype, device=device)


def _synthesize_position_ids(ctx: 'FlowContext', comp_inputs: list, comp_shapes: dict) -> None:
    """Synthesize position IDs for guidance-embedding models (Flux-style)."""

    if "txt_ids" in comp_inputs and "txt_ids" in comp_shapes:
        shape = comp_shapes["txt_ids"]
        # Create zeros tensor
        txt_ids = NBXTensor.zeros(tuple(shape), dtype=NBXDtype.int64)
        # Text tokens carry sequential position in the LAST channel (L-coordinate).
        # For triton mode, we build this via numpy and upload.
        if len(shape) == 3:
            seq_len = shape[1]
            arr = np.zeros(shape, dtype=np.int64)
            for i in range(seq_len):
                arr[0, i, -1] = i
            txt_ids = NBXTensor.from_numpy(arr)
        else:
            seq_len = shape[0]
            arr = np.zeros(shape, dtype=np.int64)
            for i in range(seq_len):
                arr[i, -1] = i
            txt_ids = NBXTensor.from_numpy(arr)
        ctx.variable_resolver.set("global.txt_ids", txt_ids)

    if "img_ids" in comp_inputs and "img_ids" in comp_shapes:
        img_shape = comp_shapes["img_ids"]
        # Shape can be 2D [positions, dims] or 3D [batch, positions, dims]
        if len(img_shape) == 3:
            n_dims = img_shape[2]
        else:
            n_dims = img_shape[1] if len(img_shape) > 1 else 3

        # Compute n_pos from RUNTIME latent dimensions
        n_pos = None
        comp_hs_shape = comp_shapes.get("hidden_states", [])
        packing = len(comp_hs_shape) == 3  # 3D hidden_states = Flux-style packing

        latent_h_rt = ctx.variable_resolver.defaults.get("latent_height")
        latent_w_rt = ctx.variable_resolver.defaults.get("latent_width")

        if latent_h_rt is not None and latent_w_rt is not None and packing:
            patch_h = int(latent_h_rt) // 2
            patch_w = int(latent_w_rt) // 2
            n_pos = patch_h * patch_w

        # Fallback to topology shapes if runtime dims not available
        if n_pos is None:
            state_shape = comp_shapes.get("hidden_states", [])
            if len(state_shape) >= 2:
                n_pos = state_shape[1] if len(state_shape) == 3 else state_shape[0]
            else:
                n_pos = img_shape[1] if len(img_shape) == 3 else img_shape[0]

        latent_h = int(math.sqrt(n_pos))
        latent_w = n_pos // latent_h

        # Build position grid: [latent_h, latent_w, n_dims] via numpy
        ids = np.zeros((latent_h, latent_w, n_dims), dtype=np.float32)
        for row in range(latent_h):
            ids[row, :, 1] = row
        if n_dims > 2:
            for col in range(latent_w):
                ids[:, col, 2] = col

        # Reshape to match expected shape
        if len(img_shape) == 3:
            ids = ids.reshape(1, -1, n_dims)
        else:
            ids = ids.reshape(-1, n_dims)

        img_ids = NBXTensor.from_numpy(ids)
        ctx.variable_resolver.set("global.img_ids", img_ids)
