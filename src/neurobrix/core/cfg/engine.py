"""
CFGEngine — Unified Classifier-Free Guidance

Consolidates CFGExecutor + CFGStrategy into a single brick.

ZERO HARDCODE: guidance_scale from topology/defaults.
ZERO SEMANTIC: Pure CFG protocol — no model-specific knowledge.

Interface:
    engine = CFGEngine.from_topology(ctx)
    if engine.is_enabled:
        output = engine.execute_with_cfg(comp_name, state, timestep)
"""

import torch
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Union

from neurobrix.core.runtime.debug import DEBUG
from neurobrix.core.dtype.config import get_torch_dtype
from neurobrix.core.runtime.resolution.i2v_conditioning import (
    CONDITION_VAR as _I2V_CONDITION_VAR,
    apply as _i2v_apply,
    condition_channel_dim as _i2v_channel_dim,
)
from neurobrix.core.runtime.resolution.vace_control_conditioning import (
    CONTROL_VAR as _VACE_CONTROL_VAR,
)

if TYPE_CHECKING:
    from neurobrix.core.flow.base import FlowContext


def _resolve_torch_dtype(dt: Union[torch.dtype, str, Any]) -> torch.dtype:
    """Coerce a dtype coming from Prism into a concrete `torch.dtype`.

    After the 'dtype-as-string above engines' refactor, anything flowing
    from Prism's component allocations is a plain string like
    ``"float16"`` or ``"bfloat16"``. Tensor ops that rely on ``.to(dt)``
    must resolve it first because ``torch.Tensor.to(str)`` treats the
    argument as a *device name*, not a dtype — so ``t.to("float16")``
    raises ``RuntimeError: Invalid device string: 'float16'``.
    """
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        return get_torch_dtype(dt.replace("torch.", ""))
    # Fallback: anything else (NBXDtype, numpy dtype, etc.) gets stringified
    # and re-parsed. Keeps the helper tolerant to future dtype carriers.
    return get_torch_dtype(str(dt).replace("torch.", ""))


class CFGMode(Enum):
    """CFG execution mode."""
    DISABLED = auto()     # No CFG (scale <= threshold)
    BATCHED = auto()      # Batch [cond, uncond] together (default)
    SEQUENTIAL = auto()   # Run cond/uncond separately (TP strategy)


class CFGEngine:
    """
    Unified Classifier-Free Guidance engine.

    Handles:
    - Detection: guidance_scale vs threshold from config cascade
    - Batching: [uncond, cond] concatenation for batch=2 execution
    - Sequential: 2x batch=1 for TP strategies
    - Application: uncond + scale * (cond - uncond)
    - Variable management: save/restore encoder states
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
    ) -> 'CFGEngine':
        """
        Create CFGEngine from topology config cascade.

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
                    # Model uses guidance embedding — inject as tensor, skip batch-2 CFG
                    ctx.variable_resolver.set(
                        "global.guidance",
                        torch.tensor([guidance_scale], dtype=torch.float32)
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
        current_state: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
        encoder_dtype: Union[torch.dtype, str],
    ) -> Dict[str, torch.Tensor]:
        """Execute component with CFG (batched or sequential)."""
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
        current_state: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
        encoder_dtype: Union[torch.dtype, str],
    ) -> Dict[str, torch.Tensor]:
        """Batched CFG: [uncond, cond] in single batch=2 forward pass."""
        encoder_key = self._get_encoder_hidden_states_key()
        if encoder_key is None:
            raise RuntimeError(
                "ZERO FALLBACK: Cannot determine encoder_hidden_states variable.\n"
                "Check topology connections from pre_loop to loop components."
            )

        encoder_comp = encoder_key.split(".")[0]
        negative_key = f"{encoder_comp}.negative_hidden_state"

        pos_hidden = self._ctx.variable_resolver.get(encoder_key)
        neg_hidden = self._ctx.variable_resolver.get(negative_key)

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
        pos_mask = self._ctx.variable_resolver.get(mask_var)
        if pos_mask is None:
            pos_mask = self._ctx.variable_resolver.get("global.attention_mask")

        if pos_hidden.shape[1] != neg_hidden.shape[1]:
            raise RuntimeError(
                f"ZERO FALLBACK: pos/neg seq_len mismatch: pos={pos_hidden.shape[1]}, neg={neg_hidden.shape[1]}."
            )

        # Save originals
        orig_hidden = self._ctx.variable_resolver.get(encoder_key)
        orig_state = current_state
        orig_timestep = self._ctx.variable_resolver.loop_state.get(self._ctx.loop_id)

        # Batch inputs: [uncond, cond]
        batched_hidden = torch.cat([neg_hidden, pos_hidden], dim=0)
        batched_state = torch.cat([current_state, current_state], dim=0)

        # Batch mask
        batched_mask = None
        if pos_mask is not None and pos_mask.shape[-1] == pos_hidden.shape[1]:
            neg_mask = self._ctx.variable_resolver.get(f"{encoder_comp}.negative_attention_mask", None)
            if neg_mask is None or neg_mask.shape[-1] != neg_hidden.shape[1]:
                neg_mask = torch.ones(neg_hidden.shape[0], neg_hidden.shape[1],
                                     dtype=pos_mask.dtype, device=pos_mask.device)
            elif neg_mask.dtype != pos_mask.dtype:
                neg_mask = neg_mask.to(dtype=pos_mask.dtype)
            batched_mask = torch.cat([neg_mask, pos_mask], dim=0)

        # Cast to encoder dtype. `encoder_dtype` is sourced from Prism's
        # allocation, which after the dtype-as-string refactor can be either
        # a torch.dtype or a str like "float16". torch.Tensor.to(str)
        # interprets strings as *devices*, so we must resolve strings to
        # torch.dtype first or we hit `Invalid device string: 'float16'`.
        encoder_dtype = _resolve_torch_dtype(encoder_dtype)
        if batched_state.dtype != encoder_dtype:
            batched_state = batched_state.to(encoder_dtype)

        # Batch timestep
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        batched_timestep = timestep.expand(2)
        if batched_timestep.dtype != batched_state.dtype:
            batched_timestep = batched_timestep.to(batched_state.dtype)

        # Set batched inputs
        self._ctx.variable_resolver.set(encoder_key, batched_hidden)
        # I2V channel-concat conditioning: concat the per-step-invariant
        # condition (built once by the flow, stored in global.i2v_condition)
        # onto the batched state for the denoiser. Shared across cond/uncond, so
        # apply() repeats it to the batch=2 state. Inert when absent (R23).
        _cond = self._ctx.variable_resolver.resolved.get(_I2V_CONDITION_VAR)
        if isinstance(_cond, torch.Tensor):
            batched_state = _i2v_apply(batched_state, _cond,
                                       _i2v_channel_dim(self._ctx, comp_name))
        self._ctx.variable_resolver.set(state_key, batched_state)

        # VACE control conditioning: control_hidden_states is a SEPARATE denoiser
        # input (not concat onto state) resolved by the global.<name> fallback, so
        # it is NOT in topology.connections and the connection-driven extra-input
        # batching below cannot see it. Repeat the batch-1 control to batch=2 here
        # to match the batched state; the per-layer scale is batch-invariant.
        # Restored after the forward. Inert when absent (R23).
        _vace_ctrl = self._ctx.variable_resolver.resolved.get(_VACE_CONTROL_VAR)
        if isinstance(_vace_ctrl, torch.Tensor) and _vace_ctrl.shape[0] == 1:
            extra_restore_vace = (_VACE_CONTROL_VAR, _vace_ctrl)
            self._ctx.variable_resolver.set(
                _VACE_CONTROL_VAR, torch.cat([_vace_ctrl, _vace_ctrl], dim=0))
        else:
            extra_restore_vace = None

        # Batch every OTHER shared conditioning input the denoiser consumes to
        # batch=2. The text encoder_hidden_states is the [neg, pos] split above;
        # everything else (e.g. Wan-I2V encoder_hidden_states_image from the CLIP
        # image_encoder) is SHARED across cond/uncond — the vendor pipeline passes
        # the SAME tensor to both forward calls, so in batched mode it must be
        # repeated [v, v] to match the batch=2 hidden_states (the denoiser
        # concatenates image+text conditioning and mismatches the batch otherwise).
        # Restored after the batched forward. Inert when there is no such input.
        extra_restore = {}
        _handled_from = {encoder_key, state_key}
        for conn in self._ctx.pkg.topology.get("connections", []):
            to_port = conn.get("to", "")
            from_port = conn.get("from", "")
            if not to_port.startswith(f"{comp_name}."):
                continue
            if from_port in _handled_from or from_port.startswith(f"{comp_name}."):
                continue
            to_input = to_port.split(".", 1)[1]
            if to_input in ("timestep", "attention_mask", "encoder_attention_mask"):
                continue
            try:
                _v = self._ctx.variable_resolver.get(from_port)
            except Exception:
                continue
            if isinstance(_v, torch.Tensor) and _v.dim() >= 1 and _v.shape[0] == 1:
                extra_restore[from_port] = _v
                self._ctx.variable_resolver.set(
                    from_port, torch.cat([_v, _v], dim=0))

        if batched_mask is not None:
            self._ctx.variable_resolver.set("global.encoder_attention_mask", batched_mask)
        self._ctx.variable_resolver.loop_state[self._ctx.loop_id] = batched_timestep

        # Execute once with batch=2
        output = self._execute_component(comp_name, "loop_cfg_batched")
        noise_pred_batched = self._extract_primary_output(comp_name, output)

        # CFG in float32 for stability
        if noise_pred_batched.dtype != torch.float32:
            noise_pred_batched = noise_pred_batched.float()

        noise_pred_uncond, noise_pred_cond = noise_pred_batched.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if DEBUG:
            print(f"[CFG] guidance={guidance_scale}: uncond={noise_pred_uncond.mean():.4f}, "
                  f"cond={noise_pred_cond.mean():.4f}, final={noise_pred.mean():.4f}")

        # Restore originals
        self._ctx.variable_resolver.set(encoder_key, orig_hidden)
        self._ctx.variable_resolver.set(state_key, orig_state)
        if batched_mask is not None:
            self._ctx.variable_resolver.set("global.encoder_attention_mask", pos_mask)
        self._ctx.variable_resolver.loop_state[self._ctx.loop_id] = orig_timestep
        for _k, _v in extra_restore.items():
            self._ctx.variable_resolver.set(_k, _v)
        if extra_restore_vace is not None:
            self._ctx.variable_resolver.set(*extra_restore_vace)

        return {"output_0": noise_pred}

    # =========================================================================
    # SEQUENTIAL CFG (TP strategy)
    # =========================================================================

    def _execute_sequential_cfg(
        self,
        comp_name: str,
        current_state: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
    ) -> Dict[str, torch.Tensor]:
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

        pos_hidden = self._ctx.variable_resolver.get(encoder_key)
        neg_hidden = self._ctx.variable_resolver.get(negative_key)
        if neg_hidden is None:
            raise RuntimeError(
                f"ZERO FALLBACK: CFG negative embeddings not found: '{negative_key}'."
            )

        pos_mask = self._ctx.variable_resolver.get("global.attention_mask")
        neg_mask = self._ctx.variable_resolver.get(f"{encoder_comp}.negative_attention_mask", None)
        if neg_mask is None or neg_mask.shape[-1] != neg_hidden.shape[1]:
            neg_mask = torch.ones(neg_hidden.shape[0], neg_hidden.shape[1],
                                 dtype=pos_mask.dtype, device=pos_mask.device)
        elif neg_mask.dtype != pos_mask.dtype:
            neg_mask = neg_mask.to(dtype=pos_mask.dtype)

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)

        # Pass 1: unconditional
        self._ctx.variable_resolver.set(encoder_key, neg_hidden)
        # I2V channel-concat conditioning (shared by both passes; batch=1 here).
        _cond = self._ctx.variable_resolver.resolved.get(_I2V_CONDITION_VAR)
        _seq_state = (_i2v_apply(current_state, _cond,
                                 _i2v_channel_dim(self._ctx, comp_name))
                      if isinstance(_cond, torch.Tensor) else current_state)
        self._ctx.variable_resolver.set(state_key, _seq_state)
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
                  f"uncond={noise_pred_uncond.mean():.4f}, cond={noise_pred_cond.mean():.4f}")

        self._ctx.variable_resolver.set(encoder_key, pos_hidden)
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
        # Pass 1: a pre_loop encoder feeds a loop component's *hidden_state*
        # input directly (text_encoder -> transformer.encoder_hidden_states).
        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")
            if "." in from_port and "." in to_port:
                from_comp = from_port.split(".")[0]
                to_comp, to_input = to_port.split(".", 1)
                if from_comp in pre_loop and to_comp in loop_components:
                    # The CFG split-key is the TEXT condition (encoder_hidden_states),
                    # which has a negative. NEVER the image condition
                    # (encoder_hidden_states_image, Wan-I2V) — it is shared across
                    # cond/uncond and has no negative, so picking it would look for a
                    # nonexistent image_encoder.negative_hidden_state.
                    ti = to_input.lower()
                    if "hidden_state" in ti and "image" not in ti:
                        return from_port
        # Pass 1b: a FLUX/MMDiT denoiser NAMES its text-condition input `txt`
        # (not `*hidden_state`), so Pass 1 misses it. Fall back to the OUTPUT
        # side: a pre_loop text-encoder whose `*_hidden_state` output feeds a
        # loop component is the text condition (it has a negative). The pooled
        # vector condition (CLIP pooler_output -> y_vec) has no `hidden_state`
        # in its from-port, so it is excluded here; the image condition is
        # excluded as in Pass 1. Runs only after Pass 1, so models that name the
        # input `encoder_hidden_states` keep their exact Pass-1 match (byte-id).
        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")
            if "." in from_port and "." in to_port:
                from_comp = from_port.split(".")[0]
                from_out = from_port.split(".", 1)[1].lower()
                to_comp, to_input = to_port.split(".", 1)
                if from_comp in pre_loop and to_comp in loop_components:
                    ti = to_input.lower()
                    if ("hidden_state" in from_out and "image" not in from_out
                            and "image" not in ti):
                        return from_port
        # Pass 2: GLOBAL-variable indirection. The I2V builds (Wan-I2V) wire the
        # text condition as `global.encoder_hidden_states -> transformer.
        # encoder_hidden_states` (the text_encoder output is finalized into the
        # global by the resolution engine) rather than a direct pre_loop->loop
        # edge. Target the TEXT condition specifically (`encoder_hidden_state`) so
        # the batched-CFG variable is the prompt embedding, NEVER the latent
        # `hidden_states` input that also carries the `hidden_state` substring.
        for conn in connections:
            from_port = conn.get("from", "")
            to_port = conn.get("to", "")
            if "." in from_port and "." in to_port:
                from_comp = from_port.split(".")[0]
                to_comp, to_input = to_port.split(".", 1)
                if from_comp == "global" and to_comp in loop_components:
                    if "encoder_hidden_state" in to_input.lower():
                        return from_port
        return None

    @staticmethod
    def apply_guidance(
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Pure CFG formula: uncond + scale * (cond - uncond)."""
        return uncond_output + guidance_scale * (cond_output - uncond_output)


def _synthesize_position_ids(ctx: 'FlowContext', comp_inputs: list, comp_shapes: dict) -> None:
    """Synthesize position IDs for guidance-embedding models (Flux-style)."""
    import math

    if "txt_ids" in comp_inputs and "txt_ids" in comp_shapes:
        shape = comp_shapes["txt_ids"]
        txt_ids = torch.zeros(shape, dtype=torch.long)
        # Text tokens carry sequential position in the LAST channel (L-coordinate).
        # Convention: [t=0, h=0, w=0, l=position_index] for 4D position encoding.
        # This matches diffusers _prepare_text_ids: cartesian_prod(t, h, w, arange(L)).
        if len(shape) == 3:
            seq_len = shape[1]
            txt_ids[0, :, -1] = torch.arange(seq_len)
        else:
            seq_len = shape[0]
            txt_ids[:, -1] = torch.arange(seq_len)
        ctx.variable_resolver.set("global.txt_ids", txt_ids)

    if "img_ids" in comp_inputs and "img_ids" in comp_shapes:
        img_shape = comp_shapes["img_ids"]
        # Shape can be 2D [positions, dims] or 3D [batch, positions, dims]
        if len(img_shape) == 3:
            n_dims = img_shape[2]
        else:
            n_dims = img_shape[1] if len(img_shape) > 1 else 3

        # Compute n_pos from RUNTIME latent dimensions (not trace-time topology shapes).
        # At synthesis time, the state variable hasn't been packed yet, so we derive
        # the packed position count from runtime latent dims in defaults.
        n_pos = None
        comp_hs_shape = comp_shapes.get("hidden_states", [])
        packing = len(comp_hs_shape) == 3  # 3D hidden_states = Flux-style packing

        latent_h_rt = ctx.variable_resolver.defaults.get("latent_height")
        latent_w_rt = ctx.variable_resolver.defaults.get("latent_width")

        if latent_h_rt is not None and latent_w_rt is not None and packing:
            # Flux-style packing: [B,C,H,W] -> [B, (H/2)*(W/2), C*4]
            # patch_size=2 consistent with _pack_latents/_unpack_latents
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

        # Build position grid: [latent_h, latent_w, n_dims]
        # dim 0 = batch_idx (0), dim 1 = row, dim 2 = col, rest = 0
        ids = torch.zeros(latent_h, latent_w, n_dims)
        ids[..., 1] = ids[..., 1] + torch.arange(latent_h)[:, None]
        if n_dims > 2:
            ids[..., 2] = ids[..., 2] + torch.arange(latent_w)[None, :]

        # Reshape to match expected shape
        if len(img_shape) == 3:
            img_ids = ids.reshape(1, -1, n_dims)
        else:
            img_ids = ids.reshape(-1, n_dims)
        ctx.variable_resolver.set("global.img_ids", img_ids)
