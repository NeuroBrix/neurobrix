"""
Iterative Process Flow Handler

ZERO SEMANTIC: Executes diffusion loop from topology.json.
ZERO HARDCODE: All flow configuration from NBX container.

Handles diffusion models with denoising loop:
1. Pre-loop components (text encoders)
2. Main loop (transformer with scheduler)
3. Post-loop components (VAE decoder)
"""

import gc
import torch
from typing import Any, Dict, List, Optional, Callable

from neurobrix.core.runtime.debug import DEBUG
from neurobrix.core.device_utils import device_empty_cache
from neurobrix.core.runtime.resolution import i2v_conditioning
from neurobrix.core.runtime.resolution import vace_control_conditioning
from .base import FlowHandler, FlowContext, register_flow


@register_flow("iterative_process")
class IterativeProcessHandler(FlowHandler):
    """
    Flow handler for iterative diffusion process.

    Executes the standard diffusion pipeline:
    - pre_loop: Text encoding (with optional CFG negative encoding)
    - loop: Denoising iterations (with optional CFG batching)
    - post_loop: VAE decoding

    ZERO SEMANTIC: No knowledge of "image", "latent", "vae".
    ZERO HARDCODE: All configuration from topology.
    """

    # Instance attribute for packing info (Flux-style models)
    _packing_info: Optional[Dict[str, int]]

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable[[str, str, Optional[torch.Tensor]], Any],
        extract_primary_output_fn: Callable[[str, Any], Any],
        cfg_engine: Optional[Any] = None,
        output_extractor: Optional[Any] = None
    ):
        """
        Initialize iterative process handler.

        Args:
            ctx: FlowContext with all shared state
            execute_component_fn: Function to execute a component
            extract_primary_output_fn: Function to extract primary output
            cfg_engine: Optional CFGEngine for classifier-free guidance
            output_extractor: Optional OutputExtractor for encoder lookups
        """
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._extract_primary_output = extract_primary_output_fn
        self._cfg_engine = cfg_engine
        self._output_extractor = output_extractor
        self._packing_info = None

    def execute(self) -> Dict[str, Any]:
        """
        Execute iterative_process flow from topology.json.

        1. Preprocess inputs (tokenization if needed)
        2. pre_loop components (once) - with CFG: run for both positive and negative prompts
        3. main loop (N iterations) - with CFG: batch 2, apply guidance formula
        4. post_loop components (once)

        Returns:
            Dict of resolved variables/outputs
        """
        # 0. Preprocess inputs (tokenization)
        self._preprocess_inputs()

        flow = self.ctx.pkg.topology.get("flow", {})

        # CFG settings from engine (already resolved in from_topology)
        do_cfg = self._cfg_engine.is_enabled if self._cfg_engine else False
        guidance_scale = self._cfg_engine.guidance_scale if self._cfg_engine else 1.0

        # 1. Pre-loop execution
        pre_loop = flow.get("pre_loop", [])
        self._execute_pre_loop(pre_loop, do_cfg)

        # 2. Main loop
        loop_def = flow.get("loop", {})
        driver_name = loop_def.get("driver")
        loop_components = loop_def.get("components", [])

        if driver_name and loop_components:
            self._execute_main_loop(driver_name, loop_components, do_cfg, guidance_scale)

        # 3. Post-loop execution
        post_loop = flow.get("post_loop", [])
        self._execute_post_loop(post_loop, loop_components)

        # 4. Return resolved outputs
        return self.ctx.variable_resolver.resolve_all()

    def _execute_pre_loop(self, pre_loop: List[str], do_cfg: bool) -> None:
        """
        Execute pre-loop components.

        For each component:
        1. Execute the component
        2. Finalize embeddings (Sana CHI slicing, etc.) via handler
        3. If CFG and produces encoder states, run negative encoding
        4. Unload weights to free memory

        Args:
            pre_loop: List of pre-loop component names
            do_cfg: Whether CFG is enabled
        """
        for comp_name in pre_loop:
            self._execute_component(comp_name, "pre_loop", None)

            # PURIFICATION: Finalize POSITIVE embeddings immediately after encoding
            # This handles Sana CHI slicing BEFORE we store the result for CFG
            executor = self.ctx.executors.get(comp_name)
            handler = getattr(executor, '_component_handler', None) if executor else None
            
            # Always finalize positive embeddings if handler supports it
            if handler and hasattr(handler, 'finalize_embeddings'):
                # DATA-DRIVEN: find actual encoder output (not hardcoded output name)
                hidden_state = None
                hidden_key = None
                for suffix in ("last_hidden_state", "output_0"):
                    candidate = f"{comp_name}.{suffix}"
                    if candidate in self.ctx.variable_resolver.resolved:
                        hidden_key = candidate
                        hidden_state = self.ctx.variable_resolver.resolved[candidate]
                        break
                if hidden_state is not None and hidden_key is not None:
                    tokenizer_vals = self._tokenizer_config_with_flags(comp_name, "tokenizer")
                    pos_mask = self.ctx.variable_resolver.get("global.attention_mask")

                    finalized = handler.finalize_embeddings(
                        hidden_state=hidden_state,
                        attention_mask=pos_mask,
                        tokenizer_config=tokenizer_vals
                    )
                    # Update the stored embedding with finalized version
                    self.ctx.variable_resolver.set(hidden_key, finalized["hidden_state"])
                    if finalized.get("attention_mask") is not None:
                        self.ctx.variable_resolver.set("global.attention_mask", finalized["attention_mask"])

            # CFG: If produces encoder hidden states, run negative encoding
            if do_cfg and self._output_extractor:
                if self._output_extractor.produces_encoder_hidden_states(comp_name):
                    # Flow control for negative encoding belongs in the handler
                    self._execute_negative_encoding(comp_name)

            if DEBUG:
                for suffix in ("output_0", "last_hidden_state"):
                    key = f"{comp_name}.{suffix}"
                    if key in self.ctx.variable_resolver.resolved:
                        t = self.ctx.variable_resolver.resolved[key]
                        if isinstance(t, torch.Tensor):
                            print(f"[AUDIT] {key}: shape={list(t.shape)} dtype={t.dtype} "
                                  f"min={t.min().item():.4f} max={t.max().item():.4f} std={t.float().std().item():.4f}")

            # Unload weights immediately. Pre_loop components run once
            # per request, so force-unload even in eager mode to free
            # VRAM for the main loop. Matches the triton path.
            self._unload_component(comp_name, force=True)

    def _tokenizer_config_with_flags(self, encoder_name: str, tokenizer_name: str) -> Dict[str, Any]:
        """Tokenizer extracted_values augmented with runtime text-conditioning
        flags from the registry. Currently `zero_pad_embeddings` (T5/UMT5
        encoders emit non-zero pad embeddings; a DiT that cross-attends to the
        full sequence must have them zeroed — the vendor pipelines trim+zero-pad).
        Read at runtime via registry_flags (no .nbx rebuild); inert for encoders
        without the flag, so image models are untouched (R23)."""
        cfg = dict(self.ctx.pkg.topology.get("extracted_values", {}).get(tokenizer_name, {}) or {})
        try:
            from neurobrix.core.runtime.registry_flags import get_component_flag
            model_name = self.ctx.pkg.manifest.get("model_name")
            if get_component_flag(model_name, encoder_name, "zero_pad_embeddings", default=False):
                cfg["zero_pad_embeddings"] = True
        except Exception:
            pass
        return cfg

    def _execute_negative_encoding(self, text_encoder_name: str) -> None:
        """
        Execute text encoder for negative/unconditional embedding (CFG).

        Flow control for negative encoding belongs in the handler.
        
        For PixArt-Sigma/Sana, the negative prompt is typically "" (empty string).
        This creates the unconditional embedding that will be concatenated
        with the positive embedding for batch 2 execution.

        Stores result in: text_encoder.negative_hidden_state

        Args:
            text_encoder_name: Name of text encoder component
        """

        # Get negative prompt (default: empty string)
        try:
            negative_prompt = self.ctx.variable_resolver.get("global.negative_prompt")
            if negative_prompt is None:
                negative_prompt = ""
        except KeyError:
            negative_prompt = ""

        # Determine per-encoder tokenizer and variable names (DATA-DRIVEN)
        suffix = ""
        if text_encoder_name != "text_encoder" and text_encoder_name.startswith("text_encoder_"):
            suffix = "_" + text_encoder_name.split("text_encoder_")[1]
        tokenizer_name = f"tokenizer{suffix}"
        input_ids_var = f"global.input_ids{suffix}"
        attention_mask_var = f"global.attention_mask{suffix}"

        tokenizer = self.ctx.modules.get(tokenizer_name)
        if tokenizer is None:
            raise RuntimeError(
                f"ZERO FALLBACK: CFG requires '{tokenizer_name}' module for negative prompt encoding of {text_encoder_name}."
            )

        # Tokenize negative prompt via TextProcessor
        from neurobrix.core.module.text.processor import TextProcessor
        tp = TextProcessor(
            tokenizer=tokenizer,
            defaults=self.ctx.pkg.defaults,
            topology=self.ctx.pkg.topology,
            variable_resolver=self.ctx.variable_resolver,
            tokenizer_name=tokenizer_name,
        )
        device = self.ctx.primary_device
        neg_input_ids, neg_attention_mask = tp.tokenize_negative(
            device, encoder_name=text_encoder_name, negative_prompt=negative_prompt
        )
        tokenizer_vals = self._tokenizer_config_with_flags(text_encoder_name, tokenizer_name)

        # Save original inputs AND positive embeddings (per-encoder variables)
        orig_input_ids = self.ctx.variable_resolver.get(input_ids_var)
        orig_attention_mask = self.ctx.variable_resolver.get(attention_mask_var)
        pos_hidden_state = self.ctx.variable_resolver.get(f"{text_encoder_name}.last_hidden_state")

        # Set negative inputs temporarily
        self.ctx.variable_resolver.set(input_ids_var, neg_input_ids)
        self.ctx.variable_resolver.set(attention_mask_var, neg_attention_mask)

        # Execute text encoder for negative
        self._execute_component(text_encoder_name, "cfg_negative", None)

        # Get negative embedding
        neg_hidden_state = self.ctx.variable_resolver.get(f"{text_encoder_name}.last_hidden_state")

        # --- PURIFICATION: Finalize embeddings via Handler ---
        executor = self.ctx.executors.get(text_encoder_name)
        handler = getattr(executor, '_component_handler', None) if executor else None
        if handler and hasattr(handler, 'finalize_embeddings'):
            finalized = handler.finalize_embeddings(
                hidden_state=neg_hidden_state,
                attention_mask=neg_attention_mask,  # REQUIRED for zero_pad_embeddings
                tokenizer_config=tokenizer_vals
            )
            neg_hidden_state = finalized["hidden_state"]
        # -----------------------------------------------------

        # Store negative embedding AND its attention mask (for CFG executor)
        self.ctx.variable_resolver.set(f"{text_encoder_name}.negative_hidden_state", neg_hidden_state)
        self.ctx.variable_resolver.set(f"{text_encoder_name}.negative_attention_mask", neg_attention_mask)

        # Restore original inputs
        self.ctx.variable_resolver.set(input_ids_var, orig_input_ids)
        self.ctx.variable_resolver.set(attention_mask_var, orig_attention_mask)
        self.ctx.variable_resolver.set(f"{text_encoder_name}.last_hidden_state", pos_hidden_state)


    def _execute_main_loop(
        self,
        driver_name: str,
        components: List[str],
        do_cfg: bool,
        guidance_scale: float
    ) -> None:
        """
        Execute the main iteration loop.

        Args:
            driver_name: Name of driver module (scheduler)
            components: List of loop component names
            do_cfg: Whether CFG is enabled
            guidance_scale: CFG guidance scale
        """
        # Get driver module
        if driver_name not in self.ctx.modules:
            raise RuntimeError(
                f"ZERO FALLBACK: Driver module '{driver_name}' not found.\n"
                f"Available modules: {list(self.ctx.modules.keys())}"
            )
        driver = self.ctx.modules[driver_name]

        # Get control variables
        num_steps = self.ctx.variable_resolver.get("global.num_inference_steps")

        # Get state variable key from topology
        loop_def = self.ctx.pkg.topology.get("flow", {}).get("loop", {})
        state_key = loop_def.get("state_variable")
        if state_key is None:
            raise RuntimeError(
                "ZERO FALLBACK: 'state_variable' not defined in topology.flow.loop."
            )
        current_state = self.ctx.variable_resolver.get(state_key)

        # DATA-DRIVEN: Detect if Flux-style packing is needed
        packing_shape = self._detect_packing(components)
        if packing_shape is not None and current_state.dim() == 4:
            # Save original spatial dims for unpacking later
            self._packing_info = {
                'channels': current_state.shape[1],
                'height': current_state.shape[2],
                'width': current_state.shape[3],
            }
            current_state = self._pack_latents(current_state)
            self.ctx.variable_resolver.set(state_key, current_state)
        else:
            self._packing_info = None

        # I2V latent channel-concat conditioning (data-driven, R23-gated). When
        # the denoiser's in_channels exceed the latent channels (Wan-I2V: 36 vs
        # 16) the runtime builds a per-step-invariant condition (VAE-encoded
        # first frame + frame mask) and channel-concats it onto the state before
        # the transformer. Built ONCE here; applied each step (non-CFG below,
        # CFG-batched/sequential inside the CFG engine via global.i2v_condition).
        self._i2v_condition = None
        loop_comp0 = components[0] if components else None
        _cond_spec = (i2v_conditioning.conditioning_spec(self.ctx, loop_comp0)
                      if loop_comp0 else None)
        if _cond_spec is not None:
            _nf = self.ctx.variable_resolver.get("global.num_frames")
            self._i2v_condition = i2v_conditioning.build_condition(
                self.ctx, _cond_spec, int(_nf))
            if self._i2v_condition is not None:
                self.ctx.variable_resolver.set(
                    i2v_conditioning.CONDITION_VAR, self._i2v_condition)

        # VACE control conditioning (data-driven, R23-gated). The denoiser takes
        # control_hidden_states (96ch latent control) + control_hidden_states_scale
        # (ones(len(vace_layers))) as separate inputs each step. Built ONCE here
        # from the vae_encoder pass (control video latents) and stored as globals
        # the InputResolver binds by the global.<name> fallback. CFG batches the
        # control to batch=2 inside the CFG engine (the scale is batch-invariant).
        _vace_spec = (vace_control_conditioning.conditioning_spec(self.ctx, loop_comp0)
                      if loop_comp0 else None)
        if _vace_spec is not None:
            _ctrl = vace_control_conditioning.build_control(self.ctx, _vace_spec)
            if _ctrl is not None:
                _scale = vace_control_conditioning.build_scale(
                    _vace_spec, _ctrl.device, _ctrl.dtype)
                self.ctx.variable_resolver.set(
                    vace_control_conditioning.CONTROL_VAR, _ctrl)
                self.ctx.variable_resolver.set(
                    vace_control_conditioning.SCALE_VAR, _scale)
                self.ctx.variable_resolver.resolved[
                    vace_control_conditioning.CONTROL_VAR] = _ctrl
                self.ctx.variable_resolver.resolved[
                    vace_control_conditioning.SCALE_VAR] = _scale

        # Initialize driver — pass image_seq_len for dynamic shifting
        if hasattr(driver, "set_timesteps"):
            kwargs = {}
            if current_state.dim() == 3:
                # Packed 3D: [B, seq_len, D] — seq_len is the image sequence length
                kwargs["image_seq_len"] = current_state.shape[1]
            driver.set_timesteps(num_steps, **kwargs)

        # Scale initial noise
        if hasattr(driver, "init_noise_sigma"):
            noise_scale = driver.init_noise_sigma
            if DEBUG:
                print(f"[Executor] current_state std before scaling: {current_state.std().item():.4f}")
            if noise_scale != 1.0:
                if DEBUG:
                    print(f"[Executor] Scaling initial state by init_noise_sigma = {noise_scale:.4f}")
                current_state = current_state * noise_scale
                if DEBUG:
                    print(f"[Executor] current_state std after scaling: {current_state.std().item():.4f}")
                self.ctx.variable_resolver.set(state_key, current_state)

        # Get iterator
        iterator = driver.timesteps if hasattr(driver, "timesteps") else range(num_steps)

        encoder_dtype = self._get_encoder_dtype()

        # Main loop
        for step_idx, timestep in enumerate(iterator):
            if DEBUG:
                ts_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
                print(f"[Loop] Step {step_idx + 1}/{num_steps} (t={ts_val:.1f})")

            # Store timestep in loop state
            if isinstance(timestep, torch.Tensor) and timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            self.ctx.variable_resolver.loop_state[self.ctx.loop_id] = timestep

            for comp_name in components:
                if DEBUG and step_idx == 0:
                    self._audit_component_inputs(comp_name)

                # Scale model input
                scaled_state = driver.scale_model_input(current_state, timestep)
                self.ctx.variable_resolver.set(state_key, scaled_state)

                # Get timestep scale for component
                comp_timestep_scale = self._get_component_timestep_scale(comp_name)
                scaled_timestep = timestep * comp_timestep_scale if comp_timestep_scale != 1.0 else timestep

                # Execute with or without CFG
                if do_cfg and self._is_loop_component(comp_name) and self._cfg_engine:
                    output = self._cfg_engine.execute_component_with_cfg(
                        comp_name, scaled_state, scaled_timestep, guidance_scale,
                        encoder_dtype=encoder_dtype
                    )
                else:
                    # I2V channel-concat conditioning on the denoiser's state
                    # (non-CFG path; the CFG engine applies it on its own batched/
                    # sequential paths via global.i2v_condition).
                    if (self._i2v_condition is not None
                            and self._is_loop_component(comp_name)):
                        self.ctx.variable_resolver.set(
                            state_key,
                            i2v_conditioning.apply(scaled_state, self._i2v_condition))
                    # Type guard: scaled_timestep may be int/float, convert to tensor if needed
                    timestep_arg: Optional[torch.Tensor] = None
                    if isinstance(scaled_timestep, torch.Tensor):
                        timestep_arg = scaled_timestep
                    output = self._execute_component(comp_name, "loop", timestep_arg)

                # Update state via driver step
                if output is not None:
                    # Restore original unscaled state
                    self.ctx.variable_resolver.set(state_key, current_state)

                    # Extract primary output
                    model_output = self._extract_primary_output(comp_name, output)

                    # Handle variance prediction split. Channels are axis 1 for
                    # both 4D image [B,C,H,W] and 5D video [B,C,T,H,W]; only the
                    # 3D packed (Flux) layout carries channels on axis 2. Branch
                    # on tensor rank, never on model family (R34) — for 5D the
                    # naive `dim()==4` path mistook axis 2 (T) for channels.
                    if current_state.dim() in (4, 5):
                        state_channels = current_state.shape[1]
                        model_channels = model_output.shape[1]
                    else:
                        # 3D packed: dim=2 is the packed channel dim
                        state_channels = current_state.shape[2]
                        model_channels = model_output.shape[2] if model_output.dim() == 3 else model_output.shape[1]
                    if model_channels == 2 * state_channels:
                        split_dim = 2 if current_state.dim() == 3 else 1
                        model_output = model_output.chunk(2, dim=split_dim)[0]

                    # DiT velocity-diff capture (NBX_DUMP_DIT): step-0 transformer
                    # inputs + raw velocity, for a matched-input comparison against
                    # the vendored WanTransformer3DModel. Run with --cfg 1.0 so this
                    # is the un-guided single forward (batch 1).
                    import os as _os
                    _dd = _os.environ.get("NBX_DUMP_DIT")
                    if _dd and step_idx == 0 and self._is_loop_component(comp_name):
                        _cap = {"velocity": model_output.detach().cpu()}
                        if isinstance(scaled_timestep, torch.Tensor):
                            _cap["timestep"] = scaled_timestep.detach().cpu()
                        else:
                            _cap["timestep"] = scaled_timestep
                        _ci = self.ctx.pkg.topology.get("components", {}).get(comp_name, {})
                        _conns = self.ctx.pkg.topology.get("connections", [])
                        _res = self.ctx.variable_resolver.resolved
                        # Interface inputs + connection-fed inputs (e.g. I2V
                        # image_latents arrives via a connection only).
                        _in_names = list(_ci.get("interface", {}).get("inputs", []))
                        for _c in _conns:
                            _to = str(_c.get("to", ""))
                            if _to.startswith(f"{comp_name}."):
                                _nm = _to.split(".", 1)[1]
                                if _nm not in _in_names:
                                    _in_names.append(_nm)
                        for _in in _in_names:
                            # Try direct keys, then resolve the topology connection
                            # (e.g. transformer.encoder_hidden_states <- text_encoder.last_hidden_state).
                            _cands = [f"{comp_name}.{_in}", f"global.{_in}"]
                            for _c in _conns:
                                if _c.get("to") == f"{comp_name}.{_in}":
                                    _cands.append(_c.get("from"))
                            for _k in _cands:
                                _v = _res.get(_k)
                                if isinstance(_v, torch.Tensor):
                                    _cap[_in] = _v.detach().cpu()
                                    break
                        # I2V latent-concat condition (20ch) so the exact 36ch
                        # transformer input can be reconstructed offline for the
                        # vendor bit-equal microtest: 36ch = cat([state16, cond20]).
                        _cond = _res.get("global.i2v_condition")
                        if isinstance(_cond, torch.Tensor):
                            _cap["i2v_condition"] = _cond.detach().cpu()
                        torch.save(_cap, _dd)
                        print(f"[DIT-DUMP] keys={list(_cap.keys())} "
                              f"velocity std={model_output.float().std().item():.4f} -> {_dd}")

                    # Handle cross-device transfer
                    if self.ctx.strategy is not None and hasattr(self.ctx.strategy, 'transfer_model_output'):
                        # Type guard: ensure the method exists (pipeline strategy)
                        transfer_fn = getattr(self.ctx.strategy, 'transfer_model_output', None)
                        if transfer_fn is not None:
                            model_output = transfer_fn(model_output, current_state)
                    elif model_output.device != current_state.device:
                        model_output = model_output.to(current_state.device)

                    if DEBUG:
                        self._print_step_diagnostics(step_idx, timestep, model_output, current_state)

                    # Driver step
                    step_result = driver.step(model_output, timestep, current_state)
                    if isinstance(step_result, dict):
                        current_state = step_result["prev_sample"]
                    else:
                        current_state = step_result.prev_sample

                    # Lightweight trajectory probe (NBX_TRAJ): per-step latent std,
                    # to compare collapse-vs-recover against the vendor U-curve
                    # (0.99->0.72->1.02). Cheaper than full NBX_DEBUG (one sync/step).
                    import os as _ost
                    if _ost.environ.get("NBX_TRAJ") == "1":
                        _ts = timestep.item() if hasattr(timestep, "item") else timestep
                        print(f"[TRAJ] step {step_idx+1:2d} t={float(_ts):.1f} "
                              f"velocity_std={model_output.float().std().item():.4f} "
                              f"latent_std={current_state.float().std().item():.4f}", flush=True)


                    if DEBUG:
                        if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                            print(f"[DEBUG] CRITICAL: NaN/Inf in current_state at step {step_idx + 1}!")

                    self.ctx.variable_resolver.set(state_key, current_state)

    def _execute_post_loop(self, post_loop: List[str], loop_components: List[str]) -> None:
        """
        Execute post-loop components (e.g., VAE decoder).

        ZERO FALLBACK: Validates latent shape before VAE to catch 5D errors early.

        Args:
            post_loop: List of post-loop component names
            loop_components: List of loop components (to unload first)
        """
        if not post_loop:
            return

        # Release loop + pre_loop component weights to free VRAM for VAE
        # Force-unload even in eager mode — within a single request, non-active
        # components must yield memory. They reload on the next request.
        pre_loop = self.ctx.pkg.topology.get("flow", {}).get("pre_loop", [])
        for comp_name in list(loop_components) + pre_loop:
            self._unload_component(comp_name, force=True)

        # Get state variable for validation
        loop_def = self.ctx.pkg.topology.get("flow", {}).get("loop", {})
        state_key = loop_def.get("state_variable")

        # Unpack latents if packing was used (Flux-style)
        if state_key and self._packing_info is not None:
            current_state = self.ctx.variable_resolver.get(state_key)
            if current_state is not None and isinstance(current_state, torch.Tensor) and current_state.dim() == 3:
                pi = self._packing_info
                assert pi is not None  # Type guard for pyright
                current_state = self._unpack_latents(
                    current_state, pi['height'], pi['width'], pi['channels']
                )
                self.ctx.variable_resolver.set(state_key, current_state)

        # Name-driven latent axis alignment (5D): video models do not share
        # one latent layout — Wan is [B,C,T,H,W] end-to-end while CogVideoX
        # denoises in [B,T,C,H,W] and decodes in [B,C,T,H,W]; the vendor
        # pipeline permutes between the loop and the VAE, outside both traced
        # components. Derive the permutation from existing contract data
        # (variables.json shape_source roles vs the consumer graph's named
        # symbol axes); identical layouts (Wan) derive None and nothing
        # changes (R23).
        if state_key and post_loop:
            from neurobrix.core.runtime.resolution.axis_alignment import (
                latent_permutation_for)
            executor = self.ctx.executors.get(post_loop[0])
            dag = getattr(executor, "_dag", None) if executor else None
            if dag:
                perm = latent_permutation_for(self.ctx.pkg.variables, dag)
                if perm is not None:
                    current_state = self.ctx.variable_resolver.get(state_key)
                    if (isinstance(current_state, torch.Tensor)
                            and current_state.dim() == len(perm)):
                        self.ctx.variable_resolver.set(
                            state_key,
                            current_state.permute(*perm).contiguous())

        # ZERO FALLBACK: Validate latent shape before VAE
        if state_key:
            from neurobrix.core.validators import TensorValidator
            current_state = self.ctx.variable_resolver.get(state_key)
            if current_state is not None and isinstance(current_state, torch.Tensor):
                # Image: 4D [B, C, H, W], Video: 5D [B, C, T, H, W]
                expected = 5 if current_state.dim() == 5 else 4
                TensorValidator.validate_latent_shape(
                    tensor=current_state,
                    component_name="pre_vae",
                    expected_dims=expected
                )

        # Execute post-loop components
        for comp_name in post_loop:
            self._execute_component(comp_name, "post_loop", None)

        # Unload post-loop components
        for comp_name in post_loop:
            self._unload_component(comp_name)

    def _unload_component(self, comp_name: str, force: bool = False) -> None:
        """
        Unload component weights and clear memory.

        Args:
            comp_name: Component to unload
            force: If True, unload even in eager mode (used by post_loop
                   to free VRAM for VAE — weights reload on next request).
                   Skipped when persistent_mode is True (serve mode): Prism
                   verified all components fit, keep weights for next request.
        """
        # Serve mode: never unload — keep all weights resident for near-zero latency
        if self.ctx.persistent_mode:
            return

        if not force:
            loading_mode = getattr(self.ctx.plan, 'loading_mode', 'lazy')
            if loading_mode == "eager":
                return

        executor = self.ctx.executors.get(comp_name)
        if executor:
            executor.unload_weights()
        gc.collect()
        device_empty_cache(self.ctx.primary_device)

    def _is_loop_component(self, comp_name: str) -> bool:
        """Check if component is in the main loop."""
        loop_def = self.ctx.pkg.topology.get("flow", {}).get("loop", {})
        return comp_name in loop_def.get("components", [])

    def _get_encoder_dtype(self) -> Any:
        """
        Get encoder dtype from Prism plan allocation.

        Priority:
        1. First pre_loop component's dtype (encoder)
        2. First available component's dtype (fallback)

        ZERO FALLBACK: Crashes if no dtype found.
        """
        pre_loop = self.ctx.pkg.topology.get("flow", {}).get("pre_loop", [])
        plan_components = getattr(self.ctx.plan, 'components', None) if self.ctx.plan else None

        if plan_components:
            # Priority 1: pre_loop encoder dtype
            for comp in pre_loop:
                if comp in plan_components:
                    dtype = getattr(plan_components[comp], 'dtype', None)
                    if dtype is not None:
                        if DEBUG:
                            print(f"[Executor] Encoder dtype from Prism plan: {comp}={dtype}")
                        return dtype

            # Priority 2: any component's dtype
            for comp_name, alloc in plan_components.items():
                dtype = getattr(alloc, 'dtype', None)
                if dtype is not None:
                    if DEBUG:
                        print(f"[Executor] Using {comp_name} dtype as encoder_dtype: {dtype}")
                    return dtype

        raise RuntimeError(
            "ZERO FALLBACK: encoder_dtype not found in Prism plan.\n"
            "Expected: plan.components[encoder].dtype"
        )

    def _audit_component_inputs(self, comp_name: str) -> None:
        """Print all component inputs for debugging (DEBUG only)."""
        comp_info = self.ctx.pkg.topology.get("components", {}).get(comp_name, {})
        for inp_name in comp_info.get("interface", {}).get("inputs", []):
            for prefix in (f"{comp_name}.", "global."):
                key = f"{prefix}{inp_name}"
                val = self.ctx.variable_resolver.resolved.get(key)
                if isinstance(val, torch.Tensor):
                    print(f"[AUDIT] {comp_name} input '{inp_name}' ({key}): "
                          f"shape={list(val.shape)} dtype={val.dtype} "
                          f"min={val.min().item():.4f} max={val.max().item():.4f}")
                    break
                elif val is not None:
                    print(f"[AUDIT] {comp_name} input '{inp_name}' ({key}): type={type(val).__name__} val={val}")
                    break

    def _print_step_diagnostics(
        self,
        step_idx: int,
        timestep: Any,
        model_output: torch.Tensor,
        current_state: torch.Tensor,
    ) -> None:
        """Print per-step diagnostic stats (DEBUG only)."""
        _t_val = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        _mo_std = model_output.float().std().item()
        _cs_std = current_state.float().std().item()
        print(f"[STEP {step_idx+1:2d}] t={_t_val:.4f} | output: [{model_output.min().item():.3f}, "
              f"{model_output.max().item():.3f}] std={_mo_std:.4f} | state: [{current_state.min().item():.3f}, "
              f"{current_state.max().item():.3f}] std={_cs_std:.4f}")

        if step_idx == 0 and model_output.dim() == 3:
            _mo = model_output[0].float()
            _seq_len = _mo.shape[0]
            _side = int(_seq_len ** 0.5)
            if _side * _side == _seq_len:
                _mo_grid = _mo.view(_side, _side, -1)
                _grad_h = (_mo_grid[1:, :, :] - _mo_grid[:-1, :, :]).abs().mean().item()
                _grad_w = (_mo_grid[:, 1:, :] - _mo_grid[:, :-1, :]).abs().mean().item()
                print(f"[SPATIAL] velocity grad_h={_grad_h:.6f} grad_w={_grad_w:.6f} "
                      f"std={_mo_std:.6f} smoothness={_mo_std/_grad_h:.2f}x")
                _cs = current_state[0].float()
                if _cs.shape[0] == _seq_len:
                    _cs_grid = _cs.view(_side, _side, -1)
                    _sg_h = (_cs_grid[1:, :, :] - _cs_grid[:-1, :, :]).abs().mean().item()
                    print(f"[SPATIAL] state    grad_h={_sg_h:.6f} std={_cs_std:.6f} "
                          f"smoothness={_cs_std/_sg_h:.2f}x")

    def _detect_packing(self, loop_components: List[str]) -> Optional[Dict[str, int]]:
        """
        Detect if Flux-style latent packing is needed.

        DATA-DRIVEN: Compares topology's hidden_states shape (3D packed)
        vs the allocated state variable shape (4D spatial).

        Returns packing info dict if needed, None otherwise.
        The dict contains: {'channels': C, 'height': H, 'width': W}
        where C, H, W are the original 4D spatial dimensions.
        """
        components_data = self.ctx.pkg.topology.get("components", {})
        for comp_name in loop_components:
            comp_shapes = components_data.get(comp_name, {}).get("shapes", {})
            hs_shape = comp_shapes.get("hidden_states", [])
            # 3D hidden_states shape = [B, seq, dim] means packing is needed
            if len(hs_shape) == 3:
                return hs_shape
        return None

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        Pack 4D latents to 3D for Flux-style transformers.

        [B, C, H, W] -> [B, (H/2)*(W/2), C*4]

        Matches diffusers FluxPipeline._pack_latents.
        """
        batch_size, channels, height, width = latents.shape
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, channels: int) -> torch.Tensor:
        """
        Unpack 3D latents back to 4D spatial form.

        [B, seq, packed_dim] -> [B, C, H, W]

        Matches diffusers FluxPipeline._unpack_latents.
        """
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels, height, width)
        return latents

    def _get_component_timestep_scale(self, comp_name: str) -> float:
        """
        Get timestep scale for a loop component.

        The traced graph captures the model's own timestep scaling (e.g.,
        FLUX `timestep * 1000` in forward()). The scheduler already produces
        raw sigmas (0-1 for flow matching). NO compensation needed — the
        graph handles the scaling internally.

        Only applies explicit override from topology if set.
        """
        comp_info = self.ctx.pkg.topology.get("components", {}).get(comp_name, {})
        attrs = comp_info.get("attributes", {})

        explicit = attrs.get("timestep_scale")
        if explicit is not None:
            return float(explicit)

        return 1.0

    def _preprocess_inputs(self) -> None:
        """
        Preprocess user inputs for iterative_process models.

        Delegates to TextProcessor brick for tokenization.
        """
        prompt = None
        for key in ["global.prompt", "prompt"]:
            if key in self.ctx.variable_resolver.resolved:
                prompt = self.ctx.variable_resolver.resolved[key]
                break

        if prompt is None:
            return

        if "tokenizer" not in self.ctx.modules:
            return

        from neurobrix.core.module.text.processor import TextProcessor

        tp = TextProcessor(
            tokenizer=self.ctx.modules["tokenizer"],
            defaults=self.ctx.pkg.defaults,
            topology=self.ctx.pkg.topology,
            variable_resolver=self.ctx.variable_resolver,
        )

        device = self.ctx.primary_device

        try:
            input_ids, attention_mask = tp.tokenize_for_diffusion(prompt, device)

            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            if attention_mask is not None:
                self.ctx.variable_resolver.resolved["global.attention_mask"] = attention_mask
                self.ctx.variable_resolver.resolved["attention_mask"] = attention_mask

        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}") from e
