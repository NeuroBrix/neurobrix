"""Triton Flow Handler — zero torch iterative process (diffusion).

Ported from core/flow/iterative_process.py. Complete separation:
native mode uses torch tensors, triton mode uses NBXTensor throughout.

No torch imports in this file.

Handles diffusion models with denoising loop:
1. Pre-loop components (text encoders)
2. Main loop (transformer with scheduler)
3. Post-loop components (VAE decoder)
"""

import gc
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator, parse_dtype
from neurobrix.core.runtime.debug import DEBUG
from neurobrix.core.flow.base import FlowContext
from neurobrix.triton.cfg import TritonCFGEngine
from neurobrix.triton import i2v_conditioning
from neurobrix.triton import vace_control_conditioning
from neurobrix.triton import flux_video_conditioning


def _vram_probe(tag: str) -> None:
    """Print free/total CUDA VRAM via cudaMemGetInfo, plus the NBX live-bytes
    watermark from the DeviceAllocator (the raw cudaMalloc pool NBXTensor uses).
    Zero torch — the triton path is torch-free end to end. Cheap — only fires
    when NBX_UNLOAD_DIAG=1."""
    import ctypes as _ct
    try:
        _libname = "libcudart.so"
        try:
            _rt = _ct.CDLL(_libname)
        except OSError:
            for _v in (12, 11, 10):
                try:
                    _rt = _ct.CDLL(f"libcudart.so.{_v}")
                    break
                except OSError:
                    continue
            else:
                print(f"[VRAM_PROBE {tag}] libcudart not found", flush=True)
                return
        _free = _ct.c_size_t(0)
        _total = _ct.c_size_t(0)
        _rt.cudaMemGetInfo(_ct.byref(_free), _ct.byref(_total))
        _free_gb = _free.value / 1e9
        _total_gb = _total.value / 1e9
        _used_gb = _total_gb - _free_gb
        try:
            _nbx_alloc_gb = DeviceAllocator.memory_allocated() / 1e9
        except Exception:
            _nbx_alloc_gb = -1.0
        print(f"[VRAM_PROBE {tag}] used={_used_gb:.2f}GB free={_free_gb:.2f}GB "
              f"total={_total_gb:.2f}GB nbx_alloc={_nbx_alloc_gb:.2f}GB",
              flush=True)
    except Exception as _e:
        print(f"[VRAM_PROBE {tag}] failed: {_e}", flush=True)


def _to_nbx(tensor, device_idx: int = 0) -> NBXTensor:
    """Convert torch.Tensor to NBXTensor at the boundary.

    CPU tensors: numpy → from_numpy (H2D via cudaMemcpy, zero torch).
    CUDA tensors: from_raw (pointer wrap, zero copy).
    """
    if isinstance(tensor, NBXTensor):
        return tensor
    if hasattr(tensor, 'data_ptr'):
        # Contiguous-guard at the torch->NBXTensor boundary. The CUDA path
        # (from_raw) wraps the data_ptr with CONTIGUOUS strides derived from
        # shape alone, and the CPU path reads the buffer flat — both ignore a
        # non-contiguous torch input's real strides. A non-contiguous input
        # (e.g. run.py's --input-image `.permute(2,0,1)` channels-last view fed
        # as global.image to an I2V vae_encoder) would then be read as if NCHW-
        # contiguous, interleaving the RGB channels -> the VAE encoder encodes
        # garbage -> wrong I2V conditioning (blue-cast). Materialise first.
        # Zero-cost for already-contiguous tensors; triton-only (compiled
        # unaffected). Mirror of the CLAUDE.md contiguous-guard pattern.
        if hasattr(tensor, 'is_contiguous') and not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if hasattr(tensor, 'is_cuda') and not tensor.is_cuda:
            # CPU → CUDA via numpy path (zero torch)
            DeviceAllocator.set_device(device_idx)
            arr = tensor.detach().numpy()
            return NBXTensor.from_numpy(arr)
        didx = tensor.device.index if hasattr(tensor.device, 'index') and tensor.device.index is not None else device_idx
        return NBXTensor.from_raw(
            tensor.data_ptr(), tuple(tensor.shape),
            parse_dtype(str(tensor.dtype)), 'cuda',
            owns_data=False, device_idx=didx, base=tensor)
    return tensor


class TritonIterativeProcessHandler:
    """
    Flow handler for iterative diffusion process — triton mode.

    Executes the standard diffusion pipeline:
    - pre_loop: Text encoding (with optional CFG negative encoding)
    - loop: Denoising iterations (with optional CFG batching)
    - post_loop: VAE decoding

    ZERO SEMANTIC: No knowledge of "image", "latent", "vae".
    ZERO HARDCODE: All configuration from topology.
    ZERO TORCH: All tensor ops via NBXTensor.
    """

    # Instance attribute for packing info (Flux-style models)
    _packing_info: Optional[Dict[str, int]]

    def __init__(
        self,
        ctx: FlowContext,
        execute_component_fn: Callable[[str, str, Optional[NBXTensor]], Any],
        extract_primary_output_fn: Callable[[str, Any], Any],
        cfg_engine: Optional[TritonCFGEngine] = None,
        output_extractor: Optional[Any] = None
    ):
        """
        Initialize iterative process handler.

        Args:
            ctx: FlowContext with all shared state
            execute_component_fn: Function to execute a component
            extract_primary_output_fn: Function to extract primary output
            cfg_engine: Optional TritonCFGEngine for classifier-free guidance
            output_extractor: Optional OutputExtractor for encoder lookups
        """
        self.ctx = ctx
        self._execute_component = execute_component_fn
        self._extract_primary_output = extract_primary_output_fn
        self._cfg_engine = cfg_engine
        self._output_extractor = output_extractor
        self._packing_info = None
        self._flux_video = False

    def _resolve_as_nbx(self, key: str):
        """Get value from variable_resolver, converting torch.Tensor → NBXTensor."""
        val = self.ctx.variable_resolver.get(key)
        return _to_nbx(val) if val is not None else None

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

        # TEMP DIAG: report loading_mode + persistent_mode + VRAM at boundaries.
        import os as _os_d
        _UNLOAD_DIAG = _os_d.environ.get("NBX_UNLOAD_DIAG") == "1"
        if _UNLOAD_DIAG:
            _plan = self.ctx.plan
            _lm = getattr(_plan, 'loading_mode', 'lazy')
            _pm = bool(getattr(self.ctx, 'persistent_mode', False))
            _strat = getattr(_plan, 'strategy', '?')
            _tdt = getattr(_plan, 'target_dtype', '?')
            print(f"[UNLOAD_DIAG] strategy={_strat!r} loading_mode={_lm!r} "
                  f"target_dtype={_tdt!r} persistent_mode={_pm}",
                  flush=True)
            _cm = getattr(_plan, 'component_memory', {}) or {}
            for _cn, _cmb in _cm.items():
                _mb = getattr(_cmb, 'total_mb', None)
                _dev = '?'
                _a = _plan.components.get(_cn) if hasattr(_plan, 'components') else None
                if _a is not None:
                    _dev = getattr(_a, 'device', '?')
                    _cdt = getattr(_a, 'dtype', '?')
                else:
                    _cdt = '?'
                print(f"[UNLOAD_DIAG]   comp={_cn!r} device={_dev} "
                      f"dtype={_cdt} prism_mb={_mb}", flush=True)
            _vram_probe("entry_to_handler")

        flow = self.ctx.pkg.topology.get("flow", {})

        # CFG settings from engine (already resolved in from_topology)
        do_cfg = self._cfg_engine.is_enabled if self._cfg_engine else False
        guidance_scale = self._cfg_engine.guidance_scale if self._cfg_engine else 1.0

        # 1. Pre-loop execution
        pre_loop = flow.get("pre_loop", [])
        self._execute_pre_loop(pre_loop, do_cfg)

        if _UNLOAD_DIAG:
            _vram_probe("after_pre_loop_returns")

        # 2. Main loop
        loop_def = flow.get("loop", {})
        driver_name = loop_def.get("driver")
        loop_components = loop_def.get("components", [])

        if driver_name and loop_components:
            if _UNLOAD_DIAG:
                _vram_probe("before_main_loop_first_call")
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

            import os as _os
            if _os.environ.get("NBX_DIAG_TRITON_PRELOOP") == "1":
                _keys = [k for k in self.ctx.variable_resolver.resolved
                         if k.startswith(f"{comp_name}.")]
                print(f"   [NBX-DIAG-PRELOOP] {comp_name} stored keys: {_keys}", flush=True)

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
                        hidden_state = _to_nbx(self.ctx.variable_resolver.resolved[candidate])
                        break
                if hidden_state is not None and hidden_key is not None:
                    tokenizer_vals = self._tokenizer_config_with_flags(comp_name, "tokenizer")
                    pos_mask = self._resolve_as_nbx("global.attention_mask")

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
                        if isinstance(t, NBXTensor):
                            _min = t.float().reshape(-1)[0].item()  # approximate
                            _max = t.float().reshape(-1)[-1].item()
                            print(f"[AUDIT] {key}: shape={list(t.shape)} dtype={t.dtype}")

            # Unload weights immediately. Pre_loop components (e.g. the
            # T5 text_encoder ~9.5 GB fp16 for PixArt) are used exactly
            # once per request, so we force-unload even in eager mode —
            # otherwise their VRAM stays resident for the entire
            # transformer loop and a cudaMalloc-based allocator (triton
            # path) fragments into small-alloc failures partway through
            # the DiT. If the caller actually wanted to keep the encoder
            # resident across requests, persistent_mode (serve mode)
            # short-circuits this at the top of _unload_component.
            self._unload_component(comp_name, force=True)

    def _tokenizer_config_with_flags(self, encoder_name: str, tokenizer_name: str) -> Dict[str, Any]:
        """Tokenizer extracted_values augmented with runtime text-conditioning
        flags from the registry. Currently `zero_pad_embeddings` (T5/UMT5
        encoders emit non-zero pad embeddings; a DiT that cross-attends to the
        full sequence must have them zeroed — the vendor pipelines trim+zero-pad).
        Read at runtime via registry_flags (no .nbx rebuild); inert for encoders
        without the flag, so image models are untouched (R23). Mirror of the
        compiled handler (R30) so triton zeroes the UMT5 padding identically."""
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
        pos_hidden_state = self._resolve_as_nbx(f"{text_encoder_name}.last_hidden_state")

        # Set negative inputs temporarily
        self.ctx.variable_resolver.set(input_ids_var, neg_input_ids)
        self.ctx.variable_resolver.set(attention_mask_var, neg_attention_mask)

        # Execute text encoder for negative
        self._execute_component(text_encoder_name, "cfg_negative", None)

        # Get negative embedding
        neg_hidden_state = self._resolve_as_nbx(f"{text_encoder_name}.last_hidden_state")

        # --- PURIFICATION: Finalize embeddings via Handler ---
        executor = self.ctx.executors.get(text_encoder_name)
        handler = getattr(executor, '_component_handler', None) if executor else None
        if handler and hasattr(handler, 'finalize_embeddings'):
            finalized = handler.finalize_embeddings(
                hidden_state=neg_hidden_state,
                attention_mask=_to_nbx(neg_attention_mask),  # REQUIRED for zero_pad_embeddings
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
        # Expose the active scheduler to _get_component_timestep_scale (video
        # FlowEuler denoisers need num_train_timesteps as their timestep scale).
        self._active_driver = driver

        # Get control variables
        num_steps = self.ctx.variable_resolver.get("global.num_inference_steps")

        # Get state variable key from topology
        loop_def = self.ctx.pkg.topology.get("flow", {}).get("loop", {})
        state_key = loop_def.get("state_variable")
        if state_key is None:
            raise RuntimeError(
                "ZERO FALLBACK: 'state_variable' not defined in topology.flow.loop."
            )
        # FLUX-video (Open-Sora) declares a non-standard state_variable
        # (global.img) that the builder does not directly allocate; alias it to
        # the allocated 5D latent so the loop state resolves. Gated on the
        # denoiser taking an img_ids input (FLUX-family only) — inert otherwise.
        # R30 mirror of core/flow/iterative_process.py.
        self._flux_video = flux_video_conditioning.is_flux_video(self.ctx, components)
        if self._flux_video:
            vr = self.ctx.variable_resolver
            _st = vr.resolved.get(state_key)
            if _st is None:
                try:
                    _st = vr.get(state_key)
                except Exception:
                    _st = None
            if _st is None:
                vr.set(state_key, vr.get("global.latents"))

        current_state = self._resolve_as_nbx(state_key)

        # DATA-DRIVEN: Detect if Flux-style packing is needed
        packing_shape = self._detect_packing(components)
        if self._flux_video and current_state.dim() == 5:
            # FLUX-video pack: [B,C,T,H,W] -> [B, T*(H/p)*(W/p), C*p^2]
            self._packing_info = {
                'channels': current_state.shape[1],
                'frames': current_state.shape[2],
                'height': current_state.shape[3],
                'width': current_state.shape[4],
                'ndim': 5,
            }
            current_state = self._pack_latents_5d(current_state)
            self.ctx.variable_resolver.set(state_key, current_state)
        elif packing_shape is not None and current_state.dim() == 4:
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

        # FLUX-video positional ids + cond (Open-Sora T2V): synthesize img_ids /
        # txt_ids / cond into the resolver now that the packed dims are known.
        if self._flux_video and self._packing_info is not None:
            flux_video_conditioning.prepare(self.ctx, current_state, self._packing_info)
        import os as _os_fvd
        if _os_fvd.environ.get("NBX_FVD_DEBUG"):
            _vr = self.ctx.variable_resolver
            print(f"[FVD] flux_video={self._flux_video} packing_info={self._packing_info} "
                  f"state.dim={current_state.dim() if current_state is not None else None}",
                  flush=True)
            for _k in ("global.img", "global.img_ids", "global.txt_ids", "global.cond",
                       "text_encoder.last_hidden_state", "text_encoder_2.pooler_output"):
                _v = _vr.resolved.get(_k)
                print(f"[FVD]   {_k} = {type(_v).__name__}"
                      f"{getattr(_v, 'shape', '') if _v is not None else ' (None)'}", flush=True)

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
                print(f"[Executor] current_state std before scaling: (triton mode, skipped)")
            if noise_scale != 1.0:
                if DEBUG:
                    print(f"[Executor] Scaling initial state by init_noise_sigma = {noise_scale:.4f}")
                current_state = current_state * noise_scale
                self.ctx.variable_resolver.set(state_key, current_state)

        # Get iterator
        iterator = driver.timesteps if hasattr(driver, "timesteps") else range(num_steps)

        encoder_dtype = self._get_encoder_dtype()

        # I2V latent channel-concat conditioning (triton brick, NBXTensor): build
        # the per-step-invariant 20ch condition once (VAE-encoded first frame +
        # frame mask) and store it; applied each step below on the non-CFG path
        # and inside the triton CFG engine (batched/sequential) via
        # global.i2v_condition. R30 mirror of the compiled core flow; inert
        # without the i2v_latent_conditioning registry flag (R23).
        self._i2v_condition = None
        self._i2v_channel_dim = 1
        _loop_comp0 = components[0] if components else None
        _cond_spec = (i2v_conditioning.conditioning_spec(self.ctx, _loop_comp0)
                      if _loop_comp0 else None)
        if _cond_spec is not None:
            self._i2v_channel_dim = int(_cond_spec.get("channel_dim", 1))
            _nf = self.ctx.variable_resolver.get("global.num_frames")
            self._i2v_condition = i2v_conditioning.build_condition(
                self.ctx, _cond_spec, int(_nf))
            if self._i2v_condition is not None:
                self.ctx.variable_resolver.set(
                    i2v_conditioning.CONDITION_VAR, self._i2v_condition)

        # VACE control conditioning (triton brick, NBXTensor): build the denoiser's
        # two extra step-invariant inputs once — control_hidden_states (96ch latent
        # control from the vae_encoder pass) and control_hidden_states_scale
        # (ones(vace_layers)) — and store them as the globals the InputResolver binds
        # by the global.<name> fallback. R30 mirror of core/flow/iterative_process.py;
        # the triton CFG engine repeats the control to batch=2 (scale is batch-
        # invariant). Inert without the vace_control_conditioning registry flag (R23).
        _vace_spec = (vace_control_conditioning.conditioning_spec(self.ctx, _loop_comp0)
                      if _loop_comp0 else None)
        if _vace_spec is not None:
            _ctrl = vace_control_conditioning.build_control(self.ctx, _vace_spec)
            if _ctrl is not None:
                _scale = vace_control_conditioning.build_scale(
                    _vace_spec, _ctrl._device_idx)
                self.ctx.variable_resolver.set(
                    vace_control_conditioning.CONTROL_VAR, _ctrl)
                self.ctx.variable_resolver.set(
                    vace_control_conditioning.SCALE_VAR, _scale)
                self.ctx.variable_resolver.resolved[
                    vace_control_conditioning.CONTROL_VAR] = _ctrl
                self.ctx.variable_resolver.resolved[
                    vace_control_conditioning.SCALE_VAR] = _scale

        # Main loop
        for step_idx, timestep in enumerate(iterator):
            if DEBUG:
                # timestep may be NBXTensor or Python scalar
                ts_val = timestep.item() if isinstance(timestep, NBXTensor) else timestep
                print(f"[Loop] Step {step_idx + 1}/{num_steps} (t={ts_val:.1f})")

            # Store timestep in loop state
            if isinstance(timestep, NBXTensor) and timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            self.ctx.variable_resolver.loop_state[self.ctx.loop_id] = timestep

            for comp_name in components:
                if DEBUG and step_idx == 0:
                    self._audit_component_inputs(comp_name)

                # Scale model input — triton scheduler is NBXTensor-native (zero torch).
                scaled_state = _to_nbx(driver.scale_model_input(current_state, timestep))
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
                    # I2V channel-concat conditioning on the denoiser state
                    # (non-CFG path; the triton CFG engine applies it on its own
                    # batched/sequential paths via global.i2v_condition).
                    if (getattr(self, "_i2v_condition", None) is not None
                            and self._is_loop_component(comp_name)):
                        self.ctx.variable_resolver.set(
                            state_key,
                            i2v_conditioning.apply(scaled_state, self._i2v_condition,
                                                   self._i2v_channel_dim))
                    # Type guard: scaled_timestep may be int/float, convert to NBXTensor if needed
                    timestep_arg: Optional[NBXTensor] = None
                    if isinstance(scaled_timestep, NBXTensor):
                        timestep_arg = scaled_timestep
                    output = self._execute_component(comp_name, "loop", timestep_arg)

                # Update state via driver step
                if output is not None:
                    # Restore original unscaled state
                    self.ctx.variable_resolver.set(state_key, current_state)

                    # Extract primary output
                    model_output = self._extract_primary_output(comp_name, output)

                    # DiT step-0 capture (NBX_DUMP_DIT): raw velocity + the
                    # connection-fed transformer inputs (I2V image_latents arrives
                    # via a connection only). R30 mirror of core/flow/
                    # iterative_process.py, saved as numpy .npz — R33 forbids
                    # torch.save on the triton path. Run with --cfg 1.0 for the
                    # un-guided single forward (batch 1).
                    import os as _os_dit
                    _dd = _os_dit.environ.get("NBX_DUMP_DIT")
                    if _dd and step_idx == 0 and self._is_loop_component(comp_name):
                        import numpy as _np_dit

                        def _np_of(_t):
                            try:
                                return _t.float().numpy() if hasattr(_t, "numpy") else None
                            except Exception:
                                return None

                        _cap = {}
                        _mo = _np_of(model_output)
                        if _mo is not None:
                            _cap["velocity"] = _mo
                        _ci = self.ctx.pkg.topology.get("components", {}).get(comp_name, {})
                        _conns = self.ctx.pkg.topology.get("connections", [])
                        _res = self.ctx.variable_resolver.resolved
                        _in_names = list(_ci.get("interface", {}).get("inputs", []))
                        for _c in _conns:
                            _to = str(_c.get("to", ""))
                            if _to.startswith(f"{comp_name}."):
                                _nm = _to.split(".", 1)[1]
                                if _nm not in _in_names:
                                    _in_names.append(_nm)
                        for _in in _in_names:
                            _cands = [f"{comp_name}.{_in}", f"global.{_in}"]
                            for _c in _conns:
                                if _c.get("to") == f"{comp_name}.{_in}":
                                    _cands.append(_c.get("from"))
                            for _k in _cands:
                                _arr = _np_of(_res.get(_k))
                                if _arr is not None:
                                    _cap[_in] = _arr
                                    break
                        _np_dit.savez(_dd if _dd.endswith(".npz") else _dd + ".npz", **_cap)
                        print(f"[DIT-DUMP-TRITON] keys={list(_cap.keys())} -> {_dd}")

                    # Handle variance prediction split. Channels are axis 1 for
                    # both 4D image [B,C,H,W] and 5D video [B,C,T,H,W]; only the
                    # 3D packed (Flux) layout carries channels on axis 2. Branch
                    # on tensor rank, never on model family (R34) — for 5D the
                    # naive `dim()==4` path mistook axis 2 (T) for channels.
                    # (R30 mirror of core/flow/iterative_process.py.)
                    if current_state.dim() in (4, 5):
                        state_channels = current_state.shape[1]
                        model_channels = model_output.shape[1]
                    else:
                        # 3D packed: dim=2 is the packed channel dim
                        state_channels = current_state.shape[2]
                        model_channels = model_output.shape[2] if model_output.dim() == 3 else model_output.shape[1]
                    if model_channels == 2 * state_channels:
                        split_dim = 2 if current_state.dim() == 3 else 1
                        # chunk(2, dim) equivalent: split in half along dim
                        half = model_output.shape[split_dim] // 2
                        # Use narrow to take the first half
                        model_output = model_output.narrow(split_dim, 0, half)

                    # Handle cross-device transfer
                    if self.ctx.strategy is not None and hasattr(self.ctx.strategy, 'transfer_model_output'):
                        transfer_fn = getattr(self.ctx.strategy, 'transfer_model_output', None)
                        if transfer_fn is not None:
                            model_output = transfer_fn(model_output, current_state)

                    if DEBUG:
                        self._print_step_diagnostics(step_idx, timestep, model_output, current_state)

                    # Driver step — triton scheduler is NBXTensor-native (zero torch).
                    step_result = driver.step(model_output, timestep, current_state)
                    prev = step_result["prev_sample"] if isinstance(step_result, dict) else step_result.prev_sample
                    current_state = _to_nbx(prev)

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
        pre_loop = self.ctx.pkg.topology.get("flow", {}).get("pre_loop", [])
        for comp_name in list(loop_components) + pre_loop:
            import os as _os_dbg
            if _os_dbg.environ.get("NBX_LIVE_DUMP_ON_OOM") == "1":
                from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
                pre_live = sum(_DA._cuda_live_bytes.values())
                self._unload_component(comp_name, force=True)
                post_live = sum(_DA._cuda_live_bytes.values())
                print(f"[UNLOAD] {comp_name}: live {pre_live/1024/1024:.0f}MB → {post_live/1024/1024:.0f}MB "
                      f"(freed {(pre_live-post_live)/1024/1024:.0f}MB)", flush=True)
            else:
                self._unload_component(comp_name, force=True)

        # Get state variable for validation
        loop_def = self.ctx.pkg.topology.get("flow", {}).get("loop", {})
        state_key = loop_def.get("state_variable")

        # Unpack latents if packing was used (Flux-style)
        if state_key and self._packing_info is not None:
            current_state = self._resolve_as_nbx(state_key)
            if current_state is not None and isinstance(current_state, NBXTensor) and current_state.dim() == 3:
                pi = self._packing_info
                assert pi is not None  # Type guard for pyright
                if pi.get('ndim') == 5:
                    # FLUX-video unpack: [B, T*(H/p)*(W/p), C*p^2] -> [B,C,T,H,W]
                    current_state = self._unpack_latents_5d(
                        current_state, pi['frames'], pi['height'], pi['width'],
                        pi['channels']
                    )
                else:
                    current_state = self._unpack_latents(
                        current_state, pi['height'], pi['width'], pi['channels']
                    )
                self.ctx.variable_resolver.set(state_key, current_state)

        # Name-driven latent axis alignment (5D) — R30 mirror of the compiled
        # flow: video models do not share one latent layout (Wan [B,C,T,H,W]
        # end-to-end; CogVideoX denoises [B,T,C,H,W], decodes [B,C,T,H,W]).
        # The permutation derives from existing contract data; identical
        # layouts derive None and nothing changes (R23). NBXTensor.permute +
        # contiguous are R33-pure.
        if state_key and post_loop:
            from neurobrix.core.runtime.resolution.axis_alignment import (
                latent_permutation_for)
            executor = self.ctx.executors.get(post_loop[0])
            dag = getattr(executor, "_dag", None) if executor else None
            if dag:
                perm = latent_permutation_for(self.ctx.pkg.variables, dag)
                if perm is not None:
                    current_state = self._resolve_as_nbx(state_key)
                    if (isinstance(current_state, NBXTensor)
                            and current_state.dim() == len(perm)):
                        self.ctx.variable_resolver.set(
                            state_key,
                            current_state.permute(*perm).contiguous())

        # ZERO FALLBACK: Validate latent shape before VAE
        if state_key:
            current_state = self._resolve_as_nbx(state_key)
            if current_state is not None and isinstance(current_state, NBXTensor):
                # Image: 4D [B, C, H, W], Video: 5D [B, C, T, H, W]
                expected = 5 if current_state.dim() == 5 else 4
                ndim = current_state.dim()
                if ndim != expected:
                    raise RuntimeError(
                        f"ZERO FALLBACK: Latent tensor has {ndim}D shape {list(current_state.shape)}, "
                        f"expected {expected}D before post_loop (VAE)."
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
        # In triton mode, use ctypes cudaDeviceSynchronize + cudaFreeAsync
        # rather than torch.cuda.empty_cache(). The DeviceAllocator handles
        # memory cleanup when tensors go out of scope.

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
                if isinstance(val, NBXTensor):
                    print(f"[AUDIT] {comp_name} input '{inp_name}' ({key}): "
                          f"shape={list(val.shape)} dtype={val.dtype}")
                    break
                elif val is not None:
                    print(f"[AUDIT] {comp_name} input '{inp_name}' ({key}): type={type(val).__name__} val={val}")
                    break

    def _print_step_diagnostics(
        self,
        step_idx: int,
        timestep: Any,
        model_output: NBXTensor,
        current_state: NBXTensor,
    ) -> None:
        """Print per-step diagnostic stats (DEBUG only).

        NOTE: In triton mode, .item()/.float() calls involve cudaMemcpy D2H
        so diagnostics are limited to avoid excessive sync overhead.
        """
        _t_val = timestep.item() if isinstance(timestep, NBXTensor) else timestep
        print(f"[STEP {step_idx+1:2d}] t={_t_val:.4f} | "
              f"output shape={list(model_output.shape)} | "
              f"state shape={list(current_state.shape)}")

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
    def _pack_latents(latents: NBXTensor) -> NBXTensor:
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
    def _unpack_latents(latents: NBXTensor, height: int, width: int, channels: int) -> NBXTensor:
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

    @staticmethod
    def _pack_latents_5d(latents: NBXTensor) -> NBXTensor:
        """Pack a 5D video latent to 3D for FLUX-style video transformers.

        [B, C, T, H, W] -> [B, T*(H/2)*(W/2), C*4]

        NBXTensor mirror of the compiled `_pack_latents_5d` (Open-Sora-v2 vendor
        `pack`: rearrange b c t (h ph) (w pw) -> b (t h w) (c ph pw), ph=pw=2).
        Token order (t h w) matches the FLUX img_ids grid. H,W = latent dims.
        """
        b, c, t, h, w = latents.shape
        latents = latents.view(b, c, t, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)  # b t (h/2) (w/2) c ph pw
        latents = latents.contiguous().reshape(b, t * (h // 2) * (w // 2), c * 4)
        return latents

    @staticmethod
    def _unpack_latents_5d(latents: NBXTensor, frames: int, height: int,
                           width: int, channels: int) -> NBXTensor:
        """Unpack 3D back to 5D video latent (inverse of _pack_latents_5d).

        [B, T*(H/2)*(W/2), C*4] -> [B, C, T, H, W]   (H, W = latent spatial)
        """
        b = latents.shape[0]
        hh, ww = height // 2, width // 2
        latents = latents.view(b, frames, hh, ww, channels, 2, 2)
        latents = latents.permute(0, 4, 1, 2, 5, 3, 6)  # b c t (h/2) ph (w/2) pw
        latents = latents.contiguous().reshape(b, channels, frames, height, width)
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

        # Video flow-match denoisers (e.g. Mochi) consume the scheduler's RAW
        # [0, num_train_timesteps] timesteps: the transformer embeds them
        # directly (no in-graph scaling). NeuroBrix's FlowEuler emits [0,1]
        # sigmas, so the loop must scale by num_train_timesteps or the timestep
        # embedding collapses to ~0 → an unconditioned (divergent) latent. R30
        # mirror of core _get_component_timestep_scale (commit 3770103 part 2),
        # gated on video + FlowEuler so non-flow video schedulers (Wan UniPC,
        # CogVideoX DDIM — already [0,1000]) and image flow (Flux) are untouched.
        driver = getattr(self, "_active_driver", None)
        if (self.ctx.pkg.manifest.get("family") == "video"
                and type(driver).__name__ == "TritonFlowEulerScheduler"
                and getattr(driver, "num_train_timesteps", None)):
            num_train = float(getattr(driver, "num_train_timesteps"))
            # Data-driven discriminator (R30 mirror of compiled, commit 1ad14c1):
            # FLUX/MMDiT denoisers scale the timestep INTERNALLY (in-graph
            # `time_factor` mul ~num_train on the timestep tensor) and expect the
            # scheduler's raw [0,1] sigmas. Applying the [0,num_train] scale here
            # too DOUBLE-scales (sigma*num_train then *time_factor in-graph),
            # blowing past fp16 range → Inf timestep embedding → NaN latent →
            # black (Open-Sora-v2). Mochi-style denoisers have NO in-graph
            # scaling and genuinely need [0,num_train]. Detect per component,
            # never branch on model name.
            if self._graph_scales_timestep_internally(comp_name, num_train):
                return 1.0
            return num_train

        return 1.0

    def _graph_scales_timestep_internally(self, comp_name: str, num_train: float) -> bool:
        """True if the component graph already up-scales the timestep input by
        ~num_train (in-graph `time_factor` mul/div on a 'timestep' tensor).

        R30 mirror of the compiled flow. Pure graph inspection (reads the
        component's graph.json / the loaded DAG) — no execution, no torch, no
        model-name branch. Result cached per component. See
        _get_component_timestep_scale for why this gates the [0,num_train] scale.
        """
        cache = getattr(self, "_ts_inscale_cache", None)
        if cache is None:
            cache = {}
            self._ts_inscale_cache = cache
        if comp_name in cache:
            return cache[comp_name]

        result = False
        try:
            ops = None
            executor = self.ctx.executors.get(comp_name) if getattr(
                self.ctx, "executors", None) else None
            dag = getattr(executor, "_dag", None) if executor is not None else None
            if isinstance(dag, dict):
                ops = dag.get("ops")
            if not ops:
                import json as _json
                gp = self.ctx.pkg.cache_path / "components" / comp_name / "graph.json"
                if gp.exists():
                    with open(gp) as _f:
                        ops = _json.load(_f).get("ops", {})
            if ops:
                lo, hi = num_train * 0.5, num_train * 2.0
                for op in ops.values():
                    if op.get("op_type") not in ("aten::mul", "aten::div"):
                        continue
                    ins = op.get("inputs", op.get("input_tensor_ids", [])) or []
                    if not any("timestep" in str(t).lower() for t in ins):
                        continue
                    for a in op.get("attributes", {}).get("args", []) or []:
                        if isinstance(a, dict) and a.get("type") == "scalar":
                            v = abs(float(a.get("value", 0.0)))
                            if lo <= v <= hi:
                                result = True
                                break
                    if result:
                        break
        except Exception:
            result = False

        cache[comp_name] = result
        return result

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
