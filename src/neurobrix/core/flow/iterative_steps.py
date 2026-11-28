"""The iterative process's steps both engine branches run alike — written once (Hocine Benkelaya 2026-10-29 22:39: no duplication).

The compiled handler (`core/flow/iterative_process.IterativeProcessHandler`) and the Triton one
(`triton/flow/iterative_process.TritonIterativeProcessHandler`) carried these eight methods twice, body for body. They
name no tensor library: the latents they pack and unpack are the branch's own tensors, reshaped through the methods
both tensor types offer. Torch-free at import (the Triton branch inherits it: R33).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from neurobrix.core.runtime.debug import DEBUG
import logging
_log = logging.getLogger(__name__)


def _dump_final_latent(tensor, to_numpy):
    """Write the final post-denoise latent to `NBX_DUMP_FINAL_LATENT=<dir>`, if armed.

    The latent that ENTERS the decoder, dumped at the one boundary where it is
    known and named — the `pre_vae` gate, beside the validator that already
    stands here. Its mirror at the other end of the run is
    `NBX_DUMP_INIT_LATENT`, which writes the noise the denoise starts from.

    It exists because inferring this tensor from the first decode op requires
    assuming a normalisation convention read from the vendor's VAE config, and a
    13 % gap measured under an unverified assumption is not a measurement — it is
    the assumption, wearing a number. With this the comparison is direct.

    Diagnostic, default off. It writes what the flow already computed.
    """
    import os
    d = os.environ.get("NBX_DUMP_FINAL_LATENT")
    if not d or tensor is None:
        return
    import numpy as np
    try:
        arr = to_numpy()
    except Exception as exc:                    # a dump must never break a run
        _log.debug(f"[final-latent] not dumped ({exc})")
        return
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "final_latent.npy")
    np.save(path, np.ascontiguousarray(arr))
    _log.debug(f"[final-latent] {tuple(arr.shape)} {arr.dtype} l2={float(np.linalg.norm(arr)):.4f}"
               f" -> {path}")


class IterativeSteps:
    """The steps of the iterative process both branches share; each handler inherits them."""

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

    def _progress_label(self) -> str:
        """Name the run in its heartbeat, so a terminal with several renders in
        it says which one is moving."""
        # The name lives in the MANIFEST, not the topology — the topology
        # describes the graph, the manifest describes the artifact.
        for src in ("manifest", "topology"):
            try:
                blob = getattr(self.ctx.pkg, src, None)
                name = (blob or {}).get("model_name") or (blob or {}).get("name")
                if name:
                    return str(name)
            except Exception:
                continue
        return "render"

    def _is_loop_component(self, comp_name: str) -> bool:
        """Check if component is in the main loop."""
        loop_def = self.ctx.pkg.topology.get("flow", {}).get("loop", {})
        return comp_name in loop_def.get("components", [])

    def _setup_dual_denoiser(self, components: List[str], driver: Any) -> Optional[dict]:
        """Dual-denoiser (Wan2.2-A14B MoE-of-experts) boundary switch.

        Returns {high, low, boundary} or None for single-denoiser models.
        R30 mirror of core/flow/iterative_process.py._setup_dual_denoiser —
        pure metadata reads (defaults / topology roles / driver), zero torch.

        boundary_ratio (defaults.json, from the registry dual-expert config)
        gives boundary_timestep = ratio * num_train_timesteps. The HIGH-noise
        expert runs for timesteps >= boundary, the LOW-noise expert for
        timesteps < boundary. Expert ordering prefers the component role
        (denoiser = high-noise, denoiser_low_noise = low-noise); falls back to
        topology loop order. GATED: None unless boundary_ratio is set AND
        >= 2 loop denoisers exist -> inert for every single-denoiser model.
        """
        br = self.ctx.pkg.defaults.get("boundary_ratio")
        if not br:
            return None
        loop_denoisers = [c for c in components if self._is_loop_component(c)]
        if len(loop_denoisers) < 2:
            return None
        comp_meta = self.ctx.pkg.topology.get("components", {})
        roles = {c: comp_meta.get(c, {}).get("role") for c in loop_denoisers}
        high = next((c for c in loop_denoisers if roles.get(c) == "denoiser"), None)
        low = next((c for c in loop_denoisers
                    if roles.get(c) in ("denoiser_low_noise", "denoiser_2")), None)
        if high is None or low is None:
            high, low = loop_denoisers[0], loop_denoisers[1]
        num_train = float(getattr(driver, "num_train_timesteps", 1000) or 1000)
        boundary = float(br) * num_train
        if DEBUG:
            _log.debug(f"[DualDenoiser] high={high} low={low} "
                       f"boundary={boundary:.1f} (ratio={br} x {num_train})")
        return {"high": high, "low": low, "boundary": boundary}

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
                            _log.debug(f"[Executor] Encoder dtype from Prism plan: {comp}={dtype}")
                        return dtype

            # Priority 2: any component's dtype
            for comp_name, alloc in plan_components.items():
                dtype = getattr(alloc, 'dtype', None)
                if dtype is not None:
                    if DEBUG:
                        _log.debug(f"[Executor] Using {comp_name} dtype as encoder_dtype: {dtype}")
                    return dtype

        raise RuntimeError(
            "ZERO FALLBACK: encoder_dtype not found in Prism plan.\n"
            "Expected: plan.components[encoder].dtype"
        )

    @staticmethod
    def _pack_latents(latents):
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
    def _unpack_latents(latents, height: int, width: int, channels: int):
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

    def _preprocess_inputs(self) -> None:
        """
        Preprocess user inputs for iterative_process models (and the compiled
        forward_pass handler, which takes this method).

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


    def _resolve_loop_state(self, state_key: str):
        """Resolve the loop state variable, aliasing to the declared state_tensor when absent."""
        from neurobrix.core.runtime.resolution.list_batched_denoiser import resolve_state_variable_alias
        return resolve_state_variable_alias(self.ctx, state_key)

    def _resolve_encoder_hidden_key(self, enc_name: str) -> str:
        """Resolve the topology-declared hidden-state variable key for a pre-loop text encoder."""
        from neurobrix.core.runtime.resolution.list_batched_denoiser import resolve_encoder_hidden_key
        return resolve_encoder_hidden_key(self.ctx, enc_name)

    def _flow_component_convention(self, comp_name: str, driver, do_cfg: bool):
        """Cached (list_batched_spec, flow_step_convention) for a loop component."""
        cache = getattr(self, "_flow_conv_cache", None)
        if cache is None:
            cache = {}
            self._flow_conv_cache = cache
        if comp_name not in cache:
            from neurobrix.core.runtime.resolution.list_batched_denoiser import (
                list_batched_spec, flow_step_convention,
            )
            lspec = list_batched_spec(self.ctx, comp_name) if self._is_loop_component(comp_name) else None
            conv = flow_step_convention(self.ctx, comp_name, driver, list_spec=lspec)
            cache[comp_name] = (lspec, conv)
        return cache[comp_name]

    def _execute_list_batched_step(
        self, comp_name: str, scaled_state, scaled_timestep,
        do_cfg: bool, guidance_scale: float, list_spec: Dict[str, Any], flow_conv: Dict[str, Any],
    ):
        """Stage dot-indexed inputs (x.0 / cap_feats.0 / t), run denoiser pass(es), unpack & apply CFG."""
        from neurobrix.core.runtime.resolution.list_batched_denoiser import (
            extract_caption_embeddings, build_list_batched_passes, unpack_list_batched_output,
        )
        caps_cache = getattr(self, "_list_caps_cache", None)
        if caps_cache is None:
            pos_caps, neg_caps = extract_caption_embeddings(self.ctx, do_cfg, comp_name=comp_name, cond_prefix=list_spec.get("cond_prefix"))
            self._list_caps_cache = (pos_caps, neg_caps)
        else:
            pos_caps, neg_caps = caps_cache

        passes = build_list_batched_passes(
            scaled_state, scaled_timestep, pos_caps, neg_caps, list_spec, do_cfg,
        )
        vr = self.ctx.variable_resolver
        pass_outputs = []
        for staged, pass_t in passes:
            staged_keys = []
            try:
                for name, val in staged.items():
                    k_comp = f"{comp_name}.{name}"
                    vr.resolved[k_comp] = val
                    staged_keys.append(k_comp)
                out = self._execute_component(comp_name, "loop", pass_t)
                pass_outputs.append(out)
            finally:
                for k_comp in staged_keys:
                    vr.resolved.pop(k_comp, None)

        return unpack_list_batched_output(
            pass_outputs, list_spec, do_cfg, guidance_scale, flow_conv["cfg_formula"],
        )


    def _step_timestep_and_convention(self, comp_name: str, timestep, driver, do_cfg: bool):
        """Scale timestep + apply flow convention; return (scaled_timestep, list_spec, flow_conv)."""
        from neurobrix.core.runtime.resolution.list_batched_denoiser import apply_model_timestep_convention
        comp_scale = self._get_component_timestep_scale(comp_name)
        scaled_ts = timestep * comp_scale if comp_scale != 1.0 else timestep
        list_spec, flow_conv = self._flow_component_convention(comp_name, driver, do_cfg)
        return apply_model_timestep_convention(scaled_ts, flow_conv), list_spec, flow_conv

    def _run_loop_step_component(
        self, comp_name: str, state_key: str, scaled_state, scaled_timestep,
        do_cfg: bool, guidance_scale: float, encoder_dtype, list_spec, flow_conv,
        i2v_apply_fn, tensor_cls,
    ):
        """Run one loop component (list-batched, CFG, or standard non-CFG)."""
        if list_spec is not None:
            return self._execute_list_batched_step(
                comp_name, scaled_state, scaled_timestep, do_cfg, guidance_scale, list_spec, flow_conv,
            )
        if do_cfg and self._is_loop_component(comp_name) and self._cfg_engine:
            return self._cfg_engine.execute_component_with_cfg(
                comp_name, scaled_state, scaled_timestep, guidance_scale, encoder_dtype=encoder_dtype,
            )
        if getattr(self, "_i2v_condition", None) is not None and self._is_loop_component(comp_name):
            self.ctx.variable_resolver.set(
                state_key, i2v_apply_fn(scaled_state, self._i2v_condition, self._i2v_channel_dim),
            )
        ts_arg = scaled_timestep if isinstance(scaled_timestep, tensor_cls) else None
        return self._execute_component(comp_name, "loop", ts_arg)

    @staticmethod
    def _apply_step_velocity_sign(model_output, flow_conv):
        from neurobrix.core.runtime.resolution.list_batched_denoiser import apply_velocity_sign
        return apply_velocity_sign(model_output, flow_conv)


def required_num_steps(ctx):
    """The step count the request or the container declares; refuses by name, the container named, when neither does."""
    try:
        return ctx.variable_resolver.get("global.num_inference_steps")
    except KeyError:
        from neurobrix.core.runtime_values import MissingRuntimeValue
        raise MissingRuntimeValue(
            f"'num_inference_steps' is required to run this model and nothing declares it: container "
            f"{ctx.pkg.manifest.get('model_name') or '?'!r} has none in runtime/defaults.json and the "
            f"request names none. Declare it in the container, or pass it with the request.") from None
