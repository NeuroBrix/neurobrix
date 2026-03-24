"""VibeVoice-1.5B native stage handlers.

VibeVoice uses a DDPM diffusion denoising loop and a native acoustic decoder
(ConvNext1d architecture) that can't run through CompiledSequence because the
full traced graph covers encode+decode but TTS only needs the decoder half.

All functions take the AudioEngine instance (``engine``) as first parameter.
"""

import gc
import time
import torch
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# Public entry points (called from AudioEngine.execute)
# ─────────────────────────────────────────────────────────────

def execute_diffusion_stage(engine, stage: Dict, audio_config: Dict) -> None:
    """Execute a diffusion denoising stage (VibeVoice prediction_head).

    Runs an iterative denoising loop using a scheduler module.
    Condition comes from a previously-executed component's output.
    Reuses existing scheduler infrastructure (DDIM/DPM++ from scheduler factory).

    Stage config expects:
        diffusion.num_inference_steps: int (default from defaults.json)
        diffusion.condition_from: str (component name for condition tensor)
        diffusion.latent_shape: [T, D] (noise shape, from graph input)
    """
    comp_name = stage["component"]
    device = engine.ctx.primary_device
    defaults = engine.ctx.pkg.defaults

    print(f"   [{comp_name}] Running diffusion denoising...")
    start = time.perf_counter()
    engine._ensure_weights_loaded(comp_name)

    # -- Get diffusion config (ALL values from defaults.json, ZERO FALLBACK) --
    diffusion_cfg = stage.get("diffusion", {})
    num_steps = diffusion_cfg.get("num_inference_steps",
                defaults.get("ddpm_num_inference_steps"))
    if num_steps is None:
        raise RuntimeError(
            "ZERO FALLBACK: ddpm_num_inference_steps missing from defaults.json.\n"
            "Builder must extract from model's diffusion config."
        )

    # -- Get or create scheduler --
    scheduler = engine.ctx.modules.get("scheduler")
    if scheduler is None:
        from neurobrix.core.module.scheduler.factory import SchedulerFactory
        sched_config = defaults.get("scheduler_config", {})
        # All scheduler params MUST be in defaults.json
        for key, default_key in [
            ("_class_name", "scheduler_type"),
            ("num_train_timesteps", "ddpm_num_steps"),
            ("prediction_type", "prediction_type"),
            ("beta_schedule", "ddpm_beta_schedule"),
        ]:
            if key not in sched_config:
                val = defaults.get(default_key)
                if val is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: '{default_key}' missing from defaults.json.\n"
                        f"Builder must extract diffusion scheduler config from model."
                    )
                sched_config[key] = val
        scheduler = SchedulerFactory.create(sched_config)

    scheduler.set_timesteps(num_steps, device=torch.device(device))

    # -- Determine latent shape from graph input --
    executor = engine.ctx.executors[comp_name]
    dag = getattr(executor, '_dag', None)
    latent_shape = None
    condition_input_name = None
    noisy_input_name = None
    if dag:
        for _tid, spec in dag.get("tensors", {}).items():
            iname = spec.get("input_name")
            if iname and "noisy" in iname:
                noisy_input_name = iname
                latent_shape = spec.get("shape", [])
            elif iname and "condition" in iname:
                condition_input_name = iname

    if latent_shape is None:
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine latent shape from {comp_name} graph.\n"
            f"Expected a graph input with 'noisy' in its name."
        )

    # Resolve symbolic dims to concrete values
    concrete_shape = []
    for s in latent_shape:
        if isinstance(s, dict):
            tv = s.get("trace_value")
            if tv is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: Symbolic dim in {comp_name} graph has no trace_value."
                )
            concrete_shape.append(tv)
        elif isinstance(s, int):
            concrete_shape.append(s)
        else:
            raise RuntimeError(
                f"ZERO FALLBACK: Unexpected dim type {type(s)} in {comp_name} graph."
            )

    # -- Get condition tensor from previous stage output --
    condition_from = diffusion_cfg.get("condition_from")
    if condition_from is None:
        raise RuntimeError(
            "ZERO FALLBACK: Diffusion stage requires 'condition_from' in topology.\n"
            "Specifies which component provides the conditioning tensor."
        )
    condition = engine._get_component_output(condition_from)
    if condition is None:
        raise RuntimeError(
            f"ZERO FALLBACK: Condition source '{condition_from}' produced no output."
        )

    # Reshape condition to match graph expectation [1, T, hidden]
    if condition is not None and condition.dim() == 2:
        condition = condition.unsqueeze(0)
    # Ensure condition sequence length matches latent sequence length
    if condition is not None and len(concrete_shape) == 3:
        target_seq = concrete_shape[1]
        if condition.shape[1] != target_seq:
            if condition.shape[1] > target_seq:
                condition = condition[:, :target_seq, :]
            else:
                pad = torch.zeros(
                    condition.shape[0], target_seq - condition.shape[1], condition.shape[2],
                    device=condition.device, dtype=condition.dtype
                )
                condition = torch.cat([condition, pad], dim=1)

    # -- Initialize noise --
    dtype = engine._get_compute_dtype()
    noisy = torch.randn(concrete_shape, device=device, dtype=dtype)

    # -- Bind condition to variable resolver --
    if condition is not None and condition_input_name:
        engine.ctx.variable_resolver.resolved[f"{comp_name}.{condition_input_name}"] = condition.to(device, dtype=dtype)
        engine.ctx.variable_resolver.resolved[condition_input_name] = condition.to(device, dtype=dtype)

    # -- Denoising loop --
    # Call executor.run() directly with all inputs -- _execute_component only
    # resolves from topology connections (condition) but noisy_images and
    # timesteps are runtime-generated per step.
    executor = engine.ctx.executors[comp_name]
    print(f"   [{comp_name}] Diffusion: {num_steps} steps, latent {concrete_shape}")
    for step_idx, t in enumerate(scheduler.timesteps):
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t_input = t.unsqueeze(0).to(device)
        else:
            t_input = torch.tensor([t], device=device, dtype=torch.long)

        # Scale model input (identity for DDIM)
        scaled_noisy = scheduler.scale_model_input(noisy, t)

        # Build complete inputs dict for this step
        comp_inputs = {}
        if noisy_input_name:
            comp_inputs[noisy_input_name] = scaled_noisy
        comp_inputs["timesteps"] = t_input
        if condition is not None and condition_input_name:
            comp_inputs[condition_input_name] = condition.to(device, dtype=dtype)

        # Execute prediction head directly
        output = executor.run(comp_inputs)

        # Extract model output
        if isinstance(output, dict):
            model_output = next(iter(output.values()))
        elif isinstance(output, torch.Tensor):
            model_output = output
        else:
            model_output = engine._get_component_output(comp_name)
        if model_output is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Diffusion stage '{comp_name}' produced no output."
            )

        # Scheduler step
        step_result = scheduler.step(model_output, t, noisy)
        if isinstance(step_result, dict):
            noisy = step_result["prev_sample"]
        else:
            noisy = step_result.prev_sample

    # -- Apply speech scaling (VibeVoice-specific, DATA-DRIVEN from defaults) --
    speech_scaling = defaults.get("speech_scaling_factor")
    speech_bias = defaults.get("speech_bias_factor")
    if speech_scaling is not None and speech_bias is not None:
        noisy = noisy / speech_scaling - speech_bias

    # -- Store denoised output --
    engine.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = noisy
    engine.ctx.variable_resolver.resolved["global.denoised_latent"] = noisy

    elapsed = (time.perf_counter() - start) * 1000
    print(f"   [{comp_name}] Diffusion done in {elapsed:.0f}ms")

    if not engine.ctx.persistent_mode:
        engine._unload_component_weights(comp_name)
        gc.collect()
        torch.cuda.empty_cache()


def execute_native_acoustic_decoder(engine, stage: Dict, audio_config: Dict) -> None:
    """Run acoustic tokenizer decoder natively.

    The traced graph covers the full encode+decode path (waveform->waveform),
    but TTS only needs the decoder (latent->waveform). Extract decoder weights
    and run the ConvNext1d architecture natively.

    Architecture params (decoder_depths, decoder_upsampling_ratios) read from
    defaults.json -- set by builder from model's config.json.

    TODO: Retrace only the decoder portion as a separate component so this
    can run through CompiledSequence as a standard forward pass.
    """
    comp_name = stage["component"]
    device = engine.ctx.primary_device
    dtype = engine._get_compute_dtype()

    # Get denoised latent from diffusion stage
    latent = engine.ctx.variable_resolver.resolved.get("global.denoised_latent")
    if latent is None:
        raise RuntimeError(
            "ZERO FALLBACK: native_acoustic_decoder requires global.denoised_latent "
            "from a preceding diffusion stage."
        )

    print(f"   [{comp_name}] Running native acoustic decoder...")
    start = time.perf_counter()

    engine._ensure_weights_loaded(comp_name)
    w = engine.ctx.executors[comp_name]._weights

    # Diffusion outputs [B, T, latent_dim=64], decoder expects [B, C=64, T]
    x = latent.to(device=device, dtype=dtype)
    if x.shape[1] != 64 or (x.dim() == 3 and x.shape[2] == 64 and x.shape[1] != 64):
        x = x.transpose(1, 2)

    # Decoder config (DATA-DRIVEN from defaults.json)
    defaults = engine.ctx.pkg.defaults
    upsample_strides = defaults.get("decoder_upsampling_ratios")
    stage_depths = defaults.get("decoder_depths")
    if upsample_strides is None or stage_depths is None:
        raise RuntimeError(
            "ZERO FALLBACK: decoder_upsampling_ratios and decoder_depths "
            "missing from defaults.json.\n"
            "Builder must extract from model's config.json."
        )

    with torch.inference_mode():
        # -- Stem: CausalConv1d(64->2048, k=7) --
        x = _vv_causal_conv1d(x, w,
            "decoder.upsample_layers.0.0.conv.conv", kernel_size=7)

        # -- Stem stage: 8 ConvNext blocks at 2048 --
        for blk in range(stage_depths[0]):
            x = _vv_convnext_block(x, w, f"decoder.stages.0.{blk}")

        # -- Upsample stages 1-5 (ConvTranspose1d) --
        for i, stride in enumerate(upsample_strides[:5]):
            kernel_size = stride * 2
            x = _vv_causal_conv_transpose1d(
                x, w, f"decoder.upsample_layers.{i+1}.0.convtr.convtr",
                stride=stride, kernel_size=kernel_size)
            for blk in range(stage_depths[i + 1]):
                x = _vv_convnext_block(x, w, f"decoder.stages.{i+1}.{blk}")

        # -- Last upsample (conv_layers.0, stride=2) --
        x = _vv_causal_conv_transpose1d(
            x, w, "decoder.conv_layers.0.convtr.convtr",
            stride=upsample_strides[5], kernel_size=upsample_strides[5] * 2)
        for blk in range(stage_depths[6]):
            x = _vv_convnext_block(x, w, f"decoder.stages.6.{blk}")

        # -- Head: CausalConv1d(32->1, k=7) --
        x = _vv_causal_conv1d(x, w, "decoder.head.conv.conv", kernel_size=7)

    elapsed = (time.perf_counter() - start) * 1000
    print(f"   [{comp_name}] Decoder done in {elapsed:.0f}ms  output={x.shape}")

    # Store waveform output
    engine.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = x
    engine.ctx.variable_resolver.resolved["global.output_audio"] = x

    if not engine.ctx.persistent_mode:
        engine._unload_component_weights(comp_name)
        gc.collect()
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# Internal helpers (ConvNext1d building blocks)
# ─────────────────────────────────────────────────────────────

def _vv_causal_conv1d(
    x: torch.Tensor, w: Dict, prefix: str,
    kernel_size: int = 7, stride: int = 1, groups: int = 1,
) -> torch.Tensor:
    """CausalConv1d: left-pad then Conv1d."""
    weight = w[f"{prefix}.weight"].to(device=x.device, dtype=x.dtype)
    bias = w.get(f"{prefix}.bias")
    if bias is not None:
        bias = bias.to(device=x.device, dtype=x.dtype)
    causal_pad = (kernel_size - 1) - (stride - 1)
    if causal_pad > 0:
        x = torch.nn.functional.pad(x, (causal_pad, 0))
    return torch.nn.functional.conv1d(x, weight, bias, stride=stride, groups=groups)


def _vv_causal_conv_transpose1d(
    x: torch.Tensor, w: Dict, prefix: str,
    stride: int, kernel_size: int,
) -> torch.Tensor:
    """CausalConvTranspose1d: ConvTranspose1d then trim right padding."""
    weight = w[f"{prefix}.weight"].to(device=x.device, dtype=x.dtype)
    bias = w.get(f"{prefix}.bias")
    if bias is not None:
        bias = bias.to(device=x.device, dtype=x.dtype)
    x = torch.nn.functional.conv_transpose1d(x, weight, bias, stride=stride)
    padding_total = kernel_size - stride
    if padding_total > 0:
        x = x[..., :-padding_total]
    return x


def _vv_convnext_block(
    x: torch.Tensor, w: Dict, prefix: str,
) -> torch.Tensor:
    """ConvNext1d block: mixer path (norm->depthwise_conv->gamma) + FFN path.

    Matches VibeVoiceAcousticTokenizerConvNext1dLayer exactly:
      mixer: residual + gamma * causal_depthwise_conv(rms_norm(x))
      ffn:   residual + ffn_gamma * linear2(gelu(linear1(ffn_norm(x))))
    """
    dev, dt = x.device, x.dtype

    def get(name):
        return w[f"{prefix}.{name}"].to(device=dev, dtype=dt)

    # -- Mixer path --
    residual = x
    channels = x.shape[1]
    # RMSNorm over last dim: transpose [B,C,T]->[B,T,C], norm, transpose back
    h = _vv_rms_norm(x.transpose(1, 2), get("norm.weight")).transpose(1, 2)
    # Depthwise causal conv (groups=channels, kernel derived from weight)
    mixer_w = get("mixer.conv.conv.conv.weight")
    mixer_b = get("mixer.conv.conv.conv.bias")
    causal_pad = mixer_w.shape[2] - 1
    h = torch.nn.functional.pad(h, (causal_pad, 0))
    h = torch.nn.functional.conv1d(h, mixer_w, mixer_b, groups=channels)
    h = h * get("gamma").unsqueeze(-1)
    x = residual + h

    # -- FFN path --
    residual = x
    h = _vv_rms_norm(x.transpose(1, 2), get("ffn_norm.weight"))
    # Linear1 -> GELU -> Linear2  (operates on [B,T,C])
    h = torch.nn.functional.linear(h, get("ffn.linear1.weight"), get("ffn.linear1.bias"))
    h = torch.nn.functional.gelu(h)
    h = torch.nn.functional.linear(h, get("ffn.linear2.weight"), get("ffn.linear2.bias"))
    h = h.transpose(1, 2)  # back to [B,C,T]
    h = h * get("ffn_gamma").unsqueeze(-1)
    x = residual + h

    return x


def _vv_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return weight * x_normed.to(x.dtype)
