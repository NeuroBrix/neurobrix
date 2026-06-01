"""Triton EncoderDecoderEngine — zero torch encoder-decoder flow.

Ported from core/flow/encoder_decoder.py. Handles models like Whisper:
encoder processes input features, decoder generates tokens autoregressively
with cross-attention from encoder output.

No torch imports in this file (except at audio preprocessing boundary).
"""

import gc
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class TritonEncoderDecoderEngine:
    """
    Triton-mode encoder-decoder cross-attention flow.

    topology.flow.audio:
        direction: stt
        stages:
          - component: model.encoder
            execution: forward
          - component: model.decoder
            execution: autoregressive
            cross_attention_from: model.encoder
            logits_source: embed_weight_tied | self | lm_head
    """

    def __init__(
        self,
        ctx,
        execute_component_fn: Callable,
        resolve_inputs_fn: Callable,
        ensure_weights_fn: Callable,
        unload_weights_fn: Callable,
    ):
        self.ctx = ctx
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

    def execute(self) -> Dict[str, Any]:
        """Execute encoder-decoder pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults

        # -- Step 1: Preprocess audio input (BOUNDARY — uses torch internally) --
        from neurobrix.core.flow.audio_utils import preprocess_audio_input
        preprocess_audio_input(self.ctx, audio_config, stages)

        # -- Step 2: Forward encoder --
        encoder_stage = None
        decoder_stage = None
        for s in stages:
            exec_type = s.get("execution", "forward")
            if exec_type == "forward":
                encoder_stage = s
            elif exec_type == "autoregressive":
                decoder_stage = s

        if encoder_stage is None or decoder_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: encoder_decoder flow requires one 'forward' stage "
                "(encoder) and one 'autoregressive' stage (decoder)."
            )

        enc_name = encoder_stage["component"]
        dec_name = decoder_stage["component"]

        print(f"   [{enc_name}] Running encoder...")
        start = time.perf_counter()
        self._ensure_weights_loaded(enc_name)
        self._execute_component(enc_name, "forward", None)
        enc_elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{enc_name}] Done in {enc_elapsed:.0f}ms")

        # Store encoder output for cross-attention
        encoder_output = _get_component_output(self.ctx, enc_name)
        if encoder_output is not None:
            self.ctx.variable_resolver.resolved[f"{enc_name}.output_0"] = encoder_output

        if not self.ctx.persistent_mode:
            self._unload_component_weights(enc_name)
            gc.collect()

        # -- Step 3: Autoregressive decode with cross-attention --
        max_tokens = defaults.get("max_tokens")
        if max_tokens is None:
            raise RuntimeError("ZERO FALLBACK: max_tokens missing from defaults.json.")
        temperature = defaults.get("temperature")
        if temperature is None:
            raise RuntimeError("ZERO FALLBACK: temperature missing from defaults.json.")
        eos_token_id = defaults.get("eos_token_id")
        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        decoder_start_token_id = defaults.get("decoder_start_token_id", eos_token_id)
        logits_source = decoder_stage.get("logits_source", "embed_weight_tied")
        repetition_penalty = defaults.get("repetition_penalty", 1.0)

        # Forced decoder IDs (language/task tokens for Whisper)
        forced_decoder_ids = defaults.get("forced_decoder_ids", [])
        forced_map = {pos: tid for pos, tid in forced_decoder_ids}

        print(f"   [{dec_name}] Generating tokens (max={max_tokens})...")
        start = time.perf_counter()
        self._ensure_weights_loaded(dec_name)

        # Get embed weight for weight-tied logits
        embed_weight = _get_embed_weight(self.ctx, dec_name)

        # Inject embed weight for weight-tied models
        if embed_weight is not None:
            executor = self.ctx.executors.get(dec_name)
            if executor is not None and hasattr(executor, '_weights'):
                dag = getattr(executor, '_dag', None)
                if dag:
                    tensors = dag.get("tensors", {})
                    for tied_name in ("head.weight", "model.token_embed.weight"):
                        if tied_name not in executor._weights and f"param::{tied_name}" in tensors:
                            executor._weights[tied_name] = embed_weight

        device_idx = _parse_device_idx(self.ctx.primary_device)
        DeviceAllocator.set_device(device_idx)
        generated_ids = [decoder_start_token_id]

        for step in range(1, max_tokens):
            # Build input_ids as NBXTensor
            ids_np = np.array([generated_ids], dtype=np.int64)
            input_ids = NBXTensor.from_numpy(ids_np)
            self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
            self.ctx.variable_resolver.resolved["input_ids"] = input_ids

            self._execute_component(dec_name, "forward", None)

            decoder_output = _get_component_output(self.ctx, dec_name)
            if decoder_output is None:
                break

            logits = _compute_logits(
                self.ctx, decoder_output, embed_weight, logits_source
            )

            current_pos = len(generated_ids)
            if current_pos in forced_map and forced_map[current_pos] is not None:
                next_token = forced_map[current_pos]
            else:
                next_token = _sample_token_nbx(
                    logits, temperature,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                )

            generated_ids.append(next_token)
            if next_token == eos_token_id:
                break

        dec_elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{dec_name}] Generated {len(generated_ids)} tokens in {dec_elapsed:.0f}ms")

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = generated_ids

        if not self.ctx.persistent_mode:
            self._unload_component_weights(dec_name)
            gc.collect()

        # -- Step 4: Decode tokens to text --
        from neurobrix.core.flow.audio_utils import postprocess_text_output
        postprocess_text_output(self.ctx)

        return self.ctx.variable_resolver.resolve_all()


# -----------------------------------------------------------------
# Module-level helpers (zero torch)
# -----------------------------------------------------------------

def _get_component_output(ctx, comp_name) -> Optional[NBXTensor]:
    """Get a component's primary output tensor."""
    resolved = ctx.variable_resolver.resolved
    for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state", f"{comp_name}.output"]:
        if key in resolved:
            val = resolved[key]
            if isinstance(val, NBXTensor) or _is_tensor(val):
                return val
    return None


def _get_embed_weight(ctx, comp_name):
    """Get TOKEN embedding weight for weight-tied logits.

    NeuroTax standard: token_embed.weight (token embeddings).
    """
    executor = ctx.executors.get(comp_name)
    if executor is not None and hasattr(executor, '_weights'):
        for key in executor._weights:
            if "token_embed" in key:
                return executor._weights[key]
        best = None
        for key in executor._weights:
            if "embed" in key and executor._weights[key].ndim == 2:
                w = executor._weights[key]
                if best is None or w.shape[0] > best.shape[0]:
                    best = w
        return best
    return None


def _compute_logits(ctx, hidden_states, embed_weight, logits_source):
    """Compute logits from hidden states.

    For triton mode, we use the executor.run() path where possible.
    For weight-tied logits, we do a manual matmul via the kernel dispatcher.
    """
    # Extract last hidden state
    if hasattr(hidden_states, 'shape') and len(hidden_states.shape) >= 3:
        # hidden_states[:, -1:, :]
        last_dim = hidden_states.shape[1]
        # Use select to get last position
        last_hidden = hidden_states.select(1, last_dim - 1).unsqueeze(1)
    else:
        last_hidden = hidden_states

    if logits_source == "lm_head" and "lm_head" in ctx.executors:
        executor = ctx.executors["lm_head"]
        output = executor.run({"input": last_hidden})
        if isinstance(output, dict):
            return next(iter(output.values()))
        return output

    if logits_source == "embed_weight_tied" and embed_weight is not None:
        # matmul: last_hidden @ embed_weight.T
        # Use the graph executor's run for the matmul if available,
        # otherwise fall back to kernel dispatch
        from neurobrix.kernels.dispatch import dispatch
        w_t = embed_weight.transpose(0, 1) if embed_weight.ndim == 2 else embed_weight
        mm = dispatch("mm")
        if last_hidden.ndim > 2:
            # lm_head over [..., H]: the 2-D `mm` kernel needs flat [M, H].
            lead, hdim = last_hidden.shape[:-1], last_hidden.shape[-1]
            m = 1
            for d in lead:
                m *= d
            out = mm(last_hidden.reshape(m, hdim), w_t)
            return out.reshape(*lead, out.shape[-1])
        return mm(last_hidden, w_t)

    return last_hidden


def _sample_token_nbx(logits, temperature, generated_ids=None, repetition_penalty=1.0) -> int:
    """Sample next token from logits.

    For sampling we need argmax or multinomial. In triton mode,
    we read logits to CPU via numpy for sampling (small tensor).
    """
    # Read last position logits to CPU
    if hasattr(logits, 'shape') and len(logits.shape) >= 3:
        last_logits_tensor = logits.select(1, logits.shape[1] - 1)
    else:
        last_logits_tensor = logits

    # Transfer to CPU via numpy for sampling
    last_logits_np = _to_numpy(last_logits_tensor)

    if last_logits_np.ndim > 1:
        last_logits_np = last_logits_np[0]  # First batch

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        for tid in set(generated_ids):
            if 0 <= tid < len(last_logits_np):
                if last_logits_np[tid] > 0:
                    last_logits_np[tid] /= repetition_penalty
                else:
                    last_logits_np[tid] *= repetition_penalty

    if temperature == 0.0:
        return int(np.argmax(last_logits_np))

    # Softmax in numpy
    logits_scaled = last_logits_np / temperature
    logits_scaled -= logits_scaled.max()  # numerical stability
    exp_logits = np.exp(logits_scaled)
    probs = exp_logits / exp_logits.sum()

    return int(np.random.choice(len(probs), p=probs))


def _to_numpy(tensor) -> np.ndarray:
    """Convert any tensor to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, NBXTensor):
        return tensor.numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def _is_tensor(val) -> bool:
    """Check if val is any tensor type."""
    return hasattr(val, 'shape') and hasattr(val, 'dtype')


def _parse_device_idx(device_str: str) -> int:
    """Parse device index from device string."""
    if device_str.startswith("cuda:"):
        try:
            return int(device_str.split(":")[-1].split(",")[0])
        except ValueError:
            return 0
    if "cuda:" in device_str:
        idx = device_str.index("cuda:")
        num_str = device_str[idx + 5:].split(",")[0].split(":")[0]
        try:
            return int(num_str)
        except ValueError:
            return 0
    try:
        return int(device_str)
    except ValueError:
        return 0
