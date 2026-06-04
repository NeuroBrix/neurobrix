"""Triton TTS-LLM Flow — zero torch Speech Language Model TTS.

Ported from core/flow/tts_llm.py. Handles models where a backbone LLM
autoregressively generates speech tokens using external embedding tables,
positional embeddings, conditioning encoders, and output heads.

Pattern:
    ve(ref_audio) -> speaker_emb
    cond_enc(speaker_emb) -> cond_embeds
    text_emb(text_tokens) + text_pos_emb -> text_embeds
    [cond_embeds, text_embeds, speech_embeds] -> backbone(inputs_embeds) -> hidden
    speech_head(hidden[:, -1]) -> logits -> sample -> next speech token
    speech_emb(token) + speech_pos_emb(step) -> append to context

Models: Chatterbox (T3 + LlamaModel backbone)

No torch imports in hot path.
"""

import gc
import time
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator


class TritonTTSLLMEngine:
    """
    Triton-mode TTS Speech LM: conditioning + text/speech embeddings
    -> backbone -> speech tokens -> vocoder.

    topology.flow:
        type: tts_llm
        direction: tts
        stages:
          - component: ve
            execution: forward
            role: speaker_embedding
          - component: t3_cfg
            execution: autoregressive
            role: speech_lm
            auxiliary_weights: {text_emb, speech_emb, speech_head, ...}
          - component: s3gen
            execution: forward
            role: vocoder
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
        """Execute TTS-LLM pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        if not audio_config:
            audio_config = flow
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults
        device_idx = _parse_device_idx(self.ctx.primary_device)
        DeviceAllocator.set_device(device_idx)

        # Separate stages by role
        ve_stage = None
        lm_stage = None
        vocoder_stage = None
        for s in stages:
            role = s.get("role", s.get("execution", ""))
            if role == "speaker_embedding" or s.get("execution") == "forward" and ve_stage is None:
                if s.get("role") == "speaker_embedding":
                    ve_stage = s
                elif vocoder_stage is None and s.get("role") != "vocoder":
                    ve_stage = s
            if s.get("execution") == "autoregressive" or role == "speech_lm":
                lm_stage = s
            if role == "vocoder" or (s.get("execution") == "forward" and lm_stage is not None):
                vocoder_stage = s

        if lm_stage is None:
            raise RuntimeError(
                "ZERO FALLBACK: tts_llm flow requires a stage with execution=autoregressive."
            )

        # -- Step 1: Tokenize text input --
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt")
        if prompt is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires global.prompt.")

        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires tokenizer module.")

        ids = tokenizer.encode(prompt, padding=False, add_special_tokens=True)
        if not isinstance(ids, list):
            ids = list(ids)
        input_ids_np = np.array([ids], dtype=np.int64)
        input_ids = NBXTensor.from_numpy(input_ids_np)

        self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
        self.ctx.variable_resolver.resolved["input_ids"] = input_ids
        print(f"   [Tokenizer] '{prompt[:50]}...' -> {len(ids)} tokens")

        # -- Step 2: Voice encoder (speaker embedding) --
        speaker_emb = None
        if ve_stage is not None:
            comp_name = ve_stage["component"]
            if comp_name in self.ctx.executors:
                ref_audio = self.ctx.variable_resolver.resolved.get("global.ref_audio")
                if ref_audio is not None and _is_tensor(ref_audio):
                    print(f"   [{comp_name}] Running voice encoder...")
                    start = time.perf_counter()
                    self._ensure_weights_loaded(comp_name)
                    self.ctx.variable_resolver.resolved["mels"] = ref_audio
                    self._execute_component(comp_name, "forward", None)
                    speaker_emb = _get_component_output(self.ctx, comp_name)
                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"   [{comp_name}] Done in {elapsed:.0f}ms")
                    if not self.ctx.persistent_mode:
                        self._unload_component_weights(comp_name)
                        gc.collect()
                else:
                    print(f"   [{comp_name}] Skipped (no reference audio)")

        # -- Step 3: Autoregressive speech token generation --
        lm_name = lm_stage["component"]
        aux_weights_config = lm_stage.get("auxiliary_weights", {})

        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        max_tokens = decode_bound(defaults.get("max_tokens", 2048))
        temperature = defaults.get("temperature", 0.8)
        eos_token_id = defaults.get("eos_token_id")
        bos_token_id = defaults.get("bos_token_id")
        top_p = defaults.get("top_p", 0.95)
        min_p = defaults.get("min_p", 0.05)
        repetition_penalty = defaults.get("repetition_penalty", 1.2)

        if eos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: eos_token_id missing from defaults.json.")
        if bos_token_id is None:
            raise RuntimeError("ZERO FALLBACK: bos_token_id missing from defaults.json.")

        print(f"   [{lm_name}] Loading weights...")
        self._ensure_weights_loaded(lm_name)

        # Load auxiliary weights
        aux_weights = _load_auxiliary_weights(self.ctx, lm_name, aux_weights_config)

        text_emb_w = aux_weights.get("text_emb")
        speech_emb_w = aux_weights.get("speech_emb")
        text_pos_emb_w = aux_weights.get("text_pos_emb")
        speech_pos_emb_w = aux_weights.get("speech_pos_emb")
        speech_head_w = aux_weights.get("speech_head")

        if speech_emb_w is None:
            raise RuntimeError(f"ZERO FALLBACK: speech_emb weight not found in {lm_name}.")
        if speech_head_w is None:
            raise RuntimeError(f"ZERO FALLBACK: speech_head weight not found in {lm_name}.")
        if text_emb_w is None:
            raise RuntimeError(f"ZERO FALLBACK: text_emb weight not found in {lm_name}.")

        # Convert aux weights to numpy for embedding lookups
        text_emb_np = _to_numpy(text_emb_w).astype(np.float32)
        speech_emb_np = _to_numpy(speech_emb_w).astype(np.float32)
        text_pos_emb_np = _to_numpy(text_pos_emb_w).astype(np.float32) if text_pos_emb_w is not None else None
        speech_pos_emb_np = _to_numpy(speech_pos_emb_w).astype(np.float32) if speech_pos_emb_w is not None else None
        speech_head_np = _to_numpy(speech_head_w).astype(np.float32)

        # -- Build conditioning embeddings --
        cond_embeds_np = _build_conditioning_np(aux_weights, speaker_emb)
        print(f"   [{lm_name}] Conditioning: {cond_embeds_np.shape}")

        # -- Build text embeddings --
        text_embeds_np = text_emb_np[ids]  # [seq_len, dim]
        text_embeds_np = text_embeds_np[np.newaxis, :, :]  # [1, seq_len, dim]
        if text_pos_emb_np is not None:
            pos_ids = np.arange(len(ids))
            text_pos = text_pos_emb_np[pos_ids]  # [seq_len, dim]
            text_embeds_np = text_embeds_np + text_pos[np.newaxis, :, :]

        print(f"   [{lm_name}] Text embeds: {text_embeds_np.shape}")

        # -- Initial speech token (BOS) --
        bos_embed_np = speech_emb_np[bos_token_id:bos_token_id + 1]  # [1, dim]
        bos_embed_np = bos_embed_np[np.newaxis, :, :]  # [1, 1, dim]
        if speech_pos_emb_np is not None:
            bos_pos = speech_pos_emb_np[0:1]  # [1, dim]
            bos_embed_np = bos_embed_np + bos_pos[np.newaxis, :, :]

        # -- Build initial context: [cond, text, bos_speech] --
        context_np = np.concatenate([cond_embeds_np, text_embeds_np, bos_embed_np], axis=1)
        cond_len = cond_embeds_np.shape[1]
        text_len = text_embeds_np.shape[1]

        print(f"   [{lm_name}] Initial context: {context_np.shape} "
              f"(cond={cond_len}, text={text_len}, speech=1)")
        print(f"   [{lm_name}] Generating speech tokens (max={max_tokens})...")
        start = time.perf_counter()

        generated_ids: list = []

        for step in range(max_tokens):
            seq_len = context_np.shape[1]
            position_ids_np = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

            # Convert context to NBXTensor for graph execution
            context_nbx = NBXTensor.from_numpy(context_np.astype(np.float32))
            position_ids_nbx = NBXTensor.from_numpy(position_ids_np)

            self.ctx.variable_resolver.resolved["global.inputs_embeds"] = context_nbx
            self.ctx.variable_resolver.resolved["inputs_embeds"] = context_nbx
            self.ctx.variable_resolver.resolved["global.position_ids"] = position_ids_nbx
            self.ctx.variable_resolver.resolved["position_ids"] = position_ids_nbx

            self._execute_component(lm_name, "forward", None)

            output = _get_component_output(self.ctx, lm_name)
            if output is None:
                break

            # Apply speech_head to last hidden state -> logits
            output_np = _to_numpy(output)
            last_hidden = output_np[:, -1:, :]  # [1, 1, dim]
            logits_np = last_hidden @ speech_head_np.T  # [1, 1, speech_vocab]

            # Sample next token
            next_token = _sample_token_np(
                logits_np[0, 0, :], temperature, top_p, min_p,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)

            if next_token == eos_token_id:
                break

            # Embed new speech token for next step
            token_embed = speech_emb_np[next_token:next_token + 1]  # [1, dim]
            token_embed = token_embed[np.newaxis, :, :]  # [1, 1, dim]
            if speech_pos_emb_np is not None:
                pos_idx = min(step + 1, speech_pos_emb_np.shape[0] - 1)
                pos_emb = speech_pos_emb_np[pos_idx:pos_idx + 1]  # [1, dim]
                token_embed = token_embed + pos_emb[np.newaxis, :, :]

            context_np = np.concatenate([context_np, token_embed], axis=1)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{lm_name}] Generated {len(generated_ids)} speech tokens in {elapsed:.0f}ms")

        # Filter special tokens for vocoder
        vocoder_vocab_size = defaults.get("vocoder_vocab_size")
        if vocoder_vocab_size is None and vocoder_stage is not None:
            voc_exec = self.ctx.executors.get(vocoder_stage["component"])
            if voc_exec is not None:
                voc_dag = getattr(voc_exec, '_dag', None)
                if voc_dag:
                    for _tid, tspec in voc_dag.get("tensors", {}).items():
                        wname = tspec.get("weight_name", "")
                        if "embedding" in wname and tspec.get("shape"):
                            vocoder_vocab_size = tspec["shape"][0]
                            break

        speech_ids = [t for t in generated_ids if vocoder_vocab_size is None or t < vocoder_vocab_size]
        if len(speech_ids) < len(generated_ids):
            print(f"   [{lm_name}] Filtered {len(generated_ids) - len(speech_ids)} special tokens")

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = speech_ids
        speech_tokens_np = np.array([speech_ids], dtype=np.int64)
        speech_tokens = NBXTensor.from_numpy(speech_tokens_np)
        self.ctx.variable_resolver.resolved["global.speech_tokens"] = speech_tokens
        self.ctx.variable_resolver.resolved["speech_tokens"] = speech_tokens

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            gc.collect()

        # -- Step 4: Vocoder (speech tokens -> audio) --
        if vocoder_stage is not None:
            voc_name = vocoder_stage["component"]
            if voc_name in self.ctx.executors:
                print(f"   [{voc_name}] Running vocoder...")
                start = time.perf_counter()
                self._ensure_weights_loaded(voc_name)

                speech_len_np = np.array([len(speech_ids)], dtype=np.int64)
                voc_executor = self.ctx.executors.get(voc_name)
                comp_inputs: Dict[str, Any] = {}
                if voc_executor is not None:
                    voc_dag = getattr(voc_executor, '_dag', None)
                    if voc_dag:
                        for _tid, tspec in voc_dag.get("tensors", {}).items():
                            iname = tspec.get("input_name")
                            if not iname:
                                continue
                            if iname == "speech_tokens":
                                comp_inputs[iname] = speech_tokens
                            elif iname == "speech_token_lens":
                                comp_inputs[iname] = NBXTensor.from_numpy(speech_len_np)
                            else:
                                shape = tspec.get("shape", [1])
                                dtype_str = tspec.get("dtype", "float32")
                                if "int" in dtype_str:
                                    dummy = np.zeros(shape, dtype=np.int64)
                                else:
                                    dummy = np.zeros(shape, dtype=np.float32)
                                comp_inputs[iname] = NBXTensor.from_numpy(dummy)

                output = voc_executor.run(comp_inputs)

                audio_output = None
                if isinstance(output, dict):
                    audio_output = next(iter(output.values())) if output else None
                elif _is_tensor(output):
                    audio_output = output

                elapsed = (time.perf_counter() - start) * 1000
                print(f"   [{voc_name}] Done in {elapsed:.0f}ms")

                if audio_output is not None:
                    self.ctx.variable_resolver.resolved["global.output_audio"] = audio_output
                    sample_rate = defaults.get("sample_rate", 24000)
                    _save_audio_np(audio_output, sample_rate)

                if not self.ctx.persistent_mode:
                    self._unload_component_weights(voc_name)
                    gc.collect()

        return self.ctx.variable_resolver.resolve_all()


# -----------------------------------------------------------------
# Module-level helpers (zero torch)
# -----------------------------------------------------------------

def _get_component_output(ctx, comp_name):
    """Get component output from variable resolver."""
    resolved = ctx.variable_resolver.resolved
    for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state",
                f"{comp_name}.output", f"{comp_name}.hidden_states"]:
        if key in resolved and _is_tensor(resolved[key]):
            return resolved[key]
    return None


def _load_auxiliary_weights(ctx, comp_name: str, aux_config: Dict) -> Dict:
    """Load auxiliary weights from a component's weight store."""
    result = {}
    executor = ctx.executors.get(comp_name)
    if executor is None:
        return result
    weights = getattr(executor, '_weights', {})

    for aux_name, weight_key in aux_config.items():
        if weight_key in weights:
            result[aux_name] = weights[weight_key]
            continue
        for wk, wv in weights.items():
            if wk.endswith(weight_key) or wk == weight_key:
                result[aux_name] = wv
                break

    loaded = list(result.keys())
    missing = [k for k in aux_config if k not in result]
    if loaded:
        print(f"   [{comp_name}] Auxiliary weights loaded: {len(loaded)}/{len(aux_config)}")
    if missing:
        print(f"   [{comp_name}] Auxiliary weights missing: {missing}")
    return result


def _build_conditioning_np(aux_weights: Dict, speaker_emb) -> np.ndarray:
    """Build conditioning embeddings from speaker/emotion/prompt info (NumPy)."""
    parts = []
    dim = None

    spkr_w = aux_weights.get("cond_enc.spkr_enc.weight")
    spkr_b = aux_weights.get("cond_enc.spkr_enc.bias")
    if spkr_w is not None:
        spkr_w_np = _to_numpy(spkr_w).astype(np.float32)
        dim = spkr_w_np.shape[0]
        if speaker_emb is not None:
            spkr_in = _to_numpy(speaker_emb).astype(np.float32).reshape(1, -1)
        else:
            spkr_size = spkr_w_np.shape[1]
            spkr_in = np.zeros((1, spkr_size), dtype=np.float32)

        cond_spkr = spkr_in @ spkr_w_np.T
        if spkr_b is not None:
            cond_spkr = cond_spkr + _to_numpy(spkr_b).astype(np.float32)
        cond_spkr = cond_spkr[:, np.newaxis, :]  # [1, 1, dim]
        parts.append(cond_spkr)

    emotion_w = aux_weights.get("cond_enc.emotion_adv_fc.weight")
    if emotion_w is not None and dim is not None:
        emotion_w_np = _to_numpy(emotion_w).astype(np.float32)
        emotion_val = np.array([[[0.5]]], dtype=np.float32)
        emotion_emb = emotion_val @ emotion_w_np.T if emotion_w_np.ndim == 2 else emotion_val * emotion_w_np
        parts.append(emotion_emb.reshape(1, 1, -1))

    if not parts:
        if dim is None:
            dim = 1024
        return np.zeros((1, 1, dim), dtype=np.float32)

    return np.concatenate(parts, axis=1)


def _sample_token_np(
    logits_1d: np.ndarray, temperature: float,
    top_p: float = 1.0, min_p: float = 0.0,
    generated_ids: Optional[list] = None,
    repetition_penalty: float = 1.0,
) -> int:
    """Sample next token from logits (NumPy)."""
    logits = logits_1d.copy().astype(np.float64)

    # Repetition penalty
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            if 0 <= tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty

    # Temperature
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # Min-p filtering
    if min_p > 0:
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()
        max_prob = probs.max()
        threshold = min_p * max_prob
        mask = probs < threshold
        logits[mask] = -np.inf

    # Top-p (nucleus) sampling
    if top_p < 1.0:
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        sorted_shifted = sorted_logits - sorted_logits.max()
        sorted_probs = np.exp(sorted_shifted) / np.exp(sorted_shifted).sum()
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        logits[sorted_idx[cutoff:]] = -np.inf

    # Final softmax + sample
    logits_shifted = logits - logits.max()
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum()
    probs = np.maximum(probs, 0)  # Ensure non-negative
    probs = probs / probs.sum()  # Re-normalize

    return int(np.random.choice(len(probs), p=probs))


def _save_audio_np(audio_tensor, sample_rate: int) -> None:
    """Save audio tensor to WAV file (numpy path)."""
    audio_np = _to_numpy(audio_tensor).astype(np.float32).squeeze()

    # Normalize to [-1, 1]
    max_val = max(abs(audio_np.max()), abs(audio_np.min()))
    if max_val > 0:
        audio_np = audio_np / max_val

    output_path = Path.cwd() / "output_chatterbox.wav"
    try:
        import soundfile as sf
        sf.write(str(output_path), audio_np, sample_rate)
    except ImportError:
        import wave
        import struct
        audio_int16 = (audio_np * 32767).astype(np.int16)
        with wave.open(str(output_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f'<{len(audio_int16)}h', *audio_int16))

    print(f"   [Audio] Saved: {output_path} ({len(audio_np) / sample_rate:.1f}s @ {sample_rate}Hz)")


def _to_numpy(tensor) -> np.ndarray:
    """Convert any tensor to numpy.

    For NBXTensor (Triton mode): upcasts fp16/bf16 to fp32 first (numpy
    has no native bf16), then D2H memcpy into a host buffer, then wrap
    as a numpy array. No torch import — this is the GPU→CPU boundary for
    sparse aux-weight lookups done on host.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, NBXTensor):
        import ctypes
        t = tensor.contiguous()
        # numpy has no bf16; fp16 we keep but the callers always .astype
        # later. Upcast bf16 to fp32 on-device before the D2H.
        if t.dtype == NBXDtype.bfloat16:
            t = t.to(NBXDtype.float32)
        nb_dtype_to_np = {
            NBXDtype.float32: np.float32,
            NBXDtype.float16: np.float16,
            NBXDtype.int32:   np.int32,
            NBXDtype.int64:   np.int64,
        }
        np_dtype = nb_dtype_to_np.get(t.dtype)
        if np_dtype is None:
            # Unknown dtype — force fp32.
            t = t.to(NBXDtype.float32)
            np_dtype = np.float32
        arr = np.empty(t.shape, dtype=np_dtype)
        DeviceAllocator.memcpy(
            arr.ctypes.data, t.data_ptr(), arr.nbytes, kind=2)
        return arr
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def _is_tensor(val) -> bool:
    """Check if val is any tensor type."""
    return isinstance(val, NBXTensor) or (hasattr(val, 'shape') and hasattr(val, 'dtype'))


def _parse_device_idx(device_str: str) -> int:
    """Parse device index from device string."""
    if device_str.startswith("cuda:"):
        try:
            return int(device_str.split(":")[-1].split(",")[0])
        except ValueError:
            return 0
    try:
        return int(device_str)
    except ValueError:
        return 0
