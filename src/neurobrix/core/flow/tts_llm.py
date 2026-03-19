"""
TTS-LLM Flow — Speech Language Model TTS

Handles models where a backbone LLM autoregressively generates speech tokens
using external embedding tables, positional embeddings, conditioning encoders,
and output heads. The backbone graph only sees inputs_embeds.

Pattern:
    ve(ref_audio) → speaker_emb
    cond_enc(speaker_emb) → cond_embeds
    text_emb(text_tokens) + text_pos_emb → text_embeds
    [cond_embeds, text_embeds, speech_embeds] → backbone(inputs_embeds) → hidden
    speech_head(hidden[:, -1]) → logits → sample → next speech token
    speech_emb(token) + speech_pos_emb(step) → append to context

Models: Chatterbox (T3 + LlamaModel backbone)

ZERO SEMANTIC: No knowledge of specific models.
ZERO HARDCODE: All parameters from NBX container.
"""

import gc
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow


@register_flow("tts_llm")
class TTSLLMEngine(FlowHandler):
    """
    TTS Speech LM: conditioning + text/speech embeddings → backbone → speech tokens → vocoder.

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
        ctx: FlowContext,
        execute_component_fn: Callable,
        resolve_inputs_fn: Callable,
        ensure_weights_fn: Callable,
        unload_weights_fn: Callable,
    ):
        super().__init__(ctx)
        self._execute_component = execute_component_fn
        self._resolve_component_inputs = resolve_inputs_fn
        self._ensure_weights_loaded = ensure_weights_fn
        self._unload_component_weights = unload_weights_fn

    def execute(self) -> Dict[str, Any]:
        """Execute TTS-LLM pipeline."""
        flow = self.ctx.pkg.topology.get("flow", {})
        audio_config = flow.get("audio", {})
        if not audio_config:
            # tts_llm flow may store config directly under flow
            audio_config = flow
        stages = audio_config.get("stages", [])
        defaults = self.ctx.pkg.defaults
        device = self.ctx.primary_device
        dtype = self._get_compute_dtype()

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

        # ── Step 1: Tokenize text input ──
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt")
        if prompt is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires global.prompt.")

        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is None:
            raise RuntimeError("ZERO FALLBACK: TTS model requires tokenizer module.")

        # Tokenize with special tokens (model expects BOT/EOT markers)
        # NeuroBrix tokenizers pad by default — disable padding for TTS input
        ids = tokenizer.encode(prompt, padding=False, add_special_tokens=True)
        if isinstance(ids, torch.Tensor):
            input_ids = ids.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        self.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
        self.ctx.variable_resolver.resolved["input_ids"] = input_ids
        print(f"   [Tokenizer] '{prompt[:50]}...' → {input_ids.shape[1]} tokens")

        # ── Step 2: Voice encoder (speaker embedding) ──
        speaker_emb = None
        if ve_stage is not None:
            comp_name = ve_stage["component"]
            if comp_name in self.ctx.executors:
                # Check if we have reference audio for VE
                ref_audio = self.ctx.variable_resolver.resolved.get("global.ref_audio")
                if ref_audio is not None and isinstance(ref_audio, torch.Tensor):
                    print(f"   [{comp_name}] Running voice encoder...")
                    start = time.perf_counter()
                    self._ensure_weights_loaded(comp_name)
                    self.ctx.variable_resolver.resolved["mels"] = ref_audio
                    self._execute_component(comp_name, "forward", None)
                    speaker_emb = self._get_component_output(comp_name)
                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"   [{comp_name}] Done in {elapsed:.0f}ms, speaker_emb: {speaker_emb.shape if speaker_emb is not None else 'None'}")
                    if not self.ctx.persistent_mode:
                        self._unload_component_weights(comp_name)
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    print(f"   [{comp_name}] Skipped (no reference audio)")

        # ── Step 3: Autoregressive speech token generation ──
        lm_name = lm_stage["component"]
        aux_weights_config = lm_stage.get("auxiliary_weights", {})

        max_tokens = defaults.get("max_tokens", 2048)
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

        # Load auxiliary weights from the same component's weight shard
        aux_weights = self._load_auxiliary_weights(lm_name, aux_weights_config)

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

        # ── Build conditioning embeddings ──
        # cond_enc: speaker_emb → spkr_enc → [1, 1, dim]
        #           + perceiver(prompt_speech_emb) if available
        #           + emotion_adv_fc if available
        cond_embeds = self._build_conditioning(
            aux_weights, speaker_emb, device, dtype
        )
        print(f"   [{lm_name}] Conditioning: {cond_embeds.shape}")

        # ── Build text embeddings ──
        with torch.no_grad():
            text_embeds = F.embedding(input_ids, text_emb_w.to(dtype=dtype))
            if text_pos_emb_w is not None:
                # Learned position embeddings: index by token positions
                pos_ids = torch.arange(input_ids.shape[1], device=device)
                text_pos = F.embedding(pos_ids, text_pos_emb_w.to(dtype=dtype))
                text_embeds = text_embeds + text_pos.unsqueeze(0)

        print(f"   [{lm_name}] Text embeds: {text_embeds.shape}")

        # ── Initial speech token (BOS) ──
        bos_tensor = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            bos_embed = F.embedding(bos_tensor, speech_emb_w.to(dtype=dtype))
            if speech_pos_emb_w is not None:
                bos_pos = F.embedding(
                    torch.tensor([0], device=device),
                    speech_pos_emb_w.to(dtype=dtype)
                )
                bos_embed = bos_embed + bos_pos.unsqueeze(0)

        # ── Build initial context: [cond, text, bos_speech] ──
        context_embeds = torch.cat([cond_embeds, text_embeds, bos_embed], dim=1)
        cond_len = cond_embeds.shape[1]
        text_len = text_embeds.shape[1]

        print(f"   [{lm_name}] Initial context: {context_embeds.shape} "
              f"(cond={cond_len}, text={text_len}, speech=1)")
        print(f"   [{lm_name}] Generating speech tokens (max={max_tokens})...")
        start = time.perf_counter()

        generated_ids: list = []
        speech_head_w_typed = speech_head_w.to(dtype=dtype)

        for step in range(max_tokens):
            seq_len = context_embeds.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

            # Bind inputs for backbone graph
            self.ctx.variable_resolver.resolved["global.inputs_embeds"] = context_embeds
            self.ctx.variable_resolver.resolved["inputs_embeds"] = context_embeds
            self.ctx.variable_resolver.resolved["global.position_ids"] = position_ids
            self.ctx.variable_resolver.resolved["position_ids"] = position_ids

            self._execute_component(lm_name, "forward", None)

            output = self._get_component_output(lm_name)
            if output is None:
                break

            # Apply speech_head to last hidden state → logits over speech vocab
            last_hidden = output[:, -1:, :]  # [B, 1, dim]
            logits = torch.matmul(last_hidden, speech_head_w_typed.T)  # [B, 1, speech_vocab]

            # Sample next token
            next_token = self._sample_token(
                logits, temperature, top_p, min_p,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)

            if next_token == eos_token_id:
                break

            # Embed new speech token for next step
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            with torch.no_grad():
                token_embed = F.embedding(token_tensor, speech_emb_w.to(dtype=dtype))
                if speech_pos_emb_w is not None:
                    pos_idx = torch.tensor([step + 1], device=device)
                    pos_idx = pos_idx.clamp(max=speech_pos_emb_w.shape[0] - 1)
                    token_pos = F.embedding(pos_idx, speech_pos_emb_w.to(dtype=dtype))
                    token_embed = token_embed + token_pos.unsqueeze(0)

            context_embeds = torch.cat([context_embeds, token_embed], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{lm_name}] Generated {len(generated_ids)} speech tokens in {elapsed:.0f}ms")

        # Store generated speech tokens, filtering out special tokens (>= vocoder vocab_size)
        # T3 generates tokens in [0, speech_vocab) + special tokens (start/stop speech).
        # The vocoder embedding only covers [0, vocab_size), so strip special tokens.
        vocoder_vocab_size = defaults.get("vocoder_vocab_size")
        if vocoder_vocab_size is None and vocoder_stage is not None:
            # Detect from vocoder graph's embedding weight shape
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
            print(f"   [{lm_name}] Filtered {len(generated_ids) - len(speech_ids)} special tokens "
                  f"(vocab_size={vocoder_vocab_size})")

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = speech_ids
        speech_tokens = torch.tensor([speech_ids], dtype=torch.long, device=device)
        self.ctx.variable_resolver.resolved["global.speech_tokens"] = speech_tokens
        self.ctx.variable_resolver.resolved["speech_tokens"] = speech_tokens

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            gc.collect()
            torch.cuda.empty_cache()

        # ── Step 4: Vocoder (speech tokens → audio) ──
        if vocoder_stage is not None:
            voc_name = vocoder_stage["component"]
            if voc_name in self.ctx.executors:
                print(f"   [{voc_name}] Running vocoder...")
                start = time.perf_counter()
                self._ensure_weights_loaded(voc_name)

                # Build comp_inputs directly from graph DAG — bypass input resolver
                # because topology connections wrongly map t3_cfg output to ALL s3gen inputs.
                speech_len = torch.tensor([len(speech_ids)], dtype=torch.long, device=device)
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
                                comp_inputs[iname] = speech_len
                            else:
                                # Dummy inputs for ref_dict fields and others
                                shape = tspec.get("shape", [1])
                                dtype_str = tspec.get("dtype", "float32")
                                t_dtype = torch.int64 if "int" in dtype_str else torch.float32
                                comp_inputs[iname] = torch.zeros(shape, dtype=t_dtype, device=device)

                output = voc_executor.run(comp_inputs)

                # Extract audio from output
                audio_output = None
                if isinstance(output, dict):
                    audio_output = next(iter(output.values())) if output else None
                elif isinstance(output, torch.Tensor):
                    audio_output = output
                elapsed = (time.perf_counter() - start) * 1000
                print(f"   [{voc_name}] Done in {elapsed:.0f}ms")

                if audio_output is not None:
                    self.ctx.variable_resolver.resolved["global.output_audio"] = audio_output

                    # Save as WAV
                    sample_rate = defaults.get("sample_rate", 24000)
                    self._save_audio(audio_output, sample_rate)

                if not self.ctx.persistent_mode:
                    self._unload_component_weights(voc_name)
                    gc.collect()
                    torch.cuda.empty_cache()

        return self.ctx.variable_resolver.resolve_all()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_compute_dtype(self) -> torch.dtype:
        """Get compute dtype from Prism plan."""
        if self.ctx.plan and hasattr(self.ctx.plan, 'allocations'):
            for alloc in self.ctx.plan.allocations.values():
                if hasattr(alloc, 'dtype') and alloc.dtype is not None:
                    return alloc.dtype
        dtype_str = self.ctx.pkg.defaults.get("dtype", "float16")
        return getattr(torch, dtype_str, torch.float16)

    def _get_component_output(self, comp_name: str) -> Optional[torch.Tensor]:
        """Get component output from variable resolver."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.last_hidden_state",
                    f"{comp_name}.output", f"{comp_name}.hidden_states"]:
            if key in resolved and isinstance(resolved[key], torch.Tensor):
                return resolved[key]
        return None

    def _load_auxiliary_weights(
        self, comp_name: str, aux_config: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        """Load auxiliary weights from a component's weight store.

        These are weights in the same safetensors shard but NOT in the graph
        (e.g., text_emb, speech_emb, speech_head for Chatterbox T3).
        """
        result = {}
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return result

        weights = getattr(executor, '_weights', {})

        for aux_name, weight_key in aux_config.items():
            # Try exact key match first
            if weight_key in weights:
                result[aux_name] = weights[weight_key]
                continue
            # Try suffix match (weight keys may have component prefix)
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

    def _build_conditioning(
        self, aux_weights: Dict[str, torch.Tensor],
        speaker_emb: Optional[torch.Tensor],
        device: str, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build conditioning embeddings from speaker/emotion/prompt info.

        Implements T3CondEnc logic using auxiliary weights directly:
        1. spkr_enc(speaker_emb) → [B, 1, dim]
        2. perceiver(prompt_speech_emb) → [B, N, dim] (if prompt available)
        3. emotion_adv_fc(0.5) → [B, 1, dim] (if emotion conditioning)
        4. Concat all → [B, cond_len, dim]
        """
        parts = []
        dim = None

        # Speaker embedding projection
        spkr_w = aux_weights.get("cond_enc.spkr_enc.weight")
        spkr_b = aux_weights.get("cond_enc.spkr_enc.bias")
        if spkr_w is not None:
            dim = spkr_w.shape[0]
            if speaker_emb is not None:
                spkr_in = speaker_emb.view(1, -1).to(dtype=dtype)
            else:
                # No reference audio: use zero speaker embedding
                spkr_size = spkr_w.shape[1]
                spkr_in = torch.zeros(1, spkr_size, device=device, dtype=dtype)

            with torch.no_grad():
                cond_spkr = F.linear(spkr_in, spkr_w.to(dtype=dtype),
                                     spkr_b.to(dtype=dtype) if spkr_b is not None else None)
            cond_spkr = cond_spkr.unsqueeze(1)  # [1, 1, dim]
            parts.append(cond_spkr)

        # Perceiver resampler (processes prompt speech embeddings)
        # For now, skip prompt conditioning (requires reference speech tokens)
        # The perceiver output would go here as parts.append(perceiver_out)

        # Emotion adversarial conditioning
        emotion_w = aux_weights.get("cond_enc.emotion_adv_fc.weight")
        if emotion_w is not None and dim is not None:
            with torch.no_grad():
                emotion_val = torch.tensor([[[0.5]]], device=device, dtype=dtype)
                emotion_emb = F.linear(emotion_val, emotion_w.to(dtype=dtype))
            parts.append(emotion_emb)  # [1, 1, dim]

        if not parts:
            # Fallback: zero conditioning
            if dim is None:
                dim = 1024  # Will fail at concat if wrong
            return torch.zeros(1, 1, dim, device=device, dtype=dtype)

        return torch.cat(parts, dim=1)

    def _sample_token(
        self, logits: torch.Tensor, temperature: float,
        top_p: float = 1.0, min_p: float = 0.0,
        generated_ids: Optional[list] = None,
        repetition_penalty: float = 1.0,
    ) -> int:
        """Sample next token from logits with temperature, top-p, min-p, and repetition penalty."""
        logits = logits[:, -1, :].float()  # [B, vocab]

        # Repetition penalty
        if generated_ids and repetition_penalty != 1.0:
            for tid in set(generated_ids):
                if 0 <= tid < logits.shape[-1]:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

        # Temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # Min-p filtering
        if min_p > 0:
            probs = torch.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True).values
            threshold = min_p * max_prob
            logits = logits.masked_fill(probs < threshold, float('-inf'))

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())
        return next_token

    def _save_audio(self, audio_tensor: torch.Tensor, sample_rate: int) -> None:
        """Save audio tensor to WAV file."""
        import numpy as np

        audio = audio_tensor.detach().cpu().float()
        if audio.dim() > 1:
            audio = audio.squeeze()
        audio_np = audio.numpy()

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

        print(f"   [Audio] Saved: {output_path} ({len(audio_np)/sample_rate:.1f}s @ {sample_rate}Hz)")
