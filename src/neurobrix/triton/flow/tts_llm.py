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

# Deterministic sampler seed — shared by both triton modes so triton-seq and
# triton-compiled produce reproducible (and, given matching logits, identical)
# speech tokens. Same discipline as core/flow/dual_ar's _DUALAR_SEED.
_TTS_LLM_SEED = 1234


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

        # Text tokenization — data-driven, scoped via the generation config.
        # Mirror of core/flow/tts_llm.py (R30): models whose config declares
        # start_text_token/stop_text_token (chatterbox's t3) reproduce the
        # vendor EnTokenizer.text_to_tokens contract — substitute a space
        # marker so spaces survive BPE, then wrap the text with sot/eot. Models
        # without those keys keep the plain encode path. Without this the t3
        # backbone is fed a different (shorter) text prefix than the oracle.
        def _norm_ids(_x):
            # Tokenizer may return list / np / torch — flatten to a python list
            # of ints with NO torch import (tolist() returns plain python).
            if hasattr(_x, "tolist"):
                _x = _x.tolist()
            if not isinstance(_x, list):
                _x = list(_x)
            if _x and isinstance(_x[0], (list, tuple)):
                _x = list(_x[0])
            return [int(t) for t in _x]

        sot = defaults.get("start_text_token")
        eot = defaults.get("stop_text_token")
        space_marker = defaults.get("text_space_marker")
        if sot is not None and eot is not None:
            _txt = prompt.replace(" ", space_marker) if space_marker else prompt
            _ids = _norm_ids(tokenizer.encode(
                _txt, padding=False, add_special_tokens=False))
            ids = [int(sot)] + _ids + [int(eot)]
        else:
            ids = _norm_ids(tokenizer.encode(
                prompt, padding=False, add_special_tokens=True))
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
        # CLI sampling overrides (global.*) take precedence over embedded defaults
        # (R30 mirror of core / dual_ar) — --temperature 0 ⇒ deterministic greedy.
        _ov = self.ctx.variable_resolver.resolved
        max_tokens = decode_bound(_ov.get("global.max_tokens", defaults.get("max_tokens", 2048)))
        temperature = _ov.get("global.temperature", defaults.get("temperature", 0.8))
        eos_token_id = defaults.get("eos_token_id")
        bos_token_id = defaults.get("bos_token_id")
        top_p = _ov.get("global.top_p", defaults.get("top_p", 0.95))
        min_p = _ov.get("global.min_p", defaults.get("min_p", 0.05))
        repetition_penalty = _ov.get("global.repetition_penalty", defaults.get("repetition_penalty", 1.2))

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

        # -- CFG detection (DATA-DRIVEN, R30 mirror of core via the centralized
        #    engine). The t3_cfg backbone is traced batch=1, so CFG runs
        #    SEQUENTIALLY: two batch=1 forwards (cond / text-zeroed uncond) per
        #    step, then the pure CFG formula. Models with no guidance_scale ->
        #    disabled (cond-only path unchanged, R23-safe). --
        from neurobrix.triton.cfg.engine import TritonCFGEngine
        do_cfg = False
        guidance_scale = 1.0
        try:
            _cfg_engine = TritonCFGEngine.from_topology(
                self.ctx, self._execute_component, lambda _c, _o: _o)
            do_cfg = _cfg_engine.is_enabled
            guidance_scale = _cfg_engine.guidance_scale
        except KeyError:
            do_cfg = False  # no guidance_scale in cascade -> cond-only
        if do_cfg:
            print(f"   [{lm_name}] CFG enabled via TritonCFGEngine "
                  f"(guidance_scale={guidance_scale}, sequential)")

        # -- Build conditioning embeddings (DATA-DRIVEN: run the traced cond_enc
        #    GRAPH, NOT a hand-rolled perceiver). R30 mirror of core
        #    _build_conditioning: cond_enc = spkr_enc + Perceiver resampler +
        #    emotion_adv_fc -> cat -> [1, 34, 1024]. Inputs from the embedded
        #    default-voice conditioning. --
        cond_embeds_np = self._build_conditioning_via_graph(
            speech_emb_np, speech_pos_emb_np, speaker_emb)
        print(f"   [{lm_name}] Conditioning: {cond_embeds_np.shape}")

        # -- Build text embeddings (+ CFG uncond = zero TOKEN embedding + learned
        #    POSITION; vendor order text_emb[1].zero_() THEN += text_pos, so the
        #    uncond text is position-only, not all-zeros). --
        text_tok_np = text_emb_np[ids][np.newaxis, :, :]  # [1, seq, dim]
        uncond_text_np = np.zeros_like(text_tok_np)
        if text_pos_emb_np is not None:
            pos_ids = np.arange(len(ids))
            text_pos = text_pos_emb_np[pos_ids][np.newaxis, :, :]  # [1, seq, dim]
            text_embeds_np = text_tok_np + text_pos
            uncond_text_np = uncond_text_np + text_pos  # zero-token + position
        else:
            text_embeds_np = text_tok_np

        print(f"   [{lm_name}] Text embeds: {text_embeds_np.shape}")

        # -- Initial speech token (BOS) --
        bos_embed_np = speech_emb_np[bos_token_id:bos_token_id + 1]  # [1, dim]
        bos_embed_np = bos_embed_np[np.newaxis, :, :]  # [1, 1, dim]
        if speech_pos_emb_np is not None:
            bos_pos = speech_pos_emb_np[0:1]  # [1, dim]
            bos_embed_np = bos_embed_np + bos_pos[np.newaxis, :, :]

        # -- Build initial context: [cond, text, bos_speech] (+ uncond) --
        context_np = np.concatenate(
            [cond_embeds_np, text_embeds_np, bos_embed_np], axis=1)
        uncond_context_np = None
        if do_cfg:
            uncond_context_np = np.concatenate(
                [cond_embeds_np, uncond_text_np, bos_embed_np], axis=1)
        cond_len = cond_embeds_np.shape[1]
        text_len = text_embeds_np.shape[1]

        print(f"   [{lm_name}] Initial context: {context_np.shape} "
              f"(cond={cond_len}, text={text_len}, speech=1)")
        print(f"   [{lm_name}] Generating speech tokens (max={max_tokens})...")
        start = time.perf_counter()

        # Deterministic sampler — PER-MODE reproducibility (not cross-mode token
        # identity: per-engine logits differ at fp16, so `choice` flips boundary
        # tokens). Seed overridable for diagnostics. Same pattern as dual_ar.
        import os as _os_seed
        rng = np.random.RandomState(
            int(_os_seed.environ.get("NBX_TTS_LLM_SEED", _TTS_LLM_SEED)))

        def _t3_logits(ctx_np):
            """One batch=1 backbone forward + speech_head -> [1,1,vocab]."""
            seq_len = ctx_np.shape[1]
            position_ids_np = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]
            context_nbx = NBXTensor.from_numpy(ctx_np.astype(np.float32))
            position_ids_nbx = NBXTensor.from_numpy(position_ids_np)
            self.ctx.variable_resolver.resolved["global.inputs_embeds"] = context_nbx
            self.ctx.variable_resolver.resolved["inputs_embeds"] = context_nbx
            self.ctx.variable_resolver.resolved["global.position_ids"] = position_ids_nbx
            self.ctx.variable_resolver.resolved["position_ids"] = position_ids_nbx
            self._execute_component(lm_name, "forward", None)
            out = _get_component_output(self.ctx, lm_name)
            if out is None:
                return None
            out_np = _to_numpy(out)
            last_hidden = out_np[:, -1:, :]  # [1, 1, dim]
            return last_hidden @ speech_head_np.T  # [1, 1, speech_vocab]

        generated_ids: list = []

        # Diagnostic component-isolation hook (gated, default-off): force a fixed
        # speech-token sequence (skip decode) so the vocoder can be diffed across
        # engines on IDENTICAL tokens. Mirror of core (R30). Same class as
        # NBX_DUMP_TIDS.
        import os as _os_fsi
        _force_ids = _os_fsi.environ.get("NBX_FORCE_SPEECH_IDS")
        _t3dump = _os_fsi.environ.get("NBX_DUMP_T3_LOGITS")  # teacher-forcing dump
        _forced_seq = None
        if _force_ids:
            import json as _j_fsi
            _forced_seq = [int(t) for t in _j_fsi.load(open(_force_ids))]
            if _t3dump:
                generated_ids = []  # filled by the teacher-forcing loop below
                print(f"   [{lm_name}] TEACHER-FORCING {len(_forced_seq)} tokens; "
                      f"dumping t3 logits -> {_t3dump}")
            else:
                generated_ids = list(_forced_seq)
                print(f"   [{lm_name}] FORCED {len(generated_ids)} speech tokens "
                      f"from {_force_ids} (decode skipped)")

        _teacher = bool(_force_ids and _t3dump)
        _nsteps = len(_forced_seq) if _teacher else (0 if _force_ids else max_tokens)
        for step in range(_nsteps):
            # Conditional (and, under CFG, unconditional) forward(s) -> logits.
            cond_logits_np = _t3_logits(context_np)
            if cond_logits_np is None:
                break
            uncond_logits_np = None
            logits_np = cond_logits_np
            if do_cfg:
                uncond_logits_np = _t3_logits(uncond_context_np)
                if uncond_logits_np is None:
                    break
                # Pure CFG formula (cond=context, uncond=text-zeroed context).
                logits_np = uncond_logits_np + guidance_scale * (
                    cond_logits_np - uncond_logits_np)

            if _teacher:
                # Teacher-forcing: dump logits, advance along the forced token path
                # (NOT the sampled one) so all modes see the identical context.
                _dump_t3_logits_np(_t3dump, step, cond_logits_np,
                                   uncond_logits_np, logits_np)
                next_token = _forced_seq[step]
            else:
                next_token = _sample_token_np(
                    logits_np[0, 0, :], temperature, top_p, min_p,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                    rng=rng,
                )
            generated_ids.append(next_token)

            if next_token == eos_token_id and not _teacher:
                break

            # Embed new speech token for next step
            token_embed = speech_emb_np[next_token:next_token + 1]  # [1, dim]
            token_embed = token_embed[np.newaxis, :, :]  # [1, 1, dim]
            if speech_pos_emb_np is not None:
                pos_idx = min(step + 1, speech_pos_emb_np.shape[0] - 1)
                pos_emb = speech_pos_emb_np[pos_idx:pos_idx + 1]  # [1, dim]
                token_embed = token_embed + pos_emb[np.newaxis, :, :]

            # Grow both contexts in lockstep (shared speech sequence).
            context_np = np.concatenate([context_np, token_embed], axis=1)
            if do_cfg:
                uncond_context_np = np.concatenate(
                    [uncond_context_np, token_embed], axis=1)

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

        if _os_fsi.environ.get("NBX_DUMP_SPEECH_IDS"):
            import json as _jd_si
            with open(_os_fsi.environ["NBX_DUMP_SPEECH_IDS"], "w") as _f_si:
                _jd_si.dump([int(t) for t in speech_ids], _f_si)
            print(f"   [{lm_name}] Dumped {len(speech_ids)} speech ids -> "
                  f"{_os_fsi.environ['NBX_DUMP_SPEECH_IDS']}")

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
                # Reference-voice ref-dict from the embedded conditioning (R30
                # mirror of core): ref_dict.* <- gen.*. Without it the s3gen
                # flow-matching gets all-zeros reference and the voice/audio is
                # wrong even when the speech tokens are correct.
                conds = _load_default_conditioning_np(self.ctx)
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
                            elif iname.startswith("ref_dict.") and conds is not None \
                                    and f"gen.{iname[len('ref_dict.'):]}" in conds:
                                _ref = conds[f"gen.{iname[len('ref_dict.'):]}"]
                                comp_inputs[iname] = NBXTensor.from_numpy(
                                    np.ascontiguousarray(_ref))
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

    def _build_conditioning_via_graph(
        self, speech_emb_np: np.ndarray,
        speech_pos_emb_np: Optional[np.ndarray], speaker_emb,
    ) -> np.ndarray:
        """Build conditioning by running the traced cond_enc GRAPH (zero torch).

        R30 mirror of core _build_conditioning: cond_enc is the data-driven
        T3CondEnc graph (spkr_enc + Perceiver resampler + emotion_adv_fc ->
        cat -> [1, 34, 1024]). Inputs come from the embedded default-voice
        conditioning, with cond_prompt_speech_tokens embedded via speech_emb +
        learned speech positions BEFORE T3CondEnc (vendor prepare_conditioning,
        is_gpt=False — omitting it flattens the t3 distribution). ZERO FALLBACK:
        no hand-rolled perceiver, no zero-conditioning path.
        """
        cond_name = "cond_enc"
        cond_executor = self.ctx.executors.get(cond_name)
        if cond_executor is None:
            raise RuntimeError(
                "ZERO FALLBACK: tts_llm conditioning requires a traced "
                "'cond_enc' component. Re-import a build that traces cond_enc.")
        conds = _load_default_conditioning_np(self.ctx)
        if conds is None:
            raise RuntimeError(
                "ZERO FALLBACK: missing runtime/default_conditioning.safetensors "
                "(default-voice inputs for cond_enc).")

        cond_tokens_np = conds["t3.cond_prompt_speech_tokens"].astype(np.int64)
        # Embedding lookup of cond_tokens via speech_emb + learned speech pos.
        cond_emb_np = speech_emb_np[cond_tokens_np[0]][np.newaxis, :, :]  # [1,L,dim]
        if speech_pos_emb_np is not None:
            _L = cond_tokens_np.shape[1]
            _pos = np.minimum(np.arange(_L), speech_pos_emb_np.shape[0] - 1)
            cond_emb_np = cond_emb_np + speech_pos_emb_np[_pos][np.newaxis, :, :]

        # Live reference speaker_emb overrides the default voice when present.
        if speaker_emb is not None:
            spk_np = _to_numpy(speaker_emb).astype(np.float32).reshape(1, -1)
        else:
            spk_np = conds["t3.speaker_emb"].astype(np.float32)

        comp_input_values = {
            "speaker_emb": NBXTensor.from_numpy(np.ascontiguousarray(spk_np)),
            "cond_prompt_speech_emb": NBXTensor.from_numpy(
                np.ascontiguousarray(cond_emb_np.astype(np.float32))),
            "emotion_adv": NBXTensor.from_numpy(
                np.ascontiguousarray(conds["t3.emotion_adv"].astype(np.float32))),
            "cond_prompt_speech_tokens": NBXTensor.from_numpy(
                np.ascontiguousarray(cond_tokens_np)),
        }
        comp_inputs = {n: comp_input_values[n]
                       for n in _graph_input_names(cond_executor)
                       if n in comp_input_values}

        self._ensure_weights_loaded(cond_name)
        output = cond_executor.run(comp_inputs)
        cond_emb = next(iter(output.values())) if isinstance(output, dict) else output
        cond_out_np = _to_numpy(cond_emb).astype(np.float32)
        if not self.ctx.persistent_mode:
            self._unload_component_weights(cond_name)
            gc.collect()
        return cond_out_np


# -----------------------------------------------------------------
# Module-level helpers (zero torch)
# -----------------------------------------------------------------

def _load_default_conditioning_np(ctx) -> Optional[Dict[str, np.ndarray]]:
    """Load the embedded default-voice conditioning as numpy (zero torch).

    runtime/default_conditioning.safetensors holds the conds inputs:
    t3.{speaker_emb, cond_prompt_speech_tokens, emotion_adv} for cond_enc and
    gen.{prompt_token, prompt_token_len, prompt_feat, embedding} for the s3gen
    ref-dict. Loaded via safetensors.numpy (NOT safetensors.torch) to stay
    torch-free. Cached on ctx. Returns None if the build has no default voice.
    """
    if hasattr(ctx, "_default_conditioning_np_cache"):
        return ctx._default_conditioning_np_cache
    from safetensors.numpy import load_file
    path = Path(ctx.pkg.cache_path) / "runtime" / "default_conditioning.safetensors"
    conds = load_file(str(path)) if path.exists() else None
    ctx._default_conditioning_np_cache = conds
    return conds


def _graph_input_names(executor) -> List[str]:
    """Return the graph DAG input_name list for a component executor."""
    dag = getattr(executor, "_dag", None)
    if not dag:
        return []
    return [t["input_name"] for t in dag.get("tensors", {}).values()
            if t.get("input_name")]

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


def _dump_t3_logits_np(path, step, cond, uncond, comb) -> None:
    """Teacher-forcing diagnostic (numpy mirror of core _dump_t3_logits): per-step
    cond/uncond/CFG-combined logit stats so the free-running backbone can be
    op-diffed across engines on an identical forced token path."""
    import json as _j

    def _st(a):
        if a is None:
            return None
        f = np.asarray(a).reshape(-1).astype(np.float64)
        return {"l2": float((f * f).sum() ** 0.5),
                "head": [round(float(x), 5) for x in f[:8]],
                "argmax": int(f.argmax())}
    with open(path, "a") as _fh:
        _j.dump({"step": step, "cond": _st(cond),
                 "uncond": _st(uncond), "comb": _st(comb)}, _fh)
        _fh.write("\n")


def _sample_token_np(
    logits_1d: np.ndarray, temperature: float,
    top_p: float = 1.0, min_p: float = 0.0,
    generated_ids: Optional[list] = None,
    repetition_penalty: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """Sample next token from logits (NumPy). `rng` makes it deterministic."""
    logits = logits_1d.copy().astype(np.float64)

    # Repetition penalty
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            if 0 <= tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty

    # Greedy floor (dual_ar discipline, R30 mirror of core): deterministic argmax
    # → all four modes pick identical tokens (logits agree to fp16), the true
    # cross-mode reconciliation with no runaway. Used by --temperature 0.
    if temperature == 0.0:
        return int(np.argmax(logits))

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

    _rng = rng if rng is not None else np.random
    return int(_rng.choice(len(probs), p=probs))


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
