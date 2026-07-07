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

from neurobrix.core.memory.manager import release_flow_memory
import time
import numpy as np
import torch
from neurobrix.core.device_utils import device_multinomial
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow

# Deterministic sampler seed — duplicated identically in triton/flow/tts_llm.py
# (_TTS_LLM_SEED) so the two separate code paths draw the SAME speech tokens from
# the SAME numpy RandomState + algorithm (the openaudio dual_ar discipline). A
# stochastic TTS (chatterbox t3 temperature sampling) is otherwise non-
# reproducible and the four modes can't be cross-validated.
_TTS_LLM_SEED = 1234


def _dump_t3_logits(path: str, step: int, cond, uncond, comb) -> None:
    """Teacher-forcing diagnostic: dump per-step t3 logit stats (l2, head, argmax)
    for cond / uncond / CFG-combined, so the free-running backbone can be
    op-diffed across engines on an IDENTICAL forced token path. cond/uncond
    separated to localise CFG-path drift. torch tensors here (core engine)."""
    import json as _j

    def _st(t):
        if t is None:
            return None
        f = t.detach().float().reshape(-1)
        return {"l2": float(torch.linalg.vector_norm(f).item()),
                "head": [round(float(x), 5) for x in f[:8].cpu().tolist()],
                "argmax": int(f.argmax().item())}
    with open(path, "a") as _fh:
        _j.dump({"step": step, "cond": _st(cond),
                 "uncond": _st(uncond), "comb": _st(comb)}, _fh)
        _fh.write("\n")


def _sample_token_np_core(
    logits_1d: np.ndarray, temperature: float,
    top_p: float = 1.0, min_p: float = 0.0,
    generated_ids: Optional[list] = None,
    repetition_penalty: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """Numpy speech-token sampler — byte-identical algorithm to the triton
    handler's _sample_token_np so both engines pick the same token from the same
    logits + seed (this is what reconciles top_p/min_p across modes). Replaces
    the previous torch.multinomial path, which used a different top_p formula and
    was unseeded (non-reproducible)."""
    logits = logits_1d.copy().astype(np.float64)
    if generated_ids and repetition_penalty != 1.0:
        for tid in set(generated_ids):
            if 0 <= tid < len(logits):
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
    if temperature == 0.0:
        # Greedy floor (dual_ar discipline): deterministic argmax → the four
        # modes pick identical tokens (logits agree to fp16), i.e. true
        # cross-mode reconciliation, no runaway. Used by --temperature 0.
        return int(np.argmax(logits))
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    if min_p > 0:
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()
        threshold = min_p * probs.max()
        logits[probs < threshold] = -np.inf
    if top_p < 1.0:
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        sorted_shifted = sorted_logits - sorted_logits.max()
        sorted_probs = np.exp(sorted_shifted) / np.exp(sorted_shifted).sum()
        cumsum = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumsum, top_p) + 1
        logits[sorted_idx[cutoff:]] = -np.inf
    logits_shifted = logits - logits.max()
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum()
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()
    _rng = rng if rng is not None else np.random
    return int(_rng.choice(len(probs), p=probs))


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

        # Text tokenization — data-driven, scoped via the generation config.
        # Models whose config declares start_text_token/stop_text_token (chatterbox's
        # t3) reproduce the vendor's EnTokenizer.text_to_tokens + generate() contract:
        # substitute a space marker so spaces survive BPE, then wrap the text with the
        # sot/eot markers. Models without those keys (OpenAudio, ...) keep the standard
        # encode path untouched — the wrap is NEVER applied unconditionally.
        sot = defaults.get("start_text_token")
        eot = defaults.get("stop_text_token")
        space_marker = defaults.get("text_space_marker")
        if sot is not None and eot is not None:
            _txt = prompt.replace(" ", space_marker) if space_marker else prompt
            _ids = tokenizer.encode(_txt, padding=False, add_special_tokens=False)
            if isinstance(_ids, torch.Tensor):
                _ids = _ids.flatten().tolist()
            elif _ids and isinstance(_ids[0], (list, tuple)):
                _ids = list(_ids[0])
            _ids = [int(sot)] + [int(t) for t in _ids] + [int(eot)]
            input_ids = torch.tensor([_ids], dtype=torch.long, device=device)
        else:
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
                        release_flow_memory(self.ctx.primary_device)
                else:
                    print(f"   [{comp_name}] Skipped (no reference audio)")

        # ── Step 3: Autoregressive speech token generation ──
        lm_name = lm_stage["component"]
        aux_weights_config = lm_stage.get("auxiliary_weights", {})

        from neurobrix.core.runtime.decode_bound import decode_bound  # NBX_DECODE_BOUND harness
        # CLI sampling overrides (global.*) take precedence over embedded
        # defaults, mirroring dual_ar / the autoregressive flow — --temperature 0
        # ⇒ deterministic greedy (the cross-mode-reconciling floor; stochastic
        # temp 0.8 is intelligible-but-not-token-identical across engines).
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

        # ── CFG via the centralized CFGEngine (single authority) ──
        # The engine owns the FORMULA (apply_guidance) + the CASCADE (from_topology:
        # CLI > topology > defaults guidance_scale + threshold) — no inline CFG here.
        # The t3_cfg backbone is traced batch=1 (a baked-batch reshape breaks batch=2),
        # so CFG runs SEQUENTIALLY: two batch=1 forwards (cond / text-zeroed uncond) per
        # step, then engine.apply_guidance. Models with no guidance_scale -> CFG disabled
        # (cond-only path unchanged, R23-safe).
        from neurobrix.core.cfg.engine import CFGEngine
        do_cfg = False
        guidance_scale = 1.0
        try:
            _cfg_engine = CFGEngine.from_topology(
                self.ctx, self._execute_component, lambda _c, _o: _o)
            do_cfg = _cfg_engine.is_enabled
            guidance_scale = _cfg_engine.guidance_scale
        except KeyError:
            do_cfg = False  # no guidance_scale in cascade -> disabled (cond-only)
        if do_cfg:
            print(f"   [{lm_name}] CFG enabled via CFGEngine (guidance_scale={guidance_scale}, sequential)")

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
        # Runs the traced cond_enc component (spkr_enc + Perceiver resampler +
        # emotion_adv_fc → cat → [1, 34, 1024]) on the embedded default-voice
        # conditioning. No hand-rolled perceiver — the graph owns the compute.
        cond_embeds = self._build_conditioning(
            speaker_emb, speech_emb_w, speech_pos_emb_w, device, dtype
        ).to(dtype=dtype)
        print(f"   [{lm_name}] Conditioning: {cond_embeds.shape}")

        # ── Build text embeddings ──
        # CFG uncond text = zero TOKEN embedding + learned POSITION (vendor order:
        # text_emb[1].zero_() THEN += text_pos_emb). So uncond_text = position-only,
        # NOT all-zeros — used to build the uncond context below.
        with torch.no_grad():
            text_embeds = F.embedding(input_ids, text_emb_w.to(dtype=dtype))
            uncond_text_embeds = torch.zeros_like(text_embeds)  # token zeroed
            if text_pos_emb_w is not None:
                # Learned position embeddings: index by token positions
                pos_ids = torch.arange(input_ids.shape[1], device=device)
                text_pos = F.embedding(pos_ids, text_pos_emb_w.to(dtype=dtype)).unsqueeze(0)
                text_embeds = text_embeds + text_pos
                uncond_text_embeds = uncond_text_embeds + text_pos  # zero-token + position

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
        # CFG uncond context (batch=1): zero the TEXT embedding (vendor text_emb[1].zero_());
        # speaker-cond + speech tokens are shared. Grown in lockstep with context_embeds.
        uncond_context = None
        if do_cfg:
            uncond_context = torch.cat(
                [cond_embeds, uncond_text_embeds, bos_embed], dim=1)
        cond_len = cond_embeds.shape[1]
        text_len = text_embeds.shape[1]

        print(f"   [{lm_name}] Initial context: {context_embeds.shape} "
              f"(cond={cond_len}, text={text_len}, speech=1)")
        print(f"   [{lm_name}] Generating speech tokens (max={max_tokens})...")
        start = time.perf_counter()

        generated_ids: list = []
        speech_head_w_typed = speech_head_w.to(dtype=dtype)

        # Diagnostic component-isolation hook (gated, default-off): force a fixed
        # speech-token sequence (skip decode) so the vocoder can be diffed across
        # engines on IDENTICAL tokens. Same class as NBX_DUMP_TIDS.
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
                      f"dumping t3 logits → {_t3dump}")
            else:
                generated_ids = list(_forced_seq)
                print(f"   [{lm_name}] FORCED {len(generated_ids)} speech tokens "
                      f"from {_force_ids} (decode skipped)")

        def _t3_logits(ctx_embeds):
            """One batch=1 backbone forward + speech_head → logits [1,1,vocab]."""
            seq_len = ctx_embeds.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            self.ctx.variable_resolver.resolved["global.inputs_embeds"] = ctx_embeds
            self.ctx.variable_resolver.resolved["inputs_embeds"] = ctx_embeds
            self.ctx.variable_resolver.resolved["global.position_ids"] = position_ids
            self.ctx.variable_resolver.resolved["position_ids"] = position_ids
            self._execute_component(lm_name, "forward", None)
            out = self._get_component_output(lm_name)
            if out is None:
                return None
            return torch.matmul(out[:, -1:, :], speech_head_w_typed.T)  # [1,1,vocab]

        # Deterministic sampler shared with the triton handler (same algorithm +
        # seed) — gives PER-MODE reproducibility. NB: it does NOT make the four
        # modes pick identical tokens (the per-engine logits differ at the fp16
        # level, so `choice` flips boundary tokens and the paths diverge). Seed
        # overridable for diagnostics (seed sweep). Same pattern as dual_ar.
        _seed = int(_os_fsi.environ.get("NBX_TTS_LLM_SEED", _TTS_LLM_SEED))
        _sampler_rng = np.random.RandomState(_seed)

        _teacher = bool(_force_ids and _t3dump)
        _nsteps = len(_forced_seq) if _teacher else (0 if _force_ids else max_tokens)
        for step in range(_nsteps):
            # Conditional (and, under CFG, unconditional) forward(s) → guided logits.
            cond_logits = _t3_logits(context_embeds)
            if cond_logits is None:
                break
            uncond_logits = None
            logits = cond_logits
            if do_cfg:
                uncond_logits = _t3_logits(uncond_context)
                if uncond_logits is None:
                    break
                # Centralized CFG formula (cond=context, uncond=text-zeroed context).
                logits = CFGEngine.apply_guidance(cond_logits, uncond_logits, guidance_scale)

            if _teacher:
                # Teacher-forcing: dump logits, advance along the forced token path
                # (NOT the sampled one) so all modes see the identical context.
                _dump_t3_logits(_t3dump, step, cond_logits, uncond_logits, logits)
                next_token = _forced_seq[step]
            else:
                # Sample via the shared seeded numpy sampler (identical algorithm to
                # the triton handler → reconciles top_p across modes).
                logits_np = logits[:, -1, :].float().detach().cpu().numpy()[0]
                next_token = _sample_token_np_core(
                    logits_np, temperature, top_p, min_p,
                    generated_ids=generated_ids,
                    repetition_penalty=repetition_penalty,
                    rng=_sampler_rng,
                )
            generated_ids.append(next_token)

            if next_token == eos_token_id and not _teacher:
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

            # Grow both contexts in lockstep (shared speech sequence).
            context_embeds = torch.cat([context_embeds, token_embed], dim=1)
            if do_cfg:
                uncond_context = torch.cat([uncond_context, token_embed], dim=1)

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

        if _os_fsi.environ.get("NBX_DUMP_SPEECH_IDS"):
            import json as _jd_si
            with open(_os_fsi.environ["NBX_DUMP_SPEECH_IDS"], "w") as _f_si:
                _jd_si.dump([int(t) for t in speech_ids], _f_si)
            print(f"   [{lm_name}] Dumped {len(speech_ids)} speech ids -> "
                  f"{_os_fsi.environ['NBX_DUMP_SPEECH_IDS']}")

        self.ctx.variable_resolver.resolved["global.generated_token_ids"] = speech_ids
        speech_tokens = torch.tensor([speech_ids], dtype=torch.long, device=device)
        self.ctx.variable_resolver.resolved["global.speech_tokens"] = speech_tokens
        self.ctx.variable_resolver.resolved["speech_tokens"] = speech_tokens

        if not self.ctx.persistent_mode:
            self._unload_component_weights(lm_name)
            release_flow_memory(self.ctx.primary_device)

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
                conds = self._load_default_conditioning()
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
                            elif iname.startswith("ref_dict.") and conds is not None \
                                    and f"gen.{iname[len('ref_dict.'):]}" in conds:
                                # Reference-voice ref-dict from the embedded conditioning
                                comp_inputs[iname] = conds[f"gen.{iname[len('ref_dict.'):]}"].to(device)
                            else:
                                # Dummy inputs for any unmapped fields
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
                    release_flow_memory(self.ctx.primary_device)

        return self.ctx.variable_resolver.resolve_all()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_compute_dtype(self) -> torch.dtype:
        """Get compute dtype from the Prism-resolved plan (FlowContext.compute_dtype).

        The previous local copy walked `plan.allocations`, an attribute the
        ExecutionPlan never had (`plan.components` is the real one) — the
        walk was dead code and the method silently fell back to
        `defaults.json`. The FlowContext resolver reads the actual plan.
        """
        return self.ctx.compute_dtype()

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

    def _load_default_conditioning(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load the embedded default-voice conditioning (built by forge).

        runtime/default_conditioning.safetensors holds the conds.pt inputs:
        t3.{speaker_emb, cond_prompt_speech_tokens, emotion_adv} for cond_enc
        and gen.{prompt_token, prompt_token_len, prompt_feat, embedding} for
        the s3gen ref-dict. Returns None if the build has no default voice.
        """
        if hasattr(self, "_default_conditioning_cache"):
            return self._default_conditioning_cache
        from safetensors.torch import load_file
        path = Path(self.ctx.pkg.cache_path) / "runtime" / "default_conditioning.safetensors"
        conds = load_file(str(path)) if path.exists() else None
        self._default_conditioning_cache = conds
        return conds

    def _graph_input_names(self, executor) -> List[str]:
        """Return the graph DAG input_name list for a component executor."""
        dag = getattr(executor, "_dag", None)
        if not dag:
            return []
        return [t["input_name"] for t in dag.get("tensors", {}).values()
                if t.get("input_name")]

    def _build_conditioning(
        self,
        speaker_emb: Optional[torch.Tensor],
        speech_emb_w: Optional[torch.Tensor],
        speech_pos_emb_w: Optional[torch.Tensor],
        device: str, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build conditioning by running the traced cond_enc component.

        cond_enc is the data-driven T3CondEnc graph: spkr_enc(speaker_emb) +
        Perceiver(speech_emb(cond_prompt_speech_tokens)) + emotion_adv_fc →
        cat → [1, 34, 1024]. Inputs come from the embedded default-voice
        conditioning (or a live speaker_emb when reference audio is given).
        ZERO FALLBACK: no hand-rolled perceiver, no zero-conditioning path.
        """
        cond_name = "cond_enc"
        cond_executor = self.ctx.executors.get(cond_name)
        if cond_executor is None:
            raise RuntimeError(
                "ZERO FALLBACK: tts_llm conditioning requires a traced 'cond_enc' "
                "component. Re-import a build that traces cond_enc."
            )
        conds = self._load_default_conditioning()
        if conds is None:
            raise RuntimeError(
                "ZERO FALLBACK: missing runtime/default_conditioning.safetensors "
                "(default-voice inputs for cond_enc)."
            )
        if speech_emb_w is None:
            raise RuntimeError(
                "ZERO FALLBACK: speech_emb weight required to embed prompt speech tokens."
            )

        cond_tokens = conds["t3.cond_prompt_speech_tokens"].to(device)
        with torch.no_grad():
            cond_prompt_speech_emb = F.embedding(cond_tokens, speech_emb_w.to(dtype=dtype))
            # Vendor prepare_conditioning (t3.py:97-100, is_gpt=False) adds learned
            # positional embeddings to the prompt-token embedding BEFORE T3CondEnc — the
            # embedding is a pre-cond_enc step in the vendor too ("instead of in T3CondEnc").
            # Omitting it weakens the conditioning (cond prefix diverges by max|diff|=2.6),
            # flattening the t3 distribution -> temp-0.8 sampling drift. Same speech_pos_emb
            # this flow already applies to the bos + generated speech tokens.
            if speech_pos_emb_w is not None:
                _L = cond_tokens.shape[1]
                _pos_idx = torch.arange(_L, device=device).clamp(max=speech_pos_emb_w.shape[0] - 1)
                cond_prompt_speech_emb = cond_prompt_speech_emb + F.embedding(
                    _pos_idx, speech_pos_emb_w.to(dtype=dtype))

        # Live reference speaker_emb overrides the default voice when present
        spk = speaker_emb.view(1, -1) if speaker_emb is not None \
            else conds["t3.speaker_emb"].to(device)

        comp_input_values = {
            "speaker_emb": spk,
            "cond_prompt_speech_emb": cond_prompt_speech_emb,
            "emotion_adv": conds["t3.emotion_adv"].to(device),
            "cond_prompt_speech_tokens": cond_tokens,
        }
        comp_inputs = {n: comp_input_values[n]
                       for n in self._graph_input_names(cond_executor)
                       if n in comp_input_values}

        self._ensure_weights_loaded(cond_name)
        output = cond_executor.run(comp_inputs)
        cond_emb = next(iter(output.values())) if isinstance(output, dict) else output
        if not self.ctx.persistent_mode:
            self._unload_component_weights(cond_name)
            release_flow_memory(self.ctx.primary_device)
        return cond_emb

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
        next_token = int(device_multinomial(probs, num_samples=1).item())
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
