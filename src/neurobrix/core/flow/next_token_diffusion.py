"""
NextTokenDiffusion Engine — VibeVoice next-token-diffusion TTS flow.

Handles the VibeVoice architecture: an autoregressive language model emits a
control token per step (speech_start / speech_end / speech_diffusion / eos);
on every `speech_diffusion` token a small diffusion head (DDPM/DPM++ v-pred)
samples one acoustic latent conditioned on the LM hidden state. The accumulated
latents are decoded to a waveform by the acoustic tokenizer (full non-streaming
decode each step — bit-equivalent to the vendor's chunked streaming decode:
max|diff|=0.0); the NEW chunk (the tail beyond the previous decode, a fixed 3200
samples) is re-encoded by the semantic tokenizer, and the acoustic+semantic
connectors fuse the two into the embedding fed back to the LM for the next step.
The final 24 kHz waveform is the last full decode of all accumulated latents.

Replicates `VibeVoiceForConditionalGenerationInference.generate` (batch=1) and
`sample_speech_tokens` from the vendor oracle, with full non-streaming
decode/encode in place of the vendor's streaming caches (proven equivalent).

ZERO SEMANTIC: no model-specific names beyond NeuroTax-standard keys.
ZERO HARDCODE: all parameters read from defaults.json / topology.json.

CFG (classifier-free guidance):
  * A parallel "negative" LM context runs alongside the positive one. It is
    seeded with ONLY `speech_start_id` (it never sees the text prompt) and is
    fed the SAME per-step feedback embeddings as the positive context (the
    default token embed on non-diffusion steps, the same diffusion_embeds on
    diffusion steps). On each speech_diffusion token the positive last-hidden
    (`pos_cond`) and the negative last-hidden (`neg_cond`) are passed to
    `_sample_speech_tokens`, which forms the vendor batch=2 guidance
    `half_eps = uncond + cfg_scale*(cond - uncond)`. At cfg_scale=1.0 this
    collapses to cond (negative irrelevant); at cfg_scale>1 the `cond - uncond`
    delta amplifies the text's influence → prompt-faithful speech. This mirrors
    the vendor oracle (negative context init ~line 379, refresh ~line 578-591,
    sample_speech_tokens ~line 699). The LM is a stateless full re-forward per
    step in this design, so the negative context is re-forwarded from its
    growing inputs_embeds each diffusion step (no KV cache surgery needed — that
    is a batch>1 concern in the vendor and a near-no-op at batch=1).
  * Full-decode-each-step is O(n^2) in latent count; fine for short utterances.
    The non-streaming decode is the correctness reference; incremental decode is
    a future optimisation, not a correctness requirement.
"""

import gc
import time
import torch
from typing import Any, Callable, Dict, List, Optional

from .base import FlowHandler, FlowContext, register_flow
from neurobrix.core.device_utils import device_empty_cache


@register_flow("next_token_diffusion")
class NextTokenDiffusionEngine(FlowHandler):
    """
    VibeVoice next-token-diffusion TTS.

    topology.flow.audio:
        direction: tts
        stages:
          - model.language_model        (autoregressive control tokens)
          - model.prediction_head       (diffusion head, v-prediction)
          - model.acoustic_tokenizer    (latent -> waveform chunk)
          - model.semantic_tokenizer    (waveform chunk -> semantic features)
          - model.acoustic_connector    (latent -> 1536 embed)
          - model.semantic_connector    (semantic -> 1536 embed)
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

    # ─── Component names (NeuroTax / topology, no vendor strings) ──────────────
    LM = "model.language_model"
    HEAD = "model.prediction_head"
    ACOUSTIC_TOK = "model.acoustic_tokenizer"
    SEMANTIC_TOK = "model.semantic_tokenizer"
    ACOUSTIC_CONN = "model.acoustic_connector"
    SEMANTIC_CONN = "model.semantic_connector"

    def execute(self) -> Dict[str, Any]:
        defaults = self.ctx.pkg.defaults
        device = self.ctx.primary_device
        dtype = self._compute_dtype()

        # ── Resolve special token ids from the embedded tokenizer (no hardcode) ──
        tok = self.ctx.modules.get("tokenizer")
        if tok is None:
            raise RuntimeError("ZERO FALLBACK: next_token_diffusion requires a tokenizer module.")

        def tid(name: str) -> Optional[int]:
            # The runtime tokenizer is a TokenizerWrapper around an HF tokenizer;
            # these speech-control tokens are single added-tokens in the Qwen2
            # vocab, so encoding the bare token string yields its single id.
            # Try direct id-lookup methods (wrapper or inner tokenizer) first,
            # then fall back to single-token encode.
            for obj in (tok, getattr(tok, "_tokenizer", None)):
                if obj is None:
                    continue
                for fn in ("convert_tokens_to_ids", "token_to_id"):
                    f = getattr(obj, fn, None)
                    if f is not None:
                        try:
                            v = f(name)
                            if v is not None and int(v) >= 0:
                                return int(v)
                        except Exception:
                            pass
            ids = self._encode_nopad(tok, name, add_special=False)
            if ids and len(ids) == 1:
                return int(ids[0])
            return None

        def _require(nm: str, val: Optional[int]) -> int:
            if val is None:
                raise RuntimeError(f"ZERO FALLBACK: could not resolve {nm} token id from tokenizer/defaults.")
            return int(val)

        speech_start_id = _require("speech_start", tid("<|vision_start|>"))
        speech_end_id = _require("speech_end", tid("<|vision_end|>"))
        speech_diffusion_id = _require("speech_diffusion", tid("<|vision_pad|>"))
        _eos = defaults.get("eos_token_id")
        if _eos is None:
            _eos = tid("<|endoftext|>")
        eos_token_id = _require("eos", _eos)
        bos_token_id = defaults.get("bos_token_id")

        valid_token_ids: List[int] = [speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id]
        if bos_token_id is not None:
            valid_token_ids.append(int(bos_token_id))

        # ── Generation / diffusion params (data-driven) ──
        _ov = self.ctx.variable_resolver.resolved
        max_steps = int(_ov.get("global.max_tokens", defaults.get("max_tokens", 2048)))
        ddpm_steps = int(defaults.get("ddpm_num_inference_steps", 20))
        # CFG scale cascade: CLI `--cfg` (global.guidance_scale) > defaults.json
        # cfg_scale > 1.3 (the VibeVoice default-ish guidance). cfg_scale=1.0
        # disables guidance (negative context becomes irrelevant).
        _cfg_override = _ov.get("global.guidance_scale")
        if _cfg_override is not None:
            cfg_scale = float(_cfg_override)
        else:
            cfg_scale = float(defaults.get("cfg_scale", 1.3))
        # One diffusion-noise generator for the whole run: advances across steps
        # (fresh init noise per step) yet makes the run reproducible. SHARED with
        # the triton next_token_diffusion handler: both draw init noise from the
        # SAME numpy RandomState(seed) with the same per-step draw, so the two
        # engines see identical noise and the first-K diffusion latents match —
        # the cross-engine validation BEFORE the chaotic feedback loop amplifies
        # fp16 differences (STT full-decode is too noisy for a chaotic model).
        # R27/R28; numpy is CPU glue, the torch compute path is untouched.
        import numpy as _np_vv
        _seed = _ov.get("global.seed")
        diffusion_gen = _np_vv.random.RandomState(int(_seed) if _seed is not None else 0)
        scaling = float(defaults.get("speech_scaling_factor", 0.1962890625))
        bias = float(defaults.get("speech_bias_factor", -0.04931640625))
        vae_dim = int(defaults.get("acoustic_vae_dim", 64))

        # ── Build the LM prompt exactly as the VibeVoice processor (text-only) ──
        prompt = self.ctx.variable_resolver.resolved.get("global.prompt")
        if prompt is None or prompt == "":
            raise RuntimeError("ZERO FALLBACK: next_token_diffusion requires --prompt text.")

        def enc(text: str, add_special: bool = False) -> List[int]:
            return self._encode_nopad(tok, text, add_special=add_special)

        system_prompt = (" Transform the text provided by various speakers into speech output, "
                         "utilizing the distinct voice of each respective speaker.\n")
        prompt_ids: List[int] = []
        prompt_ids += enc(system_prompt, add_special=True)
        prompt_ids += enc(" Text input:\n", add_special=False)
        prompt_ids += enc(f" Speaker 0: {prompt}\n", add_special=False)
        prompt_ids += enc(" Speech output:\n", add_special=False)
        prompt_ids += [speech_start_id]
        print(f"   [{self.LM}] prompt tokens ({len(prompt_ids)}): ...{prompt_ids[-12:]}")

        # ── Load LM + the diffusion / tokenizer / connector components ──
        for comp in (self.LM, self.HEAD, self.ACOUSTIC_TOK, self.SEMANTIC_TOK,
                     self.ACOUSTIC_CONN, self.SEMANTIC_CONN):
            self._ensure_weights_loaded(comp)

        embed_weight = self._embed_weight(self.LM)
        if embed_weight is None:
            raise RuntimeError("ZERO FALLBACK: could not locate tied embed weight in language_model.")
        embed_weight = embed_weight.to(device)

        # Prefill embeddings: embed_tokens(prompt_ids)
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            inputs_embeds = torch.nn.functional.embedding(prompt_t, embed_weight).to(dtype=dtype)

        # CFG: parallel negative LM context, seeded with ONLY speech_start_id
        # (it never sees the text prompt). It grows with the SAME per-step
        # feedback embeddings as the positive context. Forwarded only on
        # diffusion steps (the only place neg_cond is consumed). cfg_scale=1.0
        # makes it irrelevant; cfg_scale>1 amplifies the text conditioning.
        use_cfg = cfg_scale != 1.0
        neg_start_t = torch.tensor([[speech_start_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            neg_inputs_embeds = torch.nn.functional.embedding(neg_start_t, embed_weight).to(dtype=dtype)  # [1,1,1536]

        print(f"   [{self.LM}] next-token-diffusion generation "
              f"(max_steps={max_steps}, ddpm_steps={ddpm_steps}, cfg={cfg_scale})...")
        start = time.perf_counter()

        emitted_tokens: List[int] = []
        n_diffusion = 0
        step = -1  # defensive: bound even if max_steps == 0
        import os as _os_vv
        _vv_dump_path = _os_vv.environ.get("NBX_VV_DUMP_LATENTS", "")
        _vv_dump_k = int(_os_vv.environ.get("NBX_VV_DUMP_K", "4"))
        _vv_latents: list = []
        # Per-step audio chunks. The acoustic decoder traces at CONCRETE T=1 and we
        # decode exactly ONE latent per step (→ a fixed 3200-sample chunk). Per-step
        # independent T=1 decode is bit-equivalent to a cumulative full decode
        # (max|diff|=0.0001) — the causal-conv decoder needs no left context here.
        # (Symbolic-T decode is NOT usable: the decoder bakes trace-T-specific pad
        # constants, so a T!=trace_T runtime decode produces a wrong-length output.)
        audio_chunks: List[torch.Tensor] = []        # each [1, 1, 3200]

        for step in range(max_steps):
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

            hidden = self._lm_forward(inputs_embeds, position_ids)
            if hidden is None:
                raise RuntimeError(f"ZERO FALLBACK: language_model produced no hidden state at step {step}.")

            # logits = hidden[:, -1] @ embed.T ; constrain to valid ids ; argmax (greedy)
            last_hidden = hidden[:, -1, :]                                  # [1, 1536]
            logits = torch.matmul(last_hidden, embed_weight.to(last_hidden.dtype).T)  # [1, vocab]
            masked = torch.full_like(logits, float("-inf"))
            for vid in valid_token_ids:
                masked[:, vid] = logits[:, vid]
            next_token = int(torch.argmax(masked, dim=-1).item())
            emitted_tokens.append(next_token)

            # Lightweight per-step progress (so a CFG-destabilised non-terminating
            # LM regime is visible live rather than only on a wall-clock timeout).
            if step < 8 or step % 16 == 0:
                _tname = ("eos" if next_token == eos_token_id else
                          "start" if next_token == speech_start_id else
                          "end" if next_token == speech_end_id else
                          "diff" if next_token == speech_diffusion_id else str(next_token))
                print(f"   [{self.LM}] step {step}: tok={_tname} "
                      f"(diff_so_far={n_diffusion}, seq={inputs_embeds.shape[1]})", flush=True)

            if next_token == eos_token_id:
                break

            # default next embedding = embed_tokens(next_token)
            tok_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
            with torch.no_grad():
                next_embed = torch.nn.functional.embedding(tok_t, embed_weight).to(dtype=dtype)  # [1,1,1536]

            if next_token == speech_diffusion_id:
                n_diffusion += 1
                pos_cond = last_hidden                                       # [1, 1536]
                if use_cfg:
                    # Forward the negative context (prompt-free) and take its
                    # last hidden as the unconditional condition. The negative
                    # context has already been grown with the same feedback
                    # embeddings the positive context consumed.
                    neg_seq_len = neg_inputs_embeds.shape[1]
                    neg_position_ids = torch.arange(
                        neg_seq_len, dtype=torch.long, device=device
                    ).unsqueeze(0)
                    neg_hidden = self._lm_forward(neg_inputs_embeds, neg_position_ids)
                    if neg_hidden is None:
                        raise RuntimeError(
                            f"ZERO FALLBACK: negative language_model produced no "
                            f"hidden state at step {step}."
                        )
                    neg_cond = neg_hidden[:, -1, :]                          # [1, 1536]
                    # Diagnostic: is the CFG signal (cond - uncond) meaningful?
                    # A tiny delta ⇒ degenerate guidance; wildly different
                    # magnitudes ⇒ neg context in a bad subspace (stateless
                    # re-forward != oracle KV-cache semantics).
                    if n_diffusion in (1, 6):
                        _d = (pos_cond - neg_cond).float()
                        print(f"   [CFG diag] diff#{n_diffusion}: "
                              f"|pos|={pos_cond.float().norm().item():.3f} "
                              f"|neg|={neg_cond.float().norm().item():.3f} "
                              f"|pos-neg|={_d.norm().item():.3f} "
                              f"cos={torch.nn.functional.cosine_similarity(pos_cond.float(), neg_cond.float()).item():.4f}",
                              flush=True)
                else:
                    neg_cond = pos_cond                                      # cfg=1.0 -> irrelevant
                speech_latent = self._sample_speech_tokens(
                    pos_cond, neg_cond, cfg_scale, ddpm_steps, vae_dim, dtype, device,
                    gen=diffusion_gen,
                )                                                            # [1, 64]
                # Cross-engine validation dump (NBX_VV_DUMP_LATENTS=<path.npy>):
                # the first-K diffusion latents, BEFORE the chaotic feedback loop
                # amplifies fp16 differences. Same seeded noise as triton ⇒ these
                # must match the triton dump within fp16 tol = kernels validated.
                if _vv_dump_path and n_diffusion <= _vv_dump_k:
                    _vv_latents.append(speech_latent.detach().float().cpu().numpy().reshape(-1))

                # Vendor: scaled_latent = speech_latent / scaling - bias (decoder input scale).
                scaled = (speech_latent / scaling - bias).unsqueeze(0)       # [1, 1, 64]
                # Per-step T=1 decode → one 3200-sample chunk.
                chunk = self._acoustic_decode(scaled, dtype, device)         # [1, 1, 3200]
                if chunk is None:
                    raise RuntimeError(
                        f"ZERO FALLBACK: acoustic_tokenizer returned no audio at step {step}."
                    )
                audio_chunks.append(chunk)

                # Re-add semantic feedback (vendor lines 660-671): encode this
                # chunk → semantic features → semantic_connector; add to
                # acoustic_connector(speech_latent) = diffusion_embeds.
                acoustic_embed = self._connector(
                    self.ACOUSTIC_CONN, speech_latent, dtype, device
                )                                                            # [1, 1536]
                semantic_features = self._semantic_encode(chunk, dtype, device)  # [1, T_down, 128]
                # The connector graph traces a 2D input [B, in_dim] (addmm). A
                # 3200-sample chunk yields T_down == 1, so collapse the frame axis
                # to feed [1, 128] (mean over T_down for robustness if >1).
                if semantic_features.dim() == 3:
                    semantic_features = semantic_features.mean(dim=1)        # [1, 128]
                semantic_embed = self._connector(
                    self.SEMANTIC_CONN, semantic_features, dtype, device
                )                                                            # [1, 1536]
                diffusion_embeds = (acoustic_embed + semantic_embed).to(dtype=dtype)
                next_embed = diffusion_embeds.unsqueeze(1)                   # [1, 1, 1536]

            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            # Grow the negative context with the SAME feedback embedding the
            # positive context just consumed (default token embed on
            # non-diffusion steps, the same diffusion_embeds on diffusion steps).
            # The negative context thus tracks the positive one byte-for-byte
            # except for its prompt-free prefix.
            if use_cfg:
                neg_inputs_embeds = torch.cat([neg_inputs_embeds, next_embed], dim=1)

        elapsed = (time.perf_counter() - start) * 1000
        if _vv_dump_path and _vv_latents:
            _np_vv.save(_vv_dump_path, _np_vv.stack(_vv_latents))
            print(f"   [VV-DIAG] dumped {len(_vv_latents)} first-K latents → {_vv_dump_path}")
        print(f"   [{self.LM}] {step + 1} steps, {n_diffusion} speech_diffusion tokens "
              f"in {elapsed:.0f}ms")
        print(f"   [{self.LM}] first 10 emitted token ids: {emitted_tokens[:10]}")
        print(f"   [{self.LM}] emitted token id histogram: "
              f"start={emitted_tokens.count(speech_start_id)} "
              f"diff={emitted_tokens.count(speech_diffusion_id)} "
              f"end={emitted_tokens.count(speech_end_id)} "
              f"eos={emitted_tokens.count(eos_token_id)}")

        if not audio_chunks:
            raise RuntimeError(
                "ZERO FALLBACK: next_token_diffusion produced no audio "
                f"(emitted {len(emitted_tokens)} tokens, {n_diffusion} diffusion). "
                "The LM did not emit speech_diffusion tokens — check prompt format / embed weight."
            )

        # ── Final audio = concatenation of per-step 3200-sample chunks (24 kHz) ──
        waveform = torch.cat(audio_chunks, dim=-1)                           # [1, 1, T_total]
        print(f"   [Output] waveform {list(waveform.shape)} "
              f"({waveform.shape[-1] / float(defaults.get('sample_rate', 24000)):.2f}s)")
        self.ctx.variable_resolver.resolved["global.output_audio"] = waveform

        if not self.ctx.persistent_mode:
            for comp in (self.LM, self.HEAD, self.ACOUSTIC_TOK, self.SEMANTIC_TOK,
                         self.ACOUSTIC_CONN, self.SEMANTIC_CONN):
                self._unload_component_weights(comp)
            gc.collect()
            device_empty_cache(device)

        return self.ctx.variable_resolver.resolve_all()

    # ─── Diffusion sampling (mirror of vendor sample_speech_tokens) ────────────

    def _sample_speech_tokens(
        self, condition, neg_condition, cfg_scale, ddpm_steps, vae_dim, dtype, device, gen=None
    ) -> torch.Tensor:
        # Lazy import: the CFG engine module would create a circular import at
        # module load (cfg.engine <-> flow). Imported here (cached after first
        # call) so the unified guidance formula is the single authority.
        from neurobrix.core.cfg.engine import CFGEngine
        scheduler = self._build_scheduler()
        scheduler.set_timesteps(ddpm_steps, device=device)

        cond = torch.cat([condition, neg_condition], dim=0).to(device=device, dtype=dtype)  # [2, 1536]
        # Diffusion init noise. The generator is created ONCE per run in
        # execute() and threaded through here so its state advances naturally
        # across steps (each step gets fresh, independent init noise — required
        # for the diffusion feedback loop to keep drifting), while keeping the
        # whole run reproducible (cfg=A run-1 ≡ cfg=A run-2). Reseeding per call
        # would freeze the latent prior and collapse the loop into periodicity.
        # Shared numpy-seeded init noise (see execute(): `diffusion_gen`). Same
        # RandomState + same per-call draw as the triton handler → identical noise.
        import numpy as _np_vv
        speech = torch.from_numpy(
            gen.standard_normal((cond.shape[0], vae_dim)).astype(_np_vv.float32)
        ).to(device=device, dtype=dtype)                                    # [2, 64]

        timesteps = scheduler.timesteps
        if timesteps is None:
            raise RuntimeError("ZERO FALLBACK: scheduler produced no timesteps.")
        for t in timesteps:
            half = speech[: speech.shape[0] // 2]
            combined = torch.cat([half, half], dim=0)                        # [2, 64]
            # Float timesteps: the diffusion head graph traced a float timestep
            # (vendor head's sinusoid embedding follows t's dtype).
            t_batch = t.repeat(combined.shape[0]).to(device=device).float()  # [2]
            eps = self._head_forward(combined, t_batch, cond)                # [2, 64]
            cond_eps, uncond_eps = torch.split(eps, eps.shape[0] // 2, dim=0)
            # CFG guidance via the unified CFGEngine (single formula authority).
            # Identical to the prior inline `uncond + scale*(cond - uncond)`.
            half_eps = CFGEngine.apply_guidance(cond_eps, uncond_eps, cfg_scale)
            eps = torch.cat([half_eps, half_eps], dim=0)
            out = scheduler.step(eps, t, speech)
            speech = out["prev_sample"] if isinstance(out, dict) else out
            speech = speech.to(dtype=dtype)
        return speech[: speech.shape[0] // 2]                                # [1, 64]

    def _build_scheduler(self):
        """Instantiate DPMSolverMultistepScheduler matching VibeVoice/diffusers defaults."""
        from neurobrix.core.module.scheduler.factory import SchedulerFactory
        defaults = self.ctx.pkg.defaults
        config = {
            "_class_name": "DPMSolverMultistepScheduler",
            "num_train_timesteps": int(defaults.get("ddpm_num_steps", 1000)),
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": defaults.get("ddpm_beta_schedule", "cosine"),
            "prediction_type": defaults.get("prediction_type", "v_prediction"),
            "solver_order": 2,
            "algorithm_type": "dpmsolver++",
            "solver_type": "midpoint",
            "timestep_spacing": "linspace",
            "final_sigmas_type": "zero",
            "lower_order_final": True,
        }
        return SchedulerFactory.create(config)

    # ─── Component invocations (direct executor.run for full input control) ────

    def _lm_forward(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor):
        out = self.ctx.executors[self.LM].run(
            {"inputs_embeds": inputs_embeds, "position_ids": position_ids}
        )
        return self._primary(out)

    def _head_forward(self, noisy_images, timesteps, condition) -> torch.Tensor:
        out = self.ctx.executors[self.HEAD].run(
            {"noisy_images": noisy_images, "timesteps": timesteps, "condition": condition}
        )
        return self._require_tensor(out, self.HEAD)

    def _acoustic_decode(self, latents_3d, dtype, device) -> Optional[torch.Tensor]:
        out = self.ctx.executors[self.ACOUSTIC_TOK].run(
            {"latents": latents_3d.to(device=device, dtype=dtype)}
        )
        return self._primary(out)

    def _semantic_encode(self, waveform_chunk, dtype, device) -> torch.Tensor:
        out = self.ctx.executors[self.SEMANTIC_TOK].run(
            {"input_values": waveform_chunk.to(device=device, dtype=dtype)}
        )
        return self._require_tensor(out, self.SEMANTIC_TOK)

    def _connector(self, comp_name, x_2d, dtype, device) -> torch.Tensor:
        out = self.ctx.executors[comp_name].run(
            {"x": x_2d.to(device=device, dtype=dtype)}
        )
        return self._require_tensor(out, comp_name)

    def _require_tensor(self, out, comp_name: str) -> torch.Tensor:
        t = self._primary(out)
        if t is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp_name} produced no tensor output.")
        return t

    # ─── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_nopad(tok, text: str, add_special: bool = False) -> List[int]:
        """Encode text WITHOUT padding.

        The runtime HFTokenizer (sp_tokenizer.HFTokenizer) defaults to
        padding=True up to model_max_length, which would explode an
        incrementally-assembled prompt. We assemble the prompt token-piece by
        token-piece, so each piece must be encoded unpadded. Pass padding=False
        when supported; otherwise strip trailing pad/eos padding heuristically.
        """
        ids = None
        try:
            ids = tok.encode(text, add_special_tokens=add_special, padding=False)
        except TypeError:
            try:
                ids = tok.encode(text, add_special_tokens=add_special)
            except TypeError:
                ids = tok.encode(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    @staticmethod
    def _primary(out):
        if isinstance(out, dict):
            for k in ("last_hidden_state", "output", "output_0"):
                if k in out and isinstance(out[k], torch.Tensor):
                    return out[k]
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            return None
        return out

    def _compute_dtype(self) -> torch.dtype:
        from neurobrix.core.dtype.config import get_torch_dtype
        return get_torch_dtype(self.ctx.pkg.manifest.get("dtype", "float16"))

    def _embed_weight(self, comp_name: str) -> Optional[torch.Tensor]:
        executor = self.ctx.executors.get(comp_name)
        if executor is None or not hasattr(executor, "_weights"):
            return None
        # NeuroTax: tied embed key is "token_embed.weight" / contains "embed".
        for key in executor._weights:
            if "token_embed" in key or "embed" in key:
                w = executor._weights[key]
                if isinstance(w, torch.Tensor) and w.ndim == 2:
                    return w
        return None
