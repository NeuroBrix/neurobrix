"""Triton NextTokenDiffusion Flow — zero-torch VibeVoice TTS.

Port of core/flow/next_token_diffusion.py to the NBXTensor + Triton substrate
(R33). Same algorithm: an autoregressive LM emits a control token per step
(speech_start / speech_end / speech_diffusion / eos); on every speech_diffusion
token a small DPM-Solver++ v-prediction head samples one acoustic latent
conditioned on the LM hidden state; the latent is decoded to a 3200-sample chunk
by the acoustic tokenizer, re-encoded by the semantic tokenizer, and the
acoustic+semantic connectors fuse the two into the embedding fed back to the LM.
The final 24 kHz waveform is the concatenation of the per-step chunks.

Zero torch: the heavy compute (LM forward, diffusion head, acoustic/semantic
tokenizers, connectors) runs through the component graphs as NBXTensor; the
diffusion sampler is the zero-torch `TritonDPMSolverPPScheduler` + the
`TritonCFGEngine` guidance formula; init noise comes from a seeded numpy RNG
uploaded as an NBXTensor. numpy is used only for CPU orchestration glue
(prompt assembly, embedding-table lookups, scalar latent scaling) exactly as the
sibling triton/flow/tts_llm.py does — numpy is not torch and never touches the
GPU compute path.

CFG mirrors the compiled engine: a prompt-free negative LM context grows with the
same per-step feedback embeddings as the positive one and is forwarded on
diffusion steps; the batch=2 head output is split into cond/uncond and combined
via `uncond + scale*(cond - uncond)`. cfg_scale=1.0 disables it.
"""

import gc
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.triton.scheduler.dpm_solver_pp import TritonDPMSolverPPScheduler
from neurobrix.triton.cfg.engine import TritonCFGEngine


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, NBXTensor):
        return t.to(NBXDtype.float32).numpy() if t.nbx_dtype != NBXDtype.float32 else t.numpy()
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t)


def _parse_device_idx(device) -> int:
    s = str(device)
    return int(s.split(":")[1]) if ":" in s else 0


class TritonNextTokenDiffusionEngine:
    """Triton-mode VibeVoice next-token-diffusion TTS (NBXTensor end-to-end)."""

    LM = "model.language_model"
    HEAD = "model.prediction_head"
    ACOUSTIC_TOK = "model.acoustic_tokenizer"
    SEMANTIC_TOK = "model.semantic_tokenizer"
    ACOUSTIC_CONN = "model.acoustic_connector"
    SEMANTIC_CONN = "model.semantic_connector"

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

    # ─── public entry ──────────────────────────────────────────────────────────
    def execute(self) -> Dict[str, Any]:
        defaults = self.ctx.pkg.defaults
        DeviceAllocator.set_device(_parse_device_idx(self.ctx.primary_device))

        tok = self.ctx.modules.get("tokenizer")
        if tok is None:
            raise RuntimeError("ZERO FALLBACK: next_token_diffusion requires a tokenizer module.")

        # ── special token ids (no hardcode: resolved from tokenizer/defaults) ──
        def tid(name: str) -> Optional[int]:
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
            return int(ids[0]) if ids and len(ids) == 1 else None

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
        valid_arr = np.array(valid_token_ids, dtype=np.int64)

        # ── generation / diffusion params (data-driven) ──
        _ov = self.ctx.variable_resolver.resolved
        max_steps = int(_ov.get("global.max_tokens", defaults.get("max_tokens", 2048)))
        ddpm_steps = int(defaults.get("ddpm_num_inference_steps", 20))
        _cfg_override = _ov.get("global.guidance_scale")
        cfg_scale = float(_cfg_override) if _cfg_override is not None else float(defaults.get("cfg_scale", 1.3))
        _seed = _ov.get("global.seed")
        self._rng = np.random.RandomState(int(_seed) if _seed is not None else 0)
        scaling = float(defaults.get("speech_scaling_factor", 0.1962890625))
        bias = float(defaults.get("speech_bias_factor", -0.04931640625))
        vae_dim = int(defaults.get("acoustic_vae_dim", 64))

        prompt = _ov.get("global.prompt")
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

        for comp in (self.LM, self.HEAD, self.ACOUSTIC_TOK, self.SEMANTIC_TOK,
                     self.ACOUSTIC_CONN, self.SEMANTIC_CONN):
            self._ensure_weights_loaded(comp)

        embed_np = self._embed_weight_np(self.LM)
        if embed_np is None:
            raise RuntimeError("ZERO FALLBACK: could not locate tied embed weight in language_model.")

        # Prefill embeddings (numpy gather over the embedding table).
        inputs_embeds_np = embed_np[np.array(prompt_ids, dtype=np.int64)][np.newaxis, :, :]  # [1,S,H]
        use_cfg = cfg_scale != 1.0
        neg_inputs_embeds_np = embed_np[np.array([speech_start_id], dtype=np.int64)][np.newaxis, :, :]  # [1,1,H]

        print(f"   [{self.LM}] next-token-diffusion (max_steps={max_steps}, "
              f"ddpm_steps={ddpm_steps}, cfg={cfg_scale})...")
        start = time.perf_counter()

        emitted_tokens: List[int] = []
        audio_chunks: List[NBXTensor] = []
        n_diffusion = 0
        step = -1

        for step in range(max_steps):
            hidden_np = self._lm_forward_np(inputs_embeds_np)
            if hidden_np is None:
                raise RuntimeError(f"ZERO FALLBACK: language_model produced no hidden state at step {step}.")
            last_hidden_np = hidden_np[:, -1, :]                              # [1,H]

            # logits constrained to the valid control-token ids, greedy argmax.
            logits_valid = last_hidden_np @ embed_np[valid_arr].T             # [1, n_valid]
            next_token = int(valid_arr[int(np.argmax(logits_valid[0]))])
            emitted_tokens.append(next_token)

            if step < 8 or step % 16 == 0:
                _tname = ("eos" if next_token == eos_token_id else
                          "start" if next_token == speech_start_id else
                          "end" if next_token == speech_end_id else
                          "diff" if next_token == speech_diffusion_id else str(next_token))
                print(f"   [{self.LM}] step {step}: tok={_tname} "
                      f"(diff_so_far={n_diffusion}, seq={inputs_embeds_np.shape[1]})", flush=True)

            if next_token == eos_token_id:
                break

            next_embed_np = embed_np[np.array([next_token], dtype=np.int64)][np.newaxis, :, :]  # [1,1,H]

            if next_token == speech_diffusion_id:
                n_diffusion += 1
                pos_cond_np = last_hidden_np                                  # [1,H]
                if use_cfg:
                    neg_hidden_np = self._lm_forward_np(neg_inputs_embeds_np)
                    if neg_hidden_np is None:
                        raise RuntimeError(f"ZERO FALLBACK: negative language_model gave no hidden at step {step}.")
                    neg_cond_np = neg_hidden_np[:, -1, :]                     # [1,H]
                else:
                    neg_cond_np = pos_cond_np
                speech_latent = self._sample_speech_tokens(
                    pos_cond_np, neg_cond_np, cfg_scale, ddpm_steps, vae_dim, defaults)  # NBXTensor [1,vae]

                latent_np = _to_numpy(speech_latent)                          # [1,vae]
                scaled_np = (latent_np / scaling - bias)[:, np.newaxis, :]    # [1,1,vae]
                chunk = self._acoustic_decode(scaled_np)                      # NBXTensor [1,1,3200]
                if chunk is None:
                    raise RuntimeError(f"ZERO FALLBACK: acoustic_tokenizer returned no audio at step {step}.")
                audio_chunks.append(chunk)

                acoustic_embed_np = _to_numpy(self._connector(self.ACOUSTIC_CONN, latent_np))   # [1,H]
                semantic_features_np = _to_numpy(self._semantic_encode(chunk))                  # [1,Td,128] or [1,128]
                if semantic_features_np.ndim == 3:
                    semantic_features_np = semantic_features_np.mean(axis=1)                    # [1,128]
                semantic_embed_np = _to_numpy(self._connector(self.SEMANTIC_CONN, semantic_features_np))  # [1,H]
                next_embed_np = (acoustic_embed_np + semantic_embed_np)[:, np.newaxis, :]       # [1,1,H]

            inputs_embeds_np = np.concatenate([inputs_embeds_np, next_embed_np], axis=1)
            if use_cfg:
                neg_inputs_embeds_np = np.concatenate([neg_inputs_embeds_np, next_embed_np], axis=1)

        elapsed = (time.perf_counter() - start) * 1000
        print(f"   [{self.LM}] {step + 1} steps, {n_diffusion} speech_diffusion tokens in {elapsed:.0f}ms")
        print(f"   [{self.LM}] emitted histogram: start={emitted_tokens.count(speech_start_id)} "
              f"diff={emitted_tokens.count(speech_diffusion_id)} end={emitted_tokens.count(speech_end_id)} "
              f"eos={emitted_tokens.count(eos_token_id)}")

        if not audio_chunks:
            raise RuntimeError(
                "ZERO FALLBACK: next_token_diffusion produced no audio "
                f"(emitted {len(emitted_tokens)} tokens, {n_diffusion} diffusion).")

        waveform = audio_chunks[0] if len(audio_chunks) == 1 else NBXTensor.cat(audio_chunks, -1)  # [1,1,T]
        sr = float(defaults.get("sample_rate", 24000))
        print(f"   [Output] waveform {list(waveform.shape)} ({waveform.shape[-1] / sr:.2f}s)")
        self.ctx.variable_resolver.resolved["global.output_audio"] = waveform

        if not self.ctx.persistent_mode:
            for comp in (self.LM, self.HEAD, self.ACOUSTIC_TOK, self.SEMANTIC_TOK,
                         self.ACOUSTIC_CONN, self.SEMANTIC_CONN):
                self._unload_component_weights(comp)
            gc.collect()

        return self.ctx.variable_resolver.resolve_all()

    # ─── diffusion sampling (NBXTensor + zero-torch scheduler/CFG) ─────────────
    def _sample_speech_tokens(self, pos_cond_np, neg_cond_np, cfg_scale, ddpm_steps, vae_dim, defaults) -> NBXTensor:
        sched = TritonDPMSolverPPScheduler({
            "num_train_timesteps": int(defaults.get("ddpm_num_steps", 1000)),
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": defaults.get("ddpm_beta_schedule", "cosine"),
            "prediction_type": defaults.get("prediction_type", "v_prediction"),
            "solver_order": 2,
            "algorithm_type": "dpmsolver++",
            "solver_type": "midpoint",
            "final_sigmas_type": "zero",
            "lower_order_final": True,
        })
        sched.set_timesteps(ddpm_steps)

        cond_nbx = NBXTensor.from_numpy(np.concatenate([pos_cond_np, neg_cond_np], axis=0).astype(np.float32))  # [2,H]
        speech = NBXTensor.from_numpy(self._rng.standard_normal((2, vae_dim)).astype(np.float32))               # [2,vae]

        for i, t in enumerate(sched.timesteps):
            half = speech[:1]                                                # [1,vae]
            combined = NBXTensor.cat([half, half], 0)                        # [2,vae]
            t_val = float(sched._ts_np[i])
            t_batch = NBXTensor.from_numpy(np.array([t_val, t_val], dtype=np.float32))  # [2]
            eps = self._head_forward(combined, t_batch, cond_nbx)            # [2,vae]
            cond_eps = eps[:1]
            uncond_eps = eps[1:2]
            half_eps = TritonCFGEngine.apply_guidance(cond_eps, uncond_eps, cfg_scale)
            eps_full = NBXTensor.cat([half_eps, half_eps], 0)                # [2,vae]
            out = sched.step(eps_full, t, speech)
            speech = out["prev_sample"] if isinstance(out, dict) else out
        return speech[:1]                                                    # [1,vae]

    # ─── component invocations (executor.run, NBXTensor in/out) ────────────────
    def _lm_forward_np(self, inputs_embeds_np) -> Optional[np.ndarray]:
        seq = inputs_embeds_np.shape[1]
        embeds = NBXTensor.from_numpy(inputs_embeds_np.astype(np.float32))
        pos = NBXTensor.from_numpy(np.arange(seq, dtype=np.int64)[np.newaxis, :])
        out = self.ctx.executors[self.LM].run({"inputs_embeds": embeds, "position_ids": pos})
        t = self._primary(out)
        return _to_numpy(t) if t is not None else None

    def _head_forward(self, noisy_images, timesteps, condition) -> NBXTensor:
        out = self.ctx.executors[self.HEAD].run(
            {"noisy_images": noisy_images, "timesteps": timesteps, "condition": condition})
        return self._require(out, self.HEAD)

    def _acoustic_decode(self, scaled_np) -> Optional[NBXTensor]:
        latents = NBXTensor.from_numpy(scaled_np.astype(np.float32))
        out = self.ctx.executors[self.ACOUSTIC_TOK].run({"latents": latents})
        return self._primary(out)

    def _semantic_encode(self, waveform_chunk) -> NBXTensor:
        out = self.ctx.executors[self.SEMANTIC_TOK].run({"input_values": waveform_chunk})
        return self._require(out, self.SEMANTIC_TOK)

    def _connector(self, comp_name, x_2d_np) -> NBXTensor:
        x = NBXTensor.from_numpy(np.asarray(x_2d_np).astype(np.float32))
        out = self.ctx.executors[comp_name].run({"x": x})
        return self._require(out, comp_name)

    # ─── helpers ───────────────────────────────────────────────────────────────
    def _require(self, out, comp_name: str) -> NBXTensor:
        t = self._primary(out)
        if t is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp_name} produced no tensor output.")
        return t

    @staticmethod
    def _primary(out):
        if isinstance(out, NBXTensor):
            return out
        if isinstance(out, dict):
            for k in ("last_hidden_state", "output", "output_0"):
                if k in out and isinstance(out[k], NBXTensor):
                    return out[k]
            for v in out.values():
                if isinstance(v, NBXTensor):
                    return v
            return None
        return out

    @staticmethod
    def _encode_nopad(tok, text: str, add_special: bool = False) -> List[int]:
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

    def _embed_weight_np(self, comp_name: str) -> Optional[np.ndarray]:
        executor = self.ctx.executors.get(comp_name)
        if executor is None or not hasattr(executor, "_weights"):
            return None
        for key in executor._weights:
            if "token_embed" in key or "embed" in key:
                w = executor._weights[key]
                arr = _to_numpy(w)
                if arr.ndim == 2:
                    return arr.astype(np.float32)
        return None
