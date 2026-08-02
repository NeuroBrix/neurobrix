"""Generative-speech leg, CFM/token2wav class (P-OMNI-GEN model 3/3) —
compiled engine.

Second speech-leg VARIANT, dispatched DATA-DRIVEN on the contract's
condition_type (`hidden_text_merge`) — the tap+chatml class keeps
core/flow/speech.py. Vendor chain (MiniCPM-o class,
_generate_speech_non_streaming):

  1. tts span over the FULL id sequence (prompt carries the tts-bos
     marker via the template; generation runs until the tts-eos or the
     text eos): span ids + their LAST-layer hidden rows from the final
     decode forward (causal ⇒ identical to the vendor's per-step
     hiddens).
  2. condition: text_embedding(ids) + L2norm(hidden_projection(hidden))
     — additive merge; prefix = [merged, text_eos, audio_bos] embeds
     (repurposed reserved slots, contract data).
  3. AR loop over the backbone component (prefill embeds+positions,
     then codec_embedding(prev) per step); codec head -> ChatTTS-class
     seeded draw (shared frontier draw_chattts: temperature -> window
     penalty -> TopP/TopK min-keep -> multinomial); min_new_token
     masks the codec EOS.
  4. token2wav: speaker constants (runtime/speech_constants.safetensors
     — BUILD-precomputed x-vector + prompt tokens + prompt mel; the
     runtime NEVER runs ONNX, R34) -> flow_encoder component on
     cat(prompt, generated) tokens -> CFM Euler loop (cosine t_span,
     CFG (1+r)*cond - r*uncond, initial noise from the SHARED
     seeded_gaussian frontier) over the flow_decoder component ->
     mel slice -> vocoder component -> 24 kHz waveform in
     resolved["global.output_audio"].

Every quantity from topology.flow.speech / pkg.defaults — ZERO
FALLBACK on missing contract keys; no model names anywhere.
"""

import math
import time
from typing import Any, Dict, List

import numpy as np
import torch

from neurobrix.kernels.seeded_draw import SeededDrawStream, seeded_gaussian


def _require(block: Dict[str, Any], key: str, where: str):
    val = block.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: topology.flow.speech is missing '{key}' "
            f"({where}) — re-emit the speech contract from the registry.")
    return val


class SpeechCFMLeg:
    """Compiled-engine CFM speech leg over the flow.speech contract."""

    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.ctx
        self.resolved = engine.ctx.variable_resolver.resolved

    def _run(self, comp: str, **inputs):
        self.engine._ensure_weights_loaded(comp)
        for name, val in inputs.items():
            self.resolved[f"{comp}.{name}"] = val
            self.resolved[f"global.{name}"] = val
        self.engine._execute_component(comp, "forward", None)
        out = self.engine._get_component_output(comp)
        if out is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp} produced no output.")
        # Multi-output components (flow_encoder h+spk, hift wav+source):
        # OutputExtractor stores output_1..N aliases alongside output_0 —
        # gather the full ordered tuple when present (keys are
        # comp-prefixed, so no cross-component staleness).
        extras = []
        i = 1
        while isinstance(self.resolved.get(f"{comp}.output_{i}"),
                         torch.Tensor):
            extras.append(self.resolved[f"{comp}.output_{i}"])
            i += 1
        return (out, *extras) if extras else out

    def _load_speaker_constants(self) -> Dict[str, torch.Tensor]:
        from pathlib import Path
        from safetensors import safe_open
        base = Path(self.ctx.pkg.cache_path
                    if hasattr(self.ctx.pkg, "cache_path")
                    else self.ctx.pkg.path)
        asset = base / "runtime" / "speech_constants.safetensors"
        if not asset.exists():
            raise RuntimeError(
                "ZERO FALLBACK: runtime/speech_constants.safetensors absent"
                " — rebuild with the speech contract (speaker constants).")
        out: Dict[str, torch.Tensor] = {}
        with safe_open(str(asset), framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
        return out

    # ── the leg ──

    def run(self, state: Dict[str, Any]) -> None:
        sp = self.ctx.pkg.topology.get("flow", {}).get("speech")
        if not sp:
            raise RuntimeError(
                "ZERO FALLBACK: CFM speech leg invoked without "
                "topology.flow.speech.")
        t0 = time.perf_counter()
        device = state["device"]
        dtype = state["dtype"]
        full_ids: List[int] = state["full_ids"]        # prompt + generated
        lm_hidden: torch.Tensor = state["lm_hidden"]   # [1, S, H_llm] last fwd

        comps = _require(sp, "components", "functional component slots")
        c_bb = _require(comps, "backbone", "speech components")
        c_head = _require(comps, "codec_head", "speech components")
        c_cemb = _require(comps, "codec_embedding", "speech components")
        c_temb = _require(comps, "text_embedding", "speech components")
        c_hproj = _require(comps, "hidden_projection", "speech components")
        c_fenc = _require(comps, "flow_encoder", "speech components")
        c_fdec = _require(comps, "flow_decoder", "speech components")
        c_voc = _require(comps, "vocoder", "speech components")

        tts_bos = int(_require(sp, "tts_bos_token_id", "tts span"))
        tts_eos = int(_require(sp, "tts_eos_token_id", "tts span"))
        audio_bos = int(_require(sp, "audio_bos_token_id", "prefix ids"))
        text_eos = int(_require(sp, "text_eos_token_id", "prefix ids"))
        codec_eos = int(_require(sp, "codec_eos_token_id", "codec eos"))
        min_new = int(_require(sp, "min_new_token", "AR bounds"))
        max_new = int(_require(sp, "max_new_token", "AR bounds"))
        norm_proj = bool(_require(sp, "normalize_projected_hidden",
                                  "condition"))
        samp = _require(sp, "talker_sampling", "sampling contract")
        voc = _require(sp, "vocoder", "vocoder contract")

        _seed_v = self.resolved.get("global.seed")
        if _seed_v is None:
            _seed_v = self.ctx.pkg.defaults.get("seed")
        if _seed_v is None:
            raise RuntimeError(
                "ZERO FALLBACK: no RNG seed — the build must emit "
                "generation.seed into defaults.json.")
        seed = int(_seed_v)

        # ── 1. tts span over the full sequence ──────────────────────
        _bos_pos = [i for i, t in enumerate(full_ids) if t == tts_bos]
        if not _bos_pos:
            raise RuntimeError(
                "ZERO FALLBACK: no tts-bos marker in the sequence — the "
                "audio-mode prompt assembly must append it (contract "
                "tts_bos_token_id).")
        lo = _bos_pos[-1] + 1
        hi = next((i for i in range(lo, len(full_ids))
                   if full_ids[i] == tts_eos), None)
        if hi is None:
            # vendor: an un-emitted tts-eos ends the span at the last
            # non-eos generated token.
            hi = len(full_ids)
            _eos_set = self.ctx.pkg.defaults.get("eos_token_id")
            _eos_set = set(_eos_set if isinstance(_eos_set, (list, tuple))
                           else [_eos_set])
            while hi > lo and full_ids[hi - 1] in _eos_set:
                hi -= 1
        if hi <= lo:
            raise RuntimeError(
                "ZERO FALLBACK: empty tts span — the LM generated no "
                "speakable text between the tts markers.")
        if int(lm_hidden.shape[1]) < hi:
            raise RuntimeError(
                f"ZERO FALLBACK: the final decode forward covers "
                f"{int(lm_hidden.shape[1])} positions but the tts span "
                f"ends at {hi} — hidden rows unavailable.")
        span_ids = full_ids[lo:hi]
        span_hidden = lm_hidden[0, lo:hi].to(device)        # [T, H_llm]
        print(f"   [speech_cfm] tts span [{lo}:{hi}) — {len(span_ids)} "
              f"tokens, seed={seed}")

        # ── 2. condition: additive merge + prefix ────────────────────
        _ids_t = torch.tensor([span_ids], dtype=torch.long, device=device)
        text_emb = self._run(c_temb, input=_ids_t)          # [1, T, H]
        proj = self._run(c_hproj, audio_features=span_hidden.to(dtype))
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)                        # [1, T, H]
        if norm_proj:
            proj = torch.nn.functional.normalize(proj, p=2, dim=-1)
        merged = text_emb.to(device) + proj.to(device)      # [1, T, H]
        _fix_t = torch.tensor([[text_eos, audio_bos]], dtype=torch.long,
                              device=device)
        fix_emb = self._run(c_temb, input=_fix_t)           # [1, 2, H]
        prefix = torch.cat([merged, fix_emb.to(device)], dim=1).to(dtype)
        T0 = int(prefix.shape[1])

        # ── 3. codec AR loop (shared seeded frontier) ────────────────
        stream = SeededDrawStream(seed)
        _temp = float(_require(samp, "temperature", "sampling"))
        _top_p = float(_require(samp, "top_p", "sampling"))
        _top_k = int(_require(samp, "top_k", "sampling"))
        _min_keep = int(_require(samp, "top_min_tokens_to_keep",
                                 "sampling"))
        _rep = float(_require(samp, "repetition_penalty", "sampling"))
        _rep_win = int(_require(samp, "repetition_window", "sampling"))
        codes: List[int] = []
        # Full-context re-run per step (the validated omni-leg brick,
        # speech.py:297-301): _execute_component calls are independent
        # forwards with NO KV session between them — a 1-token step would
        # see zero context (degenerate repetitive codes, adjudicated by
        # vendor token2wav replay 2026-08-02). For the last-row read the
        # growing-context forward is exactly the vendor's KV decode.
        context = prefix
        print(f"   [speech_cfm] codec AR start (min {min_new}, "
              f"max {max_new})...")
        for t in range(max_new):
            S = int(context.shape[1])
            _pos = torch.arange(S, dtype=torch.long,
                                device=device).unsqueeze(0)
            hidden = self._run(c_bb, inputs_embeds=context,
                               position_ids=_pos)
            logits = self._run(c_head, input=hidden[:, -1:].to(dtype))
            z = logits.reshape(-1).float().cpu().numpy().astype(np.float64)
            code = stream.draw_chattts(
                z, _temp, _top_p, _top_k, min_tokens_to_keep=_min_keep,
                seen_ids=codes, repetition_penalty=_rep,
                penalty_window=_rep_win,
                eos_masked=(t < min_new), eos_id=codec_eos)
            if code == codec_eos:
                break
            codes.append(code)
            _c_t = torch.tensor([[code]], dtype=torch.long, device=device)
            context = torch.cat(
                [context, self._run(c_cemb, input=_c_t).to(dtype)], dim=1)
        if not codes:
            raise RuntimeError(
                "ZERO FALLBACK: the codec AR emitted no speech codes.")
        print(f"   [speech_cfm] {len(codes)} speech codes")

        # ── 4. token2wav: constants → flow → CFM → vocoder ──────────
        spk_prefix = str(_require(sp, "speaker_constants",
                                  "speaker constants"))
        req_spk = str(self.resolved.get("global.speaker")
                      or _require(sp, "default_speaker",
                                  "speaker selection")).lower()
        consts = self._load_speaker_constants()
        declared = sorted({k.split(".")[1] for k in consts
                           if k.startswith(f"{spk_prefix}.")})
        if req_spk not in declared:
            raise RuntimeError(
                f"ZERO FALLBACK: unknown speaker '{req_spk}' — declared "
                f"speakers: {declared}.")
        _kb = f"{spk_prefix}.{req_spk}"
        xvec = consts[f"{_kb}.xvector"].to(device=device, dtype=dtype)
        ptoks = consts[f"{_kb}.prompt_tokens"].to(device=device,
                                                  dtype=torch.long)
        pmel = consts[f"{_kb}.prompt_mel"].to(device=device,
                                              dtype=torch.float32)
        n_steps = int(_require(voc, "n_timesteps", "CFM"))
        cfg_rate = float(_require(voc, "inference_cfg_rate", "CFM"))
        mel_bins = int(_require(voc, "mel_bins", "vocoder geometry"))
        sample_rate = int(_require(voc, "sample_rate", "vocoder output"))

        gen_t = torch.tensor([codes], dtype=torch.long, device=device)
        all_tok = torch.cat([ptoks, gen_t], dim=1)          # [1, Tp+Tg]
        h, spk_p = None, None
        _fe_out = self._run(c_fenc, token=all_tok, embedding=xvec)
        if isinstance(_fe_out, (tuple, list)):
            h, spk_p = _fe_out[0], _fe_out[1]
        else:
            raise RuntimeError(
                "ZERO FALLBACK: flow_encoder must return (h, spk) — "
                "re-trace the flow_encoder component.")
        h = h.float()                                       # [1, Tmel, 512→80]
        mel_len1 = int(pmel.shape[0])
        mel_len2 = int(h.shape[1]) - mel_len1
        if mel_len2 <= 0:
            raise RuntimeError(
                f"ZERO FALLBACK: encoder mel extent {int(h.shape[1])} "
                f"does not exceed the prompt mel {mel_len1}.")
        conds = torch.zeros_like(h)
        conds[:, :mel_len1] = pmel.to(conds)
        conds = conds.transpose(1, 2).contiguous()          # [1, 80, Tm]
        mu = h.transpose(1, 2).contiguous()                 # [1, 80, Tm]
        mask = torch.ones(1, 1, int(h.shape[1]),
                          device=device, dtype=torch.float32)
        # CFM Euler on the cosine grid; initial noise from the SHARED
        # frontier; CFG batch-2 with zeroed uncond mu/spks/cond
        # (vendor solve_euler verbatim, combine (1+r)*c − r*u).
        x = torch.from_numpy(
            seeded_gaussian(seed, (1, mel_bins, int(h.shape[1])))
        ).to(device=device, dtype=torch.float32)
        ts = [1.0 - math.cos(v * 0.5 * math.pi)
              for v in np.linspace(0.0, 1.0, n_steps + 1)]
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spk_p.float(),
                             torch.zeros_like(spk_p.float())], dim=0)
        cond_in = torch.cat([conds, torch.zeros_like(conds)], dim=0)
        mask_in = torch.cat([mask, mask], dim=0)
        t_cur, dt = ts[0], ts[1] - ts[0]
        for stp in range(1, n_steps + 1):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.full((2,), float(t_cur), device=device,
                              dtype=torch.float32)
            dphi = self._run(
                c_fdec, x=x_in.to(dtype), mask=mask_in.to(dtype),
                mu=mu_in.to(dtype), t=t_in.to(dtype),
                spks=spks_in.to(dtype), cond=cond_in.to(dtype))
            if isinstance(dphi, (tuple, list)):
                dphi = dphi[0]
            dphi = dphi.float()
            d_c, d_u = dphi.chunk(2, dim=0)
            guided = (1.0 + cfg_rate) * d_c - cfg_rate * d_u
            x = x + dt * guided
            t_cur = t_cur + dt
            if stp < n_steps:
                dt = ts[stp + 1] - t_cur
        mel = x[:, :, mel_len1:]                            # [1, 80, Tg*up]
        print(f"   [speech_cfm] CFM {n_steps} steps -> mel "
              f"{list(mel.shape)}")

        wav = self._run(c_voc, speech_feat=mel.to(dtype))
        if isinstance(wav, (tuple, list)):
            wav = wav[0]
        # §8 diagnostic (default-off): NBX_SPEECH_DUMP=<dir> saves the
        # leg's bisection points (span ids, speech codes, mel, wav) for
        # vendor-replay adjudication (codes → vendor token2wav decides
        # whether a defect is upstream conditioning/AR or our token2wav).
        import os as _os_sd
        _dump_dir = _os_sd.environ.get("NBX_SPEECH_DUMP")
        if _dump_dir:
            from pathlib import Path as _P_sd
            _P_sd(_dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                str(_P_sd(_dump_dir) / "speech_leg_dump.npz"),
                span_ids=np.asarray(span_ids, dtype=np.int64),
                codes=np.asarray(codes, dtype=np.int64),
                mel=mel.detach().float().cpu().numpy(),
                wav=wav.detach().reshape(-1).float().cpu().numpy(),
                speaker=np.asarray([req_spk]),
            )
            print(f"   [speech_cfm] NBX_SPEECH_DUMP -> "
                  f"{_dump_dir}/speech_leg_dump.npz")
        self.resolved["global.output_audio"] = wav.reshape(-1).float()
        self.resolved["global.audio_sample_rate"] = sample_rate
        dt_s = time.perf_counter() - t0
        print(f"   [speech_cfm] waveform {int(wav.numel())} samples "
              f"@ {sample_rate} Hz in {dt_s:.1f}s")
