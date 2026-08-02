"""Generative-speech leg, CFM class (P-OMNI-GEN model 3/3) — triton
engines (R33-pure).

Mirror of core/flow/speech_cfm.py over the SAME flow.speech contract and
the SAME component graphs — NBXTensor end-to-end, zero torch (R33). The
engines share BOTH seeded frontiers (kernels/seeded_draw.py): the codec
draws (draw_chattts) and the CFM initial noise (seeded_gaussian) come
from one CPU fp64/np source, so codes and latents are byte-coupled
across engines; only the per-op numeric field differs (the Ming
iterative-process closure class: per-engine byte-determinism, never
cross-engine byte equality).

Structure is the compiled leg's, section for section:

  1. tts span over full_ids (python ints — engine-agnostic)
  2. condition: additive merge emb_text(ids) + L2norm(projector(hidden))
     — wrappers + NBXTensor operators, no torch
  3. codec AR, full-context re-run per step (the validated omni brick:
     component forwards hold no KV session between calls)
  4. token2wav: per-speaker constants (package safetensors, numpy
     boundary) → flow_encoder → cosine-grid CFM Euler with batch-2 CFG
     (vendor combine (1+r)·c − r·u) → mel slice → vocoder
  5. waveform to resolved["global.output_audio"] (NBX fp32; the
     tts_llm/omni triton store idiom)

Every quantity comes from topology.flow.speech / pkg.defaults — no
model names, no literals (ZERO FALLBACK on missing contract keys).
"""

import math
import time
from typing import Any, Dict, List

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from neurobrix.kernels import wrappers as w
from neurobrix.kernels.seeded_draw import SeededDrawStream, seeded_gaussian
from neurobrix.triton.device_transfer import needs_move, transfer_tensor
from neurobrix.triton.flow.speech import _require, _from_np_on


class SpeechCFMLeg:
    """Triton-engine CFM speech leg over the flow.speech contract."""

    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.ctx
        self.resolved = engine.ctx.variable_resolver.resolved
        self._dev: int = 0   # bound to lm_hidden's device in run()

    # ── component plumbing (dual-write + run, omni-leg idiom) ──

    def _run(self, comp: str, **inputs):
        self.engine._ensure_weights_loaded(comp)
        for name, val in inputs.items():
            self.resolved[f"{comp}.{name}"] = val
            self.resolved[f"global.{name}"] = val
        self.engine._execute_component(comp, "forward", None)
        out = self.engine._get_component_output(comp)
        if out is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp} produced no output.")
        if needs_move(out, self._dev):
            out = transfer_tensor(out, self._dev)
        # Multi-output components (flow_encoder h+spk, hift wav+source):
        # OutputExtractor stores output_1..N aliases alongside output_0.
        extras = []
        i = 1
        while isinstance(self.resolved.get(f"{comp}.output_{i}"), NBXTensor):
            e = self.resolved[f"{comp}.output_{i}"]
            if needs_move(e, self._dev):
                e = transfer_tensor(e, self._dev)
            extras.append(e)
            i += 1
        return (out, *extras) if extras else out

    def _load_speaker_constants_np(self) -> Dict[str, np.ndarray]:
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
        out: Dict[str, np.ndarray] = {}
        with safe_open(str(asset), framework="np") as f:
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
        full_ids: List[int] = state["full_ids"]
        lm_hidden: NBXTensor = state["lm_hidden"]      # [1, S, H_llm]
        dtype = lm_hidden.dtype
        self._dev = int(lm_hidden._device_idx)

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
        span_hidden = lm_hidden[0, lo:hi].contiguous()      # [T, H_llm]
        print(f"   [speech_cfm:triton] tts span [{lo}:{hi}) — "
              f"{len(span_ids)} tokens, seed={seed}")

        # ── 2. condition: additive merge + prefix ────────────────────
        _ids_t = _from_np_on(
            np.asarray([span_ids], dtype=np.int64), self._dev)
        text_emb = self._run(c_temb, input=_ids_t)          # [1, T, H]
        proj = self._run(c_hproj,
                         audio_features=span_hidden.to(dtype))
        if len(proj.shape) == 2:
            proj = proj.reshape((1,) + tuple(proj.shape))   # [1, T, H]
        if norm_proj:
            # F.normalize(p=2, dim=-1) mirror: x / max(||x||2, eps).
            pf = proj.to(NBXDtype.float32)
            nrm = w.sqrt_wrapper(
                w.sum_wrapper(pf * pf, dim=-1, keepdim=True))
            nrm = w.clamp_min_wrapper(nrm, 1e-12)
            proj = pf / nrm
        merged = text_emb.to(NBXDtype.float32) \
            + proj.to(NBXDtype.float32)                     # [1, T, H]
        _fix_t = _from_np_on(
            np.asarray([[text_eos, audio_bos]], dtype=np.int64), self._dev)
        fix_emb = self._run(c_temb, input=_fix_t)           # [1, 2, H]
        prefix = NBXTensor.cat(
            [merged.to(dtype), fix_emb.to(dtype)], dim=1)

        # ── 3. codec AR loop (shared seeded frontier) ────────────────
        # Full-context re-run per step — the validated omni-leg brick
        # (component forwards hold no KV session between calls).
        stream = SeededDrawStream(seed)
        _temp = float(_require(samp, "temperature", "sampling"))
        _top_p = float(_require(samp, "top_p", "sampling"))
        _top_k = int(_require(samp, "top_k", "sampling"))
        _min_keep = int(_require(samp, "top_min_tokens_to_keep",
                                 "sampling"))
        _rep = float(_require(samp, "repetition_penalty", "sampling"))
        _rep_win = int(_require(samp, "repetition_window", "sampling"))
        codes: List[int] = []
        context = prefix
        print(f"   [speech_cfm:triton] codec AR start (min {min_new}, "
              f"max {max_new})...")
        for t in range(max_new):
            S = int(context.shape[1])
            _pos = _from_np_on(
                np.arange(S, dtype=np.int64).reshape(1, -1), self._dev)
            hidden = self._run(c_bb, inputs_embeds=context,
                               position_ids=_pos)
            last = hidden[:, -1:].contiguous().to(dtype)
            logits = self._run(c_head, input=last)
            # bf16-safe boundary: exact into fp64 via fp32 (omni idiom).
            z = logits.to(NBXDtype.float32).numpy() \
                .reshape(-1).astype(np.float64)
            code = stream.draw_chattts(
                z, _temp, _top_p, _top_k, min_tokens_to_keep=_min_keep,
                seen_ids=codes, repetition_penalty=_rep,
                penalty_window=_rep_win,
                eos_masked=(t < min_new), eos_id=codec_eos)
            if code == codec_eos:
                break
            codes.append(code)
            _c_t = _from_np_on(
                np.asarray([[code]], dtype=np.int64), self._dev)
            context = NBXTensor.cat(
                [context, self._run(c_cemb, input=_c_t).to(dtype)], dim=1)
        if not codes:
            raise RuntimeError(
                "ZERO FALLBACK: the codec AR emitted no speech codes.")
        print(f"   [speech_cfm:triton] {len(codes)} speech codes")

        # ── 4. token2wav: constants → flow → CFM → vocoder ──────────
        spk_prefix = str(_require(sp, "speaker_constants",
                                  "speaker constants"))
        req_spk = str(self.resolved.get("global.speaker")
                      or _require(sp, "default_speaker",
                                  "speaker selection")).lower()
        consts = self._load_speaker_constants_np()
        declared = sorted({k.split(".")[1] for k in consts
                           if k.startswith(f"{spk_prefix}.")})
        if req_spk not in declared:
            raise RuntimeError(
                f"ZERO FALLBACK: unknown speaker '{req_spk}' — declared "
                f"speakers: {declared}.")
        _kb = f"{spk_prefix}.{req_spk}"
        xvec = _from_np_on(
            consts[f"{_kb}.xvector"].astype(np.float32), self._dev) \
            .to(dtype)
        ptoks_np = consts[f"{_kb}.prompt_tokens"].astype(np.int64)
        pmel_np = consts[f"{_kb}.prompt_mel"].astype(np.float32)
        n_steps = int(_require(voc, "n_timesteps", "CFM"))
        cfg_rate = float(_require(voc, "inference_cfg_rate", "CFM"))
        mel_bins = int(_require(voc, "mel_bins", "vocoder geometry"))
        sample_rate = int(_require(voc, "sample_rate", "vocoder output"))

        all_tok = _from_np_on(
            np.concatenate(
                [ptoks_np, np.asarray([codes], dtype=np.int64)], axis=1),
            self._dev)                                      # [1, Tp+Tg]
        _fe_out = self._run(c_fenc, token=all_tok, embedding=xvec)
        if isinstance(_fe_out, (tuple, list)):
            h, spk_p = _fe_out[0], _fe_out[1]
        else:
            raise RuntimeError(
                "ZERO FALLBACK: flow_encoder must return (h, spk) — "
                "re-trace the flow_encoder component.")
        h = h.to(NBXDtype.float32)                          # [1, Tm, 80]
        mel_len1 = int(pmel_np.shape[0])
        mel_len2 = int(h.shape[1]) - mel_len1
        if mel_len2 <= 0:
            raise RuntimeError(
                f"ZERO FALLBACK: encoder mel extent {int(h.shape[1])} "
                f"does not exceed the prompt mel {mel_len1}.")
        # conds assembly is host-constant work (prompt mel + zeros) —
        # numpy boundary, uploaded once (the Ming constants idiom).
        _Tm = int(h.shape[1])
        conds_np = np.zeros((1, _Tm, int(h.shape[2])), dtype=np.float32)
        conds_np[:, :mel_len1] = pmel_np
        conds = _from_np_on(
            np.ascontiguousarray(conds_np.transpose(0, 2, 1)), self._dev)
        mu = h.transpose(1, 2).contiguous()                 # [1, 80, Tm]
        mask = _from_np_on(
            np.ones((1, 1, _Tm), dtype=np.float32), self._dev)
        # CFM Euler on the cosine grid; initial noise from the SHARED
        # frontier; CFG batch-2 with zeroed uncond mu/spks/cond
        # (vendor solve_euler verbatim, combine (1+r)*c − r*u).
        x = _from_np_on(
            seeded_gaussian(seed, (1, mel_bins, _Tm)).astype(np.float32),
            self._dev)
        ts = [1.0 - math.cos(v * 0.5 * math.pi)
              for v in np.linspace(0.0, 1.0, n_steps + 1)]
        spk_f = spk_p.to(NBXDtype.float32)
        mu_in = NBXTensor.cat(
            [mu, NBXTensor.zeros(tuple(mu.shape), dtype=mu.dtype)], dim=0)
        spks_in = NBXTensor.cat(
            [spk_f, NBXTensor.zeros(tuple(spk_f.shape),
                                    dtype=spk_f.dtype)], dim=0)
        cond_in = NBXTensor.cat(
            [conds, NBXTensor.zeros(tuple(conds.shape),
                                    dtype=conds.dtype)], dim=0)
        mask_in = NBXTensor.cat([mask, mask], dim=0)
        t_cur, dt = ts[0], ts[1] - ts[0]
        for stp in range(1, n_steps + 1):
            x_in = NBXTensor.cat([x, x], dim=0)
            t_in = _from_np_on(
                np.full((2,), float(t_cur), dtype=np.float32), self._dev)
            dphi = self._run(
                c_fdec, x=x_in.to(dtype), mask=mask_in.to(dtype),
                mu=mu_in.to(dtype), t=t_in.to(dtype),
                spks=spks_in.to(dtype), cond=cond_in.to(dtype))
            if isinstance(dphi, (tuple, list)):
                dphi = dphi[0]
            dphi = dphi.to(NBXDtype.float32)
            d_c = dphi[0:1].contiguous()
            d_u = dphi[1:2].contiguous()
            guided = (1.0 + cfg_rate) * d_c - cfg_rate * d_u
            x = x + dt * guided
            t_cur = t_cur + dt
            if stp < n_steps:
                dt = ts[stp + 1] - t_cur
        mel = x[:, :, mel_len1:].contiguous()               # [1, 80, Tg*up]
        print(f"   [speech_cfm:triton] CFM {n_steps} steps -> mel "
              f"{list(mel.shape)}")

        wav = self._run(c_voc, speech_feat=mel.to(dtype))
        if isinstance(wav, (tuple, list)):
            wav = wav[0]
        waveform = wav.reshape((-1,)).to(NBXDtype.float32)
        # §8 diagnostic (default-off): NBX_SPEECH_DUMP=<dir> — the same
        # bisection dump as the compiled leg (vendor-replay adjudication).
        import os as _os_sd
        _dump_dir = _os_sd.environ.get("NBX_SPEECH_DUMP")
        if _dump_dir:
            from pathlib import Path as _P_sd
            _P_sd(_dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                str(_P_sd(_dump_dir) / "speech_leg_dump.npz"),
                span_ids=np.asarray(span_ids, dtype=np.int64),
                codes=np.asarray(codes, dtype=np.int64),
                mel=mel.to(NBXDtype.float32).numpy(),
                wav=waveform.numpy(),
                speaker=np.asarray([req_spk]),
            )
            print(f"   [speech_cfm:triton] NBX_SPEECH_DUMP -> "
                  f"{_dump_dir}/speech_leg_dump.npz")
        self.resolved["global.output_audio"] = waveform
        self.resolved["global.audio_sample_rate"] = sample_rate
        dt_s = time.perf_counter() - t0
        print(f"   [speech_cfm:triton] waveform {int(waveform.shape[0])} "
              f"samples @ {sample_rate} Hz in {dt_s:.1f}s")
