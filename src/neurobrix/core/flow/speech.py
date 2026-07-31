"""Generative-speech leg (P-OMNI-GEN §1) — compiled engine.

Runs AFTER the vlm text loop when the request asks --mode audio and the
container declares topology.flow.speech (the registry-emitted contract:
R15 hidden tap, codec ids, sampling contracts, vocoder windowing). The
leg mirrors the vendor pipeline op-for-op:

  thinker per-position hiddens (embeds + tap layer, both available from
  the LAST full re-forward — causal attention makes earlier positions
  invariant to appended tokens)
    -> chatml segmentation by im_start (system skipped, user parts mix
       text_projection(word embeds) with hidden_projection(tap hiddens)
       at multimodal positions, assistant part built from tts specials
       + codec specials with the speaker id)
    -> talker outer codec AR (full re-forward tier, dual_ar precedent):
       codec_head logits -> seeded draw (talker_sampling contract)
    -> per frame, the MTP code_predictor inner AR: prefill
       [past_hidden, last_id_embed] -> head[0]; step g embeds via
       codec_embedding[g-1] -> head[g] (15 residual codes, predictor
       sampling contract)
    -> next talker embed = sum of the 16 group embeds
       (+ trailing_text_hidden[step] | tts_pad_embed)
    -> code2wav chunked decode (chunk/left_context from the contract,
       trim left_context * total_upsample per chunk)
    -> resolved["global.output_audio"]  (the CLI writes the WAV; the
       flow never writes files)

Sampling runs on the shared seeded CPU fp64 draw frontier
(kernels/seeded_draw.py) — ONE code path for both engines (R30); the
triton mirror consumes the same module.

Every quantity is read from topology.flow.speech / pkg.defaults — no
model names, no literals (ZERO FALLBACK on missing contract keys).
"""

import time
from typing import Any, Dict, List, Optional

import torch

from neurobrix.kernels.seeded_draw import SeededDrawStream


def _require(block: Dict[str, Any], key: str, where: str):
    val = block.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: topology.flow.speech is missing '{key}' "
            f"({where}) — re-emit the speech contract from the registry.")
    return val


class SpeechLeg:
    """Compiled-engine speech leg over the flow.speech contract."""

    def __init__(self, engine):
        # engine = the VLMEngine instance: we reuse its component
        # execution helpers and resolver (same ctx, same lifecycle).
        self.engine = engine
        self.ctx = engine.ctx
        self.resolved = engine.ctx.variable_resolver.resolved

    # ── component plumbing (vlm projection idiom: dual-write + run) ──

    def _run(self, comp: str, **inputs: torch.Tensor) -> torch.Tensor:
        self.engine._ensure_weights_loaded(comp)
        for name, val in inputs.items():
            self.resolved[f"{comp}.{name}"] = val
            self.resolved[f"global.{name}"] = val
        self.engine._execute_component(comp, "forward", None)
        out = self.engine._get_component_output(comp)
        if out is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp} produced no output.")
        # Multi-GPU placements put components on different devices; the
        # leg concatenates across calls, so normalize to the leg device.
        return out.to(self._device) if hasattr(out, "to") else out

    @staticmethod
    def _mrope_positions(seq_len: int, device) -> torch.Tensor:
        # Text-style positions on all three M-RoPE planes (the talker
        # consumes projected hiddens — no spatial grid exists for TTS).
        base = torch.arange(seq_len, dtype=torch.long, device=device)
        return base.view(1, 1, -1).expand(3, 1, -1).contiguous()

    # ── the leg ──

    def run(self, state: Dict[str, Any]) -> None:
        sp = self.ctx.pkg.topology.get("flow", {}).get("speech")
        if not sp:
            raise RuntimeError(
                "ZERO FALLBACK: speech leg invoked without "
                "topology.flow.speech.")
        t0 = time.perf_counter()
        device = state["hidden_tap"].device
        self._device = device
        dtype = state["hidden_tap"].dtype

        comps = _require(sp, "components", "functional component slots")
        c_backbone = _require(comps, "backbone", "speech components")
        c_head = _require(comps, "codec_head", "speech components")
        c_pred = _require(comps, "predictor", "speech components")
        c_cemb = _require(comps, "codec_embedding", "speech components")
        c_pemb = _require(comps, "predictor_embedding", "speech components")
        c_tproj = _require(comps, "text_projection", "speech components")
        c_hproj = _require(comps, "hidden_projection", "speech components")
        c_voc = _require(comps, "vocoder", "speech components")

        num_groups = int(_require(sp, "num_code_groups", "codec groups"))
        codec_eos = int(_require(sp, "codec_eos_token_id", "codec eos"))
        talker_s = _require(sp, "talker_sampling", "talker contract")
        pred_s = _require(sp, "predictor", "predictor contract")
        voc = _require(sp, "vocoder", "vocoder contract")
        speakers = _require(sp, "speakers", "speaker map")
        default_speaker = str(sp.get("default_speaker", "")).lower()

        # Request-level knobs (defaults merged by the executor).
        req_speaker = str(self.resolved.get("global.speaker")
                          or default_speaker).lower()
        if req_speaker not in {k.lower(): None for k in speakers}:
            raise RuntimeError(
                f"ZERO FALLBACK: unknown speaker '{req_speaker}' — "
                f"declared speakers: {sorted(speakers)}.")
        speaker_id = int({k.lower(): v for k, v in speakers.items()}[req_speaker])
        # Registry-driven seed (generation.seed → defaults.json at build);
        # CLI `--set global.seed` overrides per request. `is None` checks —
        # seed 0 is a legitimate value, never a missing one.
        _seed_v = self.resolved.get("global.seed")
        if _seed_v is None:
            _seed_v = self.ctx.pkg.defaults.get("seed")
        if _seed_v is None:
            raise RuntimeError(
                "ZERO FALLBACK: no RNG seed — the build must emit "
                "generation.seed into defaults.json (registry-driven); "
                "re-import a current build or pass --set global.seed.")
        seed = int(_seed_v)
        max_frames = int(self.resolved.get("global.max_audio_frames")
                         or _require(talker_s, "max_new_tokens",
                                     "talker sampling contract"))

        stream = SeededDrawStream(seed)
        print(f"   [speech] leg start: speaker={req_speaker}({speaker_id}) "
              f"seed={seed} max_frames={max_frames}")

        # ── 1. thinker-side sequences ─────────────────────────────────
        input_ids: List[int] = list(state["input_ids"])
        generated_ids: List[int] = list(state["generated_ids"])
        sequences = input_ids + generated_ids
        thinker_embed = state["context_embeds"].to(device)   # [1, S0+n-1, 2048]
        thinker_hidden = state["hidden_tap"].to(device)      # [1, S0+n-1, 2048]

        lm_cfg = self.ctx.pkg.defaults.get("lm_config", {})
        mm_ids = {int(lm_cfg.get(k)) for k in
                  ("audio_token_id", "image_token_id", "video_token_id")
                  if lm_cfg.get(k) is not None}
        im_start = int(_require(sp, "im_start_token_id", "chatml"))
        system_tok = int(_require(sp, "system_token_id", "chatml"))
        user_tok = int(_require(sp, "user_token_id", "chatml"))
        assistant_tok = int(_require(sp, "assistant_token_id", "chatml"))
        tts_ids = [int(_require(sp, k, "tts specials")) for k in
                   ("tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id")]

        # ── 2. projections + special embeds ──────────────────────────
        # tts specials: thinker word embeds -> text_projection (vendor).
        thinker_embed_w = self.engine._get_embed_weight(state["lm_name"])
        if thinker_embed_w is None:
            raise RuntimeError("ZERO FALLBACK: thinker embed weight not found.")
        tts_embeds = thinker_embed_w[
            torch.tensor(tts_ids, device=thinker_embed_w.device)] \
            .to(device=device, dtype=dtype).unsqueeze(0)                       # [1, 3, 2048]
        # The thinker LM (the placement giant) is finished once its embed
        # rows are extracted — the tap/context activations are plain
        # tensors, independent of the weights. Release it BEFORE loading
        # the talker branch so the two never coexist in VRAM.
        if not self.ctx.persistent_mode:
            from neurobrix.core.memory.manager import release_flow_memory
            self.engine._unload_component_weights(state["lm_name"])
            release_flow_memory(self.ctx.primary_device)

        tts_proj = self._run(c_tproj, hidden_state=tts_embeds)
        tts_bos_embed = tts_proj[:, 0:1]
        tts_eos_embed = tts_proj[:, 1:2]
        tts_pad_embed = tts_proj[:, 2:3]

        # ── 3. chatml segmentation (vendor lines 3975-4035) ──────────
        im_starts = [i for i, t in enumerate(input_ids) if t == im_start]
        im_starts.append(len(sequences))
        mm_mask_full = torch.tensor(
            [1 if t in mm_ids else 0 for t in sequences],
            dtype=torch.bool, device=device).unsqueeze(0)
        S_avail = thinker_embed.shape[1]

        talker_embeds: List[torch.Tensor] = []
        talker_ids: List[int] = []
        trailing_text_hidden: Optional[torch.Tensor] = None
        for i in range(len(im_starts) - 1):
            a, b = im_starts[i], im_starts[i + 1]
            role = sequences[a + 1]
            if role == system_tok:
                continue
            if role == user_tok:
                b_c = min(b, S_avail)
                seg_mask = mm_mask_full[:, a:b_c]
                seg_embed = thinker_embed[:, a:b_c]
                seg_hidden = thinker_hidden[:, a:b_c]
                part = torch.empty(
                    (1, b_c - a, tts_pad_embed.shape[-1]),
                    device=device, dtype=dtype)
                if bool(seg_mask.any()):
                    mm_h = self._run(c_hproj,
                                     hidden_state=seg_hidden[seg_mask]
                                     .unsqueeze(0))
                    part[seg_mask] = mm_h.squeeze(0).to(dtype)
                txt_e = self._run(c_tproj,
                                  hidden_state=seg_embed[~seg_mask]
                                  .unsqueeze(0))
                part[~seg_mask] = txt_e.squeeze(0).to(dtype)
                talker_embeds.append(part)
                talker_ids.extend(sequences[a:b_c])
            elif role == assistant_tok and i == len(im_starts) - 2:
                b_c = min(b, S_avail)
                assistant_hidden = self._run(
                    c_tproj,
                    hidden_state=thinker_embed[:, a:b_c])
                lay = _require(sp, "assistant_layout", "assistant layout")
                _kp = int(_require(lay, "keep_prefix", "assistant layout"))
                _pc = int(_require(lay, "pad_count", "assistant layout"))
                _fti = int(_require(lay, "first_text_index", "assistant layout"))
                _tf = int(_require(lay, "trailing_from", "assistant layout"))
                _zp = int(_require(lay, "codec_zeros_prefix", "assistant layout"))
                codec_specials = [
                    speaker_id if name == "SPEAKER"
                    else int(_require(sp, name, "codec specials"))
                    for name in _require(lay, "codec_specials_order",
                                         "assistant layout")]
                codec_emb = self._run(
                    c_cemb,
                    input=torch.tensor([codec_specials], dtype=torch.long,
                                       device=device)) \
                    .to(device=device, dtype=dtype)                            # [1, 6, 1024]
                h = tts_pad_embed.shape[-1]
                assistant_text_hidden = torch.cat(
                    (assistant_hidden[:, :_kp],
                     tts_pad_embed.expand(-1, _pc, -1),
                     tts_bos_embed,
                     assistant_hidden[:, _fti:_fti + 1]), dim=1)
                assistant_codec_hidden = torch.cat(
                    (torch.zeros((1, _zp, h), device=device, dtype=dtype),
                     codec_emb), dim=1)
                trailing_text_hidden = torch.cat(
                    (assistant_hidden[:, _tf:], tts_eos_embed), dim=1)
                talker_embeds.append(
                    assistant_text_hidden + assistant_codec_hidden)
                talker_ids.extend([tts_ids[2]] *
                                  assistant_text_hidden.shape[1])
            # history assistant parts: vendor skips them

        if trailing_text_hidden is None:
            raise RuntimeError(
                "ZERO FALLBACK: no assistant segment found in the chatml "
                "sequence — speech needs a chat-templated prompt "
                "(global.chat_mode).")
        talker_context = torch.cat(talker_embeds, dim=1).to(dtype)

        # ── 4. talker outer AR + predictor MTP inner AR ──────────────
        # MTP componentization (supervisor pattern 2026-07-30): every codec
        # embedding lookup and the per-step head projection are GRAPH work
        # (codec_embedding / predictor_embedding lookup components + the
        # mono-step predictor's in-graph head_index gather). The flow owns
        # only the loops, the context concats and the seeded draws (WHEN).
        # Declared-MoE late fusion for the talker backbone (the same
        # set_moe_config path the thinker uses): the traced expert unroll
        # is superseded wholesale in all modes; routing params come from
        # the flow.speech.talker_moe contract.
        talker_moe = sp.get("talker_moe") or {}
        if int(talker_moe.get("num_experts") or 0) > 1:
            _texec = self.ctx.executors.get(c_backbone)
            if _texec is not None and hasattr(_texec, "set_moe_config"):
                _texec.set_moe_config(norm_topk_prob=bool(
                    _require(talker_moe, "norm_topk_prob",
                             "talker MoE contract")))

        # Every sampling key is _required — a silent temperature default
        # (1.0 = unscaled multinomial) is the Ming sampling-class trap.
        t_temp = float(_require(talker_s, "temperature", "talker sampling"))
        t_topk = int(_require(talker_s, "top_k", "talker sampling"))
        t_topp = float(_require(talker_s, "top_p", "talker sampling"))
        t_pen = float(_require(talker_s, "repetition_penalty",
                               "talker sampling"))
        p_temp = float(_require(pred_s, "temperature", "predictor sampling"))
        p_topk = int(_require(pred_s, "top_k", "predictor sampling"))
        p_topp = float(_require(pred_s, "top_p", "predictor sampling"))

        frames: List[List[int]] = []
        seen_codes: List[int] = list(talker_ids)
        gen_step = 0
        while gen_step < max_frames:
            S = talker_context.shape[1]
            hidden = self._run(
                c_backbone,
                inputs_embeds=talker_context,
                position_ids=self._mrope_positions(S, device))
            last_hidden = hidden[:, -1:, :]              # [1, 1, 1024]
            logits = self._run(c_head, input=last_hidden)
            z = logits.reshape(-1).to(torch.float64).cpu().numpy()
            c0 = stream.draw(z, temperature=t_temp, top_k=t_topk,
                             top_p=t_topp, seen_ids=seen_codes,
                             repetition_penalty=t_pen)
            if c0 == codec_eos:
                break
            seen_codes.append(c0)

            # predictor MTP: prefill [past_hidden, last_id_embed]
            last_id_embed = self._run(
                c_cemb,
                input=torch.tensor([[c0]], dtype=torch.long, device=device)) \
                .to(device=device, dtype=dtype)                  # [1, 1, 1024]
            pred_embeds = torch.cat((last_hidden.to(dtype), last_id_embed),
                                    dim=1)
            frame = [c0]
            group_embeds = [last_id_embed]
            for g in range(num_groups - 1):
                P = pred_embeds.shape[1]
                pos = torch.arange(P, dtype=torch.long, device=device) \
                    .unsqueeze(0)
                # Mono-step predictor: backbone + in-graph head_index
                # gather -> logits (vendor lm_head[g] applied on every
                # position; the last-position slice below is the AR read).
                pred_logits = self._run(
                    c_pred,
                    inputs_embeds=pred_embeds, position_ids=pos,
                    head_index=torch.tensor([g], dtype=torch.long,
                                            device=device))
                # VENDOR PARITY: logits arrive in the model compute dtype;
                # the fp64 conversion below is the seeded-draw frontier
                # boundary, not a compute upcast. The triton leg mirrors
                # this frontier by construction (same graph, same draw —
                # P-OMNI-GEN §1). (No hardcoded fp32 — DtypeEngine doctrine.)
                pl = pred_logits[:, -1, :]
                zg = pl.reshape(-1).to(torch.float64).cpu().numpy()
                cg = stream.draw(zg, temperature=p_temp, top_k=p_topk,
                                 top_p=p_topp)
                frame.append(cg)
                step_embed = self._run(
                    c_pemb,
                    input=torch.tensor([[cg]], dtype=torch.long,
                                       device=device),
                    table_index=torch.tensor([g], dtype=torch.long,
                                             device=device)) \
                    .to(device=device, dtype=dtype)              # [1, 1, 1024]
                group_embeds.append(step_embed)
                pred_embeds = torch.cat((pred_embeds, step_embed), dim=1)
            frames.append(frame)

            # next talker input: sum of the 16 group embeds (+ text inj)
            next_embed = torch.cat(group_embeds, dim=1).sum(1, keepdim=True)
            if gen_step < trailing_text_hidden.shape[1]:
                next_embed = next_embed + \
                    trailing_text_hidden[:, gen_step].unsqueeze(1)
            else:
                next_embed = next_embed + tts_pad_embed
            talker_context = torch.cat(
                (talker_context, next_embed.to(dtype)), dim=1)
            gen_step += 1
            if gen_step % 25 == 0:
                print(f"   [speech] {gen_step} codec frames "
                      f"({stream.draws} draws)")

        if not frames:
            raise RuntimeError(
                "ZERO FALLBACK: talker produced zero codec frames "
                "(eos at step 0) — inspect the prefill construction.")
        print(f"   [speech] talker done: {len(frames)} frames, "
              f"{stream.draws} draws")

        # ── 5. code2wav chunked decode (vendor 3752) ─────────────────
        chunk = int(_require(voc, "chunk_size", "vocoder windowing"))
        left_ctx = int(_require(voc, "left_context_size", "vocoder windowing"))
        rates = list(_require(voc, "upsample_rates", "vocoder"))
        ratios = list(_require(voc, "upsampling_ratios", "vocoder"))
        total_upsample = 1
        for r in rates + ratios:
            total_upsample *= int(r)

        codes = torch.tensor(frames, dtype=torch.long, device=device) \
            .transpose(0, 1).unsqueeze(0)                # [1, 16, T]
        wavs: List[torch.Tensor] = []
        start = 0
        T = codes.shape[-1]
        while start < T:
            end = min(start + chunk, T)
            ctx_sz = left_ctx if start - left_ctx > 0 else start
            piece = codes[..., start - ctx_sz:end]
            wav_chunk = self._run(c_voc, codes=piece)
            wavs.append(wav_chunk[..., ctx_sz * total_upsample:])
            start = end
        waveform = torch.cat(wavs, dim=-1).float()

        self.resolved["global.output_audio"] = waveform
        dt = time.perf_counter() - t0
        sr = int(_require(voc, "sample_rate", "vocoder contract"))
        print(f"   [speech] waveform {list(waveform.shape)} "
              f"({waveform.shape[-1] / sr:.2f}s @ {sr} Hz) in {dt:.1f}s")
