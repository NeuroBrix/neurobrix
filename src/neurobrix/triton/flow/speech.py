"""Generative-speech leg (P-OMNI-GEN §1) — triton engines (R33-pure).

Mirror of core/flow/speech.py over the SAME flow.speech contract and the
SAME component graphs — NBXTensor end-to-end, zero torch (R33). The two
engines share ONE seeded CPU fp64 draw frontier
(kernels/seeded_draw.py): identical seeds, identical draws, identical
codec frames whenever probabilities agree (R30 by construction).

Structure is the compiled leg's, section for section:

  1. thinker-side sequences (python ints — engine-agnostic)
  2. tts-special projections (embedding gather + text_projection run)
  3. chatml segmentation; the user-part modal/text splice is rebuilt
     as index_select over the mm/text position lists + position-order
     reassembly (ONE NBXTensor.cat) — the compiled leg's boolean
     masked-scatter has no NBX form and needs none: the mask is python
     data derived from token ids.
  4. talker outer AR + mono-step MTP inner AR — component invocations
     only (head/table selection is IN-GRAPH via head_index/table_index
     inputs; the flow owns loops, concats and draws).
  5. code2wav chunked decode (contract windowing), waveform to
     resolved["global.output_audio"] (tts_llm triton precedent).

Every quantity comes from topology.flow.speech / pkg.defaults — no
model names, no literals (ZERO FALLBACK on missing contract keys).
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.kernels import wrappers as w
from neurobrix.kernels.seeded_draw import SeededDrawStream
from neurobrix.triton.device_transfer import needs_move, transfer_tensor


def _require(block: Dict[str, Any], key: str, where: str):
    val = block.get(key)
    if val is None:
        raise RuntimeError(
            f"ZERO FALLBACK: topology.flow.speech is missing '{key}' "
            f"({where}) — re-emit the speech contract from the registry.")
    return val


def _from_np_on(arr: np.ndarray, dev_idx: int) -> NBXTensor:
    """Upload a host array to a SPECIFIC device.

    from_numpy allocates on the allocator's CURRENT device; under a
    multi-GPU placement the leg's components sit on different devices,
    so every host-created tensor is pinned explicitly (a kernel cannot
    read a pointer from another device — the 'cannot be accessed from
    Triton' class)."""
    prev = DeviceAllocator.get_device()
    try:
        DeviceAllocator.set_device(dev_idx)
        return NBXTensor.from_numpy(arr)
    finally:
        DeviceAllocator.set_device(prev)


class SpeechLeg:
    """Triton-engine speech leg over the flow.speech contract."""

    def __init__(self, engine):
        # engine = the triton VLMEngine instance: same ctx, same
        # component plumbing, same lifecycle (compiled-leg idiom).
        self.engine = engine
        self.ctx = engine.ctx
        self.resolved = engine.ctx.variable_resolver.resolved
        self._dev: int = 0   # bound to the tap's device in run()

    # ── component plumbing (dual-write + run, compiled-leg idiom) ──

    def _run(self, comp: str, **inputs) -> NBXTensor:
        self.engine._ensure_weights_loaded(comp)
        for name, val in inputs.items():
            self.resolved[f"{comp}.{name}"] = val
            self.resolved[f"global.{name}"] = val
        self.engine._execute_component(comp, "forward", None)
        out = self.engine._get_component_output(comp)
        if out is None:
            raise RuntimeError(f"ZERO FALLBACK: {comp} produced no output.")
        # Multi-GPU placements put components on different devices; the
        # leg concatenates across calls, so normalize to the leg device
        # (compiled-leg idiom; D2D via the shared transfer brick).
        if needs_move(out, self._dev):
            out = transfer_tensor(out, self._dev)
        return out

    def _ids(self, ids, shape=None) -> NBXTensor:
        arr = np.asarray(ids, dtype=np.int64)
        if shape is not None:
            arr = arr.reshape(shape)
        return _from_np_on(arr, self._dev)

    def _mrope_positions(self, seq_len: int) -> NBXTensor:
        # Text-style positions on all three M-RoPE planes (the talker
        # consumes projected hiddens — no spatial grid exists for TTS).
        base = np.arange(seq_len, dtype=np.int64)
        return _from_np_on(
            np.broadcast_to(base, (3, 1, seq_len)).copy(), self._dev)

    # ── the leg ──

    def run(self, state: Dict[str, Any]) -> None:
        sp = self.ctx.pkg.topology.get("flow", {}).get("speech")
        if not sp:
            raise RuntimeError(
                "ZERO FALLBACK: speech leg invoked without "
                "topology.flow.speech.")
        t0 = time.perf_counter()
        dtype = state["hidden_tap"].dtype
        # Leg device = the tap's device; every cross-call operand is
        # normalized onto it (multi-GPU placement discipline).
        self._dev = int(state["hidden_tap"]._device_idx)

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
        print(f"   [speech:triton] leg start: speaker={req_speaker}"
              f"({speaker_id}) seed={seed} max_frames={max_frames}")

        # ── 1. thinker-side sequences ─────────────────────────────────
        input_ids: List[int] = list(state["input_ids"])
        generated_ids: List[int] = list(state["generated_ids"])
        sequences = input_ids + generated_ids
        thinker_embed: NBXTensor = state["context_embeds"]  # [1, S0+n-1, 2048]
        thinker_hidden: NBXTensor = state["hidden_tap"]     # [1, S0+n-1, 2048]
        if needs_move(thinker_embed, self._dev):
            thinker_embed = transfer_tensor(thinker_embed, self._dev)
        if needs_move(thinker_hidden, self._dev):
            thinker_hidden = transfer_tensor(thinker_hidden, self._dev)

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
        _w_dev = int(thinker_embed_w._device_idx)
        tts_embeds = w.embedding(
            thinker_embed_w,
            _from_np_on(np.asarray([tts_ids], dtype=np.int64), _w_dev))
        if needs_move(tts_embeds, self._dev):
            tts_embeds = transfer_tensor(tts_embeds, self._dev)
        tts_embeds = tts_embeds.to(dtype)                    # [1, 3, 2048]
        # The thinker LM (the placement giant) is finished once its embed
        # rows are extracted — release it BEFORE loading the talker branch
        # so the two never coexist in VRAM (compiled-leg idiom).
        if not self.ctx.persistent_mode:
            from neurobrix.triton.memory_pool import release_flow_memory
            self.engine._unload_component_weights(state["lm_name"])
            release_flow_memory(self.ctx.primary_device)

        tts_proj = self._run(c_tproj, hidden_state=tts_embeds)
        tts_bos_embed = tts_proj[:, 0:1]
        tts_eos_embed = tts_proj[:, 1:2]
        tts_pad_embed = tts_proj[:, 2:3]

        # ── 3. chatml segmentation (vendor lines 3975-4035) ──────────
        im_starts = [i for i, t in enumerate(input_ids) if t == im_start]
        im_starts.append(len(sequences))
        S_avail = thinker_embed.shape[1]

        talker_embeds: List[NBXTensor] = []
        talker_ids: List[int] = []
        trailing_text_hidden: Optional[NBXTensor] = None
        for i in range(len(im_starts) - 1):
            a, b = im_starts[i], im_starts[i + 1]
            role = sequences[a + 1]
            if role == system_tok:
                continue
            if role == user_tok:
                b_c = min(b, S_avail)
                # The modal/text splice: the mask is PYTHON data (token
                # ids), so gather each class with index_select, project,
                # then reassemble in position order with one cat — the
                # NBX form of the compiled leg's masked scatter.
                mm_pos = [p for p in range(a, b_c) if sequences[p] in mm_ids]
                txt_pos = [p for p in range(a, b_c)
                           if sequences[p] not in mm_ids]
                pieces: Dict[int, NBXTensor] = {}
                if mm_pos:
                    seg_h = w.index_select_wrapper(
                        thinker_hidden, 1, self._ids(mm_pos))
                    mm_h = self._run(c_hproj, hidden_state=seg_h).to(dtype)
                    for k, p in enumerate(mm_pos):
                        pieces[p] = mm_h[:, k:k + 1]
                if txt_pos:
                    seg_e = w.index_select_wrapper(
                        thinker_embed, 1, self._ids(txt_pos))
                    txt_e = self._run(c_tproj, hidden_state=seg_e).to(dtype)
                    for k, p in enumerate(txt_pos):
                        pieces[p] = txt_e[:, k:k + 1]
                part = NBXTensor.cat(
                    [pieces[p] for p in range(a, b_c)], dim=1)
                talker_embeds.append(part)
                talker_ids.extend(sequences[a:b_c])
            elif role == assistant_tok and i == len(im_starts) - 2:
                b_c = min(b, S_avail)
                assistant_hidden = self._run(
                    c_tproj,
                    hidden_state=thinker_embed[:, a:b_c]).to(dtype)
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
                    input=self._ids(codec_specials,
                                    (1, len(codec_specials)))).to(dtype)       # [1, 6, 1024]
                h = tts_pad_embed.shape[-1]
                assistant_text_hidden = NBXTensor.cat(
                    ([assistant_hidden[:, :_kp]]
                     + [tts_pad_embed] * _pc
                     + [tts_bos_embed,
                        assistant_hidden[:, _fti:_fti + 1]]), dim=1)
                assistant_codec_hidden = NBXTensor.cat(
                    [NBXTensor.zeros((1, _zp, h), dtype=dtype,
                                     device=f"cuda:{self._dev}"),
                     codec_emb], dim=1)
                trailing_text_hidden = NBXTensor.cat(
                    [assistant_hidden[:, _tf:], tts_eos_embed], dim=1)
                talker_embeds.append(
                    w.add(assistant_text_hidden, assistant_codec_hidden))
                talker_ids.extend([tts_ids[2]]
                                  * assistant_text_hidden.shape[1])
            # history assistant parts: vendor skips them

        if trailing_text_hidden is None:
            raise RuntimeError(
                "ZERO FALLBACK: no assistant segment found in the chatml "
                "sequence — speech needs a chat-templated prompt "
                "(global.chat_mode).")
        talker_context = NBXTensor.cat(talker_embeds, dim=1).to(dtype)

        # ── 4. talker outer AR + predictor MTP inner AR ──────────────
        # MTP componentization (supervisor pattern): every codec embedding
        # lookup and the per-step head projection are GRAPH work — the
        # flow owns only the loops, the context concats and the seeded
        # draws (WHEN). Declared-MoE late fusion for the talker backbone,
        # same set_moe_config path as the thinker.
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
                position_ids=self._mrope_positions(S))
            last_hidden = hidden[:, -1:, :]              # [1, 1, 1024]
            logits = self._run(c_head, input=last_hidden)
            # bf16 has no numpy dtype (raw void bytes) — cast to fp32
            # FIRST (exact into fp64: the shared-draw contract holds;
            # next_token_diffusion precedent). fp32 short-circuits.
            z = logits.to(NBXDtype.float32).numpy() \
                .reshape(-1).astype(np.float64)
            c0 = stream.draw(z, temperature=t_temp, top_k=t_topk,
                             top_p=t_topp, seen_ids=seen_codes,
                             repetition_penalty=t_pen)
            if c0 == codec_eos:
                break
            seen_codes.append(c0)

            # predictor MTP: prefill [past_hidden, last_id_embed]
            last_id_embed = self._run(
                c_cemb, input=self._ids([[c0]])).to(dtype)    # [1, 1, 1024]
            pred_embeds = NBXTensor.cat(
                [last_hidden.to(dtype), last_id_embed], dim=1)
            frame = [c0]
            group_embeds = [last_id_embed]
            for g in range(num_groups - 1):
                P = pred_embeds.shape[1]
                pos = _from_np_on(
                    np.arange(P, dtype=np.int64)[None], self._dev)
                # Mono-step predictor: backbone + in-graph head_index
                # gather -> logits (vendor lm_head[g] on every position;
                # the last-position slice below is the AR read).
                pred_logits = self._run(
                    c_pred,
                    inputs_embeds=pred_embeds, position_ids=pos,
                    head_index=self._ids([g]))
                # VENDOR PARITY: logits arrive in the model compute dtype;
                # the fp64 conversion below is the seeded-draw frontier
                # boundary shared with the compiled leg (same module,
                # same stream — R30 by construction).
                pl = pred_logits[:, -1, :]
                # Same bf16-safe boundary as the talker draw above.
                zg = pl.to(NBXDtype.float32).numpy() \
                    .reshape(-1).astype(np.float64)
                cg = stream.draw(zg, temperature=p_temp, top_k=p_topk,
                                 top_p=p_topp)
                frame.append(cg)
                step_embed = self._run(
                    c_pemb,
                    input=self._ids([[cg]]),
                    table_index=self._ids([g])).to(dtype)     # [1, 1, 1024]
                group_embeds.append(step_embed)
                pred_embeds = NBXTensor.cat([pred_embeds, step_embed], dim=1)
            frames.append(frame)

            # next talker input: sum of the 16 group embeds (+ text inj)
            next_embed = w.sum_wrapper(
                NBXTensor.cat(group_embeds, dim=1), dim=1, keepdim=True)
            if gen_step < trailing_text_hidden.shape[1]:
                next_embed = w.add(
                    next_embed,
                    trailing_text_hidden[:, gen_step:gen_step + 1])
            else:
                next_embed = w.add(next_embed, tts_pad_embed)
            talker_context = NBXTensor.cat(
                [talker_context, next_embed.to(dtype)], dim=1)
            gen_step += 1
            if gen_step % 25 == 0:
                print(f"   [speech:triton] {gen_step} codec frames "
                      f"({stream.draws} draws)")

        if not frames:
            raise RuntimeError(
                "ZERO FALLBACK: talker produced zero codec frames "
                "(eos at step 0) — inspect the prefill construction.")
        print(f"   [speech:triton] talker done: {len(frames)} frames, "
              f"{stream.draws} draws")

        # ── 5. code2wav chunked decode (vendor 3752) ─────────────────
        chunk = int(_require(voc, "chunk_size", "vocoder windowing"))
        left_ctx = int(_require(voc, "left_context_size", "vocoder windowing"))
        rates = list(_require(voc, "upsample_rates", "vocoder"))
        ratios = list(_require(voc, "upsampling_ratios", "vocoder"))
        total_upsample = 1
        for r in rates + ratios:
            total_upsample *= int(r)

        codes = _from_np_on(
            np.asarray(frames, dtype=np.int64).T[None].copy(),
            self._dev)                                   # [1, 16, T]
        wavs: List[NBXTensor] = []
        start = 0
        T = codes.shape[-1]
        while start < T:
            end = min(start + chunk, T)
            ctx_sz = left_ctx if start - left_ctx > 0 else start
            # Contiguous-guard (POINT 6 H2 class): a mid-T window slice
            # of [1, 16, T] is non-contiguous on every multi-chunk
            # decode; flat-indexed consumers need materialized storage.
            # Full-width slices short-circuit at zero cost.
            piece = codes[:, :, start - ctx_sz:end].contiguous()
            wav_chunk = self._run(c_voc, codes=piece)
            wavs.append(wav_chunk[:, :, ctx_sz * total_upsample:])
            start = end
        waveform = (NBXTensor.cat(wavs, dim=2)
                    if len(wavs) > 1 else wavs[0]).to(NBXDtype.float32)

        self.resolved["global.output_audio"] = waveform
        dt = time.perf_counter() - t0
        sr = int(_require(voc, "sample_rate", "vocoder contract"))
        print(f"   [speech:triton] waveform {list(waveform.shape)} "
              f"({waveform.shape[-1] / sr:.2f}s @ {sr} Hz) in {dt:.1f}s")
