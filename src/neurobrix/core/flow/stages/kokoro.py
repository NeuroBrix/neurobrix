"""Kokoro-82M native stage handlers.

Kokoro components use pack_padded_sequence / cuDNN LSTM ops that can't run
through CompiledSequence, so they execute natively with extracted weights.

All functions take the AudioEngine instance (``engine``) as first parameter
so they can access ``engine.ctx``, executors, weight loaders, etc.
"""

# =======================================================================
#  WARNING — TEMPORARY VIOLATIONS OF THE NEUROBRIX CONTRACT
# =======================================================================
#
#  This file exists TEMPORARILY and violates two structural NeuroBrix
#  engine principles:
#
#  1)  TENSORDAG BYPASS: native PyTorch is executed inline here
#      (``torch.nn.LSTM``, ``torch.nn.functional.conv1d``,
#      pack_padded_sequence, etc.) instead of going through the
#      compiled TensorDAG. NeuroBrix is a universal inference engine:
#      *every* graph must be expressed in ATen + custom ops and
#      executed by CompiledSequence / TritonSequence, never in pure
#      Python.
#
#  2)  EXTERNAL RUNTIME DEPENDENCIES: ``preprocess_phonemizer_input``
#      imports ``kokoro``, ``phonemizer`` and invokes ``espeak-ng`` via
#      ``subprocess``. No inference engine may depend on system
#      binaries or third-party libraries at inference time — the
#      contract is zero external dependency.
#
#  RESOLUTION (forge side, not runtime):
#    - Trace LSTMs and ``pack_padded_sequence`` into the graph (emit
#      ``aten::lstm`` / ``aten::rnn_relu`` / ``aten::pack_padded_sequence``
#      in the DAG, with associated Triton kernels if needed).
#    - Integrate text -> phoneme conversion into the ``.nbx`` module
#      ``modules/tokenizer/`` (embedded lookup table or small
#      traceable network). Remove all ``import kokoro / phonemizer``
#      and ``subprocess`` calls from this file.
#
#  This file will be removed once the forge re-tracing is done.
# =======================================================================

import gc
from neurobrix.core.device_utils import device_empty_cache
import time
import torch
from typing import Dict, List, Optional


def _coerce_torch_dtype(dt) -> torch.dtype:
    """Accept either torch.dtype (native engine) or string (Triton engine).

    Triton engine returns dtype as a string to keep torch out of triton/.
    Stage handlers here are torch-boundary by design, so we coerce here.
    """
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dt, torch.float16)
    return torch.float16


# ─────────────────────────────────────────────────────────────
# Public entry points (called from AudioEngine.execute)
# ─────────────────────────────────────────────────────────────

def execute_native_kokoro(engine, stage: Dict, audio_config: Dict) -> None:
    """Native (hand-rolled) execution for Kokoro's LSTM components.

    TRITON-ONLY band-aid as of 2026-05-30. The COMPILED path no longer uses this:
    Forge now traces the LSTM forward correctly (parent_module weight-resolution
    fix for the cuDNN flatten_parameters data_ptr aliasing), so the compiled
    registry runs the predictor + text_encoder via `execution: forward` (the
    traced graph), validated element-wise vs the vendor oracle. The COMPILED
    dispatch branch for `native_kokoro` was removed (core/flow/audio.py).

    This function is retained for the TRITON path only (triton/flow/audio.py),
    which cannot yet run the `aten::lstm` forward graph (no Triton LSTM kernel) —
    a separate chantier (P-KOKORO-TRITON-LSTM-KERNEL, R33). Removing it now would
    break the untested triton path; the removal is deliberately scoped to compiled.

    Handles:
    - text_encoder: embedding -> WeightNorm Conv1d x 3 -> BiLSTM (pack_padded_sequence)
    - predictor: BiLSTM + duration + F0/N + alignment
    """
    comp_name = stage["component"]

    # Sub-dispatch by topology field, NOT by component name string
    native_subtype = stage.get("native_subtype", "predictor")
    if native_subtype == "text_encoder":
        _execute_native_text_encoder(engine, comp_name)
        return

    device = engine.ctx.primary_device
    dtype = _coerce_torch_dtype(engine._get_compute_dtype())

    print(f"   [{comp_name}] Running native Kokoro predictor...")
    start = time.perf_counter()

    # Get inputs from previous stages (DATA-DRIVEN from topology inputs_from)
    inputs_from = stage.get("inputs_from", [])
    if len(inputs_from) < 2:
        raise RuntimeError(
            f"ZERO FALLBACK: native_kokoro predictor requires 'inputs_from' "
            f"with 2 component names in topology stage config."
        )
    bert_enc_out = engine._get_component_output(inputs_from[0])
    if bert_enc_out is None:
        raise RuntimeError(f"ZERO FALLBACK: {inputs_from[0]} output not found.")
    d_en = bert_enc_out.transpose(-1, -2).to(device=device, dtype=dtype)  # [B, 512, T]

    text_enc_out = engine._get_component_output(inputs_from[1])
    if text_enc_out is None:
        raise RuntimeError(f"ZERO FALLBACK: {inputs_from[1]} output not found.")
    t_en = text_enc_out.to(device=device, dtype=dtype)  # [B, 512, T]

    style_pred = engine.ctx.variable_resolver.resolved.get("global.predictor_style")
    style_dec = engine.ctx.variable_resolver.resolved.get("global.decoder_style")
    if style_pred is None or style_dec is None:
        raise RuntimeError("ZERO FALLBACK: Kokoro predictor requires voicepack styles.")

    style_pred = style_pred.to(device=device, dtype=dtype)  # [B, 128]
    style_dec = style_dec.to(device=device, dtype=dtype)

    text_mask = engine.ctx.variable_resolver.resolved.get("global.text_mask")
    text_lengths = engine.ctx.variable_resolver.resolved.get("global.text_lengths")

    engine._ensure_weights_loaded(comp_name)
    w = dict(engine.ctx.executors[comp_name]._weights)

    target_shapes = _get_kokoro_decoder_shapes(engine)
    target_asr_frames = target_shapes["asr_frames"]
    target_f0_len = target_shapes["f0_len"]
    target_n_len = target_shapes["n_len"]

    with torch.inference_mode():
        # -- Step 1: DurationEncoder (text_encoder.lstms) --
        # Vendor: predictor.text_encoder(d_en, s, input_lengths, text_mask)
        T = d_en.shape[2]
        x = d_en.permute(2, 0, 1)  # [T, B, 512]
        s_exp = style_pred.unsqueeze(0).expand(T, -1, -1)  # [T, B, 128]
        x = torch.cat([x, s_exp], dim=-1)  # [T, B, 640]
        if text_mask is not None:
            x.masked_fill_(text_mask.unsqueeze(-1).transpose(0, 1).to(device), 0.0)
        x = x.transpose(0, 1)  # [B, T, 640]
        x = x.transpose(-1, -2)  # [B, 640, T]

        for layer_idx in range(6):
            prefix = f"text_encoder.lstms.{layer_idx}"
            lstm_key = f"{prefix}.weight_ih_l0"
            if lstm_key in w:
                # LSTM layer
                x = _run_kokoro_single_lstm(
                    engine, x, w, prefix, text_lengths, T, device, dtype
                )
            elif f"{prefix}.proj.weight" in w:
                # AdaLayerNorm layer
                x = _run_kokoro_adaln(
                    x, style_pred, w, prefix, device, dtype
                )
                # Re-concat style
                s_ch = s_exp.permute(1, 2, 0)  # [B, 128, T]
                if x.shape[2] < s_ch.shape[2]:
                    s_ch = s_ch[:, :, :x.shape[2]]
                elif x.shape[2] > s_ch.shape[2]:
                    x = x[:, :, :s_ch.shape[2]]
                x = torch.cat([x, s_ch], dim=1)  # [B, 640, T]
                if text_mask is not None:
                    x.masked_fill_(text_mask.unsqueeze(1).to(device), 0.0)

        d = x.transpose(-1, -2)  # [B, T, 640] -- DurationEncoder output

        # -- Step 2: Duration LSTM + projection --
        # Vendor: x, _ = self.predictor.lstm(d)
        dur_lstm = _build_bilstm(w, "lstm", 640, 256, device, dtype)
        d_dur, _ = dur_lstm(d)  # [B, T, 512]

        dur_w = w["dur_proj.linear_layer.weight"].to(device=device, dtype=dtype)
        dur_b = w["dur_proj.linear_layer.bias"].to(device=device, dtype=dtype)
        dur_logits = d_dur @ dur_w.T + dur_b  # [B, T, 50]

        speed = engine.ctx.pkg.defaults.get("speed", 1.0)
        raw_durations = torch.sigmoid(dur_logits).sum(dim=-1) / speed  # [B, T]

        if text_mask is not None:
            inv_mask = (~text_mask).float().to(device=device)
            raw_durations = raw_durations * inv_mask

        durations = _scale_kokoro_durations(raw_durations[0], target_asr_frames)

        # -- Step 3: Build alignment matrix --
        num_phonemes = durations.shape[0]
        alignment = torch.zeros(1, num_phonemes, target_asr_frames, device=device, dtype=dtype)
        pos = 0
        for i in range(num_phonemes):
            dur_val = int(durations[i].item())
            if dur_val > 0 and pos < target_asr_frames:
                end = min(pos + dur_val, target_asr_frames)
                alignment[0, i, pos:end] = 1.0
                pos = end

        # Real content occupies frames [0, pos); the alignment loop left the
        # tail [pos, target) zero, so asr/en are zero there too. The decoder
        # still synthesises all target_asr_frames, so record the content
        # fraction (generic, semantic-free key) for the post-decode waveform
        # crop in the audio flow handler. Stretching short prompts to fill the
        # whole window was the P0a babbling bug — see _scale_kokoro_durations.
        content_ratio = (float(pos) / float(target_asr_frames)
                         if target_asr_frames > 0 else 1.0)
        engine.ctx.variable_resolver.resolved["global.audio_content_ratio"] = content_ratio

        # -- Step 4: Compute en and asr --
        # Vendor: en = d.transpose(-1, -2) @ pred_aln_trg
        en = d.transpose(-1, -2) @ alignment  # [B, 640, T_frames]
        # Vendor: asr = t_en @ pred_aln_trg
        asr = t_en @ alignment  # [B, 512, T_frames]

        # -- Step 5: F0 and N prediction (F0Ntrain) --
        # Vendor: self.shared(en.transpose(-1, -2)) -> F0/N blocks
        shared_lstm = _build_bilstm(w, "shared", 640, 256, device, dtype)
        shared_out, _ = shared_lstm(en.transpose(-1, -2))  # [B, T_frames, 512]
        shared_out = shared_out.transpose(-1, -2)  # [B, 512, T_frames]

        F0_raw = _run_kokoro_f0n_blocks(shared_out, style_pred, w, "F0", device, dtype)
        N_raw = _run_kokoro_f0n_blocks(shared_out, style_pred, w, "N", device, dtype)

        # Expand F0/N to decoder target shapes, interpolating only the spoken
        # content prefix and zero-padding the tail (content_ratio == 1.0 when
        # the prompt fills the whole traced window).
        F0_curve = _expand_kokoro_curve(F0_raw, target_f0_len, content_ratio)
        N_curve = _expand_kokoro_curve(N_raw, target_n_len, content_ratio)

    elapsed = (time.perf_counter() - start) * 1000
    print(f"   [{comp_name}] Native done in {elapsed:.0f}ms  "
          f"asr={list(asr.shape)} F0={list(F0_curve.shape)} N={list(N_curve.shape)}  "
          f"content={pos}/{target_asr_frames} (ratio={content_ratio:.3f}) "
          f"F0_raw={list(F0_raw.shape)}")

    # Bind all decoder inputs to variable resolver
    for key in ["global.asr", "asr", f"{comp_name}.asr"]:
        engine.ctx.variable_resolver.resolved[key] = asr
    for key in ["global.F0_curve", "F0_curve"]:
        engine.ctx.variable_resolver.resolved[key] = F0_curve
    for key in ["global.N", "N"]:
        engine.ctx.variable_resolver.resolved[key] = N_curve
    for key in ["global.decoder_style", "decoder_style", "s", "global.s"]:
        engine.ctx.variable_resolver.resolved[key] = style_dec

    if not engine.ctx.persistent_mode:
        engine._unload_component_weights(comp_name)
        gc.collect()
        device_empty_cache(engine.ctx.primary_device)


def preprocess_phonemizer_input(engine, prompt: str, phoneme_vocab: Dict) -> None:
    """Convert text to phoneme IDs using espeak-ng + vocabulary mapping.

    Used by models like Kokoro that take IPA phoneme sequences instead
    of standard text tokens.
    """
    device = engine.ctx.primary_device

    # Step 1: text -> IPA via the NeuroBrix-internal g2p (ZO-3) — reads the
    # espeak-distilled lexicon embedded in the .nbx (modules/g2p/en_lexicon.txt.gz)
    # + a stdlib LTS fallback. NO `kokoro`/`phonemizer`/`espeak-ng` import at
    # runtime (R34); the embedded lexicon retains espeak's license.
    _lang_map = {"a": "en-us", "b": "en-gb"}
    klang = engine.ctx.pkg.defaults.get("phoneme_lang", "a")
    from neurobrix.core.module.audio.g2p import g2p_phonemes
    phonemes = g2p_phonemes(prompt, engine.ctx.nbx_path_str,
                            _lang_map.get(klang, "en-us"), klang)

    # Step 2: Map phonemes to IDs
    ids = [0]  # BOS/padding
    for ch in phonemes:
        if ch in phoneme_vocab:
            ids.append(phoneme_vocab[ch])
    ids.append(0)  # EOS/padding

    # Step 3: Feed the ACTUAL phoneme sequence — no padding to the trace seq_len.
    # The bert/text_encoder/predictor/decoder graphs carry a symbolic seq_len that
    # the runtime binds from the input_ids shape, so every stage runs at the true
    # length and the audio duration tracks the text. The previous code padded AND
    # truncated every utterance to the trace's 23-phoneme length, which froze the
    # output to a fixed duration (identical bytes for any prompt) and silently cut
    # any text longer than 23 phonemes.
    actual_len = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    engine.ctx.variable_resolver.resolved["global.input_ids"] = input_ids
    engine.ctx.variable_resolver.resolved["input_ids"] = input_ids
    print(f"   [Phonemizer] '{prompt[:60]}' -> {len(phonemes)} phonemes "
          f"-> {actual_len} IDs (dynamic seq_len)")

    # Bind text length and mask for downstream stages (text_encoder, predictor)
    text_lengths = torch.tensor([actual_len], dtype=torch.long, device=device)
    for key in ["global.text_lengths", "text_lengths", "input_lengths"]:
        engine.ctx.variable_resolver.resolved[key] = text_lengths

    # batch=1, no padding → every position is a real token. The mask convention is
    # True = PADDING (vendor uses gt(arange+1, input_lengths)); with no padding the
    # mask is all-False. (Consumers: masked_fill_(text_mask, 0) in the
    # DurationEncoder / text_encoder, inv_mask = ~text_mask for durations.)
    text_mask = torch.zeros(1, actual_len, dtype=torch.bool, device=device)
    for key in ["global.text_mask", "text_mask", "m"]:
        engine.ctx.variable_resolver.resolved[key] = text_mask

    # Load voicepack for TTS models with voice packs
    _load_voicepack(engine, actual_len, device)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _load_voicepack(engine, phoneme_count: int, device) -> None:
    """Load voice pack and split into predictor/decoder styles.

    Voicepacks are [N, 256] tensors stored in modules/voices/.
    Index by phoneme count, then split: [:128]=decoder, [128:]=predictor.
    """
    from pathlib import Path

    nbx_path = Path(engine.ctx.nbx_path_str)
    voices_dir = nbx_path / "modules" / "voices"

    if not voices_dir.exists():
        return

    voice_name = engine.ctx.pkg.defaults.get("voice", "af_heart")
    voice_path = voices_dir / f"{voice_name}.pt"

    if not voice_path.exists():
        voice_files = sorted(voices_dir.glob("*.pt"))
        if not voice_files:
            return
        voice_path = voice_files[0]
        voice_name = voice_path.stem

    voicepack = torch.load(voice_path, map_location=device, weights_only=True)

    if voicepack.dim() == 1:
        ref_s = voicepack.unsqueeze(0)
    elif voicepack.dim() == 2:
        idx = min(phoneme_count, voicepack.shape[0] - 1)
        ref_s = voicepack[idx:idx + 1]
    elif voicepack.dim() == 3:
        idx = min(phoneme_count, voicepack.shape[0] - 1)
        ref_s = voicepack[idx]
    else:
        vp_dim = voicepack.shape[-1]
        ref_s = voicepack.reshape(-1, vp_dim)[0:1]

    # Split voicepack: first half = decoder style, second half = predictor style
    # Dimension comes from the voicepack tensor shape itself (DATA-DRIVEN)
    split_at = ref_s.shape[-1] // 2
    style_dec = ref_s[:, :split_at]
    style_pred = ref_s[:, split_at:]

    for key in ["global.decoder_style", "decoder_style"]:
        engine.ctx.variable_resolver.resolved[key] = style_dec
    for key in ["global.predictor_style", "predictor_style"]:
        engine.ctx.variable_resolver.resolved[key] = style_pred

    print(f"   [Voicepack] Loaded '{voice_name}' (ref_s={ref_s.shape})")


def _execute_native_text_encoder(engine, comp_name: str) -> None:
    """Native execution for text_encoder: embedding -> CNN -> BiLSTM.

    cuDNN RNN internal ops can't be replayed through CompiledSequence
    (pack_padded_sequence/aten::set/cuDNN weight buffer ops).
    Extract weights and run natively using torch.nn modules.

    TODO: Fix CompiledSequence to handle aten::lstm or retrace without
    pack_padded_sequence so this can run as a compiled forward pass.
    """
    device = engine.ctx.primary_device
    dtype = _coerce_torch_dtype(engine._get_compute_dtype())

    print(f"   [{comp_name}] Running native text encoder...")
    start = time.perf_counter()

    # Get inputs
    input_ids = engine.ctx.variable_resolver.resolved.get("global.input_ids")
    input_lengths = engine.ctx.variable_resolver.resolved.get("global.text_lengths")
    text_mask = engine.ctx.variable_resolver.resolved.get("global.text_mask")

    if input_ids is None:
        raise RuntimeError("ZERO FALLBACK: input_ids not bound for text_encoder")
    if input_lengths is None:
        raise RuntimeError("ZERO FALLBACK: text_lengths not bound for text_encoder")

    # Extract weights
    engine._ensure_weights_loaded(comp_name)
    w = dict(engine.ctx.executors[comp_name]._weights)

    with torch.inference_mode():
        # Embedding
        embed_w = w["embed.weight"].to(device=device, dtype=dtype)
        x = torch.nn.functional.embedding(input_ids.to(device), embed_w)
        # x: [B, seq, 512]
        x = x.transpose(1, 2)  # [B, 512, seq]

        # Mask padding after embedding (vendor TextEncoder: x.masked_fill_(m, 0)).
        # text_mask is True=padding; m broadcasts over the channel dim.
        m = text_mask.unsqueeze(1).to(device) if text_mask is not None else None
        if m is not None:
            x = x.masked_fill(m, 0.0)

        # 3x vendor block: WeightNorm Conv1d -> LayerNorm -> LeakyReLU(0.2),
        # then mask padding. Op order and slope must match vendor exactly
        # (was conv -> LeakyReLU(0.01) -> LayerNorm, no per-block mask).
        for i in range(3):
            # WeightNorm: weight = g * v / ||v|| (dim=0, norm over in/kernel)
            wg = w[f"cnn.{i}.0.weight_g"].to(device=device, dtype=dtype)
            wv = w[f"cnn.{i}.0.weight_v"].to(device=device, dtype=dtype)
            bias = w[f"cnn.{i}.0.bias"].to(device=device, dtype=dtype)
            norm = wv.norm(dim=(1, 2), keepdim=True)
            conv_w = wg * wv / (norm + 1e-12)
            x = torch.nn.functional.conv1d(x, conv_w, bias, padding=2)
            # LayerNorm over the channel dim (vendor custom LayerNorm on [B,C,T])
            gamma = w[f"cnn.{i}.1.gamma"].to(device=device, dtype=dtype)
            beta = w[f"cnn.{i}.1.beta"].to(device=device, dtype=dtype)
            x_t = x.transpose(1, 2)  # [B, seq, C]
            x_t = torch.nn.functional.layer_norm(x_t, [gamma.shape[0]], gamma, beta)
            x = x_t.transpose(1, 2)  # [B, 512, seq]
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
            if m is not None:
                x = x.masked_fill(m, 0.0)

        # BiLSTM with pack_padded_sequence
        x = x.transpose(1, 2)  # [B, seq, 512]

        # Build nn.LSTM and load weights
        lstm = torch.nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=1, batch_first=True, bidirectional=True
        ).to(device=device, dtype=dtype)

        lstm.weight_ih_l0.data.copy_(w["lstm.weight_ih_l0"].to(device=device, dtype=dtype))
        lstm.weight_hh_l0.data.copy_(w["lstm.weight_hh_l0"].to(device=device, dtype=dtype))
        lstm.bias_ih_l0.data.copy_(w["lstm.bias_ih_l0"].to(device=device, dtype=dtype))
        lstm.bias_hh_l0.data.copy_(w["lstm.bias_hh_l0"].to(device=device, dtype=dtype))
        lstm.weight_ih_l0_reverse.data.copy_(w["lstm.weight_ih_l0_reverse"].to(device=device, dtype=dtype))
        lstm.weight_hh_l0_reverse.data.copy_(w["lstm.weight_hh_l0_reverse"].to(device=device, dtype=dtype))
        lstm.bias_ih_l0_reverse.data.copy_(w["lstm.bias_ih_l0_reverse"].to(device=device, dtype=dtype))
        lstm.bias_hh_l0_reverse.data.copy_(w["lstm.bias_hh_l0_reverse"].to(device=device, dtype=dtype))

        lengths_cpu = input_lengths.cpu().to(torch.int64)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu.clamp(min=1), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        # output: [B, T', 512] where T' = actual length

        # Transpose to [B, 512, T']
        output = output.transpose(1, 2)

        # Re-pad to full mask length (vendor: x_pad[:, :, :x.shape[-1]] = x)
        mask_len = text_mask.shape[-1] if text_mask is not None else output.shape[2]
        if output.shape[2] < mask_len:
            x_pad = torch.zeros(output.shape[0], output.shape[1], mask_len,
                                device=device, dtype=dtype)
            x_pad[:, :, :output.shape[2]] = output
            output = x_pad

        # Apply mask (vendor: x.masked_fill_(m, 0.0))
        if text_mask is not None:
            output.masked_fill_(text_mask.unsqueeze(1).to(device), 0.0)

    elapsed = (time.perf_counter() - start) * 1000
    print(f"   [{comp_name}] Native done in {elapsed:.0f}ms  output={output.shape}")

    # Store output for predictor stage
    engine.ctx.variable_resolver.resolved[f"{comp_name}.output_0"] = output


def _build_bilstm(
    w: Dict, prefix: str, input_size: int, hidden_size: int,
    device, dtype,
) -> torch.nn.LSTM:
    """Build a bidirectional LSTM from extracted weights."""
    lstm = torch.nn.LSTM(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=1, bidirectional=True, batch_first=True,
    )
    with torch.no_grad():
        for pname in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0",
                      "weight_ih_l0_reverse", "weight_hh_l0_reverse",
                      "bias_ih_l0_reverse", "bias_hh_l0_reverse"]:
            getattr(lstm, pname).copy_(w[f"{prefix}.{pname}"].to(device=device, dtype=dtype))
    lstm.eval()
    return lstm.to(device=device, dtype=dtype)


def _run_kokoro_single_lstm(
    engine, x: torch.Tensor, w: Dict, prefix: str,
    text_lengths: Optional[torch.Tensor], max_len: int, device, dtype,
) -> torch.Tensor:
    """Run one BiLSTM layer of the DurationEncoder with pack/pad."""
    wih = w[f"{prefix}.weight_ih_l0"]
    input_size = wih.shape[1]
    hidden_size = w[f"{prefix}.weight_hh_l0"].shape[1]

    lstm = _build_bilstm(w, prefix, input_size, hidden_size, device, dtype)

    x_in = x.transpose(-1, -2)  # [B, C, T] -> [B, T, C]
    if text_lengths is not None:
        lengths_cpu = text_lengths.cpu().to(torch.int64).clamp(min=1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_in, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        out, _ = lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    else:
        out, _ = lstm(x_in)
    # out: [B, T', 2*hidden] -> transpose -> [B, 2*hidden, T']
    out = out.transpose(-1, -2)
    # Pad to original mask length
    if out.shape[2] < max_len:
        pad = torch.zeros(out.shape[0], out.shape[1], max_len, device=device, dtype=dtype)
        pad[:, :, :out.shape[2]] = out
        out = pad
    return out  # [B, 2*hidden, T]


def _run_kokoro_adaln(
    x: torch.Tensor, style: torch.Tensor, w: Dict,
    prefix: str, device, dtype,
) -> torch.Tensor:
    """Run AdaLayerNorm: FC(style) -> gamma/beta -> LayerNorm -> affine."""
    channels = x.shape[1]
    fc_w = w[f"{prefix}.proj.weight"].to(device=device, dtype=dtype)
    fc_b = w[f"{prefix}.proj.bias"].to(device=device, dtype=dtype)

    # x: [B, C, T], style: [B, 128]
    h = style @ fc_w.T + fc_b  # [B, 2*C]
    h = h.unsqueeze(2)  # [B, 2*C, 1]
    gamma, beta = h.chunk(2, dim=1)  # each [B, C, 1]

    # LayerNorm over channel dim
    x_ln = x.permute(0, 2, 1)  # [B, T, C]
    x_ln = torch.nn.functional.layer_norm(x_ln, (channels,))
    x_ln = x_ln.permute(0, 2, 1)  # [B, C, T]

    return (1 + gamma) * x_ln + beta  # [B, C, T]


def _run_kokoro_f0n_blocks(
    d: torch.Tensor, style: torch.Tensor,
    weights: Dict, block_name: str, device, dtype,
) -> torch.Tensor:
    """Run F0 or N prediction: 3 AdainResBlk1d blocks + Conv1d projection.

    Returns [1, 1, T] per-phoneme prediction values.
    """
    x = d.clone()

    for block_idx in range(3):
        x = _run_kokoro_adain_resblock(
            x, style, weights, f"{block_name}.{block_idx}", device, dtype,
        )

    proj_w = weights[f"{block_name}_proj.weight"].to(device=device, dtype=dtype)
    proj_b = weights[f"{block_name}_proj.bias"].to(device=device, dtype=dtype)
    return torch.nn.functional.conv1d(x, proj_w, proj_b)  # [1, 1, T]


def _run_kokoro_adain_resblock(
    x: torch.Tensor, style: torch.Tensor,
    weights: Dict, prefix: str, device, dtype,
) -> torch.Tensor:
    """Run AdainResBlk1d matching vendor istftnet.py exactly.

    Architecture (vendor):
      residual: AdaIN->LeakyReLU->pool->Conv1->AdaIN->LeakyReLU->Conv2
      shortcut: upsample->conv1x1 (if dim_in != dim_out)
      output:   (residual + shortcut) * rsqrt(2)

    Detects upsample and learned_sc from weight presence.
    """
    def get_w(name):
        return weights[f"{prefix}.{name}"].to(device=device, dtype=dtype)

    def has_w(name):
        return f"{prefix}.{name}" in weights

    def weight_norm_conv(wg, wv, bias, h, stride=1, padding=None):
        norm = wv.norm(dim=list(range(1, wv.dim())), keepdim=True).clamp(min=1e-12)
        w = wg * wv / norm
        if padding is None:
            padding = (w.shape[2] - 1) // 2
        return torch.nn.functional.conv1d(h, w, bias, stride=stride, padding=padding)

    def weight_norm_conv_transpose(wg, wv, bias, h):
        norm = wv.norm(dim=list(range(1, wv.dim())), keepdim=True).clamp(min=1e-12)
        w = wg * wv / norm
        groups = w.shape[0]
        return torch.nn.functional.conv_transpose1d(
            h, w, bias, stride=2, padding=1, output_padding=1, groups=groups
        )

    def adain(h, norm_proj_w, norm_proj_b):
        h_norm = torch.nn.functional.instance_norm(h)
        proj = style @ norm_proj_w.T + norm_proj_b  # [B, 2*C]
        gamma, beta = proj.chunk(2, dim=-1)
        return (1 + gamma.unsqueeze(-1)) * h_norm + beta.unsqueeze(-1)

    has_upsample = has_w("pool.weight_g")
    has_learned_sc = has_w("conv1x1.weight_g")

    # -- Residual path --
    h = adain(x, get_w("norm1.proj.weight"), get_w("norm1.proj.bias"))
    h = torch.nn.functional.leaky_relu(h, 0.2)
    if has_upsample:
        h = weight_norm_conv_transpose(
            get_w("pool.weight_g"), get_w("pool.weight_v"), get_w("pool.bias"), h
        )
    h = weight_norm_conv(get_w("conv1.weight_g"), get_w("conv1.weight_v"), get_w("conv1.bias"), h)
    h = adain(h, get_w("norm2.proj.weight"), get_w("norm2.proj.bias"))
    h = torch.nn.functional.leaky_relu(h, 0.2)
    h = weight_norm_conv(get_w("conv2.weight_g"), get_w("conv2.weight_v"), get_w("conv2.bias"), h)

    # -- Shortcut path --
    sc = x
    if has_upsample:
        sc = torch.nn.functional.interpolate(sc, scale_factor=2, mode='nearest')
    if has_learned_sc:
        sc = weight_norm_conv(
            get_w("conv1x1.weight_g"), get_w("conv1x1.weight_v"), None, sc, padding=0
        )

    return (h + sc) * torch.rsqrt(torch.tensor(2.0, device=device, dtype=dtype))


def _expand_kokoro_curve(
    raw: torch.Tensor, target_len: int, content_ratio: float,
) -> torch.Tensor:
    """Interpolate a per-frame F0/N curve to the decoder target length.

    When ``content_ratio < 1.0`` the alignment matrix left the asr tail zero,
    so only the leading ``content_ratio`` fraction of ``raw`` carries real
    prosody. Interpolate that prefix to ``round(content_ratio * target_len)``
    and zero-pad the rest, keeping the iSTFTNet harmonic source silent past the
    spoken content (the post-decode crop removes the silent tail). The fraction
    is length-agnostic, so it holds whether the F0/N blocks upsample ``raw`` or
    not. ``raw`` is ``[B, 1, L]``; returns ``[B, target_len]``.
    """
    if content_ratio >= 1.0:
        return torch.nn.functional.interpolate(
            raw, size=target_len, mode="linear", align_corners=False
        ).squeeze(1)

    raw_len = raw.shape[-1]
    content_raw = max(1, round(content_ratio * raw_len))
    content_target = max(1, round(content_ratio * target_len))
    interp = torch.nn.functional.interpolate(
        raw[:, :, :content_raw], size=content_target, mode="linear", align_corners=False
    ).squeeze(1)  # [B, content_target]

    out = torch.zeros(raw.shape[0], target_len, device=raw.device, dtype=raw.dtype)
    out[:, :content_target] = interp
    return out


def _scale_kokoro_durations(raw: torch.Tensor, target: int) -> torch.Tensor:
    """Map predicted durations to integer per-phoneme frame counts.

    When the natural rounded durations already fit within ``target`` decoder
    frames, they are returned unchanged: the alignment loop leaves the unused
    tail frames zero and ``execute_native_kokoro`` crops the waveform after the
    decoder. Stretching short prompts to fill the trace-time fixed ``target``
    window was the P0a "babbling / hey hey hey" bug (each phoneme elongated
    3-5x beyond natural speech). Only when the natural sum overflows ``target``
    (prompt longer than the traced decoder window) are durations compressed to
    fit — a long-prompt fallback; the proper fix is the build-side dynamic asr
    frame count (follow-up P-BUILD-KOKORO-DYNAMIC-FRAMES).
    """
    durations = torch.round(raw).clamp(min=0)
    active = durations > 0
    if active.sum() == 0:
        result = torch.zeros_like(durations, dtype=torch.long)
        result[0] = target
        return result

    natural_sum = int(durations[active].sum().item())
    if natural_sum <= target:
        return durations.long()

    # Overflow path: compress the natural durations to exactly target frames.
    scale = target / natural_sum
    durations[active] = torch.round(durations[active] * scale).clamp(min=1)

    result = durations.long()
    diff = int(target - result.sum().item())

    active_indices = torch.where(active)[0]
    for i in range(abs(diff)):
        idx = int(active_indices[i % len(active_indices)].item())
        result[idx] += 1 if diff > 0 else -1

    return result.clamp(min=0)


def _get_kokoro_decoder_shapes(engine) -> Dict[str, int]:
    """Read decoder input shapes from graph for exact target dimensions."""
    executor = engine.ctx.executors.get("decoder")
    if executor is None:
        raise RuntimeError(
            "ZERO FALLBACK: 'decoder' component not found in executors.\n"
            "Required to determine target shapes for predictor output."
        )
    dag = getattr(executor, '_dag', None)
    if dag is None:
        raise RuntimeError(
            "ZERO FALLBACK: 'decoder' component has no DAG.\n"
            "Cannot determine target shapes for predictor output."
        )

    result: Dict[str, int] = {}
    shape_map = {"asr": ("asr_frames", 2), "F0_curve": ("f0_len", 1), "N": ("n_len", 1)}
    for spec in dag.get("tensors", {}).values():
        name = spec.get("input_name", "")
        if name in shape_map:
            key, dim_idx = shape_map[name]
            shape = spec.get("shape", [])
            if len(shape) > dim_idx:
                val = shape[dim_idx]
                if isinstance(val, int):
                    result[key] = val
                elif isinstance(val, dict) and "trace_value" in val:
                    result[key] = val["trace_value"]
                else:
                    raise RuntimeError(
                        f"ZERO FALLBACK: Cannot resolve shape dim for '{name}' in decoder graph."
                    )

    for required in ("asr_frames", "f0_len", "n_len"):
        if required not in result:
            raise RuntimeError(
                f"ZERO FALLBACK: '{required}' shape not found in decoder graph inputs."
            )
    return result
