"""Audio front-end of the compiled engine: text -> phoneme ids, and the voice style they speak in.

The R30 mirror of `triton/audio_frontend.py` (`preprocess_phonemizer_input_np`): one text-to-ids
function for both engines (`core.module.audio.g2p.phoneme_ids`, its language gate first), the
voicepack read from the container (`modules/voices/`). It lived in `core/flow/stages/kokoro.py`
beside the native Kokoro stage handlers; those had no caller (the Kokoro predictor and decoder run
their traced graphs, execution=forward, in both engines), and the module was removed on
2026-10-04 (the supervisor's decision) — this function, the one part the audio flow reaches, moved
here.
"""

import torch
from typing import Dict


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
    klang = engine.ctx.pkg.defaults.get("phoneme_lang", "a")
    from neurobrix.core.module.audio.g2p import phoneme_ids, request_voice
    # The voice and the lexicon must speak the same language — `phoneme_ids` refuses before
    # a single phoneme when the requested voice speaks a language the embedded lexicon does
    # not: the model would otherwise say other words, fluently (2026-09-16, `ff_siwis` on
    # the American lexicon). One text-to-ids function for both engines and the census.
    phonemes, ids = phoneme_ids(prompt, engine.ctx.nbx_path_str, phoneme_vocab, klang,
                                request_voice(engine.ctx.variable_resolver.resolved,
                                              engine.ctx.pkg.defaults))

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
    # The log says when it is quoting only the head of the prompt: without the
    # ellipsis a 60-character cut reads as a TRUNCATION OF THE DATA, and one hour
    # of 2026-09-16 was spent looking for a text the engine had never dropped.
    _shown = prompt if len(prompt) <= 60 else prompt[:60] + "…"
    print(f"   [Phonemizer] '{_shown}' ({len(prompt)} chars) -> {len(phonemes)} phonemes "
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

def _requested_voice(engine, available) -> str:
    """Which voice this run asked for.

    The voice is a run parameter, like the prompt. It is read, in order, from
    the run's own request (`--speaker`, bound as `global.speaker` by
    cli/commands/run.py) and then from the artefact's runtime defaults. It is
    NOT chosen here: `defaults.get("voice", "af_heart")` picked one of the 54
    voicepacks this artefact ships by a literal in the code, on every run,
    without a word — and if that one was missing it silently substituted
    another. Asking for one voice and getting another is a wrong answer.
    """
    resolved = getattr(engine.ctx.variable_resolver, "resolved", {}) or {}
    for key in ("global.speaker", "speaker", "global.voice", "voice"):
        value = resolved.get(key)
        if value:
            return str(value)

    declared = engine.ctx.pkg.defaults.get("voice")
    if declared:
        return str(declared)

    raise RuntimeError(
        f"ZERO FALLBACK: this artefact ships {len(available)} voices and "
        f"declares none as its default (`voice` is absent from "
        f"runtime/defaults.json, where `phoneme_lang` is present). Choose one "
        f"with --speaker: {', '.join(available)}.")

def _load_voicepack(engine, phoneme_count: int, device) -> None:
    """Load voice pack and split into predictor/decoder styles.

    Voicepacks are [N, 256] tensors stored in modules/voices/.
    Index by phoneme count, then split: [:128]=decoder, [128:]=predictor.
    """
    from pathlib import Path

    nbx_path = Path(engine.ctx.nbx_path_str)
    voices_dir = nbx_path / "modules" / "voices"

    if not voices_dir.exists():
        raise RuntimeError(
            f"ZERO FALLBACK: this model's flow binds a voice style, but "
            f"'{voices_dir}' does not exist. Returning here used to let the "
            f"decoder and predictor run with no style bound at all.")

    available = sorted(p.stem for p in voices_dir.glob("*.pt"))
    voice_name = _requested_voice(engine, available)
    voice_path = voices_dir / f"{voice_name}.pt"

    if not voice_path.exists():
        raise RuntimeError(
            f"ZERO FALLBACK: voice '{voice_name}' was requested but "
            f"{voice_path.name} is not in {voices_dir}. Available "
            f"({len(available)}): {', '.join(available)}. Substituting "
            f"another voice would answer a different question.")

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
