"""Long-form STT rules shared by BOTH engines — numpy + stdlib only (R34).

Same house pattern as `mel_dsp`: the compiled flow and its triton mirror
import these functions and differ only at the tensor boundary (a torch
row or an NBXTensor row becomes one numpy vector here). No torch, no
vendor package.

Whisper class (`encoder_decoder` flow) — the vendor's own long-form
algorithm: decode one fixed window WITH timestamps, seek to the end of
the last complete segment, decode again from there. Ported from the
vendor sources read on 2026-09-02:

  transformers/models/whisper/generation_whisper.py  `_retrieve_segment`,
      the init-token handling of `no_timestamps_token_id`
  transformers/generation/logits_process.py          `WhisperTimeStampLogitsProcessor`
  (openai/whisper `transcribe.py` carries the same rule set)

Every identifier is DATA carried by the build: the timestamp base is
`no_timestamps_token_id + 1` (the vendor's definition), the suppression
lists and the initial-timestamp cap come from generation_config.json,
copied into defaults.json by the build toolchain. The seconds per
timestamp index is `input_stride * hop / sr`, with `input_stride` the
mel frames per encoder frame — both measured on the run, never assumed.

RNNT class (`rnnt` flow) — buffered inference: overlapping windows, one
greedy decode per window, tokens merged by the encoder frame they were
emitted at. This is NeMo's own long-form recipe
(examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py:
overlapping buffers, decoder state per chunk, tokens kept by position).
The kept ranges partition the encoder timeline exactly (unit-tested):
no frame is decoded twice, none is dropped. The caller measures the
overlap on the encoder's FRAME axis (the axis the decode walks), never
on the feature axis.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_NEG = -float("inf")


# ---------------------------------------------------------------------------
# Whisper class
# ---------------------------------------------------------------------------
def whisper_timestamp_ids(defaults: dict) -> Optional[Tuple[int, int]]:
    """`(no_timestamps_token_id, timestamp_begin)` from the build's
    defaults, or None when the build carries no timestamp data (a
    container built before the toolchain copied it: long-form refuses
    loudly at the flow, short clips are unaffected)."""
    no_ts = defaults.get("no_timestamps_token_id")
    if no_ts is None:
        return None
    return int(no_ts), int(no_ts) + 1


def whisper_forced_map(defaults: dict, ts_ids: Optional[Tuple[int, int]],
                       *, timestamps: bool) -> Dict[int, int]:
    """Forced decoder positions (`forced_decoder_ids`) as {position: id}.

    With timestamps on, the vendor removes `<|notimestamps|>` from the
    prompt (generation_whisper.py: the init tokens drop
    `no_timestamps_token_id` when timestamps are returned) — a forced
    `<|notimestamps|>` followed by a forced timestamp would be
    out-of-distribution decoding. Positions whose id is None (the
    vendor's "let the model choose") are left to the model."""
    forced = {int(pos): tid for pos, tid in (defaults.get("forced_decoder_ids") or [])
              if tid is not None}
    if timestamps:
        if ts_ids is not None:
            forced = {pos: tid for pos, tid in forced.items() if int(tid) != ts_ids[0]}
        if not forced:
            raise RuntimeError(
                "ZERO FALLBACK: long-form decoding needs the model's prompt "
                "(forced_decoder_ids: language/task tokens) in defaults.json — the "
                "vendor never decodes with timestamps from a bare start token.")
    return forced


def whisper_begin_index(forced_map: Dict[int, int]) -> int:
    """Length of the decoder prompt: the start token plus every forced
    position, i.e. the index of the first token the model chooses —
    the vendor's `begin_index`."""
    return 1 + max(forced_map, default=0)


def apply_whisper_logit_rules(logits: np.ndarray, generated: Sequence[int],
                              defaults: dict, ts_ids: Optional[Tuple[int, int]],
                              begin: int, *, timestamps: bool) -> np.ndarray:
    """Vendor logit rules, in place on one float row (the last position).

    Always: `suppress_tokens`, and `begin_suppress_tokens` at the first
    free position — when the build carries them. In long-form
    (`timestamps=True`): the timestamp grammar of
    `WhisperTimeStampLogitsProcessor` — no `<|notimestamps|>`, stamps in
    pairs, non-decreasing, an initial stamp bounded by
    `max_initial_timestamp_index`, and a stamp whenever the timestamp
    mass beats the best text token. `generated` is the whole decoder
    sequence so far (start token, forced tokens, sampled tokens);
    `begin` its prompt length.
    """
    at_begin = len(generated) == begin
    sup = defaults.get("suppress_tokens") or []
    if sup:
        logits[np.asarray(sup, dtype=np.int64)] = _NEG
    if at_begin:
        bsup = defaults.get("begin_suppress_tokens") or []
        if bsup:
            logits[np.asarray(bsup, dtype=np.int64)] = _NEG
    if not timestamps or ts_ids is None:
        return logits

    no_ts, ts_begin = ts_ids
    eos = int(defaults["eos_token_id"])
    logits[no_ts] = _NEG

    seq = list(generated[begin:])
    last_was_ts = len(seq) >= 1 and seq[-1] >= ts_begin
    penultimate_was_ts = len(seq) < 2 or seq[-2] >= ts_begin
    if last_was_ts:
        if penultimate_was_ts:      # a segment start: text must follow
            logits[ts_begin:] = _NEG
        else:                       # a segment end: no plain text next
            logits[:eos] = _NEG
    stamps = [t for t in seq if t >= ts_begin]
    if stamps:
        if last_was_ts and not penultimate_was_ts:
            last_allowed = stamps[-1]
        else:
            last_allowed = stamps[-1] + 1   # never the same stamp twice
        logits[ts_begin:last_allowed] = _NEG
    if at_begin:
        logits[:ts_begin] = _NEG
        max_initial = defaults.get("max_initial_timestamp_index")
        if max_initial is not None:
            logits[ts_begin + int(max_initial) + 1:] = _NEG

    # A stamp whenever the timestamp mass beats the best text token
    # (log-softmax over the masked row, as the vendor does it).
    lp = logits.astype(np.float64)
    lp = lp - lp.max()
    lp = lp - np.log(np.exp(lp).sum())
    ts_mass = np.logaddexp.reduce(lp[ts_begin:])
    if ts_mass > lp[:ts_begin].max():
        logits[:ts_begin] = _NEG
    return logits


def whisper_seek(generated: Sequence[int], begin: int, ts_begin: int,
                 eos: int) -> Tuple[Optional[int], int]:
    """The vendor's seek rule on one decoded window.

    Returns `(advance, keep)`: `advance` in TIMESTAMP INDEX units
    (None = consume the whole window), `keep` = how many tokens of
    `generated` belong to complete segments. With consecutive stamps
    (`…text <|t|><|t|> text…`) and no single stamp at the very end, the
    unfinished trailing segment is dropped and the next window starts
    at the last complete segment's end; a single stamp at the end means
    no speech after it (whole window); no pairs at all: whole window.
    """
    seq = list(generated[begin:])
    if seq and seq[-1] == eos:
        seq = seq[:-1]
    is_ts = [t >= ts_begin for t in seq]
    single_ending = is_ts[-2:] == [False, True]
    consecutive = [i + 1 for i in range(len(seq) - 1) if is_ts[i] and is_ts[i + 1]]
    if consecutive and not single_ending:
        last_slice = consecutive[-1] + 1
        advance = seq[last_slice - 2] - ts_begin
        return advance, begin + last_slice
    return None, len(generated)


def whisper_advance_samples(advance_index: int, mel_frames_window: int,
                            encoder_frames: int, hop: int) -> int:
    """Timestamp index -> audio samples: one index is `input_stride` mel
    frames, `input_stride` being the mel frames per encoder frame
    (measured: window mel frames / encoder output frames)."""
    if encoder_frames <= 0 or mel_frames_window % encoder_frames:
        raise RuntimeError(
            f"ZERO FALLBACK: {mel_frames_window} mel frames per window is not a whole "
            f"multiple of the {encoder_frames} encoder frames — the vendor's input "
            "stride is an integer by construction (conv strides).")
    input_stride = mel_frames_window // encoder_frames
    return advance_index * input_stride * hop


# ---------------------------------------------------------------------------
# RNNT class
# ---------------------------------------------------------------------------
def rnnt_window_plan(actual_frames: int, window_frames: int,
                     overlap_frames: int) -> List[Tuple[int, int]]:
    """`(start, valid)` per window over the mel timeline: consecutive
    windows overlap by `overlap_frames`; the last one is shorter (the
    flow pads it like a short clip). One window when it all fits."""
    if actual_frames <= window_frames:
        return [(0, actual_frames)]
    if not 0 < overlap_frames < window_frames:
        raise ValueError(
            f"overlap {overlap_frames} must lie inside the window {window_frames}")
    stride = window_frames - overlap_frames
    n = -(-(actual_frames - overlap_frames) // stride)
    return [(i * stride, min(window_frames, actual_frames - i * stride))
            for i in range(n)]


def rnnt_keep_range(i: int, n: int, window_enc: int,
                    overlap_enc: int) -> Tuple[int, int]:
    """Encoder-frame interval `[lo, hi)` (window-relative) whose tokens
    window `i` of `n` contributes. The first window keeps its head, the
    last its tail, and every shared edge is split so the kept intervals
    tile the timeline with no gap and no double count:
    window i ends at `W - ceil(ov/2)`, window i+1 starts at `floor(ov/2)`
    — the two meet at the same absolute frame."""
    lo = 0 if i == 0 else overlap_enc // 2
    hi = window_enc if i == n - 1 else window_enc - (overlap_enc - overlap_enc // 2)
    return lo, hi


def rnnt_merge_window(tokens: Sequence[int], frames: Sequence[int],
                      i: int, n: int, window_enc: int,
                      overlap_enc: int) -> List[int]:
    """Tokens of window `i` that fall inside its kept interval.
    `window_enc` and `overlap_enc` count ENCODER FRAMES (the axis
    `frames` indexes); `overlap_enc` must be shorter than the window."""
    if n <= 1 or overlap_enc <= 0:
        return list(tokens)
    if len(tokens) != len(frames):
        raise RuntimeError(
            f"RNNT merge: {len(tokens)} tokens but {len(frames)} frame marks")
    if not 0 < overlap_enc < window_enc:
        raise RuntimeError(
            f"RNNT merge: overlap {overlap_enc} frames does not fit the window "
            f"of {window_enc} encoder frames (wrong axis?)")
    if frames and max(frames) >= window_enc:
        raise RuntimeError(
            f"RNNT merge: a token was emitted at frame {max(frames)} beyond the "
            f"window of {window_enc} encoder frames (wrong axis?)")
    lo, hi = rnnt_keep_range(i, n, window_enc, overlap_enc)
    return [t for t, f in zip(tokens, frames) if lo <= f < hi]
