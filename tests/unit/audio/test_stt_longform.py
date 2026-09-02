"""Long-form STT rules (core/module/audio/stt_longform.py) — no GPU.

The whisper cases are the vendor rule set replayed on hand-built token
sequences (transformers `_retrieve_segment` /
`WhisperTimeStampLogitsProcessor`, read 2026-09-02); the RNNT case is a
partition property: over random window/overlap pairs the kept intervals
tile the encoder timeline with no gap and no double count.
"""
import numpy as np
import pytest

from neurobrix.core.module.audio import stt_longform as L

NO_TS = 50364
TS = NO_TS + 1
EOS = 50257
DEFAULTS = {"no_timestamps_token_id": NO_TS, "eos_token_id": EOS,
            "forced_decoder_ids": [[1, 50259], [2, 50360]],
            "max_initial_timestamp_index": 50,
            "suppress_tokens": [1, 2, 7], "begin_suppress_tokens": [220, EOS]}
PROMPT = [50258, 50259, 50360]
IDS = (NO_TS, TS)
BEGIN = 3


def test_ids_and_begin_index_come_from_defaults():
    assert L.whisper_timestamp_ids(DEFAULTS) == IDS
    assert L.whisper_timestamp_ids({}) is None
    fm = L.whisper_forced_map(DEFAULTS, IDS, timestamps=True)
    assert fm == {1: 50259, 2: 50360}
    assert L.whisper_begin_index(fm) == 3
    assert L.whisper_begin_index({}) == 1
    # long-form without a prompt is refused (the vendor always has lang/task)
    with pytest.raises(RuntimeError, match="forced_decoder_ids"):
        L.whisper_forced_map({"no_timestamps_token_id": NO_TS}, IDS, timestamps=True)
    assert L.whisper_forced_map({}, IDS, timestamps=False) == {}


def test_forced_notimestamps_is_dropped_only_in_long_form():
    d = dict(DEFAULTS, forced_decoder_ids=[[1, 50259], [2, 50360], [3, NO_TS]])
    assert L.whisper_forced_map(d, IDS, timestamps=False) == {1: 50259, 2: 50360, 3: NO_TS}
    fm = L.whisper_forced_map(d, IDS, timestamps=True)
    assert fm == {1: 50259, 2: 50360} and L.whisper_begin_index(fm) == 3
    # a None position is the vendor's "model chooses"
    d = dict(DEFAULTS, forced_decoder_ids=[[1, None], [2, 50360]])
    assert L.whisper_forced_map(d, IDS, timestamps=True) == {2: 50360}


def _row(vocab=TS + 1501):
    rng = np.random.default_rng(0)
    return rng.standard_normal(vocab).astype(np.float32)


def test_begin_position_forces_a_bounded_initial_timestamp():
    lo = L.apply_whisper_logit_rules(_row(), PROMPT, DEFAULTS, IDS, BEGIN, timestamps=True)
    assert np.isneginf(lo[:TS]).all()                 # no text at the first free slot
    assert np.isneginf(lo[NO_TS])
    assert np.isfinite(lo[TS:TS + 51]).all()          # <|0.00|> .. <|1.00|> allowed
    assert np.isneginf(lo[TS + 51:]).all()            # max_initial_timestamp_index = 50


def test_suppression_lists_apply_without_long_form():
    lo = L.apply_whisper_logit_rules(_row(), PROMPT + [100], DEFAULTS, None, BEGIN, timestamps=False)
    assert np.isneginf(lo[[1, 2, 7]]).all()
    assert np.isfinite(lo[220])                       # begin_suppress only at begin
    lo = L.apply_whisper_logit_rules(_row(), PROMPT, DEFAULTS, None, BEGIN, timestamps=False)
    assert np.isneginf(lo[[220, EOS]]).all()


def test_timestamps_come_in_pairs_and_never_decrease():
    # after "<|0.00|> text": the next stamp must be >= the last one
    lo = L.apply_whisper_logit_rules(_row(), PROMPT + [TS, 100], DEFAULTS, IDS, BEGIN, timestamps=True)
    assert np.isneginf(lo[TS:TS + 1]).all()           # <|0.00|> again is forbidden
    # after "<|0.00|> text <|1.00|>": a segment end -> no plain text next
    lo = L.apply_whisper_logit_rules(_row(), PROMPT + [TS, 100, TS + 50], DEFAULTS, IDS, BEGIN, timestamps=True)
    assert np.isneginf(lo[:EOS]).all()
    assert np.isfinite(lo[TS + 50])                   # the pair may repeat the stamp
    # after "<|1.00|><|1.00|>": a segment start -> text must follow
    lo = L.apply_whisper_logit_rules(_row(), PROMPT + [TS, 100, TS + 50, TS + 50], DEFAULTS, IDS, BEGIN, timestamps=True)
    assert np.isneginf(lo[TS:]).all()


def test_seek_rules_match_the_vendor():
    b = BEGIN
    # consecutive pair then an unfinished segment: seek to the pair, drop the tail
    seq = PROMPT + [TS, 10, 11, TS + 120, TS + 120, 12, 13, EOS]
    adv, keep = L.whisper_seek(seq, b, TS, EOS)
    assert adv == 120 and seq[:keep] == PROMPT + [TS, 10, 11, TS + 120, TS + 120]
    # single stamp at the very end: no speech after it -> whole window
    seq = PROMPT + [TS, 10, TS + 120, TS + 120, 12, TS + 900, EOS]
    assert L.whisper_seek(seq, b, TS, EOS) == (None, len(seq))
    # no pair at all -> whole window
    seq = PROMPT + [TS, 10, 11, 12, EOS]
    assert L.whisper_seek(seq, b, TS, EOS) == (None, len(seq))
    # two pairs: the LAST pair decides
    seq = PROMPT + [TS, 1, TS + 40, TS + 40, 2, TS + 300, TS + 300, 3]
    adv, keep = L.whisper_seek(seq, b, TS, EOS)
    assert adv == 300 and keep == len(seq) - 1
    # empty / prompt-only sequences are safe
    assert L.whisper_seek(PROMPT, b, TS, EOS) == (None, len(PROMPT))


def test_advance_in_samples_is_measured_not_assumed():
    # 3000 mel frames -> 1500 encoder frames, hop 160: one index = 2 frames = 320 samples
    assert L.whisper_advance_samples(120, 3000, 1500, 160) == 120 * 320
    # a hypothetical 4x stride: one index = 4 mel frames
    assert L.whisper_advance_samples(10, 3000, 750, 160) == 10 * 640
    with pytest.raises(RuntimeError, match="whole multiple"):
        L.whisper_advance_samples(10, 3000, 700, 160)


@pytest.mark.parametrize("seed", range(40))
def test_rnnt_kept_ranges_partition_the_timeline(seed):
    rng = np.random.default_rng(seed)
    w_mel = int(rng.integers(400, 4000))
    ov_mel = int(rng.integers(1, w_mel // 2))
    total = int(rng.integers(w_mel + 1, 12 * w_mel))
    plan = L.rnnt_window_plan(total, w_mel, ov_mel)
    n = len(plan)
    # window plan covers the whole input, consecutive windows overlap by ov
    assert plan[0][0] == 0
    assert plan[-1][0] + plan[-1][1] == total
    for (s0, _), (s1, _) in zip(plan, plan[1:]):
        assert s1 - s0 == w_mel - ov_mel
    # encoder timeline at an arbitrary integer subsampling
    sub = int(rng.integers(1, 9))
    w_enc = -(-w_mel // sub)
    ov_enc = int(round(ov_mel * w_enc / w_mel))
    stride_enc = w_enc - ov_enc
    covered = []
    for i in range(n):
        lo, hi = L.rnnt_keep_range(i, n, w_enc, ov_enc)
        assert 0 <= lo < hi
        covered.append((i * stride_enc + lo, i * stride_enc + hi))
    for (_, e0), (s1, _) in zip(covered, covered[1:]):
        assert e0 == s1, "gap or double count at a seam"
    assert covered[0][0] == 0


def test_rnnt_merge_keeps_tokens_by_frame():
    toks = [1, 2, 3, 4, 5]
    frs = [0, 5, 12, 13, 30]
    # window 1 of 3, W_enc 40, overlap 26: keeps [13, 27)
    assert L.rnnt_merge_window(toks, frs, 1, 3, 40, 26) == [4]
    assert L.rnnt_merge_window(toks, frs, 0, 3, 40, 26) == [1, 2, 3, 4]
    assert L.rnnt_merge_window(toks, frs, 2, 3, 40, 26) == [4, 5]
    assert L.rnnt_merge_window(toks, frs, 0, 1, 40, 26) == toks
    with pytest.raises(RuntimeError):
        L.rnnt_merge_window(toks, frs[:-1], 1, 3, 40, 26)


def test_rnnt_merge_refuses_the_feature_axis():
    # The 2026-09-02 review class: the caller hands the encoder's FEATURE
    # count (1024) as the window and an overlap scaled on it (68) — the
    # emitted frames (< 375) make the mismatch detectable only when the
    # overlap outgrows the real window; the guard fires on the plain
    # inversion below and the frame-beyond-window case.
    with pytest.raises(RuntimeError, match="wrong axis"):
        L.rnnt_merge_window([1, 2], [0, 400], 0, 2, 375, 25)
    with pytest.raises(RuntimeError, match="wrong axis"):
        L.rnnt_merge_window([1, 2], [0, 4], 0, 2, 25, 375)
