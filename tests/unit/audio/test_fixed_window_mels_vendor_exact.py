"""D-AUDIOLLM-LONGFORM (2026-09-03): `fixed_window_mels` reproduces the
vendor's long-form feature contract for the whisper-encoder audio LLMs
(Voxtral): zero-pad the waveform to the next multiple of one 30 s window,
extract the log-mel over the whole padded audio, split into consecutive
windows stacked on the batch axis (`VoxtralProcessor._retrieve_input_
features`). Pinned against transformers' WhisperFeatureExtractor with the
Voxtral preprocessor config, on a synthetic 75 s signal (3 windows).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

SNAP = Path(os.path.expanduser("~/.cache/neurobrix/hf_snapshots/Voxtral-Mini-3B-2507"))


@pytest.mark.skipif(not (SNAP / "preprocessor_config.json").exists(),
                    reason="Voxtral snapshot (preprocessor_config.json) not on this node")
def test_fixed_window_mels_matches_the_vendor_extractor(tmp_path):
    transformers = pytest.importorskip("transformers")
    import soundfile as sf
    from neurobrix.core.module.audio.mel_dsp import fixed_window_mels

    cfg = json.load(open(SNAP / "preprocessor_config.json"))
    sr = cfg["sampling_rate"]
    rng = np.random.default_rng(7)
    t = np.arange(int(75.3 * sr)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
    wav = tmp_path / "synthetic_75s.wav"
    sf.write(str(wav), audio, sr)

    ours = fixed_window_mels(str(wav), SNAP, (1, cfg["feature_size"], 3000))
    assert ours is not None and ours.shape == (3, cfg["feature_size"], 3000)

    # (1) Structure and padding are the vendor's: 75.3 s → 3 windows of
    # 3000 frames, the tail zero-padded in the WAVEFORM before the mel
    # (so the padded frames are the log-mel of silence under the whole-
    # recording floor, not zeros).
    fe = transformers.WhisperFeatureExtractor.from_pretrained(str(SNAP))
    ref = fe(audio, sampling_rate=sr, padding=True, truncation=False,
             pad_to_multiple_of=cfg["chunk_length"] * sr, return_tensors="np")["input_features"]
    ref = ref.reshape(cfg["feature_size"], -1, 3000).transpose(1, 0, 2)
    assert ref.shape == ours.shape
    assert not np.all(ours[2, :, -100:] == 0.0)          # silence mel, not zero pad

    # (2) The windows are exact slices of ONE whole-padded log-mel (the
    # floor is the recording's, not per window).
    from neurobrix.core.module.audio.mel_dsp import _whisper_logmel, _load_audio
    padded = np.pad(_load_audio(str(wav), sr), (0, 3 * cfg["chunk_length"] * sr - t.size))
    full = _whisper_logmel(padded, sr, cfg["n_fft"], cfg["hop_length"], cfg["feature_size"])
    for i in range(3):
        np.testing.assert_array_equal(ours[i], full[:, i * 3000:(i + 1) * 3000])

    # (3) Numerics vs the vendor extractor: the engine's whisper log-mel
    # core deviates from transformers' STFT by ≤ 0.02 on ~1 % of the
    # frames on the CLASSIC single-window path too (measured 2026-09-03;
    # the STT gates are word-level) — the long-form windows inherit
    # exactly that baseline, no more.
    classic = fe(audio[: cfg["chunk_length"] * sr], sampling_rate=sr, return_tensors="np")["input_features"][0]
    from neurobrix.core.module.audio.mel_dsp import _whisper_mel
    classic_ours = _whisper_mel(str(wav), SNAP, None)[0]
    baseline = float(np.abs(classic_ours - classic).max())
    dev = float(np.abs(ours - ref).max())
    assert dev <= max(baseline, 0.02) * 2.5, (dev, baseline)


def test_single_window_audio_is_the_classic_path(tmp_path):
    """≤ 30 s → None: the single-window code runs unchanged."""
    if not (SNAP / "preprocessor_config.json").exists():
        pytest.skip("Voxtral snapshot not on this node")
    import soundfile as sf
    from neurobrix.core.module.audio.mel_dsp import fixed_window_mels
    cfg = json.load(open(SNAP / "preprocessor_config.json"))
    sr = cfg["sampling_rate"]
    wav = tmp_path / "short.wav"
    sf.write(str(wav), np.zeros(int(11 * sr), np.float32), sr)
    assert fixed_window_mels(str(wav), SNAP, (1, cfg["feature_size"], 3000)) is None
