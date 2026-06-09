"""Vendor-free numpy mel / waveform DSP — shared by BOTH execution modes (R34).

Pure numpy + stdlib log-mel / raw-waveform feature extraction. NO torch, NO
torchaudio, NO librosa, NO transformers. This is the single source of truth for
the audio front-end DSP: the compiled path
(`core/module/audio/input_processor.py`) and the triton path
(`triton/audio_frontend.py`) both import these functions and only differ in how
they wrap the resulting numpy array (torch.Tensor vs NBXTensor) at the boundary.

Validated bit-close to the original vendor extractors (whisper mel maxdiff
~1.7e-5, conformer ~1.3e-5; filterbank vs torchaudio ~5e-6 — fp32-level), then
proven by STT of each model. Preprocessing types:
mel_spectrogram (Whisper/Voxtral), nemo_mel (Canary/Parakeet), conformer
(Granite), raw_waveform (VibeVoice/Parakeet).
"""
import json
import math  # noqa: F401  (kept available for downstream callers / parity tooling)
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# audio IO + STFT (numpy)
# ---------------------------------------------------------------------------
def _load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n = int(len(audio) / sr * target_sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _stft_power(audio: np.ndarray, n_fft: int, win_length: int,
                hop_length: int) -> np.ndarray:
    """Centered (reflect-pad n_fft//2) power spectrogram via numpy rfft.
    Periodic Hann of win_length, zero-centred inside n_fft. [n_freqs, frames]."""
    win = np.hanning(win_length + 1)[:-1].astype(np.float64)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        win = np.pad(win, (left, n_fft - win_length - left))
    a = np.pad(audio.astype(np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
    nframes = 1 + (len(a) - n_fft) // hop_length
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(nframes)[:, None]
    frames = a[idx] * win
    spec = np.fft.rfft(frames, n=n_fft, axis=1)        # [frames, n_freqs]
    return (np.abs(spec) ** 2).T                        # [n_freqs, frames]


def _hz_to_mel(freqs: np.ndarray, *, htk: bool) -> np.ndarray:
    """Hz → mel. htk=True: O'Shaughnessy log; htk=False: Slaney auditory toolbox
    (linear below 1 kHz, log above) — both bit-for-bit as librosa.hz_to_mel."""
    freqs = np.asarray(freqs, dtype=float)
    if htk:
        return 2595.0 * np.log10(1.0 + freqs / 700.0)
    f_min, f_sp = 0.0, 200.0 / 3.0
    mels = (freqs - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = freqs >= min_log_hz
    mels = np.where(log_t, min_log_mel + np.log(freqs / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(mels: np.ndarray, *, htk: bool) -> np.ndarray:
    """mel → Hz, inverse of _hz_to_mel (librosa.mel_to_hz)."""
    mels = np.asarray(mels, dtype=float)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3.0
    freqs = f_min + f_sp * mels
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs = np.where(log_t, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)
    return freqs


def _mel_filters(sr: int, n_fft: int, n_mels: int, *, htk: bool, norm) -> np.ndarray:
    """Mel filterbank [n_mels, n_freqs] — pure numpy, R34 (no librosa). Bit-for-bit
    reimplementation of librosa.filters.mel: triangular bands over the rfft grid,
    Slaney or HTK mel scale, Slaney area-norm or none. htk+norm=None matches
    torchaudio.melscale_fbanks (~5e-6); slaney/htk=False+norm='slaney' matches whisper."""
    fmin, fmax = 0.0, sr / 2.0
    n_freqs = 1 + n_fft // 2
    fftfreqs = np.linspace(0.0, sr / 2.0, n_freqs)
    # mel band edges: n_mels+2 points equally spaced in mel
    min_mel = _hz_to_mel(np.array([fmin]), htk=htk)[0]
    max_mel = _hz_to_mel(np.array([fmax]), htk=htk)[0]
    mel_f = _mel_to_hz(np.linspace(min_mel, max_mel, n_mels + 2), htk=htk)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]            # [n_mels+2, n_freqs]
    weights = np.zeros((n_mels, n_freqs), dtype=float)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, None]
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# extractors (numpy) — bit-close mirrors of the original vendor extractors
# ---------------------------------------------------------------------------
def _whisper_mel(audio_path: str, model_path: Path, n_mels_override) -> np.ndarray:
    """Whisper log-mel (mel_spectrogram). Mirrors WhisperFeatureExtractor."""
    cfg = {}
    cp = model_path / "preprocessor_config.json"
    if cp.exists():
        cfg = json.load(open(cp))
    n_fft = cfg.get("n_fft", 400)
    hop = cfg.get("hop_length", 160)
    sr = cfg.get("sampling_rate", 16000)
    chunk = cfg.get("chunk_length", 30)
    n_mels = n_mels_override or cfg.get("feature_size", 80)
    audio = _load_audio(audio_path, sr)
    nsamp = chunk * sr
    audio = audio[:nsamp] if len(audio) >= nsamp else np.pad(audio, (0, nsamp - len(audio)))
    power = _stft_power(audio, n_fft, n_fft, hop)[:, :-1]   # drop last frame
    mf = _mel_filters(sr, n_fft, n_mels, htk=False, norm="slaney")
    mel = mf @ power
    log = np.log10(np.clip(mel, 1e-10, None))
    log = np.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    return log[None].astype(np.float32)                    # [1, n_mels, frames]


def _conformer_mel(audio_path: str, model_path: Path, n_mels_override) -> np.ndarray:
    """Granite conformer: log10 mel + amax norm + frame-stacking."""
    sr, n_fft, win, hop, n_mels, fs = 16000, 512, 400, 160, 80, 2
    cp = model_path / "preprocessor_config.json"
    if cp.exists():
        c = json.load(open(cp))
        sr = c.get("sampling_rate", sr); n_fft = c.get("n_fft", n_fft)
        win = c.get("win_length", win); hop = c.get("hop_length", hop)
        n_mels = c.get("n_mels", n_mels); fs = c.get("frame_stack", fs)
    audio = _load_audio(audio_path, sr)
    power = _stft_power(audio, n_fft, win, hop)
    mf = _mel_filters(sr, n_fft, n_mels, htk=True, norm=None)
    log = np.log10(np.clip((mf @ power).T, 1e-10, None))    # [frames, n_mels]
    log = np.maximum(log, log.max() - 8.0) / 4.0 + 1.0
    if fs > 1:
        rem = log.shape[0] % fs
        if rem:
            log = log[: log.shape[0] - rem]
        log = log.reshape(-1, fs * log.shape[-1])
    return log[None].astype(np.float32)                    # [1, frames//fs, fs*n_mels]


def _nemo_mel(audio_path: str, model_path: Path, n_mels_override, rng=None) -> np.ndarray:
    """NeMo mel (Canary): pre-emphasis + dither + log mel + per-feature normalize."""
    n_mels, n_fft, win, hop = 80, 512, 400, 160
    sr, dither, preemph = 16000, 1e-5, 0.97
    yp = model_path / "model_config.yaml"
    if yp.exists():
        try:
            import yaml
            pp = (yaml.safe_load(open(yp)) or {}).get("preprocessor", {})
            sr = pp.get("sample_rate", sr); n_fft = pp.get("n_fft", n_fft)
            n_mels = pp.get("features", n_mels); dither = pp.get("dither", dither)
            win = int(pp.get("window_size", 0.025) * sr)
            hop = int(pp.get("window_stride", 0.01) * sr)
        except Exception:
            pass
    cp = model_path / "config.json"
    if cp.exists():
        try:
            pp = (json.load(open(cp)).get("perception", {}) or {}).get("preprocessor", {})
            if pp:
                sr = pp.get("sample_rate", sr); n_fft = pp.get("n_fft", n_fft)
                n_mels = pp.get("features", n_mels); dither = pp.get("dither", dither)
                win = int(pp.get("window_size", 0.025) * sr)
                hop = int(pp.get("window_stride", 0.01) * sr)
        except Exception:
            pass
    if n_mels_override in (40, 64, 80, 128):
        n_mels = n_mels_override
    audio = _load_audio(audio_path, sr).astype(np.float64)
    if preemph > 0:
        audio = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1]])
    if dither > 0:
        _r = rng if rng is not None else np.random
        audio = audio + dither * _r.standard_normal(audio.shape)
    power = _stft_power(audio, n_fft, win, hop)
    mf = _mel_filters(sr, n_fft, n_mels, htk=True, norm=None)
    mel = np.log(np.clip((power.T @ mf.T), 1e-5, None))     # [frames, n_mels]
    mean = mel.mean(axis=0, keepdims=True)
    std = np.clip(mel.std(axis=0, keepdims=True), 1e-5, None)
    mel = (mel - mean) / std
    return mel.T[None].astype(np.float32)                  # [1, n_mels, frames]


def _raw_waveform(audio_path: str, model_path: Path, input_shape) -> np.ndarray:
    """Raw waveform (Parakeet) reshaped to the graph input shape."""
    sr = 16000
    cp = model_path / "preprocessor_config.json"
    if cp.exists():
        sr = json.load(open(cp)).get("sampling_rate", sr)
    audio = _load_audio(audio_path, sr)[None]              # [1, samples]
    if input_shape and len(input_shape) == 3:
        channels, target = input_shape[1], input_shape[2]
        wf = audio[:, None, :]                              # [1, 1, samples]
        if channels > 1:
            wf = np.repeat(wf, channels, axis=1)
        if wf.shape[2] > target:
            wf = wf[:, :, :target]
        elif wf.shape[2] < target:
            wf = np.concatenate(
                [wf, np.zeros((1, channels, target - wf.shape[2]), np.float32)], axis=2)
        return wf.astype(np.float32)
    return audio.astype(np.float32)


def extract_features_np(preprocessing_type: str, audio_path: str, model_path: Path,
                        input_shape: Optional[Tuple[int, ...]] = None,
                        rng=None) -> np.ndarray:
    n_mels_override = None
    if input_shape and len(input_shape) >= 3 and input_shape[1] in (40, 64, 80, 128):
        n_mels_override = input_shape[1]
    if preprocessing_type == "mel_spectrogram":
        return _whisper_mel(audio_path, model_path, n_mels_override)
    if preprocessing_type == "conformer":
        return _conformer_mel(audio_path, model_path, n_mels_override)
    if preprocessing_type == "nemo_mel":
        return _nemo_mel(audio_path, model_path, n_mels_override, rng=rng)
    if preprocessing_type == "raw_waveform":
        return _raw_waveform(audio_path, model_path, input_shape)
    raise RuntimeError(
        f"ZERO FALLBACK: unknown numpy audio preprocessing '{preprocessing_type}'. "
        f"Supported: mel_spectrogram, nemo_mel, conformer, raw_waveform.")
