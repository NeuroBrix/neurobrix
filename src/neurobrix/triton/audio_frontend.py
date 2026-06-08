"""Zero-torch audio front-end for the triton path.

Pure numpy + librosa (both torch-free) mel / waveform feature extraction, mirror
of core/module/audio/input_processor.py (which uses torch/torchaudio) so the
triton compute path imports NO torch. R33: numpy is CPU glue, never torch.

Validated bit-close to the torch extractors (whisper mel maxdiff ~1.7e-5,
conformer ~1.3e-5; filterbank vs torchaudio ~5e-6 — fp32-level), then proven by
STT of each model in normal command. Preprocessing types mirror the core:
mel_spectrogram (Whisper/Voxtral), nemo_mel (Canary), conformer (Granite),
raw_waveform (Parakeet).

Two totally separate front-ends (R30/two-modes): the PyTorch flows call
core/flow/audio_utils.preprocess_audio_input (torch); the triton flows call
preprocess_audio_input_np here (numpy → NBXTensor).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor


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


def _mel_filters(sr: int, n_fft: int, n_mels: int, *, htk: bool, norm) -> np.ndarray:
    """librosa mel filterbank [n_mels, n_freqs] (numpy, torch-free). htk+norm=None
    matches torchaudio.melscale_fbanks to ~5e-6; slaney/htk=False matches whisper."""
    from librosa.filters import mel as _lmel
    return _lmel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0,
                 fmax=sr / 2.0, htk=htk, norm=norm)


# ---------------------------------------------------------------------------
# extractors (numpy) — bit-close mirrors of input_processor.py
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


# ---------------------------------------------------------------------------
# binding — zero-torch mirror of audio_utils.preprocess_audio_input
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# torch-free helpers (duplicated from core/flow/audio_utils.py — deliberate
# two-modes duplication so the triton path imports nothing that pulls torch)
# ---------------------------------------------------------------------------
def _component_input_shape(ctx, comp_name):
    if comp_name is None:
        return None
    executor = ctx.executors.get(comp_name)
    if executor is None:
        return None
    dag = getattr(executor, "_dag", None)
    if dag is None:
        return None
    for tid, spec in dag.get("tensors", {}).items():
        if (spec.get("type") == "input" or spec.get("input_name") is not None
                or tid.startswith("input::")):
            resolved = []
            for dim in spec.get("shape", []):
                if isinstance(dim, dict):
                    resolved.append(dim.get("trace_value", dim.get("trace", 0)))
                elif isinstance(dim, int):
                    resolved.append(dim)
                else:
                    resolved.append(0)
            return tuple(resolved)
    return None


def _model_config_path(ctx) -> Path:
    nbx_path = Path(ctx.nbx_path_str)
    for subdir in ["modules/processor", "modules/tokenizer"]:
        cand = nbx_path / subdir
        if cand.exists():
            return cand
    raise RuntimeError(
        "Cannot find model config path. Expected modules/processor/ or "
        "modules/tokenizer/ inside the .nbx.")


def postprocess_text_output_np(ctx) -> None:
    """Decode generated token IDs to text (torch-free mirror of audio_utils)."""
    generated_ids = ctx.variable_resolver.resolved.get("global.generated_token_ids")
    if generated_ids is None:
        return
    tokenizer = ctx.modules.get("tokenizer")
    if tokenizer is not None:
        from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
        text = AudioOutputProcessor.decode_tokens(generated_ids, tokenizer)
    else:
        text = str(generated_ids)
    ctx.variable_resolver.resolved["global.transcription"] = text
    print(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")


def preprocess_audio_input_np(ctx, audio_config: Dict, stages: List[Dict]) -> None:
    """Load audio + extract features in numpy, bind as NBXTensor (no torch)."""
    get_component_input_shape = _component_input_shape
    find_model_config_path = _model_config_path

    input_config = audio_config.get("input", {})
    audio_path = ctx.variable_resolver.resolved.get("global.audio_path")
    if audio_path is None:
        raise RuntimeError("ZERO FALLBACK: Audio model requires global.audio_path "
                           "(use --audio <path>).")
    preprocessing = input_config.get("preprocessing")
    if preprocessing is None:
        raise RuntimeError("ZERO FALLBACK: topology.flow.audio.input.preprocessing required.")
    variable = input_config.get("variable", "global.input_features")

    first_comp = stages[0]["component"] if stages else None
    input_shape = get_component_input_shape(ctx, first_comp)
    # Auto-correct preprocessing from graph shape (mirror of the torch path).
    if input_shape and len(input_shape) >= 3:
        d1, d2 = input_shape[1], input_shape[2]
        if preprocessing == "raw_waveform":
            if d1 in (40, 64, 80, 128) and d2 > d1:
                preprocessing = "mel_spectrogram"
            elif d2 in (40, 64, 80, 128, 160, 256) and d1 > d2:
                preprocessing = "conformer"

    print(f"   [Audio·np] Loading: {audio_path}")
    feats = extract_features_np(preprocessing, str(audio_path),
                                Path(find_model_config_path(ctx)), input_shape)

    # Pad/truncate to trace-time dims (mirror of the torch path).
    if input_shape and len(input_shape) == feats.ndim and feats.ndim >= 3:
        for d in range(1, len(input_shape)):
            trace, actual = input_shape[d], feats.shape[d]
            if actual > trace:
                sl = [slice(None)] * feats.ndim
                sl[d] = slice(None, trace)
                feats = feats[tuple(sl)]
            elif actual < trace:
                ps = list(feats.shape); ps[d] = trace - actual
                feats = np.concatenate([feats, np.zeros(ps, np.float32)], axis=d)
    feats = np.ascontiguousarray(feats.astype(np.float32))
    print(f"   [Audio·np] Features: {tuple(feats.shape)} ({preprocessing})")

    # Place the feature tensor on the encoder's device (NBXTensor.from_numpy uses
    # the CURRENT DeviceAllocator device — without this it lands on cuda:0 while a
    # multi-GPU-placed encoder may be on another device → "cpu tensor" at conv).
    from neurobrix.kernels.nbx_tensor import DeviceAllocator as _DA
    dev = str(getattr(ctx, "primary_device", "cuda:0"))
    try:
        _DA.set_device(int(dev.split(":")[-1].split(",")[0]) if "cuda" in dev else 0)
    except (ValueError, IndexError):
        _DA.set_device(0)
    nbx = NBXTensor.from_numpy(feats)
    ctx.variable_resolver.resolved[variable] = nbx
    short = variable.split(".")[-1] if "." in variable else variable
    ctx.variable_resolver.resolved[short] = nbx

    frames = feats.shape[-1] if preprocessing in ("mel_spectrogram", "nemo_mel") else feats.shape[1]
    length = NBXTensor.from_numpy(np.array([frames], dtype=np.int64))
    for k in ["global.audio_signal_length", "audio_signal_length", "global.length", "length"]:
        ctx.variable_resolver.resolved[k] = length
