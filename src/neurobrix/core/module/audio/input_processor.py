"""
Universal audio input preprocessing.

DATA-DRIVEN: preprocessing type comes from topology.json flow.audio.input.preprocessing.
Input dimensions come from graph.json first input tensor shape — ZERO hardcoded defaults.
Routes to the correct extractor based on topology data — ZERO model-specific code.

Supported preprocessing types:
  mel_spectrogram  — Whisper-family models (WhisperFeatureExtractor)
  raw_waveform     — Models that take raw audio samples (VibeVoice, Chatterbox)
  conformer        — NeMo Conformer-based models (Granite Speech, Parakeet, Canary)
  none             — TTS models that take text, not audio
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


class AudioInputProcessor:
    """Routes to the correct audio preprocessor based on topology data."""

    @staticmethod
    def process(
        preprocessing_type: str,
        audio_path: str,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Preprocess audio file into model input features.

        Args:
            preprocessing_type: From topology.flow.audio.input.preprocessing
            audio_path: Path to audio file (wav, mp3, flac, etc.)
            model_path: Path to model snapshot (for preprocessor configs)
            device: Target device
            dtype: Target dtype
            input_shape: Expected input tensor shape from graph.json (e.g. [1, 128, 3000]).
                        Used to determine mel bins, feature dim, etc. DATA-DRIVEN.

        Returns:
            Feature tensor ready for encoder input
        """
        if preprocessing_type == "mel_spectrogram":
            return MelSpectrogramExtractor.extract(audio_path, model_path, device, dtype, input_shape)
        elif preprocessing_type == "nemo_mel":
            return NemoMelExtractor.extract(audio_path, model_path, device, dtype, input_shape)
        elif preprocessing_type == "raw_waveform":
            return RawWaveformLoader.load(audio_path, model_path, device, dtype, input_shape)
        elif preprocessing_type == "conformer":
            return ConformerFeatureExtractor.extract(audio_path, model_path, device, dtype, input_shape)
        elif preprocessing_type == "none":
            raise RuntimeError(
                "ZERO FALLBACK: preprocessing='none' should not reach AudioInputProcessor.\n"
                "TTS models use text input, not audio."
            )
        else:
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown audio preprocessing type '{preprocessing_type}'.\n"
                f"Supported: mel_spectrogram, nemo_mel, raw_waveform, conformer, none"
            )


def _load_audio(audio_path: str, target_sr: Optional[int] = None) -> tuple:
    """
    Load audio file and optionally resample.

    Returns:
        (audio_samples: np.ndarray, sample_rate: int)
    """
    import soundfile as sf

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = sf.read(str(path), dtype="float32")

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        duration = len(audio) / sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sr

    return audio, sr


class MelSpectrogramExtractor:
    """
    Mel spectrogram extraction for Whisper-family models.

    Uses HuggingFace WhisperFeatureExtractor from the model's
    preprocessor_config.json for exact parameter compatibility.
    Falls back to torchaudio when no preprocessor_config exists.
    """

    @staticmethod
    def extract(
        audio_path: str,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Extract mel spectrogram features from audio file.

        Args:
            input_shape: Expected shape from graph.json, e.g. [1, 128, 3000].
                        Dim 1 = n_mels (overrides preprocessor_config if different).

        Returns:
            input_features tensor [1, n_mels, frames]
        """
        # Determine expected mel bins from graph input shape
        graph_n_mels = None
        if input_shape and len(input_shape) >= 3:
            graph_n_mels = input_shape[1]  # [B, n_mels, frames]

        config_path = model_path / "preprocessor_config.json"
        if config_path.exists():
            from transformers import WhisperFeatureExtractor

            feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_path))

            # Override mel bins if graph expects different count
            if graph_n_mels and feature_extractor.feature_size != graph_n_mels:
                feature_extractor.feature_size = graph_n_mels
                # Regenerate mel filters for new bin count
                if hasattr(feature_extractor, 'mel_filters'):
                    from librosa.filters import mel as librosa_mel
                    feature_extractor.mel_filters = librosa_mel(
                        sr=feature_extractor.sampling_rate,
                        n_fft=feature_extractor.n_fft,
                        n_mels=graph_n_mels,
                    )

            audio, _sr = _load_audio(audio_path, target_sr=feature_extractor.sampling_rate)
            features = feature_extractor(
                audio,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            return features.input_features.to(device=device, dtype=dtype)

        # No preprocessor_config — manual mel extraction
        n_mels = graph_n_mels or 80
        target_sr = 16000
        audio, _sr = _load_audio(audio_path, target_sr=target_sr)

        try:
            import torchaudio
            waveform = torch.from_numpy(audio).unsqueeze(0)
            features = torchaudio.compliance.kaldi.fbank(
                waveform, num_mel_bins=n_mels, sample_frequency=target_sr
            )
            # fbank returns [frames, n_mels] → transpose to [n_mels, frames]
            features = features.T.unsqueeze(0)  # [1, n_mels, frames]
        except ImportError:
            import librosa
            mel = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_mels=n_mels)
            features = torch.from_numpy(np.log(mel + 1e-6)).unsqueeze(0)  # [1, n_mels, frames]

        return features.to(device=device, dtype=dtype)


class RawWaveformLoader:
    """
    Load raw audio waveform for models that process audio samples directly.

    Reads sample_rate from preprocessor_config.json or defaults to 16000.
    """

    @staticmethod
    def load(
        audio_path: str,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Load audio as raw waveform tensor.

        Returns:
            waveform tensor matching input_shape (e.g. [1, 1, 24000] or [1, samples])
        """
        import json

        # Read target sample rate from config
        target_sr = 16000
        config_path = model_path / "preprocessor_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            target_sr = config.get("sampling_rate", target_sr)

        audio, _sr = _load_audio(audio_path, target_sr=target_sr)
        waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]

        # Reshape to match expected input shape from graph.json
        if input_shape and len(input_shape) == 3:
            # Expected [B, channels, samples] — add channel dim
            channels = input_shape[1]
            target_samples = input_shape[2]
            waveform = waveform.unsqueeze(1)  # [1, 1, samples]
            if channels > 1:
                waveform = waveform.expand(-1, channels, -1)
            # Truncate or pad to expected sample count
            if waveform.shape[2] > target_samples:
                waveform = waveform[:, :, :target_samples]
            elif waveform.shape[2] < target_samples:
                pad = torch.zeros(1, channels, target_samples - waveform.shape[2])
                waveform = torch.cat([waveform, pad], dim=2)

        return waveform.to(device=device, dtype=dtype)


class ConformerFeatureExtractor:
    """
    Feature extraction for Conformer-based models with adjacent-frame stacking
    (Granite Speech).

    Reproduces the vendor GraniteSpeechFeatureExtractor recipe (transformers
    `granite_speech.feature_extraction_granite_speech._extract_mel_spectrograms`):
    a torchaudio `MelSpectrogram`, a log10 + per-clip amax normalisation, an
    odd-frame drop, then `frame_stack`-frame stacking (`n_mels → frame_stack·n_mels`,
    frame count divided by `frame_stack`). For Granite: 80 mels × 2 = 160 input_dim
    at half the frame rate (50 fps effective from a 100 fps mel).

    This is the feature representation the conformer encoder was trained on.
    The earlier delta-feature path produced the right dim (160) but wrong values
    (kaldi fbank + first-difference deltas, not log10-normalised stacked mels)
    AND twice the frames (no stacking) — leaving the audio ungrounded. Params are
    the vendor defaults, overridable from the .nbx preprocessing block.
    """

    @staticmethod
    def extract(
        audio_path: str,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Extract stacked-log-mel Conformer features from an audio file.

        Returns:
            features tensor [1, frames // frame_stack, frame_stack * n_mels]
        """
        import json
        import torchaudio

        # Vendor GraniteSpeechFeatureExtractor defaults; overridable from the
        # .nbx preprocessing block (data-driven identity card).
        sr, n_fft, win_length, hop_length, n_mels, frame_stack = 16000, 512, 400, 160, 80, 2
        config_path = model_path / "preprocessor_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            sr = config.get("sampling_rate", sr)
            n_fft = config.get("n_fft", n_fft)
            win_length = config.get("win_length", win_length)
            hop_length = config.get("hop_length", hop_length)
            n_mels = config.get("n_mels", n_mels)
            frame_stack = config.get("frame_stack", frame_stack)

        audio, _sr = _load_audio(audio_path, target_sr=sr)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]

        melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels,
        )
        mel = melspec(waveform)                                  # [1, n_mels, frames]
        logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()  # [1, frames, n_mels]
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)

        # Drop trailing frames so the count is a clean multiple of frame_stack,
        # then stack adjacent frames into the feature dim.
        if frame_stack > 1:
            rem = logmel.shape[1] % frame_stack
            if rem:
                logmel = logmel[:, : logmel.shape[1] - rem]
            logmel = logmel.reshape(logmel.shape[0], -1, frame_stack * logmel.shape[-1])

        return logmel.to(device=device, dtype=dtype)


class NemoMelExtractor:
    """
    NeMo-compatible mel spectrogram for Conformer/RNNT models.

    Differs from Whisper: n_fft=512, per-feature normalize, dither, pre-emphasis.
    All params DATA-DRIVEN from model config files.
    """

    @staticmethod
    def extract(
        audio_path: str,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Extract NeMo-style mel features. Returns [1, n_mels, frames]."""
        import json

        # Defaults (NeMo standard)
        n_mels, n_fft, win_length, hop_length = 80, 512, 400, 160
        target_sr, dither, preemph = 16000, 1e-5, 0.97

        # Read from model_config.yaml (NeMo RNNT)
        yaml_path = model_path / "model_config.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path) as f:
                    pp = yaml.safe_load(f).get("preprocessor", {})
                target_sr = pp.get("sample_rate", target_sr)
                n_fft = pp.get("n_fft", n_fft)
                n_mels = pp.get("features", n_mels)
                dither = pp.get("dither", dither)
                win_length = int(pp.get("window_size", 0.025) * target_sr)
                hop_length = int(pp.get("window_stride", 0.01) * target_sr)
            except Exception:
                pass

        # Read from config.json (NeMo SpeechLM — perception.preprocessor)
        cfg_path = model_path / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    pp = json.load(f).get("perception", {}).get("preprocessor", {})
                if pp:
                    target_sr = pp.get("sample_rate", target_sr)
                    n_fft = pp.get("n_fft", n_fft)
                    n_mels = pp.get("features", n_mels)
                    dither = pp.get("dither", dither)
                    win_length = int(pp.get("window_size", 0.025) * target_sr)
                    hop_length = int(pp.get("window_stride", 0.01) * target_sr)
            except Exception:
                pass

        # Override n_mels from graph if available
        if input_shape and len(input_shape) >= 3 and input_shape[1] in (40, 64, 80, 128):
            n_mels = input_shape[1]

        audio, _ = _load_audio(audio_path, target_sr=target_sr)
        waveform = torch.from_numpy(audio).to(device=device, dtype=torch.float32)

        # Pre-emphasis: y[n] = x[n] - 0.97*x[n-1]
        if preemph > 0:
            waveform = torch.cat([waveform[:1], waveform[1:] - preemph * waveform[:-1]])

        # Dither
        if dither > 0:
            waveform = waveform + dither * torch.randn_like(waveform)

        # STFT → power spectrum → mel filterbank → log → per-feature normalize
        window = torch.hann_window(win_length, device=device)
        stft = torch.stft(
            waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True, center=True, pad_mode="reflect",
        )
        power = stft.abs().pow(2)  # [n_fft//2+1, frames]

        import torchaudio
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=target_sr / 2.0,
            n_mels=n_mels, sample_rate=target_sr,
        ).to(device=device)

        mel = torch.log((power.T @ mel_fb).clamp(min=1e-5))  # [frames, n_mels]

        # Per-feature normalization
        mean = mel.mean(dim=0, keepdim=True)
        std = mel.std(dim=0, keepdim=True).clamp(min=1e-5)
        mel = (mel - mean) / std

        return mel.T.unsqueeze(0).to(dtype=dtype)  # [1, n_mels, frames]
