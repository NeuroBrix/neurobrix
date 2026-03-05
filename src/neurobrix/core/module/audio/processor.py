"""
Audio preprocessing for encoder-decoder speech models.

Loads audio files and computes mel spectrograms for Whisper-family models.
Uses the HuggingFace WhisperProcessor for feature extraction.

ZERO HARDCODE: All parameters come from the model's preprocessor_config.json.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


class AudioProcessor:
    """
    Loads audio and extracts mel spectrogram features.

    Uses WhisperFeatureExtractor from the model's snapshot for exact
    parameter compatibility (sampling_rate, n_fft, n_mels, etc.).
    """

    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._feature_extractor = None
        self._model_path = model_path
        self._load_feature_extractor()

    def _load_feature_extractor(self) -> None:
        """Load WhisperFeatureExtractor from preprocessor_config.json."""
        from transformers import WhisperFeatureExtractor

        config_path = self._model_path / "preprocessor_config.json"
        if config_path.exists():
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                str(self._model_path)
            )
        else:
            # ZERO FALLBACK: crash if no preprocessor config
            raise RuntimeError(
                f"ZERO FALLBACK: No preprocessor_config.json found at {self._model_path}"
            )

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and resample to model's expected sample rate.

        Returns numpy array of float32 samples at the correct sample rate.
        """
        import soundfile as sf

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        audio, sr = sf.read(str(audio_path), dtype="float32")

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        target_sr = self._feature_extractor.sampling_rate
        if sr != target_sr:
            audio = self._resample(audio, sr, target_sr)

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using linear interpolation."""
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract mel spectrogram features from audio samples.

        Args:
            audio: Float32 numpy array of audio samples at model's sample rate

        Returns:
            input_features tensor [1, n_mels, frames] ready for encoder
        """
        features = self._feature_extractor(
            audio,
            sampling_rate=self._feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        input_features = features.input_features  # [1, n_mels, frames]
        return input_features.to(device=self.device, dtype=self.dtype)

    def process_audio_file(self, audio_path: str) -> torch.Tensor:
        """
        Full pipeline: load audio file → extract mel features.

        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)

        Returns:
            input_features tensor [1, n_mels, frames]
        """
        audio = self.load_audio(audio_path)
        return self.extract_features(audio)
