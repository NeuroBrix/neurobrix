"""
Universal audio input preprocessing (compiled mode).

DATA-DRIVEN: preprocessing type comes from topology.json flow.audio.input.preprocessing.
Input dimensions come from graph.json first input tensor shape — ZERO hardcoded defaults.
Routes to the correct extractor based on topology data — ZERO model-specific code.

R34 (Zero Outsider): the feature DSP is computed by the vendor-free numpy core
`mel_dsp.extract_features_np` (shared with the triton path) — NO transformers,
NO torchaudio, NO librosa. Compiled mode only converts the resulting numpy array
to a torch.Tensor at the device boundary.

Supported preprocessing types:
  mel_spectrogram  — Whisper-family models
  raw_waveform     — Models that take raw audio samples (VibeVoice, Chatterbox)
  conformer        — Conformer-based models (Granite Speech)
  nemo_mel         — NeMo-style mel models (Parakeet, Canary)
  none             — TTS models that take text, not audio
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from neurobrix.core.module.audio import mel_dsp


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
        params: Optional[dict] = None,
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
        if preprocessing_type == "none":
            raise RuntimeError(
                "ZERO FALLBACK: preprocessing='none' should not reach AudioInputProcessor.\n"
                "TTS models use text input, not audio."
            )
        if preprocessing_type not in (
            "mel_spectrogram", "nemo_mel", "raw_waveform", "conformer",
        ):
            raise RuntimeError(
                f"ZERO FALLBACK: Unknown audio preprocessing type '{preprocessing_type}'.\n"
                f"Supported: mel_spectrogram, nemo_mel, raw_waveform, conformer, none"
            )

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Vendor-free numpy DSP (shared with the triton path, R34).
        feats = mel_dsp.extract_features_np(
            preprocessing_type, str(audio_path), Path(model_path), input_shape,
            params=params,
        )

        # Convert to torch only at the device boundary (compiled mode = torch allowed).
        return torch.from_numpy(np.ascontiguousarray(feats)).to(device=device, dtype=dtype)
