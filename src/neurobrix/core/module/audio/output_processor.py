"""
Universal audio output processing.

Handles both STT (tokens → text) and TTS (waveform tensor → .wav file).

ZERO HARDCODE: sample_rate and bit_depth from topology/defaults.
"""

from typing import Any, List

import torch


class AudioOutputProcessor:
    """Audio output processing for STT and TTS models."""

    @staticmethod
    def decode_tokens(token_ids: List[int], tokenizer: Any, skip_special: bool = True) -> str:
        """
        Decode token IDs to text string (STT).

        Args:
            token_ids: Generated token IDs from autoregressive decoder
            tokenizer: HuggingFace tokenizer with decode()
            skip_special: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_ids, skip_special_tokens=skip_special)
        return str(token_ids)

    @staticmethod
    def save_waveform(
        waveform: torch.Tensor,
        output_path: str,
        sample_rate: int = 16000,
    ) -> str:
        """
        Save waveform tensor as .wav file (TTS).

        Args:
            waveform: Audio tensor [samples] or [1, samples] or [batch, samples]
            output_path: Path to save .wav file
            sample_rate: Audio sample rate in Hz

        Returns:
            Saved file path
        """
        import soundfile as sf

        audio_np = waveform.cpu().float().numpy()

        # Squeeze batch/channel dimensions
        while audio_np.ndim > 1 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)

        # Normalize to [-1, 1] if needed
        max_val = abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        sf.write(output_path, audio_np, sample_rate, subtype="PCM_16")
        return output_path
