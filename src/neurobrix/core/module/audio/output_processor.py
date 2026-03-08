"""
Universal audio output processing.

Handles both STT (tokens → text) and TTS (waveform tensor → .wav file).
Supports SNAC codec decoding for Orpheus-style TTS models.

ZERO HARDCODE: sample_rate and bit_depth from topology/defaults.
"""

from typing import Any, List

import torch


class AudioOutputProcessor:
    """Audio output processing for STT and TTS models."""

    @staticmethod
    def decode_tokens(token_ids: List[int], tokenizer: Any, skip_special: bool = True) -> str:
        """Decode token IDs to text string (STT)."""
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(token_ids, skip_special_tokens=skip_special)
        return str(token_ids)

    @staticmethod
    def save_waveform(
        waveform: torch.Tensor,
        output_path: str,
        sample_rate: int = 16000,
    ) -> str:
        """Save waveform tensor as .wav file (TTS)."""
        import soundfile as sf

        audio_np = waveform.cpu().float().numpy()

        while audio_np.ndim > 1 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)

        max_val = abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        sf.write(output_path, audio_np, sample_rate, subtype="PCM_16")
        return output_path

    @staticmethod
    def decode_snac_tokens(
        token_ids: List[int],
        vocab_size: int = 156940,
        audio_token_start: int = 128266,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Decode Orpheus-style audio token IDs to waveform via SNAC codec.

        Orpheus generates token IDs in range [audio_token_start, vocab_size).
        These map to SNAC codebook indices across 3 levels (1+2+4=7 tokens/frame).

        Args:
            token_ids: Raw token IDs from autoregressive generation
            vocab_size: Model vocab size (156940 for Orpheus)
            audio_token_start: Start of audio token range (128266 for Orpheus)
            device: Device for SNAC decode

        Returns:
            Waveform tensor [1, 1, samples] at 24kHz
        """
        # Filter to audio tokens only
        audio_tokens = [t - audio_token_start for t in token_ids
                        if audio_token_start <= t < vocab_size]

        if len(audio_tokens) < 7:
            return torch.zeros(1, 1, 1)

        # SNAC has 3 codebook levels: level 0 (1 token), level 1 (2 tokens), level 2 (4 tokens)
        # Each frame = 7 tokens in order: [L0, L1a, L1b, L2a, L2b, L2c, L2d]
        n_frames = len(audio_tokens) // 7
        audio_tokens = audio_tokens[:n_frames * 7]

        # Separate into codebook levels
        codes_0 = []  # 1 per frame
        codes_1 = []  # 2 per frame
        codes_2 = []  # 4 per frame

        for i in range(n_frames):
            base = i * 7
            codes_0.append(audio_tokens[base])
            codes_1.extend(audio_tokens[base + 1:base + 3])
            codes_2.extend(audio_tokens[base + 3:base + 7])

        # Each level has its own codebook with 4096 entries
        # Tokens are offset: level 0 = [0, 4096), level 1 = [4096, 8192), level 2 = [8192, 12288)
        codes_0 = [c % 4096 for c in codes_0]
        codes_1 = [c % 4096 for c in codes_1]
        codes_2 = [c % 4096 for c in codes_2]

        codes_0_t = torch.tensor(codes_0, dtype=torch.long, device=device).unsqueeze(0)
        codes_1_t = torch.tensor(codes_1, dtype=torch.long, device=device).unsqueeze(0)
        codes_2_t = torch.tensor(codes_2, dtype=torch.long, device=device).unsqueeze(0)

        try:
            import snac
            snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
            with torch.inference_mode():
                audio = snac_model.decode([codes_0_t, codes_1_t, codes_2_t])
            return audio  # [1, 1, samples] at 24kHz
        except ImportError:
            print("   [WARN] SNAC not installed — cannot decode audio tokens")
            print("   Install with: pip install snac")
            return torch.zeros(1, 1, 1)
        except Exception as e:
            print(f"   [WARN] SNAC decode failed: {e}")
            return torch.zeros(1, 1, 1)
