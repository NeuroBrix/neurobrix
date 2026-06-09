"""Unit test for Kokoro per-phoneme duration mapping (P-AUDIO-P0a).

Regression guard for the "babbling phonemes / hey hey hey" symptom Hocine
flagged on Kokoro-82M (Dette E baseline). `_scale_kokoro_durations` used to
*force* the predicted per-phoneme durations to sum to exactly the trace-time
fixed decoder window (`target_asr_frames`, 128 ≈ 10 s @ 80 ms/frame). A short
prompt whose natural durations sum to ~15 frames got stretched ~8x, so every
phoneme was elongated 3-5x beyond natural speech -> vowel smearing.

The fix: when the natural rounded durations already fit within `target`, return
them unchanged (the alignment loop leaves the unused tail frames zero and the
waveform is cropped post-decode). Only compress when the natural sum overflows
`target` (prompt longer than the traced decoder window).

`test_durations_preserved_when_fitting_target` MUST fail on the pre-fix code
(it returned a sum of 128, not 15) and pass once the early-return lands. The
other two tests guard the overflow and all-zero edge paths against regression.

Runnable two ways:
  - pytest:  PYTHONPATH=src python -m pytest tests/unit/audio/test_kokoro_durations.py -v
  - script:  PYTHONPATH=src python tests/unit/audio/test_kokoro_durations.py
"""
from __future__ import annotations

try:
    import pytest
except ModuleNotFoundError:  # script-mode under the pytest-less GPU runtime venv
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore

import torch

from neurobrix.core.flow.stages.kokoro import (
    _expand_kokoro_curve,
    _scale_kokoro_durations,
)


def test_durations_preserved_when_fitting_target():
    """Short prompt: natural durations fit the decoder window -> keep them.

    round([3.2, 4.1, 1.8, 5.9]) = [3, 4, 2, 6], sum = 15 << target 128.
    Pre-fix this was stretched to sum 128 (the babbling bug).
    """
    raw = torch.tensor([3.2, 4.1, 1.8, 5.9])
    result = _scale_kokoro_durations(raw, target=128)

    assert result.tolist() == [3, 4, 2, 6], (
        f"natural durations should be preserved unchanged, got {result.tolist()}"
    )
    assert int(result.sum().item()) == 15, (
        f"sum must stay at the natural 15 frames, not the stretched 128 — "
        f"got {int(result.sum().item())}"
    )


def test_durations_compressed_when_overflowing_target():
    """Long prompt: natural durations overflow the decoder window -> compress.

    sum([50, 60, 70]) = 180 > target 128. The overflow path must still pack
    everything into exactly `target` frames (long-prompt fallback; the proper
    fix is the build-side dynamic-frames re-trace, P-BUILD-KOKORO-DYNAMIC-FRAMES).
    """
    raw = torch.tensor([50.0, 60.0, 70.0])
    result = _scale_kokoro_durations(raw, target=128)

    assert int(result.sum().item()) == 128, (
        f"overflowing durations must compress to exactly target 128, "
        f"got {int(result.sum().item())}"
    )
    assert (result >= 1).all(), "active phonemes must keep at least 1 frame"


def test_all_zero_durations_fills_first_slot():
    """Degenerate all-zero prediction: first slot absorbs the whole window."""
    raw = torch.zeros(5)
    result = _scale_kokoro_durations(raw, target=128)

    assert int(result.sum().item()) == 128
    assert int(result[0].item()) == 128


def test_expand_curve_full_window_matches_plain_interpolate():
    """content_ratio == 1.0 reproduces the original full interpolation."""
    raw = torch.linspace(0.0, 1.0, 8).reshape(1, 1, 8)
    out = _expand_kokoro_curve(raw, target_len=16, content_ratio=1.0)
    ref = torch.nn.functional.interpolate(
        raw, size=16, mode="linear", align_corners=False
    ).squeeze(1)
    assert out.shape == (1, 16)
    assert torch.allclose(out, ref)


def test_expand_curve_zero_pads_tail_past_content():
    """content_ratio == 0.5: only the first half carries prosody, tail is zero.

    A naive "interpolate the whole raw curve to target_len" (the pre-fix
    behaviour) would leave the tail non-zero — this asserts the content-prefix
    + zero-pad semantics that keep the iSTFTNet source silent past the speech.
    """
    raw = torch.ones(1, 1, 10)  # uniform non-zero curve
    out = _expand_kokoro_curve(raw, target_len=20, content_ratio=0.5)
    assert out.shape == (1, 20)
    # First half (content) is the interpolated non-zero prefix.
    assert (out[:, :10] > 0).all(), "content region must carry the curve"
    # Second half (tail past content) must be exactly zero.
    assert torch.count_nonzero(out[:, 10:]) == 0, "tail past content must be zero-padded"


if __name__ == "__main__":
    test_durations_preserved_when_fitting_target()
    test_durations_compressed_when_overflowing_target()
    test_all_zero_durations_fills_first_slot()
    test_expand_curve_full_window_matches_plain_interpolate()
    test_expand_curve_zero_pads_tail_past_content()
    print("all kokoro duration + curve tests passed")
