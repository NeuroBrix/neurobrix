"""Regression tests for Mac M4 Pro video bugs (SANA-Video 0-byte MP4):

1. `restore_plan` in `resolution_binning.py` must never under-round `resized_h` or
   `resized_w` below the requested size due to IEEE 754 truncation (e.g.
   `704 * (480 / 704) == 479.99999999999994 -> int() == 479 < 480`), which turned
   `start_y = (479 - 480) // 2 = -1` and collapsed a 480-px frame to 1 px tall.
2. `_write_video_h264` in `output_dispatch.py` must never silently swallow a non-zero
   ffmpeg exit code or a 0-byte output file.
"""
from __future__ import annotations

import types
import numpy as np
import pytest

from neurobrix.core.runtime.resolution import resolution_binning as rb
from neurobrix.core.runtime import output_dispatch


def _contract() -> rb.BinningContract:
    return rb.BinningContract(
        default=True,
        source="SanaVideoPipeline.__call__(use_resolution_binning)",
        bins=(("0.57", 704, 1280), ("1.0", 960, 960)),
        mode="bilinear",
        align_corners=False,
    )


def test_restore_plan_sana_video_832x480_does_not_under_round():
    """Exact case from stat_runs 857/858: SANA-Video_2B_720p binned 480x832 -> 704x1280."""
    plan = rb.restore_plan(704, 1280, 480, 832, _contract())
    assert plan is not None
    assert plan.resized_h == 480
    assert plan.resized_w == 872
    assert plan.top == 0
    assert plan.bottom == 480
    assert plan.bottom - plan.top == 480
    assert plan.right - plan.left == 832


@pytest.mark.parametrize(
    "orig_h, orig_w, new_h, new_w",
    [
        (704, 1280, 480, 832),   # 704 * (480 / 704) == 479.99999999999994
        (1280, 704, 832, 480),   # width-dominated mirror of the same ULP drop
        (49, 100, 10, 10),       # 49 * (10 / 49) == 9.999999999999998
        (100, 49, 10, 10),       # width-dominated 49 * (10 / 49)
        (704, 1280, 360, 640),
        (1024, 1024, 512, 768),
    ],
)
def test_restore_plan_always_covers_requested_extent(orig_h, orig_w, new_h, new_w):
    plan = rb.restore_plan(orig_h, orig_w, new_h, new_w, _contract())
    assert plan is not None
    assert plan.resized_h >= new_h
    assert plan.resized_w >= new_w
    assert 0 <= plan.top < plan.bottom <= plan.resized_h
    assert 0 <= plan.left < plan.right <= plan.resized_w
    assert plan.bottom - plan.top == new_h
    assert plan.right - plan.left == new_w


def test_write_video_h264_raises_on_ffmpeg_nonzero_rc(tmp_path, monkeypatch):
    out = tmp_path / "bad.mp4"

    class _FakeProc:
        returncode = 1

        def communicate(self, input=None):
            out.write_bytes(b"")  # ffmpeg leaves 0-byte file on error
            return b"", b"[libx264] height not divisible by 2 (832x1)\n"

    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: "fake-ffmpeg"),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: _FakeProc())

    frames = np.zeros((2, 1, 832, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match=r"ffmpeg failed \(rc=1\).*height not divisible by 2"):
        output_dispatch._write_video_h264(str(out), frames, 16)


def test_write_video_h264_raises_on_empty_output_even_if_rc_zero(tmp_path, monkeypatch):
    out = tmp_path / "empty.mp4"

    class _FakeProc:
        returncode = 0

        def communicate(self, input=None):
            out.write_bytes(b"")
            return b"", b""

    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: "fake-ffmpeg"),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: _FakeProc())

    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="wrote an empty file"):
        output_dispatch._write_video_h264(str(out), frames, 16)


def test_write_video_h264_succeeds_when_nonempty(tmp_path, monkeypatch):
    out = tmp_path / "ok.mp4"

    class _FakeProc:
        returncode = 0

        def communicate(self, input=None):
            assert input is not None and len(input) == 2 * 16 * 16 * 3
            out.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return b"", b""

    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: "fake-ffmpeg"),
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: _FakeProc())

    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    output_dispatch._write_video_h264(str(out), frames, 16)
    assert out.stat().st_size > 0


def test_write_video_h264_rejects_zero_frames(tmp_path):
    frames = np.zeros((0, 480, 832, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="empty video"):
        output_dispatch._write_video_h264(str(tmp_path / "zero.mp4"), frames, 16)
