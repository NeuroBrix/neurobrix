"""A video cell's mechanical check reads the artefact's size and compares it with the request.

SANA-Video on a 32 GB card wrote a 160x64 video for a 1280x512 request, and the matrix row read
rc=0 with a byte count (2026-09-27): the mechanical half of R29 checked geometry for images only,
so the one fault a size makes obvious passed as a green row until a judge looked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))

import regression_matrix as R  # noqa: E402


def _video(path, h, w, flat=False):
    import imageio.v2 as iio
    rng = np.random.default_rng(0)
    with iio.get_writer(str(path), fps=8, macro_block_size=1) as wr:
        for _ in range(5):
            f = np.full((h, w, 3), 90, np.uint8) if flat else rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            wr.append_data(f)
    return path


def test_a_video_at_the_latent_size_is_named_degenerate(tmp_path):
    m = R.mechanical(_video(tmp_path / "native.mp4", 64, 160), "video", (512, 1280))
    assert m["degenerate"] and "geometry (64, 160) is not the requested (512, 1280)" in m["reasons"][0]


def test_a_video_at_the_requested_size_passes(tmp_path):
    m = R.mechanical(_video(tmp_path / "native.mp4", 64, 160), "video", (64, 160))
    assert not m["degenerate"] and m["shape"] == [64, 160] and m["frames"] == 5


def test_a_flat_video_is_named(tmp_path):
    m = R.mechanical(_video(tmp_path / "native.mp4", 64, 160, flat=True), "video", (64, 160))
    assert m["degenerate"] and "flat colour" in m["reasons"][0]
