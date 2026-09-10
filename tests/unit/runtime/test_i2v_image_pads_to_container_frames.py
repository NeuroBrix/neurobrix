"""The image→clip padding uses the frame count the container declares.

`cmd_run` resolves the request's runtime dims once, near the top::

    height     = args.height     or defaults.json or family config or 1024
    width      = args.width      or defaults.json or family config or 1024
    num_frames = args.num_frames or defaults.json

and hands them to Prism. Seventy lines further down, the call that turns a
still image into the I2V conditioning clip re-derived one of them from the raw
argument instead::

    num_frames=int(getattr(args, "num_frames", 0) or 0)

With no `--frames` on the command line that is 0, so `pad_to_num_frames` is 0
and the image stays a ONE-frame clip — for every model declaring the
`vae_encoder.pad_image_to_num_frames` flag, whatever its container says.

The containers do say. Allegro-TI2V declares 88, the Wan I2V pair 81, VACE 81.
Allegro-TI2V's VAE compresses time by 4 and refuses an extent that is not a
multiple of it, so the arm died in six seconds on::

    [TilingEngine] downscale input temporal extent 1 is not a multiple of
    the temporal compression ratio 4

The same file already carries the scar of this bug being fixed once, for
height/width: "run.py read only defaults.json + a hardcoded 1024 fallback, so
for video Prism estimated the VAE at 8x the real activation". `num_frames`
never got the same treatment.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_i2v_image_pads_to_container_frames.py
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ASSET = (pathlib.Path(__file__).resolve().parents[3]
         / "benchmarks" / "assets" / "apple_448.png")
RUN_PY = (pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
          / "cli" / "commands" / "run.py")


def _t_extent(clip):
    """The clip is [1, 3, T, H, W] in either engine's container."""
    shape = getattr(clip, "shape", None)
    assert shape is not None and len(shape) == 5, f"expected 5-D clip, got {shape}"
    return int(shape[2])


def test_padding_to_a_frame_count_produces_that_many_frames():
    from neurobrix.core.module.vision.input_processor import ImageInputProcessor

    if not ASSET.exists():
        pytest.skip(f"test asset missing: {ASSET}")
    clip = ImageInputProcessor.process("i2v_vae_condition", str(ASSET),
                                       height=64, width=64,
                                       pad_to_num_frames=88)
    assert _t_extent(clip) == 88


def test_padding_to_zero_leaves_a_single_frame():
    """The current default when no --frames is given — one frame, which a
    temporal VAE with a compression ratio of 4 refuses."""
    from neurobrix.core.module.vision.input_processor import ImageInputProcessor

    if not ASSET.exists():
        pytest.skip(f"test asset missing: {ASSET}")
    clip = ImageInputProcessor.process("i2v_vae_condition", str(ASSET),
                                       height=64, width=64,
                                       pad_to_num_frames=0)
    assert _t_extent(clip) == 1


def test_the_image_call_site_uses_the_resolved_frame_count():
    """One resolution, both consumers. The failure mode of a second derivation
    is silence: the clip is simply one frame and nothing says why, until a VAE
    dozens of ops later objects to its temporal extent."""
    tree = ast.parse(RUN_PY.read_text(), filename=str(RUN_PY))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "prepare_image_inputs"]
    assert calls, "prepare_image_inputs is no longer called from run.py"
    for call in calls:
        kw = next((k for k in call.keywords if k.arg == "num_frames"), None)
        assert kw is not None, "prepare_image_inputs called without num_frames"
        names = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(kw.value) if isinstance(n, ast.Attribute)}
        assert "num_frames" in names, (
            f"line {call.lineno}: num_frames is re-derived here "
            f"(names={sorted(names)}, attrs={sorted(attrs)}) instead of using "
            "the value cmd_run already resolved from args / defaults.json / "
            "the family config")
