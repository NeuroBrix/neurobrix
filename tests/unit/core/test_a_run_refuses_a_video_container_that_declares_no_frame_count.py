"""A run plans a video container with the frame count the container declares, and refuses by name one that declares none.

Before: `request_input_config` resolved `num_frames` as request -> container -> the FAMILY config, and
`config/families/video.yml` gave `defaults.num_frames: 16`. A video container that declared no frame count was planned
at 16 frames, a number no model stated (zero hardcode: every model value is embedded in the container); the serve path
already planned such a container with no count and refused it by name.

What this stands on: `request_input_config` on a container directory holding a `runtime/defaults.json` and a manifest
naming the `video` family; no weights, no topology, no device.

What this test would do if the code were wrong: a family default (the yml or a literal) brings the count back, so the
no-frame-count case returns a plan at 16 instead of raising; a resolver that dropped the container's count fails the
second case.
"""
import json
from types import SimpleNamespace as NS

import pytest

_BASE = {"height": 480, "width": 832, "vae_scale_factor": 8, "dtype": "float16", "batch_size": 1}


def _plan(tmp_path, defaults):
    from neurobrix.cli.commands.run import request_input_config
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps(defaults))
    args = NS(prompt="a cat")          # the request names no size, no frame count
    return request_input_config(args, {"family": "video", "model_name": "a video model"}, "video", tmp_path)


def test_a_run_refuses_a_video_container_that_declares_no_frame_count(tmp_path):
    from neurobrix.core.runtime_values import MissingRuntimeValue
    with pytest.raises(MissingRuntimeValue, match="num_frames"):
        _plan(tmp_path, dict(_BASE))


def test_a_run_plans_a_video_container_with_the_frame_count_it_declares(tmp_path):
    ic = _plan(tmp_path, dict(_BASE, num_frames=81, temporal_compression_ratio=4))
    assert ic.num_frames == 81
