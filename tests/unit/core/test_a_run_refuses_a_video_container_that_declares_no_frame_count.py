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
_ALL = dict(_BASE, num_frames=81, temporal_compression_ratio=4, num_inference_steps=50, guidance_scale=5.0, fps=16)


def _plan(tmp_path, defaults, args=None):
    from neurobrix.cli.commands.run import request_input_config
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps(defaults))
    args = args or NS(prompt="a cat")          # the request names no size, no frame count
    return request_input_config(args, {"family": "video", "model_name": "a video model"}, "video", tmp_path)


def test_a_run_refuses_a_video_container_that_declares_no_frame_count(tmp_path):
    from neurobrix.core.runtime_values import MissingRuntimeValue
    with pytest.raises(MissingRuntimeValue, match="num_frames"):
        _plan(tmp_path, dict(_BASE))


def test_a_run_plans_a_video_container_with_the_frame_count_it_declares(tmp_path):
    ic = _plan(tmp_path, dict(_ALL))
    assert ic.num_frames == 81


@pytest.mark.parametrize("key", ["num_frames", "num_inference_steps", "guidance_scale", "fps"])
def test_a_run_refuses_by_name_a_video_container_that_leaves_out_any_value_the_family_lists(tmp_path, key):
    from neurobrix.core.runtime_values import MissingRuntimeValue
    declared = {k: v for k, v in _ALL.items() if k != key}
    with pytest.raises(MissingRuntimeValue, match=key) as e:
        _plan(tmp_path, declared)
    assert "a video model" in str(e.value) and str(tmp_path) in str(e.value)    # the container, as the run knows it


def test_a_request_flag_stands_in_for_the_steps_and_the_guidance_a_container_leaves_out(tmp_path):
    declared = {k: v for k, v in _ALL.items() if k not in ("num_inference_steps", "guidance_scale")}
    ic = _plan(tmp_path, declared, NS(prompt="a cat", steps=4, cfg=3.0))
    assert ic.num_frames == 81


def test_the_check_is_driven_by_every_key_the_family_lists_not_by_the_frame_count_alone(tmp_path, monkeypatch):
    import neurobrix.core.config as cfg
    from neurobrix.core.runtime_values import MissingRuntimeValue
    real = cfg.get_family_config
    monkeypatch.setattr(cfg, "get_family_config", lambda f: dict(real(f), required_container_values=["a_second_key"]))
    with pytest.raises(MissingRuntimeValue, match="a_second_key"):
        _plan(tmp_path, dict(_ALL))


def test_the_video_family_file_holds_no_frame_rate():
    from neurobrix.core.config import get_family_config
    cfg = get_family_config("video")
    assert "fps" not in (cfg.get("output_processing") or {})
    assert "fps" in cfg["required_container_values"]


def test_the_video_writer_refuses_by_name_a_container_that_declares_no_frame_rate(tmp_path):
    from neurobrix.core.runtime.output_dispatch import save_video
    from neurobrix.core.runtime_values import MissingRuntimeValue
    executor = NS(get_final_output=lambda outputs: object(), variable_resolver=NS(resolved={}))
    pkg = NS(defaults={}, manifest={"model_name": "a video model"})
    with pytest.raises(MissingRuntimeValue, match="fps") as e:
        save_video({}, str(tmp_path / "o.mp4"), "video", executor, pkg)
    assert "a video model" in str(e.value)
