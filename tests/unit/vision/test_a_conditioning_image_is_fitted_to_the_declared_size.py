"""The conditioning image is fitted to the size the run generates at, in the executor's order: the request's, else
the container's declared height/width, else the image's own (CogVideoX-5b-I2V: a 512 image met 480x720 noise)."""
import json

from neurobrix.core.module.vision.input_processor import declared_condition_size


def _box(tmp_path, defaults):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "defaults.json").write_text(json.dumps(defaults))
    return tmp_path


def test_the_request_size_wins(tmp_path):
    assert declared_condition_size(_box(tmp_path, {"height": 480, "width": 720}), 256, 384) == (256, 384)


def test_the_declared_size_is_read_when_the_request_names_none(tmp_path):
    assert declared_condition_size(_box(tmp_path, {"height": 480, "width": 720}), None, None) == (480, 720)


def test_a_container_that_declares_no_size_keeps_the_image_size(tmp_path):
    assert declared_condition_size(_box(tmp_path, {"num_frames": 49}), None, None) == (None, None)
