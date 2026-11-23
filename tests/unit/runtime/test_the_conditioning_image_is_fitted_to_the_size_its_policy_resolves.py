"""The conditioning image is fitted to the size the container's size policy resolves, as its vae_encoder declares the
fit, and that size rides with the inputs (global.height / global.width) for the executor — through the one door every
caller takes (`input_processor.prepare_image_inputs`: cli run, the daemon, the plan's VLM pricing). It replaces B4's
test (18a3b1bcb), whose last case kept the image's own size when the container declared none: that is now refused.

The five rulings of 23/11 11:13, each on a container built here (runtime/defaults.json + the vae_encoder flag):
  Wan2.1-I2V-14B-480P  from_image 399360 / 16, bicubic, no crop
  Wan2.2-I2V-A14B      from_image 921600 / 16, bicubic, no crop
  CogVideoX-5b-I2V     fixed 480x720, lanczos, no crop
  Allegro-TI2V         fixed 720x1280, centre crop THEN antialiased linear
  Wan2.1-VACE-1.3B     fixed 480x832, the resample its container declares (lanczos or bicubic, both run here)

What would turn it red: the clip at the image's own size or at the request's raw size instead of the resolved one,
the resolved size not handed on (the executor then generates at another), the resample not the declared PIL filter,
Allegro's crop skipped or done after the resample, or a container with no size_policy / no fit accepted."""
import json

import numpy as np
import pytest

from neurobrix.core.module.vision.input_processor import prepare_image_inputs
from neurobrix.nbx import component_flags

LINEAR = {"crop": "center", "resample": "linear", "antialias": True, "align_corners": True}
RULINGS = {
    "Wan2.1-I2V-14B-480P": ({"size_policy": "from_image", "max_area": 399360, "size_multiple": 16},
                            {"crop": "none", "resample": "bicubic"}),
    "Wan2.2-I2V-A14B": ({"size_policy": "from_image", "max_area": 921600, "size_multiple": 16},
                        {"crop": "none", "resample": "bicubic"}),
    "CogVideoX-5b-I2V": ({"size_policy": "fixed", "height": 480, "width": 720}, {"crop": "none", "resample": "lanczos"}),
    "Allegro-TI2V": ({"size_policy": "fixed", "height": 720, "width": 1280}, LINEAR),
    "Wan2.1-VACE-1.3B": ({"size_policy": "fixed", "height": 480, "width": 832}, {"crop": "none", "resample": "lanczos"}),
}


def _png(tmp_path, h, w, name="img.png"):
    from PIL import Image
    a = np.random.default_rng(h * 7 + w).integers(0, 256, (h, w, 3), dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(a).save(p)
    return str(p), a


def _container(tmp_path, monkeypatch, name, defaults, fit):
    """A container cache: manifest (a video model), runtime/defaults.json, the vae_encoder's fit flag carried."""
    root = tmp_path / "cache"
    (root / "runtime").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"family": "video", "model_name": name}))
    if defaults is not None:
        (root / "runtime" / "defaults.json").write_text(json.dumps(defaults))
    key = f"size-policy-test::{name}"                       # a name no developer registry declares
    monkeypatch.setitem(component_flags._TABLE, key, {} if fit is None else {"vae_encoder": {"condition_image_fit": fit}})
    return root, key


def _fit(root, key, image, height=None, width=None):
    return prepare_image_inputs({"flow": {"type": "iterative_process"}}, key, image, root, height=height, width=width)


@pytest.mark.parametrize("name,image_hw,want", [
    ("Wan2.1-I2V-14B-480P", (448, 448), (624, 624)), ("Wan2.1-I2V-14B-480P", (480, 832), (480, 832)),
    ("Wan2.2-I2V-A14B", (448, 448), (960, 960)), ("CogVideoX-5b-I2V", (512, 512), (480, 720)),
    ("Allegro-TI2V", (448, 448), (720, 1280)), ("Wan2.1-VACE-1.3B", (512, 512), (480, 832)),
])
def test_each_ruling_fits_the_image_to_its_resolved_size_and_hands_it_on(tmp_path, monkeypatch, name, image_hw, want):
    root, key = _container(tmp_path, monkeypatch, name, *RULINGS[name])
    image, _ = _png(tmp_path, *image_hw)
    ins = _fit(root, key, image)
    assert ins["global.image"].shape == (1, 3, 1) + want
    assert (ins["global.height"], ins["global.width"]) == want


@pytest.mark.parametrize("name", ["Wan2.1-I2V-14B-480P", "CogVideoX-5b-I2V", "Wan2.1-VACE-1.3B"])
def test_a_stretch_is_the_declared_pil_filter_at_the_resolved_size(tmp_path, monkeypatch, name):
    from PIL import Image
    defaults, fit = RULINGS[name]
    root, key = _container(tmp_path, monkeypatch, name, defaults, fit)
    image, a = _png(tmp_path, 448, 448)
    ins = _fit(root, key, image)
    h, w = ins["global.height"], ins["global.width"]
    filt = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC}[fit["resample"]]
    want = np.asarray(Image.fromarray(a).resize((w, h), filt), dtype=np.float32) / 127.5 - 1.0
    np.testing.assert_array_equal(ins["global.image"][0, :, 0], want.transpose(2, 0, 1))


@pytest.mark.parametrize("resample", ["lanczos", "bicubic"])
def test_vace_takes_the_resample_its_container_declares(tmp_path, monkeypatch, resample):
    from PIL import Image
    root, key = _container(tmp_path, monkeypatch, "Wan2.1-VACE-1.3B", RULINGS["Wan2.1-VACE-1.3B"][0],
                           {"crop": "none", "resample": resample})
    image, a = _png(tmp_path, 300, 500)
    got = _fit(root, key, image)["global.image"][0, :, 0]
    want = np.asarray(Image.fromarray(a).resize((832, 480), getattr(Image, resample.upper())), dtype=np.float32)
    np.testing.assert_array_equal(got, (want / 127.5 - 1.0).transpose(2, 0, 1))


def test_allegro_crops_to_the_target_aspect_then_resamples(tmp_path, monkeypatch):
    """The centre window of a 448 square at 16:9 (252 rows from row 98), saved as its own image and only resampled,
    gives the same clip: the crop came first."""
    from PIL import Image
    root, key = _container(tmp_path, monkeypatch, "Allegro-TI2V", *RULINGS["Allegro-TI2V"])
    image, a = _png(tmp_path, 448, 448)
    window = tmp_path / "window.png"
    Image.fromarray(a[98:98 + 252]).save(window)
    from neurobrix.core.module.vision import image_dsp
    want = image_dsp.i2v_vae_condition_np(str(window), height=720, width=1280, fit={**LINEAR, "crop": "none"})
    np.testing.assert_array_equal(_fit(root, key, image)["global.image"], want)


def test_the_request_size_is_fitted_when_it_is_a_multiple(tmp_path, monkeypatch):
    root, key = _container(tmp_path, monkeypatch, "Wan2.2-I2V-A14B", *RULINGS["Wan2.2-I2V-A14B"])
    image, _ = _png(tmp_path, 448, 448)
    ins = _fit(root, key, image, height=480, width=832)
    assert ins["global.image"].shape == (1, 3, 1, 480, 832) and (ins["global.height"], ins["global.width"]) == (480, 832)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: the requested width 830 is not a multiple of size_multiple = 16"):
        _fit(root, key, image, height=480, width=830)


@pytest.mark.parametrize("defaults", [None, {"num_frames": 49}, {"height": 480, "width": 720}])
def test_a_container_without_a_size_policy_is_refused_not_given_the_images_size(tmp_path, monkeypatch, defaults):
    root, key = _container(tmp_path, monkeypatch, "M", defaults, {"crop": "none", "resample": "lanczos"})
    image, _ = _png(tmp_path, 512, 512)
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: size_policy is not declared in .*runtime/defaults.json"):
        _fit(root, key, image)


def test_a_container_without_a_fit_is_refused_by_name(tmp_path, monkeypatch):
    root, key = _container(tmp_path, monkeypatch, "M", RULINGS["CogVideoX-5b-I2V"][0], None)
    image, _ = _png(tmp_path, 512, 512)
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: condition_image_fit is not declared on the vae_encoder"):
        _fit(root, key, image)
