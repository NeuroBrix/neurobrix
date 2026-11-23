"""The conditioning image's fit is its container's: the vae_encoder's condition_image_fit declares crop (none|center)
and resample (lanczos|bicubic|linear), in the vendor's order of operations. An undeclared fit, crop or resample is
refused by name (the key and topology.json named); the hidden LANCZOS stretch the DSP applied when nothing was declared
is gone, and the library refuses what it does not reproduce (a resample declared before a centre crop).

What would turn it red: a container declaring no fit, or no resample, running anyway (the old LANCZOS stretch), a
refusal that does not say which key and where, or a centre crop applied after the resample."""
import numpy as np
import pytest

from neurobrix.core.module.vision import image_dsp
from neurobrix.core.runtime.size_policy import image_fit


@pytest.mark.parametrize("fit", [None, {}, "lanczos", {"source": "x"}])
def test_an_undeclared_fit_is_refused_naming_the_flag_and_the_file(fit):
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: condition_image_fit\S* .*vae_encoder of Wan2.1-VACE-1.3B "
                                           r"\(topology.json extracted_values.vae_encoder.condition_image_fit\)"):
        image_fit(fit, "Wan2.1-VACE-1.3B")


@pytest.mark.parametrize("fit,key", [({"crop": "none"}, "resample"), ({"resample": "bicubic"}, "crop"),
                                     ({"crop": None, "resample": "lanczos"}, "crop")])
def test_an_undeclared_crop_or_resample_is_refused_naming_the_key(fit, key):
    with pytest.raises(RuntimeError, match=rf"ZERO FALLBACK: condition_image_fit\.{key} is not declared"):
        image_fit(fit, "M")


@pytest.mark.parametrize("fit,want", [({"crop": "none", "resample": "bicubic"}, ("none", "bicubic")),
                                      ({"crop": "center", "resample": "linear", "antialias": True,
                                        "align_corners": True}, ("center", "linear"))])
def test_a_declared_fit_is_read_as_declared(fit, want):
    assert image_fit(fit, "M") == want


def _png(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.fromarray(np.random.default_rng(5).integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(p)
    return str(p)


def test_the_dsp_no_longer_stretches_an_undeclared_image(tmp_path):
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: condition_image_fit is required"):
        image_dsp.i2v_vae_condition_np(_png(tmp_path), height=20, width=60, fit=None)


def test_the_dsp_refuses_a_resample_before_a_centre_crop(tmp_path):
    with pytest.raises(RuntimeError, match="resamples before its centre crop"):
        image_dsp.i2v_vae_condition_np(_png(tmp_path), height=20, width=60,
                                       fit={"resample": "bicubic", "crop": "center"})


@pytest.mark.parametrize("fit", [{"crop": "none", "resample": "lanczos", "antialias": True},
                                 {"crop": "none", "resample": "linear"},
                                 {"crop": "none", "resample": "nearest"}])
def test_the_dsp_refuses_a_fit_it_does_not_reproduce(tmp_path, fit):
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: condition_image_fit"):
        image_dsp.i2v_vae_condition_np(_png(tmp_path), height=20, width=60, fit=fit)
