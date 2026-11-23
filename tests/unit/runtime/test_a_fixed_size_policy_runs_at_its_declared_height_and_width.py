"""A container declaring size_policy "fixed" runs at its declared height and width whatever the conditioning image's
own size (CogVideoX-5b-I2V 480x720, Allegro-TI2V 720x1280, Wan2.1-VACE-1.3B 480x832 — the rulings of 23/11 11:13).

What would turn it red: the image's own size leaking back as the run's (a 512x512 image run at 512x512: CogVideoX-5b-I2V
'Expected size 60 but got size 64'), the declared pair swapped, or a declared side ignored when the request names the
other side only."""
import pytest

from neurobrix.core.runtime.size_policy import resolve_size


@pytest.mark.parametrize("name,h,w", [("CogVideoX-5b-I2V", 480, 720), ("Allegro-TI2V", 720, 1280),
                                      ("Wan2.1-VACE-1.3B", 480, 832)])
@pytest.mark.parametrize("image_hw", [(512, 512), (448, 448), (1080, 1920), (1280, 720)])
def test_the_declared_pair_is_the_run_size_whatever_the_image(name, h, w, image_hw):
    defaults = {"size_policy": "fixed", "height": h, "width": w}
    assert resolve_size(defaults, image_hw, None, None) == (h, w), name


def test_a_fixed_policy_needs_no_image_size():
    assert resolve_size({"size_policy": "fixed", "height": 480, "width": 720}, None, None, None) == (480, 720)


def test_a_side_the_request_does_not_name_stays_the_declared_one():
    defaults = {"size_policy": "fixed", "height": 480, "width": 720}
    assert resolve_size(defaults, (512, 512), 240, None) == (240, 720)
    assert resolve_size(defaults, (512, 512), None, 360) == (480, 360)
