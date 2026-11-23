"""A container declaring size_policy "from_image" runs at max_area pixels in the conditioning image's aspect, each side
floored to size_multiple, by the ONE diffusers form (the Wan I2V pipelines' example): with ar = h / w,
height = round(sqrt(max_area * ar)) // m * m, width = round(sqrt(max_area / ar)) // m * m.
Wan2.1-I2V-14B-480P declares 399360 / 16, Wan2.2-I2V-A14B 921600 / 16 (the rulings of 23/11 11:13).

The expected figures are worked by hand here, not by the formula under test. What would turn it red: floor instead of
round before the floor-to-multiple (576x1000 at 399360: sqrt(230031.36) = 479.62 rounds to 480 -> 480, floors to
479 -> 464), the aspect inverted (height and width swapped for a portrait image), ceil to the multiple instead of
floor (632 -> 640), or max_area read as a side."""
import pytest

from neurobrix.core.runtime.size_policy import from_image_size, resolve_size

WAN21_480P = {"size_policy": "from_image", "max_area": 399360, "size_multiple": 16}
WAN22_A14B = {"size_policy": "from_image", "max_area": 921600, "size_multiple": 16}


@pytest.mark.parametrize("defaults,image_hw,want", [
    (WAN21_480P, (480, 832), (480, 832)),      # the vendor's own 480P frame: sqrt(230400) = 480, sqrt(692224) = 832
    (WAN21_480P, (448, 448), (624, 624)),      # sqrt(399360) = 631.95 -> 632 -> 39 * 16
    (WAN21_480P, (832, 480), (832, 480)),      # portrait: the aspect is h / w, not w / h
    (WAN21_480P, (576, 1000), (480, 832)),     # round, then floor: 479.62 -> 480 (a floor would give 464); 832.67 -> 833 -> 832
    (WAN22_A14B, (720, 1280), (720, 1280)),    # sqrt(518400) = 720, sqrt(1638400) = 1280
    (WAN22_A14B, (448, 448), (960, 960)),      # sqrt(921600) = 960
    (WAN22_A14B, (1024, 768), (1104, 816)),    # sqrt(1228800) = 1108.51 -> 1109 -> 69 * 16; sqrt(691200) = 831.38 -> 51 * 16
])
def test_the_size_follows_the_image_aspect_at_the_declared_area(defaults, image_hw, want):
    assert resolve_size(defaults, image_hw, None, None) == want


def test_every_side_is_a_multiple_and_the_area_stays_within_a_rounding_of_the_declared_one():
    for hw in ((300, 500), (333, 211), (1080, 1920), (1, 1), (720, 721)):
        h, w = from_image_size(921600, 16, hw)
        assert h % 16 == 0 and w % 16 == 0 and h * w <= 921600 + 16 * (h + w), hw


def test_a_from_image_policy_without_an_image_size_is_refused_by_name():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: size_policy = 'from_image' .* gave no image size"):
        resolve_size(WAN21_480P, None, None, None)


def test_an_aspect_that_leaves_a_side_under_the_multiple_is_refused_by_name():
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: the conditioning image's aspect"):
        resolve_size(WAN21_480P, (1, 100000), None, None)
