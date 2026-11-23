"""An explicit --height/--width wins over the size policy: for "from_image" when each side is a multiple of the declared
size_multiple, else refused naming the multiple; for "fixed" any value, as the existing door accepted it.

What would turn it red: a request that is a multiple overridden by the image-derived size, a non-multiple silently
rounded (the run then differs from what was asked and from what the image was fitted to), or a refusal that does not
name the multiple to ask."""
import pytest

from neurobrix.core.runtime.size_policy import resolve_size

WAN21_480P = {"size_policy": "from_image", "max_area": 399360, "size_multiple": 16}


def test_a_requested_multiple_wins_over_the_image():
    assert resolve_size(WAN21_480P, (448, 448), 480, 832) == (480, 832)


def test_a_requested_multiple_needs_no_image():
    assert resolve_size(WAN21_480P, None, 320, 576) == (320, 576)


def test_one_requested_side_wins_and_the_other_follows_the_image():
    assert resolve_size(WAN21_480P, (448, 448), 480, None) == (480, 624)


@pytest.mark.parametrize("h,w,side,value", [(500, 832, "height", 500), (480, 830, "width", 830), (481, None, "height", 481)])
def test_a_request_that_is_not_a_multiple_is_refused_naming_the_multiple(h, w, side, value):
    with pytest.raises(RuntimeError, match=f"ZERO FALLBACK: the requested {side} {value} is not a multiple of "
                                           f"size_multiple = 16 .*ask a multiple of 16"):
        resolve_size(WAN21_480P, (448, 448), h, w)


def test_a_fixed_policy_takes_any_requested_size():
    assert resolve_size({"size_policy": "fixed", "height": 480, "width": 720}, (512, 512), 250, 333) == (250, 333)
