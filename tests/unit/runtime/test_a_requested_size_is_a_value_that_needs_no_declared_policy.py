"""The user's size is a value (owner 13:48, project lead 13:51, 23/11): a request naming BOTH height and width runs at
them even when the container declares no size_policy, nothing being missing; the one check is the declared
size_multiple, where one is declared. A request naming ONE side to a container with no policy is refused naming the
missing side.

What would turn it red: --height/--width refused for a container without a policy (the B4b gate), a declared
size_multiple not checked on a request when no policy is declared, or a one-sided request refused without naming the
side it lacks (or filled from the image)."""
import pytest

from neurobrix.core.runtime.size_policy import resolve_size


@pytest.mark.parametrize("defaults", [None, {}, {"num_frames": 49}, {"height": 480, "width": 720}])
def test_both_flags_with_no_policy_run_at_those_flags(defaults):
    assert resolve_size(defaults, (512, 512), 368, 640) == (368, 640)
    assert resolve_size(defaults, None, 368, 640) == (368, 640)


def test_a_declared_multiple_is_the_one_check_without_a_policy():
    assert resolve_size({"size_multiple": 16}, None, 480, 832) == (480, 832)
    with pytest.raises(RuntimeError, match="the requested width 830 is not a multiple of size_multiple = 16"):
        resolve_size({"size_multiple": 16}, None, 480, 830)


def test_a_default_policy_checks_a_declared_multiple():
    defaults = {"size_policy": "default", "height": 480, "width": 720, "size_multiple": 16}
    assert resolve_size(defaults, (512, 512), 240, 368) == (240, 368)
    with pytest.raises(RuntimeError, match="the requested height 250 is not a multiple of size_multiple = 16"):
        resolve_size(defaults, (512, 512), 250, 368)


@pytest.mark.parametrize("h,w,given,missing", [(480, None, "height", "width"), (None, 832, "width", "height")])
def test_one_flag_with_no_policy_is_refused_naming_the_missing_side(h, w, given, missing):
    with pytest.raises(RuntimeError, match=f"ZERO FALLBACK: the request names its {given} but not its {missing}, "
                                           f".*declares no size_policy to give it: pass --{missing} too"):
        resolve_size({}, (512, 512), h, w)
