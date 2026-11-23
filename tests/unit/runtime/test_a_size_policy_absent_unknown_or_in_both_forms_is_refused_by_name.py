"""A run with a conditioning image refuses, by name and with the file named, a container whose runtime/defaults.json
declares no size_policy, an unknown one, both forms at once, or a form with a key missing or not a positive integer.
The engine never supplies the size (no default, no family value, no image's own size).

What would turn it red: a container without size_policy running at the image's size or a family size (the fallback of
18a3b1bcb), "from_image" running with height/width beside max_area, or a missing key filled in."""
import pytest

from neurobrix.core.runtime.size_policy import resolve_size

WHERE = "/cache/M/runtime/defaults.json"


@pytest.mark.parametrize("defaults", [None, {}, {"height": 480, "width": 720}, {"num_frames": 49}])
def test_an_absent_policy_is_refused_naming_the_key_and_the_file(defaults):
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: size_policy is not declared in /cache/M/runtime/defaults"):
        resolve_size(defaults, (512, 512), None, None, where=WHERE)


@pytest.mark.parametrize("policy", ["auto", "image", "FIXED", 1, True])
def test_an_unknown_policy_is_refused_by_name(policy):
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: size_policy = .* is not a size policy"):
        resolve_size({"size_policy": policy, "height": 480, "width": 720}, (512, 512), None, None, where=WHERE)


@pytest.mark.parametrize("defaults", [
    {"size_policy": "fixed", "height": 480, "width": 720, "max_area": 345600},
    {"size_policy": "fixed", "height": 480, "width": 720, "size_multiple": 16},
    {"size_policy": "from_image", "max_area": 399360, "size_multiple": 16, "height": 480},
    {"size_policy": "from_image", "max_area": 399360, "size_multiple": 16, "width": 832},
])
def test_both_forms_at_once_are_refused(defaults):
    with pytest.raises(RuntimeError, match="one form, not both"):
        resolve_size(defaults, (512, 512), None, None, where=WHERE)


@pytest.mark.parametrize("defaults,key", [
    ({"size_policy": "fixed", "height": 480}, "width"),
    ({"size_policy": "fixed", "width": 720}, "height"),
    ({"size_policy": "from_image", "size_multiple": 16}, "max_area"),
    ({"size_policy": "from_image", "max_area": 399360}, "size_multiple"),
])
def test_a_form_with_a_key_missing_is_refused_naming_the_key(defaults, key):
    with pytest.raises(RuntimeError, match=f"needs {key}, which is not declared in {WHERE}"):
        resolve_size(defaults, (512, 512), None, None, where=WHERE)


@pytest.mark.parametrize("bad", [0, -16, 16.0, "16", True, None])
def test_a_key_that_is_not_a_positive_integer_is_refused(bad):
    defaults = {"size_policy": "from_image", "max_area": 399360, "size_multiple": bad}
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: size_policy = 'from_image' needs size_multiple|"
                                           "ZERO FALLBACK: size_multiple = .* is not a positive integer"):
        resolve_size(defaults, (512, 512), None, None, where=WHERE)


def test_a_container_without_the_policy_is_told_its_remedy():
    """The refusal names the remedy as well as the fault: the corrected container, in the engine's own wording for
    incomplete model data (core/module/cache/factory.py). Red if the text names only the missing key."""
    with pytest.raises(RuntimeError, match=r"predates the declared size policy: install the corrected container "
                                           r"\(neurobrix remove <model> && neurobrix import <org>/<model>\)"):
        resolve_size({}, (512, 512), None, None, where=WHERE)
