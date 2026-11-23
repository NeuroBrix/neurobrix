"""A run with a conditioning image generates at the size its container's size policy gives (`size_policy.resolve_size`),
never at the image's own size: the executor's `_conditioning_image_size` branch (the image set the resolution when the
container declared none) is gone. The image processor hands the size it fitted the image to as global.height /
global.width, which the executor applies; an image that arrives without them is sized by the same reader. Without a
conditioning image the path is unchanged: the container's declared size, else its built output extent.

Replaces test_the_conditioning_image_sets_the_resolution.py, which asserted the withdrawn order (a 448x448 image
outranking the built latent and setting the run at 448x448).

What would turn it red: a 512x512 image run at 512x512 under a default 480x720 policy, an image accepted from a container
declaring no size_policy, the processor's size overridden, or a run without an image newly refused for lack of a policy
(the VACE all-generate pair carries its size and no conditioning image)."""
from __future__ import annotations

import types

import numpy as np
import pytest

from neurobrix.core.runtime.executor import RuntimeExecutor


def _ex(defaults=None, topology=None):
    ex = object.__new__(RuntimeExecutor)
    ex.pkg = types.SimpleNamespace(
        topology=topology or {"version": "0.1", "flow": {"type": "iterative_process"}, "components": {}},
        manifest={"vae_scale_factor": 8, "family": "video"},
        defaults=defaults or {},
        components={})
    return ex


def _hw(merged):
    return merged["height"], merged["width"]


def test_a_default_policy_is_the_run_size_not_the_images():
    ex = _ex({"size_policy": "default", "height": 480, "width": 720})
    assert _hw(ex._prepare_defaults({"global.image": np.zeros((1, 3, 1, 512, 512), dtype=np.float32)})) == (480, 720)


def test_a_from_image_policy_sizes_an_image_that_arrives_without_a_size():
    ex = _ex({"size_policy": "from_image", "max_area": 399360, "size_multiple": 16})
    assert _hw(ex._prepare_defaults({"global.image": np.zeros((1, 3, 1, 480, 832), dtype=np.float32)})) == (480, 832)


def test_the_size_the_processor_fitted_the_image_to_is_the_run_size():
    ex = _ex({"size_policy": "from_image", "max_area": 921600, "size_multiple": 16})
    merged = ex._prepare_defaults({"global.image": np.zeros((1, 3, 1, 960, 960), dtype=np.float32),
                                   "global.height": 960, "global.width": 960})
    assert _hw(merged) == (960, 960)


@pytest.mark.parametrize("defaults", [{}, {"num_frames": 49}])
def test_an_image_from_a_container_without_a_policy_is_refused_by_name(defaults):
    ex = _ex(defaults)
    with pytest.raises(RuntimeError, match="ZERO FALLBACK: size_policy is not declared in the container's runtime"):
        ex._prepare_defaults({"global.image": np.zeros((1, 3, 1, 448, 448), dtype=np.float32)})


def test_without_a_conditioning_image_the_built_latent_still_sizes_the_run():
    topology = {"version": "0.1", "flow": {"type": "iterative_process"},
                "components": {"transformer": {"shapes": {"hidden_states": [3, 12, 7, 18, 26]}}}}
    ex = _ex(topology=topology)
    assert _hw(ex._prepare_defaults({})) == (144, 208)


def test_without_a_conditioning_image_the_declared_size_and_the_request_are_unchanged():
    ex = _ex({"height": 720, "width": 1280})
    assert _hw(ex._prepare_defaults({})) == (720, 1280)
    assert _hw(ex._prepare_defaults({"global.height": 144, "global.width": 208})) == (144, 208)


def test_a_pair_that_carries_its_size_needs_no_policy():
    """VACE all-generate: the control pair rides on global.image with the run's resolved size and no image."""
    ex = _ex({"height": 480, "width": 832})
    merged = ex._prepare_defaults({"global.image": np.zeros((2, 3, 81, 480, 832), dtype=np.float32),
                                   "global.height": 480, "global.width": 832})
    assert _hw(merged) == (480, 832)
