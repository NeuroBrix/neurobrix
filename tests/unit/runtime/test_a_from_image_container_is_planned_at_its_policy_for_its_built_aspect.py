"""A daemon plans a "from_image" container at load, before any image, at what its policy resolves for the
container's own built aspect — the likeliest request (for an image-to-video container built at its vendor's default
size, exactly that size), not its built latent's pixel count and not a bound (project lead 23/11/2026).

Red if `_container_extent` keeps the built latent (96, 160) of a container whose budget is larger."""
from neurobrix.core.runtime.size_policy import from_image_size, plan_size
from neurobrix.serving import engine


def test_a_container_built_under_its_budget_is_planned_at_its_policy(monkeypatch):
    from neurobrix.core.runtime.resolution import container_size
    monkeypatch.setattr(container_size, "container_output_size", lambda *a: (96, 160))
    monkeypatch.setattr(container_size, "vae_scale_factor", lambda *a: 8)
    defaults = {"size_policy": "from_image", "max_area": 40000, "size_multiple": 16}
    h, w, scale = engine._container_extent({}, defaults, {})
    assert (h, w) == from_image_size(40000, 16, (96, 160)) == (144, 256) and scale == 8


def test_a_container_built_at_its_vendor_default_is_planned_at_it():
    assert plan_size({"size_policy": "from_image", "max_area": 399360, "size_multiple": 16}, (480, 832)) == (480, 832)


def test_a_container_of_another_policy_keeps_its_declared_or_built_size():
    assert plan_size({"size_policy": "fixed", "height": 480, "width": 720}, (480, 720)) is None
    assert plan_size({}, (96, 160)) is None
