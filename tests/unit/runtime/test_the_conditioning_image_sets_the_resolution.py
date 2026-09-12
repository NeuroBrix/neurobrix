"""When a request supplies a conditioning image, the image sets the resolution.

Two decisions were being taken separately about one quantity. The image
processor keeps the source image's own size when no `--height/--width` is given
— its own comment says so — while the resolution cascade independently derived
the size from the TRACED latent. On 2026-09-12 Allegro-TI2V met its own 448x448
conditioning image at a pipeline running 144x208 and died on *"Expected size 18
but got size 56"*: the traced latent extent against the image's.

The order this pins, and it is the whole answer:

    explicit --height/--width          request-side, and explicit
    the container's declared defaults  build-side, and deliberate
    the conditioning image             request-side, and specific
    the traced latent                  a stimulus, not an intention
    the family constant                last resort

A request-side fact outranks a build-side one, and a stimulus chosen at trace
time is the last thing that should decide what a user gets.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_the_conditioning_image_sets_the_resolution.py
"""
from __future__ import annotations

import types

import numpy as np

from neurobrix.core.runtime.executor import RuntimeExecutor


def _ex(defaults=None, topology=None):
    ex = object.__new__(RuntimeExecutor)
    ex.pkg = types.SimpleNamespace(
        topology=topology or {"version": "0.1", "flow": {"type": "iterative_process"},
                              "components": {}},
        manifest={"vae_scale_factor": 8, "family": "video"},
        defaults=defaults or {},
        components={})
    return ex


def test_the_last_two_axes_are_the_spatial_ones():
    """Holds for [C,H,W], [C,T,H,W] and [B,C,T,H,W] alike."""
    ex = _ex()
    for shape in ((3, 448, 448), (3, 8, 144, 208), (1, 3, 8, 144, 208)):
        got = ex._conditioning_image_size({"global.image": np.zeros(shape, dtype=np.uint8)})
        assert got == (shape[-2], shape[-1]), shape


def test_no_image_is_silence_not_a_guess():
    """Silence falls through to the container, which is the next authority."""
    assert _ex()._conditioning_image_size({}) is None


def test_a_non_spatial_array_is_silence():
    ex = _ex()
    assert ex._conditioning_image_size({"global.image": np.zeros((7,))}) is None


def test_a_zero_extent_is_silence():
    """A degenerate array must not become a resolution of zero."""
    ex = _ex()
    assert ex._conditioning_image_size(
        {"global.image": np.zeros((3, 0, 16), dtype=np.uint8)}) is None


def test_the_image_outranks_the_traced_latent():
    """The measured Allegro case: a 448x448 image against a 18x26 traced latent.

    Without this the pipeline runs at 144x208 while the processor hands it a
    448x448 tensor, and the two meet at a broadcast.
    """
    topology = {"version": "0.1", "flow": {"type": "iterative_process"},
                "components": {"transformer": {"shapes":
                                               {"hidden_states": [3, 12, 7, 18, 26]}}}}
    ex = _ex(topology=topology)
    # The container alone would answer 18*8 x 26*8.
    assert ex._container_output_size({}) == (144, 208)
    merged = ex._prepare_defaults(
        {"global.image": np.zeros((3, 8, 448, 448), dtype=np.uint8)})
    assert (merged["height"], merged["width"]) == (448, 448)


def test_an_explicit_request_still_wins_over_the_image():
    """The image is specific; a flag is explicit, and explicit outranks specific."""
    ex = _ex()
    merged = ex._prepare_defaults({
        "global.image": np.zeros((3, 8, 448, 448), dtype=np.uint8),
        "global.height": 144, "global.width": 208})
    assert (merged["height"], merged["width"]) == (144, 208)


def test_a_container_that_declares_its_size_is_not_overridden():
    """A declared default is a build-side decision; this branch never runs."""
    ex = _ex(defaults={"height": 720, "width": 1280})
    merged = ex._prepare_defaults(
        {"global.image": np.zeros((3, 8, 448, 448), dtype=np.uint8)})
    assert (merged["height"], merged["width"]) == (720, 1280)
