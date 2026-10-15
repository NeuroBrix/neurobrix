"""A request the denoising loop cannot keep is refused by name before anything runs (core/prism/loop_state.py).

The Mac, 2026-10-15 05:30: Allegro at 360x640 (latent height 45) ran its first step and died in the scheduler, "the
size of tensor a (45) must match the size of tensor b (44)": the transformer patchifies with a stride-2 convolution,
its output height is ((h - 2) // 2 + 1) * 2, which is h only when h is even. Read from the graph: the state input's
dims and the output's, at the request's own binding.

What this test does if the code is wrong: the refusal cell fails when an odd latent height is accepted (red on main:
nothing checks it) or when the refusal does not name the neighbours the loop keeps; the accept cell fails when a size
the loop keeps is refused; the synthetic cells fail when the rule reads anything but the graph (a loop that keeps every
size is never refused; an output that shrinks by one is).

Run: python -m pytest tests/unit/prism/test_a_request_the_loop_cannot_keep_is_refused.py
"""
import pytest

from tests.unit.prism._pinned_machine import container_root


def _input_config(model, h, w):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.nbx import NBXContainer
    c = NBXContainer.load(str(container_root(model)))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", model, "--prompt", "apple", "--seed", "42", "--steps", "2",
                                       "--height", str(h), "--width", str(w)])
    return request_input_config(args, man, man.get("family"), c.cache_path)


def test_an_odd_latent_height_is_refused_with_the_sizes_the_loop_keeps():
    with pytest.raises(ValueError, match=r"height \(s2, axis 3\): the state is 45 and 'transformer' returns 44; "
                                         r"the loop keeps 44 x8 = 352 below and 46 x8 = 368 above"):
        _input_config("Allegro", 360, 640)


def test_a_size_the_loop_keeps_is_accepted():
    assert _input_config("Allegro", 368, 640) is not None


S = lambda sid: {"type": "symbol", "id": sid}  # noqa: E731


def _graph(out_h):
    return {"tensors": {"input::x": {"symbolic_shape": {"dims": [1, 4, S("h")]}},
                        "o": {"symbolic_shape": {"dims": [1, 4, out_h]}}},
            "output_tensor_ids": ["o"]}


def test_a_loop_that_keeps_every_size_is_never_refused():
    from neurobrix.core.prism.loop_state import mismatches
    g = _graph(S("h"))
    assert all(mismatches(g, "x", {"h": v}) == [] for v in range(1, 200))


def test_an_output_that_loses_a_row_is_named():
    from neurobrix.core.prism.loop_state import kept_neighbours, mismatches
    g = _graph({"type": "mul", "left": {"type": "floordiv", "left": S("h"), "right": 3}, "right": 3})
    assert mismatches(g, "x", {"h": 10}) == [(2, "h", 10, 9)]
    assert kept_neighbours(g, "x", {"h": 10}, "h") == (9, 12)
