"""A name that promises an answer must not deliver a guess.

`InputConfig.to_symbol_map()` returned the POSITIONAL base — `s0` batch, `s1`
latent height, `s2` latent width — under a name that reads like "the symbol map".
It is the image legacy, and it is wrong for every graph that declares otherwise:
a video container says `s1: time` in its `symbolic_context`, and bound
positionally the time axis takes a spatial extent.

The cost, twice:

* **2026-08-10** — Qwen3-Omni's `thinker.audio_tower` mel-frame axis, named
  `seq_len`, bound to the global text config (~128) instead of its 441-frame
  trace. The activation estimate collapsed and `block_scatter` packed a 16 GB
  card to 15.77 GiB with no forward headroom.
* **2026-09-12** — four instruments built on it in one day, each binding `s1` to
  a latent height on containers declaring `s1: time`. Each produced a confident
  census; one was reported upstream as an established fact ("22 components across
  12 containers, every video container in the zoo") before the instrument was
  confronted with a measurement taken another way. The true count is one
  container, one component, one dimension.

**The warning was already written, in `build_symbol_map`'s own docstring — the
function that was not called.** That is why this is a refusal and not a note: a
note in the function you did not call cannot reach you, and the same note had
already failed once.

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_symbol_map_names_its_hypothesis.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.profiler import ActivationProfiler, InputConfig


REQUEST = InputConfig(batch_size=1, height=512, width=512, num_frames=17)

# A video graph, as the live containers declare themselves: the symbol table
# says `s1` is TIME, and the positional guess says it is a latent height.
VIDEO_GRAPH = {
    "execution_order": [],
    "ops": {},
    "tensors": {
        "input::args": {
            "tensor_id": "input::args", "shape": [1, 3, 9, 112, 176],
            "dtype": "float32", "is_input": True, "input_name": "args",
        },
    },
    "symbolic_context": {
        "symbols": {
            "s0": {"name": "batch", "trace_value": 1, "source": "input::args::dim_0"},
            "s1": {"name": "time", "trace_value": 9, "source": "input::args::dim_2"},
            "s2": {"name": "height", "trace_value": 112, "source": "input::args::dim_3"},
            "s3": {"name": "width", "trace_value": 176, "source": "input::args::dim_4"},
        },
        "expressions": {},
    },
}


def test_the_promising_name_is_refused():
    with pytest.raises(AttributeError) as exc:
        REQUEST.to_symbol_map()
    why = str(exc.value)
    # The refusal must say what the old name did, or it is just an error.
    assert "POSITIONAL GUESS" in why
    # And name BOTH ways forward, so the caller is not left to search.
    assert "build_symbol_map" in why
    assert "positional_symbol_map" in why


def test_the_guess_is_still_available_under_a_name_that_says_so():
    """Not removed — renamed. `build_symbol_map` is built on it."""
    base = REQUEST.positional_symbol_map()
    assert base["s1"] == 512 // 8 and base["s2"] == 512 // 8


def test_the_declared_map_binds_time_to_time():
    """The measurement the refusal exists to force.

    The same container, the same request, the two maps — and the difference is
    a factor of 12 on one axis, silently.
    """
    prof = ActivationProfiler(VIDEO_GRAPH)
    guess = REQUEST.positional_symbol_map()
    per_request = prof.build_symbol_map(REQUEST)
    placement = prof.build_symbol_map(REQUEST, placement_floor=True)

    # 17 frames at temporal compression 4 -> (17-1)//4 + 1 = 5 latent frames.
    assert per_request["s1"] == 5, "time binds to the latent frame count"
    # The placement estimate never binds a named symbol BELOW its witnessed
    # trace (9 here), which is the 2026-08-10 floor and not a disagreement.
    assert placement["s1"] == 9
    # The guess binds the same axis to a latent HEIGHT — a factor of 12 away
    # from the per-request answer, silently.
    assert guess["s1"] == 64
    # And the guess's `s3` is a PRODUCT (latent_h x latent_w), where the
    # declared table says it is a width. Nothing in the positional map is
    # checked against the table it contradicts.
    assert guess["s3"] == 64 * 64
    assert placement["s3"] == 176, "width keeps its witnessed trace under the floor"


def test_an_unnamed_graph_still_gets_the_positional_base():
    """R23: a legacy graph with no symbol table is untouched by all of this."""
    prof = ActivationProfiler({"execution_order": [], "ops": {}, "tensors": {}})
    assert prof.build_symbol_map(REQUEST) == REQUEST.positional_symbol_map()
