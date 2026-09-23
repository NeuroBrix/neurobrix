"""A graph constant is resident beside every segment, and nothing counted it.

`_try_layer_streaming` computed `segment_budget = budget_bytes - resident_beside`, where
`resident_beside` sums COMPONENTS. A constant baked into `graph.json` is not a component. It
is also the FIRST thing an executor allocates — `_load_constants_from_graph` runs at
construction, before a single weight — so the budget was blind to exactly the bytes that are
already gone when the first segment asks for its own.

WHAT IT COST, measured 2026-09-22 on a 16 GB V100 (card 1, `default-ff6008b7`)
----------------------------------------------------------------------------
DeepSeek-Coder-V2-Lite-Instruct, `NBX_FORCE_STRATEGY=layer_streaming`, `--max-tokens 8`:

* `NBX_PHASE_TRACE=1` (extended here so the two stamps raised inside `_create_session` carry
  the allocator baseline the other three already carried): `live=2160MB` at
  `flow.session.weights_ensured` — the FIRST stamp. The KV cache allocates nothing; the
  residency is already there before the session exists.
* `NBX_MALLOC_TRACE`, the engine's own site recorder, names every byte of it:
  `108 blocks x 20 971 520 B = 2160 MB  graph_executor.py:2432 _load_constant_triton`.
* The container says which: 54 tensors declared `[163840, 64]` bfloat16 —
  `block.{0..26}.attn.rotary_embed.{cos,sin}_cached`. 163 840 is `max_position_embeddings`,
  and `163840 * 64 * 2 == 20971520` exactly. The RoPE tables are materialised at the model's
  maximum context for a request of eight tokens.
* The partition cut against 15 539 MB; execution offered 13 680 MB; the segment asked
  13 966 MB and missed by 286 MB. Four MoE models died this way, every one at
  `live_tracked=2160MB`.

TWO DEFECTS, FIXED SEPARATELY, AND WHY BOTH WERE NEEDED
-------------------------------------------------------
54 distinct constants, **108 live blocks**. Every executor loads the constants of its own
graph, so the three segment executors hold the whole set between them — and the base executor
holds a second complete copy while its `run` is about to be replaced by `segmented_run` and
never executes another op. Half the residency was a copy of the other half.

Subtracting the constants from the budget alone still would not have run: budget 16 151 −
845 (lm_head) − 1 080 (constants) = 14 226 MB against 13 680 MB free. Releasing the base copy
alone would not have run either: the partition would still cut 15 539 MB segments. Measured
after both, same command, same card: `free=14760MB`, `54x20.00MB=1080MB`, and segment 0
executed — the first time this rung has run on this rack.

The cells below assert the INVARIANTS, not the numbers: a budget that reserves what the
executor will allocate, and a base executor that holds nothing the segments already hold.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from neurobrix.core.prism.solver import _graph_constant_bytes

CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
MODEL = "DeepSeek-Coder-V2-Lite-Instruct"


def _graph(model: str = MODEL, component: str = "model"):
    p = CACHE / model / "components" / component / "graph.json"
    if not p.exists():
        pytest.skip(f"{model}/{component} is not in this cache")
    return json.loads(p.read_text())


# ───────────────────── the measurement, against the container ─────────────────────

def test_the_constants_are_measured_and_they_are_not_small():
    """The figure the budget now subtracts, read from the container itself."""
    n = _graph_constant_bytes(_graph())
    assert n > 0, "no constants measured — the budget would be unchanged and the fix inert"
    assert n / 2**20 > 512, (
        f"{n / 2**20:.0f} MB of constants; the recorded measurement was 1080 MB. A figure "
        f"this much smaller means the tensor records no longer mark constants the same way.")


def test_the_measurement_matches_the_blocks_the_allocator_actually_took():
    """`NBX_MALLOC_TRACE` recorded 54 blocks of 20 971 520 B for this graph. The budget's
    arithmetic must land on the same bytes, or it reserves a number for a different thing."""
    g = _graph()
    rope = [t for t in (g.get("tensors") or {}).values()
            if isinstance(t, dict) and list(t.get("shape") or []) == [163840, 64]]
    if not rope:
        pytest.skip("this container no longer carries the [163840, 64] constants")
    assert len(rope) == 54
    assert _graph_constant_bytes(g) >= len(rope) * 163840 * 64 * 2


# ───────────────────── the refusals that keep the figure honest ─────────────────────

def test_an_unknown_constant_dtype_is_REFUSED_not_assumed():
    """ZERO FALLBACK. A dtype assumed at 4 bytes under-reserves by exactly the amount that
    matters, and silently — the failure mode this whole file exists to remove."""
    with pytest.raises(ValueError, match="unknown constant dtype"):
        _graph_constant_bytes({"tensors": {"t": {
            "constant": True, "constant_data": "eA==", "shape": [4, 4],
            "dtype": "float8_e4m3fn"}}})


def test_a_computable_buffer_is_NOT_counted():
    """The loader skips its `constant_data` and recomputes it at runtime resolution, so its
    traced size is not what it occupies. Counting it would over-reserve."""
    t = {"constant": True, "constant_data": "eA==", "shape": [1024, 1024],
         "dtype": "float32", "is_computable": True, "weight_name": "w",
         "computation_spec": {}}
    assert _graph_constant_bytes({"tensors": {"t": t}}) == 0


def test_a_symbolic_dim_is_skipped_rather_than_guessed():
    assert _graph_constant_bytes({"tensors": {"t": {
        "constant": True, "constant_data": "eA==", "shape": ["s0", 64],
        "dtype": "float16"}}}) == 0


def test_the_narrowing_the_loader_performs_is_mirrored():
    """`_load_constant_triton` narrows fp64 to fp32 and complex128 to complex64 on the way in
    (the kernels are fp32-max). Counting the declared width would over-reserve by half."""
    mk = lambda dt: {"tensors": {"t": {"constant": True, "constant_data": "eA==",
                                       "shape": [1024], "dtype": dt}}}
    assert _graph_constant_bytes(mk("float64")) == _graph_constant_bytes(mk("float32"))
    assert _graph_constant_bytes(mk("complex128")) == _graph_constant_bytes(mk("complex64"))


def test_a_graph_without_constants_reserves_nothing():
    """The pass must be INERT where there is nothing to reserve, or every other model's
    budget shrinks for no reason."""
    assert _graph_constant_bytes({"tensors": {"t": {"shape": [8, 8], "dtype": "float16"}}}) == 0
    assert _graph_constant_bytes(None) == 0
    assert _graph_constant_bytes({}) == 0
