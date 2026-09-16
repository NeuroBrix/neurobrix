"""The weight-key reconcile rebuilds the component's weight dict by the
binding; a key the binding does not name must keep its own name — above all
`_arenas`, the Triton loader's device blocks that every weight tensor points
INTO. When the rebuilt dict dropped it, the dict was the arenas' last
reference: the blocks were freed under the weights and the first kernel to
read one met a freed address (Ming-Lite-Omni triton, 2026-09-16: the door
refused `embedding_kernel`'s weight_ptr inside a 14 GB block freed at malloc
event 295 of 426, named by tools/address_origin.py).

Injection: with the carry-over removed, `_arenas` vanished and the sentinel's
finalizer ran — the first test was RED."""
from __future__ import annotations

import gc
from types import SimpleNamespace

from neurobrix.core.runtime.graph_executor import GraphExecutor


class _Arena:
    freed = []

    def __del__(self):
        _Arena.freed.append(id(self))


def _stub(weights, params):
    return SimpleNamespace(
        _dag={"ops": []}, _weights=weights, _pending_weight_binding=None,
        _graph_param_names=lambda: set(params),
        bind_weight_keys=GraphExecutor.bind_weight_keys,
        _reconcile_weight_keys=GraphExecutor._reconcile_weight_keys,
    )


def test_the_arenas_survive_a_rebinding_reconcile():
    _Arena.freed.clear()
    arena = _Arena()
    w = {"language_model.model.block.0.attn.key.weight": "t0", "_arenas": {2: arena}}
    stub = _stub(w, {"model.language_model.block.0.attn.key.weight"})
    stub._reconcile_weight_keys(stub)
    arena_id = id(arena)
    del arena
    gc.collect()
    assert "model.language_model.block.0.attn.key.weight" in stub._weights, stub._weights.keys()
    assert "_arenas" in stub._weights, "the arenas' last reference was dropped by the reconcile"
    assert arena_id not in _Arena.freed


def test_a_dict_that_needs_no_binding_is_left_alone():
    w = {"block.0.attn.key.weight": "t0", "_arenas": {}}
    stub = _stub(dict(w), {"block.0.attn.key.weight"})
    stub._reconcile_weight_keys(stub)
    assert stub._weights == w


def test_a_second_load_joins_its_arenas_instead_of_replacing_the_first():
    """The declared-MoE fusion loads the weights it adds AFTER the first load
    (`_load_what_the_rewrite_added`, `only=`); the second load's `_arenas`
    used to REPLACE the first's in the dict — the first arenas' last
    reference, so the blocks were freed under the live weights (Ming triton,
    2026-09-16: the malloc trace named `_load_weights_triton` as the freer).
    Injection: with `_join_arenas` replaced by the bare `update`, `first` was
    finalized — RED."""
    _Arena.freed.clear()
    first, second = _Arena(), _Arena()
    held = {"w0": "t0", "_arenas": {2: first}}
    loaded = {"w1": "t1", "_arenas": {2: second}}
    held.update(GraphExecutor._join_arenas(held, loaded))
    first_id = id(first)
    del first, second, loaded
    gc.collect()
    assert first_id not in _Arena.freed, "the first load's arena was freed under its weights"
    assert set(held) == {"w0", "w1", "_arenas"}
    assert len(held["_arenas"]) == 2 and {a.device_idx for a in held["_arenas"].values() if hasattr(a, "device_idx")} == set()


def test_a_load_without_arenas_joins_nothing():
    held = {"w0": "t0", "_arenas": {2: object()}}
    out = GraphExecutor._join_arenas(held, {"w1": "t1"})
    assert out == {"w1": "t1"}


def test_an_empty_arenas_dict_from_the_second_load_does_not_replace_the_first():
    """The fusion's added weights may put nothing on a device: `_arenas: {}`.
    The first wiring returned `loaded` untouched when the new dict was empty,
    and `update` replaced the first arenas with `{}` — found by the same malloc
    trace on the proof run (2026-09-16 12:45). Injection: the empty-dict
    shortcut restored → RED."""
    _Arena.freed.clear()
    first = _Arena()
    held = {"w0": "t0", "_arenas": {2: first}}
    held.update(GraphExecutor._join_arenas(held, {"w1": "t1", "_arenas": {}}))
    first_id = id(first)
    del first
    gc.collect()
    assert first_id not in _Arena.freed and len(held["_arenas"]) == 1
