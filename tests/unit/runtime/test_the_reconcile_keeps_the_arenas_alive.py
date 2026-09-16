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
