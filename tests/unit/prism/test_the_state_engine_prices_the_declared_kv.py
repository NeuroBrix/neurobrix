"""Step 1 of the state engine (docs/design/state-engine.md, validated 2026-10-15): the KV a container declares is priced by
one pure function (core/state/price.py), and Prism's KV check takes the decoder's cache out of the decoder's own figures
and sizes it on the decoder's own cards.

What this test does if the code is wrong: the price cells fail when the bytes are not layers x kv heads x (k + v) x width x
sequences, when a cache exceeds the declared window, when a decoding flow without a declaration is not refused by name;
the check cells fail when the KV estimate is taken off another component's figures (the MAX defect) or when the decoder's
cards are not the only ones the cache is sized on (the sum-of-rungs defect).
"""
import pytest

from neurobrix.core.state.price import (StateUndeclared, kv_cache_size, kv_declaration, kv_min_tokens,
                                        kv_token_bytes)

KV = {"kind": "kv", "layers": 22, "kv_heads": 4, "k_dim": 64, "v_dim": 64, "dtype": "bfloat16", "window": 2048}
TOPO = {"components": {"model": {}}, "flow": {"type": "autoregressive_generation",
                                               "generation": {"lm_component": "model"}},
        "state": {"model": [KV]}}


def test_the_bytes_are_the_declared_geometry():
    assert kv_token_bytes(KV, "float16", 1) == 22 * 4 * 128 * 2
    assert kv_token_bytes(KV, "float32", 2) == 22 * 4 * 128 * 4 * 2


def test_a_decoding_flow_without_a_declaration_is_refused_and_a_non_decoding_one_is_none():
    assert kv_declaration(TOPO, ["model"])[0] == "model"
    with pytest.raises(StateUndeclared, match="declares no KV state"):
        kv_declaration({**TOPO, "state": {}}, ["model"])
    assert kv_declaration({"components": {"transformer": {}}, "flow": {"type": "iterative_process"}},
                          ["transformer"]) is None


def test_the_cache_never_exceeds_the_declared_window():
    tb = kv_token_bytes(KV, "float16", 1)
    serve = kv_cache_size(KV, tb, 256, 128, True, 10 ** 12)
    assert serve["max_cache_len"] == 2048 and serve["initial_cache_len"] == 384
    run = kv_cache_size(KV, tb, 4096, 128, False, 10 ** 12)
    assert run["max_cache_len"] == 2048, "a request longer than the window is held to the window"
    with pytest.raises(RuntimeError, match="KV cache budget too small"):
        kv_cache_size(KV, tb, 256, 128, False, tb * 10)
    assert kv_min_tokens(256, 128, True) == 640 and kv_min_tokens(0, 128, False) == 64


class _M:
    def __init__(self, w, a, o=0):
        self.weight_bytes, self.activation_bytes, self.overhead_bytes = w, a, o

    @property
    def total_bytes(self):
        return self.weight_bytes + self.activation_bytes + self.overhead_bytes


def test_the_kv_estimate_comes_out_of_the_decoders_own_figures():
    from neurobrix.core.prism.solver import PrismSolver
    s = PrismSolver()
    s._lm_component_name = "lm"
    mem = {"lm": _M(1000, 600), "vision": _M(1500, 300)}
    out = s._kv_stripped(mem, 500)
    assert out["lm"].total_bytes == 1100 and out["vision"].total_bytes == 1800
    assert mem["lm"].activation_bytes == 600, "the plan's own figures are untouched"
    # the MAX over components is the vision tower's 1 800 — taken off the MAX, the estimate came off it (1 300)
    assert max(m.total_bytes for m in out.values()) == 1800


def test_the_cache_is_sized_on_the_decoders_cards_only():
    from neurobrix.core.prism.solver import PrismSolver
    s = PrismSolver()
    s._lm_component_name = "lm"
    allocs = {"lm": ("cuda:1", {"block.0": "cuda:1", "block.1": "cuda:2"}), "vae": "cuda:0", "enc": "zero3:cuda:3"}
    assert s._kv_devices(allocs) == {"cuda:1", "cuda:2"}
    assert PrismSolver._alloc_devices("zero3:cuda:3") == {"cuda:3"}
    s._lm_component_name = None
    assert s._kv_devices(allocs) is None
