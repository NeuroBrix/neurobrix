"""The solver names the allocator's reserve a decode cache may not grow into, and its plans keep it.

`PrismSolver._kv_growth_reserve_bytes` is the solver's half of the rule `price.kv_cache_size` applies: the reserve is
the allocator's (`_margin_mb`) on every card whose capacity is the raw rung — the compiled engine on a card no
residency is granted on after the cache — and 0 where `_kv_capacity_bytes` already holds the band back (a bounded
card; the Triton engine). The cache plan built with it grows a run's cache, a served window and the session reach it
states only above the reserve, each floor still granted from the whole room.

What this test does if the code is wrong: the three engine cells fail when the reserve is no longer named for the
compiled raw rung, or is charged twice where the capacity already holds it (bounded, Triton); the plan cells fail
when `_compute_kv_cache_plan` does not carry the reserve to the price — a run's cache, a served window or the session
reach then takes the band again (the 7B decoder held lazy on a 16 GB card: 8 724 positions, 477 MiB, plan + cache = the
budget, out of memory in its first prefill, 2026-11-15) — or when a floor is refused for the reserve. Red with
solver.py at its base form: no such method, no such argument.
"""
from types import SimpleNamespace

import pytest

from neurobrix.core.prism import PrismSolver
from neurobrix.core.state.price import DEFAULT_PROMPT_MARGIN, kv_min_tokens
from tests.unit.prism._pinned_machine import (V100_16GB, container_root, no_door, pin_dedicated_card, pin_no_calibration,
                                              profile)

MIB = 1024 * 1024
RESERVE_MB = 3072
VENDOR_MB = 1024
RUNGS = {"cuda:0": 15845.0, "cuda:1": 32195.0}
KV = {"kind": "kv", "layers": 28, "kv_heads": 4, "k_dim": 128, "v_dim": 128, "window": 32768}
TOKEN = 28 * 4 * 256 * 2
PACKAGE = {"max_tokens": 8192, "prompt_margin": 532}
MODEL = "Qwen2.5-Coder-7B-Instruct"
CARDS = [SimpleNamespace(device_string="cuda:0"), SimpleNamespace(device_string="cuda:1")]


def _solver(mode, vendor_mb=None, serve=False, asked=5):
    s = PrismSolver.__new__(PrismSolver)
    s._mode = mode
    s.oom_reserve_mb = RESERVE_MB
    s._vendor_plan_reserve_mb = vendor_mb
    s._effective_capacity_mb = lambda d: RUNGS[d.device_string]
    s._serve_mode = serve
    s._kv_declared = lambda container: ("language_model", KV)
    s._read_defaults = lambda container: dict(PACKAGE)
    s._kv_token_bytes = lambda container, dtype: TOKEN
    s._request_max_tokens = lambda defaults: defaults["max_tokens"] if serve else asked
    s._prefill_chunk = lambda *a: (None, "not priced in this cell")
    return s


def test_the_compiled_engine_names_the_reserve_of_every_raw_rung_card():
    assert _solver("compiled")._kv_growth_reserve_bytes(CARDS[:1], set()) == RESERVE_MB * MIB
    assert _solver("compiled")._kv_growth_reserve_bytes(CARDS, set()) == 2 * RESERVE_MB * MIB, "one band a card"
    assert _solver("compiled", VENDOR_MB)._kv_growth_reserve_bytes(CARDS[:1], set()) == VENDOR_MB * MIB, \
        "the vendor file's measured reserve is the band (`_margin_mb`), never a second figure"


def test_a_bounded_card_holds_the_band_already():
    s = _solver("compiled")
    assert s._kv_growth_reserve_bytes(CARDS[:1], {"cuda:0"}) == 0, "its capacity is the usable part"
    assert s._kv_growth_reserve_bytes(CARDS, {"cuda:1"}) == RESERVE_MB * MIB, "only the raw-rung card pays"


def test_the_triton_engine_prices_its_growth_in_the_capacity():
    assert _solver("triton")._kv_growth_reserve_bytes(CARDS, set()) == 0
    assert _solver("triton_sequential")._kv_growth_reserve_bytes(CARDS, set()) == 0


def test_a_run_plan_grows_its_cache_only_above_the_reserve():
    s, reserve = _solver("compiled"), RESERVE_MB * MIB
    floor = 5 + PACKAGE["prompt_margin"]
    tight = s._compute_kv_cache_plan(None, "float16", 477 * MIB, growth_reserve_bytes=reserve)
    assert tight.max_cache_len == floor, "inside the reserve the cache is its request's positions, not the package's"
    assert tight.memory_bytes == floor * TOKEN
    roomy = s._compute_kv_cache_plan(None, "float16", reserve + 100 * MIB, growth_reserve_bytes=reserve)
    assert roomy.max_cache_len == (100 * MIB) // TOKEN, "the growth takes the room above the reserve"
    whole = s._compute_kv_cache_plan(None, "float16", 500 * MIB)
    assert whole.max_cache_len == PACKAGE["max_tokens"] + PACKAGE["prompt_margin"], "no reserve named: the package bound"
    with pytest.raises(RuntimeError, match="KV cache budget too small"):
        s._compute_kv_cache_plan(None, "float16", (floor - 1) * TOKEN, growth_reserve_bytes=reserve)


def test_a_served_window_and_the_session_reach_keep_the_same_reserve():
    """A served cache is allocated beside the same weights and prefills through the same allocator: the window that
    took the band is the run's defect at serve size. The reach a run plan states is what a served cache would be
    granted in the same room, so it keeps the reserve too — a larger reach would promise positions a served session on
    this card cannot hold."""
    reserve = RESERVE_MB * MIB
    session_floor = kv_min_tokens(PACKAGE["max_tokens"], PACKAGE["prompt_margin"], True)
    room = reserve + session_floor * TOKEN + 100 * MIB
    above = (room - reserve) // TOKEN
    served = _solver("compiled", serve=True)._compute_kv_cache_plan(None, "float16", room, growth_reserve_bytes=reserve)
    assert served.max_cache_len == above == served.session_reach, "the served window grows above the reserve only"
    assert _solver("compiled", serve=True)._compute_kv_cache_plan(None, "float16", room).max_cache_len == KV["window"]
    run = _solver("compiled")._compute_kv_cache_plan(None, "float16", room, growth_reserve_bytes=reserve)
    assert run.session_reach == above, "a run plan's reach is what serving would hold in the same room"
    inside = _solver("compiled", serve=True)._compute_kv_cache_plan(None, "float16", session_floor * TOKEN,
                                                                    growth_reserve_bytes=reserve)
    assert inside.max_cache_len == session_floor == inside.session_reach, "the session floor, from the whole room"


def _plan(monkeypatch, mode):
    """The 7B decoder's run plan on the rack's 16 GB card, idle, through the CLI's own request (120 tokens asked)."""
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.nbx import NBXContainer
    no_door(monkeypatch)
    pin_no_calibration(monkeypatch)
    pin_dedicated_card(monkeypatch, 16151, 306, "the rack's V100-16GB, nothing else on it")
    c = NBXContainer.load(str(container_root(MODEL)))
    man = c.get_manifest() or {}
    argv = ["run", "--model", MODEL, "--prompt", "Write a Python function that returns the n-th Fibonacci number.",
            "--max-tokens", "120"] + (["--triton"] if mode == "triton" else [])
    s = PrismSolver()
    plan = s.solve(c, profile(V100_16GB), request_input_config(create_parser().parse_args(argv), man, man.get("family"),
                                                               c.cache_path), mode=mode)
    return s, c, plan


def test_the_7b_run_plan_on_a_16gb_card_keeps_its_cache_at_the_request(monkeypatch):
    """The whole wiring, `_solve_at_rung` to the price: the plan that ran out of memory (lazy_sequential, the raw rung)
    sizes its cache at the request's floor. Red at the base form: 8 689 positions, 475 MiB, the package's bound."""
    s, c, plan = _plan(monkeypatch, "compiled")
    defaults = s._read_defaults(c)
    margin = defaults.get("prompt_margin", DEFAULT_PROMPT_MARGIN)
    assert plan.strategy == "lazy_sequential", "the cell judges the raw-rung plan; another strategy is another cell"
    assert plan.kv_cache_plan.max_cache_len == kv_min_tokens(s._request_max_tokens(defaults), margin, False)
    assert plan.kv_cache_plan.max_cache_len < int(defaults["max_tokens"]) + margin


def test_the_7b_triton_run_plan_is_not_charged_the_reserve_twice(monkeypatch):
    _, _, plan = _plan(monkeypatch, "triton")
    assert plan.kv_cache_plan.max_cache_len == 32768, "its capacity holds the band already: the declared window, as before"
