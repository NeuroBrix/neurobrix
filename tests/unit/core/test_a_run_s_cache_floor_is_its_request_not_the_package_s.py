"""A run's KV floor and pre-sizing estimate are priced at the tokens the request asks (--max-tokens), not at the
package's declared ceiling; the cache is still sized up to the package's bound when the room affords it; serving keeps
the package's figures.

What this test does if the code is wrong: the floor cell fails when a run asking 32 tokens of a package declaring
32 768 is refused for the 32 768 (the Mac, 2026-10-19: Qwen3-30B-A3B-Thinking's expert store refused for 3.2 GB of
KV, the model then ran layer_streaming at 0.02 tok/s); the bound cell fails when the request also caps the cache below
what the room affords; the solver cells fail when the CLI's request does not reach the floor, or reaches a serve plan.
"""
from types import SimpleNamespace

import pytest

from neurobrix.core.state.price import kv_cache_size

KV = {"kind": "kv", "layers": 48, "kv_heads": 4, "k_dim": 128, "v_dim": 128, "window": 262144}
TOKEN = 48 * 4 * 256 * 2


def test_a_run_asking_few_tokens_is_floored_at_its_request():
    room = 851 * 10 ** 6
    with pytest.raises(RuntimeError, match="KV cache budget too small"):
        kv_cache_size(KV, TOKEN, 32768, 128, False, room)
    size = kv_cache_size(KV, TOKEN, 32768, 128, False, room, floor_max_tokens=32)
    assert size["max_cache_len"] == room // TOKEN, "the cache takes what the room affords"


def test_the_request_does_not_cap_the_cache_below_the_package_bound():
    size = kv_cache_size(KV, TOKEN, 1000, 128, False, 10 ** 12, floor_max_tokens=32)
    assert size["max_cache_len"] == 1128


def _solver(asked, serve):
    from neurobrix.core.prism.solver import PrismSolver
    s = PrismSolver.__new__(PrismSolver)
    s._input_config = SimpleNamespace(max_tokens=asked)
    s._serve_requested = serve
    return s


def test_the_solver_floors_a_run_at_the_cli_request_and_a_serve_at_the_package():
    d = {"max_tokens": 32768, "prompt_margin": 128}
    assert _solver(32, False)._request_max_tokens(d) == 32
    assert _solver(None, False)._request_max_tokens(d) == 32768
    assert _solver(32, True)._request_max_tokens(d) == 32768


def test_the_cli_carries_its_max_tokens_into_the_plan_s_input():
    import inspect
    from neurobrix.cli.commands import run
    assert "max_tokens=getattr(args, 'max_tokens', None)" in inspect.getsource(run.request_input_config)
