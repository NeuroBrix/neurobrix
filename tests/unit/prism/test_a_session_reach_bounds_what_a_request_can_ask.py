"""A plan states how far a session can reach, and a KV plan survives its own save and restore.

2026-11-08: the derived census keyed a decoder's prompt lengths only up to the cache Prism planned for the GATE request
(TinyLlama: 640 positions of a 2 048 context), so a user's 530- or 900-token prompt stopped under --certified-only on
matmul M_BUCKET 672 / 1152 — keys no census row held. Prism now states `session_reach`: what a serving session would
hold in the room the run cache was sized from, within the decoder's declared window; the census keys prompts up to it.
The same lines restored a plan without its `initial_cache_len` and with `memory_bytes` through a float MB defaulting to
0 (the defect the KV design's step 0 names).

What this test does if the code is wrong: the first cell fails when the plan states no reach or one no larger than the
run cache (red on main: no field); the second when a restored plan loses initial_cache_len, memory_bytes or the reach
(red on main: lost); the third when the census does not key TinyLlama's prompts from 1 to the reach (red on main:
from the gate's own length to 640).

Run: python -m pytest tests/unit/prism/test_a_session_reach_bounds_what_a_request_can_ask.py
"""
import json
import sys
from pathlib import Path

import pytest

from tests.unit.prism._pinned_machine import container_root, profile, V100_16GB, pin_no_calibration

MODEL = "TinyLlama-1.1B-Chat-v1.0"


def _plan(monkeypatch):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.core.prism import PrismSolver
    from neurobrix.nbx import NBXContainer
    pin_no_calibration(monkeypatch)
    c = NBXContainer.load(str(container_root(MODEL)))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", "hi", "--max-tokens", "32", "--triton"])
    return PrismSolver().solve(c, profile(V100_16GB), request_input_config(args, man, man.get("family"), c.cache_path),
                               mode="triton")


def test_the_plan_states_a_session_reach_beyond_the_run_cache(monkeypatch):
    kv = _plan(monkeypatch).kv_cache_plan
    assert kv.session_reach is not None and kv.session_reach > kv.max_cache_len, (kv.session_reach, kv.max_cache_len)
    assert kv.session_reach <= 2048                       # never beyond the declared window


def test_a_kv_plan_survives_its_save_and_restore(monkeypatch, tmp_path):
    from neurobrix.core.prism.solver import ExecutionPlan
    p = _plan(monkeypatch)
    p.kv_cache_plan.initial_cache_len = 96               # a serve plan's first request size
    p.save(tmp_path / "plan.json")
    back = ExecutionPlan.load(tmp_path / "plan.json").kv_cache_plan
    assert back.memory_bytes == p.kv_cache_plan.memory_bytes
    assert back.initial_cache_len == 96
    assert back.session_reach == p.kv_cache_plan.session_reach


def test_the_census_keys_every_prompt_up_to_the_session_reach():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import derived_census as D
    plan = {"kv_cache": {"max_cache_len": 640, "session_reach": 2048, "dtype": "float16", "num_kv_heads": 4}}
    sites = D._decoder_sites(MODEL, "model", plan, 25, 32, lambda n: {})
    prefill = [s for s in sites if "prefill" in s[0]][0]
    decode = [s for s in sites if "decode" in s[0]][0]
    assert (prefill[1], prefill[2]) == (1, 2047), prefill[:3]
    assert decode[2] == 2048, decode[:3]
