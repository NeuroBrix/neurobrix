"""Unit tests for explicit `--max-context <int>` support in `neurobrix serve` / `chat` / `run`,
Prism KV budget capping (both PyTorch and Triton solvers), and PyTorch + Triton session propagation.

Verifies:
1. CLI parsing of `--max-context` on `serve`, `chat`, and `run`, plus rejection of invalid values (< 1, bool).
2. `ServingDaemon._foreground_command()` preserves `--max-context` across subprocess daemonization.
3. `ensure_serving()` reuses when `max_context` matches or is omitted, and swaps when a different `max_context` is requested.
4. Prism KV pricing (`kv_cache_size`, `kv_estimate_tokens`, and both PyTorch & Triton `PrismSolver._compute_kv_cache_plan` /
   `_kv_min_tokens` / `_estimate_kv_cache_bytes`) shrinks a 202,752-window model's KV budget proportionally when
   `--max-context 4096` is passed, clamps cleanly to `min(max_context, model_window)` when `max_context > model_window`,
   and preserves ZERO FALLBACK with an actionable `--max-context` hint when uncapped and OOM.
5. Runtime propagation: PyTorch `KVCacheFactory.create()` and Triton `session_kv_params()` + `effective_kv_window()`
   and `InferenceEngine._window()` / `served_reach()` all honour `max_context`.
"""
from __future__ import annotations

import types
import pytest

from neurobrix.cli import create_parser
from neurobrix.core.state.price import (
    effective_kv_window,
    kv_cache_size,
    kv_estimate_tokens,
    kv_token_bytes,
    validate_max_context,
)
from neurobrix.serving.server import ServingDaemon


# 200K-context model geometry (47 layers, 20 kv heads, 256+256 head dims, 202,752 window -> ~195 GB at fp16/bf16)
KV_200K = {
    "kind": "kv",
    "layers": 47,
    "kv_heads": 20,
    "k_dim": 256,
    "v_dim": 256,
    "dtype": "bfloat16",
    "window": 202752,
}


def test_validate_max_context_accepts_positive_ints_and_rejects_invalid():
    assert validate_max_context(None) is None
    assert validate_max_context(1) == 1
    assert validate_max_context(4096) == 4096
    for bad in (0, -1, -4096, True, False, "4096", 3.5):
        with pytest.raises(ValueError, match="--max-context must be a positive integer"):
            validate_max_context(bad)


def test_effective_kv_window_clamps_to_model_window():
    assert effective_kv_window(202752, None) == 202752
    assert effective_kv_window(202752, 4096) == 4096
    assert effective_kv_window(2048, 4096) == 2048
    assert effective_kv_window(0, 4096) == 4096


def test_cli_parses_max_context_on_serve_chat_and_run():
    parser = create_parser()
    a_serve = parser.parse_args(["serve", "--model", "m", "--max-context", "4096"])
    assert a_serve.command == "serve" and a_serve.max_context == 4096

    a_chat = parser.parse_args(["chat", "--model", "m", "--max-context", "8192"])
    assert a_chat.command == "chat" and a_chat.max_context == 8192

    a_run = parser.parse_args(["run", "--model", "m", "--prompt", "hi", "--max-context", "2048"])
    assert a_run.command == "run" and a_run.max_context == 2048

    a_default = parser.parse_args(["serve", "--model", "m"])
    assert a_default.max_context is None


def test_foreground_command_preserves_max_context():
    class _FakeEngine:
        model_name = "fake-model"
        hardware_id = "v100-32g"
        mode = "triton"
        max_context = 4096

    d = ServingDaemon("fake-model", "v100-32g", mode="triton", idle_timeout=600.0,
                      engine=_FakeEngine(), max_context=4096)
    cmd = d._foreground_command()
    args = create_parser().parse_args(cmd[3:])
    assert args.command == "serve"
    assert args.foreground is True
    assert args.max_context == 4096


def test_kv_cache_size_shrinks_proportionally_and_preserves_zero_fallback_hint():
    tb = kv_token_bytes(KV_200K, "float16", 1)
    # Uncapped 202,752 positions with defaults max_tokens=202752 exceeds 40 GB remaining -> ZERO FALLBACK + hint
    remaining_40gb = 40 * (1024 ** 3)
    with pytest.raises(RuntimeError, match=r"ZERO FALLBACK: KV cache budget too small.*--max-context"):
        kv_cache_size(KV_200K, tb, 202752, 128, True, remaining_40gb)

    # Capped to 4096 -> fits easily and memory shrinks proportionally (4096 / 202752)
    capped = kv_cache_size(KV_200K, tb, 202752, 128, True, remaining_40gb, max_context=4096)
    assert capped["max_cache_len"] == 4096
    assert capped["memory_bytes"] == 4096 * tb
    full_bytes = 202752 * tb
    assert capped["memory_bytes"] * 202752 == full_bytes * 4096

    # Capping larger than model window clamps to model window
    small_kv = {**KV_200K, "window": 2048}
    clamped = kv_cache_size(small_kv, tb, 256, 128, True, remaining_40gb, max_context=8192)
    assert clamped["max_cache_len"] == 2048

    # kv_estimate_tokens also respects max_context
    assert kv_estimate_tokens(KV_200K, 202752, 128, max_context=4096) == 4096
    assert kv_estimate_tokens(small_kv, 0, 128, max_context=8192) == 2048


@pytest.mark.parametrize("side", ["pytorch", "triton"])
def test_prism_solver_both_branches_cap_kv_budget_and_serialize_max_context(side):
    import importlib
    solver_mod = importlib.import_module(f"neurobrix.{side}.prism.solver")
    profiler_mod = importlib.import_module(f"neurobrix.{side}.prism.profiler")
    PrismSolver = solver_mod.PrismSolver
    ExecutionPlan = solver_mod.ExecutionPlan
    InputConfig = profiler_mod.InputConfig

    topo = {
        "components": {"model": {}},
        "flow": {"type": "autoregressive_generation", "generation": {"lm_component": "model"}},
        "state": {"model": [KV_200K]},
    }
    defaults = {"max_tokens": 202752, "prompt_margin": 128}

    s = PrismSolver()
    s._serve_mode = True
    s._serve_requested = True
    s._mode = "compiled" if side == "pytorch" else "triton"
    s._flow_topology = lambda _c: topo
    s._read_defaults = lambda _c: defaults
    s._kv_sequences = lambda _c: 1

    # Uncapped: _kv_min_tokens & _estimate_kv_cache_bytes demand huge budget
    s._input_config = InputConfig(max_context=None)
    assert s._kv_min_tokens(object()) > 200000
    est_uncapped = s._estimate_kv_cache_bytes(object(), "float16")
    assert est_uncapped == 202752 * kv_token_bytes(KV_200K, "float16", 1)

    # Capped to 4096: floor, estimate, and KVCachePlan all cap to 4096
    s._input_config = InputConfig(max_context=4096)
    assert s._kv_min_tokens(object()) == 4096
    est_capped = s._estimate_kv_cache_bytes(object(), "float16")
    assert est_capped == 4096 * kv_token_bytes(KV_200K, "float16", 1)

    remaining_40gb = 40 * (1024 ** 3)
    kv_plan = s._compute_kv_cache_plan(object(), "float16", remaining_40gb)
    assert kv_plan is not None
    assert kv_plan.max_cache_len == 4096
    assert kv_plan.session_reach == 4096
    assert kv_plan.max_context == 4096
    assert kv_plan.memory_bytes == 4096 * kv_plan.per_token_bytes

    # Serialization preserves max_context
    ep = ExecutionPlan(
        components={},
        target_dtype="float16",
        total_memory_mb=0.0,
        strategy="pipeline",
        component_memory={},
        kv_cache_plan=kv_plan,
    )
    d = ep.to_dict()
    assert d["kv_cache_plan"]["max_context"] == 4096
    assert d["kv_cache_plan"]["max_cache_len"] == 4096


def test_pytorch_and_triton_session_kv_propagation():
    from neurobrix.pytorch.prism.solver import KVCachePlan as PtKVCachePlan
    from neurobrix.triton.prism.solver import KVCachePlan as TrKVCachePlan
    from neurobrix.core.module.cache.factory import StateCacheFactory
    from neurobrix.triton.flow.autoregressive import session_kv_params
    from neurobrix.core.state.kv_tier import served_reach

    lm_config = {
        "num_layers": 47,
        "num_heads": 32,
        "hidden_size": 2048,
        "max_position_embeddings": 202752,
    }
    tb = kv_token_bytes(KV_200K, "float16", 1)

    # PyTorch StateCacheFactory must NOT raise max_cache_len back to 202752 when kv_plan.max_context=4096
    pt_kv = PtKVCachePlan(
        max_cache_len=4096,
        num_layers=47,
        num_kv_heads=20,
        k_head_dim=256,
        v_head_dim=256,
        dtype="float16",
        memory_bytes=4096 * tb,
        per_token_bytes=tb,
        initial_cache_len=4096,
        session_reach=4096,
        max_context=4096,
    )
    ctx = types.SimpleNamespace(plan=types.SimpleNamespace(kv_cache_plan=pt_kv))
    wrapper = StateCacheFactory.create(ctx, lm_config, device="cpu", dtype="float16", hardware=("nvidia", "volta"))
    assert wrapper.config.max_cache_len == 4096
    assert wrapper.config.ceiling_is_window is False

    # Triton session_kv_params with effective_kv_window caps window_ceiling to 4096
    tr_kv = TrKVCachePlan(
        max_cache_len=4096,
        num_layers=47,
        num_kv_heads=20,
        k_head_dim=256,
        v_head_dim=256,
        dtype="float16",
        memory_bytes=4096 * tb,
        per_token_bytes=tb,
        initial_cache_len=4096,
        session_reach=4096,
        max_context=4096,
    )
    eff_win = effective_kv_window(lm_config["max_position_embeddings"], tr_kv.max_context)
    assert eff_win == 4096
    params = session_kv_params(tr_kv, eff_win, decode_budget=128)
    assert params["max_cache_len"] == 4096
    assert params["window_ceiling"] == 4096

    # served_reach also stays capped at 4096
    reach, _src = served_reach(tr_kv, tier=None, window=eff_win)
    assert reach == 4096


def test_ensure_serving_reuses_or_swaps_based_on_max_context(monkeypatch):
    from neurobrix.cli.commands import serve as serve_cmd
    from neurobrix.serving import client as client_mod
    from neurobrix.serving import server as server_mod
    from neurobrix.core.runtime import engines

    monkeypatch.setattr(engines, "require_engine", lambda _m: None)
    monkeypatch.setattr(engines, "default_mode", lambda: ("triton", "default triton"))

    state = {"running": True, "status": {"model": "m", "mode": "triton", "hardware": "hw", "max_context": 4096}}
    started = []

    class _FakeClient:
        @staticmethod
        def is_running():
            return state["running"]

        def connect(self):
            pass

        def status(self):
            return dict(state["status"])

        def close(self):
            pass

    class _FakeDaemon:
        def __init__(self, model_name, hardware_id, mode, idle_timeout, max_context=None, **kw):
            started.append((model_name, hardware_id, mode, max_context))

        def start(self, **kw):
            state["running"] = True

    monkeypatch.setattr(client_mod, "DaemonClient", _FakeClient)
    monkeypatch.setattr(server_mod, "ServingDaemon", _FakeDaemon)
    monkeypatch.setattr(serve_cmd, "stop_daemon", lambda quiet=True: True)

    # Matching max_context -> reused
    assert serve_cmd.ensure_serving("m", hardware_id="hw", mode="triton", max_context=4096) == "reused"
    # Omitted max_context -> reused
    assert serve_cmd.ensure_serving("m", hardware_id="hw", mode="triton", max_context=None) == "reused"
    assert started == []

    # Different max_context -> swapped and new daemon receives max_context=8192
    assert serve_cmd.ensure_serving("m", hardware_id="hw", mode="triton", max_context=8192) == "swapped"
    assert started == [("m", "hw", "triton", 8192)]
