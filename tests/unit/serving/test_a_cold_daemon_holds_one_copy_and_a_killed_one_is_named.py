"""A cold-served daemon (weights reload per request) holds ONE copy of its weights, and a daemon that was killed is
named by `neurobrix stop`.

Measured 2026-11-08 (Qwen3-Coder-30B, lazy_sequential, card 2): the per-request cleanup left the request's context —
which holds the weight dict the run was built on — and the pinned host block on the executor, so every weight lived
until the NEXT run rebuilt its context, after that run's load: 61 GB resident between requests, 118 GB during each
load; the tools request's daemon died silently on a busy host. A SIGKILL cannot be logged by its victim; the pid file
it leaves behind is how `stop` says so.

Injections (each RED, then GREEN; see the commit message): the cleanup without `self._ctx = None` / the block pop —
the first test fails; `stop` printing its old line for a dead pid file — the second test fails.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_the_per_request_cleanup_releases_the_context_and_the_pinned_block():
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    ex = GraphExecutor.__new__(GraphExecutor)
    ex._persistent = False
    ex._compiled_seq = None
    ex._pieces_capture = None
    ex._weights = {"w": object()}
    ex._ctx = None
    ex._weights_loaded = True
    ex._ctx = type("Ctx", (), {"weights": {"w": object()}})()
    ex._pinned_host_block = object()
    try:
        ex.cleanup()
    except AttributeError as e:       # a field the minimal executor does not carry: said, so the test is read
        raise AssertionError(f"the minimal executor lacks a field cleanup reads: {e}")
    assert ex._weights_loaded is False
    assert ex._ctx is None, "the request's context (it holds the weight dict) outlived the cleanup"
    assert "_pinned_host_block" not in ex.__dict__, "the pinned host block outlived the cleanup"


def test_stop_names_a_daemon_that_was_killed(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (home / "daemon.pid").write_text(str(dead.pid))
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "NEUROBRIX_HOME": str(home), "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1",
           "PYTHONPATH": os.pathsep.join([str(repo / "src"), str(repo / "nbxtensor" / "src")])}
    env.pop("NBX_SOCKET_PATH", None)
    p = subprocess.run([sys.executable, "-m", "neurobrix", "stop"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stderr
    assert f"the daemon (PID {dead.pid}) ended without its shutdown" in p.stdout, p.stdout
    assert not (home / "daemon.pid").exists()


def test_a_cold_request_ends_with_every_component_uninstalled_and_its_block_released():
    """The strategy's install (zero3's ratchet keeps the host originals) is released at the end of a cold request;
    a warm daemon keeps everything."""
    from types import SimpleNamespace
    from neurobrix.serving.engine import InferenceEngine
    calls = []
    strategy = SimpleNamespace(uninstall_for_executor=lambda name, ex: calls.append(name))
    unloaded = []
    execs = {"model": SimpleNamespace(), "lm_head": SimpleNamespace()}
    for n, e in execs.items():
        e._pinned_host_block = object()
        e._devices_computed = True
        e.unload_weights = (lambda n=n: unloaded.append(n))
    eng = InferenceEngine.__new__(InferenceEngine)
    eng._executor = SimpleNamespace(strategy=strategy, executors=execs)
    eng._warm_serving = True
    eng._release_cold_request()
    assert calls == [] and all("_pinned_host_block" in e.__dict__ for e in execs.values())
    eng._warm_serving = False
    eng._release_cold_request()
    assert sorted(calls) == ["lm_head", "model"] and sorted(unloaded) == ["lm_head", "model"]
    assert not any("_pinned_host_block" in e.__dict__ for e in execs.values())
    assert not any(e._devices_computed for e in execs.values()), "the one-shot device classification must recompute"
