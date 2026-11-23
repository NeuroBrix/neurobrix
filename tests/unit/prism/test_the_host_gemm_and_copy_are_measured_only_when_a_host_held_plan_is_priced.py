"""Orchestrator ruling of 22/11/2026 17:43, line 5: the host's GEMM and copy (`host_matmul_gflops`, `host_memory_gbps`)
are measured only when a plan placing a component on the host is priced — once, and kept in the generated profile —
never at every first launch; a plan that places nothing on the host is priced without them.

What each test would do if the code were wrong: a first launch measuring the host (red); a ranking with no host-placed
component measuring it, or one with a host-placed component not measuring it, or measuring it twice (red); a device plan
refused for the host's GEMM it never reads (red); a host-placed component priced without it instead of refused (red)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import yaml

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import owed_facts as OF
from neurobrix.pytorch.prism import facts_probe as P
from nbxtensor.backend import host_measure as H
from neurobrix.pytorch.prism import time_estimate as T
from neurobrix.pytorch.prism.work import Work

DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16"],
       "runtime_total_mb": 16000, "runtime_context_mb": 300,
       "measured": {"matmul_tflops": {"float16": 10.0}, "memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0,
                    "vmm_chunk_bytes": 2 << 20}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(P, "vmm_first_replay", lambda *a, **k: None)
    monkeypatch.setattr(P, "sweep_chunk", lambda *a, **k: (2 << 20, None))
    monkeypatch.setattr(P, "ruled_chunk", lambda *a, **k: None)   # a class the library's rule does not answer: swept
    calls = []
    monkeypatch.setattr(H, "host_matmul_gflops", lambda *a, **k: calls.append("gemm") or 300.0)
    monkeypatch.setattr(H, "host_memory_gbps", lambda *a, **k: calls.append("copy") or 20.0)
    (tmp_path / "default-abc.yml").write_text(yaml.safe_dump({"id": "default-abc", "cpu": {"model": "x"},
                                                              "devices": [dict(DEV)]}))
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-abc")
    import neurobrix.pytorch.prism.loader as L        # the profile read back: the file this test's home holds
    monkeypatch.setattr(L, "load_profile", lambda hw: NS(host_measured=_host(tmp_path)))
    return tmp_path, calls


def _host(d):
    return yaml.safe_load((d / "default-abc.yml").read_text()).get("host_measured") or {}


def test_the_first_launch_measures_no_host_fact(home):
    d, calls = home
    A.ensure_measured_facts("default-abc")
    assert calls == [] and "host_matmul_gflops" not in _host(d)


def test_the_host_is_measured_when_a_host_placed_candidate_is_priced_once(home):
    d, calls = home
    profile = NS(host_measured=None, devices=[NS(measured={"memory_gbps": 900.0})])   # the facts a timed plan has
    solver = NS(_mode="compiled")
    OF.measure_owed(solver._mode, [(0.0, "single_gpu", {"lm": "cuda:0"}, None)], NS(cache_path=None), profile)
    assert calls == [] and profile.host_measured is None
    OF.measure_owed(solver._mode, [(0.0, "lazy_sequential", {"lm": "cuda:0", "vae": "cpu"}, None)],
                                    NS(cache_path=None), profile)
    assert calls == ["gemm", "copy"] and _host(d)["host_memory_gbps"] == 20 == profile.host_measured["host_memory_gbps"]
    OF.measure_owed(solver._mode, [(0.0, "lazy_sequential", {"vae": "cpu"}, None)], NS(cache_path=None), profile)
    assert calls == ["gemm", "copy"], "measured again"


def test_a_device_plan_is_priced_without_the_host_gemm_and_a_host_placed_one_is_refused():
    gpu = T.DeviceFacts(matmul_flops={"float16": 1e12}, memory_bps=1e11, h2d_bps=1e10,
                        load_rates={"compiled": {"F16": {"device": {"cold": 1.0, "warm": 2.0}}}})
    host = T.host_facts({"weight_reload_gbps": {"compiled": {"F16": 1.0}}})
    kw = dict(engine="compiled", stored={"c": "float16"})
    assert T.plan_seconds("single_gpu", {"c": "cuda:0"}, {"c": Work(1, 1, 1)}, {}, {"cuda:0": gpu}, host, "float16", **kw) > 0
    with pytest.raises(T.MissingFact, match="host_matmul_gflops"):
        T.plan_seconds("single_gpu", {"c": "cpu"}, {"c": Work(1, 1, 1)}, {}, {"cuda:0": gpu}, host, "float16", **kw)
