"""Orchestrator ruling of 22/11/2026 17:43, line 4: the reload probe (the per-piece reload rate a streamed plan is
priced with, `time_estimate.reload_bps`) runs only when Prism prices a STREAMED candidate — measured once and kept —
never at every first launch (TinyLlama on a 16 GB card owed one, 3.3 s, and its os.stat ended the launch rc 1).

What each test would do if the code were wrong: a first launch reloading a component before any plan is priced (red);
a ranking without a streamed candidate probing (red); a ranking with one not probing, or not reading the written rate
back into the profile it prices with (red)."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
import yaml

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import owed_facts as OF
from neurobrix.core.prism import facts_probe as P

DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16"],
       "memory_mb": 16384, "runtime_total_mb": 16000, "runtime_context_mb": 300,
       "measured": {"matmul_tflops": {"float16": 10.0}, "memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0,
                    "vmm_chunk_bytes": 2 << 20, "gemv_gbps": {"compiled": {"float16": 100.0}}}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    import neurobrix.core.paths as paths
    from neurobrix.core.prism import arena_probe as AP
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    cache = tmp_path / "cache"
    c = cache / "M" / "components" / "lm"
    c.mkdir(parents=True)
    (c / "weights_index.json").write_text(json.dumps({"tensors": {"w": {"dtype": "bfloat16", "size_bytes": 1}}}))
    with open(c / "w.safetensors", "wb") as f:
        f.truncate(2 * A.LOADER_RATE_MIN_BYTES)          # sparse: the size is what the probe reads
    monkeypatch.setattr(paths, "cache_dir", lambda: cache)
    monkeypatch.setattr(P, "vmm_first_replay", lambda *a, **k: None)
    monkeypatch.setattr(P, "sweep_chunk", lambda *a, **k: (2 << 20, None))
    monkeypatch.setattr(P, "ruled_chunk", lambda *a, **k: None)   # a class the library's rule does not answer: swept
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda engine: 800)
    probed = []
    monkeypatch.setattr(AP, "child", lambda d, comp, h, e: probed.append((comp, h, e)) or
                        {"container": d, "component": comp, "draws": []})
    (tmp_path / "default-abc.yml").write_text(yaml.safe_dump({
        "id": "default-abc", "cpu": {"model": "x"}, "devices": [dict(DEV)],
        "host_measured": {"host_matmul_gflops": 300, "host_memory_gbps": 20}}))
    return cache / "M", probed


def test_the_first_launch_reloads_nothing(home):
    container, probed = home
    A.ensure_measured_facts("default-abc", "compiled", container_dir=container)
    assert probed == []


@pytest.mark.parametrize("ranked, owed", [((("lazy_sequential", {}),), False), ((("single_gpu", {}),), False),
                                          # a streamed candidate the time decides (its rank is the best present;
                                          # behind a resident plan it owes nothing: test_a_measure_is_owed_only_...)
                                          ((("lazy_sequential", {}), ("layer_streaming", {})), True)])
def test_the_probe_runs_when_a_streamed_candidate_is_priced(home, monkeypatch, ranked, owed):
    container, probed = home
    import neurobrix.core.prism.loader as L
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-abc")
    monkeypatch.setattr(L, "load_profile", lambda hw: NS(host_measured={"weight_reload_gbps": {"compiled": {"BF16": 1.0}}}))
    profile = NS(host_measured=None, devices=[NS(measured={"memory_gbps": 900.0})])   # the facts a timed plan has
    OF.measure_owed("compiled", [(0.0, n, a, None) for n, a in ranked],
                                    NS(cache_path=str(container)), profile)
    assert probed == ([("lm", "default-abc", "compiled")] if owed else [])
    assert (profile.host_measured is not None) == owed, "the written rate is read back into the profile priced"
