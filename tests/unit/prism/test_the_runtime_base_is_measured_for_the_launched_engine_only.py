"""Orchestrator ruling of 22/11/2026 17:43, line 3: the runtime base (`cpu.runtime_base_mb`) stays measured in a child,
and only the child of the engine branch being launched, at its first launch, kept; the other branch's base is measured
the first time THAT branch is launched (before: both children at the profile's generation, 7.4 s of a first launch,
the Triton child compiling kernels for an engine the launch never ran).

What each test would do if the code were wrong: both engines measured at one launch, or at the profile's generation
(red); a kept base measured again (red); the other engine's base never measured at its own first launch (red); a
sibling profile's measure of this engine measured again (red)."""
from __future__ import annotations

import subprocess

import pytest
import yaml

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import facts_probe as P

DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16"],
       "runtime_total_mb": 16000, "runtime_context_mb": 300,
       "measured": {"matmul_tflops": {"float16": 10.0}, "memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0,
                    "vmm_chunk_bytes": 2 << 20, "expert_copy": {"bytes_per_s": 1e9},
                    "gemv_gbps": {"compiled": {"float16": 100.0}, "triton": {"float16": 100.0}}}}


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    import neurobrix.core.paths as paths
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(paths, "cache_dir", lambda: tmp_path / "cache")         # no container: no reload probe owed
    monkeypatch.setattr(P, "vmm_first_replay", lambda *a, **k: None)
    monkeypatch.setattr(P, "vmm_chunk_bytes", lambda *a, **k: (2 << 20, None))  # the chunk: not this test's
    for name in ("matmul_tflops", "copies", "card_facts_in_a_child", "gemv_gbps", "host_matmul_gflops",
                 "host_memory_gbps"):
        monkeypatch.setattr(P, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(AssertionError(f"{_n} measured")))
    measured = []
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda engine: measured.append(engine) or {"compiled": 800,
                                                                                                    "triton": 90}[engine])
    return tmp_path, measured


def _profile(directory, name, cpu):
    (directory / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "cpu": cpu, "devices": [dict(DEV)],
                                                           "host_measured": {"host_matmul_gflops": 300,
                                                                             "host_memory_gbps": 20}}))


def _base(directory, name):
    return yaml.safe_load((directory / f"{name}.yml").read_text())["cpu"].get("runtime_base_mb")


def test_each_engine_s_base_is_measured_at_its_own_first_launch_and_kept(hw):
    d, measured = hw
    _profile(d, "default-abc", {"model": "x"})
    A.ensure_measured_facts("default-abc", "triton")
    assert measured == ["triton"] and _base(d, "default-abc") == {"triton": 90}
    A.ensure_measured_facts("default-abc", "triton")
    assert measured == ["triton"], "a kept base is measured again"
    A.ensure_measured_facts("default-abc", "compiled")
    assert measured == ["triton", "compiled"] and _base(d, "default-abc") == {"triton": 90, "compiled": 800}


def test_a_sibling_s_measure_of_this_engine_is_taken(hw):
    d, measured = hw
    _profile(d, "default-four", {"model": "x", "runtime_base_mb": {"compiled": 777}})
    _profile(d, "default-one", {"model": "x", "runtime_base_mb": {"triton": 91}})
    A.ensure_measured_facts("default-one", "compiled")
    assert measured == [] and _base(d, "default-one") == {"triton": 91, "compiled": 777}


def test_one_child_of_the_asked_engine_alone(monkeypatch):
    run = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **k: run.append(argv) or
                        subprocess.CompletedProcess(argv, 0, "33\n123\n", ""))
    assert A._measure_runtime_base_mb("triton") == 90
    (argv,), = [run]
    assert A._BASE_PROBES["triton"] in argv[-1] and "import torch" not in argv[-1]


def test_a_second_sibling_s_measure_is_taken_when_the_first_lacks_this_engine(hw):
    """The siblings are read in order (`other_card_models.sibling_profiles`): the first measured the other engine only,
    the second this one — the second's is taken, and no child runs (qa 1231-3)."""
    d, measured = hw
    _profile(d, "default-a", {"model": "x", "runtime_base_mb": {"triton": 91}})
    _profile(d, "default-b", {"model": "x", "runtime_base_mb": {"compiled": 777}})
    _profile(d, "default-z", {"model": "x"})
    A.ensure_measured_facts("default-z", "compiled")
    assert measured == [] and _base(d, "default-z")["compiled"] == 777
