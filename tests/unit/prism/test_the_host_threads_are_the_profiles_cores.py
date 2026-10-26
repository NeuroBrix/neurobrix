"""The host's compute threads come from the hardware profile, for both engines and every variable that sets them
(the owner's rule 1, 2026-10-26 23:49): a profile declaring 4 cores runs 4 threads, whatever the machine. Before: the
CLI set OMP_NUM_THREADS / MKL_NUM_THREADS from `os.cpu_count()` at import, the Triton branch only `setdefault`-ed over it
(the profile was never applied), the compiled branch skipped the profile when the plan kept the CPU idle, and the
executor re-derived `os.cpu_count() // 2`. On Apple silicon the profile's performance cores are the compute threads
(the M4 Pro: 8 performance + 4 efficiency).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from neurobrix.core.prism.cpu_config import CPUConfig, apply_thread_config

_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TRITON_CPU_MAX_THREADS")


def _cpu(cores, performance=None, efficiency=None):
    d = {"model": "test", "cores": cores, "threads": cores, "ram_mb": 16384, "architecture": "arm64"}
    if performance is not None:
        d["performance_cores"], d["efficiency_cores"] = performance, efficiency
    return CPUConfig.from_yaml_dict(d)


@pytest.mark.parametrize("mode", ["triton", "triton_sequential", "compiled"])
def test_a_four_core_profile_runs_four_threads_whatever_was_set_before(monkeypatch, mode):
    for v in _VARS:
        monkeypatch.setenv(v, "40")                 # what the import-time default left (a 40-core machine)
    apply_thread_config(_cpu(4), involves_cpu=False, mode=mode)
    assert all(os.environ[v] == "4" for v in _VARS), {v: os.environ[v] for v in _VARS}
    if mode == "compiled":
        import torch
        assert torch.get_num_threads() == 4


def test_the_performance_cores_are_the_compute_threads_on_a_hybrid_cpu(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    apply_thread_config(_cpu(12, performance=8, efficiency=4), involves_cpu=True, mode="triton")
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_no_thread_count_is_read_from_the_machine():
    src = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
    cli = (src / "cli" / "__init__.py").read_text()
    assert "os.cpu_count()" not in cli.replace("_os_startup.cpu_count()", "os.cpu_count()"), \
        "the CLI derives thread counts from the machine at import"
    executor = (src / "core" / "runtime" / "executor.py").read_text()
    assert "_optimize_cpu_threading" not in executor, "the executor re-derives threads from os.cpu_count()"


def test_a_profile_without_a_cpu_is_refused_by_name_where_the_solver_prices_cores():
    solver = (Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "core" / "prism" / "solver.py").read_text()
    assert "profile.cpu.cores if profile.cpu else 4" not in solver, "a profile without cpu falls back to 4 cores"
