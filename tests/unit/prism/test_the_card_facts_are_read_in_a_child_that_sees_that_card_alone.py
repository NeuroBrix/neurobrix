"""Orchestrator ruling of 22/11/2026 17:43, line 1: the serving process opens no context on a card its plan does not
name. A generated profile's runtime card facts (`runtime_total_mb`, `runtime_context_mb`) are READ once in a short child
that sees ONE card (`nbxtensor.backend.install_help.one_device_env`, the chunk probe's replay child's pattern), through
the library's own reads (the allocator's total, the driver's per-process accounting), written to the profile and read
from it at every later launch.

What each test would do if the code were wrong: a child shown every card of the launch, or asked for the card by the
parent's index (red); a context read through the engine's own nvidia-smi query instead of the library (red); a read that
allocates or times (red: the library's two reads are the only calls)."""
from __future__ import annotations

import subprocess
import sys

import pytest

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import facts_probe as P


def test_the_child_sees_the_card_alone_and_reads_it_as_its_device_0(monkeypatch):
    from nbxtensor.backend import runtime as RT
    monkeypatch.setattr(RT, "backend_table", lambda: {"visible_env": ("CUDA_VISIBLE_DEVICES",)})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    seen = []

    def run(argv, **kw):
        seen.append((argv, kw["env"]))
        return subprocess.CompletedProcess(argv, 0, 'CARD_FACTS {"runtime_total_mb": 16151, "runtime_context_mb": 306}\n', "")
    monkeypatch.setattr(subprocess, "run", run)
    assert P.card_facts_in_a_child(1) == {"runtime_total_mb": 16151, "runtime_context_mb": 306}
    (argv, env), = seen
    assert argv[0] == sys.executable and env["CUDA_VISIBLE_DEVICES"] == "3", env.get("CUDA_VISIBLE_DEVICES")
    assert "runtime_card_facts(0)" in argv[-1], argv[-1]


def test_the_facts_are_the_librarys_reads_and_nothing_else(monkeypatch):
    from nbxtensor.backend import process_memory as PM

    def no_tool(*a, **k):
        raise AssertionError(f"the engine queried a tool of its own: {a}")
    monkeypatch.setattr(subprocess, "run", no_tool)
    monkeypatch.setattr(A._nbx_memory, "device_total_bytes", lambda i: 16151 << 20)
    monkeypatch.setattr(PM, "process_device_bytes", lambda i: 306 << 20)
    assert A.runtime_card_facts(0) == {"runtime_total_mb": 16151, "runtime_context_mb": 306}
    monkeypatch.setattr(PM, "process_device_bytes", lambda i: None)      # no per-process accounting: no reading
    assert A.runtime_card_facts(0) == {}
