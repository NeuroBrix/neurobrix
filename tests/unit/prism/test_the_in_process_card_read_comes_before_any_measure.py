"""The runtime's card facts read IN this process (one card seen, `facts_probe.card_facts_reader`) are read BEFORE any
measure this process runs at the first launch (qa 1235-1).

The driver attributes to a process everything it holds on the card: read after the GEMM, the copies and the GEMV of
the same process, `runtime_context_mb` kept what the PyTorch branch's allocator still reserved (Dell, V100 16 GB, fresh
home, the unknown-card case: compiled 564 MiB in process against 306 in a child, p1235 TABLE_c1). A child holds nothing
of this process, so its place (after the chunk is written, which it reads) does not move.

What each test would do if the code were wrong: the read after any first-replay, GEMM, copy or GEMV call (red on
91bd545 + e8bf399: the read came after them); the child asked before the chunk was written (red)."""
from __future__ import annotations

import pytest
import yaml

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import facts_probe as P

CARD = {"brand": "nvidia", "model": "Card-16", "architecture": "unknownarch", "supports_dtypes": ["float16"],
        "memory_mb": 16384, "index": 0}
CALLS = []


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    CALLS.clear()
    monkeypatch.setattr(P, "matmul_tflops", lambda t, *a, **k: CALLS.append("gemm") or {"best": 10.0})
    monkeypatch.setattr(P, "copies", lambda *a, **k: CALLS.append("copies") or
                        {"memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0, "h2d_pageable_gbps": 5.0})
    monkeypatch.setattr(P, "gemv_gbps", lambda *a, **k: CALLS.append("gemv") or 400.0)
    monkeypatch.setattr(P, "ruled_chunk", lambda d, memory_mb: None)           # the rule answers nothing: swept
    monkeypatch.setattr(P, "vmm_first_replay", lambda d: CALLS.append("first_replay") or (2.0, 3.0))
    monkeypatch.setattr(P, "sweep_chunk", lambda d, *a, **k: CALLS.append("sweep") or (32 << 20, None))
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda engine: 500)
    (tmp_path / "default-one16.yml").write_text(yaml.safe_dump({"id": "default-one16", "devices": [dict(CARD)]}))
    return tmp_path


def _facts(where):
    return lambda index: CALLS.append(where) or {"runtime_total_mb": 16151, "runtime_context_mb": 306}


def test_the_in_process_read_comes_before_every_measure_of_this_process(hw, monkeypatch):
    read = _facts("read")
    monkeypatch.setattr(P, "card_facts_reader", lambda devices: read)            # one card seen
    A.ensure_measured_facts("default-one16", engine="compiled")
    assert CALLS and CALLS[0] == "read", CALLS
    assert {"first_replay", "gemm", "copies", "gemv"} <= set(CALLS), CALLS   # the measures did run, after it
    dev = yaml.safe_load((hw / "default-one16.yml").read_text())["devices"][0]
    assert dev["runtime_context_mb"] == 306


def test_the_child_is_asked_after_the_chunk_it_reads_is_set(hw, monkeypatch):
    monkeypatch.setattr(P, "card_facts_in_a_child", _facts("child"))
    monkeypatch.setattr(P, "card_facts_reader", lambda devices: P.card_facts_in_a_child)
    A.ensure_measured_facts("default-one16", engine="compiled")
    assert "child" in CALLS and CALLS.index("child") > CALLS.index("sweep"), CALLS
