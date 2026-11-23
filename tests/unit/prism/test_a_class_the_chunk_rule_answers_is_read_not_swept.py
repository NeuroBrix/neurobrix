"""A card class the library's chunk rule answers is READ at the first launch, never swept (OWNER RESET 22/11 point 1;
first-launch fact page line 1, mac_cards/first_launch_read/FACTS.md).

The library's allocator takes the rule's chunk (`nbxtensor.memory.chunk_rule.vmm_chunk_bytes`, by the profile's
`memory_mb`) before any measured one, so the engine's sweep of such a class ran its first replay and 11-13 children
(40-48 s of the Dell's 61.7 s first launch on a V100-16GB) and was then ignored. A class the rule does not answer is
still swept once and kept.

What each test would do if the code were wrong: a sweep or its first replay run for a class the rule answers (red on
main 9eb471d: the engine never asks the rule), a class the rule does not answer left without its chunk (red)."""
from __future__ import annotations

import pytest
import yaml

from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import facts_probe as P

CALLS = []
C16 = {"brand": "nvidia", "model": "Card-16", "architecture": "unknownarch", "supports_dtypes": ["float16"],
       "memory_mb": 16384}
C48 = dict(C16, model="Card-48", memory_mb=49152)


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    CALLS.clear()
    monkeypatch.setattr(P, "matmul_tflops", lambda t, *a, **k: {"best": 10.0})
    monkeypatch.setattr(P, "copies", lambda *a, **k:
                        {"memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0, "h2d_pageable_gbps": 5.0})
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda i: {"runtime_total_mb": 16000, "runtime_context_mb": 300})
    monkeypatch.setattr(P, "sweep_chunk", lambda d, *a, **k: CALLS.append(("sweep", d)) or (32 << 20, None))
    monkeypatch.setattr(P, "vmm_first_replay", lambda d: CALLS.append(("first_replay", d)) or (2.0, 3.0))
    # the rule's classes, as the library's chunk_rule.yml states them for this test: 16 GB answered, 48 GB not
    monkeypatch.setattr(P, "ruled_chunk", lambda d, memory_mb: (64 << 20) if memory_mb == 16384 else None)
    return tmp_path


def _write(hw, name, devices):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": devices}))


def test_a_class_the_rule_answers_runs_no_sweep_and_no_first_replay(hw):
    _write(hw, "default-one16", [dict(C16, index=0)])
    A.ensure_measured_facts("default-one16")
    assert CALLS == [], CALLS


def test_a_class_the_rule_does_not_answer_is_swept_once_and_kept(hw):
    _write(hw, "default-mixed", [dict(C16, index=0), dict(C48, index=1)])
    A.ensure_measured_facts("default-mixed")
    assert ("sweep", 1) in CALLS and ("sweep", 0) not in CALLS and ("first_replay", 0) not in CALLS, CALLS
    devs = yaml.safe_load((hw / "default-mixed.yml").read_text())["devices"]
    assert "vmm_chunk_bytes" not in (devs[0].get("measured") or {})         # read from the rule at every launch
