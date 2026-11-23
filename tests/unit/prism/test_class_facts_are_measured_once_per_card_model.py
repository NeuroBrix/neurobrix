"""A card model's facts are measured ONCE per machine and reused by every card set (the launch-time cure, 2026-11-19).

A fresh home on the 4-card rack measured the 16 GB class for its 1-card profile, again for its 2-card one and again for
its 4-card one: each profile (one per visible card set, `default-<tag>.yml`) ran the 24 chunk-probe replay children of
each class it holds, the engine-base probe and the host facts again. Now what a sibling generated profile of the same
machine measured — on a card of the SAME model — is taken (`other_card_models.from_siblings`), and the host's facts and
the engine base with it; only a model no profile has measured is measured.

What each test would do if the code were wrong: a probe run for a card model a sibling already measured (red), a
model's facts taken from a card of another model (red), the host facts or the engine base measured again (red), a kept
reload rate lost or its unsteady spread dropped on the way (red)."""
from __future__ import annotations

import pytest
import yaml

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import owed_facts as OF
from neurobrix.pytorch.prism import kept_launch_facts as K
from neurobrix.pytorch.prism import facts_probe as P
from nbxtensor.backend import host_measure as H

CALLS = []
C16 = {"brand": "nvidia", "model": "Card-16", "architecture": "unknownarch", "supports_dtypes": ["float16"],
       "memory_mb": 16384}
C32 = dict(C16, model="Card-32", memory_mb=32768)


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    CALLS.clear()
    monkeypatch.setattr(P, "matmul_tflops", lambda t, *a, **k: CALLS.append(("mm", t)) or {"best": 10.0})
    monkeypatch.setattr(P, "copies", lambda *a, **k: CALLS.append(("copies",)) or
                        {"memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0, "h2d_pageable_gbps": 5.0})
    monkeypatch.setattr(H, "host_matmul_gflops", lambda *a, **k: CALLS.append(("hmm",)) or 300.0)
    monkeypatch.setattr(H, "host_memory_gbps", lambda *a, **k: CALLS.append(("hmem",)) or 20.0)
    monkeypatch.setattr(P, "sweep_chunk", lambda d, *a, **k: CALLS.append(("vmm", d)) or ((32 << 20) * (1 + d), None))
    monkeypatch.setattr(P, "ruled_chunk", lambda *a, **k: None)   # a class the library's rule does not answer: swept
    monkeypatch.setattr(P, "vmm_first_replay", lambda *a, **k: None)
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda i: CALLS.append(("card", i)) or
                        {"runtime_total_mb": 16000 + i, "runtime_context_mb": 300})
    return tmp_path


def _write(hw, name, devices, **extra):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": devices, **extra}))


def _read(hw, name):
    return yaml.safe_load((hw / f"{name}.yml").read_text())


def test_a_card_set_of_a_measured_model_measures_nothing(hw):
    _write(hw, "default-four", [dict(C16, index=0), dict(C16, index=1), dict(C32, index=2), dict(C32, index=3)])
    A.ensure_measured_facts("default-four")
    assert ("vmm", 0) in CALLS and ("vmm", 2) in CALLS
    OF.measure_host_facts("default-four")          # a host-held plan priced there (line 5 of the ruling of 22/11/2026)
    four = _read(hw, "default-four")
    CALLS.clear()
    _write(hw, "default-one32", [dict(C32, index=0)])           # a 32 GB card alone: its own set, its own index
    added = A.ensure_measured_facts("default-one32")
    assert CALLS == [], CALLS
    one = _read(hw, "default-one32")
    got, want = one["devices"][0], four["devices"][2]
    assert {k: v for k, v in got["measured"].items() if k != "source"} == \
        {k: v for k, v in want["measured"].items() if k != "source"}
    assert got["measured"]["vmm_chunk_bytes"] == (32 << 20) * 3            # the 32 GB card's, never card 0's
    assert got["runtime_context_mb"] == want["runtime_context_mb"]
    assert one["host_measured"]["host_memory_gbps"] == four["host_measured"]["host_memory_gbps"]
    assert "default-four.yml" in got["measured"]["source"] and added["class_facts_from"]


def test_a_model_no_sibling_measured_is_measured(hw):
    _write(hw, "default-a", [dict(C16, index=0)])
    A.ensure_measured_facts("default-a")
    CALLS.clear()
    _write(hw, "default-b", [dict(C32, index=0)])
    A.ensure_measured_facts("default-b")
    assert ("vmm", 0) in CALLS and ("mm", "float16") in CALLS           # another model: measured on its own card
    assert not any(c[0] in ("hmm", "hmem") for c in CALLS)              # the host's facts were the machine's


def test_identical_cards_of_one_set_share_their_runtime_facts(hw):
    _write(hw, "default-two", [dict(C16, index=0), dict(C16, index=1)])
    A.ensure_measured_facts("default-two")
    assert [c for c in CALLS if c[0] == "card"] == [("card", 0)], CALLS


def test_a_kept_reload_rate_and_its_spread_come_along(hw):
    spread = {"low": 0.2, "high": 0.9, "measured": "2026-11-19 07:00:00 CET"}
    _write(hw, "default-four", [dict(C16, index=0)],
           host_measured={"host_matmul_gflops": 300, "host_memory_gbps": 20,
                          "weight_reload_gbps": {"triton": {"BF16": 0.2, "F32": 0.19}},
                          "weight_reload_spread": {"triton": {"BF16": spread}},
                          "weight_reload_under": K.card_key()})       # measured under this driver and library (1215)
    A.ensure_measured_facts("default-four")
    _write(hw, "default-one", [dict(C16, index=0)])
    A.ensure_measured_facts("default-one")
    host = _read(hw, "default-one")["host_measured"]
    assert host["weight_reload_gbps"]["triton"] == {"BF16": 0.2, "F32": 0.19}
    assert host["weight_reload_spread"]["triton"]["BF16"] == spread


def test_the_engine_base_is_the_machine_s(hw, monkeypatch):
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda: (_ for _ in ()).throw(AssertionError("measured")))
    _write(hw, "default-four", [dict(C16, index=0)], cpu={"model": "x", "runtime_base_mb": {"triton": 200}})
    _write(hw, "default-one", [dict(C16, index=0)], cpu={"model": "x"})
    A._complete_measured_fields(hw / "default-one.yml")
    assert _read(hw, "default-one")["cpu"]["runtime_base_mb"] == {"triton": 200}
