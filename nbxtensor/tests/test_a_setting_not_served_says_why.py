"""A certified setting the directory holds but does not serve says why: proven under another code generator than the
running one, or on another memory class. The Mac's new-user run on the published Metal wheels (2026-11-07): every Apple
entry was proven under the development environment's generator, none was served, and the line said "certified for
18 GB, this card is 18 GB, not served (register 56)" — the memory class matched and the cause was not named.

What this test would do if the code were wrong: a generator mismatch reported as a memory class (red), a memory-class
miss reported as a generator (red)."""
from __future__ import annotations

from nbxtensor.ops import autotune_certified as C


def _entry(gen_version: str, memory_mb: int):
    return {"proof": {"backend": {"triton": gen_version, "name": "metal"},
                      "machine": {"device": {"memory_mb": memory_mb}}}}


def test_a_generator_mismatch_on_the_right_memory_class_names_the_generators(monkeypatch):
    monkeypatch.setattr(C, "running_generator", lambda: "triton 3.8.0+git4a15f415.nbx1 metal")
    raw = _entry("3.8.0+git4a15f415", 18432)
    gens = sorted(C.proof_backends(raw))
    why = C._not_served_because(raw, 18)
    assert why.startswith("certified for this card's 18 GB under "), why
    assert gens[0] in why and "triton 3.8.0+git4a15f415.nbx1 metal" in why and "--reprove-generator" in why, why
    assert "register 56" not in why


def test_another_memory_class_is_said_as_such(monkeypatch):
    raw = _entry("3.8.0", 32768)
    monkeypatch.setattr(C, "running_generator", lambda: sorted(C.proof_backends(raw))[0])
    why = C._not_served_because(raw, 16)
    assert "certified for 32 GB, this card is 16 GB, not served (register 56)" == why, why
