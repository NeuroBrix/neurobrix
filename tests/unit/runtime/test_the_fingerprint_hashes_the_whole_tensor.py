"""The op-output fingerprint hashes the whole tensor unless a cap is asked for by name.

On 2026-09-20 the instrument hashed the first 8 192 bytes of every output by default and
named, for Kokoro, a layer-norm mean as "the first differing op" while the layer-norm's
INPUT already differed past byte 8 192; the levers were two `aten::pow` commits, found by
bisection. Every attribution taken on the prefix is void (the owner, 2026-09-21).

What would this file do if the code were wrong? A default that still caps → the first
cell reads 8 192 and fails; a cap that is not honoured when asked → the second fails; two
buffers equal on the prefix and different past it hashing the same → the third fails.
"""
from __future__ import annotations

import hashlib

from neurobrix.core.runtime.fingerprint import ENV_CAP, hashed_span


def test_the_default_is_the_whole_tensor(monkeypatch):
    monkeypatch.delenv(ENV_CAP, raising=False)
    assert hashed_span(24_576) == 24_576
    assert hashed_span(36) == 36


def test_a_cap_is_an_explicit_choice(monkeypatch):
    monkeypatch.setenv(ENV_CAP, "8192")
    assert hashed_span(24_576) == 8_192
    assert hashed_span(36) == 36
    monkeypatch.setenv(ENV_CAP, "0")
    assert hashed_span(24_576) == 24_576


def test_a_difference_past_the_old_prefix_changes_the_hash(monkeypatch):
    monkeypatch.delenv(ENV_CAP, raising=False)
    a = bytearray(20_000)
    b = bytearray(20_000)
    b[15_000] = 1                                    # identical on the first 8 192 bytes
    span = hashed_span(len(a))
    ha = hashlib.sha256(bytes(a[:span])).hexdigest()
    hb = hashlib.sha256(bytes(b[:span])).hexdigest()
    assert ha != hb
    monkeypatch.setenv(ENV_CAP, "8192")
    old = hashed_span(len(a))                         # the instrument as it stood: blind to byte 15 000
    assert hashlib.sha256(bytes(a[:old])).hexdigest() == hashlib.sha256(bytes(b[:old])).hexdigest()


def test_both_fingerprint_paths_take_their_span_from_the_one_door():
    """The door above was written (2026-09-21) and its cells were green, but both engines still read the cap
    themselves with an 8 192-byte default — the ATen sequential loop (graph_executor) and the Triton sequence — so the
    instrument kept hashing the prefix the owner had voided, and this file tested a function nothing called (the
    engine-slim scan found it dead, 2026-10-29). No engine file but the door reads the cap; the two paths call it."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[3] / "src" / "neurobrix"
    readers = sorted(str(p.relative_to(src)) for p in src.rglob("*.py")
                     if ENV_CAP in p.read_text() and p.name != "fingerprint.py")
    assert not readers, f"the fingerprint cap read outside its door: {readers}"
    for rel in ("core/runtime/graph_executor.py", "triton/sequence.py"):
        assert "fingerprint import hashed_span" in (src / rel).read_text(), f"{rel} hashes its own span"
