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
    assert hashed_span(24_576, cap=0) == 24_576


def test_a_difference_past_the_old_prefix_changes_the_hash(monkeypatch):
    monkeypatch.delenv(ENV_CAP, raising=False)
    a = bytearray(20_000)
    b = bytearray(20_000)
    b[15_000] = 1                                    # identical on the first 8 192 bytes
    span = hashed_span(len(a))
    ha = hashlib.sha256(bytes(a[:span])).hexdigest()
    hb = hashlib.sha256(bytes(b[:span])).hexdigest()
    assert ha != hb
    old = hashed_span(len(a), cap=8192)               # the instrument as it stood: blind to byte 15 000
    assert hashlib.sha256(bytes(a[:old])).hexdigest() == hashlib.sha256(bytes(b[:old])).hexdigest()
