"""A launch whose key cannot be formed is REFUSED in a confirmation run, never swept.

`run_with_notice` forms the key with `autotune_cache.key_of`; when that raised, the key became None, and
the certified-only check (`key is not None and key not in cache`) was skipped: the call went on to the
autotuner, which swept at runtime inside a run that promised to serve the certified directory and
nothing else (found reading the door, 2026-11-03, the list of what remains, 1.1).

What this test would do if the code were wrong: with the refusal removed, `key_of` raising lets the
fake autotuner run — `test_an_unformable_key_fails_a_confirmation_run` sees the call and fails (seen red).
The control: the same unformable key outside a confirmation run still reaches the autotuner.
"""
from __future__ import annotations

import pytest

from nbxtensor.ops import _configs
from nbxtensor.ops import autotune_cache as _atc
from nbxtensor.ops import autotune_certified as _cert


class _FakeAutotuner:
    """Stands in for triton's Autotuner: a `cache`, and a `run` that counts the sweeps it would do."""

    def __init__(self):
        self.cache = {}
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return "result"


@pytest.fixture
def unformable(monkeypatch):
    def _raise(*a, **k):
        raise KeyError("an argument the key names is absent")
    monkeypatch.setattr(_atc, "key_of", _raise)
    monkeypatch.setattr(_configs, "_SEEDED", [True])


def test_an_unformable_key_fails_a_confirmation_run(unformable, monkeypatch):
    monkeypatch.setenv("NBX_AUTOTUNE_CERTIFIED_ONLY", "1")
    fake = _FakeAutotuner()
    tuned = _configs._announce_first_sweep(fake)
    with pytest.raises(_cert.KeyNotCertified, match="cannot be formed"):
        tuned.run(1, 2)
    assert fake.calls == 0, "a confirmation run reached the autotuner with no key"


def test_outside_a_confirmation_run_the_launch_still_runs(unformable, monkeypatch):
    monkeypatch.delenv("NBX_AUTOTUNE_CERTIFIED_ONLY", raising=False)
    fake = _FakeAutotuner()
    tuned = _configs._announce_first_sweep(fake)
    assert tuned.run(1, 2) == "result"
    assert fake.calls == 1
