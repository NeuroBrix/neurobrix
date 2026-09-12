"""A configuration seated without an oracle is recorded as what it is.

The owner's ruling, 2026-09-12: the bare consensus screen may go on ranking by
speed — the engine must never refuse to run — but it may no longer produce a
verdict that reads as a validation. The distinction to hold is between *"this
configuration was validated"* and *"this configuration was the fastest among
candidates nobody verified"*. Until this record existed the two were written the
same way and read the same way.

Three things this pins, because the ruling has three parts:

* the seating still happens — nothing here refuses to run;
* the provenance is recorded and says plainly what it is;
* it cannot reach the certified directory, which is a different path filled only
  by `neurobrix autotune certify` and its own fp64 oracle.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_an_unscreened_configuration_says_so.py
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import launcher


@pytest.fixture(autouse=True)
def _clean():
    launcher.clear_screened()
    yield
    launcher.clear_screened()


def test_seating_without_an_oracle_still_returns_the_configs():
    """The engine runs. A provenance is not a refusal."""
    configs = ["a", "b", "c"]
    got = launcher._seat_unscreened("matmul_kernel", (1, 2), configs, 3, "no oracle")
    assert got is configs


def test_the_provenance_is_recorded_with_its_reason():
    launcher._seat_unscreened("conv2d_forward_kernel", (64, 64), ["a"], 7,
                              "the provider covers no oracle for this kernel")
    (entry,) = launcher.unscreened()
    assert entry.kernel == "conv2d_forward_kernel"
    assert entry.key == (64, 64)
    assert entry.candidates == 7
    assert "covers no oracle" in entry.reason


def test_the_announcement_does_not_read_as_a_validation(capsys):
    """The words are the point: this line is read by someone deciding whether a
    number can be quoted."""
    launcher._seat_unscreened("matmul_kernel", (8,), ["a"], 4, "no oracle provider")
    out = capsys.readouterr().out
    assert "UNSCREENED" in out
    assert "not a validated setting" in out.lower()
    assert "nothing verified" in out.lower()
    assert "never written to the certified directory" in out
    # And it must NOT contain the words that read as an endorsement.
    for word in ("validated setting is", "verified", "certified for"):
        assert f" {word} " not in out.lower().replace("nothing verified", "")


def test_clearing_forgets_both_registers():
    launcher._seat_unscreened("matmul_kernel", (8,), ["a"], 4, "no oracle")
    assert launcher.unscreened()
    launcher.clear_screened()
    assert launcher.unscreened() == []


def test_the_reason_names_the_provider_state():
    """Three distinguishable causes, because the remedy differs for each."""
    assert "no oracle provider is installed" in launcher._no_oracle_reason(None)
    launcher.set_screen_oracle(lambda *a: None)
    try:
        assert "covers no oracle" in launcher._no_oracle_reason(None)
        assert "produced no reference" in launcher._no_oracle_reason(object())
    finally:
        launcher.set_screen_oracle(None)


def test_the_certified_directory_is_a_different_path_entirely():
    """Structural, not a promise: the runtime cache writes under the machine's
    replay cache and the directory lives in the installed package."""
    from neurobrix.triton import autotune_cache
    assert "replay_cache" in autotune_cache._DIR
    assert "config/autotune" not in autotune_cache._DIR
