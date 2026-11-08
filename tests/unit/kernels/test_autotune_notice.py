"""The engine says it is tuning, once, instead of looking hung.

A first `whisper-large-v3-turbo` transcription spends **247 seconds**
autotuning before producing a word, and printed nothing while it did. The
second run of the identical command takes 7.9 s. Measured 2026-09-03,
`validation_outputs/audio_launch_census_2026_09_03/VERDICT.md`.

The cost is legitimate — it buys ~12 % throughput per the autotune policy and
is paid once per shape per machine. The silence was not: nothing on screen
distinguished four minutes of tuning from a hang, which is the same class of
defect as the CUDA/Volta mismatch that 0.5.3 was published to fix.

These pins cover the two ways a notice like this goes wrong: saying nothing,
and never shutting up.
"""

from __future__ import annotations

import pytest

from nbxtensor.ops import _configs


class _FakeAutotuner:
    """Stands in for triton's Autotuner: a `cache` that grows on a miss."""

    def __init__(self):
        self.cache = {}
        self.calls = 0

    def run(self, key, miss=False):
        self.calls += 1
        if miss:
            self.cache[key] = "config"
        return "result"


@pytest.fixture(autouse=True)
def _reset_announced():
    _configs._SWEEP_ANNOUNCED[0] = False
    yield
    _configs._SWEEP_ANNOUNCED[0] = False


def test_a_cache_miss_is_announced(engine_log):
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    tuned.run("shape-a", miss=True)
    assert "measuring kernel configurations" in engine_log()


def test_a_cache_hit_says_nothing(engine_log):
    """The warm path is the overwhelming majority of launches. A notice there
    would be noise on every single run."""
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    tuned.run("shape-a", miss=False)
    assert engine_log() == ""


def test_it_is_said_once_not_once_per_kernel(engine_log):
    """There are hundreds of autotuned kernels. One notice, not hundreds."""
    first, second = (_configs._announce_first_sweep(_FakeAutotuner())
                     for _ in range(2))
    first.run("a", miss=True)
    second.run("b", miss=True)
    err = engine_log()
    assert err.count("measuring kernel configurations") == 1


def test_the_wrapper_stays_and_announces_once():
    """Since 2026-09-05 the wrapper is the engine's persistence of a sweep
    (upstream's on-disk cache was dropped: its key was a torch-importing
    driver probe): it stays installed to capture every new selection, and
    what it costs a warm launch is one `len(cache)` comparison. The notice
    is printed once."""
    import io
    import sys
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    assert "run" in tuned.__dict__, "the override should be installed"
    tuned.run("a", miss=True)          # announces
    tuned.run("b", miss=False)         # stays: a later sweep must be captured too
    assert "run" in tuned.__dict__, "the override persists the sweeps; it must stay"


def test_the_notice_names_the_cost_and_where_it_goes(engine_log):
    """A message that says 'please wait' without saying how long, whether it
    recurs, or where the result is kept, does not answer the question the user
    actually has."""
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    tuned.run("a", miss=True)
    err = engine_log()
    assert len(err.splitlines()) == 1, "one line, not a page"
    assert "once" in err, "must say it does not recur"
    assert "kept for later runs, which skip it" in err, "must say the result is kept and later runs skip it"


def test_a_non_autotuner_is_returned_untouched():
    """`nbx_autotune` also decorates on Tritons that lack the keyword; the
    wrapper must not assume what it is handed."""
    class Plain:
        pass

    plain = Plain()
    assert _configs._announce_first_sweep(plain) is plain
