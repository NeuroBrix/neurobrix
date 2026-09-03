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

from neurobrix.kernels.ops import _configs


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


def test_a_cache_miss_is_announced(capsys):
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    tuned.run("shape-a", miss=True)
    assert "measuring kernel configurations" in capsys.readouterr().err


def test_a_cache_hit_says_nothing(capsys):
    """The warm path is the overwhelming majority of launches. A notice there
    would be noise on every single run."""
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    tuned.run("shape-a", miss=False)
    assert capsys.readouterr().err == ""


def test_it_is_said_once_not_once_per_kernel(capsys):
    """There are hundreds of autotuned kernels. One notice, not hundreds."""
    first, second = (_configs._announce_first_sweep(_FakeAutotuner())
                     for _ in range(2))
    first.run("a", miss=True)
    second.run("b", miss=True)
    err = capsys.readouterr().err
    assert err.count("measuring kernel configurations") == 1


def test_the_wrapper_removes_itself_from_the_hot_path():
    """Instrumentation that outlives its purpose is a permanent tax on every
    launch. Once the notice is out, the override is dropped and the class
    method takes over again."""
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    assert "run" in tuned.__dict__, "the override should be installed"
    tuned.run("a", miss=True)          # announces
    tuned.run("b", miss=False)         # sees the flag, removes itself
    assert "run" not in tuned.__dict__, (
        "the instance override must be gone once it has nothing left to say")


def test_the_notice_names_the_cost_and_where_it_goes(capsys):
    """A message that says 'please wait' without saying how long, whether it
    recurs, or where the result is kept, does not answer the question the user
    actually has."""
    tuned = _configs._announce_first_sweep(_FakeAutotuner())
    tuned.run("a", miss=True)
    err = capsys.readouterr().err
    assert "ONCE" in err, "must say it does not recur"
    assert "~/.triton/cache" in err, "must say where the result is kept"
    assert "254.9" in err and "7.9" in err, "must carry the measured figures"


def test_a_non_autotuner_is_returned_untouched():
    """`nbx_autotune` also decorates on Tritons that lack the keyword; the
    wrapper must not assume what it is handed."""
    class Plain:
        pass

    plain = Plain()
    assert _configs._announce_first_sweep(plain) is plain
