"""A long run must be distinguishable from a hung one.

A 100-step 720p Allegro render takes about fourteen hours on a V100 and, until
2026-09-04, printed nothing at all for the whole of it. There is no way to tell
that from a hang, and the operator cannot know whether to wait or to kill it.
It is a product defect before it is a tooling one: anyone rendering video with
this engine saw the same blank terminal.

The two failure modes of a fix like this are opposite, and both are pinned
here: saying nothing on a long run, and spamming a short one.
"""

from __future__ import annotations

import io

import pytest

from neurobrix.core.runtime.progress import StepProgress


def _p(total=100, interval=0.0, **kw):
    return StepProgress(total, stream=io.StringIO(), interval_s=interval, **kw)


# --- it speaks on a long run ------------------------------------------------

def test_a_due_step_is_reported_with_position_and_elapsed():
    p = _p()
    p.step(0)
    out = p.stream.getvalue()
    assert "step 1/100" in out
    assert "elapsed" in out


def test_it_projects_the_remainder():
    """"Still going" is not enough — the operator needs to know whether to
    wait ten minutes or ten hours."""
    p = _p()
    p.step(0)
    assert "remaining" in p.stream.getvalue()


def test_the_label_names_the_run():
    """A terminal with several renders in it must say which one is moving."""
    p = _p(label="Allegro-TI2V")
    p.step(0)
    assert "Allegro-TI2V" in p.stream.getvalue()


# --- it stays quiet on a short one ------------------------------------------

def test_a_run_shorter_than_the_interval_says_nothing():
    """The same rule has to cover a 20-step image render finishing in seconds
    and a 100-step video render taking hours. Time-gating is that rule."""
    p = _p(interval=3600.0)
    for i in range(20):
        p.step(i)
    p.done()
    assert p.stream.getvalue() == ""


def test_the_completion_line_only_appears_if_the_run_ever_spoke():
    quiet = _p(interval=3600.0)
    quiet.step(0)
    quiet.done()
    assert quiet.stream.getvalue() == ""

    loud = _p(interval=0.0)
    loud.step(0)
    loud.done()
    assert "complete" in loud.stream.getvalue()


def test_it_can_be_silenced_entirely(monkeypatch):
    monkeypatch.setenv("NBX_NO_PROGRESS", "1")
    p = _p()
    p.step(0)
    p.done()
    assert p.stream.getvalue() == ""


def test_the_interval_is_configurable(monkeypatch):
    monkeypatch.setenv("NBX_PROGRESS_EVERY", "0")
    p = StepProgress(10, stream=io.StringIO())
    p.step(0)
    assert p.stream.getvalue() != ""


def test_a_nonsense_interval_falls_back_instead_of_crashing(monkeypatch):
    """A malformed env var must never take down a fourteen-hour render."""
    monkeypatch.setenv("NBX_PROGRESS_EVERY", "not-a-number")
    p = StepProgress(10, stream=io.StringIO())
    assert p.interval > 0


# --- it can never be the thing that kills the run ---------------------------

def test_a_zero_step_loop_is_not_a_division_by_zero():
    p = _p(total=0)
    p.step(0)
    p.done()


def test_a_closed_stream_is_not_fatal():
    """Progress goes to a pipe. A consumer that exits early must cost the run
    its heartbeat, not its result."""
    p = _p()
    p.stream.close()
    p.step(0)
    p.done()


def test_it_writes_to_stderr_by_default():
    """stdout carries the model's OUTPUT and is parsed by the harness; a
    heartbeat there would corrupt what callers read."""
    import sys
    assert StepProgress(10).stream is sys.stderr


def test_the_module_is_torch_free():
    """R33: the triton engine needs the same heartbeat, and both modes must
    share ONE implementation rather than drifting apart (R30)."""
    import inspect

    from neurobrix.core.runtime import progress

    assert "import torch" not in inspect.getsource(progress)
