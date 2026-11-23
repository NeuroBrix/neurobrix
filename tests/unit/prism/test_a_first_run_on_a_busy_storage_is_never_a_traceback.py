"""A user's first `neurobrix run` never ends in a Traceback because its storage was busy, and no later launch pays a
read-rate trial again. On a clean container whose model cache was a network mount (2026-11-07), the first-launch
measure of the loaders' parallel reads found no stable read rate and its refusal ended the run; the fix of that day
read serially, kept no count and MEASURED AGAIN AT THE NEXT RUN — every kept-home Triton serve on the busy rack then
paid 130 to 367 s of trials before it was ready (2026-11-19). Now nothing is measured ahead and nothing is kept: the
weight reader measures itself on the model's own load (`core.workspace.weight_reader`). No card.

What this test would do if the code were wrong: a read-rate trial reached from the profile's completion (red), a count
written into a profile (red), the trial's code path still present in the detector (red)."""
from __future__ import annotations

import inspect

import pytest


def _no_trial(*a, **k):
    raise AssertionError("a read-rate trial ran ahead of the load")


@pytest.fixture
def no_trials(monkeypatch):
    import nbxtensor.io.read_rate as RR
    for name in ("workers_for", "measure", "trial"):          # the trial API leaves the library after this branch
        monkeypatch.setattr(RR, name, _no_trial, raising=False)


def test_a_profile_s_completion_measures_no_read_rate_and_keeps_no_count(no_trials, tmp_path, monkeypatch):
    import yaml
    from neurobrix.pytorch.prism import autodetect as A
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda: {"triton": 1})
    p = tmp_path / "default-x.yml"
    p.write_text(yaml.safe_dump({"cpu": {"model": "m", "runtime_base_mb": {"triton": 1}}}))
    A._complete_measured_fields(p)
    assert "io_workers" not in yaml.safe_load(p.read_text())["cpu"]


def test_the_detector_holds_no_read_rate_trial():
    from neurobrix.pytorch.prism import autodetect as A
    src = inspect.getsource(A)
    assert not hasattr(A, "_measure_io_workers")
    assert "workers_for" not in src and "io_workers" not in src
