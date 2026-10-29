"""The runtime's growth record (core/runtime/runtime_growth.py) is read only when it is this code's measurement, and a
census shadow writes none. 2026-10-29: the store held 0.0 MB for 35 of 48 models, written by the no-card census shadow
(the coverage check: its allocator answers no device), and 202 MB for Wan2.2-I2V-A14B from a branch's other reading
where this code measures 872 MB; Wan2.2's 16 GB plan, cut to 202 MB of growth, ran out of memory at an attention.

What each cell would do if the code were wrong: a shadow run recording a figure (red); a figure that states no method,
or another method, read as a measurement (red); a larger measurement of the same method not kept (red)."""
import json

import pytest

from neurobrix.core.runtime import runtime_growth as RG


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "_root", lambda: tmp_path)
    return tmp_path


def test_a_census_shadow_records_nothing(store, monkeypatch):
    from neurobrix.kernels import census
    monkeypatch.setitem(census._ACTIVE, "census", True)
    assert RG.record("M", "triton", "volta", {0: 300.0}, {0: 300.0}) is None
    assert not RG.path("M").exists()


def test_a_figure_of_no_method_or_another_is_not_read(store):
    RG.path("M").parent.mkdir(parents=True)
    RG.path("M").write_text(json.dumps({"triton": {"volta": 202.4}}))
    assert RG.load("M", "triton", "volta") is None
    RG.path("M").write_text(json.dumps({"triton": {"volta": {"mb": 202.4, "method": "a branch's height reading"}}}))
    assert RG.load("M", "triton", "volta") is None


def test_this_methods_largest_measurement_is_kept_and_read(store):
    RG.path("M").parent.mkdir(parents=True)
    RG.path("M").write_text(json.dumps({"triton": {"volta": 202.4}}))           # a legacy figure is replaced
    assert RG.record("M", "triton", "volta", {0: 310.0}, {0: 1182.5}) == 872.5
    assert RG.load("M", "triton", "volta") == 872.5
    RG.record("M", "triton", "volta", {0: 310.0}, {0: 610.0})                    # a smaller run keeps the larger
    assert RG.load("M", "triton", "volta") == 872.5
