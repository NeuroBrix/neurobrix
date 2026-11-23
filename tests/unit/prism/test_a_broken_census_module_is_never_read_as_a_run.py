"""The guard of every first-launch and owed measure (`autodetect.generated_profile`) reads the census to know whether a
census shadow is running, which measures nothing. The census module ships with the engine: one that cannot be imported,
or that does not answer `active`, is a broken install raised by its own name, never read as "no census, a run" — which
would let a shadow measure a card (qa 1231-n1)."""
from __future__ import annotations

import sys

import pytest

import neurobrix.kernels as K
from neurobrix.pytorch.prism import autodetect as A
from neurobrix.kernels import census


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    path = tmp_path / "default-x.yml"
    path.write_text("devices: []\n")
    return path


def test_a_run_gets_its_generated_profile_and_a_census_shadow_gets_none(profile, monkeypatch):
    monkeypatch.setattr(census, "active", lambda: False)
    assert A.generated_profile("default-x") == profile
    assert A.generated_profile("c4140-committed") is None and A.generated_profile("default-absent") is None
    monkeypatch.setattr(census, "active", lambda: True)
    assert A.generated_profile("default-x") is None


def test_a_census_module_without_its_reader_is_raised_not_taken_for_a_run(profile, monkeypatch):
    monkeypatch.delattr(census, "active")
    with pytest.raises(AttributeError):
        A.generated_profile("default-x")


def test_a_census_module_that_cannot_be_imported_is_raised_not_taken_for_a_run(profile, monkeypatch):
    monkeypatch.delattr(K, "census")
    monkeypatch.setitem(sys.modules, "neurobrix.kernels.census", None)
    with pytest.raises(ImportError):
        A.generated_profile("default-x")
