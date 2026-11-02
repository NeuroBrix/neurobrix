"""A machine's GENERATED hardware profiles (`default`, `default-<tag>`) live in ONE place, beside its measured-facts
store, never inside a source tree (the team's review, 2026-11-02: every worktree carried its own copy of the rack's
profiles, four different versions of default-ff6008b7 at once, and a fact the engine measured in one tree never
reached the others).

What this test would do if the code were wrong: a generated name resolved inside the package, the engine writing its
detection or a measured fact into the package, a stale copy in the package read silently instead of refused, or the
machine's profiles missing from the listing — RED."""
from __future__ import annotations

import pytest
import yaml

from neurobrix.core.dtype import calibration
from neurobrix.core.prism import autodetect as A
from neurobrix.core.prism import loader as L


def test_the_generated_profiles_sit_beside_the_measured_facts():
    assert L.GENERATED_DIR.parent == calibration.STORE_ROOT.parent
    assert L.GENERATED_DIR != L.HARDWARE_DIR and L.HARDWARE_DIR.parent.parent == L.PROJECT_ROOT


def test_a_generated_name_resolves_there_and_a_committed_one_in_the_package(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "HARDWARE_DIR", tmp_path / "package")
    monkeypatch.setattr(L, "GENERATED_DIR", tmp_path / "home")
    assert L.profile_path("default-ff6008b7") == tmp_path / "home" / "default-ff6008b7.yml"
    assert L.profile_path("default") == tmp_path / "home" / "default.yml"
    assert L.profile_path("c4140-4xv100-16GB-nvlink") == tmp_path / "package" / "c4140-4xv100-16GB-nvlink.yml"


def test_a_generated_copy_in_the_package_is_refused_by_name(tmp_path, monkeypatch):
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "default-ff6008b7.yml").write_text("id: stale\n")
    monkeypatch.setattr(L, "HARDWARE_DIR", tmp_path / "package")
    monkeypatch.setattr(L, "GENERATED_DIR", tmp_path / "home")
    with pytest.raises(RuntimeError, match="generated hardware profile sits in the package"):
        L.load_profile("default-ff6008b7")


def test_the_engine_writes_its_detection_and_its_facts_there():
    assert A.HARDWARE_DIR == L.GENERATED_DIR and A.DEFAULT_PROFILE_PATH.parent == L.GENERATED_DIR


def test_the_listing_holds_both_places(tmp_path, monkeypatch):
    for d, name in (("package", "c4140-x"), ("home", "default-abc")):
        (tmp_path / d).mkdir()
        (tmp_path / d / f"{name}.yml").write_text(yaml.safe_dump({"id": name}))
    monkeypatch.setattr(L, "HARDWARE_DIR", tmp_path / "package")
    monkeypatch.setattr(L, "GENERATED_DIR", tmp_path / "home")
    assert L.list_available_profiles() == ["c4140-x", "default-abc"]
