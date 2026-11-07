"""The Triton arena's factor observed on the user's own runs (voie B, the supervisor 2026-11-07 17:05): one allocator
window around the FIRST run of each component a Triton plan runs whole, its hold over the walk Prism priced; written into
the generated profile when three observations agree, read by Prism after the device's vendor file and before its
family. No run in addition, no shape the run does not form. No card.

What this test would do if the code were wrong: a factor written from disagreeing or too few observations (red), a
profile the user named written (red), an observed value read over an architecture's declaration or ignored over the
family's (red), a second run of the same component observed again (red)."""
from __future__ import annotations

import types

import pytest
import yaml

from neurobrix.core.prism import arena_observation as O


def test_three_agreeing_observations_give_their_median_else_nothing():
    assert O.agreed([1.40, 1.41], 0.05) is None
    assert O.agreed([1.40, 1.41, 1.39], 0.05) == pytest.approx(1.40)
    assert O.agreed([1.40, 1.41, 2.37], 0.05) is None


def _profile(tmp_path, monkeypatch, hw, measured=None):
    from neurobrix.core.prism import autodetect as AD
    monkeypatch.setattr(AD, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(AD, "ACTIVE_HARDWARE_ID", hw)
    (tmp_path / f"{hw}.yml").write_text(yaml.safe_dump({"devices": [{"model": "M", "measured": measured or {}}]}))
    return tmp_path / f"{hw}.yml"


def test_the_factor_is_written_into_a_generated_profile_once_three_agree(tmp_path, monkeypatch):
    p = _profile(tmp_path, monkeypatch, "default-x")
    for c, ratio in (("a", 1.40), ("b", 1.42), ("c", 1.41)):
        assert O.observed_factor("default-x") is None
        O.record(c, int(ratio * 1000), 1000)
    m = yaml.safe_load(p.read_text())["devices"][0]["measured"]
    assert m["triton_arena_activation_factor_observed"] == pytest.approx(1.41) and len(m["arena_observations"]) == 3


def test_a_named_profile_is_never_observed(tmp_path, monkeypatch):
    _profile(tmp_path, monkeypatch, "my-lab-profile")
    assert O._wanted(1000) is False
    _profile(tmp_path, monkeypatch, "default-y", {"triton_arena_activation_factor_observed": 1.4})
    assert O._wanted(1000) is False                     # already observed: no window any more


def test_only_the_first_run_of_a_component_is_watched(tmp_path, monkeypatch):
    _profile(tmp_path, monkeypatch, "default-z")
    import nbxtensor.memory.allocator as AL
    opened = []
    monkeypatch.setattr(AL.DeviceAllocator, "device_synchronize", staticmethod(lambda d: None))
    monkeypatch.setattr(AL.DeviceAllocator, "open_window", staticmethod(lambda d: opened.append(d) or "tok"))
    ex = types.SimpleNamespace()
    assert O.open_first(ex, 1000, 0) == "tok"
    assert O.open_first(ex, 1000, 0) is None and opened == [0]


def test_prism_reads_the_observed_factor_after_the_vendor_file_and_before_the_family(monkeypatch):
    from neurobrix.core.prism import solver as S
    dev = types.SimpleNamespace(brand="apple", architecture="apple-m2", compute_capability="0.0", name="Apple M2",
                                measured={"triton_arena_activation_factor_observed": 1.52}, memory_mb=1,
                                device_string="mps:0")
    assert S.PrismSolver()._triton_arena_factor(types.SimpleNamespace(spec=dev)) == pytest.approx(1.52)
    dev.brand, dev.architecture, dev.compute_capability, dev.name = "nvidia", "volta", "7.0", "Tesla V100-SXM2-32GB"
    assert S.PrismSolver()._triton_arena_factor(types.SimpleNamespace(spec=dev)) == pytest.approx(1.35)   # its file
