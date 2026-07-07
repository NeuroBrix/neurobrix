"""P-ZERO-FALLBACK-SWEEP — registry_flags absent-vs-malformed boundary.

Pins the HIGH finding of engine audit #2 (2026-07-05): the registry
reader's former "never raises" contract meant a MALFORMED
forge/config/model_registry.yml silently disabled every per-component
annotation engine-wide (activations_fp16_safe, requires_fp32_compute —
the fp32-overflow protection).

Boundary under test:
  - ABSENT registry (deployed install, no build system co-located):
    legitimate → every flag read resolves to its documented default.
  - ABSENT model / component / flag in a well-formed registry:
    legitimate (annotations are opt-in) → default.
  - PRESENT-but-unreadable/malformed registry: ERROR → raises with an
    actionable message naming the file (ZERO FALLBACK).

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_registry_flags_zero_fallback.py
"""
from __future__ import annotations

import pytest

import neurobrix.core.runtime.registry_flags as rf


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Each test sees an unloaded registry cache (module-global)."""
    monkeypatch.setattr(rf, "_REGISTRY_CACHE", None)
    yield
    rf._REGISTRY_CACHE = None


def _point_registry_at(monkeypatch, path):
    monkeypatch.setattr(rf, "_find_registry_yaml", lambda: path)


# ── Legitimate absences → default ────────────────────────────────────

def test_absent_registry_returns_default(monkeypatch):
    _point_registry_at(monkeypatch, None)
    assert rf.get_component_flag("m", "c", "requires_fp32_compute",
                                 default=False) is False
    assert rf.get_component_flag("m", "c", "some_flag", default="d") == "d"


def test_empty_registry_file_returns_default(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text("")
    _point_registry_at(monkeypatch, p)
    assert rf.get_component_flag("m", "c", "f", default=42) == 42


def test_missing_model_component_flag_return_default(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text(
        "image:\n"
        "  known-model:\n"
        "    components:\n"
        "      vae_decoder:\n"
        "        requires_fp32_compute: true\n"
    )
    _point_registry_at(monkeypatch, p)
    # Present flag resolves.
    assert rf.get_component_flag("known-model", "vae_decoder",
                                 "requires_fp32_compute",
                                 default=False) is True
    # Absent flag / component / model → default (opt-in annotations).
    assert rf.get_component_flag("known-model", "vae_decoder",
                                 "activations_fp16_safe",
                                 default=False) is False
    assert rf.get_component_flag("known-model", "text_encoder",
                                 "requires_fp32_compute",
                                 default=False) is False
    assert rf.get_component_flag("other-model", "vae_decoder",
                                 "requires_fp32_compute",
                                 default=False) is False


def test_env_override_wins(monkeypatch):
    _point_registry_at(monkeypatch, None)
    monkeypatch.setenv("NBX_TEST_FLAG_OVERRIDE", "1")
    assert rf.get_component_flag(
        "m", "c", "f", default=False,
        env_override="NBX_TEST_FLAG_OVERRIDE") is True


# ── Malformed registry → raise ───────────────────────────────────────

def test_unparseable_registry_raises(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text("image:\n  model: [unclosed\n  bad: : :\n")
    _point_registry_at(monkeypatch, p)
    with pytest.raises(RuntimeError) as excinfo:
        rf.get_component_flag("m", "c", "f", default=False)
    msg = str(excinfo.value)
    assert "ZERO FALLBACK" in msg
    assert str(p) in msg


def test_non_mapping_registry_raises(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text("- just\n- a\n- list\n")
    _point_registry_at(monkeypatch, p)
    with pytest.raises(RuntimeError) as excinfo:
        rf.get_component_flag("m", "c", "f", default=False)
    assert "YAML mapping" in str(excinfo.value)


def test_unreadable_registry_raises(monkeypatch, tmp_path):
    p = tmp_path / "model_registry.yml"
    p.write_text("image: {}\n")
    p.chmod(0o000)
    _point_registry_at(monkeypatch, p)
    try:
        with pytest.raises(RuntimeError):
            rf.get_component_flag("m", "c", "f", default=False)
    finally:
        p.chmod(0o644)
