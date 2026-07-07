"""P-ZERO-FALLBACK-SWEEP — fp32-protection reads propagate failures.

Pins the HIGH finding of engine audit #2 (2026-07-05): the swallow
sites in `PrismSolver._components_force_fp32` / `_auto_fp32_components`
silently returned an EMPTY fp32 pin set on a broken container / family
config / hardware profile — disabling the fp32-overflow protection
engine-wide (NaN outputs on fp16-unsafe architectures) with zero error.

Boundary under test:
  - broken container / profile / unknown family: ERROR → propagates.
  - legitimate absence (no family in the manifest, no flag annotation,
    policy absent from the family YAML): still resolves to the empty
    set — optional-with-default reads are NOT converted into crashes.

Run: PYTHONPATH=src python -m pytest tests/unit/dtype/test_force_fp32_zero_fallback.py
"""
from __future__ import annotations

from typing import Dict, List

import pytest

from neurobrix.core.prism.solver import PrismSolver


class _StubComponent:
    def __init__(self, name: str, graph: Dict):
        self.name = name
        self.graph = graph


class _StubContainer:
    def __init__(self, manifest: Dict, components: List[_StubComponent]):
        self._manifest = manifest
        self._components = components

    def get_manifest(self) -> Dict:
        return self._manifest

    def get_neural_components(self) -> List[_StubComponent]:
        return self._components


class _BrokenManifestContainer(_StubContainer):
    def get_manifest(self):
        raise OSError("manifest read failed")


class _BrokenComponentsContainer(_StubContainer):
    def get_neural_components(self):
        raise OSError("component scan failed")


class _StubProfile:
    def __init__(self, supports_bf16: bool):
        self._bf16 = supports_bf16
        self.preferred_dtype = "float16"

    def devices_support_dtype(self, dtype: str) -> bool:
        return self._bf16 if dtype == "bfloat16" else True


class _BrokenProfile(_StubProfile):
    def devices_support_dtype(self, dtype: str) -> bool:
        raise OSError("profile query failed")


def _comp(name: str = "block") -> _StubComponent:
    return _StubComponent(name, {"torch_dtype": "float32", "ops": {}})


# ── Errors propagate ────────────────────────────────────────────────

def test_broken_manifest_propagates():
    c = _BrokenManifestContainer({}, [_comp()])
    with pytest.raises(OSError, match="manifest read failed"):
        PrismSolver()._components_force_fp32(c, _StubProfile(False))


def test_broken_component_scan_propagates():
    c = _BrokenComponentsContainer({"family": None}, [_comp()])
    with pytest.raises(OSError, match="component scan failed"):
        PrismSolver()._components_force_fp32(c, _StubProfile(False))


def test_unknown_family_propagates():
    """A manifest naming a family with no config YAML is malformed data:
    the loader's ZERO FALLBACK FileNotFoundError must reach the caller
    (the former swallow silently disabled the auto-detect)."""
    c = _StubContainer({"family": "definitely_not_a_family"}, [_comp()])
    with pytest.raises(FileNotFoundError):
        PrismSolver()._components_force_fp32(c, _StubProfile(False))


def test_broken_profile_propagates(monkeypatch):
    """The bf16 hardware gate reads the profile; a broken profile is an
    ERROR, not a reason to silently keep scanning."""
    # 'image' has a real family YAML with the overflow policy enabled,
    # so the gate is actually consulted.
    c = _StubContainer({"family": "image"}, [_comp()])
    monkeypatch.delenv("NBX_DISABLE_AUTO_FP32", raising=False)
    with pytest.raises(OSError, match="profile query failed"):
        PrismSolver()._components_force_fp32(c, _BrokenProfile(False))


# ── Legitimate absences stay defaults ───────────────────────────────

def test_no_family_yields_empty_set(monkeypatch):
    monkeypatch.delenv("NBX_DISABLE_AUTO_FP32", raising=False)
    c = _StubContainer({}, [_comp()])
    assert PrismSolver()._components_force_fp32(c, _StubProfile(False)) == set()


def test_family_without_policy_yields_empty_set(monkeypatch):
    """'llm' has a real family YAML with no auto_fp32 overflow policy:
    default-absent ⇒ disabled ⇒ empty set (not a crash)."""
    monkeypatch.delenv("NBX_DISABLE_AUTO_FP32", raising=False)
    c = _StubContainer({"family": "llm"}, [_comp()])
    assert PrismSolver()._components_force_fp32(c, _StubProfile(False)) == set()


def test_disable_env_short_circuits_auto(monkeypatch):
    """NBX_DISABLE_AUTO_FP32=1 keeps manual-only behavior even for an
    unknown family (the auto-detect path is bypassed for diagnosis)."""
    monkeypatch.setenv("NBX_DISABLE_AUTO_FP32", "1")
    c = _StubContainer({"family": "definitely_not_a_family"}, [_comp()])
    assert PrismSolver()._components_force_fp32(c, _StubProfile(False)) == set()
