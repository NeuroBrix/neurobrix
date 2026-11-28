"""The engine loads a container of its NeuroTax major and a minor at most its own; any other is refused by name.

The minor rises when the vocabulary is extended (names added, none renamed or removed), so a
container of an older minor holds only names this engine knows and loads unchanged. A higher minor
holds names this engine does not know: refused, the message names both versions and says to update
neurobrix. Another major renamed or removed names: refused, both versions named. A manifest with no
version, or none readable, is refused, never guessed. Versions compare as integers: 5.10 is newer
than 5.9.

What these would do with the door of equality: a lower minor is refused; the higher-minor message
has no update instruction.
"""
from __future__ import annotations

import json

import pytest

from neurobrix.core.runtime.loader import NBXRuntimeLoader
from neurobrix.nbx.spec import REQUIRED_NBX_CONTAINER_VERSION
from neurobrix.pytorch.neurotax import NEUROTAX_VERSION, _neurotax_major_minor

MAJOR, MINOR = _neurotax_major_minor(NEUROTAX_VERSION)


def _container(tmp_path, manifest):
    # The directory carries the manifest's model_name: the loader refuses a misnamed container
    # before it reads the version, and that refusal would pass for the door (or hide it).
    d = tmp_path / manifest["model_name"]
    (d / "runtime").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    for rel in ("topology.json", "runtime/variables.json", "runtime/defaults.json"):
        (d / rel).write_text("{}")
    return d


def _load(tmp_path, manifest):
    NBXRuntimeLoader().load(str(_container(tmp_path, {"model_name": "m", "nbx_container_version": REQUIRED_NBX_CONTAINER_VERSION, **manifest})))


def _passes_the_door(tmp_path, version):
    try:
        _load(tmp_path, {"neurotax_version": version})
    except RuntimeError as e:
        assert "NEUROTAX VERSION" not in str(e)
    except Exception:
        pass  # the empty fixture fails further on; only the door is under test


@pytest.mark.parametrize("manifest", [{"neurotax_version": "0.1"}, {}, {"neurotax_version": 5.0},
                                      {"neurotax_version": "5"}])
def test_another_version_is_refused_by_name(tmp_path, manifest):
    with pytest.raises(RuntimeError, match="NEUROTAX VERSION") as e:
        _load(tmp_path, manifest)
    assert NEUROTAX_VERSION in str(e.value) and "FIX:" in str(e.value)


def test_the_engines_version_passes_the_door(tmp_path):
    _passes_the_door(tmp_path, NEUROTAX_VERSION)


def test_a_lower_minor_of_the_same_major_passes_the_door(tmp_path):
    assert MINOR >= 1
    _passes_the_door(tmp_path, f"{MAJOR}.{MINOR - 1}")
    _passes_the_door(tmp_path / "zero", f"{MAJOR}.0")


@pytest.mark.parametrize("minor", [MINOR + 1, MINOR + 10])
def test_a_higher_minor_is_refused_with_the_update_instruction(tmp_path, minor):
    newer = f"{MAJOR}.{minor}"
    with pytest.raises(RuntimeError, match="NEUROTAX VERSION") as e:
        _load(tmp_path, {"neurotax_version": newer})
    msg = str(e.value)
    assert f"NeuroTax {newer}," in msg and f"NeuroTax {NEUROTAX_VERSION}:" in msg
    assert "update neurobrix" in msg


@pytest.mark.parametrize("major", [MAJOR - 1, MAJOR + 1])
def test_another_major_is_refused_naming_both_versions(tmp_path, major):
    other = f"{major}.{MINOR}"
    with pytest.raises(RuntimeError, match="NEUROTAX VERSION") as e:
        _load(tmp_path, {"neurotax_version": other})
    msg = str(e.value)
    assert f"NeuroTax {other};" in msg and f"NeuroTax {NEUROTAX_VERSION}," in msg
    assert "another major" in msg
