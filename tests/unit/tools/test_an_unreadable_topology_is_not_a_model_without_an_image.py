"""A topology that cannot be read does not mean the model takes no image (2026-09-24).

`_declares_image_input` decides whether the campaign feeds an I2V/TI2V model its asset
image, by looking for `"global.image"` in the container's topology.json. It answered

    except OSError:
        return False

so an unreadable topology was indistinguishable from a model that needs no image. On this
Mac the container cache is an NFS mount over Wi-Fi at ~9 MB/s; a slow or stale read raises
OSError, the detector says "no image", and the census then runs a request the flow cannot
satisfy:

    RuntimeError: ZERO FALLBACK: None of the sources resolved: ['global.image']

That is what happened to three models in the 2026-09-22 Apple census. They were filed under
CENSUS HARNESS (input never supplied), and the recorded command proves the omission:

    neurobrix run --model CogVideoX-5b-I2V --prompt ... --seed 42 --steps 4 --triton ...

no `--input-image`, while `request_args` produces one for all three today and the models
each declare `global.image`. The family was recorded as `video`, so the family guard passed
and only the detector could have answered False.

The failure must be LOUD: a census that silently drops a required input does not measure the
model, and the result reads as an engine defect. Absence of the key is a real answer; being
unable to look is not.

Runnable: PYTHONPATH=src python3 -m pytest \
  tests/unit/tools/test_an_unreadable_topology_is_not_a_model_without_an_image.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

import precision_zoo_campaign as zoo  # noqa: E402


def _cache(tmp_path, model, topology):
    d = tmp_path / model
    d.mkdir(parents=True)
    if topology is not None:
        (d / "topology.json").write_text(json.dumps(topology))
    return tmp_path


def test_a_declared_image_is_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(zoo, "CACHE", _cache(tmp_path, "M", {"inputs": ["global.image"]}))
    assert zoo._declares_image_input("M") is True


def test_a_model_without_an_image_is_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(zoo, "CACHE", _cache(tmp_path, "M", {"inputs": ["global.prompt"]}))
    assert zoo._declares_image_input("M") is False


def test_a_missing_topology_refuses_instead_of_answering_no(tmp_path, monkeypatch):
    """The 2026-09-22 case: the read fails and the answer must not be a quiet False."""
    monkeypatch.setattr(zoo, "CACHE", _cache(tmp_path, "M", None))
    with pytest.raises(Exception) as got:
        zoo._declares_image_input("M")
    assert "topology" in str(got.value).lower(), str(got.value)


def test_an_unreadable_topology_refuses_instead_of_answering_no(tmp_path, monkeypatch):
    """A stale NFS mount raises OSError on read, not on exists()."""
    root = _cache(tmp_path, "M", {"inputs": ["global.image"]})

    real = Path.read_text

    def boom(self, *a, **k):
        if self.name == "topology.json":
            raise OSError(5, "Input/output error")      # what a stale mount gives
        return real(self, *a, **k)

    monkeypatch.setattr(zoo, "CACHE", root)
    monkeypatch.setattr(Path, "read_text", boom)
    with pytest.raises(Exception) as got:
        zoo._declares_image_input("M")
    assert "topology" in str(got.value).lower(), str(got.value)
