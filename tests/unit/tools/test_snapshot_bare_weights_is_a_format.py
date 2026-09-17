"""A bare checkpoint is a format the toolchain reads, and the re-trace door must read it too.

The door listed four layouts. The build toolchain's detector accepts a fifth — bare weights,
a `.pth`/`.pt`/`.safetensors`/`.ckpt` with no config of any kind — which is exactly how every
upscaler ships. So a re-trace of real-esrgan-x2 stopped with "no COMPLETE snapshot" while the
checkpoint sat in the directory it had just been told to read (2026-09-16).

The fix may not re-open what the door was built for: a STOPPED diffusers download is bare
weights too (Sana 4K, 6 GB of shards, no model_index.json, 2026-09-07) and must stay refused.
The discriminant is the registry — it NAMES the checkpoint for the models that ship as one,
because several variants live in one upstream repository. Bare weights are admitted only when
the entry names the file and the file is there.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import retrace_zoo as R  # noqa: E402


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A registry the test owns, read the way the tool reads the real one."""
    forge = tmp_path / "forge"
    (forge / "config").mkdir(parents=True)
    (forge / "config" / "model_registry.yml").write_text(
        "models:\n"
        "  upscaler-two:\n"
        "    family: upscaler\n"
        '    checkpoint_file: "Weights_x2.pth"\n'
        "  a-diffusers-pipeline:\n"
        "    family: image\n"
    )
    monkeypatch.setattr(R, "FORGE", forge / "forge.py")
    return forge


def test_the_registry_names_the_checkpoint(registry):
    assert R.registry_checkpoint_file("upscaler-two") == "Weights_x2.pth"
    assert R.registry_checkpoint_file("a-diffusers-pipeline") is None
    assert R.registry_checkpoint_file("not-in-the-registry") is None


def test_a_bare_checkpoint_the_registry_names_is_a_format(tmp_path, registry):
    """The case the door refused for its whole life."""
    snap = tmp_path / "upscaler-two"
    snap.mkdir()
    (snap / "Weights_x2.pth").write_bytes(b"\x00" * 16)
    assert R.snapshot_has_a_format(snap, "upscaler-two") is True


def test_the_checkpoint_the_registry_names_must_be_the_one_present(tmp_path, registry):
    """A sibling variant's file is not this model's checkpoint — one repository, several."""
    snap = tmp_path / "upscaler-two"
    snap.mkdir()
    (snap / "Weights_x8.pth").write_bytes(b"\x00" * 16)
    assert R.snapshot_has_a_format(snap, "upscaler-two") is False


def test_a_stopped_diffusers_download_stays_refused(tmp_path, registry):
    """What the door was built for: shards, no index, and no entry naming a checkpoint.

    This is the injection that turns the test red against a fix that simply accepts bare
    weights: the directory below IS bare weights by the detector's rule.
    """
    snap = tmp_path / "a-diffusers-pipeline"
    snap.mkdir()
    (snap / "diffusion_pytorch_model-00001-of-00003.safetensors").write_bytes(b"\x00" * 16)
    assert R.snapshot_has_a_format(snap, "a-diffusers-pipeline") is False


def test_the_four_layouts_are_untouched(tmp_path, registry):
    for filename in ("model_index.json", "config.json"):
        snap = tmp_path / filename.replace(".", "_")
        snap.mkdir()
        (snap / filename).write_text("{}")
        assert R.snapshot_has_a_format(snap, "a-diffusers-pipeline") is True
    nemo = tmp_path / "nemo"
    nemo.mkdir()
    (nemo / "model.nemo").write_bytes(b"\x00")
    assert R.snapshot_has_a_format(nemo, None) is True
