"""A weight load hands the process's weight reader the file it actually reads on disk (the extracted cache's file, or
the archive itself), never the container-relative name of a shard: the reader keys its kept width on that file's
storage (`kept_launch_facts.read_key` stats it), and a name like `components/model/weights/shard_000.safetensors`
resolves against no directory.

2026-11-22: every first launch on a fresh home ended rc 1 in its reload probe on
`os.stat('components/model/weights/shard_000.safetensors')` (FileNotFoundError), and the lead's smoke home failed the
same way. What the test would do if the code were wrong: the source handed to the reader is a relative name that does
not exist from another working directory (red), and the kept-width read on a generated profile raises (red)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from neurobrix.core.io import weight_loader as WL


class _Seen(Exception):
    pass


def _container(root: Path) -> Path:
    weights = root / "box" / "components" / "model" / "weights"
    weights.mkdir(parents=True)
    save_file({"w": torch.ones(2)}, str(weights / "shard_000.safetensors"))
    (root / "box" / "components" / "model" / "weights_index.json").write_text(
        json.dumps({"tensors": {"w": {"shard": "shard_000.safetensors", "dtype": "F32", "shape": [2]}}}))
    (root / "box" / "manifest.json").write_text(json.dumps({"model_name": "box", "components": {}}))
    return root / "box"


@pytest.mark.parametrize("entry", ["load_component", "load_component_with_shard_map"])
def test_the_reader_is_handed_a_file_that_exists_from_any_directory(tmp_path, monkeypatch, entry):
    box = _container(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                        # the launch's working directory is the home, not the box
    seen = []

    def reader(source=None):
        seen.append(source)
        raise _Seen

    monkeypatch.setattr(WL, "_weight_reader", reader)
    loader = WL.WeightLoader(str(box))
    with pytest.raises(_Seen):
        if entry == "load_component":
            loader.load_component("model", device="cpu")
        else:
            loader.load_component_with_shard_map("model", {"shard_000.safetensors": "cpu"})
    assert seen and seen[0] is not None and Path(seen[0]).is_file(), seen
    assert Path(seen[0]).resolve() == (box / "components" / "model" / "weights" / "shard_000.safetensors").resolve()


def test_the_kept_width_is_read_on_a_generated_profile_without_raising(tmp_path, monkeypatch):
    from neurobrix.pytorch.prism import autodetect as A, kept_launch_facts as K
    box = _container(tmp_path)
    (tmp_path / "hw").mkdir()
    (tmp_path / "hw" / "default-one.yml").write_text("devices: []\n")
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path / "hw")
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-one")
    monkeypatch.setattr(K, "library_key", lambda: "lib")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    asked = []

    def reader(source=None):
        asked.append(K.kept_read_width("default-one", source))
        raise _Seen

    monkeypatch.setattr(WL, "_weight_reader", reader)
    with pytest.raises(_Seen):
        WL.WeightLoader(str(box)).load_component("model", device="cpu")
    assert asked == [None]                              # read under the file's own key: nothing kept yet, no stat error
