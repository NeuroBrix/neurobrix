"""qa on 1215 (MAJOR): the weight reader's kept width had no key but the GPU set's profile and nothing invalidated it:
a width settled under one library version, one storage or one kernel was reused under another as if measured today.
It is kept under a key naming what it measures (`kept_launch_facts.read_key`: the library with its commit for a source
tree, the storage the weights are read from, the OS kernel); a width kept under another key is not reused, the reader
starts at one and measures, and the new key is kept beside the old.

What each test would do if the code were wrong: the width of another library or another disk reused (red), a reader
starting at the old width (red), the new key written over the old one (red), a width kept with no key reused (red)."""
from __future__ import annotations

import yaml

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import kept_launch_facts as K
from tests.unit.prism.test_class_facts_are_measured_once_per_card_model import (  # noqa: F401 — the fixture
    C16, _read, _write, hw)

SETTLED = {"workers": 8, "settled": True, "start": None, "rates": {4: 3e9, 8: 3.5e9, 16: 3.4e9},
           "draws": {8: [3.5e9, 3.3e9]}, "spread": (3.3e9, 3.5e9)}


def _reader_from_one(hw, monkeypatch):
    from neurobrix.core import workspace as W
    monkeypatch.setattr(W, "_READER", None)
    monkeypatch.setattr(W, "_LOAD_CEILING", 32)
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-one")
    return W.weight_reader(hw)


def test_a_width_kept_under_another_library_is_not_reused(hw, monkeypatch):
    _write(hw, "default-one", [dict(C16, index=0)])
    monkeypatch.setattr(K, "library_key", lambda: "0.1.11+aaaa")
    assert K.note_read_width("default-one", SETTLED, hw)
    assert K.kept_read_width("default-one", hw) == 8
    monkeypatch.setattr(K, "library_key", lambda: "0.1.11+bbbb")            # another commit of the same version
    assert K.kept_read_width("default-one", hw) is None
    r = _reader_from_one(hw, monkeypatch)
    assert r.width == 1 and r.measured()["start"] is None                  # measured again, from one
    assert K.note_read_width("default-one", dict(SETTLED, workers=4, draws={4: [2.9e9]}), hw)
    widths = _read(hw, "default-one")["host_measured"]["weight_read"]
    assert sorted(w["workers"] for w in widths.values()) == [4, 8]          # the new key beside the old one


def test_a_width_kept_for_another_storage_is_not_reused(hw, monkeypatch):
    _write(hw, "default-one", [dict(C16, index=0)])
    monkeypatch.setattr(K, "storage_key", lambda source: "device 0:51, nfs 192.168.100.1:/nvme/cache")
    assert K.note_read_width("default-one", SETTLED, hw)
    assert K.kept_read_width("default-one", hw) == 8
    monkeypatch.setattr(K, "storage_key", lambda source: "device 259:2, ext4 /dev/nvme0n1p2")   # the cache moved
    assert K.kept_read_width("default-one", hw) is None
    assert _reader_from_one(hw, monkeypatch).width == 1


def test_the_storage_key_is_the_same_for_two_files_of_one_disk_and_names_its_device(tmp_path):
    (tmp_path / "a").write_bytes(b"0"); (tmp_path / "b").write_bytes(b"1")
    assert K.storage_key(tmp_path / "a") == K.storage_key(tmp_path / "b")
    assert K.storage_key(tmp_path / "a").startswith("device ")
    assert K.read_key(tmp_path / "a").startswith(f"library {K.library_key()}; storage ")


def test_a_width_kept_with_no_key_is_not_reused_and_a_source_is_needed(hw, monkeypatch):
    _write(hw, "default-one", [dict(C16, index=0)])
    path = A.HARDWARE_DIR / "default-one.yml"
    data = yaml.safe_load(path.read_text())
    data.setdefault("host_measured", {})["weight_read"] = {"workers": 8, "settled": True, "low_gbps": 3.3}  # before the key
    path.write_text(yaml.safe_dump(data))
    assert K.kept_read_width("default-one", hw) is None
    assert K.kept_read_width("default-one") is None                         # no source: nothing to key on
    assert not K.note_read_width("default-one", SETTLED)                    # nothing kept without its source
    assert K.note_read_width("default-one", SETTLED, hw)                    # the keyed width replaces the unkeyed one
    assert list(_read(hw, "default-one")["host_measured"]["weight_read"]) == [K.read_key(hw)]
