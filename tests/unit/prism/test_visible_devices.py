"""Prism must plan on the GPUs the process can actually address.

`nvidia-smi` reports every card in the machine and ignores
`CUDA_VISIBLE_DEVICES` entirely. Until 2026-09-03 Prism planned on that raw
list, so a process pinned to one card was handed a four-GPU profile and
placed components on `cuda:2` — an ordinal that does not exist inside that
process. The CUDA runtime then refused with "invalid device ordinal", after
the container had already been read from disk.

Worse than the error: it silently contradicts an explicit instruction. A
user who pins a card on a shared machine has said which GPU they may use.

These pins cover the filter itself, CUDA's renumbering semantics, and the
cache invalidation — the profile is written once but the visible set is a
property of the environment and can change between two runs on one host.
"""

from __future__ import annotations

import pytest

from neurobrix.core.prism import autodetect


@pytest.fixture
def four_gpus(monkeypatch):
    """A fixed four-card machine, so the pins do not depend on the host."""
    devices = [
        {"index": 0, "brand": "nvidia", "model": "Tesla V100-SXM2-16GB",
         "memory_mb": 16384, "compute_capability": "7.0",
         "supports_dtypes": ["float32", "float16"], "architecture": "volta",
         "pcie_version": "3.0"},
        {"index": 1, "brand": "nvidia", "model": "Tesla V100-SXM2-16GB",
         "memory_mb": 16384, "compute_capability": "7.0",
         "supports_dtypes": ["float32", "float16"], "architecture": "volta",
         "pcie_version": "3.0"},
        {"index": 2, "brand": "nvidia", "model": "Tesla V100-SXM2-32GB",
         "memory_mb": 32768, "compute_capability": "7.0",
         "supports_dtypes": ["float32", "float16"], "architecture": "volta",
         "pcie_version": "3.0"},
        {"index": 3, "brand": "nvidia", "model": "Tesla V100-SXM2-32GB",
         "memory_mb": 32768, "compute_capability": "7.0",
         "supports_dtypes": ["float32", "float16"], "architecture": "volta",
         "pcie_version": "3.0"},
    ]

    def fake_smi():
        return [dict(d) for d in devices]

    # The filter is applied by the DETECTION CASCADE, not inside one detector:
    # filtering only nvidia-smi made it look like that detector had failed, so
    # detection fell through to the lspci fallback — which reads the PCI bus
    # and cannot see the variable — and rebuilt a full-rig profile on a
    # process that could address none of it (caught by the CPU-only cell,
    # 2026-09-03). So the pins go through the cascade.
    monkeypatch.setattr(autodetect, "_parse_nvidia_smi", fake_smi)
    monkeypatch.setattr(autodetect, "_parse_lspci", lambda: ([], "nvidia"))
    return fake_smi


# --- the filter itself ------------------------------------------------------

def test_unset_means_no_filtering(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert autodetect._visible_device_filter() is None


def test_empty_string_means_no_gpu_visible(monkeypatch):
    """CUDA treats an empty value as 'hide everything' — not as 'unset'."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert autodetect._visible_device_filter() == []


def test_indices_are_parsed_in_order(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert autodetect._visible_device_filter() == [3, 1]


def test_whitespace_is_tolerated(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 2 , 0 ")
    assert autodetect._visible_device_filter() == [2, 0]


def test_uuid_form_is_not_guessed_at(monkeypatch):
    """GPU-UUID selection is legal. We cannot map it from nvidia-smi indices
    here, and guessing would place work on the wrong card — so the filter
    declines rather than inventing an answer."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fe8b3a1c-0000-0000-0000-000000000000")
    assert autodetect._visible_device_filter() is None


# --- renumbering, which is the part that actually bites ---------------------

def test_pinning_one_card_yields_one_device_at_ordinal_zero(four_gpus, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    devices, _ = autodetect._detect_gpus_linux()
    assert len(devices) == 1
    assert devices[0]["index"] == 0, "CUDA renumbers the visible set from 0"
    assert devices[0]["physical_index"] == 2
    assert devices[0]["memory_mb"] == 32768, "and it must be the card asked for"


def test_order_of_the_variable_defines_the_ordinals(four_gpus, monkeypatch):
    """`CUDA_VISIBLE_DEVICES=3,1` makes physical 3 become cuda:0."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    devices, _ = autodetect._detect_gpus_linux()
    assert [(d["index"], d["physical_index"]) for d in devices] == [(0, 3), (1, 1)]
    assert devices[0]["memory_mb"] == 32768
    assert devices[1]["memory_mb"] == 16384


def test_unset_keeps_every_card(four_gpus, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    devices, _ = autodetect._detect_gpus_linux()
    assert [d["index"] for d in devices] == [0, 1, 2, 3]


def test_invalid_entry_truncates_like_cuda(four_gpus, monkeypatch):
    """CUDA stops at the first entry it cannot resolve; everything after it is
    invisible, even if individually valid."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,9,0")
    devices, _ = autodetect._detect_gpus_linux()
    assert [d["physical_index"] for d in devices] == [1]


# --- the cached profile must not outlive the environment it describes -------

def _two_environments(monkeypatch, tmp_path):
    """Two processes on one host, pinned to different cards."""
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    seen = {"models": ["Tesla V100-SXM2-32GB"]}
    monkeypatch.setattr(autodetect, "_detect_gpus",
                        lambda _sys: ([{"model": m} for m in seen["models"]], "nvidia"))
    monkeypatch.setattr(autodetect, "detect_hardware",
                        lambda: {"devices": [{"model": m, "memory_mb": 32768 if "32GB" in m else 16384}
                                             for m in seen["models"]], "notes": "test"})
    return seen


def test_each_visible_set_reads_its_own_profile_file(monkeypatch, tmp_path):
    seen = _two_environments(monkeypatch, tmp_path)
    id_32 = autodetect.get_or_create_default_profile()
    seen["models"] = ["Tesla V100-SXM2-16GB"]
    id_16 = autodetect.get_or_create_default_profile()
    assert id_32 != id_16 and id_32.startswith("default-") and id_16.startswith("default-")
    assert (tmp_path / f"{id_32}.yml").exists() and (tmp_path / f"{id_16}.yml").exists()
    import yaml
    assert yaml.safe_load((tmp_path / f"{id_32}.yml").read_text())["devices"][0]["memory_mb"] == 32768
    assert yaml.safe_load((tmp_path / f"{id_16}.yml").read_text())["devices"][0]["memory_mb"] == 16384
    # the second detection did not touch the first environment's file
    seen["models"] = ["Tesla V100-SXM2-32GB"]
    assert autodetect.get_or_create_default_profile() == id_32
    # the human-facing default.yml mirrors the latest detection, whole file, no leftovers
    assert (tmp_path / "default.yml").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_detection_unavailable_falls_back_to_the_shared_default(monkeypatch, tmp_path):
    monkeypatch.setattr(autodetect, "HARDWARE_DIR", tmp_path)
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", tmp_path / "default.yml")
    monkeypatch.setattr(autodetect, "_detect_gpus", lambda _sys: (_ for _ in ()).throw(RuntimeError("no smi")))
    monkeypatch.setattr(autodetect, "detect_hardware", lambda: {"devices": [], "notes": "cpu"})
    assert autodetect.get_or_create_default_profile() == "default"
    assert (tmp_path / "default.yml").exists()
