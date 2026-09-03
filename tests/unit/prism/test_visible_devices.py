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

def test_cached_profile_is_rejected_when_the_visible_set_changed(monkeypatch, tmp_path):
    """A four-GPU profile reused inside a one-GPU process is exactly how
    'invalid device ordinal' reached users."""
    cache = tmp_path / "default.yml"
    cache.write_text(
        "id: auto-4xv100\n"
        "devices:\n"
        "  - {model: Tesla V100-SXM2-16GB}\n"
        "  - {model: Tesla V100-SXM2-16GB}\n"
        "  - {model: Tesla V100-SXM2-32GB}\n"
        "  - {model: Tesla V100-SXM2-32GB}\n"
    )
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", cache)
    monkeypatch.setattr(autodetect, "_detect_gpus",
                        lambda _os: ([{"model": "Tesla V100-SXM2-16GB"}], "nvidia"))
    assert autodetect._cached_profile_matches_visible_gpus() is False


def test_cached_profile_is_kept_when_the_machine_is_unchanged(monkeypatch, tmp_path):
    cache = tmp_path / "default.yml"
    cache.write_text(
        "id: auto-2xv100\n"
        "devices:\n"
        "  - {model: Tesla V100-SXM2-16GB}\n"
        "  - {model: Tesla V100-SXM2-32GB}\n"
    )
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", cache)
    monkeypatch.setattr(autodetect, "_detect_gpus", lambda _os: (
        [{"model": "Tesla V100-SXM2-16GB"}, {"model": "Tesla V100-SXM2-32GB"}], "nvidia"))
    assert autodetect._cached_profile_matches_visible_gpus() is True


def test_unreadable_cache_redetects_rather_than_trusting_it(monkeypatch, tmp_path):
    cache = tmp_path / "default.yml"
    cache.write_text("id: [unclosed\n")
    monkeypatch.setattr(autodetect, "DEFAULT_PROFILE_PATH", cache)
    assert autodetect._cached_profile_matches_visible_gpus() is False


def test_hiding_every_gpu_is_conclusive(four_gpus, monkeypatch):
    """`CUDA_VISIBLE_DEVICES=""` means no GPU, and must not send detection
    looking for another source. It used to fall through to the lspci
    fallback, which reads the PCI bus and rebuilt a four-GPU profile — the
    process then planned onto cards it could not address and died at the
    first `.to(device)`."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    devices, brand = autodetect._detect_gpus_linux()
    assert devices == []
    assert brand == "none"


def test_the_filter_is_reusable_across_detectors():
    """It is a cascade-level step, so it must work on any detector's output,
    not only nvidia-smi's."""
    rocm_like = [{"index": 0, "model": "MI250X"}, {"index": 1, "model": "MI250X"}]
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    try:
        out = autodetect._apply_visible_filter(rocm_like)
        assert [(d["index"], d["physical_index"]) for d in out] == [(0, 1)]
    finally:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
