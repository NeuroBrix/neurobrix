"""A machine whose cards are not all one model (16 GB and 32 GB V100 in one box): its generated profile gets, on EVERY
card, the facts a run needs, each model's measured on its own first card. Before, the first launch measured card 0
and copied its facts only onto cards of card 0's model; the library then refused every allocation on the others
("no measured `vmm_chunk_bytes` for device 2"), on every run, since the profile never repaired itself (2026-11-17).

What each test would do if the code were wrong: a card of the other model left without its chunk (red); the other
model given card 0's chunk instead of its own measure (red); an already written profile missing the other model's
chunk left as it is (red)."""
from __future__ import annotations

import pytest
import yaml

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import facts_probe as P
from nbxtensor.backend import host_measure as H
from neurobrix.pytorch.prism.kept_launch_facts import card_key

SMALL = {"brand": "nvidia", "model": "Tesla V100-SXM2-16GB", "architecture": "volta", "compute_capability": "7.0",
         "memory_mb": 16384, "supports_dtypes": ["float16", "float32"], "runtime_context_mb": 300}
LARGE = dict(SMALL, model="Tesla V100-SXM2-32GB", memory_mb=32768)
CHUNK = {0: 32 << 20, 1: 32 << 20, 2: 64 << 20}          # what each card's own probe answers, by process index


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    probed = []
    monkeypatch.setattr(P, "vmm_first_replay", lambda dev: None)
    monkeypatch.setattr(P, "sweep_chunk", lambda dev, *a, **k: probed.append(dev) or (CHUNK[dev], None))
    monkeypatch.setattr(P, "ruled_chunk", lambda *a, **k: None)   # a class the library's rule does not answer: swept
    monkeypatch.setattr(H, "host_matmul_gflops", lambda *a, **k: 300.0)
    monkeypatch.setattr(H, "host_memory_gbps", lambda *a, **k: 20.0)
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda index: {})
    monkeypatch.setattr("nbxtensor.device.count", lambda: 2)   # several cards seen: the child reads them
    return tmp_path, probed


def _write(path, devices):
    path.write_text(yaml.safe_dump({"id": path.stem, "devices": devices}))


def test_every_card_of_a_mixed_machine_gets_its_own_model_s_chunk(hw):
    root, probed = hw
    _write(root / "default-mix.yml", [dict(SMALL, index=0), dict(SMALL, index=1), dict(LARGE, index=2)])
    A.ensure_measured_facts("default-mix")
    devices = yaml.safe_load((root / "default-mix.yml").read_text())["devices"]
    assert [d["measured"].get("vmm_chunk_bytes") for d in devices] == [32 << 20, 32 << 20, 64 << 20]
    assert probed == [0, 2], "each model measured once, on its own first card"
    # each model keeps its own vendor facts beside the chunk (volta.yml states the device speeds)
    assert all("matmul_tflops" in d["measured"] for d in devices)


def test_a_profile_written_before_without_the_other_model_s_chunk_is_completed(hw):
    root, probed = hw
    measured = {"vmm_chunk_bytes": 32 << 20, "vmm_chunk_under": card_key(), "matmul_tflops": {"float16": 100.0}}
    _write(root / "default-mix.yml", [dict(SMALL, index=0, measured=measured), dict(LARGE, index=1)])
    A.ensure_measured_facts("default-mix")
    devices = yaml.safe_load((root / "default-mix.yml").read_text())["devices"]
    assert devices[0]["measured"] == measured and devices[1]["measured"]["vmm_chunk_bytes"] == CHUNK[1]
    assert probed == [1]
    probed.clear()
    A.ensure_measured_facts("default-mix")
    assert probed == [], "measured once: the next launch reads it from the profile"
