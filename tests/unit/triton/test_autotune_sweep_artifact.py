"""The kernel sweep is never run inside a request (owner directive,
2026-09-06). A model's sweep artifact — per kernel, per shape, per hardware
profile — is the only source of autotune configs in a request; a shape it
never measured is served by the nearest measured shape of the same kernel; a
kernel it never measured, or a model without an artifact, is an explicit
refusal naming the command that measures it. `--sweep` (NBX_AUTOTUNE=sweep)
is the producer.

Pins (CPU, no GPU: the Autotuner is a stand-in with the attributes
`Autotuner.run` reads):
  key_of mirrors Autotuner.run's key; activate refuses without an artifact
  and seeds with one (membership-gated); resolve_missing serves the nearest
  shape, refuses an unmeasured kernel, measures under sweep; the embedded
  artifact wins over the store; capture_model writes the store artifact.
"""
from __future__ import annotations

import json
import os

import pytest
import triton

from neurobrix.triton import autotune_cache as atc


class _Tuner:
    """What `Autotuner.run` and the artifact module read from an Autotuner."""

    def __init__(self):
        self.keys = ["M", "N", "K"]
        self.arg_names = ["a_ptr", "b_ptr", "c_ptr", "M", "N", "K"]
        self.configs = [triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
                        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=2, num_stages=2)]
        self.cache = {}

        def base_fn():
            pass
        base_fn.__name__ = "fake_kernel"
        self.base_fn = base_fn


@pytest.fixture
def rig(monkeypatch, tmp_path):
    tuner = _Tuner()
    monkeypatch.setattr(atc, "_autotuners", lambda: iter([("neurobrix.kernels.ops.fake.fake_kernel", tuner)]))
    monkeypatch.setattr(atc, "_arch_fingerprint", lambda: "cuda-70")
    monkeypatch.setenv("NEUROBRIX_AUTOTUNE_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(atc, "_DIR", str(tmp_path / "machine_cache"))     # never the real machine cache
    monkeypatch.delenv("NBX_AUTOTUNE", raising=False)
    monkeypatch.setattr(atc, "_ACTIVE", None)
    container = tmp_path / "container"
    (container / "runtime").mkdir(parents=True)
    return tuner, container, tmp_path


def _artifact(entries):
    return {"format": atc.FORMAT, "model_name": "m", "arch": "cuda-70", "entries": entries}


def _cfg(bm, bn, warps=4, stages=2):
    return {"kwargs": {"BLOCK_M": bm, "BLOCK_N": bn}, "num_warps": warps, "num_stages": stages, "num_ctas": 1, "maxnreg": None}


def test_key_of_mirrors_the_autotuner(rig):
    tuner, _, _ = rig

    class _T:
        dtype = "fp16"
    key = atc.key_of(tuner, (_T(), _T(), _T(), 128, 256, 512), {})
    assert key == (128, 256, 512, "fp16", "fp16", "fp16")


def test_no_artifact_is_a_refusal_naming_the_command(rig):
    tuner, container, _ = rig
    with pytest.raises(RuntimeError, match="no kernel sweep artifact.*--sweep"):
        atc.activate("m", str(container))


def test_artifact_seeds_and_membership_gates(rig):
    tuner, container, tmp = rig
    store = tmp / "store" / "m"
    store.mkdir(parents=True)
    (store / "cuda-70.json").write_text(json.dumps(_artifact({
        "neurobrix.kernels.ops.fake.fake_kernel::(128, 256, 512)": _cfg(64, 64),
        "neurobrix.kernels.ops.fake.fake_kernel::(64, 256, 512)": _cfg(999, 999),   # not in the config space
    })))
    n = atc.activate("m", str(container))
    assert n == 1 and (128, 256, 512) in tuner.cache and (64, 256, 512) not in tuner.cache


def test_nearest_measured_shape_serves_an_unseen_one(rig):
    tuner, container, tmp = rig
    store = tmp / "store" / "m"
    store.mkdir(parents=True)
    (store / "cuda-70.json").write_text(json.dumps(_artifact({
        "neurobrix.kernels.ops.fake.fake_kernel::(128, 256, 512)": _cfg(64, 64),
        "neurobrix.kernels.ops.fake.fake_kernel::(1024, 256, 512)": _cfg(32, 64, warps=2),
    })))
    atc.activate("m", str(container))
    atc.resolve_missing(tuner, (900, 256, 512))
    assert tuner.cache[(900, 256, 512)].kwargs["BLOCK_M"] == 32      # nearest in M is 1024
    atc.resolve_missing(tuner, (150, 300, 512))                         # nearest over every extent: (128, 256, 512)
    assert tuner.cache[(150, 300, 512)].kwargs["BLOCK_M"] == 64
    with pytest.raises(RuntimeError, match="no measured configuration"):
        atc.resolve_missing(tuner, (900, 256, 512, "bf16"))           # another kernel variant (dtype tail): never measured


def test_sweep_mode_measures_and_captures(rig, monkeypatch):
    tuner, container, tmp = rig
    monkeypatch.setenv("NBX_AUTOTUNE", "sweep")
    atc.activate("m", str(container))                       # no artifact, allowed under sweep
    atc.resolve_missing(tuner, (7, 8, 9))                   # measure: nothing inserted, nothing raised
    assert (7, 8, 9) not in tuner.cache
    tuner.cache[(7, 8, 9)] = tuner.configs[0]              # what the bench would have selected
    tuner.cache[(1, 1, 1)] = tuner.configs[1]              # seeded from the machine cache, never used here
    atc.note_use(tuner, (7, 8, 9))
    path = atc.capture_model()
    doc = json.loads(open(path).read())
    assert doc["format"] == atc.FORMAT and "neurobrix.kernels.ops.fake.fake_kernel::(7, 8, 9)" in doc["entries"]
    assert "neurobrix.kernels.ops.fake.fake_kernel::(1, 1, 1)" not in doc["entries"]   # only the shapes the model used
    assert path == atc.store_path("m")


def test_embedded_artifact_wins_over_the_store(rig):
    tuner, container, tmp = rig
    store = tmp / "store" / "m"
    store.mkdir(parents=True)
    (store / "cuda-70.json").write_text(json.dumps(_artifact({
        "neurobrix.kernels.ops.fake.fake_kernel::(1, 2, 3)": _cfg(64, 64)})))
    (container / "runtime" / "autotune").mkdir()
    (container / "runtime" / "autotune" / "cuda-70.json").write_text(json.dumps(_artifact({
        "neurobrix.kernels.ops.fake.fake_kernel::(4, 5, 6)": _cfg(32, 64, warps=2)})))
    atc.activate("m", str(container))
    assert (4, 5, 6) in tuner.cache and (1, 2, 3) not in tuner.cache
    assert atc.active()["source"].endswith(os.path.join("runtime", "autotune", "cuda-70.json"))
