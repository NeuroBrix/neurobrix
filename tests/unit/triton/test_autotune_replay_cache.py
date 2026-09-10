"""The machine's replay cache — the LOCAL persistence of a runtime sweep.

Since the certified directory (owner directive 2026-09-06) the engine keeps
what a runtime sweep found in `~/.neurobrix/replay_cache` (relocatable with
NEUROBRIX_REPLAY_CACHE), never in the engine's directory: seeded under the
membership gate, captured with each key's bench margin beside its config.
"""
from __future__ import annotations

import json
import os

import pytest
import triton

from neurobrix.triton import autotune_cache as atc


class _Tuner:
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
    monkeypatch.setattr(atc, "_DIR", str(tmp_path / "replay"))     # never the real machine cache
    atc._TIMINGS.clear()
    return tuner, tmp_path


class _T:
    def __init__(self, dtype):
        self.dtype = dtype


def test_key_of_mirrors_the_autotuner(rig):
    tuner, _ = rig
    key = atc.key_of(tuner, [_T("fp16"), _T("fp16"), _T("fp16"), 8, 16, 32], {})
    assert key == (8, 16, 32, "fp16", "fp16", "fp16")


def test_a_swept_key_is_captured_with_its_margin_and_seeded_back(rig):
    tuner, tmp = rig
    key = (8, 16, 32, "fp16", "fp16", "fp16")
    tuner.cache[key] = tuner.configs[0]
    atc.note_timings(tuner, key, {tuner.configs[0]: 1.0, tuner.configs[1]: 1.3})
    assert atc.capture() == 1
    doc = json.load(open(atc._artifact_path()))
    rec = doc["neurobrix.kernels.ops.fake.fake_kernel::" + repr(key)]
    assert rec["kwargs"] == {"BLOCK_M": 64, "BLOCK_N": 64}
    assert rec["timing"]["best_ms"] == 1.0 and abs(rec["timing"]["margin"] - 0.3) < 1e-9
    tuner.cache.clear()
    assert atc.seed() == 1 and tuner.cache[key].kwargs == {"BLOCK_M": 64, "BLOCK_N": 64}


def test_the_membership_gate_refuses_a_config_outside_the_kernels_space(rig):
    tuner, tmp = rig
    os.makedirs(atc._DIR, exist_ok=True)
    json.dump({"neurobrix.kernels.ops.fake.fake_kernel::(1, 2, 3, 'fp16', 'fp16', 'fp16')":
               {"kwargs": {"BLOCK_M": 999, "BLOCK_N": 64}, "num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None}},
              open(atc._artifact_path(), "w"))
    assert atc.seed() == 0 and not tuner.cache


def test_the_replay_cache_relocates_with_the_environment(monkeypatch, tmp_path):
    import importlib
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "elsewhere"))
    mod = importlib.reload(atc)
    try:
        assert mod._DIR == str(tmp_path / "elsewhere")
    finally:
        monkeypatch.delenv("NEUROBRIX_REPLAY_CACHE")
        importlib.reload(atc)
