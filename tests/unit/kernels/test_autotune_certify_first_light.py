"""`neurobrix autotune certify`, first light on the card this process sees.

One small matmul shape: the wrapper's own key is recorded, every candidate
config is run against the fp64 oracle, the survivors are timed, the entry
is written with its proof, the directory gate re-reads it, and the loader
then serves it without a sweep. An injected wrong config is excluded and
recorded with its deviation.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from neurobrix.kernels import autotune_certified as C
from neurobrix.kernels import autotune_certify as Z

cuda = pytest.importorskip("triton")


def _cuda_available():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        return DeviceAllocator.device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA card")


def _recorded_key(tuner, call):
    """The key the wrapper computes for this call (the autotuner's own key)."""
    from neurobrix.triton import autotune_cache as atc
    seen = {}
    saved = tuner.run

    def spy(*args, **kwargs):
        seen["key"] = atc.key_of(tuner, args, kwargs)
        return saved(*args, **kwargs)
    tuner.run = spy
    try:
        call()
    finally:
        tuner.run = saved
    return seen["key"]


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR", str(tmp_path))
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay"))
    C.reset()
    yield tmp_path
    C.reset()


def test_one_matmul_shape_is_certified_gated_and_served(root, monkeypatch):
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops.matmul import matmul_kernel
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    prof = C.active_profile()
    assert prof is not None, "the launcher resolved no vendor profile on this card"
    vendor, profile = prof
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((96, 64)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((64, 80)) * 0.1).astype(np.float16)
    key = _recorded_key(matmul_kernel, lambda: W.mm(NBXTensor.from_numpy(a), NBXTensor.from_numpy(b)))
    matmul_kernel.cache.pop(key, None)
    dtype = C.output_dtype(matmul_kernel, key)
    tol = Z._tolerance(vendor, profile, dtype)

    entry = Z.certify_key(qual, matmul_kernel, key, tol, np.random.default_rng(7),
                          bench=lambda fn: (fn(), 0.5)[1])
    assert entry["proof"]["deviation"] <= tol and entry["proof"]["accepted"] >= 1
    assert entry["excluded"] == [], "a correct kernel's configs all agree with the fp64 oracle"
    assert entry["proof"]["shape"] == list(key) and entry["proof"]["oracle"].startswith("fp64")

    path = C.file_for(vendor, profile, qual, dtype, root=root)
    Z._write_file(path, vendor, profile, qual, dtype, {C.key_repr(key): {k: entry[k] for k in ("config", "proof", "excluded")}})
    doc = json.loads(path.read_text())
    assert C.validate(doc, path, matmul_kernel) == []

    # served at load, without a sweep
    matmul_kernel.cache.pop(key, None)
    C.reset()
    assert C.apply(qual, matmul_kernel, key) is True
    assert matmul_kernel.cache[key].kwargs == entry["config"]["kwargs"]


def test_a_config_that_diverges_from_the_oracle_is_excluded_and_recorded(root, monkeypatch):
    """The certifier's judgement is the oracle's: a config whose result is
    made wrong is excluded with its deviation, and the file's gate still
    re-reads (its deviation is above the tolerance)."""
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops.matmul import matmul_kernel
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    vendor, profile = C.active_profile()
    rng = np.random.default_rng(3)
    a = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
    key = _recorded_key(matmul_kernel, lambda: W.mm(NBXTensor.from_numpy(a), NBXTensor.from_numpy(b)))
    matmul_kernel.cache.pop(key, None)
    tol = Z._tolerance(vendor, profile, C.output_dtype(matmul_kernel, key))
    real = Z.oracle_deviation
    calls = {"n": 0}

    def sabotage(out, oracle):          # the first candidate's result reads as garbage
        calls["n"] += 1
        return 0.75 if calls["n"] == 1 else real(out, oracle)
    monkeypatch.setattr(Z, "oracle_deviation", sabotage)
    entry = Z.certify_key(qual, matmul_kernel, key, tol, np.random.default_rng(3), bench=lambda fn: (fn(), 0.5)[1])
    assert len(entry["excluded"]) == 1 and entry["excluded"][0]["deviation"] == 0.75
    assert entry["proof"]["accepted"] == entry["proof"]["candidates"] - 1
    path = C.file_for(vendor, profile, qual, C.output_dtype(matmul_kernel, key), root=root)
    dtype = C.output_dtype(matmul_kernel, key)
    Z._write_file(path, vendor, profile, qual, dtype, {C.key_repr(key): {k: entry[k] for k in ("config", "proof", "excluded")}})
    assert C.validate(json.loads(path.read_text()), path, matmul_kernel) == []
