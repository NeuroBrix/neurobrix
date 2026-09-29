"""The owner's ruling of 2026-09-29 01:37 (release-decisions): a certificate made under a retired
code generator is not thrown away — `--reprove-generator` re-PROVES that stored configuration under
the served generator (the fp64 oracle and one timing under the witness: one configuration), serves it
recorded as re-proven, and sweeps only where the stored configuration fails the oracle or no entry
exists. Red under the old behaviour: the whole configuration space was benched again for such a key
(a re-ranking, one to twelve keys a minute on the Mac, one key in five refused for witness drift), and
no proof said it had been re-proven from an earlier generator."""
from __future__ import annotations

import json
import numpy as np
import pytest

from neurobrix.kernels import autotune_certify as Z
from neurobrix.kernels import autotune_certified as C
from neurobrix.triton import autotune_cache as atc
from neurobrix.kernels import census_table as T

from tests.unit.kernels.test_autotune_certify_first_light import _cuda_available, _recorded_key  # noqa: E402

pytestmark = pytest.mark.skipif(not _cuda_available(), reason="a card is needed to launch the kernel")


@pytest.fixture
def root(tmp_path, monkeypatch):
    Z._bind_hardware_profile()
    monkeypatch.setenv("NEUROBRIX_AUTOTUNE_CERTIFIED_DIR", str(tmp_path))
    monkeypatch.setenv("NEUROBRIX_REPLAY_CACHE", str(tmp_path / "replay"))
    C.reset()
    yield tmp_path
    C.reset()


def _matmul_key():
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as W
    from neurobrix.kernels.ops.matmul import matmul_kernel
    rng = np.random.default_rng(7)
    a = (rng.standard_normal((96, 64)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((64, 80)) * 0.1).astype(np.float16)
    key = _recorded_key(matmul_kernel, lambda: W.mm(NBXTensor.from_numpy(a), NBXTensor.from_numpy(b)))
    matmul_kernel.cache.pop(key, None)
    return matmul_kernel, key


def _certify_once(root, monkeypatch, stored_proof_backend):
    """One `certify(reprove_generator=True)` over a one-key census whose directory already holds the
    key under `stored_proof_backend`; returns (summary, entry after, benches counted)."""
    from neurobrix.kernels.ops.matmul import matmul_kernel
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    tuner, key = _matmul_key()
    vendor, profile = C.active_profile()
    dtype = C.output_dtype(tuner, key)
    tol = Z._tolerance(vendor, profile, dtype)
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: 3.0)        # the setup's own sweep under a still witness
    # the certificate as the retired generator left it: a full sweep's entry, its backend relabelled
    first = Z.certify_key(qual, tuner, key, tol, np.random.default_rng(7), bench=lambda fn: (fn(), 0.5)[1])
    old = {k: first[k] for k in ("config", "proof", "excluded")}
    old["proof"] = dict(old["proof"], backend=stored_proof_backend)
    path = C.file_for(vendor, profile, qual, dtype, root=root)
    Z._write_file(path, vendor, profile, qual, dtype, {C.key_repr(key): old})
    assert C.proof_backend(old["proof"]) != C.proof_backend({"backend": Z._backend()}), "the stored proof must name another generator"
    tuner.cache.pop(key, None); C.reset()
    # the census is THE TABLE of this card's class (P2): one row for this key, nothing else
    cls = C.memory_class_gb(Z._certifying_device()["memory_mb"]) if hasattr(Z, "_certifying_device") else C.memory_class_gb(C.executing_memory_class_mb()) if hasattr(C, "executing_memory_class_mb") else None
    if cls is None:
        cls = C.proof_memory_class(first["proof"])
    monkeypatch.setattr(T, "ROOT", root / "table")
    T.write(T.table_path(vendor, profile, cls), [{"model": "t", "container": "s", "mode": "triton", "rungs_mb": None, "ops": [None],
                                                  "kernel": qual, "key": C.key_repr(key), "dtype": T.dtypes_of(C.key_repr(key)), "tool": "t"}])
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: 3.0)        # the witness held still: this case is not the drift case
    benches = []
    summary = Z.certify(profile, vendor=vendor, reprove_generator=True, log=lambda *a, **k: None,
                        bench=lambda fn: (benches.append(1), fn(), 0.5)[2])
    return summary, json.loads(path.read_text())["entries"][C.key_repr(key)], benches, first


def test_a_stored_configuration_is_re_proven_under_the_served_generator_with_one_launch(root, monkeypatch):
    old_backend = dict(Z._backend()); old_backend["backend_hash"] = "msl-v0.1-retired000"; old_backend["triton"] = "3.7.9+retired"
    summary, entry, benches, first = _certify_once(root, monkeypatch, old_backend)
    assert summary.get("reproven", 0) == 1 and summary.get("swept", 0) == 0, summary
    assert len(benches) == 1, f"one configuration is timed once, not the whole space ({len(benches)} benches)"
    assert entry["config"] == first["config"], "the stored configuration is kept, not re-ranked"
    assert C.proof_backend(entry["proof"]) == C.proof_backend({"backend": Z._backend()}), "served under the running generator"
    assert entry["proof"].get("reproven_from") == C.proof_backend({"backend": old_backend}), "recorded as re-proven from the retired generator"
    assert entry["proof"]["candidates"] == 1 and entry["proof"]["accepted"] == 1


def test_a_stored_configuration_that_fails_the_oracle_is_swept_and_the_reason_counted(root, monkeypatch):
    """The other branch of the ruling: a stored configuration the fp64 oracle rejects under the served
    generator is not served — the key is swept, and the summary says why. Injection: the stored entry's
    configuration is made wrong for this kernel (a block size outside its config space is refused by the
    membership gate; a configuration inside the space that computes wrongly is what a compiler change
    would produce — here the stored kwargs are replaced by ones the kernel's own configs do not carry, so
    the single-candidate re-prove cannot pass)."""
    from neurobrix.kernels.ops.matmul import matmul_kernel
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    tuner, key = _matmul_key()
    vendor, profile = C.active_profile()
    dtype = C.output_dtype(tuner, key)
    tol = Z._tolerance(vendor, profile, dtype)
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: 3.0)        # the setup's own sweep under a still witness
    first = Z.certify_key(qual, tuner, key, tol, np.random.default_rng(7), bench=lambda fn: (fn(), 0.5)[1])
    old = {k: first[k] for k in ("config", "proof", "excluded")}
    old_backend = dict(Z._backend()); old_backend["backend_hash"] = "msl-v0.1-retired000"; old_backend["triton"] = "3.7.9+retired"
    old["proof"] = dict(old["proof"], backend=old_backend)
    old["config"] = dict(old["config"], kwargs=dict(old["config"]["kwargs"], BLOCK_K=1))   # a configuration this kernel cannot run correctly at
    path = C.file_for(vendor, profile, qual, dtype, root=root)
    Z._write_file(path, vendor, profile, qual, dtype, {C.key_repr(key): old})
    tuner.cache.pop(key, None); C.reset()
    cls = C.proof_memory_class(first["proof"])
    monkeypatch.setattr(T, "ROOT", root / "table")
    T.write(T.table_path(vendor, profile, cls), [{"model": "t", "container": "s", "mode": "triton", "rungs_mb": None, "ops": [None],
                                                  "kernel": qual, "key": C.key_repr(key), "dtype": T.dtypes_of(C.key_repr(key)), "tool": "t"}])
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: 3.0)        # the witness held still
    benches = []
    summary = Z.certify(profile, vendor=vendor, reprove_generator=True, log=lambda *a, **k: None,
                        bench=lambda fn: (benches.append(1), fn(), 0.5)[2])
    entry = json.loads(path.read_text())["entries"][C.key_repr(key)]
    assert summary.get("reproven", 0) == 0 and summary.get("swept", 0) == 1, summary
    assert summary.get("swept_why", {}).get("stored configuration failed the oracle") == 1, summary.get("swept_why")
    assert len(benches) > 1, "the sweep timed the configuration space, not one configuration"
    assert entry["config"]["kwargs"].get("BLOCK_K") != 1, "the failing configuration was not served"
    assert "reproven_from" not in entry["proof"], "a swept key is not recorded as re-proven"


def test_a_drift_on_the_single_re_prove_timing_is_a_refusal_for_the_retry_never_a_sweep(root, monkeypatch):
    """Inbox 60 and 65 together: a key is never certified on a drifting witness and never left without its
    retry — and a drift is not an oracle failure, so it must not open a sweep. Injection: the witness
    reads two values 30 % apart around the single timing."""
    from neurobrix.kernels.ops.matmul import matmul_kernel
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    tuner, key = _matmul_key()
    vendor, profile = C.active_profile()
    dtype = C.output_dtype(tuner, key)
    tol = Z._tolerance(vendor, profile, dtype)
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: 3.0)        # the setup's own sweep under a still witness
    first = Z.certify_key(qual, tuner, key, tol, np.random.default_rng(7), bench=lambda fn: (fn(), 0.5)[1])
    old = {k: first[k] for k in ("config", "proof", "excluded")}
    old_backend = dict(Z._backend()); old_backend["backend_hash"] = "msl-v0.1-retired000"; old_backend["triton"] = "3.7.9+retired"
    old["proof"] = dict(old["proof"], backend=old_backend)
    path = C.file_for(vendor, profile, qual, dtype, root=root)
    Z._write_file(path, vendor, profile, qual, dtype, {C.key_repr(key): old})
    before = path.read_text()
    tuner.cache.pop(key, None); C.reset()
    cls = C.proof_memory_class(first["proof"])
    monkeypatch.setattr(T, "ROOT", root / "table")
    T.write(T.table_path(vendor, profile, cls), [{"model": "t", "container": "s", "mode": "triton", "rungs_mb": None, "ops": [None],
                                                  "kernel": qual, "key": C.key_repr(key), "dtype": T.dtypes_of(C.key_repr(key)), "tool": "t"}])
    kind, _ = Z._regime()
    if kind != "witness":
        pytest.skip("this card's stability regime is a clock lock, not a witness")
    reads = iter([3.0, 3.9] * 8)                                      # open 3.0, close 3.9: a 30 % drift on every bracket
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: next(reads))   # the module-level reader the local witness calls
    benches = []
    summary = Z.certify(profile, vendor=vendor, reprove_generator=True, log=lambda *a, **k: None,
                        bench=lambda fn: (benches.append(1), fn(), 0.5)[2])
    assert summary.get("swept", 0) == 0, f"a drift opened a sweep: {summary}"
    assert summary.get("reproven", 0) == 0 and summary.get("failed", 0) == 1, summary
    assert path.read_text() == before, "a refused key leaves the stored entry as it was, for the retry"


def test_another_memory_class_s_configuration_is_re_proven_on_this_card(root, monkeypatch):
    """The same ruling across memory classes (the supervisor, 2026-09-29 11:33): a key this card's class
    has no certificate for, where the other class holds one, keeps that configuration — the oracle and
    one timing on THIS card, filed under this card's class beside the other's, recorded as re-proven
    from that class. Red before `--reprove-class`: the key was swept over the whole space."""
    qual = "neurobrix.kernels.ops.matmul.matmul_kernel"
    tuner, key = _matmul_key()
    vendor, profile = C.active_profile()
    dtype = C.output_dtype(tuner, key)
    tol = Z._tolerance(vendor, profile, dtype)
    monkeypatch.setattr(Z, "_witness_time_ms", lambda proto: 3.0)
    first = Z.certify_key(qual, tuner, key, tol, np.random.default_rng(7), bench=lambda fn: (fn(), 0.5)[1])
    here = C.proof_memory_class(first["proof"])
    there_mb = 16384 if here != 16 else 32768
    there = C.memory_class_gb(there_mb)
    old = {k: first[k] for k in ("config", "proof", "excluded")}
    machine = dict(old["proof"]["machine"], device=dict(old["proof"]["machine"]["device"], memory_mb=there_mb))
    old["proof"] = dict(old["proof"], machine=machine)
    assert C.proof_memory_class(old["proof"]) == there != here
    path = C.file_for(vendor, profile, qual, dtype, root=root)
    Z._write_file(path, vendor, profile, qual, dtype, {C.key_repr(key): old})
    tuner.cache.pop(key, None); C.reset()
    monkeypatch.setattr(T, "ROOT", root / "table")
    T.write(T.table_path(vendor, profile, here), [{"model": "t", "container": "s", "mode": "triton", "rungs_mb": None, "ops": [None],
                                                   "kernel": qual, "key": C.key_repr(key), "dtype": T.dtypes_of(C.key_repr(key)), "tool": "t"}])
    benches = []
    summary = Z.certify(profile, vendor=vendor, only_missing=True, reprove_class=there, log=lambda *a, **k: None,
                        bench=lambda fn: (benches.append(1), fn(), 0.5)[2])
    entries = json.loads(path.read_text())["entries"]
    mine = C.entry_for_memory_class(entries[C.key_repr(key)], here)
    assert summary.get("reproven", 0) == 1 and summary.get("swept", 0) == 0, summary
    assert len(benches) == 1, f"one configuration timed once ({len(benches)} benches)"
    assert mine is not None and mine["config"] == old["config"], "the other class's configuration, filed for this class"
    assert mine["proof"].get("reproven_from") == f"memory class {there} GB"
    assert C.entry_for_memory_class(entries[C.key_repr(key)], there)["proof"] == old["proof"], "the other class's proof kept"
