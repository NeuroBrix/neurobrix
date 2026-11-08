"""An `mm` / `addmm` certificate made before the matrix unit was a candidate (1.8) is owed the unit's measurement, and
a pass that fills what is missing measures it beside the stored configuration — never a re-sweep, never skipped.

Until 1.8 a GEMM the unit could take had no key and ran on the unit by threshold; 1.8 gave it back its FMA key, whose
certificate chooses the route. The store still held FMA certificates of those keys from before the unit route (never
discarded, as the owner orders), with no unit measurement: `--only-missing` counted them served, and the runtime would
have served the FMA kernel where the unit ran 2-3x faster (2026-11-04: ~880 keys a class, reached by 41 models).

What these tests would do if the code were wrong: the owed entry skipped as covered — `certify_key` never called (red
at the first loop case); re-swept instead of measured beside its configuration — called without `only_config` (red
at the same case); an entry that holds its unit measurement, or a key below the unit's smallest tile, drawn again
(red at the counts)."""
from __future__ import annotations

import types

import pytest

from neurobrix.kernels import autotune_certify as AC
from tests.unit.prism._pinned_machine import V100_16GB, profile

MM = "nbxtensor.ops.matmul.matmul_kernel"
ABOVE = (1024, 3072, 3072, False, False, "fp16", "fp16", "fp16")      # rows the unit's smallest tile fills
BELOW = (16, 3072, 3072, False, False, "fp16", "fp16", "fp16")        # rows below it: the FMA kernel's only
CONFIG = {"kwargs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, "num_warps": 4, "num_stages": 3}


@pytest.fixture
def volta(monkeypatch):
    from nbxtensor.ops import wrappers as W
    from nbxtensor.ops import _configs as CF
    monkeypatch.setattr(W, "_NBX_HW_PROFILE", W._NBX_HW_PROFILE)
    monkeypatch.setattr(W, "_NBX_HAS_NATIVE_BF16", W._NBX_HAS_NATIVE_BF16)
    W.set_hardware_profile(profile(V100_16GB))
    monkeypatch.setattr(CF, "_ACTIVE_PROFILE", CF.vendor_profile_for_arch("7.0"))


def _entry(measured: bool):
    proof = {"date": "2026-10-01T00:00:00+00:00", "best_ms": 1.0,
             "backend": {"triton": "3.8.0", "name": "cuda"},       # the generator `_loop` runs: a covered key names it
             "machine": {"device": {"memory_mb": 16384}}}
    if measured:
        proof["unit_measured"] = {"tiles": [], "fma_best_ms": 1.0, "chosen": "fma"}
    return {"config": CONFIG, "proof": proof, "excluded": []}


def test_only_an_unmeasured_gemm_the_unit_can_take_is_owed(volta):
    assert AC.unit_owed(MM, ABOVE, _entry(False))
    assert not AC.unit_owed(MM, ABOVE, _entry(True))
    assert not AC.unit_owed(MM, BELOW, _entry(False))
    assert not AC.unit_owed("nbxtensor.ops.baddbmm_op.baddbmm_kernel", ABOVE, _entry(False))
    assert not AC.unit_owed(MM, ABOVE, None)


def _loop(monkeypatch, tmp_path, entries, keys, calls):
    monkeypatch.setattr(AC.C, "output_dtype", lambda tuner, key: "fp16")
    monkeypatch.setattr(AC.C, "file_for", lambda v, p, q, d, root=None: tmp_path / f"{q.rsplit('.', 1)[-1]}.{d}.json")
    monkeypatch.setattr(AC.C, "entry_covers", lambda e, k, *a, **kw: k in e)
    monkeypatch.setattr(AC.C, "entry_for_memory_gb", lambda e, cls: e)
    monkeypatch.setattr(AC, "_file_entries", lambda per, d, path: entries)
    monkeypatch.setattr(AC, "_tolerance", lambda *a: 1e-4)
    monkeypatch.setattr(AC, "_num_stages_space", lambda *a: [3])
    monkeypatch.setattr(AC, "_release_between_keys", lambda: None)
    monkeypatch.setattr(AC, "_backend", lambda: {"triton": "3.8.0", "name": "cuda"})

    def certify_key(qual, tuner, key, tol, rng, only_config=None, **k):
        calls.append((key, only_config))
        raise RuntimeError("stop here")       # the loop's call is what is judged

    monkeypatch.setattr(AC, "certify_key", certify_key)
    summary = {"certified": 0, "skipped": 0, "failed": 0, "unreachable": 0, "excluded_configs": 0,
               "kernels": {}, "started": 0.0}
    dev = {"memory_mb": 16384, "name": "card", "ordinal": 0, "visible_devices": "0"}
    return AC._certify_loop({MM: keys}, {MM: object()}, "nvidia", "volta", tmp_path, None, True, False, False,
                            dev, 16, None, None, None, summary, lambda _m: None, types.SimpleNamespace(add=None), None)


def test_a_pass_that_fills_what_is_missing_measures_the_owed_unit_beside_the_stored_configuration(
        volta, monkeypatch, tmp_path):
    calls = []
    entries = {AC.C.key_repr(ABOVE): _entry(False), AC.C.key_repr(BELOW): _entry(False)}
    _loop(monkeypatch, tmp_path, entries, [ABOVE, BELOW], calls)
    assert calls == [(ABOVE, CONFIG)], calls


def test_a_measured_entry_is_not_drawn_again(volta, monkeypatch, tmp_path):
    calls = []
    _loop(monkeypatch, tmp_path, {AC.C.key_repr(ABOVE): _entry(True)}, [ABOVE], calls)
    assert calls == []
