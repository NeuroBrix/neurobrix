"""A shipped certified file is replaced by its epoch re-proof (`tools/certified_epoch_stamp.py`) only when every
certification it owes is re-proven at the library's bits epoch with the shipped configuration, or listed failed for
the full recertification; the replacement is the draft (never mixed); the failed ones leave the file and join the
recertification list; one table row per file.

What would this file do if the code were wrong? A file with a key not yet tried replaced -> the first test RED; a
draft configuration that differs counted re-proven -> the second RED; a failed key dropped silently, not listed, or a
replacement holding an epoch-less proof, or no table row -> the third RED; a re-chosen unit tile counted re-proven ->
the unit test RED; a stop during the replacement losing a failed key or truncating the file -> the stop test RED.
"""
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from nbxtensor.ops import autotune_certified as C

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import certified_epoch_stamp as S  # noqa: E402

QUAL = "pkg.mod.toy_kernel"
K1, K2 = C.key_repr((1, "fp32")), C.key_repr((2, "fp32"))


def _cfg(b):
    return {"kwargs": {"BLOCK": b}, "num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None}


def _proof(epoch=None, n=1):
    """A proof of the key (n, 'fp32'): the library's gate refuses a proof whose shape is not its entry's key, and a
    draft refused for that is never refused for the reason a test means (qa 1227-3: the configuration test was)."""
    p = {"date": "2026-11-01T00:00:00+00:00", "engine_version": "0", "backend": {"triton": "3.8.0", "name": "cuda"},
         "shape": [n, "fp32"], "deviation": 0.0, "tolerance": 1e-4, "oracle": "fp64",
         "machine": {"device": {"memory_mb": 16384}, "clocks_mhz": {"0": {"graphics": 1290}}},
         "built": {"gpu": True}}
    if epoch is not None:
        p["bits_epoch"] = epoch
    return p


def _write(root, entries):
    d = root / "nvidia" / "volta"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "toy_kernel.fp32.json"
    path.write_text(json.dumps({"format": "nbx-autotune-certified/2", "vendor": "nvidia", "profile": "volta",
                                "kernel": QUAL, "dtype": "fp32", "entries": entries}))
    return path


def _ledger(draft, rows):
    with open(draft / S.LEDGER, "w") as fh:
        fh.write("file\tkey\tmemory_class\toutcome\tseconds\treason\tdate\n")
        for k, outcome in rows:
            fh.write(f"nvidia/volta/toy_kernel.fp32.json\t{k}\t16g\t{outcome}\t2.5\t{'oracle: too far' if outcome == 'failed' else ''}"
                     f"\t2026-11-22T00:00:00+00:00\n")


def _setup(tmp_path, draft_entries, ledger_rows):
    shipped = _write(tmp_path / "shipped", {K1: {"config": _cfg(1), "proof": _proof()},
                                            K2: {"config": _cfg(2), "proof": _proof(n=2)}})
    _write(tmp_path / "draft", draft_entries)
    _ledger(tmp_path / "draft", ledger_rows)
    return shipped


def _run(tmp_path):
    return S.run(tmp_path / "shipped", tmp_path / "draft", tmp_path / "table.tsv", tmp_path / "recert.tsv",
                 apply=True, say=lambda *a: None)


def test_a_file_with_a_key_not_yet_tried_is_not_replaced(tmp_path):
    shipped = _setup(tmp_path, {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}}, [(K1, "reproven")])
    before = shipped.read_bytes()
    out = _run(tmp_path)
    assert out["replaced"] == [] and shipped.read_bytes() == before


def test_a_draft_configuration_that_differs_is_not_a_reproof(tmp_path):
    shipped = _setup(tmp_path, {K1: {"config": _cfg(9), "proof": _proof(C.BITS_EPOCH)},
                                K2: {"config": _cfg(2), "proof": _proof(C.BITS_EPOCH, n=2)}},
                     [(K1, "reproven"), (K2, "reproven")])
    before = shipped.read_bytes()
    assert _run(tmp_path)["replaced"] == [] and shipped.read_bytes() == before
    # the same draft with the shipped configuration IS a re-proof: the refusal above is the configuration's alone
    _write(tmp_path / "draft", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)},
                                K2: {"config": _cfg(2), "proof": _proof(C.BITS_EPOCH, n=2)}})
    assert _run(tmp_path)["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]


def test_a_draft_unit_tile_that_differs_is_not_a_reproof(tmp_path):
    """qa 1227-1: the library serves a unit-bearing launch with `unit.tile`; a re-chosen tile is a re-tuning."""
    unit = lambda m: {"unit": {"tile": {"block_m": m, "block_n": 64, "block_k": 32, "warps_m": 2, "warps_n": 2,
                                        "group_m": 8}, "deviation": 0.0, "best_ms": 1.0}}
    shipped = _write(tmp_path / "shipped", {K1: {"config": _cfg(1), "proof": _proof(), **unit(64)}})
    _write(tmp_path / "draft", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH), **unit(128)}})
    _ledger(tmp_path / "draft", [(K1, "reproven")])
    before = shipped.read_bytes()
    out = _run(tmp_path)
    assert out["replaced"] == [] and shipped.read_bytes() == before
    _write(tmp_path / "draft", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH), **unit(64)}})
    assert _run(tmp_path)["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]


def test_the_failed_keys_leave_the_file_and_join_the_recert_list(tmp_path):
    shipped = _setup(tmp_path, {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}},
                     [(K1, "reproven"), (K2, "failed")])
    out = _run(tmp_path)
    assert out["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]
    doc = json.loads(shipped.read_text())
    assert list(doc["entries"]) == [K1] and C.epoch_refusal(doc["entries"]) is None     # never mixed
    rec = list(csv.DictReader(open(tmp_path / "recert.tsv"), delimiter="\t"))
    assert [(r["file"], r["key"], r["memory_class"]) for r in rec] == [("nvidia/volta/toy_kernel.fp32.json", K2, "16g")]
    assert rec[0]["reason"] == "oracle: too far"
    table = list(csv.DictReader(open(tmp_path / "table.tsv"), delimiter="\t"))
    assert table == [{"file": "nvidia/volta/toy_kernel.fp32.json", "keys": "2", "re_proven": "1", "failed": "1",
                      "seconds": "5.0", "generators_dropped": "0"}]


def test_a_stop_during_the_replacement_loses_no_failed_key_and_truncates_nothing(tmp_path, monkeypatch):
    """qa 1227-2: the failed key leaves the shipped file with the replacement, so its recert row must be on disk
    first; and a stop mid-copy must leave the shipped file whole."""
    shipped = _setup(tmp_path, {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}},
                     [(K1, "reproven"), (K2, "failed")])
    before = shipped.read_bytes()
    copy, rename = shutil.copyfile, os.replace

    def stopped_copy(src, dst, *a, **kw):
        if Path(dst) == shipped:                       # killed mid-copy: half the bytes written in place
            Path(dst).write_bytes(Path(src).read_bytes()[:20])
            raise KeyboardInterrupt
        return copy(src, dst, *a, **kw)

    def stopped_rename(src, dst):
        if Path(dst) == shipped:
            raise KeyboardInterrupt
        return rename(src, dst)
    monkeypatch.setattr(shutil, "copyfile", stopped_copy)
    monkeypatch.setattr(os, "replace", stopped_rename)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path)
    assert shipped.read_bytes() == before                                         # whole, as it was
    rec = list(csv.DictReader(open(tmp_path / "recert.tsv"), delimiter="\t"))
    assert [(r["key"], r["memory_class"]) for r in rec] == [(K2, "16g")]           # listed before the replacement
    assert [r["file"] for r in csv.DictReader(open(tmp_path / "table.tsv"), delimiter="\t")] == [
        "nvidia/volta/toy_kernel.fp32.json"]
    monkeypatch.undo()
    assert _run(tmp_path)["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]
    assert list(json.loads(shipped.read_text())["entries"]) == [K1]
    assert len(list(csv.DictReader(open(tmp_path / "recert.tsv"), delimiter="\t"))) == 1
    assert sorted(p.name for p in shipped.parent.iterdir()) == ["toy_kernel.fp32.json"]   # no tmp left


def test_the_proofs_of_other_generators_that_leave_are_counted_and_listed(tmp_path):
    """qa 1227-4: the re-prove runs under one generator; a shipped head also proven under another (`also_proven`)
    loses that proof with the replacement — counted in the table, listed by generator, never dropped silently."""
    other = {**_proof(), "backend": {"triton": "3.7.0", "name": "cuda"}}
    shipped = _write(tmp_path / "shipped", {K1: {"config": _cfg(1), "proof": _proof(), "also_proven": [other]}})
    _write(tmp_path / "draft", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}})
    _ledger(tmp_path / "draft", [(K1, "reproven")])
    assert _run(tmp_path)["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]
    rec = list(csv.DictReader(open(tmp_path / "recert.tsv"), delimiter="\t"))
    assert [(r["key"], r["memory_class"], r["generator"]) for r in rec] == [(K1, "16g", C.proof_backend(other))]
    assert C.proof_backend(_proof()) in rec[0]["reason"]
    table = list(csv.DictReader(open(tmp_path / "table.tsv"), delimiter="\t"))
    assert (table[0]["re_proven"], table[0]["generators_dropped"]) == ("1", "1")
