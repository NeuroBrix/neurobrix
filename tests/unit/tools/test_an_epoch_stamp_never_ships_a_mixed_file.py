"""A shipped certified file is replaced by its epoch re-proof (`tools/certified_epoch_stamp.py`) only when every
certification it owes is re-proven at the library's bits epoch with the shipped configuration, or listed failed for
the full recertification; the replacement is the draft (never mixed); the failed ones leave the file and join the
recertification list; one table row per file.

What would this file do if the code were wrong? A file with a key not yet tried replaced -> the first test RED; a
draft configuration that differs counted re-proven -> the second RED; a failed key dropped silently, not listed, or a
replacement holding an epoch-less proof, or no table row -> the third RED.
"""
import csv
import json
import sys
from pathlib import Path

from nbxtensor.ops import autotune_certified as C

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import certified_epoch_stamp as S  # noqa: E402

QUAL = "pkg.mod.toy_kernel"
K1, K2 = C.key_repr((1, "fp32")), C.key_repr((2, "fp32"))


def _cfg(b):
    return {"kwargs": {"BLOCK": b}, "num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None}


def _proof(epoch=None):
    p = {"date": "2026-11-01T00:00:00+00:00", "engine_version": "0", "backend": {"triton": "3.8.0", "name": "cuda"},
         "shape": [1, "fp32"], "deviation": 0.0, "tolerance": 1e-4, "oracle": "fp64",
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
                                            K2: {"config": _cfg(2), "proof": _proof()}})
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
                                K2: {"config": _cfg(2), "proof": _proof(C.BITS_EPOCH)}},
                     [(K1, "reproven"), (K2, "reproven")])
    before = shipped.read_bytes()
    assert _run(tmp_path)["replaced"] == [] and shipped.read_bytes() == before


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
                      "seconds": "5.0"}]
