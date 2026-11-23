"""A shipped certified file is replaced by its epoch re-proof (`tools/certified_epoch_stamp.py`) only when every
certification it owes is re-proven at the library's bits epoch with the shipped configuration, or listed failed for
the full recertification; the replacement is the draft (never mixed); the failed ones leave the file and join the
recertification list; one table row per file.

What would this file do if the code were wrong? A file with a key not yet tried replaced -> the first test RED; a
draft configuration that differs counted re-proven -> the second RED; a failed key dropped silently, not listed, or a
replacement holding an epoch-less proof, or no table row -> the third RED; a re-chosen unit tile counted re-proven ->
the unit test RED; a stop during the replacement losing a failed key or truncating the file -> the stop test RED; a
retired key kept in the file, put on the recertification list or not counted apart -> the retired test RED; a file
whose every key is retired shipped (empty or as it was), or one with a key failed removed -> the removal test RED. A
subset draft's certification not added, added over another class's proof, or an existing certification of the same
key and class overwritten -> the two add tests RED. Each guard of the add removed (qa 1245-1) -> its own test RED: a
proof of another bits epoch accepted, a draft entry of two classes accepted, a merge that fails the library's validate
written, a draft tree that is the shipped tree run.
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


def _proof(epoch=None, n=1, mem=16384):
    """A proof of the key (n, 'fp32'): the library's gate refuses a proof whose shape is not its entry's key, and a
    draft refused for that is never refused for the reason a test means (qa 1227-3: the configuration test was)."""
    p = {"date": "2026-11-01T00:00:00+00:00", "engine_version": "0", "backend": {"triton": "3.8.0", "name": "cuda"},
         "shape": [n, "fp32"], "deviation": 0.0, "tolerance": 1e-4, "oracle": "fp64",
         "machine": {"device": {"memory_mb": mem}, "clocks_mhz": {"0": {"graphics": 1290}}},
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
                      "seconds": "5.0", "generators_dropped": "0", "retired": "0"}]


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


def test_a_shipped_certification_already_at_the_epoch_is_not_lost_by_the_replacement(tmp_path):
    """qa 1227-6: the check runs both ways — a shipped certification not owed (already of the library's epoch) that
    the draft does not hold keeps the file from being replaced; the draft holding it as shipped lets it through."""
    K3 = C.key_repr((3, "fp32"))
    at_epoch = {"config": _cfg(3), "proof": _proof(C.BITS_EPOCH, n=3)}
    shipped = _write(tmp_path / "shipped", {K1: {"config": _cfg(1), "proof": _proof()}, K3: at_epoch})
    _write(tmp_path / "draft", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}})
    _ledger(tmp_path / "draft", [(K1, "reproven")])
    before = shipped.read_bytes()
    assert _run(tmp_path)["replaced"] == [] and shipped.read_bytes() == before
    _write(tmp_path / "draft", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}, K3: at_epoch})
    assert _run(tmp_path)["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]
    assert sorted(json.loads(shipped.read_text())["entries"]) == sorted([K1, K3])


def test_a_retired_key_leaves_the_file_counted_apart_and_off_the_recert_list(tmp_path):
    """A key the frozen library gives no launch (the matrix unit takes it: the wrapper never reached the autotuner) is
    named in a --retired list (file, key, memory_class, verdict retired_*): it leaves the file with the replacement,
    is counted apart in the table, and never joins the recertification list (there is nothing to recertify). A row
    of another verdict in the same list retires nothing."""
    shipped = _setup(tmp_path, {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}},
                     [(K1, "reproven"), (K2, "failed")])
    retired = tmp_path / "retired.tsv"
    retired.write_text("file\tmemory_class\tverdict\tkey\n"
                       f"nvidia/volta/toy_kernel.fp32.json\t16g\tretired_unit_routed\t{K2}\n"
                       f"nvidia/volta/toy_kernel.fp32.json\t16g\treproven_live\t{K1}\n")
    out = S.run(tmp_path / "shipped", tmp_path / "draft", tmp_path / "table.tsv", tmp_path / "recert.tsv",
                apply=True, say=lambda *a: None, retired=[retired])
    assert out["replaced"] == ["nvidia/volta/toy_kernel.fp32.json"]
    assert list(json.loads(shipped.read_text())["entries"]) == [K1]
    assert not (tmp_path / "recert.tsv").exists() and out["recert"] == []
    table = list(csv.DictReader(open(tmp_path / "table.tsv"), delimiter="\t"))
    assert [(r["re_proven"], r["failed"], r["retired"]) for r in table] == [("1", "0", "1")]


def test_a_file_whose_every_key_is_retired_is_removed_not_shipped_empty(tmp_path):
    """A file every certification of which is retired (no key the library forms) has nothing left to ship: --apply
    REMOVES it (an empty certified file is not shipped), its table row counting the keys retired, nothing on the
    recertification list. A file with one key failed and not retired is still left as it is, for a ruling."""
    shipped = _write(tmp_path / "shipped", {K1: {"config": _cfg(1), "proof": _proof()},
                                            K2: {"config": _cfg(2), "proof": _proof(n=2)}})
    (tmp_path / "draft").mkdir()
    _ledger(tmp_path / "draft", [(K1, "failed"), (K2, "failed")])
    retired = tmp_path / "retired.tsv"
    retired.write_text("file\tmemory_class\tverdict\tkey\n"
                       f"nvidia/volta/toy_kernel.fp32.json\t16g\tretired_out_dtype_unformed\t{K1}\n")
    run = lambda: S.run(tmp_path / "shipped", tmp_path / "draft", tmp_path / "table.tsv", tmp_path / "recert.tsv",
                        apply=True, say=lambda *a: None, retired=[retired])
    before = shipped.read_bytes()
    out = run()                                                   # K2 failed, not retired: left for a ruling
    assert out["replaced"] == [] and out["removed"] == [] and shipped.read_bytes() == before
    retired.write_text(retired.read_text() + f"nvidia/volta/toy_kernel.fp32.json\t16g\tretired_out_dtype_unformed\t{K2}\n")
    out = run()
    assert out["removed"] == ["nvidia/volta/toy_kernel.fp32.json"] and not shipped.exists()
    assert out["recert"] == [] and not (tmp_path / "recert.tsv").exists()
    table = list(csv.DictReader(open(tmp_path / "table.tsv"), delimiter="\t"))
    assert [(r["keys"], r["re_proven"], r["failed"], r["retired"]) for r in table] == [("2", "0", "0", "2")]


def test_a_retired_key_the_draft_still_holds_stays_pending_and_the_file_is_not_replaced(tmp_path):
    """qa 1244-2: a key named retired that the draft holds is a contradiction for a ruling, never counted retired: the
    file stays as shipped, the key said pending."""
    shipped = _setup(tmp_path, {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)},
                                K2: {"config": _cfg(2), "proof": _proof(C.BITS_EPOCH, n=2)}},
                     [(K1, "reproven"), (K2, "reproven")])
    retired = tmp_path / "retired.tsv"
    retired.write_text("file\tmemory_class\tverdict\tkey\n"
                       f"nvidia/volta/toy_kernel.fp32.json\t16g\tretired_unit_routed\t{K2}\n")
    said = []
    before = shipped.read_bytes()
    out = S.run(tmp_path / "shipped", tmp_path / "draft", tmp_path / "table.tsv", tmp_path / "recert.tsv",
                apply=True, say=said.append, retired=[retired])
    assert out["replaced"] == [] and out["removed"] == [] and shipped.read_bytes() == before
    assert any("named retired, yet the draft holds it" in s for s in said)
    table = list(csv.DictReader(open(tmp_path / "table.tsv"), delimiter="\t"))
    assert [(r["re_proven"], r["retired"]) for r in table] == [("1", "0")]


def _unit_cert(b, n, mem):
    return {"config": _cfg(b), "proof": _proof(C.BITS_EPOCH, n=n, mem=mem), "excluded": [],
            "unit": {"tile": {"block_m": 64, "block_n": 64, "block_k": 32, "warps_m": 2, "warps_n": 2, "group_m": 8},
                     "deviation": 0.0, "best_ms": 1.0}}


def _add(tmp_path, *drafts, apply=True):
    return S.add(tmp_path / "shipped", [tmp_path / d for d in drafts], tmp_path / "add_table.tsv", apply=apply,
                 say=lambda *a: None)


def test_a_subset_draft_adds_its_certifications_and_a_class_beside_the_other(tmp_path):
    """1245: a recertification draft holds a SUBSET of a file's keys, one memory class. Its certification is ADDED: a
    key the file lacks gets the draft's entry as it stands; a key the file holds for the OTHER class gets this class
    beside it (the variant slot), the other class's certification untouched; every other entry stays as it was."""
    K3 = C.key_repr((3, "fp32"))
    k1 = {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}
    k3 = {"config": _cfg(3), "proof": _proof(C.BITS_EPOCH, n=3)}
    shipped = _write(tmp_path / "shipped", {K1: k1, K3: k3})
    _write(tmp_path / "d16", {K2: _unit_cert(2, 2, 16384)})
    _write(tmp_path / "d32", {K1: _unit_cert(5, 1, 32768)})
    assert _add(tmp_path, "d16", "d32", apply=False)["written"] == [] and json.loads(shipped.read_text())["entries"] == {
        K1: k1, K3: k3}                                                        # no --apply: nothing written
    out = _add(tmp_path, "d16", "d32")
    assert out["written"] == ["nvidia/volta/toy_kernel.fp32.json"]
    doc = json.loads(shipped.read_text())
    e = doc["entries"]
    assert e[K3] == k3 and e[K2] == _unit_cert(2, 2, 16384)
    assert {k: v for k, v in e[K1].items() if k != "variants"} == k1          # the 16 GB certification untouched
    assert e[K1]["variants"] == {"32g": _unit_cert(5, 1, 32768)}
    assert C.entry_for_memory_gb(e[K1], 32) == _unit_cert(5, 1, 32768) and C.entry_for_memory_gb(e[K1], 16) == e[K1]
    assert C.validate(doc, shipped) == []
    table = list(csv.DictReader(open(tmp_path / "add_table.tsv"), delimiter="\t"))
    assert [(r["added"], r["variants"], r["refused"]) for r in table] == [("1", "1", "0")]


def test_an_existing_certification_of_the_same_key_and_class_is_never_overwritten(tmp_path):
    """The add never replaces: a draft certification whose key and class the file already holds is refused by name,
    and the file is not written at all (whole or nothing), even for the other keys of the same draft."""
    k1 = {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}
    shipped = _write(tmp_path / "shipped", {K1: k1})
    _write(tmp_path / "d16", {K1: _unit_cert(7, 1, 16384), K2: _unit_cert(2, 2, 16384)})
    before = shipped.read_bytes()
    said = []
    out = S.add(tmp_path / "shipped", [tmp_path / "d16"], tmp_path / "add_table.tsv", apply=True, say=said.append)
    assert out["written"] == [] and shipped.read_bytes() == before
    assert any("never overwritten" in s and K1 in s for s in said)
    table = list(csv.DictReader(open(tmp_path / "add_table.tsv"), delimiter="\t"))
    assert [(r["added"], r["variants"], r["refused"]) for r in table] == [("1", "0", "1")]


def _refused(tmp_path, cert, why):
    """A draft whose one certification is refused by `why`: refused by name, counted, and the file not written."""
    k1 = {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}
    shipped = _write(tmp_path / "shipped", {K1: k1})
    _write(tmp_path / "d16", {K2: cert})
    before, said = shipped.read_bytes(), []
    out = S.add(tmp_path / "shipped", [tmp_path / "d16"], tmp_path / "add_table.tsv", apply=True, say=said.append)
    assert out["written"] == [] and shipped.read_bytes() == before
    assert any(why in s and K2 in s and "REFUSED" in s for s in said)
    table = list(csv.DictReader(open(tmp_path / "add_table.tsv"), delimiter="\t"))
    assert [(r["added"], r["variants"], r["refused"]) for r in table] == [("0", "0", "1")]


def test_a_draft_proof_of_another_bits_epoch_is_refused_by_name(tmp_path):
    """1245-1, the epoch guard: a draft certification whose proof is not at the library's bits epoch is refused by its
    own reason (not left to the later gate), and the file is not written."""
    cert = _unit_cert(2, 2, 16384)
    cert["proof"]["bits_epoch"] = C.BITS_EPOCH + 1
    _refused(tmp_path, cert, "not of the library's bits epoch")


def test_a_draft_entry_of_two_memory_classes_is_refused_by_name(tmp_path):
    """1245-1, the one-class guard: a draft entry that carries a variant (it covers 16 and 32 GB) is refused, never
    added whole, and the file is not written."""
    cert = _unit_cert(2, 2, 16384)
    cert["variants"] = {"32g": _unit_cert(3, 2, 32768)}
    assert sorted(C.covered_memory_gbs(cert)) == [16, 32]
    _refused(tmp_path, cert, "not one memory class")


def test_a_merge_the_library_validate_refuses_is_not_written(tmp_path):
    """1245-1, the validate gate: a draft certification no guard of the add refuses (one class, at the epoch, a key the
    file lacks) whose merge the library's `validate` refuses (its proof's shape is not its key) leaves the file as it
    was, and says the gate."""
    k1 = {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}
    shipped = _write(tmp_path / "shipped", {K1: k1})
    _write(tmp_path / "d16", {K2: _unit_cert(2, 1, 16384)})                  # key (2, fp32), proof shape (1, fp32)
    assert S._add_one({}, K2, _unit_cert(2, 1, 16384)) == "added"            # no guard of the add refuses it
    before, said = shipped.read_bytes(), []
    out = S.add(tmp_path / "shipped", [tmp_path / "d16"], tmp_path / "add_table.tsv", apply=True, say=said.append)
    assert out["written"] == [] and shipped.read_bytes() == before
    assert any("NOT WRITTEN, gate:" in s for s in said)
    table = list(csv.DictReader(open(tmp_path / "add_table.tsv"), delimiter="\t"))
    assert [(r["added"], r["variants"], r["refused"]) for r in table] == [("1", "0", "0")]


def test_a_draft_tree_that_is_the_shipped_tree_is_refused_before_any_read(tmp_path):
    """1245-1, the same-tree guard: --add-from naming the shipped tree itself is refused by the command, with --apply,
    before anything is read or written (no table)."""
    shipped = _write(tmp_path / "shipped", {K1: {"config": _cfg(1), "proof": _proof(C.BITS_EPOCH)}})
    before = shipped.read_bytes()
    with pytest.raises(SystemExit) as stop:
        S.main(["--shipped", str(tmp_path / "shipped"), "--table", str(tmp_path / "add_table.tsv"), "--apply",
                "--add-from", str(tmp_path / "x" / ".." / "shipped")])
    assert stop.value.code == "a draft tree is the shipped tree: refused"
    assert shipped.read_bytes() == before and not (tmp_path / "add_table.tsv").exists()
