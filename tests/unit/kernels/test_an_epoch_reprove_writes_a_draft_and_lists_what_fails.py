"""The epoch re-prove (`autotune certify --reprove-epoch-from <shipped> --out <draft>`, orchestrator 2026-11-22 15:37):
each stored configuration whose proof is of another bits epoch than the library's is proven AGAIN, alone, into a
draft tree; the shipped tree is only read; a key whose stored configuration does not re-prove is not swept and not
dropped silently but written to the draft's ledger, for the full recertification.

What would this file do if the code were wrong? A draft at (or inside) the shipped tree accepted -> the first test
RED; a key already of the epoch, or of the other class, re-proven -> the second RED; a failed key swept (a call with no
`only_config`) or left out of the ledger, the shipped file written, a draft proof with no epoch, a key the draft holds
tried again -> the third RED.
"""
import json
from types import SimpleNamespace

import pytest

from nbxtensor.ops import autotune_certified as C
from neurobrix.kernels import autotune_certify as CF

QUAL = "pkg.mod.toy_kernel"


def _proof(mb, epoch=None):
    p = {"date": "2026-11-01T00:00:00+00:00", "backend": {"triton": "3.8.0", "name": "cuda"},
         "machine": {"device": {"memory_mb": mb}}}
    if epoch is not None:
        p["bits_epoch"] = epoch
    return p


def _cfg(b):
    return {"kwargs": {"BLOCK": b}, "num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None}


def _shipped(root, entries):
    d = root / "nvidia" / "volta"
    d.mkdir(parents=True)
    path = d / "toy_kernel.fp32.json"
    path.write_text(json.dumps({"format": "nbx-autotune-certified/2", "vendor": "nvidia", "profile": "volta",
                                "kernel": QUAL, "dtype": "fp32", "entries": entries}))
    return path


@pytest.mark.parametrize("out", [None, "same", "inside"])
def test_a_reprove_writes_only_a_draft_tree_apart_from_the_shipped_one(tmp_path, out):
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    where = {None: None, "same": str(shipped), "inside": str(shipped / "draft")}[out]
    with pytest.raises(RuntimeError, match="draft"):
        CF.certify("volta", vendor="nvidia", out=where, reprove_epoch_from=str(shipped))


def test_the_keys_owed_are_this_class_heads_not_of_the_library_epoch(tmp_path):
    _shipped(tmp_path, {
        "(1, 'fp32')": {"config": _cfg(1), "proof": _proof(16384),
                        "variants": {"32g": {"config": _cfg(2), "proof": _proof(32768)}}},
        "(2, 'fp32')": {"config": _cfg(3), "proof": _proof(16384, C.BITS_EPOCH)},
        "(3, 'fp32')": {"config": _cfg(4), "proof": _proof(32768)},
    })
    shapes, stored, unreadable = CF.stored_epoch_keys("nvidia", "volta", tmp_path, 16)
    assert shapes == {QUAL: [(1, "fp32")]} and not unreadable
    assert stored[(QUAL, C.key_repr((1, "fp32")))]["config"] == _cfg(1)
    shapes32, stored32, _ = CF.stored_epoch_keys("nvidia", "volta", tmp_path, 32)
    assert sorted(shapes32[QUAL]) == [(1, "fp32"), (3, "fp32")]
    assert stored32[(QUAL, C.key_repr((1, "fp32")))]["config"] == _cfg(2)


def test_a_failed_reprove_is_listed_never_swept_and_the_shipped_file_is_not_written(tmp_path, monkeypatch):
    shipped = tmp_path / "shipped"
    draft = tmp_path / "draft"
    spath = _shipped(shipped, {"(1, 'fp32')": {"config": _cfg(1), "proof": _proof(16384)},
                               "(2, 'fp32')": {"config": _cfg(2), "proof": _proof(16384)}})
    before = spath.read_bytes()
    monkeypatch.setattr(C, "output_dtype", lambda tuner, key: "fp32")
    monkeypatch.setattr(C, "describe_key", lambda tuner, key: str(key))
    monkeypatch.setattr(CF, "_tolerance", lambda *a: 1e-3)
    monkeypatch.setattr(CF, "_num_stages_space", lambda *a: [1])
    monkeypatch.setattr(CF, "_backend", lambda: {"triton": "3.8.0", "name": "cuda"})
    calls = []

    def fake_certify_key(qual, tuner, key, tol, rng, only_config=None, reproven_from=None, **kw):
        calls.append((key, only_config))
        if key == (2, "fp32"):
            raise RuntimeError("the stored configuration exceeds the oracle's tolerance")
        p = {**_proof(16384), "deviation": 0.0, "best_ms": 1.0, "accepted": 1, "candidates": 1,
             "seconds": {"oracle": 0, "runs": 0, "bench": 0}, "reproven_from": reproven_from}
        return {"config": dict(only_config), "proof": p, "excluded": []}
    monkeypatch.setattr(CF, "certify_key", fake_certify_key)

    def loop():
        shapes, stored, _ = CF.stored_epoch_keys("nvidia", "volta", shipped, 16)
        summary = {"unreachable": 0, "skipped": 0, "failed": 0, "certified": 0, "excluded_configs": 0,
                   "kernels": {}, "started": 0.0}
        writer = CF._BoundedWriter("nvidia", "volta")
        try:
            CF._certify_loop(shapes, {QUAL: SimpleNamespace()}, "nvidia", "volta", draft, None, False, False, False,
                             {"memory_mb": 16384, "unified": False}, 16, None, 0, None, summary, lambda *a: None,
                             writer, None, reprove_from=stored, ledger=CF.ReproveLedger(draft / CF.REPROVE_LEDGER))
        finally:
            writer.flush_all()

    loop()
    assert spath.read_bytes() == before                                  # the shipped tree is only read
    assert all(oc is not None for _, oc in calls) and len(calls) == 2    # never a sweep
    doc = json.loads((draft / "nvidia" / "volta" / "toy_kernel.fp32.json").read_text())
    assert list(doc["entries"]) == [C.key_repr((1, "fp32"))]              # the failed key is not filed
    e = doc["entries"][C.key_repr((1, "fp32"))]
    assert e["config"] == _cfg(1) and e["proof"]["bits_epoch"] == C.BITS_EPOCH
    rows = [l.split("\t") for l in (draft / CF.REPROVE_LEDGER).read_text().splitlines()]
    assert rows[0] == list(CF.ReproveLedger.COLS)
    assert {r[0] for r in rows[1:]} == {"nvidia/volta/toy_kernel.fp32.json"}   # the profile's file, not a bare name
    got = {(r[1], r[2], r[3]) for r in rows[1:]}
    assert got == {(C.key_repr((1, "fp32")), "16g", "reproven"), (C.key_repr((2, "fp32")), "16g", "failed")}
    assert "tolerance" in [r for r in rows[1:] if r[3] == "failed"][0][5]
    calls.clear()
    loop()                                                               # resumed: the draft's key is not tried again
    assert [k for k, _ in calls] == [(2, "fp32")]
