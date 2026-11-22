"""The epoch re-prove proves a matrix-unit certification's SHIPPED tile as shipped (qa 1227-1): the library serves the
launch with the entry's `unit.tile`, not its `config`, so a re-proof that times the declared tiles again and names
another is a re-tuning, not a re-proof. The re-prove times the shipped tile alone (`shipped_unit_runs`), and a key whose
re-proven tile is not the shipped one — or that gains or loses a tile — is failed in the ledger, for the full
recertification, and never filed in the draft.

What would this file do if the code were wrong? The unit section choosing among the declared tiles again, or a
disagreement between the shipped tile and the rule passed over -> the first two tests RED; the loop filing a re-chosen,
gained or lost tile as re-proven, or not handing the shipped tile to the re-proof -> the third RED.
"""
import json
from types import SimpleNamespace

import pytest

from nbxtensor.ops import autotune_certified as C
from neurobrix.kernels import autotune_certify as CF

QUAL = "pkg.mod.toy_kernel"
T64 = {"block_m": 64, "block_n": 64}
T128 = {"block_m": 128, "block_n": 64}


def _proof(mb=16384):
    return {"date": "2026-11-01T00:00:00+00:00", "backend": {"triton": "3.8.0", "name": "cuda"},
            "machine": {"device": {"memory_mb": mb}}}


def _cfg(b):
    return {"kwargs": {"BLOCK": b}, "num_warps": 4, "num_stages": 2, "num_ctas": 1, "maxnreg": None}


def test_a_reproof_times_the_shipped_tile_alone():
    runs = [(dict(T128), "run128"), (dict(T64), "run64")]
    assert CF.shipped_unit_runs(runs, CF.unit_tile({"unit": {"tile": T64}}), "k") == [(T64, "run64")]
    assert CF.shipped_unit_runs([], None, "k") == []
    assert CF.unit_tile({"config": _cfg(1)}) is None


@pytest.mark.parametrize("runs, shipped, why", [
    ([(T64, "r")], None, "names no unit tile"),                       # the rule now gives it to the unit
    ([], T64, "no unit run"),                                         # the rule no longer does
    ([(T128, "r")], T64, "not one of the profile's"),                 # the shipped tile is no longer declared
])
def test_a_shipped_tile_the_rule_disagrees_with_is_not_re_proven(runs, shipped, why):
    with pytest.raises(RuntimeError, match=why):
        CF.shipped_unit_runs(runs, shipped, "k")


def test_a_re_chosen_gained_or_lost_unit_tile_is_failed_never_filed(tmp_path, monkeypatch):
    shipped = tmp_path / "shipped"
    draft = tmp_path / "draft"
    d = shipped / "nvidia" / "volta"
    d.mkdir(parents=True)
    head = lambda b, tile=None: {"config": _cfg(b), "proof": _proof(), **({"unit": {"tile": tile}} if tile else {})}
    entries = {C.key_repr((1, "fp32")): head(1, T64),     # re-proven with its tile: filed
               C.key_repr((2, "fp32")): head(2, T64),     # the re-proof comes back with another tile: failed
               C.key_repr((3, "fp32")): head(3),          # gains a tile: failed
               C.key_repr((4, "fp32")): head(4, T64)}     # loses its tile: failed
    (d / "toy_kernel.fp32.json").write_text(json.dumps({
        "format": "nbx-autotune-certified/2", "vendor": "nvidia", "profile": "volta", "kernel": QUAL,
        "dtype": "fp32", "entries": entries}))
    monkeypatch.setattr(C, "output_dtype", lambda tuner, key: "fp32")
    monkeypatch.setattr(C, "describe_key", lambda tuner, key: str(key))
    monkeypatch.setattr(CF, "_tolerance", lambda *a: 1e-3)
    monkeypatch.setattr(CF, "_num_stages_space", lambda *a: [1])
    monkeypatch.setattr(CF, "_backend", lambda: {"triton": "3.8.0", "name": "cuda"})
    handed = {}
    back = {(1, "fp32"): T64, (2, "fp32"): T128, (3, "fp32"): T64, (4, "fp32"): None}

    def fake_certify_key(qual, tuner, key, tol, rng, only_config=None, reproven_from=None, **kw):
        handed[key] = kw.get("keep_unit", "not handed")
        p = {**_proof(), "deviation": 0.0, "best_ms": 1.0, "accepted": 1, "candidates": 1,
             "seconds": {"oracle": 0, "runs": 0, "bench": 0}, "reproven_from": reproven_from}
        tile = back[key]
        return {"config": dict(only_config), "proof": p, "excluded": [],
                **({"unit": {"tile": tile, "deviation": 0.0, "best_ms": 1.0}} if tile else {})}
    monkeypatch.setattr(CF, "certify_key", fake_certify_key)

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
    assert handed == {(1, "fp32"): T64, (2, "fp32"): T64, (3, "fp32"): None, (4, "fp32"): T64}
    doc = json.loads((draft / "nvidia" / "volta" / "toy_kernel.fp32.json").read_text())
    assert list(doc["entries"]) == [C.key_repr((1, "fp32"))]
    assert doc["entries"][C.key_repr((1, "fp32"))]["unit"]["tile"] == T64
    rows = [l.split("\t") for l in (draft / CF.REPROVE_LEDGER).read_text().splitlines()[1:]]
    out = {r[1]: (r[3], r[5]) for r in rows}
    assert out[C.key_repr((1, "fp32"))][0] == "reproven"
    for k in (2, 3, 4):
        assert out[C.key_repr((k, "fp32"))][0] == "failed" and "unit tile" in out[C.key_repr((k, "fp32"))][1]
