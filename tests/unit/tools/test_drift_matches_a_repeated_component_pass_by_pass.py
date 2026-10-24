"""`neurobrix drift` compares a component run several times (a vendor-tiled VAE's tiles, a decode's steps) PASS BY
PASS: every per-op dump record carries its pass, the drift command asks both arms for every pass
(NBX_DUMP_TIDS_PASS=all), and the walk keys a record by its component tagged with that pass.

The Mac, Allegro-TI2V (2026-10-24): the oracle arm dumped the vendor-tiled vae_encoder 30 times, the engine arm once
(its dump kept the first pass per tid), and the walk compared different tiles — no drift site could be named.

What this test would do if the code were wrong: records keyed by tid alone collapse the three passes to one — the
first assertion counts them; a walk that matched pass 2 of one side to pass 0 of the other would report a deviation
on identical tiles — the second assertion fails.
"""
import json

from neurobrix.core.dtype import drift as D


def _dump(path, passes):
    with open(path, "w") as f:
        for p, head in passes:
            f.write(json.dumps({"engine": "x", "record": {"component": "vae_encoder", "pass": p, "tid": "t1",
                                                          "op_uid": "aten.conv::1", "op_type": "aten::convolution",
                                                          "head10": head, "l2_norm": sum(head)}}) + "\n")


def test_every_pass_of_a_repeated_component_is_its_own_record(tmp_path):
    _dump(tmp_path / "a.jsonl", [(0, [1.0] * 10), (1, [2.0] * 10), (2, [3.0] * 10)])
    recs, order = D.load_dump(tmp_path / "a.jsonl")
    assert len(order) == 3, order


def test_the_walk_pairs_pass_with_pass(tmp_path):
    _dump(tmp_path / "a.jsonl", [(0, [1.0] * 10), (1, [2.0] * 10), (2, [3.0] * 10)])
    _dump(tmp_path / "b.jsonl", [(0, [1.0] * 10), (1, [2.0] * 10), (2, [3.0] * 10)])
    rep = D.detect(tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    assert rep.matched == 3 and rep.over_bound == 0 and rep.first is None, rep.to_dict()
