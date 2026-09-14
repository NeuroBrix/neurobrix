"""A proof that does not say its clock is not coverage under `--reprove-unclocked`.

5 628 proofs of 2026-09-07 record no `clocks_mhz`: made before the clock door
at a frequency nothing recorded, their timings read 1.176× a proof at the
protocol lock (register 56, adjudicated 2026-09-14 01:47 — the 32 GB and
16 GB cards agree at 0.998 the same night). `--only-missing` keeps them;
`--reprove-unclocked` re-proves them and keeps the ones behind the door.

Injection (2026-09-14): with `need_clock` ignored in `entry_covers`, the
second test went RED; restored, green.
"""
from neurobrix.kernels import autotune_certified as C


def _entry(clocks):
    proof = {"machine": {"device": {"ordinal": 0, "visible_devices": "0", "name": "V100", "memory_mb": 16384}}}
    if clocks:
        proof["machine"]["clocks_mhz"] = {"0": {"graphics": 1290, "memory": 877}}
    return {"config": {"kwargs": {}, "num_warps": 4, "num_stages": 3}, "proof": proof, "excluded": []}


def test_only_missing_keeps_a_clockless_proof():
    entries = {"k": _entry(clocks=False)}
    assert C.entry_covers(entries, "k", 16) is True


def test_reprove_unclocked_does_not_count_a_clockless_proof_as_coverage():
    assert C.entry_covers({"k": _entry(clocks=False)}, "k", 16, need_clock=True) is False
    assert C.entry_covers({"k": _entry(clocks=True)}, "k", 16, need_clock=True) is True
    assert C.entry_covers({}, "k", 16, need_clock=True) is False


def test_the_clock_predicate_reads_the_record_not_its_presence():
    assert C.proof_records_clock({"machine": {"clocks_mhz": {}}}) is False
    assert C.proof_records_clock({"machine": {"clocks_mhz": {"0": {"graphics": 1290}}}}) is True
    assert C.proof_records_clock(None) is False
