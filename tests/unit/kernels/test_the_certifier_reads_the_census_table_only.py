"""The certifier certifies the committed census table of its profile and card memory class, and
nothing else: not the machine's replay cache (earlier runtime sweeps), not a campaign census file.

The owner's method (2026-09-28, point 2): one certified directory per profile, certified from the one
census table. `autotune_certify.census()` used to default to the replay cache and take any census JSON.

What each test would do if the code were wrong: a census() still reading the replay cache or a file
returns keys the table does not hold (the first test's equality fails); one that falls back when the
table is absent returns something instead of raising (the second fails); a certify() that accepts
`--census` proceeds (the third fails). (Seen red: census() reading the 32 GB table for a 16 GB card.)
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import autotune_certify as A
from neurobrix.kernels import census_table as T

K1 = "(19, 2048, 2048, True, True, 'fp16', 'fp16', 'fp16')"
K2 = "(64, 2048, 2048, True, True, 'fp16', 'fp16', 'fp16')"
QUAL = "neurobrix.kernels.ops.matmul.matmul_kernel"


def _row(model, key, mode="triton"):
    return {"model": model, "container": "s", "mode": mode, "rungs_mb": None, "ops": [None],
            "kernel": QUAL, "key": key, "dtype": T.dtypes_of(key), "tool": "t"}


def test_the_census_is_the_table_of_the_class(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "ROOT", tmp_path)
    T.write(T.table_path("nvidia", "volta", 16), [_row("A", K1), _row("B", K1, mode="triton-sequential")])
    T.write(T.table_path("nvidia", "volta", 32), [_row("A", K2)])
    got = A.census("nvidia", "volta", 16)
    assert list(got) == [QUAL] and [tuple(k) for k in got[QUAL]] == [tuple(eval(K1))]


def test_no_table_no_census(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="no census table"):
        A.census("nvidia", "volta", 16)


def test_a_census_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "rig_protocol_refusal", lambda **k: None)
    monkeypatch.setattr(A.C, "active_profile", lambda: ("nvidia", "volta"))
    monkeypatch.setattr(A, "_bind_hardware_profile", lambda: "test")
    monkeypatch.setattr(A, "_certifying_device", lambda: {"name": "card", "ordinal": 0, "visible_devices": "0",
                                                          "memory_mb": 16384})
    with pytest.raises(RuntimeError, match="REFUSED"):
        A.certify("volta", census_path=str(tmp_path / "census.json"), log=lambda *a: None)
