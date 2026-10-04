"""The census is ONE table per vendor profile and memory class: rows with fixed columns, sorted,
de-duplicated, a model's rows replaced — never added to — when its container is censused again.

What each test would do if the code were wrong: a writer that appended would keep a retraced model's
old rows (the replace test fails); a reader that accepted a row without a column would let a table
lose the container sha or the mode unnoticed (the refusal test fails); two tables written for two
classes in one file would answer a 16 GB question with a 32 GB row (the path test fails); a writer
keeping one row per (op, key) would grow a transformer's table by its layer count (the ops test
fails); a reader accepting the old schema would let an unconverted table pass for a converted one, and
a migration that dropped an op or a rung would lose what the census measured (the migration tests fail).
(Seen red: the write identity with the op put back.)
"""
from __future__ import annotations

import pytest

from neurobrix.kernels import census_table as T


def _row(model, key, container="aaaa", mode="triton", rung=16384):
    return {"model": model, "container": container, "mode": mode, "rungs_mb": [rung], "ops": [None],
            "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "key": key,
            "dtype": T.dtypes_of(key), "tool": "t"}


K1 = "(19, 2048, 2048, True, True, 'fp16', 'fp16', 'fp16')"
K2 = "(64, 2048, 2048, True, True, 'fp16', 'fp16', 'fp16')"


def test_one_table_per_profile_and_class(tmp_path):
    assert T.table_path("nvidia", "volta", 16, tmp_path) != T.table_path("nvidia", "volta", 32, tmp_path)
    assert T.table_path("nvidia", "volta", 16, tmp_path).name == "16g.jsonl"


def test_rows_are_sorted_and_deduplicated(tmp_path):
    p = tmp_path / "t.jsonl"
    n = T.write(p, [_row("B", K1), _row("A", K2, rung=8192), _row("A", K2)])
    assert n == 2
    assert [r["rungs_mb"] for r in T.read(p)] == [[8192, 16384], [16384]]
    assert [r["model"] for r in T.read(p)] == ["A", "B"]
    assert T.dtypes_of(K1) == "fp16,fp16,fp16"


def test_a_model_censused_again_replaces_its_rows(tmp_path):
    p = tmp_path / "t.jsonl"
    T.write(p, [_row("A", K1, container="old"), _row("B", K1)])
    removed, written = T.replace_model(p, "A", [_row("A", K2, container="new")])
    rows = T.read(p)
    assert (removed, written) == (1, 1)
    assert {(r["model"], r["key"], r["container"]) for r in rows} == {("A", K2, "new"), ("B", K1, "aaaa")}


def test_a_row_without_a_column_is_refused(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"model": "A", "kernel": "k", "key": "(1,)"}\n')
    with pytest.raises(ValueError, match="without"):
        T.read(p)


def test_a_key_is_found_by_its_line(tmp_path):
    T.write(T.table_path("nvidia", "volta", 16, tmp_path), [_row("A", K1)])
    path, rows = T.rows_for("neurobrix.kernels.ops.matmul.matmul_kernel", K1, "nvidia", "volta", 16, tmp_path)
    assert [r["model"] for r in rows] == ["A"]
    assert T.rows_for("neurobrix.kernels.ops.matmul.matmul_kernel", K2, "nvidia", "volta", 16, tmp_path)[1] == []


def test_consolidation_reads_a_logs_dir_and_skips_foreign_json(tmp_path, monkeypatch):
    """The Mac's campaigns (2026-09-28 20:51): key files straight in a logs directory, beside JSON lists
    that are not census documents. Consolidation must read the former and step over the latter."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    import census_table as CT
    src = tmp_path / "logs_x"
    src.mkdir()
    (src / "perf_keys.json").write_text(json.dumps(["not", "a", "census"]))
    (src / "census.json").write_text(json.dumps({"models": {"M": {"graph_sha": "abc"}}}))
    (src / "M.triton.walk.keys").write_text(f"neurobrix.kernels.ops.matmul.matmul_kernel::{K1}\n")
    by_model, unhashed = CT.rows_of_source(src, "t")
    assert list(by_model) == ["M"] and by_model["M"][0]["mode"] == "triton" and not unhashed


def _replace_many(args):
    path, model = args
    from pathlib import Path
    from neurobrix.kernels import census_table as TT
    TT.replace_model(Path(path), model, [_row(model, K1), _row(model, K2)])


def test_many_writers_lose_no_model(tmp_path):
    """One census process per model writing one class table (2026-09-28 21:56): every model's rows
    survive. Without the lock, a read-modify-write that raced another dropped its rows."""
    import multiprocessing as mp
    p = tmp_path / "t.jsonl"
    models = [f"M{i:02d}" for i in range(24)]
    with mp.Pool(8) as pool:
        pool.map(_replace_many, [(str(p), m) for m in models])
    assert sorted({r["model"] for r in T.read(p)}) == models


def test_one_row_per_key_with_every_op_that_formed_it(tmp_path):
    p = tmp_path / "t.jsonl"
    a, b, c = _row("A", K1), _row("A", K1, rung=8192), _row("A", K1)
    a["ops"], b["ops"], c["ops"] = ["aten.mm::9"], ["aten.mm::7"], [None]
    assert T.write(p, [a, b, c]) == 1
    (row,) = T.read(p)
    assert row["ops"] == [None, "aten.mm::7", "aten.mm::9"]
    assert row["rungs_mb"] == [8192, 16384]
    with pytest.raises(ValueError, match="ops must be a list"):
        T.write(p, [dict(a, ops="aten.mm::9")])


def _old_row(model, key, op, rung):
    r = _row(model, key, rung=rung)
    del r["ops"]
    r["op"] = op
    return r


def test_an_old_schema_table_is_refused_by_name_and_migrated_once(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    p = tmp_path / "16g.jsonl"
    old = [_old_row("A", K1, "aten.mm::7", 16384), _old_row("A", K1, "aten.mm::9", 8192),
           _old_row("A", K2, None, 16384), _old_row("B", K1, "aten.mm::7", 16384)]
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in old))
    with pytest.raises(ValueError, match="census_table.py migrate"):
        T.read(p)
    tool = Path(__file__).resolve().parents[3] / "tools" / "census_table.py"
    for _ in range(2):                                   # the second run finds nothing to convert
        r = subprocess.run([sys.executable, str(tool), "migrate", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    rows = T.read(p)
    got = [(r["model"], r["key"][:4], r["ops"], r["rungs_mb"]) for r in rows]
    assert got == [("A", "(19,", ["aten.mm::7", "aten.mm::9"], [8192, 16384]),
                   ("A", "(64,", [None], [16384]),
                   ("B", "(19,", ["aten.mm::7"], [16384])]


def test_a_renamed_container_retires_its_old_rows(tmp_path):
    # A retrace under a new name replaces, never adds: the old name's rows leave with its container.
    # Wrong code (no retirement) keeps "old" and this fails; seen red with the filter inverted.
    p = tmp_path / "t.jsonl"
    T.write(p, [_row("old", K1), _row("old", K2), _row("new", K1), _row("B", K2)])
    gone = T.retire_absent(p, {"new", "B"})
    assert gone == {"old": 2}
    assert {(r["model"], r["key"]) for r in T.read(p)} == {("new", K1), ("B", K2)}
    assert T.retire_absent(p, {"new", "B"}) == {}          # idempotent: nothing left to retire


def test_an_empty_listing_is_refused_and_the_table_kept(tmp_path):
    # A mis-pointed cache lists nothing; read as "every container is gone" it would empty the table.
    p = tmp_path / "t.jsonl"
    T.write(p, [_row("A", K1)])
    with pytest.raises(T.EmptyCensus, match="no container is present"):
        T.retire_absent(p, set())
    assert [r["model"] for r in T.read(p)] == ["A"]
