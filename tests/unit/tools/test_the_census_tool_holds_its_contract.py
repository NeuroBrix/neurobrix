"""The census tool's contract (docs/reference/tool-contracts.md, "The census"), one test per clause.

The owner, 2026-09-29 13:45: many of the project's problems came from the tools that census,
certify and gate; each gets a written contract and the tests that hold it. The audit of that
day found the census table silently emptied three ways (an empty --modes, a shadow that formed
no key, a derivation over no mode or rung), an empty --models censusing the whole cache, a model
absent from the cache ending in a traceback, a report written in place (a stale one surviving an
interrupted run), and a two-writer test that could pass on timing alone.

What would this file do if the code were wrong? Each test names its injection in its docstring;
every one was run and seen RED before this file was committed.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import certified_census as CC  # noqa: E402
from neurobrix.kernels import census_table as T  # noqa: E402

PROFILE = "c4140-4xv100-16GB-nvlink"          # a committed profile: its memory class names the table


def _row(model, key="(64, 64, 64, 'fp16')", mode="triton"):
    return {"model": model, "container": "c0ffee", "mode": mode, "rungs_mb": [4096],
            "ops": ["aten.mm::0"], "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "key": key,
            "dtype": "fp16", "tool": "test"}


# --- the table -------------------------------------------------------------------------------

def test_a_model_s_rows_are_never_replaced_by_none(tmp_path):
    """Injection: the EmptyCensus door removed -> the table loses model A, RED."""
    p = tmp_path / "16g.jsonl"
    T.replace_model(p, "A", [_row("A")])
    before = p.read_bytes()
    with pytest.raises(T.EmptyCensus, match="no rows"):
        T.replace_model(p, "A", [])
    assert p.read_bytes() == before


def test_a_model_proven_keyless_loses_its_rows_and_only_then(tmp_path):
    """A derivation that PLACED its kernel-bearing ops with no key (a matrix unit's launches) is the knowledge
    that the model forms none: its stale rows go, every other model's stay. Without that proof the door holds.
    Injection: the `keyless_ops` proof ignored -> the stale rows stay, RED; the door opened for 0 -> RED above."""
    p = tmp_path / "16g.jsonl"
    T.replace_model(p, "A", [_row("A")])
    T.replace_model(p, "B", [_row("B")])
    with pytest.raises(T.EmptyCensus, match="no keyless op placed"):
        T.replace_model(p, "A", [], keyless_ops=0)
    assert T.replace_model(p, "A", [], keyless_ops=3) == (1, 0)
    assert {r["model"] for r in T.read(p)} == {"B"}


def test_replacing_a_model_twice_with_the_same_rows_is_byte_identical(tmp_path):
    """Injection: rows written unsorted or ops un-canonicalised -> the bytes differ, RED."""
    p = tmp_path / "16g.jsonl"
    T.replace_model(p, "B", [_row("B", "(1,)"), _row("B", "(2,)")])
    T.replace_model(p, "A", [_row("A", "(3,)"), _row("A", "(1,)")])
    once = p.read_bytes()
    T.replace_model(p, "A", [_row("A", "(1,)"), _row("A", "(3,)")])
    assert p.read_bytes() == once


def _slow_writer(path, model, go):
    """Reads the table, waits long enough for the other writer to read the same table, writes."""
    real_read = T.read

    def slow_read(p):
        rows = real_read(p)
        time.sleep(0.6)
        return rows
    T.read = slow_read
    go.wait()
    T.replace_model(Path(path), model, [_row(model)])


def test_two_writers_on_one_table_lose_nothing(tmp_path):
    """Two processes, each stalled between its read and its write, released together.
    Injection: `locked` made a no-op -> both read the empty table, the last write drops the
    other's model, RED (deterministic: the stall is longer than the release skew)."""
    p = tmp_path / "16g.jsonl"
    T.replace_model(p, "Z", [_row("Z")])
    ctx = mp.get_context("fork")
    go = ctx.Event()
    procs = [ctx.Process(target=_slow_writer, args=(str(p), m, go)) for m in ("A", "B")]
    for pr in procs:
        pr.start()
    go.set()
    for pr in procs:
        pr.join(30)
        assert pr.exitcode == 0
    assert {r["model"] for r in T.read(p)} == {"A", "B", "Z"}


def test_a_failed_write_leaves_the_table_and_no_temporary(tmp_path, monkeypatch):
    """Injection: the cleanup removed -> a `.tmp` file is left beside the table, RED."""
    p = tmp_path / "16g.jsonl"
    T.replace_model(p, "A", [_row("A")])
    before = p.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        T.write(p, [_row("A"), _row("B")])
    monkeypatch.undo()
    assert p.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


# --- the walk's inputs -----------------------------------------------------------------------

def _cache(tmp_path, *models):
    for m in models:
        (tmp_path / "cache" / m).mkdir(parents=True)
        (tmp_path / "cache" / m / "manifest.json").write_text(json.dumps({"family": "llm"}))
    return tmp_path / "cache"


def _main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(sys, "argv", ["certified_census.py", "--hardware", PROFILE, "--table",
                                      "nvidia/volta", "--logs", str(tmp_path / "logs"), *args])
    return CC.main()


@pytest.mark.parametrize("args,said", [
    (["--models", ""], "an empty name"),
    (["--models", "A,,B"], "an empty name"),
    (["--models", "A,Nowhere"], "not in the cache"),
    (["--models", "A", "--modes", ""], "no mode"),
    (["--models", "A", "--modes", " , "], "no mode"),
])
def test_an_input_that_names_nothing_is_refused_by_name(tmp_path, monkeypatch, capsys, args, said):
    """Injection: the input checks removed -> an empty --models censuses the whole cache, a
    missing model ends in a traceback, an empty --modes writes no row and exits 0 — RED."""
    monkeypatch.setattr(CC, "CACHE", _cache(tmp_path, "A", "B"))
    monkeypatch.setattr(CC, "census_model", lambda *a, **k: pytest.fail("a shadow ran"))
    assert _main(monkeypatch, tmp_path, *args) == 2
    assert said in capsys.readouterr().err


def test_an_empty_cache_is_refused(tmp_path, monkeypatch, capsys):
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(CC, "CACHE", tmp_path / "cache")
    assert _main(monkeypatch, tmp_path) == 2
    assert "holds no container" in capsys.readouterr().err


def test_a_key_record_cut_mid_line_yields_only_whole_keys(tmp_path):
    """Injection: `record_lines` reading every line -> the fragment is counted as a key, RED."""
    rec = tmp_path / "k.keys"
    rec.write_text("q::(1, 2)\nq::(3, 4)\nq::(5,")
    assert CC.record_lines(rec) == ["q::(1, 2)", "q::(3, 4)"]
    rec.write_text("q::(1, 2)\n")
    assert CC.record_lines(rec) == ["q::(1, 2)"]
    assert CC.record_lines(tmp_path / "absent") == []


# --- the walk's exits ------------------------------------------------------------------------

def _no_request(monkeypatch):
    monkeypatch.setattr(CC, "census_requests", lambda *a, **k: [["--prompt", "x"]])
    monkeypatch.setattr(CC, "_family", lambda m: "llm")


def test_a_census_that_formed_no_key_keeps_the_model_s_rows(tmp_path, monkeypatch):
    """Injection: the no_keys branch removed -> replace_model's door raises out of the census
    (or, with the door gone too, the rows are wiped), RED."""
    monkeypatch.setattr(CC, "CACHE", _cache(tmp_path, "A"))
    _no_request(monkeypatch)
    table = tmp_path / "16g.jsonl"
    T.replace_model(table, "A", [_row("A")])
    before = table.read_bytes()
    monkeypatch.setattr(T, "table_path", lambda *a, **k: table)
    monkeypatch.setattr(CC, "census_model", lambda m, *a, **k: {
        "family": "llm", "status": "ok", "keys": 0, "modes": {}, "requests": [], "frozen": [],
        "_table": [], "derived_breaks": [], "graph_sha": "c0ffee", "_keys": []})
    out = tmp_path / "census.json"
    assert _main(monkeypatch, tmp_path, "--models", "A", "--rungs", "none", "--out", str(out)) == 1
    assert table.read_bytes() == before
    rep = json.loads(out.read_text())
    assert rep["models"]["A"]["status"] == "no_keys" and "A" in rep["failed"]


def test_an_interrupted_census_leaves_a_report_that_says_so(tmp_path, monkeypatch):
    """Injection: the except branch removed -> the earlier run's report survives unchanged, read
    as this run's, RED."""
    monkeypatch.setattr(CC, "CACHE", _cache(tmp_path, "A"))
    _no_request(monkeypatch)
    monkeypatch.setattr(T, "table_path", lambda *a, **k: tmp_path / "16g.jsonl")
    out = tmp_path / "census.json"
    out.write_text(json.dumps({"models": {"A": {"status": "ok"}}, "stale": True}))

    def dies(*a, **k):
        raise RuntimeError("the shadow's host went away")
    monkeypatch.setattr(CC, "census_model", dies)
    with pytest.raises(RuntimeError):
        _main(monkeypatch, tmp_path, "--models", "A", "--rungs", "none", "--out", str(out))
    rep = json.loads(out.read_text())
    assert "stale" not in rep and "the shadow's host went away" in rep["interrupted"]
    assert not list(tmp_path.glob("census.json.*.tmp"))


# --- the derivation's table ------------------------------------------------------------------

def _table_args(tmp_path, **kw):
    a = dict(hardware=PROFILE, rungs="4096", modes="triton", models="A", table="nvidia/volta",
             logs=str(tmp_path / "logs"))
    a.update(kw)
    return SimpleNamespace(**a)


@pytest.mark.parametrize("kw,said", [
    ({"modes": ""}, "--modes"), ({"rungs": ","}, "--rungs"), ({"models": ""}, "an empty name"),
    ({"models": "Nowhere"}, "not in the cache"),
])
def test_the_derivation_refuses_an_input_that_names_nothing(tmp_path, monkeypatch, kw, said):
    """Injection: the derivation's input checks removed -> an empty --modes wipes model A, RED."""
    import derived_census as D
    monkeypatch.setattr(D, "CACHE", _cache(tmp_path, "A"))
    with pytest.raises(SystemExit, match=said):
        D.table(_table_args(tmp_path, **kw))


def test_a_derivation_that_formed_no_key_writes_nothing(tmp_path, monkeypatch):
    """Injection: the no-row refusal removed -> replace_model's door raises out of the table
    (or, with the door gone too, the rows are wiped), RED."""
    import derived_census as D
    monkeypatch.setattr(D, "CACHE", _cache(tmp_path, "A"))
    table = tmp_path / "16g.jsonl"
    T.replace_model(table, "A", [_row("A")])
    before = table.read_bytes()
    monkeypatch.setattr(T, "table_path", lambda *a, **k: table)
    monkeypatch.setattr(CC, "_family", lambda m: "llm")
    monkeypatch.setattr(CC, "census_requests", lambda *a, **k: [["--prompt", "x"]])
    monkeypatch.setattr(CC, "_tiling_probe", lambda *a, **k: None)
    monkeypatch.setattr(CC, "_graph_sha", lambda m: "c0ffee")
    import collections
    monkeypatch.setattr(D, "derive_keys", lambda *a, **k: (set(), {}, None, collections.Counter()))
    assert D.table(_table_args(tmp_path)) == 1
    assert table.read_bytes() == before
