"""A gate carries a cell's verdict instead of running it only with its provenance (the supervisor,
2026-10-11 18:55), under three rules of `tools/regression_matrix.py run`:

1. CARRIED VERDICTS (`--carry`): only from a record naming the source gate, the source row's date, its
   engine sha and a reason, the row READ from the source gate and found to be what the record says; the
   carried row is written as such and the table says CARRIED before its verdict. A cell neither run nor
   carried is "not measured". A carried row is never reused as a measurement by a later run.
2. KNOWN-CAUSE REDS (`--known-reds`, required for a gate; `--claims-fix`): a red with an identified cause is
   carried, red, with its cause, until a batch names it; a claim on an unrecorded cell is refused; a record
   whose source row is green is stale and refused.
3. PER-BRANCH GATE (`--base-gate`, `--base-sha`, tools/gate_branches.yml): a diff touching only one engine
   branch carries the other branch's cells from the base gate measured at the base sha; a shared path, or a
   branch file the other branch imports, re-runs both; a path the mapping cannot classify is refused.

Before this branch none of the flags exist and a gate either ran every cell or reused a row by its sha alone.
Every test here was SEEN red on an injection into the tool (clean -> injected -> restored, recorded in the
commit message). No cell is launched: `run_cell` is stubbed; no GPU, no model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

HERE = "fedcba9876"
BASE = "0123456789"


def _row(model, mode, rc=0, engine=BASE, date="2026-10-10T10:00:00Z", **kw):
    return {"model": model, "mode": mode, "family": "llm", "rc": rc, "wall_s": 5.0, "date": date,
            "engine": engine, "log": "x", **({"error": "RuntimeError: boom"} if rc else {}), **kw}


def _gate(tmp_path, name, rows):
    g = tmp_path / name
    g.mkdir(parents=True)
    (g / "rows_card0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return g


def _yaml(tmp_path, name, key, entries):
    p = tmp_path / name
    p.write_text(json.dumps({key: entries}))            # JSON is YAML
    return p


@pytest.fixture
def world(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    for m in ("A", "B"):
        (cache / m).mkdir(parents=True)
        (cache / m / "manifest.json").write_text("{}")
    monkeypatch.setattr(R, "CACHE", cache)
    monkeypatch.setattr(R, "engine_sha", lambda tree: HERE)
    monkeypatch.setattr(R, "reference_state", lambda src: {"tree_dirty": False, "tree_dirty_count": 0,
                                                          "tree_dirty_paths": []})
    ran = []

    def fake_run_cell(model, mode, gpu, out, timeout, src, wait=True, need=None):
        ran.append((model, mode))
        return _row(model, mode, engine=HERE, date="2026-10-11T18:00:00Z")
    monkeypatch.setattr(R, "run_cell", fake_run_cell)
    return NS(tmp=tmp_path, ran=ran)


def _args(w, **kw):
    base = dict(models="A,B", modes="native,triton", gpu="0", out=str(w.tmp / "out"), src=str(w.tmp / "src"),
                timeout=60, rerun=False, gate_lists=None, defer_after=None, known_reds=None, claims_fix=None,
                carry=None, base_gate=None, base_sha=None, branches=str(R.GATE_BRANCHES))
    base.update(kw)
    return NS(**base)


def _rows(w):
    return {(r["model"], r["mode"]): r for r in R.read_jsonl(w.tmp / "out" / "rows_card0.jsonl")}


# --- 1. carried verdicts with provenance ---------------------------------------------------------------------

def test_a_recorded_carry_is_written_as_carried_and_not_run(world):
    g = _gate(world.tmp, "g1", [_row("A", "native")])
    rec = _yaml(world.tmp, "c.yml", "carries", [{"model": "A", "mode": "native", "source_gate": str(g),
                                                  "date": "2026-10-10T10:00:00Z", "engine": BASE[:7],
                                                  "reason": "unchanged since g1"}])
    assert R.cmd_run(_args(world, carry=str(rec))) == 0
    assert ("A", "native") not in world.ran and len(world.ran) == 3
    c = _rows(world)[("A", "native")]["carried"]
    assert (c["kind"], c["source_gate"], c["engine"], c["date"], c["reason"]) == \
        ("record", str(g), BASE, "2026-10-10T10:00:00Z", "unchanged since g1")
    assert c["carried_by_engine"] == HERE
    rows = {(r["model"], r["mode"]): r for r in R.load_rows(world.tmp / "out")}
    assert R.table_cell(rows[("A", "native")]).startswith("CARRIED record from g1 @" + BASE)
    # idempotent: the same carry again writes no second row
    assert R.cmd_run(_args(world, carry=str(rec))) == 0
    assert sum(1 for r in R.read_jsonl(world.tmp / "out" / "rows_card0.jsonl") if r["model"] == "A"
               and r["mode"] == "native") == 1


@pytest.mark.parametrize("field,value,match", [
    ("engine", "aaaaaaa", "not the measurement the record names"),
    ("date", "2026-10-09T10:00:00Z", "not the measurement the record names"),
    ("reason", "", "names no reason"),
    ("source_gate", "NOWHERE", "holds no rows_card"),
])
def test_a_carry_the_source_does_not_bear_out_is_refused(world, field, value, match):
    g = _gate(world.tmp, "g1", [_row("A", "native")])
    e = {"model": "A", "mode": "native", "source_gate": str(g), "date": "2026-10-10T10:00:00Z", "engine": BASE,
         "reason": "r"}
    e[field] = str(world.tmp / value) if field == "source_gate" else value
    rec = _yaml(world.tmp, "c.yml", "carries", [e])
    with pytest.raises(SystemExit, match=match):
        R.cmd_run(_args(world, carry=str(rec)))
    assert world.ran == []


def test_a_cell_neither_run_nor_carried_is_not_measured(world):
    assert R.table_cell(None) == "not measured"


def test_a_carried_row_is_never_reused_as_a_measurement(world):
    out = world.tmp / "out"
    out.mkdir()
    carried = _row("A", "native", engine=HERE, tree_dirty=False)   # every other reason to rerun removed
    carried["carried"] = {"kind": "record", "source_gate": "g", "date": "d", "engine": HERE, "reason": "r",
                          "carried_by_engine": HERE}
    (out / "rows_card0.jsonl").write_text(json.dumps(carried) + "\n")
    assert R.cmd_run(_args(world, models="A", modes="native")) == 0
    assert world.ran == [("A", "native")]


# --- 2. known-cause reds ------------------------------------------------------------------------------------

def _red_record(world, g, mode="triton"):
    return _yaml(world.tmp, "reds.yml", "reds", [{"model": "B", "mode": mode, "source_gate": str(g),
                                                  "date": "2026-10-10T10:00:00Z", "engine": BASE,
                                                  "cause": "frozen pos table", "red_cells": "B | trace"}])


def test_a_known_red_is_carried_red_with_its_cause(world):
    g = _gate(world.tmp, "g1", [_row("B", "triton", rc=1)])
    assert R.cmd_run(_args(world, known_reds=str(_red_record(world, g)))) == 0
    assert ("B", "triton") not in world.ran
    r = _rows(world)[("B", "triton")]
    assert r["rc"] == 1 and r["carried"]["kind"] == "known-red" and r["carried"]["reason"] == "frozen pos table"
    assert r["carried"]["red_cells"] == "B | trace"


def test_a_claimed_red_runs(world):
    g = _gate(world.tmp, "g1", [_row("B", "triton", rc=1)])
    assert R.cmd_run(_args(world, known_reds=str(_red_record(world, g)), claims_fix="B/triton")) == 0
    assert ("B", "triton") in world.ran and "carried" not in _rows(world)[("B", "triton")]


def test_a_claim_on_an_unrecorded_cell_is_refused(world):
    g = _gate(world.tmp, "g1", [_row("B", "triton", rc=1)])
    with pytest.raises(SystemExit, match="no known-cause red"):
        R.cmd_run(_args(world, known_reds=str(_red_record(world, g)), claims_fix="B/native"))
    assert world.ran == []


def test_a_known_red_whose_source_is_green_is_refused(world):
    g = _gate(world.tmp, "g1", [_row("B", "triton", rc=0)])
    with pytest.raises(SystemExit, match="is not red"):
        R.cmd_run(_args(world, known_reds=str(_red_record(world, g))))


def test_a_gate_names_its_known_cause_record(world):
    lst = world.tmp / "l"
    lst.write_text("A native,triton\nB native,triton\n")
    assert R.cmd_run(_args(world, gate_lists=[str(lst)])) == 2
    assert world.ran == []


# --- 3. the per-branch gate ---------------------------------------------------------------------------------

def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


def _repo(tmp_path):
    """A tree with both engines: core/solo_c.py (compiled only), kernels/solo_k.py (triton only),
    kernels/shared_k.py (imported by core/uses_k.py, so shared), and docs."""
    repo = tmp_path / "tree"
    files = {
        "src/neurobrix/__init__.py": "",
        "src/neurobrix/core/__init__.py": "",
        "src/neurobrix/core/solo_c.py": "X = 1\n",
        "src/neurobrix/core/uses_k.py": "def f():\n    from neurobrix.kernels import shared_k\n",
        "src/neurobrix/kernels/__init__.py": "",
        "src/neurobrix/kernels/solo_k.py": "Y = 1\n",
        "src/neurobrix/kernels/shared_k.py": "Z = 1\n",
        "src/neurobrix/triton/__init__.py": "from ..kernels import solo_k\n",
        "docs/x.md": "a\n",
    }
    for p, t in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _change(repo, *paths):
    for p in paths:
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text((repo / p).read_text() + "# changed\n" if (repo / p).exists() else "new\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "batch")


def _base_sha(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def test_the_mapping_classifies_by_branch_and_by_import(tmp_path):
    repo = _repo(tmp_path)
    m = R.load_branches()
    c = R.classify_paths(repo, ["src/neurobrix/core/solo_c.py", "src/neurobrix/kernels/solo_k.py",
                                "src/neurobrix/kernels/shared_k.py", "docs/x.md", "tools/a.py",
                                "src/neurobrix/core/data.yml"], m)
    assert c == {"src/neurobrix/core/solo_c.py": "compiled", "src/neurobrix/kernels/solo_k.py": "triton",
                 "src/neurobrix/kernels/shared_k.py": "shared", "docs/x.md": "inert", "tools/a.py": "shared",
                 "src/neurobrix/core/data.yml": "shared"}
    with pytest.raises(SystemExit, match="no prefix of gate_branches.yml classifies it"):
        R.classify_paths(repo, ["weird/file.py"], m)


def _branch_world(world, *changed, base_engine=None):
    repo = _repo(world.tmp)
    base = _base_sha(repo)
    g = _gate(world.tmp, "base_gate", [_row(mo, md, engine=(base_engine or base)[:10]) for mo in "AB"
                                       for md in ("native", "triton")])
    _change(repo, *changed)
    return repo, base, g


def test_a_triton_only_batch_carries_the_native_cells(world):
    repo, base, g = _branch_world(world, "src/neurobrix/kernels/solo_k.py", "docs/x.md")
    a = _args(world, src=str(repo / "src"), base_gate=str(g), base_sha=base[:10])
    assert R.cmd_run(a) == 0
    assert sorted(world.ran) == [("A", "triton"), ("B", "triton")]
    c = _rows(world)[("A", "native")]["carried"]
    assert c["kind"] == "branch" and c["branch"] == "triton" and "touches only the triton branch" in c["reason"]
    assert c["diff_paths"] == ["docs/x.md", "src/neurobrix/kernels/solo_k.py"]


def test_a_compiled_only_batch_carries_the_triton_cells(world):
    repo, base, g = _branch_world(world, "src/neurobrix/core/solo_c.py")
    assert R.cmd_run(_args(world, src=str(repo / "src"), base_gate=str(g), base_sha=base[:10])) == 0
    assert sorted(world.ran) == [("A", "native"), ("B", "native")]


@pytest.mark.parametrize("changed", [("src/neurobrix/kernels/shared_k.py",), ("tools/x.py",),
                                     ("src/neurobrix/kernels/solo_k.py", "tools/x.py"),
                                     ("src/neurobrix/core/solo_c.py", "src/neurobrix/kernels/solo_k.py")])
def test_a_shared_or_two_branch_batch_reruns_both(world, changed):
    repo, base, g = _branch_world(world, *changed)
    assert R.cmd_run(_args(world, src=str(repo / "src"), base_gate=str(g), base_sha=base[:10])) == 0
    assert len(world.ran) == 4


def test_a_base_gate_at_another_sha_is_refused(world):
    repo, base, g = _branch_world(world, "src/neurobrix/kernels/solo_k.py", base_engine="abcdef0123")
    with pytest.raises(SystemExit, match="not the base"):
        R.cmd_run(_args(world, src=str(repo / "src"), base_gate=str(g), base_sha=base[:10]))
    assert world.ran == []


def test_a_dirty_tree_proves_no_reach(world):
    repo, base, g = _branch_world(world, "src/neurobrix/kernels/solo_k.py")
    (repo / "src/neurobrix/core/solo_c.py").write_text("X = 2\n")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        R.cmd_run(_args(world, src=str(repo / "src"), base_gate=str(g), base_sha=base[:10]))


def test_the_committed_records_load_and_classify_every_tracked_path():
    """The committed known-cause record reads, and the mapping classifies every path this repository tracks:
    a new top-level path fails here, not in the middle of a gate."""
    repo = Path(R.__file__).resolve().parents[1]
    R._read_records(repo / "docs" / "reference" / "known-cause-reds.yml", "reds", R.KNOWN_RED_FIELDS)
    tracked = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True).stdout.split()
    assert tracked
    R.classify_paths(repo, [p for p in tracked if not p.startswith("src/")], R.load_branches())
