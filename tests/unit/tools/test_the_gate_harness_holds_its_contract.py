"""The gate harness (`tools/regression_matrix.py`) holds its written contract.

The audit of 2026-09-29 read the harness against what a gate needs and found it did not:

1. It counted no miss. A row carried `rc` and `first_error(log)`, and a `KeyNotCertified` line
   matched none of the error patterns, so a confirmation run's missing key and its census row were
   never recorded — the one thing a certified-only cell exists to report.
2. It accepted an empty request. `--models ""` gave an empty to-do list and exit 0; an empty cache
   gave an empty matrix that empty gate lists "covered"; nothing checked at the end that every cell
   asked for had a row.
3. It reused any earlier output. Without `--rerun`, a cell with ANY earlier row was skipped whatever
   engine that row had measured.
4. It did not say what it compared against. Cells import `--src` and inherit the environment, so
   NEUROBRIX_AUTOTUNE_CERTIFIED_DIR passed through; the certified directory read was the tree's working
   copy; a row recorded HEAD but not whether that working copy differed from it.
5. Its files could become unreadable. Rows were appended with a bare open("a"); a kill mid-append
   left a cut last line and every reader then crashed on json.loads for the whole file; table.md
   and the export were rewritten in place.

Every test here names the injection it was SEEN red on (clean -> injected -> red -> restored ->
green). No cell is launched: `Z.run` or `run_cell` is stubbed; no GPU, no model.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import regression_matrix as R  # noqa: E402

KERNEL = "kernels.ops.mm._mm_kernel"
REFUSED_MISS = (f"CERTIFIED-ONLY: no certified setting for _mm_kernel fp16 (M=64 N=128 K=256 fp16,fp16,fp16) "
                f"on nvidia/v100-sxm2. ABSENT from the census table census/nvidia/v100-sxm2.16.jsonl — a CENSUS "
                f"defect: fix the census tool so the shadow forms it, certify it, re-run the cell. A "
                f"confirmation run never sweeps.")
LOOKUP_MISS = f"CERTIFIED-ONLY: the certified lookup failed for {KERNEL} at (64, 128, 'torch.float8'): 'float8'"


def _log(message, inner=None):
    """A certified-only cell's log as the CLI leaves it: its own ERROR line (no class name), then the
    traceback — whose frames print the source comment `# raises KeyNotCertified: never a sweep` —
    and the exception line; a chained inner error first when the miss is raised `from` one."""
    chained = (f"Traceback (most recent call last):\n  File \"autotune_certified.py\", line 900, in apply\n"
               f"{inner}\n\nThe above exception was the direct cause of the following exception:\n\n") if inner else ""
    return (f"$ python -m neurobrix run --certified-only\n[progress] load\n\nUNEXPECTED ERROR: {message}\n"
            f"{chained}Traceback (most recent call last):\n"
            f"  File \"src/neurobrix/kernels/ops/_configs.py\", line 192, in wrapper\n"
            f"    _cert.refuse_missing(qual, tuned, key)   # raises KeyNotCertified: never a sweep\n"
            f"neurobrix.kernels.autotune_certified.KeyNotCertified: {message}\n")


def _stub_cell(monkeypatch, log_text, rc=1):
    def fake_run(cmd, env, log, timeout, **kw):
        Path(log).write_text(log_text)
        return rc, 0.1
    monkeypatch.setattr(R.Z, "run", fake_run)
    monkeypatch.setattr(R.Z, "family_of", lambda model: "llm")
    monkeypatch.setattr(R.Z, "output_ext", lambda family, req: ".txt")
    monkeypatch.setattr(R, "cell_request", lambda model: ["--prompt", "x"])
    monkeypatch.setattr(R, "off_trace_size", lambda model, family: None)


def _cache(tmp_path, names):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    for n in names:
        (cache / n).mkdir()
        (cache / n / "manifest.json").write_text("{}")
    return cache


def _args(tmp_path, models="a", modes="native", **kw):
    return NS(models=models, modes=modes, gpu="0", out=str(tmp_path / "o"), src=str(tmp_path / "src"),
              timeout=60, rerun=False, gate_lists=None, **kw)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _matrix(tmp_path, monkeypatch, names=("a",), engine="abc1234", here_dirty=False):
    """A cache, this tree at `engine` with its reference clean (or not), and a stub `run_cell` that
    records which cells ran and returns their row."""
    monkeypatch.setattr(R, "CACHE", _cache(tmp_path, names))
    monkeypatch.delenv(R.CERTIFIED_DIR_ENV, raising=False)
    monkeypatch.setattr(R, "engine_sha", lambda tree: engine)
    monkeypatch.setattr(R, "reference_state", lambda src: {"tree_dirty": here_dirty, "tree_dirty_count": 0,
                                                            "tree_dirty_paths": []})
    ran = []

    def fake_run_cell(model, mode, gpu, out, timeout, src, wait=True):
        ran.append((model, mode))
        return {"model": model, "mode": mode, "gpu": gpu, "rc": 0, "date": _now(), "engine": engine,
                "tree_dirty": here_dirty}
    monkeypatch.setattr(R, "run_cell", fake_run_cell)
    return ran


# ------------------------------------------------------------------------------------------------
# 1. a miss is counted and its key named
# ------------------------------------------------------------------------------------------------
def test_a_miss_is_counted_once_and_its_key_named(tmp_path, monkeypatch):
    """Injection: KEY_NOT_CERTIFIED = r"(KeyNotCertified[^\\n]*)" (the bare class name) — the source
    comment of _configs.py in the traceback is counted as a second miss: red on `misses == 1`."""
    _stub_cell(monkeypatch, _log(REFUSED_MISS))
    row = R._run_cell("M", "triton", "0", tmp_path, 60, tmp_path / "src")
    assert row["misses"] == 1, row
    assert row["first_missing_key"] == "_mm_kernel fp16 (M=64 N=128 K=256 fp16,fp16,fp16)", row
    assert row["error"].startswith("KeyNotCertified: CERTIFIED-ONLY: no certified setting for _mm_kernel"), row


def test_a_lookup_failed_miss_names_itself_over_its_inner_error(tmp_path, monkeypatch):
    """Injection: the KeyNotCertified pattern removed from `first_error` — the chained inner
    `KeyError: 'float8'` names the cell: red on the error assertion."""
    _stub_cell(monkeypatch, _log(LOOKUP_MISS, inner="KeyError: 'float8'"))
    row = R._run_cell("M", "triton", "0", tmp_path, 60, tmp_path / "src")
    assert row["misses"] == 1, row
    assert row["first_missing_key"] == f"{KERNEL} at (64, 128, 'torch.float8')", row
    assert row["error"].startswith("KeyNotCertified: CERTIFIED-ONLY: the certified lookup failed"), row


def test_a_cell_without_a_miss_records_zero(tmp_path, monkeypatch):
    """Injection: the `misses` / `first_missing_key` fields dropped from the row — red on KeyError.
    (Every row carries them, so a table can tell 'no miss' from 'not counted'.)"""
    _stub_cell(monkeypatch, "[progress] done\n", rc=0)
    row = R._run_cell("M", "native", "0", tmp_path, 60, tmp_path / "src")
    assert row["misses"] == 0 and row["first_missing_key"] is None, row


# ------------------------------------------------------------------------------------------------
# 2. an empty or absent request is refused by name, and every asked cell has a row at the end
# ------------------------------------------------------------------------------------------------
def test_an_empty_model_list_is_refused_before_any_cell(tmp_path, monkeypatch, capsys):
    """Injection: the old filter `models = [m.strip() for m in a.models.split(",") if m.strip()]` —
    the empty list becomes an empty run that exits 0: red on `rc == 2`."""
    ran = _matrix(tmp_path, monkeypatch)
    for models in ("", " ", "a,,a"):
        assert R.cmd_run(_args(tmp_path, models=models)) == 2, models
        assert "names no model" in capsys.readouterr().err
    assert ran == [] and not (tmp_path / "o").exists()


def test_a_model_absent_from_the_cache_is_refused_by_name(tmp_path, monkeypatch, capsys):
    """Injection: `absent = []` — the cell of a model the cache does not hold is launched: red on
    `rc == 2` and on `ran == []`."""
    ran = _matrix(tmp_path, monkeypatch)
    assert R.cmd_run(_args(tmp_path, models="a,ghost")) == 2
    assert "ghost" in capsys.readouterr().err
    assert ran == [] and not (tmp_path / "o").exists()


def test_an_empty_or_unknown_mode_list_is_refused_by_name(tmp_path, monkeypatch, capsys):
    """Injection: the old filter `modes = [m.strip() for m in a.modes.split(",") if m.strip()]` —
    `--modes ""` becomes an empty run that exits 0: red on `rc == 2`."""
    ran = _matrix(tmp_path, monkeypatch)
    assert R.cmd_run(_args(tmp_path, modes="")) == 2
    assert "names no mode" in capsys.readouterr().err
    assert R.cmd_run(_args(tmp_path, modes="native,compiled")) == 2
    assert "compiled" in capsys.readouterr().err
    assert ran == [] and not (tmp_path / "o").exists()


def test_a_gate_over_an_empty_cache_is_refused(tmp_path):
    """Injection: the `if not full:` refusal removed — empty lists 'cover' the empty matrix and the
    gate starts: red on `pytest.raises`."""
    cache = _cache(tmp_path, [])
    (tmp_path / "l").write_text("")
    with pytest.raises(SystemExit, match="holds no container"):
        R.refuse_a_partial_gate([tmp_path / "l"], cache)


def test_a_cell_asked_for_with_no_row_is_an_error_naming_it(tmp_path, monkeypatch, capsys):
    """Injection: `lost = []` (no end check) — a run whose triton cell left no row exits 0: red on
    `rc != 0`. The stub writes every cell's row under mode 'native', so only a check that reads the
    rows back from the files sees the hole."""
    _matrix(tmp_path, monkeypatch)

    def mislabelled(model, mode, gpu, out, timeout, src, wait=True):
        return {"model": model, "mode": "native", "gpu": gpu, "rc": 0, "date": _now(), "engine": "abc1234",
                "tree_dirty": False}
    monkeypatch.setattr(R, "run_cell", mislabelled)
    assert R.cmd_run(_args(tmp_path, modes="native,triton")) != 0
    assert "a/triton" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------
# 3. an earlier row is reused only when it measured this engine against a clean reference
# ------------------------------------------------------------------------------------------------
def _earlier(tmp_path, rows):
    out = tmp_path / "o"
    out.mkdir(exist_ok=True)
    (out / "rows_card1.jsonl").write_text("".join(json.dumps(
        {"mode": "native", "gpu": "1", "rc": 0, "date": "2026-09-28T10:00:00Z", **r}) + "\n" for r in rows))


def test_an_earlier_row_of_another_engine_or_a_dirty_tree_runs_again(tmp_path, monkeypatch, capsys):
    """Injection: `done.add(cell); continue` before the engine/tree checks (the old rule: any earlier
    row is done) — the cells of another engine, a dirty tree and an unrecorded tree are skipped: red
    on `ran`."""
    ran = _matrix(tmp_path, monkeypatch, names=("same", "other", "dirty", "unrecorded"))
    _earlier(tmp_path, [
        {"model": "same", "engine": "abc1234", "tree_dirty": False},
        {"model": "other", "engine": "0ld5ha1", "tree_dirty": False},
        {"model": "dirty", "engine": "abc1234", "tree_dirty": True},
        {"model": "unrecorded", "engine": "abc1234"},
    ])
    assert R.cmd_run(_args(tmp_path, models="same,other,dirty,unrecorded")) == 0
    assert ran == [("other", "native"), ("dirty", "native"), ("unrecorded", "native")], ran
    said = capsys.readouterr().out
    assert "0ld5ha1 is not this run's abc1234" in said and "was dirty" in said and "unrecorded" in said
    rows = R.read_jsonl(tmp_path / "o" / "rows_card0.jsonl")
    assert rows[0]["supersedes"]["engine"] == "0ld5ha1", "the re-run names the row it supersedes"


def test_no_earlier_row_is_reused_when_this_tree_is_dirty(tmp_path, monkeypatch, capsys):
    """Injection: `done.add(cell); continue` before the checks — the row of this very engine is
    reused although this tree's certified reference differs from its HEAD: red on `ran`."""
    ran = _matrix(tmp_path, monkeypatch, here_dirty=True)
    _earlier(tmp_path, [{"model": "a", "engine": "abc1234", "tree_dirty": False}])
    assert R.cmd_run(_args(tmp_path)) == 0
    assert ran == [("a", "native")]
    assert "this tree's certified reference is dirty" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------
# 4. the row says what the cell compared against; an override is refused unless allowed
# ------------------------------------------------------------------------------------------------
def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def _repo(tmp_path):
    repo = tmp_path / "tree"
    for rel in ("src/neurobrix/config/autotune/nvidia/v100/mm.fp16.json", "src/neurobrix/config/census/t.jsonl",
                "src/neurobrix/core/x.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("{}\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "t")
    return repo


def test_only_the_certified_reference_makes_a_tree_dirty(tmp_path):
    """Injection: the pathspec limit dropped from `reference_state`'s `git status` — an edit to
    core/x.py makes the tree 'dirty': red on the first `tree_dirty is False`."""
    repo = _repo(tmp_path)
    src = repo / "src"
    (src / "neurobrix/core/x.py").write_text("changed\n")
    assert R.reference_state(src)["tree_dirty"] is False
    (src / "neurobrix/config/autotune/nvidia/v100/mm.fp16.json").write_text('{"x": 1}\n')
    (src / "neurobrix/config/census/new.jsonl").write_text("{}\n")
    st = R.reference_state(src)
    assert st["tree_dirty"] is True and st["tree_dirty_count"] == 2, st
    assert set(st["tree_dirty_paths"]) == {"src/neurobrix/config/autotune/nvidia/v100/mm.fp16.json",
                                           "src/neurobrix/config/census/new.jsonl"}, st
    assert R.reference_state(tmp_path / "not-a-tree" / "src")["tree_dirty"] is None


def test_the_row_records_the_reference_it_compared_against(tmp_path, monkeypatch):
    """Injection: `**reference_state(src)` and `certified_dir_override` removed from the row —
    red on KeyError."""
    repo = _repo(tmp_path)
    (repo / "src/neurobrix/config/autotune/nvidia/v100/mm.fp16.json").write_text('{"x": 1}\n')
    monkeypatch.setenv(R.CERTIFIED_DIR_ENV, str(tmp_path / "draft"))
    _stub_cell(monkeypatch, "[progress] done\n", rc=0)
    row = R._run_cell("M", "native", "0", tmp_path / "o", 60, repo / "src")
    assert row["tree_dirty"] is True and row["tree_dirty_paths"], row
    assert row["certified_dir_override"] == str(tmp_path / "draft"), row
    assert row["engine"], row


def test_a_run_under_a_certified_dir_override_is_refused_unless_allowed(tmp_path, monkeypatch, capsys):
    """Injection: the override refusal removed (`if False and override ...`) — the cells run against
    the relocated directory: red on `rc == 2` and on `ran == []`."""
    ran = _matrix(tmp_path, monkeypatch)
    monkeypatch.setenv(R.CERTIFIED_DIR_ENV, str(tmp_path / "draft"))
    assert R.cmd_run(_args(tmp_path)) == 2
    assert R.CERTIFIED_DIR_ENV in capsys.readouterr().err
    assert ran == [] and not (tmp_path / "o").exists()
    assert R.cmd_run(_args(tmp_path, allow_certified_dir_override=True)) == 0
    assert ran == [("a", "native")]


# ------------------------------------------------------------------------------------------------
# 5. the matrix's files stay readable
# ------------------------------------------------------------------------------------------------
_ROW = {"model": "a", "mode": "native", "gpu": "1", "rc": 0, "date": "2026-09-28T10:00:00Z", "wall_s": 1.0,
        "log": "x"}


def test_a_cut_last_line_is_skipped_and_said(tmp_path, capsys):
    """Injection: `tail = ""` in `read_jsonl` (the cut line read as a row) — the reader refuses the
    whole file: red on the SystemExit."""
    (tmp_path / "rows_card1.jsonl").write_text(json.dumps(_ROW) + "\n" + '{"model": "b", "mo')
    rows = R.load_rows(tmp_path)
    assert [r["model"] for r in rows] == ["a"]
    assert "no newline" in capsys.readouterr().err


def test_a_malformed_line_that_is_not_last_is_refused_by_name(tmp_path):
    """Injection: `except ValueError: continue` in `read_jsonl` (a bad row silently dropped) — the
    file is read without it: red on `pytest.raises`."""
    f = tmp_path / "rows_card1.jsonl"
    f.write_text(json.dumps(_ROW) + "\n" + "{not json\n" + json.dumps({**_ROW, "model": "b"}) + "\n")
    with pytest.raises(SystemExit, match=r"rows_card1\.jsonl:2 is not a row"):
        R.load_rows(tmp_path)


def test_an_append_after_a_cut_line_keeps_the_file_readable(tmp_path, monkeypatch):
    """Injection: the cut-tail repair in `append_jsonl` disabled (`if False:`) — the new row glues
    onto the fragment, a malformed line no longer last: red on the refusal."""
    _matrix(tmp_path, monkeypatch, names=("a", "b"))
    out = tmp_path / "o"
    out.mkdir()
    (out / "rows_card0.jsonl").write_text(json.dumps({**_ROW, "engine": "abc1234", "tree_dirty": False})
                                          + "\n" + '{"model": "b", "mo')
    assert R.cmd_run(_args(tmp_path, models="a,b")) == 0
    assert [r["model"] for r in R.read_jsonl(out / "rows_card0.jsonl")] == ["a", "b"]
    assert (out / "rows_card0.jsonl.torn").read_text() == '{"model": "b", "mo\n'


def test_every_appended_row_is_fsynced(tmp_path, monkeypatch):
    """Injection: the `os.fsync` after the row's write removed from `append_jsonl` — red on the count."""
    _matrix(tmp_path, monkeypatch, names=("a", "b"))
    synced = []
    real = R.os.fsync
    monkeypatch.setattr(R.os, "fsync", lambda fd: (synced.append(fd), real(fd)))
    assert R.cmd_run(_args(tmp_path, models="a,b")) == 0
    assert len(synced) == 2, synced


def test_a_kill_before_the_table_is_committed_leaves_the_old_one_whole(tmp_path, monkeypatch):
    """Injection: `write_atomic` writes the destination in place (`path.write_text(text)`) — the
    table is already rewritten when the kill lands: red on the byte comparison. The kill is injected
    at `os.replace`, the last instant before the atomic write's commit point."""
    (tmp_path / "rows_card1.jsonl").write_text(json.dumps({**_ROW, "family": "llm"}) + "\n")
    old = "| the previous table, whole |\n"
    (tmp_path / "table.md").write_text(old)

    def killed(*a, **k):
        raise KeyboardInterrupt("killed before the rename")
    monkeypatch.setattr(R.os, "replace", killed)
    try:
        R.cmd_table(NS(out=str(tmp_path), proofs=None))
    except KeyboardInterrupt:
        pass                                    # the kill; an in-place writer never reaches it
    assert (tmp_path / "table.md").read_text() == old, "the table was rewritten in place before the kill"
