"""The checkpointer commits a file ONLY on the gate's positive verdict for that file.

Found by the stale-tests census (2026-11-24): `tools/certified_checkpoint.py` read the gate's
exit code 1 as "some refused", but a Python crash exits 1 too. A gate that crashed (a
`ModuleNotFoundError` under a broken interpreter) named nothing refused, so EVERY changed
file was committed unvalidated: under a broken interpreter, the checkpointer cell of
tests/unit/kernels/test_a_certification_keeps_one_proof_per_generator.py committed its
deliberately bad file.

The rule now: every changed file carries an explicit `ok` (committed) or an explicit
`REFUSED` (kept out). A file without a verdict, a file with both, a Python crash header
anywhere in the gate's output, an exit code other than 0 / 1, a 1 that names no refusal, or a 0 that names
one: NOTHING is committed and the checkpoint says so by name ("the gate did not judge
<files>: nothing committed"); the final checkpoint exits 4.

Every repository here is a temporary one under `tmp_path`; the real repository is never
touched. Each test was run RED on the tool as it stood at c90e752553af before the cure.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import certified_checkpoint as CP  # noqa: E402

DIR = "src/neurobrix/config/autotune"

#: A gate whose interpreter cannot import the engine: the exact fault the census reproduced.
CRASHING_GATE = r'''
import neurobrix_that_is_not_installed  # noqa: F401  -> ModuleNotFoundError, a crash report, exit 1
'''

#: A gate that judges the first file and then crashes: a partial output, then a crash report.
CRASHES_HALFWAY = r'''
import sys, pathlib
files = sorted(pathlib.Path(sys.argv[-1]).rglob("*.json"))
print(f"ok      {files[0]} (1 shape(s))", flush=True)
raise RuntimeError("the gate broke after its first file")
'''

#: A gate that exits 0 having judged only the first file: the others carry no verdict.
JUDGES_ONE = r'''
import sys, pathlib
files = sorted(pathlib.Path(sys.argv[-1]).rglob("*.json"))
print(f"ok      {files[0]} (1 shape(s))")
sys.exit(0)
'''

#: A gate that exits 1 and says nothing: a failure that is not a refusal.
SILENT_ONE = r'''
import sys
sys.exit(1)
'''

#: A gate that exits 0 yet names a refusal: a status that contradicts its own lines.
CONTRADICTS_ITSELF = r'''
import sys, pathlib
for p in sorted(pathlib.Path(sys.argv[-1]).rglob("*.json")):
    print(f"REFUSED {p}: injected" if p.name.startswith("bad") else f"ok      {p} (1 shape(s))")
sys.exit(0)
'''

#: The honest gate: ok for every file, REFUSED for bad*.json, exit 1 when any is refused.
MIXED_GATE = r'''
import sys, pathlib
bad = 0
for p in sorted(pathlib.Path(sys.argv[-1]).rglob("*.json")):
    if p.name.startswith("bad"):
        print(f"REFUSED {p}: injected"); bad += 1
    else:
        print(f"ok      {p} (1 shape(s))")
print(f"{bad} refused"); sys.exit(1 if bad else 0)
'''


def _sh(*cmd, cwd=None):
    """Run a command in a temporary repository and return its stdout (raises on failure)."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    """A temporary repository on a working branch with one committed certified file."""
    r = tmp_path / "repo"; r.mkdir()
    _sh("git", "init", "-q", "-b", "certify-branch", cwd=r)
    _sh("git", "config", "user.email", "t@t", cwd=r); _sh("git", "config", "user.name", "t", cwd=r)
    d = r / DIR / "v" / "p"; d.mkdir(parents=True)
    (d / "k.fp32.json").write_text(json.dumps({"entries": {"(1,)": {"config": 1}}}))
    _sh("git", "add", ".", cwd=r); _sh("git", "commit", "-q", "-m", "base", cwd=r)
    return {"path": str(r), "dir": d, "tmp": tmp_path}


def _gate(repo, source: str):
    """Write a stand-in gate script into the test's tmp dir and return its command line."""
    g = repo["tmp"] / "gate.py"; g.write_text(source)
    return [sys.executable, str(g)]


def _write(d: Path, name: str, n: int) -> None:
    """Write a certified-looking file of `n` entries (the stand-in gates never read its content)."""
    (d / name).write_text(json.dumps({"entries": {f"({i},)": {"config": i} for i in range(n)}}))


def _head(repo) -> str:
    """The temporary repository's HEAD sha."""
    return _sh("git", "-C", repo["path"], "rev-parse", "HEAD").strip()


def _two_changed_files(repo):
    """One good and one deliberately bad changed file; returns their repo-relative paths."""
    _write(repo["dir"], "k.fp32.json", 3)
    _write(repo["dir"], "bad.fp16.json", 5)
    return [f"{DIR}/v/p/bad.fp16.json", f"{DIR}/v/p/k.fp32.json"]


def _assert_nothing_committed(repo, res, before, files, said):
    """Nothing reached HEAD, the result names the fault, and the record line names every file."""
    assert _head(repo) == before, "the gate did not judge: nothing may be committed"
    assert res["committed"] == [] and res["sha"] is None, res
    why = res.get("gate_failed") or ""
    assert why.startswith("the gate did not judge ") and "nothing committed" in why, res
    for f in files:
        assert f in why, (f, why)
    assert any("the gate did not judge" in line for line in said), said


@pytest.mark.parametrize("source", [CRASHING_GATE, CRASHES_HALFWAY, SILENT_ONE, CONTRADICTS_ITSELF],
                         ids=["crash-on-import", "crash-halfway", "exit-1-names-nothing", "exit-0-names-a-refusal"])
def test_a_gate_that_did_not_judge_commits_nothing_and_says_so_by_name(repo, source):
    """A crashing, silent or self-contradicting gate leaves HEAD where it was and names every file."""
    files = _two_changed_files(repo)
    before = _head(repo); said = []
    res = CP.checkpoint(repo["path"], DIR, [], _gate(repo, source), [], say=said.append)
    _assert_nothing_committed(repo, res, before, files, said)


def test_a_gate_exiting_with_an_undocumented_code_commits_nothing(repo):
    """An exit code other than 0 or 1 is the gate broken, said by name, never a raise mid-tick."""
    files = _two_changed_files(repo)
    before = _head(repo); said = []
    res = CP.checkpoint(repo["path"], DIR, [], _gate(repo, "import sys; sys.exit(2)\n"), [], say=said.append)
    _assert_nothing_committed(repo, res, before, files, said)


def test_a_gate_that_judges_only_some_files_commits_nothing(repo):
    """The file the gate never named is the one the commit would carry unvalidated: nothing goes in."""
    _two_changed_files(repo)
    before = _head(repo); said = []
    res = CP.checkpoint(repo["path"], DIR, [], _gate(repo, JUDGES_ONE), [], say=said.append)
    # the gate judged bad.fp16.json (first in order) and never k.fp32.json
    _assert_nothing_committed(repo, res, before, [f"{DIR}/v/p/k.fp32.json"], said)


def test_a_mixed_run_still_commits_exactly_the_files_judged_ok(repo):
    """The honest gate's ok files are committed, its REFUSED file is named and kept out."""
    _two_changed_files(repo)
    before = _head(repo)
    res = CP.checkpoint(repo["path"], DIR, [], _gate(repo, MIXED_GATE), [], say=lambda *a: None)
    assert not res.get("gate_failed"), res
    assert res["committed"] == [f"{DIR}/v/p/k.fp32.json"], res
    assert res["refused"] == [f"{DIR}/v/p/bad.fp16.json"], res
    assert _head(repo) != before
    in_head = _sh("git", "-C", repo["path"], "ls-tree", "-r", "--name-only", "HEAD")
    assert "k.fp32.json" in in_head and "bad.fp16.json" not in in_head


def test_the_final_checkpoint_of_a_gate_that_did_not_judge_exits_4_by_name(repo):
    """`--once` over a crashing gate returns 4 and says which files the gate did not judge."""
    files = _two_changed_files(repo)
    before = _head(repo); said = []
    rc = CP.run(repo["path"], DIR, [], 600.0, [], _gate(repo, CRASHING_GATE), [], None, once=True, say=said.append)
    assert rc == 4, (rc, said)
    assert _head(repo) == before
    assert any("the gate did not judge" in s and all(f in s for f in files) for s in said), said
