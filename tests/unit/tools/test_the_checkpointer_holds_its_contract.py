"""The checkpointer's contract (docs/reference/tool-contracts.md, "The checkpointer").

The tools audit of 2026-09-29 found: the gate read the disk and the commit took the working
tree as it stood LATER, so a certifier write in between was committed ungated; "never push main"
was checked once, in `main()`, at start — `checkpoint()` and `run()` pushed whatever branch the
tree was on; "one checkpointer per repository" was written in a skill and enforced nowhere (two
collided on the index lock); a refused push was retried at every tick, three attempts per
30-minute window; a `--dir` that does not exist was accepted and committed nothing forever; and
the only test of the main refusal lived in a file calling functions the tool no longer has.

What would this file do if the code were wrong? Each test names its injection; every one was run
and seen RED before this file was committed.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import certified_checkpoint as CP  # noqa: E402

GATE_STUB = r'''
import os, sys, pathlib
d = pathlib.Path(sys.argv[-1])
if os.environ.get("CUDA_VISIBLE_DEVICES", "unset") != "":
    for p in sorted(d.rglob("*.json")):
        print(f"REFUSED {p}: the gate ran with a card visible")
    sys.exit(1)
bad = 0
for p in sorted(d.rglob("*.json")):
    if p.name.startswith("bad"):
        print(f"REFUSED {p}: injected"); bad += 1
    else:
        print(f"ok      {p} (1 shape(s))")
print(f"{bad} refused"); sys.exit(1 if bad else 0)
'''


def _sh(*cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    """A repo with one committed certified file and two bare remotes named as on the rack."""
    r = tmp_path / "repo"; r.mkdir()
    _sh("git", "init", "-q", "-b", "certify-branch", cwd=r)   # a working branch: `main` is never pushed by the brick
    _sh("git", "config", "user.email", "t@t", cwd=r); _sh("git", "config", "user.name", "t", cwd=r)
    d = r / "src/neurobrix/config/autotune/v/p"; d.mkdir(parents=True)
    (d / "k.fp32.json").write_text(json.dumps({"entries": {"(1,)": {"config": 1}}}))
    (r / "other.txt").write_text("not ours")
    _sh("git", "add", ".", cwd=r); _sh("git", "commit", "-q", "-m", "base", cwd=r)
    for name in ("origin", "gitlab"):
        bare = tmp_path / f"{name}.git"; _sh("git", "init", "-q", "--bare", "-b", "certify-branch", str(bare))
        _sh("git", "remote", "add", name, str(bare), cwd=r)
        _sh("git", "push", "-q", name, "certify-branch", cwd=r)
    gate = tmp_path / "gate.py"; gate.write_text(GATE_STUB)
    return {"path": str(r), "dir": d, "gate": [sys.executable, str(gate)], "tmp": tmp_path}




def _head_blob(repo, rel):
    return subprocess.run(["git", "-C", repo, "show", f"HEAD:{rel}"], capture_output=True, text=True,
                          check=True).stdout


def test_the_bytes_committed_are_the_bytes_the_gate_passed(repo, tmp_path, monkeypatch):
    """The gate stand-in rewrites the working file WHILE it runs (a certifier's write landing
    between the gate and the commit). Injection: commit the working tree (`git add` after the
    gate) -> the ungated rewrite is committed, RED."""
    f = repo["dir"] / "k.fp32.json"
    gated = json.dumps({"entries": {"(1,)": {"config": 1}, "(2,)": {"config": 2}}})
    f.write_text(gated)
    later = tmp_path / "later.py"
    later.write_text(f"""
import pathlib, runpy, sys
pathlib.Path({str(f)!r}).write_text('{{"entries": {{"ungated": 1}}}}')
runpy.run_path({repo['gate'][1]!r}, run_name="__main__")
""")
    rel = os.path.relpath(f, repo["path"])
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", [], [sys.executable, str(later)], [])
    assert res["sha"], res
    assert _head_blob(repo["path"], rel) == gated
    status = subprocess.run(["git", "-C", repo["path"], "status", "--porcelain", "--", rel],
                            capture_output=True, text=True).stdout
    assert status.startswith(" M"), f"the later write stays a change for the next checkpoint: {status!r}"


def test_main_is_never_pushed_even_when_the_tree_is_on_main_mid_run(repo):
    """Injection: the refusal left in `main()` only -> `checkpoint()` pushes main, RED."""
    r = repo["path"]
    subprocess.run(["git", "-C", r, "checkout", "-q", "-b", "main"], check=True)
    (repo["dir"] / "k.fp32.json").write_text(json.dumps({"entries": {"(3,)": {"config": 3}}}))
    res = CP.checkpoint(r, "src/neurobrix/config/autotune", ["origin"], repo["gate"], [])
    assert res["sha"] and "REFUSED" in res["remotes"]["origin"]
    heads = subprocess.run(["git", "--git-dir", str(repo["tmp"] / "origin.git"), "branch", "--list", "main"],
                           capture_output=True, text=True).stdout
    assert not heads.strip(), "main reached the remote"


def _hold(repo_path, ready, release):
    assert CP.hold_the_repository(repo_path) is not None
    ready.set()
    release.wait(30)


def test_a_second_checkpointer_on_one_repository_is_refused(repo):
    """Injection: `hold_the_repository` always granting -> the second run proceeds, RED."""
    ctx = mp.get_context("fork")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold, args=(repo["path"], ready, release))
    holder.start()
    assert ready.wait(20)
    try:
        said = []
        rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [], 600, [], repo["gate"], [], None,
                    once=True, say=said.append)
        assert rc == 2 and "another checkpointer holds" in said[0]
    finally:
        release.set()
        holder.join(20)


def test_a_refused_push_spends_the_window(repo):
    """A remote that refuses. Injection: the window stamped only after a push that LANDED -> the
    stamp stays unset and the next tick pushes again, RED."""
    r = repo["path"]
    subprocess.run(["git", "-C", r, "remote", "set-url", "origin", str(repo["tmp"] / "absent.git")], check=True)
    (repo["dir"] / "k.fp32.json").write_text(json.dumps({"entries": {"(4,)": {"config": 4}}}))
    assert CP.last_push_time(r) == 0.0
    CP._HELD.clear()
    rc = CP.run(r, "src/neurobrix/config/autotune", [], 600, ["origin"], repo["gate"], [], None,
                once=True, say=lambda *_: None)
    assert rc == 3
    assert time.time() - CP.last_push_time(r) < 60, "a refused attempt did not spend the window"


def test_a_directory_that_does_not_exist_is_refused(repo, capsys):
    """Injection: the --dir check removed -> the checkpointer runs over nothing, RED."""
    rc = CP.main(["--repo", repo["path"], "--dir", "src/neurobrix/config/autotun", "--once", "--remotes", "",
                  "--gate-cmd", json.dumps(repo["gate"])])
    assert rc == 2 and "no such directory" in capsys.readouterr().err
