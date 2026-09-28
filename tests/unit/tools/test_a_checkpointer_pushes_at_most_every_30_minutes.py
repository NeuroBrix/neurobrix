"""An unattended process pushes at most once every 30 minutes per repository, never `main`
(the owner's GitHub account was suspended twice for automated pushes; the supervisor, 2026-09-28
05:41 and 06:16). `tools/certified_checkpoint.py` keeps committing at `--interval` and rations its
PUSHES through a record in the clone's common git directory, shared by every checkpointer of that
clone; the final checkpoint waits for its turn instead of skipping its push.

What each test would do if the code were wrong:
* no ration -> the second checkpoint pushes: `test_a_second_checkpoint_inside_the_ration_commits_but_does_not_push` RED;
* a per-process ration (not shared) -> `test_the_ration_is_shared_by_every_checkpointer_of_the_clone` RED;
* a final checkpoint that skips a deferred push -> `test_the_final_checkpoint_waits_for_its_turn_and_pushes` RED
  (the remote would lack HEAD); one that pushes early -> RED on the recorded sleep;
* a tick with nothing new that forgets the deferred commits -> `test_a_quiet_tick_carries_the_commits_a_deferred_push_left` RED;
* `main` pushed -> `test_main_is_never_pushed_by_an_unattended_process` RED.

Injection record (2026-09-28 06:3x CEST), each seen RED then restored green: the ration removed -> 4 red;
the record made per call instead of per clone -> 4 red; the final checkpoint not waiting -> 1 red;
`main` allowed -> 1 red.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import certified_checkpoint as CP  # noqa: E402

GATE_OK = "import sys, pathlib\n[print(f'ok      {p} (1 shape(s))') for p in sorted(pathlib.Path(sys.argv[-1]).rglob('*.json'))]\n"
REL = "src/neurobrix/config/autotune"


def _sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"; r.mkdir()
    _sh("git", "init", "-q", "-b", "certify-branch", cwd=r)
    _sh("git", "config", "user.email", "t@t", cwd=r); _sh("git", "config", "user.name", "t", cwd=r)
    d = r / REL / "v/p"; d.mkdir(parents=True)
    (d / "k.fp32.json").write_text(json.dumps({"entries": {"(1,)": {"config": 1}}}))
    _sh("git", "add", ".", cwd=r); _sh("git", "commit", "-q", "-m", "base", cwd=r)
    bare = tmp_path / "origin.git"; _sh("git", "init", "-q", "--bare", str(bare))
    _sh("git", "remote", "add", "origin", str(bare), cwd=r); _sh("git", "push", "-q", "origin", "certify-branch", cwd=r)
    gate = tmp_path / "gate.py"; gate.write_text(GATE_OK)
    return {"path": str(r), "dir": d, "gate": [sys.executable, str(gate)], "bare": bare}


def _write(repo, n):
    (repo["dir"] / "k.fp32.json").write_text(json.dumps({"entries": {f"({i},)": {"config": i} for i in range(n)}}))


def _remote(repo, branch="certify-branch"):
    return _sh("git", "--git-dir", str(repo["bare"]), "rev-parse", branch).strip()


def _head(repo):
    return _sh("git", "rev-parse", "HEAD", cwd=repo["path"]).strip()


def _cp(repo, lines, **kw):
    return CP.checkpoint(repo["path"], REL, ["origin"], repo["gate"], [], say=lines.append, **kw)


def test_a_second_checkpoint_inside_the_ration_commits_but_does_not_push(repo):
    lines = []
    _write(repo, 2); first = _cp(repo, lines, push_every=1800)
    assert first["remotes"] == {"origin": ""} and _remote(repo) == _head(repo)
    pushed_sha = _head(repo)
    _write(repo, 3); second = _cp(repo, lines, push_every=1800)
    assert second["sha"] and _head(repo) != pushed_sha          # committed locally
    assert _remote(repo) == pushed_sha                           # not pushed
    assert "push not due" in lines[-1] and "one per 30 min per repository" in lines[-1]


def test_the_ration_is_shared_by_every_checkpointer_of_the_clone(repo):
    # another checkpointer of this clone (another process, another worktree) pushed a moment ago
    CP._stamp_path(repo["path"]).write_text(f"{time.time():.3f} certify-branch\n")
    before = _remote(repo)
    lines = []
    _write(repo, 4); res = _cp(repo, lines, push_every=1800)
    assert res["sha"] and _remote(repo) == before and "push not due" in lines[-1]


def test_the_final_checkpoint_waits_for_its_turn_and_pushes(repo, monkeypatch):
    CP._stamp_path(repo["path"]).write_text(f"{time.time():.3f} certify-branch\n")
    slept = []
    real_push = CP.push_when_due

    def push_with_a_recorded_sleep(*a, **kw):
        def sleep(s):
            slept.append(s)
            # the other checkpointer's turn has passed: the stamp ages by the ration
            p = CP._stamp_path(repo["path"]); t = float(p.read_text().split()[0]); p.write_text(f"{t - 1800:.3f} x\n")
        return real_push(*a, **{**kw, "sleep": sleep})
    monkeypatch.setattr(CP, "push_when_due", push_with_a_recorded_sleep)
    lines = []
    _write(repo, 5); res = _cp(repo, lines, push_every=1800, final=True)
    assert slept and slept[0] > 1700                             # waited for the ration, did not push early
    assert res["remotes"] == {"origin": ""} and _remote(repo) == _head(repo)


def test_a_quiet_tick_carries_the_commits_a_deferred_push_left(repo):
    lines = []
    _write(repo, 6); _cp(repo, lines, push_every=1800)          # pushed
    _write(repo, 7); _cp(repo, lines, push_every=1800)          # committed, deferred
    assert _remote(repo) != _head(repo)
    p = CP._stamp_path(repo["path"]); p.write_text(f"{time.time() - 1801:.3f} x\n")   # the ration has passed
    res = _cp(repo, lines, push_every=1800)                     # nothing new to commit
    assert res["sha"] is None and res["remotes"] == {"origin": ""} and _remote(repo) == _head(repo)


def test_main_is_never_pushed_by_an_unattended_process(repo):
    _sh("git", "checkout", "-q", "-b", "main", cwd=repo["path"])
    lines = []
    _write(repo, 8); res = _cp(repo, lines, push_every=0)
    assert res["remotes"]["origin"].startswith("REFUSED")
    assert "main" not in _sh("git", "--git-dir", str(repo["bare"]), "branch", "--list", "main")


def test_the_command_line_rations_by_default():
    import argparse  # noqa: F401 — the default is read from the parser the command line builds
    assert CP.PUSH_EVERY_S == 1800.0
    src = (TOOLS / "certified_checkpoint.py").read_text()
    assert 'default=PUSH_EVERY_S' in src
