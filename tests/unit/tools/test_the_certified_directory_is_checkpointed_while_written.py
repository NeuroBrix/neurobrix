"""A cut must cost minutes, not a pass — the certified directory is committed
and pushed while the certifier writes it (`tools/certified_checkpoint.py`).

Three mains cuts in three days (2026-09-11/12/13) each left hundreds of
certified entries on disk and nowhere else (971, then 1 222). The brick under
test commits the files that pass the directory's gate, pushes every remote and
reads each back, holds its producers, and never touches a card: the gate runs
with `CUDA_VISIBLE_DEVICES=` in its environment.

The gate here is a stand-in script (`--gate-cmd`) so the test is hermetic and
fast; one test wires the ENGINE's real gate on a real entry copied from the
directory and on that entry with its deviation pushed above the tolerance.

Injection record (2026-09-14 00:03-00:05 UTC): with the gate's refusals
neutralised (`refused_rel = []`),
`test_a_file_the_gate_refuses_is_named_and_left_uncommitted` went RED (the
refused file was committed); restored, green. With the `CUDA_VISIBLE_DEVICES`
line removed, `test_the_gate_runs_behind_the_no_card_door` stayed GREEN on
its first form — the stub refused a fictitious `door.json`, which the brick
rightly ignored, so the test proved nothing about the door. The stub now
refuses EVERY file when a card is visible; with that form the injection went
RED, restored, green. The vacuous first form is recorded here because it is
the shape the register warns of: a green over a check that reached nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
    _sh("git", "init", "-q", "-b", "main", cwd=r)
    _sh("git", "config", "user.email", "t@t", cwd=r); _sh("git", "config", "user.name", "t", cwd=r)
    d = r / "src/neurobrix/config/autotune/v/p"; d.mkdir(parents=True)
    (d / "k.fp32.json").write_text(json.dumps({"entries": {"(1,)": {"config": 1}}}))
    (r / "other.txt").write_text("not ours")
    _sh("git", "add", ".", cwd=r); _sh("git", "commit", "-q", "-m", "base", cwd=r)
    for name in ("origin", "gitlab"):
        bare = tmp_path / f"{name}.git"; _sh("git", "init", "-q", "--bare", "-b", "main", str(bare))
        _sh("git", "remote", "add", name, str(bare), cwd=r)
        _sh("git", "push", "-q", name, "main", cwd=r)
    gate = tmp_path / "gate.py"; gate.write_text(GATE_STUB)
    return {"path": str(r), "dir": d, "gate": [sys.executable, str(gate)], "tmp": tmp_path}


def _remote_head(tmp, name):
    return _sh("git", "--git-dir", str(tmp / f"{name}.git"), "rev-parse", "main").strip()


def _head(repo):
    return _sh("git", "-C", repo, "rev-parse", "HEAD").strip()


def _write(d, name, n_entries):
    (d / name).write_text(json.dumps({"entries": {f"({i},)": {"config": i} for i in range(n_entries)}}))


def test_a_changed_file_is_committed_and_pushed_to_every_remote_and_read_back(repo):
    _write(repo["dir"], "k.fp32.json", 4)                       # +3 entries against HEAD
    (Path(repo["path"]) / "other.txt").write_text("someone else's edit")
    rec = repo["tmp"] / "RUN.md"
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", ["origin", "gitlab"], repo["gate"],
                        ["Trailer: x"], record=str(rec), say=lambda *a: None)
    assert res["committed"] == ["src/neurobrix/config/autotune/v/p/k.fp32.json"]
    head = _head(repo["path"])
    assert _remote_head(repo["tmp"], "origin") == head and _remote_head(repo["tmp"], "gitlab") == head
    msg = _sh("git", "-C", repo["path"], "log", "-1", "--format=%B")
    assert "3 entries added, 0 changed" in msg and "Trailer: x" in msg      # measured after the add
    assert "+3 ~0 -0" in msg
    line = rec.read_text()
    assert "origin ok; gitlab ok" in line and "+3 entries" in line
    # only the directory's files went in: the other edit is still in the tree
    st = _sh("git", "-C", repo["path"], "status", "--porcelain")
    assert "other.txt" in st and "k.fp32.json" not in st


def test_a_file_the_gate_refuses_is_named_and_left_uncommitted(repo):
    """The injection: a file named bad*.json is refused by the stub gate."""
    _write(repo["dir"], "k.fp32.json", 2)
    _write(repo["dir"], "bad.fp16.json", 5)
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", ["origin"], repo["gate"], [], say=lambda *a: None)
    assert res["refused"] == ["src/neurobrix/config/autotune/v/p/bad.fp16.json"]
    assert res["committed"] == ["src/neurobrix/config/autotune/v/p/k.fp32.json"]
    tracked = _sh("git", "-C", repo["path"], "ls-tree", "-r", "--name-only", "HEAD")
    assert "bad.fp16.json" not in tracked, "a refused file must never be committed"
    msg = _sh("git", "-C", repo["path"], "log", "-1", "--format=%B")
    assert "bad.fp16.json" in msg and "Refused by the gate" in msg


def test_the_gate_runs_behind_the_no_card_door(repo, monkeypatch):
    """The stub refuses everything unless CUDA_VISIBLE_DEVICES is the empty string."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    _write(repo["dir"], "k.fp32.json", 2)
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", ["origin"], repo["gate"], [], say=lambda *a: None)
    assert res["committed"], "behind the door the gate sees no card and passes the file"


def test_the_final_checkpoint_fires_when_the_last_producer_dies(repo):
    p = subprocess.Popen(["sleep", "60"]); p.kill(); p.wait()
    _write(repo["dir"], "k.fp32.json", 3)
    rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [p.pid], interval=10_000, remotes=["origin", "gitlab"],
                gate_cmd=repo["gate"], trailers=[], record=None, poll=0.01, say=lambda *a: None)
    assert rc == 0
    assert _remote_head(repo["tmp"], "gitlab") == _head(repo["path"])


def test_a_remote_that_refuses_is_said_and_the_run_exits_three(repo):
    import shutil
    shutil.rmtree(repo["tmp"] / "gitlab.git")
    _write(repo["dir"], "k.fp32.json", 3)
    said = []
    rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [], interval=1, remotes=["origin", "gitlab"],
                gate_cmd=repo["gate"], trailers=[], record=None, once=True, say=said.append)
    assert rc == 3
    assert any("gitlab FAILED" in s for s in said)
    assert _remote_head(repo["tmp"], "origin") == _head(repo["path"])   # the one that answered holds it


def test_a_tmp_file_of_a_write_in_flight_is_never_committed(repo):
    """The certifier writes `<name>.json.tmp` then os.replace: a status read inside
    that window sees an untracked tmp file. Injection: without the `.json` filter
    the tmp file was committed (seen RED 2026-09-14 00:27 UTC)."""
    _write(repo["dir"], "k.fp32.json", 3)
    (repo["dir"] / "k.fp32.json.tmp").write_text("{half-written")
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", ["origin"], repo["gate"], [], say=lambda *a: None)
    assert res["committed"] == ["src/neurobrix/config/autotune/v/p/k.fp32.json"]
    tracked = _sh("git", "-C", repo["path"], "ls-tree", "-r", "--name-only", "HEAD")
    assert ".tmp" not in tracked


def test_nothing_changed_commits_nothing(repo):
    before = _head(repo["path"])
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", ["origin"], repo["gate"], [], say=lambda *a: None)
    assert res["committed"] == [] and _head(repo["path"]) == before


ENGINE_DIR = Path(__file__).resolve().parents[3] / "src/neurobrix/config/autotune/nvidia/volta"


@pytest.mark.skipif(not (ENGINE_DIR / "depthwise_conv2d_kernel.fp32.json").exists(), reason="engine directory absent")
def test_the_engines_real_gate_is_wired_and_refuses_a_deviation_above_tolerance(repo):
    """The real gate, on a copy of a real file, and on that file with one
    deviation pushed above the tolerance — the refused copy stays out."""
    src = json.loads((ENGINE_DIR / "depthwise_conv2d_kernel.fp32.json").read_text())
    d = Path(repo["path"]) / "src/neurobrix/config/autotune/nvidia/volta"; d.mkdir(parents=True)
    (d / "depthwise_conv2d_kernel.fp32.json").write_text(json.dumps(src))
    bad = json.loads(json.dumps(src)); bad["dtype"] = "fp16"
    first = next(iter(bad["entries"].values())); first["proof"]["deviation"] = first["proof"]["tolerance"] * 10
    (d / "depthwise_conv2d_kernel.fp16.json").write_text(json.dumps(bad))
    res = CP.checkpoint(repo["path"], "src/neurobrix/config/autotune", ["origin"], CP.DEFAULT_GATE, [], say=lambda *a: None)
    names = [Path(f).name for f in res["committed"]]
    assert "depthwise_conv2d_kernel.fp32.json" in names
    assert "depthwise_conv2d_kernel.fp16.json" in [Path(f).name for f in res["refused"]]
    assert "depthwise_conv2d_kernel.fp16.json" not in names   # the fixture's stand-in k.fp32.json is refused too, rightly


def test_pushes_are_batched_to_one_per_window_and_the_final_one_does_not_wait(repo):
    """Commits every interval, pushes at most once per `push_interval`; a final checkpoint inside the window
    commits and returns at once, recorded as not pushed — the next run whose window is open carries it. Why: the
    owner's account was suspended twice for automated pushes (supervisor 05:41, 2026-09-28: one push per 30 min
    per repository); and a final checkpoint that slept out the window held a chain's GPU idle for the whole window
    (13 minutes on 2026-09-28 09:31). Seen red on the tool that pushed at every checkpoint (every commit pushed at
    once), and red again on the tool whose final checkpoint waited (this run took the whole window)."""
    import threading, time
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2.6)"])
    def writer():
        for k in range(2, 5):                               # a new certified entry every second
            time.sleep(1.0)
            _write(repo["dir"], "k.fp32.json", k)
    threading.Thread(target=writer, daemon=True).start()
    t0 = time.monotonic()
    rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [p.pid], interval=1.0, remotes=["origin", "gitlab"],
                gate_cmd=repo["gate"], trailers=[], record=None, poll=0.2, say=lambda *_: None, push_interval=30.0)
    elapsed = time.monotonic() - t0
    assert rc == 0
    commits = [l for l in _sh("git", "-C", repo["path"], "log", "--format=%h %s").splitlines() if "checkpoint" in l]
    assert len(commits) >= 2, commits
    pushed = _remote_head(repo["tmp"], "origin")
    assert pushed == _remote_head(repo["tmp"], "gitlab") and pushed in _sh("git", "-C", repo["path"], "log", "--format=%H"), \
        "the first checkpoint's push did not reach every remote"
    assert pushed != _head(repo["path"]), "a second push happened inside the 30 s window"
    assert elapsed < 8.0, f"the run took {elapsed:.1f} s: the final checkpoint waited for the window"


def test_the_push_window_is_the_repositorys_not_the_process_s(repo):
    """A push recorded for the repository by another process (a manual batched push, `--touch-push`) holds this
    run's pushes too: on 2026-09-28 a per-process window let the tool push 23 minutes after a manual push of the
    same repository. Seen red on the per-process window (the first checkpoint pushed at once)."""
    import time
    CP.touch_push(repo["path"])                                  # a push of this repository, just now, by someone else
    _write(repo["dir"], "k.fp32.json", 3)
    before = _remote_head(repo["tmp"], "origin")
    rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [], interval=1.0, remotes=["origin", "gitlab"],
                gate_cmd=repo["gate"], trailers=[], record=None, once=True, poll=0.2, say=lambda *_: None,
                push_interval=30.0)
    assert rc == 0
    assert _head(repo["path"]) != before, "the entry was not committed"
    assert _remote_head(repo["tmp"], "origin") == before, "the tool pushed inside the repository's window"
    # a later run whose window is open carries it
    CP.touch_push(repo["path"], when=time.time() - 60)
    rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [], interval=1.0, remotes=["origin", "gitlab"],
                gate_cmd=repo["gate"], trailers=[], record=None, once=True, poll=0.2, say=lambda *_: None,
                push_interval=30.0)
    assert rc == 0 and _remote_head(repo["tmp"], "origin") == _head(repo["path"]) != before


def test_a_final_checkpoint_with_nothing_to_push_does_not_wait_for_the_window(repo):
    """With nothing to commit and nothing unpushed, the final checkpoint returns at once even inside
    the push window. Seen red: it slept the whole window (13 minutes behind a certifier that certified
    nothing on 2026-09-28, the GPU idle)."""
    import time
    CP.touch_push(repo["path"])                                  # the window is closed for 30 s
    t0 = time.monotonic()
    rc = CP.run(repo["path"], "src/neurobrix/config/autotune", [], interval=1.0, remotes=["origin", "gitlab"],
                gate_cmd=repo["gate"], trailers=[], record=None, once=True, poll=0.2, say=lambda *_: None,
                push_interval=30.0)
    elapsed = time.monotonic() - t0
    assert rc == 0
    assert elapsed < 5.0, f"the final checkpoint waited {elapsed:.1f} s with nothing to push"
