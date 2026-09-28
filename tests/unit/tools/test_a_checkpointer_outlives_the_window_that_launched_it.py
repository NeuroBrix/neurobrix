"""A checkpointer's final checkpoint survives the hangup of the terminal that launched it.

Measured 2026-09-28: the 32 GB certification chain ran in a tmux window with its checkpointer as a
background child. The certifier ended at 12:10:05; the chain slept 20 s and exited; the window
closed and SIGHUP'ed its process group — the checkpointer died in its final gate, before the
commit. The certifier's last four entries sat uncommitted for six and a half hours, and the
12:07 checkpoint, held locally by the 30-minute push ration, was never pushed. SIGTERM was
handled (it runs the final checkpoint); SIGHUP was not.

What this test would do if the code were wrong: with SIGHUP at its default disposition the
checkpointer (and its gate) die at the hangup, nothing is committed, and the assertion on the
remotes' head fails (seen red, 2026-09-28, with the SIGHUP line removed).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / "tools"

SLOW_GATE = r'''
import pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text("gate started")
time.sleep(2)
for p in sorted(pathlib.Path(sys.argv[-1]).rglob("*.json")):
    print(f"ok      {p} (1 shape(s))")
print("0 refused")
'''


def _sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


def test_the_final_checkpoint_survives_the_window_closing(tmp_path):
    r = tmp_path / "repo"; r.mkdir()
    _sh("git", "init", "-q", "-b", "certify-branch", cwd=r)
    _sh("git", "config", "user.email", "t@t", cwd=r); _sh("git", "config", "user.name", "t", cwd=r)
    d = r / "src/neurobrix/config/autotune/v/p"; d.mkdir(parents=True)
    (d / "k.fp32.json").write_text(json.dumps({"entries": {}}))
    _sh("git", "add", ".", cwd=r); _sh("git", "commit", "-q", "-m", "base", cwd=r)
    for name in ("origin", "gitlab"):
        bare = tmp_path / f"{name}.git"; _sh("git", "init", "-q", "--bare", "-b", "certify-branch", str(bare))
        _sh("git", "remote", "add", name, str(bare), cwd=r); _sh("git", "push", "-q", name, "certify-branch", cwd=r)
    started = tmp_path / "gate.started"
    gate = tmp_path / "gate.py"; gate.write_text(SLOW_GATE)
    producer = subprocess.Popen(["sleep", "0.5"])
    (d / "k.fp32.json").write_text(json.dumps({"entries": {"(1,)": {"config": 1}}}))   # the certifier's last write
    # the window: a session of its own, as a tmux pane's process group
    cp = subprocess.Popen([sys.executable, str(TOOLS / "certified_checkpoint.py"), "--repo", str(r),
                           "--producer-pid", str(producer.pid), "--interval", "3600", "--poll", "0.1",
                           "--push-interval", "0", "--gate-cmd", json.dumps([sys.executable, str(gate), str(started)])],
                          start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    producer.wait()
    deadline = time.time() + 30
    while not started.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert started.exists(), "the final checkpoint never reached its gate"
    os.killpg(cp.pid, signal.SIGHUP)                     # the window closes mid-gate
    out, _ = cp.communicate(timeout=60)
    head = _sh("git", "rev-parse", "HEAD", cwd=r).strip()
    for name in ("origin", "gitlab"):
        assert _sh("git", "--git-dir", str(tmp_path / f"{name}.git"), "rev-parse", "certify-branch").strip() == head, out
    assert "(1,)" in _sh("git", "show", "HEAD:src/neurobrix/config/autotune/v/p/k.fp32.json", cwd=r)
