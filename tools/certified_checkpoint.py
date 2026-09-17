#!/usr/bin/env python3
"""The certified directory is committed and pushed WHILE it is being written.

    tools/certified_checkpoint.py --repo /home/mlops/NeuroBrix_System \\
        --dir src/neurobrix/config/autotune --producer-pid 1234 --producer-pid 1235 \\
        --interval 600 --record /home/mlops/nbx/campaigns/<date>_<name>/RUN.md

Three mains cuts in three days (2026-09-11, 09-12, 09-13) each found hundreds
of certified entries on disk and nowhere else — 971 on the Friday, 1 222 on
the Saturday night — because the certifier wrote its files entry by entry and
nothing carried them to a remote before the pass ended. They survived by the
file system's journal; a truncated write at the wrong moment would have cost a
file, and a dead disk the whole pass. A cut must cost minutes, not a pass.

What this brick does, and why it is its own process and not a flag of the
certifier:

* ONE checkpointer per repository. Two certifiers write two disjoint sets of
  files at once (one card per kernel), and two `git commit`s racing on one
  index lock would each fail the other; a single committer serialises them.
  The engine's certifier stays what it is — a measurement that writes JSON —
  and never learns what git is.
* It holds its producers (register entry 55): every `--producer-pid` is
  watched with the start-time check of `tools/wait_for.py`; when the last one
  is gone the final checkpoint runs and the process exits, so a chain can wait
  on IT.
* Every checkpoint runs the directory's gate first (`neurobrix autotune
  check --dir`) behind the no-card door (`CUDA_VISIBLE_DEVICES=` in the gate's
  environment: it cannot open a context on a real card whatever the stack
  does). A file the gate refuses is NAMED and left uncommitted; the others are
  committed. Nothing is ever restamped or rewritten here — the certifier is
  the only writer of those files.
* The commit carries the counts measured AFTER the add (entries added and
  changed per file against HEAD), never a number written before the
  measurement. Only the directory's files are committed (`git commit --
  <paths>`); anything else in the tree stays where it is.
* Each remote is pushed and then READ BACK (`git ls-remote`) — a push that
  printed nothing is not a push that landed. A remote that refuses is said,
  and retried at the next tick; the commit exists locally either way.

Exit: 0 when the producers are gone and the last checkpoint reached every
remote; 3 when they are gone and a remote still lacks it (said, with the
remote's name). `--once` runs a single checkpoint and exits with the same
codes. Without a producer and without `--once` it runs until SIGTERM, which
triggers a last checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wait_for import producer_alive, producer_name  # noqa: E402  — the one liveness brick

DEFAULT_GATE = [sys.executable, "-c", "import sys; from neurobrix.cli import main; sys.exit(main())",
                "autotune", "check", "--dir"]


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=check)


def changed_files(repo: str, rel_dir: str) -> List[str]:
    """`.json` paths under `rel_dir` that differ from HEAD (modified, added, untracked).
    Only `.json`: the certifier writes each file through `<name>.json.tmp` + `os.replace`,
    and a status read inside that window would otherwise hand the tmp file to `git add`."""
    out = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", rel_dir).stdout
    files = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not path.endswith(".json"):
            continue        # the certifier's `.json.tmp` (atomic replace in flight) is never a file of the directory
        files.append(path)
    return sorted(set(files))


def run_gate(repo: str, abs_dir: str, gate_cmd: List[str], env_extra: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """The directory's gate, behind the no-card door. Returns {"refused": [abs paths], "ok": [abs paths]}."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""            # the door: no context can exist on a real card
    env.setdefault("PYTHONPATH", str(Path(repo) / "src"))
    env.update(env_extra or {})
    r = subprocess.run([*gate_cmd, abs_dir], capture_output=True, text=True, env=env, cwd=repo)
    refused, ok = [], []
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("REFUSED "):
            refused.append(line[len("REFUSED "):].split(":", 1)[0].strip())
        elif line.startswith("ok "):
            ok.append(line[3:].strip().split(" (", 1)[0])
    if r.returncode not in (0, 1):               # 1 = "some refused" (said file by file); anything else = the gate itself broke
        raise RuntimeError(f"the gate failed to run (rc={r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return {"refused": refused, "ok": ok}


def entry_delta(repo: str, rel_path: str) -> Dict[str, int]:
    """Entries added / changed / removed in a certified file against HEAD (0/0/0 for a non-JSON file)."""
    try:
        disk = json.loads((Path(repo) / rel_path).read_text(encoding="utf-8")).get("entries") or {}
    except (OSError, ValueError, AttributeError):
        return {"added": 0, "changed": 0, "removed": 0}
    shown = _git(repo, "show", f"HEAD:{rel_path}", check=False)
    try:
        head = json.loads(shown.stdout).get("entries") or {} if shown.returncode == 0 else {}
    except (ValueError, AttributeError):
        head = {}
    return {"added": sum(1 for k in disk if k not in head),
            "changed": sum(1 for k in disk if k in head and disk[k] != head[k]),
            "removed": sum(1 for k in head if k not in disk)}


def current_branch(repo: str) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def push_and_verify(repo: str, remote: str, branch: str) -> str:
    """Push, then read the remote back. Returns "" when the remote holds HEAD, else the reason."""
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    r = _git(repo, "push", remote, branch, check=False)
    if r.returncode != 0:
        return f"push refused ({(r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else 'no output'})"
    ls = _git(repo, "ls-remote", remote, f"refs/heads/{branch}", check=False)
    if ls.returncode != 0 or head not in ls.stdout:
        return f"pushed but the remote reads {ls.stdout.split()[0][:12] if ls.stdout.split() else 'nothing'}, not {head[:12]}"
    return ""


def checkpoint(repo: str, rel_dir: str, remotes: List[str], gate_cmd: List[str], trailers: List[str],
               record: Optional[str] = None, say=print, label: str = "") -> Dict[str, object]:
    """One checkpoint: gate → commit the files that pass → push every remote → read back → record."""
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    files = changed_files(repo, rel_dir)
    result: Dict[str, object] = {"committed": [], "refused": [], "sha": None, "remotes": {}, "files": files}
    if not files:
        _record(record, f"== checkpoint {stamp}: nothing to commit under {rel_dir}", say)
        return result
    abs_dir = str(Path(repo) / rel_dir)
    gate = run_gate(repo, abs_dir, gate_cmd)
    refused_rel = sorted({os.path.relpath(p, repo) for p in gate["refused"]})
    to_commit = [f for f in files if f not in refused_rel]
    result["refused"] = refused_rel
    if not to_commit:
        _record(record, f"== checkpoint {stamp}: {len(files)} changed file(s), ALL refused by the gate, nothing committed: "
                        + ", ".join(refused_rel), say)
        return result
    _git(repo, "add", "--", *to_commit)
    deltas = {f: entry_delta(repo, f) for f in to_commit}           # measured AFTER the add, before the message
    added = sum(d["added"] for d in deltas.values()); changed = sum(d["changed"] for d in deltas.values())
    lines = [f"certified: checkpoint{(' ' + label) if label else ''} — {added} entries added, {changed} changed, "
             f"{len(to_commit)} file(s), written while the certifier runs", ""]
    for f, d in deltas.items():
        lines.append(f"- {Path(f).name}: +{d['added']} ~{d['changed']} -{d['removed']}")
    if refused_rel:
        lines += ["", "Refused by the gate at this checkpoint and left uncommitted: " + ", ".join(refused_rel)]
    lines += ["", "Committed by tools/certified_checkpoint.py: a cut must cost minutes, not a pass."]
    if trailers:
        lines += ["", *trailers]
    rr = subprocess.run(["git", "-C", repo, "commit", "-q", "-F", "-", "--", *to_commit], input="\n".join(lines),
                        capture_output=True, text=True)
    if rr.returncode != 0:
        # the commit itself failed — another writer's index lock, a hook — said, retried at the next tick
        _git(repo, "reset", "-q", "--", *to_commit, check=False)
        _record(record, f"== checkpoint {stamp}: commit FAILED — {(rr.stderr or rr.stdout).strip()[-300:]}", say)
        return result
    sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    result["sha"] = sha; result["committed"] = to_commit
    branch = current_branch(repo)
    status = []
    for remote in remotes:
        why = push_and_verify(repo, remote, branch)
        result["remotes"][remote] = why
        status.append(f"{remote} {'ok' if not why else 'FAILED (' + why + ')'}")
    _record(record, f"== checkpoint {sha} {stamp}: {len(to_commit)} file(s), +{added} entries, ~{changed} changed; "
                    + "; ".join(status)
                    + (f"; REFUSED by the gate, not committed: {', '.join(refused_rel)}" if refused_rel else ""), say)
    return result


def _record(record: Optional[str], line: str, say=print) -> None:
    say(line)
    if record:
        with open(record, "a") as f:
            f.write(line + "\n")


def run(repo: str, rel_dir: str, producers: List[int], interval: float, remotes: List[str], gate_cmd: List[str],
        trailers: List[str], record: Optional[str], once: bool = False, poll: float = 10.0, say=print,
        label: str = "") -> int:
    for pid in producers:
        if not producer_alive(pid):
            say(f"[checkpoint] producer {pid} ({producer_name(pid)}) is already gone at start — one last checkpoint, then exit")
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    last = time.monotonic()
    while True:
        alive = [p for p in producers if producer_alive(p)]
        final = once or stop["now"] or (bool(producers) and not alive)
        due = time.monotonic() - last >= interval
        if due or final:
            res = checkpoint(repo, rel_dir, remotes, gate_cmd, trailers, record, say=say, label=label)
            last = time.monotonic()
            if final:
                failed = [r for r, why in (res.get("remotes") or {}).items() if why]
                if failed:
                    say(f"[checkpoint] final checkpoint did not reach: {', '.join(failed)} — rc 3")
                    return 3
                return 0
        time.sleep(poll)


def refuse_a_producer_that_will_wait_for_us(producer_pids, parent_pid) -> str:
    """The message refusing a producer that is our own parent shell, or "".

    The contract in this file's header is that the checkpointer holds its
    PRODUCERS and exits when the last is gone, "so a chain can wait on IT".
    Naming the chain itself as the producer inverts that into a mutual wait:
    the chain waits for this process, this process waits for the chain, and
    neither moves again.

    That is not hypothetical. `reproof_t38_v2.sh` passed `--producer-pid $$`
    and then ran `wait` over its own job table. All four certifiers finished
    (card 3 at 02:53, card 1 at 02:57, card 2 at 04:13 — 4172 and 5942 shapes
    proven) and the chain never wrote `== reproof t38 done`. It sat there for
    the eight hours after its work was complete, and a waiter on that marker
    would have starved behind a chain that had already succeeded — register 55
    exactly, from the other side.

    The producers are the processes doing the WORK — the certifiers — not the
    shell that schedules them. When those are launched later and their pids are
    not known up front, the shell should `wait` for them itself and let this
    process go when it exits, passing `--allow-parent-as-producer` to say that
    is what it means.
    """
    if parent_pid in set(producer_pids):
        return ("REFUSED: --producer-pid %d is the shell that launched this checkpointer.\n"
                "  It holds that shell, so if that shell also waits on this process neither ever\n"
                "  moves (reproof_t38_v2.sh, 2026-09-17: all work done by 04:13, marker never\n"
                "  written, eight hours stuck).\n"
                "  Pass the pids of the processes doing the WORK instead, or\n"
                "  --allow-parent-as-producer if that shell will not wait on this one."
                % parent_pid)
    return ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Commit and push the certified directory while it is written.")
    p.add_argument("--repo", required=True)
    p.add_argument("--dir", default="src/neurobrix/config/autotune", help="relative to --repo")
    p.add_argument("--producer-pid", type=int, action="append", default=[],
                   help="a certifier (or its chain) to hold; the last one gone triggers the final checkpoint")
    p.add_argument("--interval", type=float, default=600.0, help="seconds between checkpoints")
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--remotes", default="origin,gitlab")
    p.add_argument("--record", default=None, help="a campaign RUN.md every checkpoint appends one line to")
    p.add_argument("--trailer", action="append", default=[], help="a line appended to each commit message")
    p.add_argument("--gate-cmd", default=None,
                   help="JSON list; the gate command that receives the directory as its last argument "
                        "(default: the engine's `autotune check --dir`)")
    p.add_argument("--label", default="", help="a short name for the pass, in each commit's subject")
    p.add_argument("--once", action="store_true")
    p.add_argument("--allow-parent-as-producer", action="store_true",
                   help="permit --producer-pid to name the shell that launched this "
                        "checkpointer; only correct if that shell will NOT wait on it")
    a = p.parse_args(argv)
    deadlock = refuse_a_producer_that_will_wait_for_us(a.producer_pid, os.getppid())
    if deadlock and not a.allow_parent_as_producer:
        print(deadlock, file=sys.stderr)
        return 2
    gate = json.loads(a.gate_cmd) if a.gate_cmd else DEFAULT_GATE
    return run(a.repo, a.dir, a.producer_pid, a.interval, [r for r in a.remotes.split(",") if r], gate,
               a.trailer, a.record, once=a.once, poll=a.poll, label=a.label)


if __name__ == "__main__":
    sys.exit(main())
