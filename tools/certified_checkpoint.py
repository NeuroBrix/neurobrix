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


def entry_delta(repo: str, rel_path: str, content: Optional[bytes] = None) -> Dict[str, int]:
    """Entries added / changed / removed in a certified file against HEAD (0/0/0 for a non-JSON file).
    `content`: the bytes being committed (read once, gated, committed); the disk otherwise."""
    try:
        text = content.decode("utf-8") if content is not None else (Path(repo) / rel_path).read_text(encoding="utf-8")
        disk = json.loads(text).get("entries") or {}
    except (OSError, ValueError, AttributeError, UnicodeDecodeError):
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


def _push_stamp_path(repo: str) -> Path:
    """The repository's last-push stamp, in its common git dir so every worktree shares it."""
    common = subprocess.run(["git", "-C", repo, "rev-parse", "--git-common-dir"], capture_output=True, text=True)
    d = common.stdout.strip() if common.returncode == 0 and common.stdout.strip() else ".git"
    if not os.path.isabs(d):
        d = os.path.join(repo, d)
    return Path(d) / "nbx-last-push"


def last_push_time(repo: str) -> float:
    """Epoch seconds of the repository's last recorded push (0.0 when none was recorded)."""
    try:
        return float(_push_stamp_path(repo).read_text().strip())
    except (OSError, ValueError):
        return 0.0


def touch_push(repo: str, when: Optional[float] = None) -> None:
    """Record a push of this repository now (this tool's pushes and push ATTEMPTS, and manual batched
    ones via `--touch-push`). Written beside itself and replaced: a torn stamp read as 0 opened the window."""
    p = _push_stamp_path(repo)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(f"{when if when is not None else time.time():.0f}\n")
    os.replace(tmp, p)


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


def _git_dir(repo: str) -> Path:
    common = subprocess.run(["git", "-C", repo, "rev-parse", "--git-common-dir"], capture_output=True, text=True)
    d = common.stdout.strip() if common.returncode == 0 and common.stdout.strip() else ".git"
    return Path(d if os.path.isabs(d) else os.path.join(repo, d))


def _gate_snapshot(repo: str, rel_dir: str, snap: Dict[str, bytes], gate_cmd: List[str]) -> List[str]:
    """The gate run on a COPY of exactly the bytes about to be committed (read once); returns the
    refused files as repo-relative paths. The gate used to read the disk and the commit to take the
    working tree as it stood later: a certifier write in between was committed ungated (the tools
    audit, 2026-09-29)."""
    import shutil
    import tempfile
    td = tempfile.mkdtemp(prefix="nbx-ckpt-gate-", dir=str(_git_dir(repo)))
    try:
        root = Path(td) / "dir"
        for rel, data in snap.items():
            dst = root / os.path.relpath(rel, rel_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        gate = run_gate(repo, str(root), gate_cmd)
        return sorted({os.path.join(rel_dir, os.path.relpath(p, root)) for p in gate["refused"]})
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _commit_blobs(repo: str, blobs: Dict[str, bytes], message: str) -> subprocess.CompletedProcess:
    """Commit EXACTLY these bytes at these paths on top of HEAD, through a temporary index (the
    repository's hooks run as for any commit); the real index is then reset for those paths, so the
    working tree's later writes stay changes for the next checkpoint. HEAD moved meanwhile (another
    committer): nothing is committed, said, retried at the next tick."""
    import tempfile
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    fd, idx = tempfile.mkstemp(prefix="nbx-ckpt-index-", dir=str(_git_dir(repo)))
    os.close(fd)
    os.unlink(idx)
    env = dict(os.environ, GIT_INDEX_FILE=idx)
    try:
        subprocess.run(["git", "-C", repo, "read-tree", head], env=env, check=True, capture_output=True)
        for rel, data in blobs.items():
            sha = subprocess.run(["git", "-C", repo, "hash-object", "-w", "--stdin"], input=data,
                                 capture_output=True, check=True).stdout.decode().strip()
            subprocess.run(["git", "-C", repo, "update-index", "--add", "--cacheinfo", f"100644,{sha},{rel}"],
                           env=env, check=True, capture_output=True)
        if _git(repo, "rev-parse", "HEAD").stdout.strip() != head:
            return subprocess.CompletedProcess([], 1, "", "HEAD moved while the checkpoint was prepared")
        r = subprocess.run(["git", "-C", repo, "commit", "-q", "-F", "-"], input=message, env=env,
                           capture_output=True, text=True)
        if r.returncode == 0:
            _git(repo, "reset", "-q", "--", *blobs, check=False)
        return r
    finally:
        if os.path.exists(idx):
            os.unlink(idx)


def checkpoint(repo: str, rel_dir: str, remotes: List[str], gate_cmd: List[str], trailers: List[str],
               record: Optional[str] = None, say=print, label: str = "") -> Dict[str, object]:
    """One checkpoint: read the changed files ONCE → gate those bytes → commit those bytes → push
    every remote (never main) → read back → record."""
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    files = changed_files(repo, rel_dir)
    result: Dict[str, object] = {"committed": [], "refused": [], "sha": None, "remotes": {}, "files": files}
    if not files:
        _record(record, f"== checkpoint {stamp}: nothing to commit under {rel_dir}", say)
        return result
    snap = {}
    for f in files:
        try:
            snap[f] = (Path(repo) / f).read_bytes()
        except FileNotFoundError:
            continue                    # replaced away between the status and the read: the next tick sees it
    refused_rel = _gate_snapshot(repo, rel_dir, snap, gate_cmd)
    to_commit = [f for f in snap if f not in refused_rel]
    result["refused"] = refused_rel
    if not to_commit:
        _record(record, f"== checkpoint {stamp}: {len(files)} changed file(s), ALL refused by the gate, nothing committed: "
                        + ", ".join(refused_rel), say)
        return result
    deltas = {f: entry_delta(repo, f, snap[f]) for f in to_commit}   # measured on the committed bytes
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
    rr = _commit_blobs(repo, {f: snap[f] for f in to_commit}, "\n".join(lines))
    if rr.returncode != 0:
        # the commit itself failed — a moved HEAD, a hook — said, retried at the next tick
        _record(record, f"== checkpoint {stamp}: commit FAILED — {(rr.stderr or rr.stdout).strip()[-300:]}", say)
        return result
    sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()
    result["sha"] = sha; result["committed"] = to_commit
    branch = current_branch(repo)
    status = []
    if remotes and branch == "main":
        # an unattended process never pushes main (the supervisor, 2026-09-28 06:16) — checked HERE, where
        # the push is, on the branch as it is now; it used to be checked once, in main(), at start only
        for remote in remotes:
            result["remotes"][remote] = "REFUSED: main is never pushed by an unattended process"
        status.append("NOT pushed: the branch is main")
        remotes = []
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
        label: str = "", push_interval: float = 1800.0) -> int:
    """Checkpoints every `interval` seconds; PUSHES at most once every `push_interval` seconds.

    The owner's account was suspended twice for automated pushes, and on 2026-09-28 this
    checkpointer pushed a working branch four times in an hour (supervisor 05:41): an
    unattended process pushes at most once every 30 minutes per repository, to a working
    branch only. Commits are local and as frequent as the interval; a commit made inside the
    push window stays local and is said so; the final checkpoint waits for the window to open
    rather than leave a proof only where it was written."""
    holder = hold_the_repository(repo)
    if holder is None:
        say(f"[checkpoint] REFUSED: another checkpointer holds {repo} ({_holder_line(repo)}) — one per repository")
        return 2
    for pid in producers:
        if not producer_alive(pid):
            say(f"[checkpoint] producer {pid} ({producer_name(pid)}) is already gone at start — one last checkpoint, then exit")
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    # A checkpointer outlives the terminal that launched it: a tmux window closing when its chain
    # ends SIGHUPs the whole group, and it killed a final checkpoint mid-gate on 2026-09-28 (four
    # certified entries uncommitted for 6.5 h, a checkpoint never pushed). Ignored, and inherited
    # as ignored by the gate it runs, the hangup cannot cost a result; SIGTERM still ends it cleanly.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    last = time.monotonic()
    while True:
        alive = [p for p in producers if producer_alive(p)]
        final = once or stop["now"] or (bool(producers) and not alive)
        due = time.monotonic() - last >= interval
        if due or final:
            def seconds_until_push_window():
                # the window is the REPOSITORY's, shared by every worktree and by manual pushes
                # (`--touch-push`): on 2026-09-28 a per-process window let this tool push 23 minutes
                # after a manual batched push of the same repository (06:06 and 06:29).
                return max(0.0, push_interval - (time.time() - last_push_time(repo)))
            if final and remotes and seconds_until_push_window() > 0:
                wait = seconds_until_push_window()
                say(f"[checkpoint] final checkpoint: the repository's push window opens in {wait:.0f} s (at most one "
                    f"push per {push_interval:.0f} s); waiting, the commit is local meanwhile")
                time.sleep(wait)
            push_now = bool(remotes) and seconds_until_push_window() == 0
            res = checkpoint(repo, rel_dir, remotes if push_now else [], gate_cmd, trailers, record, say=say, label=label)
            last = time.monotonic()
            if res.get("sha"):
                if push_now and res.get("remotes"):
                    # an ATTEMPT spends the window, landed or refused: a refused push was retried at every
                    # tick (10 min), three attempts per window (the tools audit, 2026-09-29)
                    touch_push(repo)
                elif not push_now:
                    _record(record, f"== {res['sha']}: committed, NOT pushed (the repository's push window opens in "
                                    f"{seconds_until_push_window():.0f} s)", say)
            if final:
                failed = [r for r, why in (res.get("remotes") or {}).items() if why]
                if failed:
                    say(f"[checkpoint] final checkpoint did not reach: {', '.join(failed)} — rc 3")
                    return 3
                return 0
        time.sleep(poll)


_HELD: Dict[str, object] = {}


def _worktree_git_dir(repo: str) -> Path:
    own = subprocess.run(["git", "-C", repo, "rev-parse", "--git-dir"], capture_output=True, text=True)
    d = own.stdout.strip() if own.returncode == 0 and own.stdout.strip() else ".git"
    return Path(d if os.path.isabs(d) else os.path.join(repo, d))


def _holder_path(repo: str) -> Path:
    """The hold lives in the WORKTREE's git dir: what two checkpointers collided on was the index
    lock, and a worktree has its own index. The push window stays in the common git dir
    (`_push_stamp_path`), so the one-push-per-30-minutes rule still counts the whole repository.
    Measured 2026-10-04 19:07: rc1's checkpointer held the clone-wide lock, so no checkpointer could
    start for the value-axis worktree, and the certifier, which correctly asks for one on its own
    tree, refused both memory classes (rc=3)."""
    return _worktree_git_dir(repo) / "nbx-checkpointer.lock"


def _holder_line(repo: str) -> str:
    try:
        return _holder_path(repo).read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def hold_the_repository(repo: str):
    """ONE checkpointer per worktree (the index it commits through is the worktree's own): an exclusive,
    non-blocking lock held for this process's life. None when another process holds it — the rule
    was written in a skill and enforced nowhere (the tools audit, 2026-09-29); two checkpointers
    collided on the index lock. Idempotent within one process."""
    import fcntl
    key = str(_holder_path(repo))
    if key in _HELD:
        return _HELD[key]
    fh = open(key, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.seek(0); fh.truncate(); fh.write(f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n"); fh.flush()
    _HELD[key] = fh
    return fh


def refuse_a_repo_this_tool_is_not_from(repo: str, tool: Optional[str] = None) -> str:
    """The message refusing a `--repo` that carries its OWN, different checkpointer, or "".

    The push window, the hangup immunity and every later rule live in this file; a checkpointer run
    from another tree's copy applies that tree's rules to this one. Measured 2026-09-28 18:55: two
    chains launched from tmux windows (cwd: the main checkout) ran `tools/certified_checkpoint.py` by
    a relative path after a backgrounded `cd` — main's copy, older than the certify tree's, without
    the push window — and two unattended pushes of one branch went out 97 s apart. A repo without a
    copy of its own (a test's scratch repo) has no rules of its own to lose, and passes."""
    me = Path(tool or __file__).resolve()
    top = _git(repo, "rev-parse", "--show-toplevel", check=False).stdout.strip()
    own = Path(top) / "tools" / me.name if top else None
    if own is None or not own.exists() or own.resolve() == me or own.read_bytes() == me.read_bytes():
        return ""
    return (f"REFUSED: --repo {top} carries its own checkpointer ({own}), different from this one ({me});\n"
            f"  the tree's rules live in its own copy — run {own}.")


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
    p.add_argument("--interval", type=float, default=600.0, help="seconds between checkpoints (commits)")
    p.add_argument("--push-interval", type=float, default=1800.0,
                   help="seconds between PUSHES, at least; the owner's rule is one automated push per 30 minutes per "
                        "repository, working branches only (2026-09-28)")
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--remotes", default="origin,gitlab")
    p.add_argument("--record", default=None, help="a campaign RUN.md every checkpoint appends one line to")
    p.add_argument("--trailer", action="append", default=[], help="a line appended to each commit message")
    p.add_argument("--gate-cmd", default=None,
                   help="JSON list; the gate command that receives the directory as its last argument "
                        "(default: the engine's `autotune check --dir`)")
    p.add_argument("--label", default="", help="a short name for the pass, in each commit's subject")
    p.add_argument("--once", action="store_true")
    p.add_argument("--touch-push", action="store_true",
                   help="record a push of --repo made by hand now, so this tool's window counts it; then exit")
    p.add_argument("--allow-parent-as-producer", action="store_true",
                   help="permit --producer-pid to name the shell that launched this "
                        "checkpointer; only correct if that shell will NOT wait on it")
    a = p.parse_args(argv)
    deadlock = refuse_a_producer_that_will_wait_for_us(a.producer_pid, os.getppid())
    if deadlock and not a.allow_parent_as_producer:
        print(deadlock, file=sys.stderr)
        return 2
    foreign = refuse_a_repo_this_tool_is_not_from(a.repo)
    if foreign:
        print(foreign, file=sys.stderr)
        return 2
    branch = _git(a.repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    if branch == "main" and [r for r in a.remotes.split(",") if r]:
        # an unattended process never pushes main (the supervisor, 2026-09-28 06:16)
        print("REFUSED: an unattended checkpointer never pushes main — run it on a working branch", file=sys.stderr)
        return 2
    gate = json.loads(a.gate_cmd) if a.gate_cmd else DEFAULT_GATE
    d = Path(a.repo) / a.dir
    if not a.touch_push and not d.is_dir():
        print(f"REFUSED: --dir {a.dir}: no such directory under {a.repo} — a checkpointer over nothing "
              f"commits nothing forever", file=sys.stderr)
        return 2
    if a.touch_push:
        touch_push(a.repo); print(f"[checkpoint] push of {a.repo} recorded at {time.strftime('%H:%M:%S')}")
        return 0
    return run(a.repo, a.dir, a.producer_pid, a.interval, [r for r in a.remotes.split(",") if r], gate,
               a.trailer, a.record, once=a.once, poll=a.poll, label=a.label, push_interval=a.push_interval)


if __name__ == "__main__":
    sys.exit(main())
