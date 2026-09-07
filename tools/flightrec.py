#!/usr/bin/env python3
"""Flight recorder for long-running jobs (power-loss / crash resume).

The machine hosting the GPUs has no UPS: a power cut reboots the server
mid-computation and every in-flight process dies silently. This tool
makes such interruptions *visible and resumable*:

  1. Launch any long job through the wrapper::

         python3 tools/flightrec.py run --label "bench heavy row" \
             --gpu 3 --note "columns left: neurobrix_pytorch,triton" \
             -- python3 benchmarks/harness/run_bench.py --row ...

     A flight record (JSON, fsync'ed so it survives the outage) is
     written to .flightrec/ BEFORE the child starts, and updated with
     the exit status when the child finishes.

  2. At every session start, `check --hook` scans the records. A record
     still "in_flight" whose boot_id differs from the current one is
     mechanical proof of a reboot during the run (power loss); a dead
     PID on the same boot is a crash. Either way the hook prints a loud
     resume block with the exact command to re-run.

  3. `clear <id>` acknowledges a record once resumed or obsolete.

  4. The repository gate. A power cut truncates whatever git was writing
     (2026-09-03 00:12: two loose objects left empty, found four days
     later); a job resumed on a corrupt repository commits on top of the
     damage. So the procedure runs `git fsck --full` once per boot — at
     the session hook, and before any `run` — and `run` REFUSES to
     launch while the verdict is not clean. `fsck` re-checks by hand
     after the repair (quarantine the corrupt object files out of
     .git/objects, fetch both remotes, rebuild what neither has).

  5. `wait <id>` blocks until a record leaves in_flight and exits 0 only
     when the job ended `done` — a chain's next step waits on the RECORD,
     so a resumed job (a new wrapper, a new pid) still fires its waiters,
     and a killed or failed job never does.

Commands: run, status, check [--hook], clear <id>|--all-stale, fsck, wait <id>.
Stdlib only; records live in <repo>/.flightrec/ (gitignored).
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REC_DIR = REPO / ".flightrec"
BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")


def current_boot_id() -> str:
    try:
        return BOOT_ID_FILE.read_text().strip()
    except OSError:
        return "unknown"


def write_record(path: Path, record: dict) -> None:
    """Write + fsync the record AND its directory entry.

    A power cut can happen milliseconds after launch: without fsync the
    record may still sit in the page cache and vanish with the outage,
    which is exactly the event this tool exists to survive.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def load_records() -> list:
    records = []
    if not REC_DIR.is_dir():
        return records
    for p in sorted(REC_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
            rec["_path"] = str(p)
            records.append(rec)
        except (json.JSONDecodeError, OSError):
            print(f"[flightrec] warning: unreadable record {p}",
                  file=sys.stderr)
    return records


def pid_is_this_wrapper(pid: int) -> bool:
    """True iff `pid` is alive AND still a flightrec wrapper process.

    PID liveness alone is not enough: after weeks of uptime a freed PID
    can be recycled by an unrelated process, which would misclassify a
    crashed job as still running.
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"flightrec" in cmdline


def classify(rec: dict) -> str:
    """RUNNING | POWER_LOSS | DIED | <terminal status as-is>."""
    if rec.get("status") != "in_flight":
        return rec.get("status", "unknown")
    if rec.get("boot_id") != current_boot_id():
        return "POWER_LOSS"
    if pid_is_this_wrapper(int(rec.get("pid", -1))):
        return "RUNNING"
    return "DIED"


def resume_block(rec: dict, cause: str) -> str:
    rec_id = rec.get("id", "?")
    lines = [
        f"[{rec_id}] {rec.get('label', '?')}",
        f"    started : {rec.get('started_iso', '?')}"
        + (f"  (GPU {rec['gpu']})" if rec.get("gpu") is not None else ""),
        f"    cause   : {cause}"
        + (" — machine rebooted while the job was in flight"
           if cause == "POWER_LOSS" else
           " — process died on the same boot (crash or kill)"
           if cause == "DIED" else ""),
    ]
    if rec.get("note"):
        lines.append(f"    note    : {rec['note']}")
    if rec.get("log"):
        lines.append(f"    log     : {rec['log']}")
    lines.append(f"    RESUME  : cd {rec.get('cwd', str(REPO))} && "
                 f"{rec.get('resume_cmd', '?')}")
    lines.append(f"    then    : python3 tools/flightrec.py clear {rec_id}")
    return "\n".join(lines)


FSCK_FILE = REC_DIR / "fsck.json"
FSCK_REFUSED = 3


def run_git_fsck() -> "tuple[int, str]":
    """`git fsck --full` on the repository: (exit code, output). Dangling
    objects are not a defect (every rebase leaves some) and are not listed."""
    try:
        proc = subprocess.run(["git", "-C", str(REPO), "fsck", "--full", "--no-dangling"],
                              capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"git fsck could not run: {exc}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def fsck_verdict(force: bool = False) -> dict:
    """The repository's verdict for THIS boot: read from the record when one
    exists for the current boot id (a power cut is a boot boundary, so a clean
    verdict holds until the next one), else measured now and recorded."""
    boot = current_boot_id()
    if not force and FSCK_FILE.exists():
        try:
            rec = json.loads(FSCK_FILE.read_text())
            if rec.get("boot_id") == boot and "clean" in rec:
                return rec
        except (json.JSONDecodeError, OSError):
            pass
    code, out = run_git_fsck()
    rec = {"boot_id": boot, "clean": code == 0, "exit_code": code,
           "checked_iso": time.strftime("%Y-%m-%d %H:%M:%S"), "report": out}
    REC_DIR.mkdir(exist_ok=True)
    write_record(FSCK_FILE, rec)
    return rec


def fsck_block(rec: dict) -> str:
    if rec["clean"]:
        return (f"[flightrec] repository: git fsck --full clean "
                f"(checked {rec['checked_iso']}, this boot)")
    lines = ["=" * 64,
             "FLIGHT RECORDER — REPOSITORY CORRUPT: RESUME REFUSED",
             "=" * 64,
             f"git fsck --full failed (exit {rec['exit_code']}, checked {rec['checked_iso']}):"]
    lines += ["    " + l for l in rec["report"].splitlines()[:20]]
    lines += ["Repair before any resume: note what references each bad object (fsck says),",
              "move the bad files out of .git/objects, `git fetch origin` and `git fetch gitlab`;",
              "what neither remote brings back is rebuilt from the working tree (blob: git",
              "hash-object -w, hash checked; tree/commit: recommitted, sha noted in the journal).",
              "Then `python3 tools/flightrec.py fsck` — `run` launches again once it is clean."]
    return "\n".join(lines)


def cmd_wait(args) -> int:
    """Poll the record every --every seconds until its status leaves in_flight;
    0 iff done. A record that dies (crash, power loss) ends the wait with 1."""
    while True:
        recs = [r for r in load_records() if r.get("id") == args.id]
        if not recs:
            print(f"[flightrec] wait: no record {args.id}", file=sys.stderr)
            return 2
        cause = classify(recs[0])
        if cause != "RUNNING":
            print(f"[flightrec] wait: {args.id} ended {cause}", flush=True)
            return 0 if cause == "done" else 1
        time.sleep(args.every)


def cmd_fsck(args) -> int:
    rec = fsck_verdict(force=True)
    print(fsck_block(rec))
    return 0 if rec["clean"] else FSCK_REFUSED


def cmd_run(args) -> int:
    if not args.cmd:
        print("[flightrec] run: no command given after --", file=sys.stderr)
        return 2
    verdict = fsck_verdict()
    if not verdict["clean"]:
        print(fsck_block(verdict), file=sys.stderr, flush=True)
        return FSCK_REFUSED
    REC_DIR.mkdir(exist_ok=True)
    label_slug = re.sub(r"[^a-zA-Z0-9]+", "-", args.label).strip("-")[:60]
    rec_id = time.strftime("%Y%m%d_%H%M%S") + "_" + label_slug
    path = REC_DIR / f"{rec_id}.json"
    record = {
        "id": rec_id,
        "label": args.label,
        "status": "in_flight",
        "cmd": shlex.join(args.cmd),
        "resume_cmd": args.resume_cmd or shlex.join(args.cmd),
        "note": args.note,
        "gpu": args.gpu,
        "log": args.log,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "boot_id": current_boot_id(),
        "started_unix": time.time(),
        "started_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_record(path, record)
    print(f"[flightrec] in flight: {rec_id}", flush=True)

    # Forward SIGTERM/SIGINT to the child so a deliberate kill of the
    # wrapper stops the job AND gets recorded as "killed", not "DIED".
    child = subprocess.Popen(args.cmd)
    killed = []

    def forward(signum, _frame):
        killed.append(signum)
        child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = child.wait()
    record["status"] = ("killed" if killed
                        else "done" if code == 0 else "failed")
    record["exit_code"] = code
    record["ended_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_record(path, record)
    print(f"[flightrec] {record['status']} (exit {code}): {rec_id}",
          flush=True)
    return code


def cmd_status(args) -> int:
    records = load_records()
    if not records:
        print("[flightrec] no records")
        return 0
    for rec in records:
        if rec.get("status") == "cleared" and not args.all:
            continue
        print(f"{classify(rec):>10}  {rec.get('id')}  {rec.get('label')}")
    return 0


def cmd_check(args) -> int:
    stale, running = [], []
    for rec in load_records():
        cause = classify(rec)
        if cause in ("POWER_LOSS", "DIED"):
            stale.append((rec, cause))
        elif cause == "RUNNING":
            running.append(rec)
    # The repository before the records: a cut that killed the jobs may have
    # truncated git's writes too. Measured once per boot; in hook mode a clean
    # verdict is said only when there is something else to say.
    verdict = fsck_verdict()
    if not verdict["clean"] or stale or not args.hook:
        print(fsck_block(verdict))
        print()
    if stale:
        print("=" * 64)
        print("FLIGHT RECORDER — UNCLEAN SHUTDOWN: WORK WAS IN FLIGHT")
        print("=" * 64)
        for rec, cause in stale:
            print(resume_block(rec, cause))
            print()
        print("Check `nvidia-smi` (GPUs must be idle) before resuming.")
        print("Partial output files of the dead run may exist — the "
              "resume command must be idempotent or re-run the cell.")
    if running:
        print(f"[flightrec] {len(running)} job(s) alive and in flight:")
        for rec in running:
            print(f"    [{rec.get('id')}] {rec.get('label')}"
                  f" (pid {rec.get('pid')})")
    if not stale and not running and not args.hook:
        print("[flightrec] all clear")
    return 0


def cmd_clear(args) -> int:
    records = load_records()
    targets = []
    for rec in records:
        cause = classify(rec)
        if args.all_stale and cause in ("POWER_LOSS", "DIED"):
            targets.append(rec)
        elif args.id and rec.get("id") == args.id:
            targets.append(rec)
    if not targets:
        print("[flightrec] nothing matched", file=sys.stderr)
        return 1
    for rec in targets:
        path = Path(rec.pop("_path"))
        rec["status"] = "cleared"
        rec["cleared_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_record(path, rec)
        print(f"[flightrec] cleared: {rec['id']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="launch a job under the recorder")
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--note", default=None,
                       help="campaign context: what step, what comes next")
    p_run.add_argument("--gpu", type=int, default=None)
    p_run.add_argument("--log", default=None,
                       help="path to the job's own log file, for the "
                            "resume block")
    p_run.add_argument("--resume-cmd", default=None,
                       help="command to resume (default: re-run the "
                            "wrapped command)")
    p_run.add_argument("cmd", nargs=argparse.REMAINDER,
                       help="-- command to run")
    p_status = sub.add_parser("status", help="list records")
    p_status.add_argument("--all", action="store_true")
    p_check = sub.add_parser("check", help="detect stale in-flight records")
    p_check.add_argument("--hook", action="store_true",
                         help="session-start mode: silent when clear")
    p_clear = sub.add_parser("clear", help="acknowledge a record")
    p_clear.add_argument("id", nargs="?")
    p_clear.add_argument("--all-stale", action="store_true")
    sub.add_parser("fsck", help="re-check the repository now (git fsck --full); "
                               "run refuses to launch while it is not clean")
    p_wait = sub.add_parser("wait", help="block until a record leaves in_flight; exit 0 iff done")
    p_wait.add_argument("id")
    p_wait.add_argument("--every", type=float, default=60.0, help="poll period in seconds")

    args = ap.parse_args()
    if args.command == "run" and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    return {"run": cmd_run, "status": cmd_status, "check": cmd_check,
            "clear": cmd_clear, "fsck": cmd_fsck, "wait": cmd_wait}[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
