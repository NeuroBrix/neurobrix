#!/usr/bin/env python3
"""The rig's scheduler: one rule, read from the job, not decided again each time.

Three of four cards sat idle for hours while five campaigns queued behind one, because every
campaign inherited the same waiting rule — and that rule only ever belonged to a fraction of
them. A measurement of TIME needs the machine to itself; a check of BYTES does not.

  TIMED — benchmark rows, lever A/B, throughput, TTFT, anything whose output is a duration.
          The machine is exclusive, the clock is locked, one card at a time, the rest at rest.
          Nothing else runs, because a second job on a second card moves the number.

  UNTIMED — byte gates, retraces, builds, downloads, audits, correctness gates, byte-identity
          checks, certifications of correctness. Their output is an equality, and an equality
          does not care what the neighbouring card is doing. They run in PARALLEL on the free
          cards, one card per job, the card chosen from the profile's capacity and the job's
          declared weight — never guessed.

Two limits are the rack's, not the scheduler's taste, and they are written here so nobody has to
remember them: the cabinet has already tripped its breaker under load with the air conditioning
running, so no more than `--max-heavy` heavy jobs run at once and a card is left with margin; and
an untimed job YIELDS the moment a timed campaign is ready, because a falsified measurement costs
more than the work thrown away — it is stopped and put back at the head of its queue.

    python3 tools/rig_scheduler.py run   --queue queue.json
    python3 tools/rig_scheduler.py status

The queue is a list of jobs:

    {"name": "audio kv byte gate", "family": "untimed", "weight_gb": 10,
     "cmd": ["bash", "validation_outputs/.../gate.sh"], "done_when": "path/to/report.json",
     "impossible_when": "path/to/a_fact_that_makes_the_verdict_unreachable"}

`done_when` says the verdict is already KNOWN, `impossible_when` that it is
already IMPOSSIBLE. A job consults both before spending any of its time: the
certified campaign once spent 8 h on a second arm whose gate — `all(arms)` — was
already False, and the lesson is not about byte gates.

`family` is mandatory and has no default: a job that does not say whether its output is a time is
a job whose scheduling nobody has thought about.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

POLL_S = 20


def _gpus() -> List[dict]:
    """Every card, with the capacity the profile reports and what is on it right now."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    out = []
    for line in q.stdout.strip().splitlines():
        idx, total, used, util = [x.strip() for x in line.split(",")]
        out.append({"index": int(idx), "total_gb": int(total) / 1024,
                    "used_gb": int(used) / 1024, "util": int(util)})
    return out


def _compute_pids() -> List[int]:
    """PIDs that actually hold a card. A shell waiting on a log is not a job."""
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return sorted({int(p) for p in q.stdout.split() if p.strip().isdigit()})


@dataclass
class Job:
    name: str
    family: str
    cmd: List[str]
    weight_gb: float = 0.0
    heavy: bool = True
    done_when: Optional[str] = None
    # a path whose existence means this job's verdict can no longer
    # change — the precondition of its own gate, consulted before
    # any of its time is spent
    impossible_when: Optional[str] = None
    gpus: Optional[List[int]] = None
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    card: Optional[int] = None
    started: float = 0.0

    @property
    def finished(self) -> bool:
        return self.proc is not None and self.proc.poll() is not None


def _load(queue_path: Path) -> List[Job]:
    jobs = []
    for raw in json.loads(queue_path.read_text()):
        if raw.get("family") not in ("timed", "untimed"):
            raise SystemExit(
                f"job {raw.get('name')!r} declares family={raw.get('family')!r}: every job says "
                f"whether its output is a TIME (exclusive) or an EQUALITY (parallel). There is "
                f"no default, because a default is how five untimed campaigns came to queue "
                f"behind one.")
        cmd = raw["cmd"]
        jobs.append(Job(name=raw["name"], family=raw["family"],
                        cmd=shlex.split(cmd) if isinstance(cmd, str) else list(cmd),
                        weight_gb=float(raw.get("weight_gb", 0)),
                        heavy=bool(raw.get("heavy", True)),
                        done_when=raw.get("done_when"),
                        impossible_when=raw.get("impossible_when"),
                        gpus=raw.get("gpus")))
    return jobs


def _pick_card(job: Job, busy: set, margin_gb: float) -> Optional[int]:
    """A card whose capacity holds the job's declared weight plus the margin. Read, not guessed:
    a job that declares no weight takes the largest free card, and one that declares more than
    any card holds is refused by name rather than launched to fail."""
    free = [g for g in _gpus() if g["index"] not in busy]
    if not free:
        return None
    fits = [g for g in free if g["total_gb"] - g["used_gb"] >= job.weight_gb + margin_gb]
    if not fits:
        biggest = max(_gpus(), key=lambda g: g["total_gb"])
        if job.weight_gb + margin_gb > biggest["total_gb"]:
            raise SystemExit(
                f"job {job.name!r} declares {job.weight_gb:.1f} GB + {margin_gb:.1f} GB margin; "
                f"the largest card holds {biggest['total_gb']:.1f} GB. Refused here rather than "
                f"launched to fail on the card.")
        return None
    return min(fits, key=lambda g: g["total_gb"])["index"]      # the smallest card that fits


def cmd_run(args) -> int:
    queue = _load(Path(args.queue))
    pending, running, done = list(queue), [], []
    log = Path(args.log or "rig_scheduler.log")

    def say(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    say(f"queue: {sum(1 for j in queue if j.family == 'timed')} timed, "
        f"{sum(1 for j in queue if j.family == 'untimed')} untimed")

    while pending or running:
        for j in list(running):
            if j.finished:
                running.remove(j)
                done.append(j)
                say(f"done   {j.name} (rc={j.proc.returncode}, "
                    f"{(time.time() - j.started) / 60:.0f} min, card {j.card})")

        # THE PRECONDITION RULE. Every job consults the precondition of its own
        # VERDICT before it starts, and a job whose verdict is already known or
        # already impossible does not start — with its reason written.
        #
        # It was learned on the byte gate: the certified campaign spent 8 h on a
        # second arm that could not change anything, because `ran = all(arms)`
        # was already False. A per-unit budget with no notion of an already
        # undecidable unit spends its full allocation twice to learn one thing;
        # what was missing was not the budget but the consultation of the gate's
        # precondition before spending. That is not a property of byte gates, so
        # it does not live on one.
        #
        #   done_when exists      -> the verdict is already KNOWN
        #   impossible_when exists-> the verdict is already IMPOSSIBLE
        #
        # Both are paths, because a path is a fact on disk that survives a power
        # cut, and this rack has no UPS.
        still = []
        for j in pending:
            if j.done_when and Path(j.done_when).exists():
                say(f"skip   {j.name} -> its verdict is already known "
                    f"({j.done_when})")
                continue
            imp = getattr(j, "impossible_when", None)
            if imp and Path(imp).exists():
                say(f"skip   {j.name} -> its verdict is already IMPOSSIBLE "
                    f"({imp}); running it buys no information")
                continue
            still.append(j)
        pending = still

        head_timed = next((j for j in pending if j.family == "timed"), None)
        # A timed job takes the floor only when it could actually USE it. While a job this
        # scheduler does not own holds a card, the timed campaign cannot start whatever anyone
        # yields — so letting it block the untimed queue would idle the free cards for nothing,
        # which is the exact behaviour this scheduler was written to end.
        ours = {j.proc.pid for j in running if j.proc is not None}
        foreign = [p for p in _compute_pids() if p not in ours]
        if head_timed is not None and foreign:
            head_timed = None
        if head_timed is not None and not any(j.family == "timed" for j in running):
            # A timed campaign is ready. Everything untimed yields to it: a measurement taken
            # beside another job is not a measurement, and the work stopped here is cheap
            # against a number nobody can trust.
            if running:
                for j in running:
                    say(f"yield  {j.name} -> the timed campaign {head_timed.name!r} is ready")
                    j.proc.send_signal(signal.SIGTERM)
                for j in running:
                    try:
                        j.proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        j.proc.kill()
                    pending.insert(0, j)          # back at the head, it will be re-run whole
                    j.proc = None
                running = []
                continue
            if foreign:
                time.sleep(POLL_S)
                continue
            pending.remove(head_timed)
            head_timed.card = (head_timed.gpus or [args.timed_card])[0]
            head_timed.started = time.time()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(head_timed.card)}
            head_timed.proc = subprocess.Popen(head_timed.cmd, env=env)
            running.append(head_timed)
            say(f"TIMED  {head_timed.name} on card {head_timed.card}, machine exclusive")
            continue

        # The floor a timed job takes is held for its whole LIFE, not just the instant it
        # takes it. The branch above yields what was already running; without this line the
        # very next poll starts a fresh untimed job beside the measurement, because the
        # exclusivity test above is skipped once a timed job is running. That is the exact
        # falsification this scheduler exists to prevent, and it is cheap to hold: the cards
        # idle for the minutes a timed row takes, and every number it reports is a number.
        if any(j.family == "timed" for j in running):
            time.sleep(POLL_S)
            continue

        busy = {j.card for j in running if j.card is not None}
        heavy_running = sum(1 for j in running if j.heavy)
        for j in list(pending):
            if j.family != "untimed":
                continue
            if heavy_running >= args.max_heavy:
                break
            # A job that names several cards is one Prism will place across them: it takes the
            # rig, so it waits for the rig rather than starting beside a neighbour whose memory
            # it was going to need.
            if j.gpus and len(j.gpus) > 1:
                if running or _compute_pids():
                    continue
                card = j.gpus[0]
            else:
                card = j.gpus[0] if j.gpus else _pick_card(j, busy, args.margin_gb)
            if card is None:
                continue
            pending.remove(j)
            j.card = card
            j.started = time.time()
            busy.add(card)
            heavy_running += 1 if j.heavy else 0
            visible = ",".join(str(g) for g in (j.gpus if j.gpus else [card]))
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
            j.proc = subprocess.Popen(j.cmd, env=env)
            running.append(j)
            say(f"start  {j.name} on card(s) {visible} ({j.weight_gb:.0f} GB declared)")
        time.sleep(POLL_S)

    say(f"queue empty: {len(done)} job(s) ran")
    return 0


def cmd_status(args) -> int:
    pids = _compute_pids()
    print("card  total   used  util")
    for g in _gpus():
        print(f"{g['index']:4d} {g['total_gb']:6.0f} {g['used_gb']:6.1f} {g['util']:5d}%")
    print(f"\njobs that COMPUTE: {len(pids)}"
          + (f" (pids {pids})" if pids else " — the rig is idle"))
    waiting = subprocess.run(["bash", "-c",
                              "ps -eo pid,etime,cmd | grep -E 'tail -n 0 -F|sleep 60' "
                              "| grep -v grep | wc -l"], capture_output=True, text=True)
    print(f"shells that WAIT:   {waiting.stdout.strip()} — not counted as work")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queue", required=True)
    r.add_argument("--log", default=None)
    r.add_argument("--max-heavy", type=int, default=3,
                   help="the cabinet tripped its breaker under load; three at once, margin kept")
    r.add_argument("--margin-gb", type=float, default=2.0)
    r.add_argument("--timed-card", type=int, default=2)
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
