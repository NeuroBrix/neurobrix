#!/usr/bin/env python3
"""gate_run.py — THE gate, one script, one wake (the validation regime, Hocine 2026-10-14 14:20 / 18:05).

    computed scope (CPU)  →  ONE class (the card's real VRAM)  →  two scheduler instances  →  the tally.

1. The scope: `tools/gate_scope.py` from the tree's merge-base with its trunk to its head, against the base gate
   (the last gate of the class, whose rows the unreached cells are carried from). Everything or nothing is never
   a scope: the tool names the units a cell reaches.
2. One class per gate: the class the base gate holds rows for (default 32g). The other class is not executed: the
   models whose Prism plan is IDENTICAL under both profiles (the committed list `--plan-identical`, written by
   `tools/plan_listing.py` without executing) have ONE cell, which the ANY-CARD scheduler instance places on the
   cards of the other class and records under the gate's class; the other models run on the class's cards.
   A model validates once per distinct Prism plan, not per card class.
3. Cells above ten minutes at their last measurement are carried by record (`<gate>/gate/carries.yml`, beside the rows, written by the
   caller with provenance) until the gate request at the model's floor lands in the registries.
4. The tally (`tools/gate_tally.py`): replayed, carried, missing, never measured, judged / not judged, against the
   committed references; its first line is the gate's verdict line in RUN.md.

Every step appends one line to RUN.md and the whole run ends with "<label> GATE DONE": one waiter, one wake.
`--plan` prints the steps and runs nothing.

    gate_run.py --campaign <dir> --tree <worktree> --base-gate <dir> --plan-identical <file> --run-md <RUN.md>
                --holds <dir> --hardware <id> --class-split-mib 24576 --heavy-share 0.25 --need-workers 16
                [--cls 32g] [--peak-rows <pattern>]... [--label <word>] [--plan]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Refused(RuntimeError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def say(run_md: Path, msg: str) -> None:
    with open(run_md, "a") as f:
        f.write(f"{msg} {now()}\n")
    print(msg, flush=True)


def git(tree: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise Refused(f"git {' '.join(args)} in {tree}: {(r.stderr or r.stdout).strip()[-200:]}")
    return r.stdout.strip()


def read_list(path: Path) -> set:
    if not path.is_file():
        raise Refused(f"{path}: the plan-identical list is not there — tools/plan_listing.py writes it; nothing scheduled")
    return {l.split()[0] for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}


def split_queue(lines: list, identical: set) -> tuple:
    """(lines for the class's cards, lines for any card): a queue line's model in the plan-identical list goes to
    the any-card instance, every other line to the class's cards. Comments and blanks are dropped."""
    own, anyc = [], []
    for l in lines:
        if not l.strip() or l.startswith("#"):
            continue
        (anyc if l.split()[0] in identical else own).append(l)
    return own, anyc


def any_card_classes(cls: str) -> list:
    """The card classes the any-card instance serves: every class the scheduler knows but the gate's."""
    import gate_sched
    if cls not in gate_sched.CLASS_LABELS:
        raise Refused(f"class {cls!r} is not one the scheduler knows {gate_sched.CLASS_LABELS}")
    return [c for c in gate_sched.CLASS_LABELS if c != cls]


def layout() -> dict:
    """THE gate layout, the scope tool's (tools/gate_scope.yml `gate:`): lists, rows, timeouts, relative to a gate's
    directory. One definition: the scope tool reads a base gate where this tool writes it."""
    import yaml
    lay = (yaml.safe_load((HERE / "gate_scope.yml").read_text()) or {}).get("gate") or {}
    for k in ("lists", "rows", "timeouts"):
        if k not in lay:
            raise Refused(f"tools/gate_scope.yml names no gate.{k}: the gate layout is read there")
    if not lay["rows"].endswith("matrix_{cls}"):
        raise Refused(f"gate.rows {lay['rows']!r}: the scheduler writes <dir>/matrix_<cls>, the layout must end so")
    return lay


def rows_root(C: Path) -> Path:
    """The directory whose matrix_<cls>/ holds a gate's rows (the scheduler's --gate), and its carries.yml."""
    rel = layout()["rows"][:-len("matrix_{cls}")].rstrip("/")
    return C / rel if rel else C


def carried(C: Path) -> int:
    """The cells the gate's carry record names (<rows root>/carries.yml), 0 without one."""
    f = rows_root(C) / "carries.yml"
    if not f.is_file():
        return 0
    import yaml
    return len((yaml.safe_load(f.read_text()) or {}).get("carries") or [])


def check_carries(tree: Path, C: Path) -> int:
    """Every entry of the gate's carry record, checked by the HARNESS's own reader (`record_carries` of the tree
    under test: the source row exists and is the measurement the entry names) before any card is spent. The
    reduction gate of 2026-10-15 lost two rounds to a record whose source rows had moved, then named the wrong
    engine: the harness refused each carried line only after the scheduler had placed it. Returns the entries."""
    f = rows_root(C) / "carries.yml"
    if not f.is_file():
        return 0
    import importlib.util
    import yaml
    spec = importlib.util.spec_from_file_location("_gate_harness", tree / "tools" / "regression_matrix.py")
    rm = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(tree / "tools"), str(tree / "src")]
    spec.loader.exec_module(rm)
    asked = {(e["model"], e["mode"]) for e in (yaml.safe_load(f.read_text()) or {}).get("carries") or []}
    try:
        got = rm.record_carries(f, asked, str(tree))
    except SystemExit as e:
        raise Refused(f"the carry record {f}: {e}") from e
    return len(got)


def steps(a) -> list:
    C, W, B = Path(a.campaign), Path(a.tree), Path(a.base_gate)
    lay = layout()
    for p, what in ((W / "tools" / "gate_scope.py", "the scope tool"), (B / lay["lists"].format(cls=a.cls), "the base gate's list"),
                    (B / lay["rows"].format(cls=a.cls), "the base gate's rows"),
                    (B / lay["timeouts"], "the base gate's per-model timeouts (the scope tool refuses a gate without them, "
                                         "after computing — 27 min lost on 2026-10-14)")):
        if not p.exists():
            raise Refused(f"{p}: {what} is not there; nothing scheduled")
    identical = read_list(Path(a.plan_identical))
    base_sha = git(W, "merge-base", a.trunk, "HEAD")
    head = git(W, "rev-parse", "--short", "HEAD")
    scope = C / "scope"
    py = a.python
    env = dict(os.environ, PYTHONNOUSERSITE="1", NEUROBRIX_PYTHON=py, NBX_PYTHON=py)
    return [
        ("scope", [py, str(W / "tools" / "gate_scope.py"), "--base", base_sha, "--head", head, "--gate-dir", str(B),
                   "--out", str(scope)], dict(env, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(W / "src")), W),
        ("split", identical, scope, None),
        ("sched", [py, str(HERE / "gate_sched.py"), "--gate2", str(C / "gate2"), "--gate", str(rows_root(C)), "--tree", str(W),
                   "--run-md", str(a.run_md), "--holds", str(a.holds), "--hardware", a.hardware, "--class-split-mib",
                   str(a.class_split_mib), "--heavy-share", str(a.heavy_share), "--need-workers", str(a.need_workers),
                   "--state", str(C / "sched"), "--label", a.label, "--python", py]
                  + sum((["--peak-rows", x] for x in a.peak_rows), []), env, W),
        ("sched_any", [py, str(HERE / "gate_sched.py"), "--gate2", str(C / "gate2_any"), "--gate", str(rows_root(C)), "--tree", str(W),
                       "--run-md", str(a.run_md), "--holds", str(a.holds), "--hardware", a.hardware, "--class-split-mib",
                       str(a.class_split_mib), "--heavy-share", str(a.heavy_share), "--need-workers", str(max(1, a.need_workers // 2)),
                       "--state", str(C / "sched_any"), "--out-class", a.cls, "--label", a.label, "--python", py]
                      + sum((["--peak-rows", x] for x in a.peak_rows), []), env, W),
        ("tally", [py, str(HERE / "gate_tally.py"), "--gate", str(C), "--base", str(B), "--scope", str(scope), "--tree", str(W),
                   "--classes", a.cls], dict(env, CUDA_VISIBLE_DEVICES=""), C),
    ]


def run(a) -> int:
    C, W, B = Path(a.campaign), Path(a.tree), Path(a.base_gate)
    run_md = Path(a.run_md)
    plan = steps(a)
    if a.plan:
        for name, cmd, *_ in plan:
            print(name, ":", " ".join(cmd) if isinstance(cmd, list) else f"plan-identical models: {len(cmd)}")
        return 0
    C.mkdir(parents=True, exist_ok=True)
    for d in ("gate2", "gate2_any", "sched", "sched_any", "scope"):
        (C / d).mkdir(exist_ok=True)
    rows_root(C).mkdir(parents=True, exist_ok=True)
    # the gate's list is the base gate's: the same cells wanted, judged against the same reference
    # every class's list is carried forward (the scope tool reads them all, the gated class's and the others'):
    # the second gate built on the first refused for want of its 16g list (2026-10-14 22:27)
    for lst in sorted((B / Path(layout()["lists"]).parent).glob("gate_*.list")):
        (C / "gate2" / lst.name).write_text(lst.read_text())
        (C / "gate2_any" / lst.name).write_text(lst.read_text())
    tmo = layout()["timeouts"]
    if (B / tmo).is_file() and not (C / tmo).is_file():
        (C / tmo).write_text((B / tmo).read_text())
    label = a.label
    head = git(W, "rev-parse", "--short", "HEAD")
    if a.resume:
        # the scope and the split are already on disk (a stopped run): the two instances take their own queues back
        for d in ("sched", "sched_any"):
            if not (C / d).is_dir():
                raise Refused(f"--resume: {C / d} is not there — the gate never split; run it without --resume")
        say(run_md, f"{label} GATE RESUME tree {head} class {a.cls}: the scope and the split on disk are kept")
        say(run_md, f"{label} carry record checked by the harness: {check_carries(W, C)} entries")
        return _schedule_and_tally(a, C, B, run_md, plan, label, any_lines=True)
    say(run_md, f"{label} GATE START tree {head} class {a.cls} base {B}")
    # 1. the scope
    name, cmd, env, cwd = plan[0]
    (C / "scope").mkdir(exist_ok=True)
    with open(C / "scope" / "scope.log", "w") as log:
        rc = subprocess.run(cmd, env=env, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, timeout=a.scope_timeout).returncode
    queue = C / "scope" / f"queue_{a.cls}.txt"
    lines = queue.read_text().splitlines() if queue.is_file() else []
    say(run_md, f"{label} scope rc={rc} {len(lines)} lines {a.cls}")
    if rc != 0:
        say(run_md, f"{label} GATE DONE (scope refused: see {C / 'scope' / 'scope.log'})")
        return 1
    # 2. the split: plan-identical models to any card, the others to the class's cards
    own, anyc = split_queue(lines, plan[1][1])
    (C / "gate2" / f"queue_{a.cls}.txt").write_text("".join(l + "\n" for l in own))
    # the any-card instance serves the cards of the OTHER class (a scheduler reads the queue named after the class of
    # the card it places on): the class instance and it never share a card, and both read the shared holds — the
    # first run gave the any-card lines the gate's own class and private holds, and placed two cells on a held card
    for other in any_card_classes(a.cls):
        (C / "gate2_any" / f"queue_{other}.txt").write_text("".join(l + "\n" for l in anyc))
    say(run_md, f"{label} split: {len(own)} lines on the {a.cls} cards + {len(anyc)} plan-identical lines on the "
                f"{','.join(any_card_classes(a.cls))} cards; {carried(C)} cells carried by record")
    say(run_md, f"{label} carry record checked by the harness: {check_carries(W, C)} entries")
    return _schedule_and_tally(a, C, B, run_md, plan, label, any_lines=bool(anyc), own_lines=bool(own))


def _schedule_and_tally(a, C, B, run_md, plan, label, any_lines=True, own_lines=True) -> int:
    # 3. the two scheduler instances, together
    procs = []
    for name, cmd, env, cwd in plan[2:4]:
        if name == "sched_any" and not any_lines:
            continue
        if name == "sched" and not own_lines:
            continue
        with open(C / f"{name}.log", "a") as log:
            procs.append((name, subprocess.Popen(cmd, env=env, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)))
    rcs = {name: p.wait() for name, p in procs}
    say(run_md, f"{label} sched rc={rcs}")
    # 4. the tally
    name, cmd, env, cwd = plan[4]
    with open(C / "tally.txt", "w") as out:
        rc = subprocess.run(cmd, env=env, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT).returncode
    first = (C / "tally.txt").read_text().splitlines()[:1]
    say(run_md, f"{label} TALLY rc={rc} {first[0][:200] if first else ''}")
    say(run_md, f"{label} GATE DONE")
    return rc


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", required=True, help="this gate's directory (created), in the scope tool's layout (tools/gate_scope.yml gate:): gate2/, gate/matrix_<cls>/, timeouts.txt; and scope/, gate2_any/, sched/, sched_any/")
    ap.add_argument("--tree", required=True, help="the worktree under test")
    ap.add_argument("--trunk", default="origin/main", help="the ref the scope's base is the merge-base with")
    ap.add_argument("--base-gate", required=True, help="the last gate of the class: gate2/gate_<cls>.list, matrix_<cls>/ rows")
    ap.add_argument("--plan-identical", required=True, help="the committed list of models whose plan is identical under both profiles")
    ap.add_argument("--cls", default="32g")
    ap.add_argument("--run-md", required=True)
    ap.add_argument("--holds", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--class-split-mib", type=int, required=True)
    ap.add_argument("--heavy-share", type=float, required=True)
    ap.add_argument("--need-workers", type=int, required=True)
    ap.add_argument("--peak-rows", action="append", default=[])
    ap.add_argument("--label", default="gate")
    ap.add_argument("--scope-timeout", type=int, default=7200)
    ap.add_argument("--python", default=os.environ.get("NEUROBRIX_PYTHON") or os.environ.get("NBX_PYTHON") or sys.executable)
    ap.add_argument("--plan", action="store_true", help="print the steps, run nothing")
    ap.add_argument("--resume", action="store_true",
                    help="a stopped gate: keep the scope and the split on disk, run the two instances and the tally")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return run(a)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
