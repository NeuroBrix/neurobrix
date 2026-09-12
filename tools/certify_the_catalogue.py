#!/usr/bin/env python3
"""Certify the hub catalogue for one profile — plan first, run second.

THE TWO PHASES, AND WHY THE EXPENSIVE ONE IS NOT THE ONE YOU EXPECT

Certification needs a CENSUS of the shapes a model actually meets, and the
census comes from the machine's replay cache. So:

  phase MEET     one run per model with the certified directory OFF, so the
                 launcher sweeps and records every shape key it met. ONE run,
                 not a paired A/B with repetitions — this pass is not measuring
                 a gain, it is collecting shapes.
  phase CERTIFY  `neurobrix autotune certify --profile <p> --only-missing` runs
                 every candidate on every shape of the census against the fp64
                 oracle and writes the entries WITH their proofs.
  phase GATE     `neurobrix autotune check` — a file without a proof, or whose
                 proof does not re-read, is refused.
  phase REPORT   regenerate the catalogue document.

THE BUDGET GUARD, AND WHAT IT MAY NOT DO

A cell whose KNOWN cost exceeds what remains is refused at the door with that
cost named. A cell with NO known cost is never refused on a guess — the guard's
job is to stop a known cost, not to invent one. That rule is why
`Allegro` is refused (a recorded ~31 h per arm) and why an unrun 84 GB video
model is not.

WHAT IS NOT A FAILURE

A config the screen excludes is DATA. It is counted, named with its kernel, its
shape key and its deviation, and it appears in the report. An exclusion that
contradicts a certified entry is a finding and is reported as one — never a
silence.

MATCHING IS EXPLICIT

Hub slug to local container is case-insensitive exact, plus the alias table
below. No fuzzy rule: a prefix match in the first version of the report tool
attributed one model's measurement to a variant that had never been run. Where
two local containers could answer to one hub slug, the entry is AMBIGUOUS and is
reported as needing a decision — never resolved by picking one.

Usage:
    python tools/certify_the_catalogue.py --plan            # writes nothing, runs nothing
    python tools/certify_the_catalogue.py --profile volta --budget 28800
"""
from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import subprocess
import sys
import time
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
CACHE = Path(os.environ.get("NEUROBRIX_CACHE", Path.home() / ".neurobrix" / "cache"))
PY = os.environ.get("NBX_PYTHON", "/home/mlops/ml/venv/bin/python")

#: hub slug (lowercased) -> local container directory. ONLY where the two names
#: genuinely differ. Each line is a decision someone made by reading both names,
#: not a rule a machine inferred.
ALIASES = {
    "qwen3-30b-a3b-thinking": "Qwen3-30B-A3B-Thinking-2507",
    "sana-1600m-4kpx-bf16": "Sana_1600M_4Kpx_BF16",
    "wan2.1-t2v-1.3b": "Wan2.1-T2V-1.3B-Diffusers",
    "wan2.1-vace-1.3b": "Wan2.1-VACE-1.3B-diffusers",
    "wan2.1-i2v-14b-480p": "Wan2.1-I2V-14B-480P-Diffusers",
    "wan2.2-i2v-a14b": "Wan2.2-I2V-A14B-Diffusers",
    "sana-video-2b-720p": "SANA-Video_2B_720p_diffusers",
    "swin2sr-classical-x4": "swin2SR-classical-sr-x4-64",
    "swin2sr-classical-x2": "swin2SR-classical-sr-x2-64",
    "swin2sr-realworld-x4": "swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "whisper-v3-turbo": "whisper-large-v3-turbo",
    "voxtral-mini-3b": "Voxtral-Mini-3B-2507",
    # Resolved by a size bijection, not by the name: the hub lists exactly two
    # whisper builds, 5.8 GB and 1.5 GB, and this machine holds exactly two,
    # 5.8 GB (`whisper-large`) and 1.6 GB (`whisper-large-v3-turbo`). The
    # mapping is forced. The manifest itself does not carry the version, which
    # is a gap worth closing at the source rather than re-deducing here.
    "whisper-large-v2": "whisper-large",
}

#: Two local containers could answer to these hub slugs and NOBODY HAS DECIDED
#: which. Resolving one by picking the likelier name would put a measurement of
#: one artefact under the name of another — the defect this file is written
#: against. They are reported as ambiguous until a person says which.
AMBIGUOUS = {
    # Both builds come from the SAME HF repo (`canopylabs/orpheus-3b-0.1-ft`)
    # and both are 15 GB on disk; they differ by whether the SNAC decoder is
    # embedded in the container — the R34 question, a product decision rather
    # than a lookup. Which one the hub publishes as `Orpheus-3B` is not
    # readable from here, and picking the likelier name would file a
    # measurement of one artefact under the name of another.
    "orpheus-3b": ["orpheus-3b-0.1-ft", "orpheus-3b-0.1-ft-snac"],
}


def _hub_rows(snapshot: Path) -> list[dict]:
    rows, started = [], False
    for line in snapshot.read_text().splitlines():
        if line.startswith("---"):
            started = True
            continue
        if not started or not line.strip() or line.startswith(("Total:", "Install:", "Installed locally:")):
            continue
        m = re.match(r"^(\S+)\s+(\S+)\s+([\d.]+\s*[GM]B)\s+(.*?)\s+(\d+)\s*(installed)?\s*$", line)
        if not m:
            continue
        size = m.group(3).replace(" ", "")
        rows.append({
            "hub": m.group(1),
            "slug": m.group(1).split("/")[-1],
            "family": m.group(2).lower(),
            "gb": float(size[:-2]) / (1024 if size.endswith("MB") else 1),
        })
    return rows


def _local(slug: str) -> tuple[str | None, str]:
    """(container directory, why) — never a guess."""
    low = slug.lower()
    if low in AMBIGUOUS:
        return None, "AMBIGUOUS: " + " or ".join(AMBIGUOUS[low][:2])
    if low in ALIASES:
        d = CACHE / ALIASES[low]
        return (ALIASES[low], "alias") if d.is_dir() else (None, f"alias {ALIASES[low]} absent")
    for p in CACHE.iterdir():
        if p.is_dir() and p.name.lower() == low:
            return p.name, "exact"
    return None, "not installed"


def _known_costs(campaigns: Path) -> dict:
    """Recorded cost per model, from campaigns nobody voided.

    Priced at `arms=1`, because MEET runs each model ONCE. The recorded cells it
    reads are paired A/B campaigns, and their default price is the SUM of both
    arms — charging that to a one-run phase over-counts every model with a record
    by about a factor of two, and the budget then refuses models it could afford.

    TWO numbers per model, because two guards ask two questions:

      known_s   the WIDEST arm — what to RESERVE against the remaining budget.
      floor_s   the NARROWEST — what the doom test reads, since a model that has
                already finished once under the clock is not doomed by it.

    Reading the widest for both would refuse `Qwen3-VL-30B` against a 2700 s
    clock on the strength of a 3135 s sweeping arm, when it met in 365 s.
    """
    from precision_zoo_campaign import cell_cost_estimate  # reuse the brick
    void = {d.name for d in campaigns.glob("*/")
            if any((d / m).exists() for m in ("INVALIDATED.md", "PERTURBATION_NOTE.md"))}
    out = {}
    for result in campaigns.glob("*/proof/*/result.json"):
        if result.parent.parent.parent.name in void:
            continue
        est = cell_cost_estimate(result.parent, timeout=28800, arms=1)
        low = cell_cost_estimate(result.parent, timeout=28800, arms=1, narrowest=True)
        if est:
            model = result.parent.name
            row = {"known_s": est[0], "basis": est[1],
                   "floor_s": low[0] if low else est[0],
                   "floor_basis": low[1] if low else est[1]}
            if model not in out or row["known_s"] > out[model]["known_s"]:
                out[model] = row
    return out


def timeout_refusal(known_s, timeout: int, basis=None):
    """Why this run may not start, or None — its own cost against its own clock.

    The LEDGER and the KILL TIMER are two numbers about the same model, and until
    2026-09-12 nothing compared them. On 2026-09-11 the plan accepted
    `Wan2.1-T2V-1.3B` at its recorded cost and the runner killed it at `--timeout`
    anyway; `Allegro`, a recorded eight-hour model, went the same way. Two kills,
    ninety minutes of rig, and not one shape collected between them.

    A run whose OWN accepted cost exceeds the time it will be given is doomed
    before it starts, so the refusal strictly dominates the kill: today's
    behaviour spends the entire timeout to arrive at the same outcome, and loses
    the timeout as well. No information is lost by refusing — a killed run
    produces none.

    Nothing is guessed. A model with no recorded cost returns None and runs,
    exactly as the budget guard already promises: the job is to stop a KNOWN
    cost, never to invent one.
    """
    if not known_s or float(known_s) <= float(timeout):
        return None
    need = int(float(known_s)) + 1
    return (f"its own recorded cost {float(known_s):.0f} s exceeds the {timeout} s "
            f"it would be given — it would be killed at the wall having produced "
            f"nothing. Run it with --timeout {need} or more."
            + (f" Basis: {basis}" if basis else ""))


def _plan(args) -> list[dict]:
    rows = _hub_rows(args.snapshot)
    costs = _known_costs(args.campaigns)
    for r in rows:
        container, why = _local(r["slug"])
        r["container"], r["why"] = container, why
        est = costs.get(container or "") or {}
        r["known_s"], r["basis"] = est.get("known_s"), est.get("basis")
        r["floor_s"], r["floor_basis"] = est.get("floor_s"), est.get("floor_basis")
    rows.sort(key=lambda r: (r["container"] is None,          # runnable first
                             r["known_s"] is None,            # known cost first
                             r["known_s"] or 0,
                             r["gb"]))
    return rows


#: A campaign BETWEEN two runs holds no compute process. Asking nvidia-smi at
#: that instant returns an empty list and "the rig is free" is then a statement
#: about a moment, not about the rig. That is how a second instance was launched
#: onto a live measurement on 2026-09-10, from a log that looked idle — and this
#: check repeated the mistake in another form on 2026-09-11, returning 0 while a
#: gate held GPU0 five seconds later.
#:
#: So the rig is free when BOTH are true: no compute process, and no driver
#: alive that is about to start one.
_DRIVERS = ("precision_zoo_campaign.py", "flightrec.py", "certify_the_catalogue.py",
            "autotune_certify")


def _rig_busy() -> int:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return -1
    out = subprocess.run([smi, "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True)
    procs = len([l for l in out.stdout.splitlines() if l.strip()])
    if procs:
        _rig_busy.drivers = []
        return procs

    # Exclude MY OWN process GROUP, not my own pid. The first version compared
    # against `os.getpid()` and refused itself on 2026-09-11: the shell wrapper
    # that launches this tool carries the same script name on its command line
    # and has a different pid. An instrument that cannot recognise itself is the
    # same family as one that cannot recognise an absence.
    # Exclude MY OWN SESSION, not my pid and not my process group. Two versions
    # of this check refused their own run on 2026-09-11: the first compared
    # against `os.getpid()` and was defeated by the shell wrapper carrying the
    # same script name; the second compared against the process GROUP and was
    # defeated by a launcher that puts the wrapper in another group. The session
    # is the widest thing that is still unambiguously "this run".
    try:
        mine = os.getsid(0)
    except OSError:                                   # pragma: no cover
        mine = -1
    ps = subprocess.run(["ps", "-eo", "sid,pid,cmd"], capture_output=True, text=True).stdout
    drivers = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        sid, _pid, cmd = parts
        if sid.isdigit() and int(sid) == mine:
            continue
        if any(d in cmd for d in _DRIVERS) and "--plan" not in cmd:
            drivers.append(cmd)
    _rig_busy.drivers = drivers          # so the refusal can SAY what it saw
    return len(drivers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path,
                    default=REPO / "validation_outputs" / "certified_catalogue_2026_09_10" / "hub_snapshot.txt")
    ap.add_argument("--campaigns", type=Path, default=Path("/home/mlops/nbx/campaigns"))
    ap.add_argument("--plan", action="store_true", help="print the order and stop; runs nothing")
    ap.add_argument("--profile", default="volta")
    ap.add_argument("--budget", type=int, default=28800, help="seconds of rig time available")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--src", default=None,
                    help="a frozen worktree's src — required to run, for the same "
                         "reason every campaign needs one")
    import rig_clock as _rc
    _rc.add_argument(ap)
    ap.add_argument("--timeout", type=int, default=5400,
                    help="seconds a single MEET run may take before it is killed "
                         "with its whole process group")
    args = ap.parse_args()

    rows = _plan(args)
    runnable = [r for r in rows if r["container"]]
    absent = [r for r in rows if not r["container"]]

    print(f"{len(rows)} hub models · {len(runnable)} present locally · "
          f"{len(absent)} not runnable tonight")
    print(f"{'#':>3} {'model':<52} {'fam':<11} {'GB':>6} {'known cost':>22}  local")
    budget_left, refused = args.budget, []
    for i, r in enumerate(rows, 1):
        if r["known_s"]:
            cost = f"{r['known_s']:.0f} s (recorded)"
        else:
            cost = "not measured"
        note = r["container"] or r["why"]
        if r["container"] and r["known_s"] and r["known_s"] > budget_left:
            refused.append((r, budget_left))
            note += "  ← REFUSED AT THE DOOR"
        elif r["container"] and r["known_s"]:
            budget_left -= r["known_s"]
        print(f"{i:>3} {r['hub']:<52} {r['family']:<11} {r['gb']:>6.1f} {cost:>22}  {note}")

    print(f"\nBudget: {args.budget} s in, {budget_left:.0f} s left after the cells "
          f"whose cost is known.")
    print("Cells with no known cost consume from that remainder without being "
          "refused in advance: the guard stops a KNOWN cost, it never invents one.")
    if absent:
        gb = sum(r["gb"] for r in absent)
        free = shutil.disk_usage("/").free / 1e9
        print(f"\n{len(absent)} not present locally — {gb:.1f} GB to fetch against "
              f"{free:.1f} GB free on /. They are NOT part of tonight's pass, and "
              f"the report says `not measured` for them rather than a projection:")
        for r in absent:
            print(f"    {r['gb']:>7.2f} GB  {r['hub']:<52} {r['why']}")

    if args.plan:
        print("\n--plan: nothing was run.")
        return 0

    busy = _rig_busy()
    if busy != 0:
        seen = getattr(_rig_busy, "drivers", [])
        what = ("nvidia-smi is absent, so the rig cannot be established free"
                if busy < 0 else
                (f"{busy} driver process(es) alive, about to take a card"
                 if seen else f"{busy} compute process(es) on a card"))
        print(f"\nREFUSED: {what}.", file=sys.stderr)
        for cmd in seen[:3]:
            print(f"    {cmd[:150]}", file=sys.stderr)
        print("    A refusal that does not say what it saw cannot be acted on.",
              file=sys.stderr)
        return 1

    # The rig runs at the protocol clock or it does not measure. Application
    # clocks do not survive a reboot and this rack's two SKUs return to DIFFERENT
    # factory defaults, so after an outage half of it can sit at the protocol
    # value by coincidence — which is how the 2026-09-11 certification ran across
    # cards at 1312 and cards at 1290. The refusal reads every card.
    from rig_clock import OffProtocol, require_protocol_clock
    try:
        require_protocol_clock(
            allow_off_protocol=getattr(args, "allow_off_protocol_clock", False))
    except OffProtocol as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    from precision_zoo_campaign import frozen_src_refusal, request_args, run

    why = frozen_src_refusal(args.src, REPO)
    if why:
        print(f"\nREFUSED: the MEET phase runs the ENGINE, and a shape key "
              f"depends on the engine version — a new constexpr in an autotune "
              f"key unserves the whole directory. So it measures a frozen tree "
              f"like every other campaign: {why}", file=sys.stderr)
        return 1

    out = args.out or (REPO / "validation_outputs" /
                       f"catalogue_meet_{time.strftime('%Y%m%d_%H%M')}")
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

    record, left = [], args.budget
    for r in rows:
        if not r["container"]:
            record.append({**{k: r[k] for k in ("hub", "family", "gb", "why")},
                           "state": "not runnable"})
            continue
        if r["known_s"] and r["known_s"] > left:
            print(f"REFUSED AT THE DOOR  {r['hub']}: known cost "
                  f"{r['known_s']:.0f} s exceeds the {left:.0f} s left")
            record.append({**{k: r[k] for k in ("hub", "family", "gb")},
                           "state": "refused at the door",
                           "known_s": r["known_s"], "left_s": left})
            continue
        # The FLOOR, not the reservation: refusing on the widest arm would refuse
        # Qwen3-VL (a 3135 s sweeping arm, 211 s on the other) for a model that
        # met in 365 s. A run is doomed only when EVERY arm that ended took longer
        # than the clock it is about to be given.
        why_timeout = timeout_refusal(r["floor_s"], args.timeout, r["floor_basis"])
        if why_timeout:
            print(f"REFUSED AT THE DOOR  {r['hub']}: {why_timeout}")
            record.append({**{k: r[k] for k in ("hub", "family", "gb")},
                           "container": r["container"],
                           "state": "refused at the door",
                           "why": "known cost exceeds the run timeout",
                           "known_s": r["known_s"], "floor_s": r["floor_s"],
                           "timeout_s": args.timeout, "basis": r["floor_basis"]})
            continue

        log = out / f"{r['container']}.log"
        # The FAMILY comes from the container, never from the hub listing. The
        # hub's CATEGORY is a shelf label — it carries `CODE`, and the engine
        # has no `code` family, so composing a request from it raised and took
        # the whole pass down after three models on 2026-09-11. The container
        # declares what it is; that is the authority.
        family = r["family"]
        try:
            manifest = json.loads((CACHE / r["container"] / "manifest.json").read_text())
            family = manifest.get("family") or family
        except (OSError, ValueError):
            pass
        try:
            req = request_args(r["container"], family, ["--triton"])
        except Exception as exc:
            # A request that cannot be composed is a NAMED skip, never a crash
            # that ends the pass: forty-three models must not be lost because
            # the forty-fourth has no stimulus.
            print(f"SKIP {r['hub']:<52} no request for family {family!r} "
                  f"({type(exc).__name__})")
            record.append({**{k: r[k] for k in ("hub", "gb")}, "family": family,
                           "container": r["container"], "state": "no request",
                           "reason": f"{type(exc).__name__}: {exc}"})
            continue
        cmd = [str(PY), "-c", "import sys; from neurobrix.cli import main; sys.exit(main())",
               "run", "--model", r["container"], *req]
        env = {**os.environ, "PYTHONPATH": str(args.src),
               "NBX_AUTOTUNE_CERTIFIED": "off"}
        record_family = family
        t0 = time.time()
        rc, wall = run(cmd, env, log, args.timeout)
        left -= wall
        text = log.read_text(errors="replace")
        swept = len(re.findall(r"no certified setting for", text))
        excluded = len(re.findall(r"\[AUTOTUNE_SCREEN\] .*config excluded", text))
        print(f"{'MET ' if rc == 0 else 'FAIL'} {r['hub']:<52} "
              f"{wall:>7.0f} s  swept {swept:>4}  screened-out {excluded:>3}  "
              f"{left:>7.0f} s left")
        record.append({**{k: r[k] for k in ("hub", "gb")}, "family": record_family,
                       "container": r["container"], "state": "met" if rc == 0 else "failed",
                       "rc": rc, "wall_s": wall, "swept": swept,
                       "screened_out": excluded, "src_sha": sha, "log": log.name})
        if left <= 0:
            print("Budget exhausted; the rest is untouched and reads `not measured`.")
            break

    (out / "meet.json").write_text(json.dumps(record, indent=1))
    met = [x for x in record if x.get("state") == "met"]
    print(f"\nMEET: {len(met)} of {len(rows)} models met their shapes. "
          f"Record: {out / 'meet.json'}")
    print("Next: `neurobrix autotune certify --profile "
          f"{args.profile} --only-missing`, then `autotune check`, then the "
          f"catalogue report. The census is the machine's replay cache and it "
          f"ACCUMULATES — it already held 6,277 keys against 5,628 certified "
          f"entries before this phase ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
