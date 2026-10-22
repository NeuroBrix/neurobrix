#!/usr/bin/env python3
"""gate_sched.py — THE gate scheduler, committed: the dynamic card queue of the validation regime.

One scheduler, two instances per gate (the validation regime, Hocine 2026-10-14 14:20, rule 1 and rule 4):
  * the CLASS instance places the gate's cells on the cards of the gate's class (the card's real VRAM, 32g);
  * the ANY-CARD instance (`--out-class <cls>`) places the cells of the models whose Prism plan is IDENTICAL under
    both profiles (tools/plan_listing.py) on the cards of the other class and RECORDS them under the gate's class
    — the row names the card it ran on. A card at 0 % outside loading is a noted fault; the scheduler never waits.
Every path is an argument: nothing of a campaign is written in this file.

The dynamic card queue (supervisor inbox 2026-10-07 20:54, Q4, validated by Hocine).

Each time a card is free it takes, at once, the FIRST ADMISSIBLE task: a gate cell of the card's memory class whose
host need fits now (and, if heavy, no other heavy cell is running), else the next one in the queue (a blocked head
does not hold the card), else a TC task (they fill the gaps). No card waits while something admissible exists.

Inputs (all data, nothing about a model in here):
  * gate queues  <gate2>/queue_<cls>.txt   "model modes timeout [rerun]" — taken over atomically at start (under the
    same flock the gw2 workers use), so a worker finishes its current model, finds its queue empty and exits.
  * card class   from nvidia-smi memory.total against --class-split-mib (the two V100 SKUs of this rack).
  * host need    regression_matrix.cell_need (the plan's own figure, the harness's estimate otherwise), max over modes.
  * admission    need + headroom <= MemAvailable now, AND the needs of every running cell + this one + headroom
    <= host RAM (Q4's intent, one host that is never over-committed, priced by the needs themselves: the old
    "one heavy cell at a time" rule, heavy = need >= --heavy-share x host RAM, held every 32g LLM line (89-169 GiB,
    all "heavy") behind one 108 GiB Wan cell on a 251 GiB host, three cards idle, 2026-10-08 01:13). heavy is now
    only printed. A need costs ~17 s to plan, so it is asked lazily, in queue order, and each one is printed.
  * measured peak  once a cell's host peak is MEASURED (RssAnon+RssFile+RssShmem of its process tree, sampled at
    every reap, recorded in peaks.jsonl at its end), its reservation AND its admission follow that peak, never the
    price above it: an over-bound price is a refusal to place, i.e. idle cards (supervisor 2026-10-11 19:35, Ming-Lite
    16g priced 189.4 GiB, measured ~28 GiB, cards 0 and 2 idle). A running cell reserves max(its recorded peak, the
    peak sampled so far); MemAvailable still gates every admission, so a peak that grows is seen by the host itself.
    The harness's own figure comes first: every gate row carries host_peak_rss, the run's sampled peak per mode, and
    the cell's peak is the max over its modes, read from this gate's rows AND --peak-rows (earlier gates of the same
    class, mq23: 123 rows per class). The sched's own 5 s sample under-reads it (mochi 16g: 40.3 vs 50.8 GiB).
  * TC tasks     <dir>/tc_queue.txt, one shell command per line, run with CUDA_VISIBLE_DEVICES=<card>; a line
    starting "@<cls> " runs only on a card of that class (a 32 GB reference never lands on a 16 GB card).
  * Pace         a finished child is seen within --reap seconds (default 5): the 30 s poll left a card idle up to
    30 s after each cell (supervisor, 21:37: cards 0 and 2 read at 0 %). nvidia-smi is asked only when a card
    is free.
A card is free when nothing of ours runs on it, nvidia-smi lists no compute process, and no other holder's
hold_c<card> exists. Rows are appended to RUN.md as "<label> <cls> <model> card <c> harness=<rc> cells=[...] sched".
"""
from __future__ import annotations

import argparse, fcntl, json, multiprocessing, os, signal, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate2", required=True, help="the gate's lists and queues: gate_<cls>.list, queue_<cls>.txt")
    ap.add_argument("--gate", required=True, help="the gate directory: matrix_<cls>/ rows, carries.yml")
    ap.add_argument("--tree", required=True, help="the frozen tree under test (its tools/regression_matrix.py runs the cells)")
    ap.add_argument("--run-md", required=True, help="the campaign's RUN.md the rows are appended to")
    ap.add_argument("--holds", required=True, help="the holds directory: hold_c<card> marks a card held by another holder")
    ap.add_argument("--class-split-mib", type=int, required=True)
    ap.add_argument("--heavy-share", type=float, required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--reap", type=float, default=5.0)
    ap.add_argument("--need-workers", type=int, required=True)
    ap.add_argument("--peak-rows", action="append", default=[],
                    help="a gate matrix dir pattern with {cls} (e.g. .../mq23/gate/matrix_{cls}) whose rows' host_peak_rss "
                         "price a cell already measured there; this gate's own rows are always read and win")
    ap.add_argument("--adopt", action="append", default=[],
                    help="card:pid:heavy:label — a cell started by an earlier scheduler instance, held and logged here; a label "
                         "starting tc= is a TC task")
    ap.add_argument("--state", required=True,
                    help="this instance's own directory: queue_<cls>.txt taken over, needs.jsonl, peaks.jsonl, tc_queue.txt, logs")
    ap.add_argument("--out-class", default=None,
                    help="the ANY-CARD instance: record every cell under this class (matrix_<cls>, gate_<cls>.list, the "
                         "class's known-cause record) whatever the card's own class")
    ap.add_argument("--label", default="gate2", help="the first word of the rows appended to RUN.md")
    ap.add_argument("--python", default=os.environ.get("NEUROBRIX_PYTHON") or os.environ.get("NBX_PYTHON") or sys.executable,
                    help="the engine python the cells run on (NEUROBRIX_PYTHON / NBX_PYTHON, else this interpreter)")
    return ap


a = None
PY = HERE = G2 = G = W = None
RM = None


def OUT(cls):
    """The class a cell is recorded under: the gate's (--out-class) for the any-card instance, else the card's."""
    return a.out_class or cls


def configure(args):
    """Bind the module to one instance's arguments (the campaign scripts bound them at import)."""
    global a, PY, HERE, G2, G, W, RM
    a = args
    PY = a.python
    HERE = Path(a.state)
    HERE.mkdir(parents=True, exist_ok=True)
    G2, G, W = Path(a.gate2), Path(a.gate), Path(a.tree)
    if RM is None:
        sys.path[:0] = [str(W / "tools"), str(W / "src")]
        import regression_matrix as _rm  # noqa: E402  (the frozen tree's own reader of a cell's host need)
        RM = _rm


def now():
    return time.strftime("%F %T %Z")


def say(msg):
    with open(a.run_md, "a") as f:
        f.write(f"{msg} {now()}\n")
    print(msg, flush=True)


CLASS_LABELS = ("16g", "32g")   # a card's class: below --class-split-mib, or at or above it


def cards():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {int(i): (CLASS_LABELS[0] if int(m) < a.class_split_mib else CLASS_LABELS[1]) for i, m in
            (l.split(",") for l in out.strip().splitlines())}


def busy_on_device(card):
    out = subprocess.run(["nvidia-smi", "-i", str(card), "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def take_over(cls):
    """Move every line of the gw2 queue into ours, under the workers' own lock."""
    q, mine = G2 / f"queue_{cls}.txt", HERE / f"queue_{cls}.txt"
    with open(f"{q}.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = [l for l in q.read_text().splitlines() if l.strip()] if q.exists() else []
        with open(mine, "a") as f:
            f.writelines(l + "\n" for l in lines)
        q.write_text("")
    return len(lines)


NEED, PENDING, POOL = {}, {}, None


PEAK = {}  # (model, modes, cls) -> measured host peak in bytes (peaks.jsonl)


def tree_rss(pid):
    """RssAnon+RssFile+RssShmem of pid and every descendant (the harness starts its run in its own session)."""
    kids, total = {}, 0
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d.name))
    todo = [pid]
    while todo:
        q = todo.pop()
        todo += kids.get(q, [])
        try:
            for l in Path(f"/proc/{q}/status").read_text().splitlines():
                if l.startswith(("RssAnon:", "RssFile:", "RssShmem:")):
                    total += int(l.split()[1]) * 1024
        except OSError:
            continue
    return total


def row_peaks(cls):
    """(model, mode) -> host_peak_rss, latest row per pair, earlier gates first so this gate's rows win."""
    out = {}
    for d in [x.format(cls=cls) for x in a.peak_rows] + [str(G / f"matrix_{OUT(cls)}")]:
        rows = []
        for f in sorted(Path(d).glob("rows_card*.jsonl")):
            for l in f.read_text().splitlines():
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("host_peak_rss"):
                    rows.append(r)
        for r in sorted(rows, key=lambda r: r.get("date", "")):
            out[(r["model"], r["mode"])] = r["host_peak_rss"]
    return out


def refresh_row_peaks(classes, running=()):
    """A cell is priced by its rows only when EVERY one of its modes was measured. `running` holds the
    (model, modes, cls) of the cells on the cards, which have left the queue but still reserve."""
    for cls in classes:
        rp = ROWP[cls] = row_peaks(cls)
        cells = [q.split()[:2] for q in ((HERE / f"queue_{cls}.txt").read_text().splitlines()
                                        if (HERE / f"queue_{cls}.txt").exists() else []) if q.strip()]
        cells += [[m, md] for m, md, c in running if c == cls]
        for model, modes in cells:
            got = [rp.get((model, m)) for m in modes.split(",")]
            old = PEAK.get((model, modes, cls), 0)
            if all(got) and max(old, max(got)) != old:
                PEAK[(model, modes, cls)] = max(old, max(got))
                say(f"sched peak {cls} {model} {PEAK[(model, modes, cls)] / RM.GIB:.1f} GiB from the gate rows' host_peak_rss")


def record_peak(model, modes, cls, peak, src):
    PEAK[(model, modes, cls)] = peak
    with open(HERE / "peaks.jsonl", "a") as fh:
        fh.write(json.dumps({"cls": cls, "model": model, "modes": modes, "peak": peak, "src": src, "at": now()}) + "\n")
    say(f"sched peak {cls} {model} {peak / RM.GIB:.1f} GiB measured ({src})")


def _need(model, modes, card, cls):
    return max(RM.cell_need(model, m, str(card), G / f"matrix_{OUT(cls)}", W / "src")[0] for m in modes.split(","))


def need_of(model, modes, card, cls):
    """A cell's host need, or None while it is still being planned: a plan costs ~17 s, so needs are asked of a
    process pool and a free card never waits for one (it takes the next known line, or a TC task)."""
    key = (model, modes, cls)
    if key in NEED:
        return NEED[key]
    f = PENDING.get(key)
    if f is None:
        PENDING[key] = POOL.submit(_need, model, modes, card, cls)
        return None
    if not f.done():
        return None
    del PENDING[key]
    NEED[key] = f.result()
    with open(HERE / "needs.jsonl", "a") as fh:  # a restart keeps every need already planned (same frozen tree)
        fh.write(json.dumps({"cls": cls, "model": model, "modes": modes, "need": NEED[key]}) + "\n")
    print(f"need {cls} {model} {NEED[key] / RM.GIB:.1f} GiB heavy={NEED[key] >= a.heavy_share * RM._host_bytes()} "
          f"{now()}", flush=True)
    return NEED[key]


def others_reserved() -> int:
    """Host bytes the OTHER schedulers' cells hold reserved: the need each wrote into its card's hold (`take_hold`'s
    file: "<pid> <need>"; a filler's hold names no need). Each instance summed only its own cells: two gates running
    at once admitted a Qwen3-VL each, 169 GiB of host apiece, and the kernel killed one (2026-10-20 14:47). A census
    beside the gate writes its workers' bytes as `hold_host_<pid>` (`derived_census._host_hold`)."""
    total = 0
    for f in list(Path(a.holds).glob("hold_c*")) + list(Path(a.holds).glob("hold_host_*")):   # a census's too
        try:
            parts = f.read_text().split()
        except FileNotFoundError:
            continue
        if len(parts) >= 2 and parts[0] != str(os.getpid()):
            total += int(parts[1])
    return total


def pick_gate(cls, card, reserved):
    mine = HERE / f"queue_{cls}.txt"
    # one admission at a time across every scheduler sharing the holds: a need is read and written under it
    with open(f"{mine}.lock", "a") as lk, open(Path(a.holds) / ".admission.lock", "a") as adm:
        fcntl.flock(lk, fcntl.LOCK_EX)
        fcntl.flock(adm, fcntl.LOCK_EX)
        reserved = reserved + others_reserved()
        lines = [l for l in mine.read_text().splitlines() if l.strip()] if mine.exists() else []
        avail, headroom = RM._mem_available(), int(RM._host_bytes() * RM.HOST_HEADROOM_SHARE)
        for i, line in enumerate(lines):
            model, modes = line.split()[:2]
            need = PEAK.get((model, modes, cls)) or need_of(model, modes, card, cls)
            if need is None:
                continue
            heavy = need >= a.heavy_share * RM._host_bytes()
            if need + headroom > avail or reserved + need + headroom > RM._host_bytes():
                continue
            mine.write_text("".join(l + "\n" for j, l in enumerate(lines) if j != i))
            (Path(a.holds) / f"hold_c{card}").write_text(f"{os.getpid()} {int(need)}\n")   # the need, for the others
            return line, need, heavy
    return None


def pick_tc(cls):
    """The first TC line this card's class may run ("@<cls> cmd" binds a line to a class)."""
    q = HERE / "tc_queue.txt"
    with open(f"{q}.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = [l for l in q.read_text().splitlines() if l.strip()] if q.exists() else []
        for i, l in enumerate(lines):
            if l.startswith("@") and l.split()[0][1:] != cls:
                continue
            q.write_text("".join(x + "\n" for j, x in enumerate(lines) if j != i))
            return l.split(None, 1)[1] if l.startswith("@") else l
        return None


def gate_cmd(card, cls, line):
    model, modes, tmo, *extra = line.split()
    cmd = [PY, str(W / "tools/regression_matrix.py"), "run", "--models", model, "--modes", modes, "--gpu", str(card),
           "--out", str(G / f"matrix_{OUT(cls)}"), "--src", str(W / "src"), "--timeout", tmo,
           "--gate-lists", str(G2 / f"gate_{OUT(cls)}.list"),
           "--known-reds", str(W / "docs" / "reference" / f"known-cause-reds.{OUT(cls)}.yml")]   # the TREE's committed per-class record
    if (G / "carries.yml").is_file():          # the gate's carry record (rule 2 of the regime: cells carried with provenance)
        cmd += ["--carry", str(G / "carries.yml")]
    if extra[:1] == ["rerun"]:
        cmd.append("--rerun")
    return cmd


def cells(cls, card, model):
    p = G / f"matrix_{OUT(cls)}" / f"rows_card{card}.jsonl"
    last = {}
    for l in (p.read_text().splitlines() if p.exists() else []):
        if l.strip().startswith("{"):
            r = json.loads(l)
            if r.get("model") == model:
                last[r["mode"]] = r
    return " ".join(f"{m}:rc{r.get('rc')}/{r.get('wall_s')}s/misses{r.get('misses')}" for m, r in sorted(last.items())) or "NO ROW"


MODES, SEEN, START = {}, {}, {}  # card -> the running gate cell's modes / host peak sampled so far / start (UTC ISO)
LINE, REQUEUED = {}, set()  # card -> the queue line it runs; the (cls, model, mode) cells re-queued once already


def modes_without_a_row(cls, card, model, modes, since):
    """The modes of a finished cell for which the harness wrote NO row since the cell started (a refused or
    crashed harness writes none): the cell was consumed, not judged."""
    p = G / f"matrix_{OUT(cls)}" / f"rows_card{card}.jsonl"
    seen = set()
    for l in (p.read_text().splitlines() if p.exists() else []):
        if l.strip().startswith("{"):
            r = json.loads(l)
            if r.get("model") == model and (r.get("date") or "") >= (since or ""):
                seen.add(r.get("mode"))
    return [m for m in modes.split(",") if m not in seen]


def requeue(cls, line, missing):
    """The row-less modes of a consumed line put back at the head of their class queue, ONCE per cell (the second
    consumption without a row is a wall, not a loop), as a rerun. Returns the line re-queued, or None."""
    model, modes, timeout, *extra = line.split()
    cells = [(cls, model, m) for m in missing if (cls, model, m) not in REQUEUED]
    if not cells:
        return None
    REQUEUED.update(cells)
    new = " ".join([model, ",".join(m for _, _, m in cells), timeout, "rerun"])
    q = HERE / f"queue_{cls}.txt"
    with open(f"{q}.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = [l for l in q.read_text().splitlines() if l.strip()] if q.exists() else []
        q.write_text("".join(l + "\n" for l in [new] + lines))
    return new
ROWP = {}  # cls -> row_peaks(cls), refreshed each loop


def started_utc(pid):
    """A process's start as the rows' UTC ISO form, from /proc (boot time + start ticks)."""
    btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime"))
    ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(btime + ticks / os.sysconf("SC_CLK_TCK")))


def reservation(card, cls, label, priced):
    """A running cell reserves the measured peak of the modes it has NOT finished yet (its harness writes one row per
    mode as it ends: a cell in its triton mode no longer holds its native peak — Wan2.2-I2V 16g: 112.0 GiB native,
    53.1 triton); a mode never measured keeps the price. The peak sampled so far keeps a growing cell honest."""
    modes = MODES.get(card, "").split(",")
    rows = G / f"matrix_{OUT(cls)}" / f"rows_card{card}.jsonl"
    done = set()
    for l in (rows.read_text().splitlines() if rows.exists() else []):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("model") == label and r.get("date", "") >= START.get(card, "~"):
            done.add(r.get("mode"))
    left = [m for m in modes if m not in done] or modes[-1:]
    rp = ROWP.get(cls, {})
    if all(rp.get((label, m)) for m in left):
        return max(max(rp[(label, m)] for m in left), SEEN.get(card, 0))
    known = PEAK.get((label, MODES.get(card, ""), cls))
    return max(known, SEEN.get(card, 0)) if known else max(priced, SEEN.get(card, 0))


class Adopted:
    """A process this instance did not start: alive while /proc holds it; its harness writes its own rows."""
    returncode = "adopted"

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if Path(f"/proc/{self.pid}").exists() else self.returncode


def take_hold(card: int) -> bool:
    """Hold the card for this instance: hold_c<card> CREATED EXCLUSIVELY (O_EXCL), so of two schedulers (two gates
    running at once) that both find a card free, one holds it and the other sees it held — reading the holds and
    the device alone, two gates placed Qwen3-Omni and DeepSeek-Coder on card 2 one second apart (2026-10-20 08:01)
    and the second cell died out of memory. False when another holder has it."""
    try:
        fd = os.open(str(Path(a.holds) / f"hold_c{card}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{os.getpid()}\n")
    return True


def drop_hold(card: int) -> None:
    """Release this instance's hold on the card (never another holder's)."""
    f = Path(a.holds) / f"hold_c{card}"
    try:
        if f.read_text().split()[0] == str(os.getpid()):
            f.unlink()
    except (FileNotFoundError, IndexError):
        pass


def main(argv=None):
    global POOL
    configure(build_parser().parse_args(argv))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    # SPAWNED workers: a forked worker born inside pick_gate's flock inherits the lock's open file description and holds
    # it after the parent closes it — every later pick_gate then waits forever (mq24, 2026-10-10 22:09: 18 h, no cell;
    # lock_proof.py: fork relock BLOCKED, spawn relock OK). A spawned worker inherits no descriptor.
    # each spawned worker binds itself to this instance's arguments: a worker imports this module afresh, and the
    # harness reader (RM) is bound by `configure` — the first port left it None in the workers and the any-card
    # instance died at its first need not yet planned (2026-10-14 20:3x)
    POOL = ProcessPoolExecutor(a.need_workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=configure, initargs=(a,))
    cmap = cards()
    for l in (HERE / "needs.jsonl").read_text().splitlines() if (HERE / "needs.jsonl").exists() else []:
        r = json.loads(l)
        NEED[(r["model"], r["modes"], r["cls"])] = r["need"]
    for l in (HERE / "peaks.jsonl").read_text().splitlines() if (HERE / "peaks.jsonl").exists() else []:
        r = json.loads(l)
        PEAK[(r["model"], r["modes"], r["cls"])] = r["peak"]
    for cls in sorted(set(cmap.values())):
        say(f"sched took over {take_over(cls)} queue lines of {cls}")
    refresh_row_peaks(sorted(set(cmap.values())))
    # every need is asked at once, in the background, the classes interleaved so no class waits behind another's queue
    heads = {cls: [l.split()[:2] for l in (HERE / f"queue_{cls}.txt").read_text().splitlines() if l.strip()]
             for cls in sorted(set(cmap.values())) if (HERE / f"queue_{cls}.txt").exists()}
    for i in range(max(map(len, heads.values()), default=0)):
        for cls, ls in heads.items():
            if i < len(ls):
                need_of(*ls[i], min(c for c, k in cmap.items() if k == cls), cls)
    env = dict(os.environ, NEUROBRIX_PYTHON=PY, NBX_PYTHON=PY, PYTHONNOUSERSITE="1")
    running = {}  # card -> (Popen or Adopted, kind, label, heavy)
    for ad in a.adopt:
        card, pid, heavy, label = ad.split(":")[:4]
        modes = (ad.split(":")[4:] or [""])[0]
        kind = "tc" if label.startswith("tc=") else "gate"
        held = [n for (m, md, c), n in NEED.items() if m == label and c == cmap[int(card)] and (not modes or md == modes)]
        if kind == "gate" and not held:
            sys.exit(f"--adopt {ad}: no recorded need for {label} in {cmap[int(card)]} (needs.jsonl)")
        running[int(card)] = (Adopted(int(pid)), kind, label, heavy == "1", max(held, default=0))
        MODES[int(card)], SEEN[int(card)], START[int(card)] = modes, 0, started_utc(int(pid))
        say(f"sched adopted {kind} {label} card {card} pid {pid} heavy={heavy == '1'}")
    while True:
        for card, (p, kind, label, heavy, _n) in list(running.items()):
            if kind == "gate":
                SEEN[card] = max(SEEN.get(card, 0), tree_rss(p.pid))
            if p.poll() is None:
                continue
            del running[card]
            drop_hold(card)
            cls = cmap[card]
            if kind == "gate":
                if SEEN.get(card):
                    record_peak(label, MODES[card], cls, max(SEEN[card], PEAK.get((label, MODES[card], cls), 0)),
                                f"sampled every {a.reap:.0f} s over the whole cell")
                SEEN.pop(card, None)
                say(f"{a.label} {OUT(cls)} {label} card {card} harness={p.returncode} cells=[{cells(cls, card, label)}] sched")
                missing = modes_without_a_row(cls, card, label, MODES.get(card, ""), START.get(card))
                if missing:
                    back = requeue(cls, LINE.get(card, f"{label} {MODES.get(card, '')} 7200"), missing)
                    say(f"sched {cls} {label} card {card}: no row for {','.join(missing)} since {START.get(card)} "
                        f"(harness rc={p.returncode}) -> " + (f"re-queued once: {back}" if back else "already re-queued once: WALL, left to the report"))
            else:
                say(f"sched tc card {card} rc={p.returncode}: {label[-160:]}")
        refresh_row_peaks(sorted(set(cmap.values())),
                          [(lb, MODES.get(c, ""), cmap[c]) for c, (_p, k, lb, _h, _n) in running.items() if k == "gate"])
        reserved = sum(reservation(card, cmap[card], label, n) if kind == "gate" else n
                       for card, (_p, kind, label, _h, n) in running.items())
        for card, cls in cmap.items():
            if card in running or busy_on_device(card) or (Path(a.holds) / f"hold_c{card}").exists():
                continue
            if not take_hold(card):        # another scheduler took it between the reading and now
                continue
            tc = pick_tc(cls)  # TC work is the absolute priority (inbox 19:51) and short: it goes first on any free card
            if tc:
                log = open(HERE / f"tc_card{card}.log", "a")
                p = subprocess.Popen(["bash", "-c", tc], env=dict(env, CUDA_VISIBLE_DEVICES=str(card)), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                running[card] = (p, "tc", tc, False, 0)
                say(f"sched tc card {card} start: {tc[-160:]}")
                continue
            got = pick_gate(cls, card, reserved)
            if got:
                line, need, heavy = got
                log = open(HERE / f"gate_{OUT(cls)}_card{card}.log", "a")
                p = subprocess.Popen(gate_cmd(card, cls, line), cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                running[card] = (p, "gate", line.split()[0], heavy, need)
                MODES[card], SEEN[card], START[card], LINE[card] = line.split()[1], 0, started_utc(p.pid), line
                reserved += need
                say(f"sched gate {cls} {line.split()[0]} card {card} need {need / RM.GIB:.1f} GiB heavy={heavy} "
                    f"reserved {reserved / RM.GIB:.1f} GiB")
            else:
                drop_hold(card)              # nothing admissible for this card: the hold goes back
        left = sum(len([l for l in (HERE / f"queue_{c}.txt").read_text().splitlines() if l.strip()])
                   for c in set(cmap.values()))
        if not running and not left:
            say("sched done: queues empty")
            return
        t_end = time.time() + a.poll          # the poll re-reads queues and cards; a child's exit is seen at once
        while time.time() < t_end and all(p.poll() is None for p, *_ in running.values()):
            time.sleep(a.reap)


if __name__ == "__main__":
    main()
