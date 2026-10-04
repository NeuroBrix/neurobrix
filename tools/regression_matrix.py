#!/usr/bin/env python3
"""The regression matrix: every catalogue model x every served mode, TODAY, at a size other
than the trace size, each cell's artefact kept for an outside judgment.

The owner's standing directive (2026-09-26): a model that worked before and fails now is a
REGRESSION measured against its last dated proof, not a new bug, and the answer is global —
one table of the catalogue, causes grouped, one fix per cause at the brick. This tool produces
today's half of that table on the machine it runs on (CUDA here, Metal on the Mac); the last
dated proofs are compiled beside it (`last_proofs.json`), and the two are joined by `table`.

One cell = one `neurobrix run` of the model's judged request (`precision_zoo_campaign.
request_args`: the family's calibration section, its media and its bound — the one brick the
batteries and the retrace gate use) in one mode:

    native            the compiled (ATen) engine, the reference arm
    triton            the Triton engine, compiled sequence
    triton-sequential the Triton engine, op by op

at ONE SIZE OTHER THAN THE TRACE SIZE. The judged requests of the language, speech and upscaler
families already differ from their trace extents (a prompt is not 23 tokens, a clip is not the
trace clip, a 448-pixel image is not the upscaler's trace tile). An image or video request
names no size, and the engine then renders at the container's own size — the traced one
(`resolution.container_size`). So those two families get an explicit size: the container's own
height taken to three quarters on the lattice (64 pixels for an image, 32 for a video), the
width kept — a non-square request away from the trace, the one class that catches a swapped or
frozen spatial axis.

Each cell records: rc, wall time, the engine's own execute time, the artefact's path, size and
sha256, the mechanical judgment (`judge_artefact`: degeneracy and geometry for an image, empty
or single-token for a text), the first error line on failure, and the request it ran; the
confirmation run's misses (`misses`, `first_missing_key` — the key a KeyNotCertified named) and
the reference it compared against (`engine`, `tree_dirty` of the certified directory and census,
`certified_dir_override`). A row is reused only when it measured this engine against a clean
reference; a run's every asked cell must have a row at its end. The
CONTENT judgment (an eye for an image, an STT for a WAV, a reader for a text) is written into
the row afterwards by the judge, with the artefact's path, so the table carries links that
open — never a PASS pronounced from rc or from two arms agreeing (R29).

Caches: the Triton compilation cache is the machine's (a kernel compiled is the same kernel);
the replay cache (runtime sweeps of uncertified keys) is owned per CARD, so parallel cards
never write one JSON store at once.

    tools/regression_matrix.py run --models A,B --gpu 1 --out nbx/campaigns/<dated>/matrix
    tools/regression_matrix.py table --out nbx/campaigns/<dated>/matrix [--proofs last_proofs.json]
"""
from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import precision_zoo_campaign as Z  # noqa: E402  the judged request, the output kind
from judge_artefact import image_degeneracy, text_degeneracy, video_degeneracy  # noqa: E402
# The request a cell runs at is DERIVED from the current container's trace, in ONE place the
# certification census imports too: a census and the verification it serves ask for one request.
from trace_request import derived_request, off_trace_size  # noqa: E402

from container_renames import by_current_name, current_name  # noqa: E402

MODES = {"native": [], "triton": ["--triton"], "triton-sequential": ["--triton-sequential"]}
#: Where the containers are: the ENGINE's own door (`NEUROBRIX_CACHE`, then `~/.neurobrix/paths.json`, then the
#: default), because the cells this harness launches read that door. A literal default stood here, so a
#: container the engine finds elsewhere (a NAS mount, for one that does not fit the local disk) was refused as
#: absent before its cell could run (the Mac, 2026-10-04: Qwen3-Coder-30B in place from the mount).
from neurobrix.core import paths as _engine_paths  # noqa: E402

CACHE = _engine_paths.cache_dir()
#: The certified reference a cell reads, under the `--src` tree: the directory and the census tables.
#: A gate compares against the COMMITTED reference — a row says whether the working copy differed.
CERTIFIED_REFERENCE = ("neurobrix/config/autotune", "neurobrix/config/census")
#: The engine's relocation of the certified directory (kernels/autotune_certified.directory()).
CERTIFIED_DIR_ENV = "NEUROBRIX_AUTOTUNE_CERTIFIED_DIR"
DIRTY_PATHS_CAP = 20
#: The exception line a confirmation run's miss leaves in the traceback (`<module>.KeyNotCertified:
#: CERTIFIED-ONLY: ...`). Anchored on the message head: a traceback also prints the source line
#: `# raises KeyNotCertified: never a sweep` of kernels/ops/_configs.py, which is not a miss.
KEY_NOT_CERTIFIED = r"KeyNotCertified: (CERTIFIED-ONLY:[^\n]*)"


def first_error(log: Path) -> str:
    text = log.read_text(errors="replace") if log.exists() else ""
    # The miss comes before the generic errors: the lookup-failed miss is raised `from` an inner
    # KeyError/ValueError whose line is in the same traceback and would otherwise name the cell.
    for pat in (r"(KILLED by SIGKILL[^\n]*)", r"(TIMEOUT after[^\n]*)", r"(KeyNotCertified: CERTIFIED-ONLY[^\n]*)",
                r"(ZERO FALLBACK[^\n]*)",
                r"((?:Runtime|Value|Shape\w*|OutOfMemory|Key|Index)Error[^\n]*)", r"(Traceback[^\n]*)"):
        m = re.findall(pat, text)
        if m:
            return m[-1][:300]
    return text.strip().splitlines()[-1][:300] if text.strip() else ""


def missing_keys(log: Path) -> list:
    """The KeyNotCertified messages of a cell's log, distinct, in order of first appearance — one per
    key the certified directory did not serve (a message printed twice is one miss)."""
    text = log.read_text(errors="replace") if log.exists() else ""
    seen = []
    for msg in re.findall(KEY_NOT_CERTIFIED, text):
        if msg not in seen:
            seen.append(msg)
    return seen


def missing_key_text(msg: str) -> str:
    """The kernel and the key a KeyNotCertified message names — both of its raise sites
    (autotune_certified.refuse_missing, kernels/ops/_configs.py's failed lookup); the message's head
    when neither form reads, never nothing."""
    m = (re.match(r"CERTIFIED-ONLY: no certified setting for ([^()\n]*\([^()\n]*\))", msg)
         or re.match(r"CERTIFIED-ONLY: the certified lookup failed for (\S+ at \([^()\n]*\))", msg))
    return m.group(1) if m else msg[:300]


def engine_sha(tree: Path) -> str:
    """The tree's HEAD, short — "" when git cannot say (never equal to anything)."""
    return subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def reference_state(src: Path) -> dict:
    """Whether the certified reference the cells read differs from the tree's HEAD: `git status
    --porcelain` of the `--src` tree's repository limited to CERTIFIED_REFERENCE. `tree_dirty` is
    True/False, None when git cannot say (the reason in the paths) — None is never clean."""
    p = subprocess.run(["git", "-C", str(src.parent), "status", "--porcelain", "--untracked-files=all", "--",
                        *(str(src / d) for d in CERTIFIED_REFERENCE)], capture_output=True, text=True)
    if p.returncode != 0:
        return {"tree_dirty": None, "tree_dirty_count": None,
                "tree_dirty_paths": [f"git status failed (rc {p.returncode}): {p.stderr.strip()[:160]}"]}
    paths = [l[3:] for l in p.stdout.splitlines() if l.strip()]
    return {"tree_dirty": bool(paths), "tree_dirty_count": len(paths), "tree_dirty_paths": paths[:DIRTY_PATHS_CAP]}


def read_jsonl(path: Path) -> list:
    """Every record of a JSON-lines file the matrix appends to. A LAST line with no newline is a write
    cut mid-append (a kill): said on stderr and skipped, never read as a row. Any other line that does
    not parse is refused by name — a row is never guessed."""
    lines = path.read_text().split("\n")
    tail = lines.pop()                                 # "" when the file ends in a newline
    if tail:
        print(f"[matrix] {path}: the last line ({len(tail)} bytes) has no newline — a write cut mid-append; "
              f"skipped, never read as a row", file=sys.stderr, flush=True)
    recs = []
    for n, line in enumerate(lines, 1):
        try:
            recs.append(json.loads(line))
        except ValueError as exc:
            raise SystemExit(f"REFUSED: {path}:{n} is not a row ({exc}): {line[:120]!r} — only a cut LAST "
                             f"line is skipped")
    return recs


def append_jsonl(path: Path, rec: dict) -> None:
    """One whole line per record, flushed and fsynced before the call returns. A cut last line an
    earlier kill left is moved to `<file>.torn` first (said on stderr): glued to this line it would
    become a malformed line that is no longer last, and the reader would refuse the whole file."""
    with open(path, "ab+") as f:
        end = f.seek(0, os.SEEK_END)
        if end:
            f.seek(end - 1)
            if f.read(1) != b"\n":
                f.seek(0)
                data = f.read()
                keep = data.rfind(b"\n") + 1
                torn = path.with_name(path.name + ".torn")
                with open(torn, "ab") as t:
                    t.write(data[keep:] + b"\n")
                    t.flush()
                    os.fsync(t.fileno())
                f.truncate(keep)
                print(f"[matrix] {path}: a cut last line ({end - keep} bytes) moved to {torn} before this append",
                      file=sys.stderr, flush=True)
        f.write((json.dumps(rec) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def write_atomic(path: Path, text: str) -> None:
    """The whole file or the previous one: written to a temporary beside it, fsynced, then renamed."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def last_stage(log: Path) -> dict:
    """Where a cell was when it ended without an artefact: its last progress line and its last
    line of output. A TIMEOUT row carries it, so a red cell names the stage it held its card in."""
    text = log.read_text(errors="replace") if log.exists() else ""
    text = text.split(Z.STACK_MARK)[0]
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.startswith(("TIMEOUT after", "KILLED by SIGKILL"))]
    progress = [l for l in lines if l.startswith("[progress]")]
    return {"progress": progress[-1][:300] if progress else None, "last_line": lines[-1][:300] if lines else None}


def stack_at_timeout(log: Path) -> list:
    """The engine frames of the main process's first thread in the stack taken before a TIMEOUT
    kill, innermost first — what the cell was DOING when its budget ran out (py-spy's frame
    lines: "    func (path:line)"). Empty when no stack was taken."""
    text = log.read_text(errors="replace") if log.exists() else ""
    if Z.STACK_MARK not in text:
        return []
    section = text.split(Z.STACK_MARK, 1)[1]
    frames, in_thread = [], False
    for line in section.splitlines():
        if line.startswith("Thread "):
            if in_thread:
                break
            in_thread = True
            continue
        if in_thread and line.startswith("    ") and "(" in line:
            frames.append(line.strip())
        elif in_thread and frames and not line.strip():
            break
    engine = [f for f in frames if "neurobrix" in f or "kernels" in f or "triton" in f]
    return (engine or frames)[:12]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mechanical(path: Path, family: str, expect_hw=None) -> dict:
    """The mechanical half of R29 — the content half is the judge's."""
    if not path.exists():
        return {"missing": True}
    suffix = path.suffix.lower()
    if suffix == ".png":
        return image_degeneracy(path, expect_shape=expect_hw)
    if suffix == ".txt":
        return text_degeneracy(path)
    if suffix == ".mp4":
        return video_degeneracy(path, expect_shape=expect_hw)
    return {"path": str(path), "bytes": path.stat().st_size}


@functools.lru_cache(maxsize=None)
def container_bytes(model: str) -> int:
    return sum(f.stat().st_size for f in (CACHE / model).glob("components/*/weights/*"))


#: A cell's host footprint per byte of weights: pinned staging + the arena's host side. Measured
#: 2026-09-26: Qwen3-Coder-30B 87 GB RSS for 57 GB of weights, DeepSeek-Coder-V2-Lite 51 GB for 31,
#: deepseek-moe 43 GB for 32 — 1.35x to 1.65x.
HOST_PER_WEIGHT_BYTE = 1.7
#: The share of the host the matrix may hold at once: the rest belongs to the census, the gate's
#: harness and the kernel. Three concurrent cells at 180 GB of 251 drove memory pressure to 33 %
#: "full" and the gate's cells to their timeouts (2026-09-26).
HOST_SHARE = 0.8
#: Kept free beyond every running cell's owed growth (the kernel, the census, the gate harness), as a
#: share of the host: 16 GiB on the rack's 256 GB, 1.5 GiB on a 24 GB Mac. A constant 16 GiB is more
#: than a 24 GB host ever has available, so no cell was ever admitted there (the Mac, 2026-10-03).
HOST_HEADROOM_SHARE = 1 / 16
#: How often a running cell's host footprint is sampled for its peak.
PEAK_SAMPLE_S = 1.0
#: How often a cell the host cannot admit yet says so (at its first refusal, then every WAIT_SAY_S): a wait
#: that writes nothing holds the card with no trace (the Mac, 2026-10-03 22:34-22:36, Voxtral).
WAIT_SAY_S = 300.0
GIB = float(1 << 30)


def _host_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _mem_available() -> int:
    """The host's available bytes from the ENGINE's one reader (`core.host_memory.memory_state`:
    MemAvailable on Linux, vm_stat + sysctl on macOS) — the harness read /proc/meminfo itself and
    every cell was refused on the Mac, where no /proc exists (the Mac, 2026-09-29 16:02)."""
    from neurobrix.core.host_memory import memory_state
    st = memory_state()
    if st.available_mb is None:
        raise SystemExit(f"the host's available memory cannot be read ({st.source}): the host budget cannot be measured")
    return int(st.available_mb) << 20


def _rss_tree(pid: int) -> int:
    """Resident bytes of a runner and every process under it (its cell) — psutil, which every host
    answers (it read /proc/<pid>/status and /proc/<pid>/task/<pid>/children, Linux only)."""
    import psutil
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            continue                                    # ended between the listing and the read
    return total


def _pid_alive(pid: int) -> bool:
    """Whether a reservation's process still exists — `os.kill(pid, 0)`, which every POSIX host
    answers. It read `/proc/<pid>` before, which macOS does not have: on the Mac every
    reservation was pruned as dead the moment another process read the ledger, so two cells
    over half the budget both reserved (test_the_matrix_budgets_the_host, red 2026-09-28)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                    # exists, not ours
    return True


def _children_rss() -> int:
    """Resident bytes of every process this runner started (its cell's process tree), the runner excluded."""
    import psutil
    try:
        kids = [c.pid for c in psutil.Process(os.getpid()).children()]
    except psutil.Error:
        return 0
    return sum(_rss_tree(k) for k in kids)


class PeakRSS:
    """The peak of `_children_rss()` while it runs, sampled every PEAK_SAMPLE_S — a cell's host peak,
    written in its row. It is the PROOF of the host estimate, never its source (the owner, 2026-09-27
    14:27): the engine treats technologies, not model names, so a reservation is what the plan the
    engine chose says the run will hold on the host — a per-model table of peaks is refused."""

    def __init__(self, interval: float = PEAK_SAMPLE_S):
        import threading
        self.peak, self._interval = 0, interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while True:
            self.peak = max(self.peak, _children_rss())
            if self._stop.wait(self._interval):
                return

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _children_rss())


def _ledger(out: Path, change):
    """Read-modify-write the host reservations {pid: bytes} under an exclusive flock, dead pids pruned."""
    path = out / "host_ledger.json"
    with open(out / "host_ledger.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            led = json.loads(path.read_text()) if path.exists() else {}
            led = {p: n for p, n in led.items() if _pid_alive(int(p))}
            result = change(led)
            path.write_text(json.dumps(led))
            return result
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def reserve_host(out: Path, need: int, headroom: Optional[int] = None) -> bool:
    """`headroom`: what the host must keep free beyond the cell and the owed growth; the host's
    HOST_HEADROOM_SHARE by default. 0 for a cell priced by a plan that already drew its device memory
    from this host's free reading (unified memory: Prism's ladder rounding IS the margin, the owner's
    rule) — one margin, not two (the Mac, 2026-10-03 23:59)."""
    budget = int(_host_bytes() * HOST_SHARE)
    if headroom is None:
        headroom = int(_host_bytes() * HOST_HEADROOM_SHARE)

    def take(led):
        if need > budget:
            raise SystemExit(f"a cell needing {need >> 30} GiB of host exceeds the matrix's whole budget "
                             f"({budget >> 30} GiB): refused by name")
        if sum(led.values()) + need > budget:
            return False
        # MEASURED as well as reserved: a running cell owes at most its reservation minus what it
        # already holds; the new cell starts only if the host's available memory covers it, that
        # owed growth, and a headroom. Reservations alone held three cards idle at 22:57 with
        # 201 GB available (2026-09-26); measurement alone let three 30B loads OOM the host at 18:40.
        owed = sum(max(0, n - _rss_tree(int(p))) for p, n in led.items())
        if _mem_available() < need + owed + headroom:
            return False
        led[str(os.getpid())] = need
        return True

    def in_turn(led):
        # IN ARRIVAL ORDER: a cell that could not be admitted waits in `host_waiting.json` {pid: since},
        # and no later cell is admitted past a living earlier waiter. Without it a cell needing most of
        # the budget never saw the ledger empty — smaller cells kept slipping in: Wan2.2-I2V-A14B
        # (200 of 201 GiB) waited 1 h 46 min on card 0 with both gates running (2026-09-27). The
        # queue is its own file so a runner still on the older code keeps reading a plain ledger.
        wpath = out / "host_waiting.json"
        waiting = json.loads(wpath.read_text()) if wpath.exists() else {}
        waiting = {p: t for p, t in waiting.items() if _pid_alive(int(p))}
        me = str(os.getpid())
        mine = waiting.get(me, time.time())
        ahead = [p for p, t in waiting.items() if p != me and t < mine]
        admitted = not ahead and take(led)
        if admitted:
            waiting.pop(me, None)
        else:
            waiting[me] = mine
        wpath.write_text(json.dumps(waiting))
        return admitted
    return _ledger(out, in_turn)


def release_host(out: Path) -> None:
    _ledger(out, lambda led: led.pop(str(os.getpid()), None))


def cell_request(model: str) -> list:
    """The request a cell runs, and the one its plan is asked for: THE shared derivation
    (tools/trace_request.derived_request — the family's confirmation request at its confirmation
    size), the same the census takes, so the table holds what a cell forms."""
    return derived_request(model, Z.family_of(model))


def plan_host_need(model: str, mode: str, gpu: str, src: Path):
    """(bytes, "plan") — the host footprint the ENGINE's plan states for this request on this card
    (`plan.host_footprint.total_bytes`, Prism's host estimate, the owner's 14:27 rule: a reservation is
    what the plan the engine chose says the run will hold), or None when the tree under test prints
    no such figure (a tree before prism-prices-the-host). Asked with `--explain-plan --json` in the
    cell's own environment; nothing runs."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(src), "PYTHONNOUSERSITE": "1"}
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model, *cell_request(model), *MODES[mode],
           "--explain-plan", "--json"]
    def _none(why):
        # Said, never silent: the cell then reserves the static estimate and the reason is in its log.
        print(f"[matrix] {model} {mode}: no plan host figure ({why}) — the static estimate is reserved", flush=True)
        return None
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return _none("the plan query timed out at 900 s")
    t = p.stdout
    try:
        doc = json.loads(t[t.index("{"):])
    except ValueError:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
        return _none(f"rc {p.returncode}, no plan JSON: {tail[0][:160]}")
    hf = (doc.get("plan", doc) or {}).get("host_footprint") or {}
    total = hf.get("total_bytes")
    if not (isinstance(total, int) and total > 0):
        return _none("the tree states no host_footprint")
    # A plan whose device memory comes out of the host (unified) was sized against this host's free
    # reading, its margin already taken by the ladder: the harness keeps no second one.
    return int(total), ("plan-unified" if int(hf.get("device_bytes") or 0) > 0 else "plan")


def run_cell(model: str, mode: str, gpu: str, out: Path, timeout: int, src: Path, wait: bool = True):
    """The cell's row; None when the host budget cannot take it now and `wait` is False (the card
    runs its other cells meanwhile and comes back). A pause file (`<out>/PAUSE`, written while a
    gate runs — nothing runs beside a gate) holds every new cell.

    The reservation is the plan's own host figure when the tree states one, the static estimate
    (container bytes x HOST_PER_WEIGHT_BYTE) otherwise — written in the row either way
    (`host_reserved_from`). Two static reservations held a gate's card idle behind 162 GB of
    estimate while the host used 17 GB (2026-09-28 07:49)."""
    planned = plan_host_need(model, mode, gpu, src)
    if planned is None:
        # A tree that states no host figure (one before prism-prices-the-host) may be PRICED by another
        # tree named in `<out>/price_src` — only one whose plans are proven identical to it (a plan
        # census, both engines), so the figure is the host footprint of the very plan this cell runs.
        # Said in the row as `plan@<tree>`.
        pf = out / "price_src"
        if pf.exists():
            ptree = Path(pf.read_text().strip())
            got = plan_host_need(model, mode, gpu, ptree / "src")
            if got:
                planned = (got[0], f"{got[1]}@{ptree.name}")
    need, need_from = planned if planned else (int(container_bytes(model) * HOST_PER_WEIGHT_BYTE), "estimate")
    said_at, waiting_since = None, time.time()
    while True:
        while (out / "PAUSE").exists():
            time.sleep(30)
        if (reserve_host(out, need, headroom=0) if need_from.startswith("plan-unified")
                else reserve_host(out, need)):
            break
        if not wait:
            return None
        if said_at is None or time.time() - said_at >= WAIT_SAY_S:
            said_at = time.time()
            print(f"[matrix] {model} {mode}: waits for host admission since "
                  f"{time.strftime('%H:%M:%S', time.localtime(waiting_since))} — need {need / GIB:.1f} GiB "
                  f"({need_from}), available {_mem_available() / GIB:.1f} GiB, headroom "
                  f"{_host_bytes() * HOST_HEADROOM_SHARE / GIB:.1f} GiB; every {WAIT_SAY_S:.0f} s until admitted",
                  flush=True)
        time.sleep(30)
    try:
        print(f"[matrix] {model} {mode}: {need >> 30} GiB of host reserved ({need_from})", flush=True)
        with PeakRSS() as peak:
            row = _run_cell(model, mode, gpu, out, timeout, src)
        row.update(host_peak_rss=peak.peak, host_reserved=need, host_reserved_from=need_from)
        return row
    finally:
        release_host(out)


def _run_cell(model: str, mode: str, gpu: str, out: Path, timeout: int, src: Path) -> dict:
    family = Z.family_of(model)
    req = cell_request(model)
    size = off_trace_size(model, family)
    ext = Z.output_ext(family, req)
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    art = d / f"{mode}{ext}"
    log = d / f"{mode}.log"
    if art.exists():
        art.unlink()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(src),
           "NEUROBRIX_REPLAY_CACHE": str(out / f"replay_card{gpu}"), "PYTHONNOUSERSITE": "1"}
    # The cell runs under the interpreter the matrix was launched with (the pinned engine
    # python), written in the row: a matrix measures ONE stack, and the stack is part of the cell.
    # Every cell is a CONFIRMATION run (the owner's method, 2026-09-28 20:06): served entirely from
    # the certified directory, a missing key an error naming its census row — never a runtime sweep.
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model, *req, *MODES[mode],
           "--certified-only", "--output", str(art)]
    rc, wall = Z.run(cmd, env, log, timeout, stack_at_timeout=True)
    tree = src.parent
    misses = missing_keys(log)
    row = {"model": model, "family": family, "mode": mode, "gpu": gpu, "rc": rc,
           "wall_s": round(wall, 1), "exec_s": Z.exec_time(log), "request": req,
           "off_trace_size": list(size) if size else None, "log": str(log),
           "python": sys.executable,
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "engine": engine_sha(tree),
           "engine_tree": str(tree),
           # the reference the cell compared against: the tree's working copy of the certified
           # directory and census (dirty or not), and any relocation of the directory it inherited
           **reference_state(src),
           "certified_dir_override": env.get(CERTIFIED_DIR_ENV) or None,
           # a confirmation run's misses: the key and its census row, from the KeyNotCertified line
           "misses": len(misses),
           "first_missing_key": missing_key_text(misses[0]) if misses else None}
    if art.exists() and rc == 0:
        row.update(artefact=str(art), sha256=sha256(art), bytes=art.stat().st_size,
                   mechanical=mechanical(art, family, size))
    else:
        row["error"] = first_error(log) if rc != 0 else "rc 0 and no artefact"
        row["last_stage"] = last_stage(log)
        stack = stack_at_timeout(log)
        if stack:
            row["stack_at_timeout"] = stack
    return row


def full_matrix(cache: Path = None) -> set:
    """Every cell the matrix holds: each container of the shared cache (a directory with a manifest)
    in every mode — derived from the cache, never a hand-kept list."""
    cache = cache or CACHE
    return {(d.name, mode) for d in cache.iterdir() if (d / "manifest.json").exists() for mode in MODES}


def cells_of_lists(paths) -> set:
    """The cells a gate's card lists name ("<model> <mode>,<mode>" per line)."""
    cells = set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                model, modes = line.split()
                cells |= {(model, m) for m in modes.split(",")}
    return cells


def refuse_a_partial_gate(lists, cache: Path = None) -> None:
    """A queue's gate covers EVERY cell of the matrix (the supervisor, 2026-09-27 16:25): queue-9's
    gate ran lists inherited from queue-8 and never ran Sana_1600M_4Kpx_BF16 native — a regression
    it would have seen landed on main. Refused by name when the union of the gate's lists is not
    the full matrix."""
    full, named = full_matrix(cache), cells_of_lists(lists)
    if not full:
        # An empty cache is an empty matrix, and empty lists "cover" it: a gate over no cell proves nothing.
        raise SystemExit(f"REFUSED: the cache {cache or CACHE} holds no container — a gate over an empty "
                         f"matrix proves nothing.")
    missing, unknown = sorted(full - named), sorted(named - full)
    if missing or unknown:
        raise SystemExit(
            f"REFUSED: this gate's cell lists are not the matrix's full list — {len(missing)} of "
            f"{len(full)} cells missing{': ' + ', '.join(f'{m}/{mo}' for m, mo in missing[:8]) if missing else ''}"
            f"{'; unknown: ' + ', '.join(f'{m}/{mo}' for m, mo in unknown[:8]) if unknown else ''}. "
            f"A gate spares no cell.")


def refuse(why: str) -> int:
    """A run refused by name before any cell: exit 2."""
    print(f"REFUSED: {why}", file=sys.stderr, flush=True)
    return 2


def cmd_run(a) -> int:
    if getattr(a, "gate_lists", None):
        refuse_a_partial_gate(a.gate_lists)
    # The request, refused by name before any cell: an empty list is not a run, and a name the cache
    # does not hold is not a cell (it would be judged by its absence).
    models = [m.strip() for m in a.models.split(",")]
    modes = [m.strip() for m in a.modes.split(",")]
    blank = [i for i, m in enumerate(models, 1) if not m]
    if blank:
        return refuse(f"--models {a.models!r}: entry {', '.join(map(str, blank))} names no model")
    absent = [m for m in models if not (CACHE / m / "manifest.json").exists()]
    if absent:
        return refuse(f"--models: {', '.join(absent)} — no container of that name in the cache {CACHE}")
    blank = [i for i, m in enumerate(modes, 1) if not m]
    if blank:
        return refuse(f"--modes {a.modes!r}: entry {', '.join(map(str, blank))} names no mode")
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        return refuse(f"--modes: {', '.join(unknown)} — not a mode of the matrix ({', '.join(MODES)})")
    override = os.environ.get(CERTIFIED_DIR_ENV) or None
    if override and not getattr(a, "allow_certified_dir_override", False):
        return refuse(f"{CERTIFIED_DIR_ENV}={override} is set: every cell would read that directory, not the "
                      f"committed one of --src. A gate compares against the committed reference; unset it, or "
                      f"pass --allow-certified-dir-override for a run that is not a gate.")
    asked = list(dict.fromkeys((m, mode) for m in models for mode in modes))
    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows_card{a.gpu}.jsonl"
    # The matrix is one: a cell with a row on ANY card is done, so a model's remaining cells can be
    # handed to another card without running its finished ones twice.
    latest = {}
    for f in out.glob("rows_card*.jsonl"):
        for r in read_jsonl(f):
            latest[(r["model"], r["mode"])] = r if _row_time(r) >= _row_time(latest.get((r["model"], r["mode"]))) \
                else latest[(r["model"], r["mode"])]
    # --rerun: the listed cells run again although they have a row; the new row names the one it
    # supersedes (kept, never deleted — the supervisor's rule of 2026-09-27 02:57).
    # Without it, an earlier row is done only when it measured THIS engine against a clean reference:
    # its engine sha is this run's and neither its tree nor this one had the certified reference dirty.
    here, here_state = engine_sha(src.parent), reference_state(src)
    done = set()
    for cell in asked:
        prev = latest.get(cell)
        if a.rerun or prev is None:
            continue
        if not here:
            why = "this run's engine sha is unknown"
        elif prev.get("engine") != here:
            why = f"its engine {prev.get('engine') or 'unknown'} is not this run's {here}"
        elif prev.get("tree_dirty") is not False:
            why = f"its tree's certified reference was {'dirty' if prev.get('tree_dirty') else 'unrecorded'}"
        elif here_state["tree_dirty"] is not False:
            why = f"this tree's certified reference is {'dirty' if here_state['tree_dirty'] else 'unreadable'}"
        else:
            done.add(cell)
            continue
        print(f"[matrix] {cell[0]} {cell[1]}: the earlier row ({prev.get('date')}) is not reused — {why}; "
                  f"the cell runs again", flush=True)
    todo = [cell for cell in asked if cell not in done]
    while todo:
        deferred = []
        for i, (model, mode) in enumerate(todo):
            # A cell the host budget cannot take now is deferred while this pass has cells left;
            # the pass's last cell waits for the budget (and the deferred ones come next pass).
            row = run_cell(model, mode, a.gpu, out, a.timeout, src, wait=i == len(todo) - 1)
            if row is None:
                deferred.append((model, mode))
                continue
            prev = latest.get((model, mode))
            if prev is not None:
                row["supersedes"] = {k: prev.get(k) for k in ("date", "gpu", "rc", "wall_s", "error", "engine", "sha256")}
            append_jsonl(rows_path, row)
            print(f"[matrix] {model} {mode} rc={row['rc']} {row.get('wall_s')}s "
                  f"{row.get('error', '')[:120]}", flush=True)
        todo = deferred
    # Every cell asked for has a row in the matrix — read back from the files, never from this loop's
    # own bookkeeping: a cell with no row is an error naming it.
    have = {(r["model"], r["mode"]) for f in out.glob("rows_card*.jsonl") for r in read_jsonl(f)}
    lost = [cell for cell in asked if cell not in have]
    if lost:
        print(f"ERROR: {len(lost)} cell(s) asked for have no row in {out}: "
              f"{', '.join(f'{m}/{mo}' for m, mo in lost)}", file=sys.stderr, flush=True)
        return 1
    return 0


def _row_time(r) -> float:
    """A row's own time (its UTC `date`), 0 for none."""
    if not r or not r.get("date"):
        return 0.0
    import calendar
    return float(calendar.timegm(time.strptime(r["date"], "%Y-%m-%dT%H:%M:%SZ")))


def _judgment_time(j) -> float:
    """A judgment's time: `judge` writes local time with its zone name (CEST/CET on this fleet)."""
    import calendar
    stamp, _, zone = j["date"].rpartition(" ")
    offset = {"CEST": 2, "CET": 1, "UTC": 0, "GMT": 0}.get(zone)
    if offset is None:
        raise SystemExit(f"judgment date {j['date']!r}: unknown zone {zone!r}")
    fmt = "%Y-%m-%d %H:%M:%S" if stamp.count(":") == 2 else "%Y-%m-%d %H:%M"   # a judge may stamp seconds
    return float(calendar.timegm(time.strptime(stamp, fmt)) - offset * 3600)


def load_rows(out: Path) -> list:
    """The LATEST row of every cell (a re-run supersedes, the older rows ride along under
    `superseded`), with the latest outside judgment of that (model, mode) merged in — only a judgment
    taken after the row it would judge: a re-run cell is pending until judged again."""
    latest, older = {}, {}
    for f in sorted(out.glob("rows_card*.jsonl")):
        for r in read_jsonl(f):
            # A row keeps the name its container had when it ran; it is one cell with today's.
            if current_name(r["model"]) != r["model"]:
                r["recorded_as"], r["model"] = r["model"], current_name(r["model"])
            k = (r["model"], r["mode"])
            if k in latest and _row_time(r) < _row_time(latest[k]):
                older.setdefault(k, []).append(r)
                continue
            if k in latest:
                older.setdefault(k, []).append(latest[k])
            latest[k] = r
    judged = {}
    jf = out / "judgments.jsonl"
    if jf.exists():
        for j in read_jsonl(jf):
            judged[(current_name(j["model"]), j["mode"])] = j
    rows = []
    for k, r in latest.items():
        if k in older:
            r["superseded"] = sorted(older[k], key=_row_time)
        j = judged.get(k)
        if j and _judgment_time(j) >= _row_time(r) - 60:        # judge stamps to the minute
            r.update(judged=j["judged"], verdict=j["verdict"], judged_on=j["date"])
        rows.append(r)
    return rows


def cmd_judge(a) -> int:
    """Record the OUTSIDE judgment of one cell (R29): the instrument and what it saw, and the verdict."""
    if a.verdict not in ("works", "broken") and not a.verdict.startswith("not-runnable-here("):
        raise SystemExit("verdict: works | broken | not-runnable-here(<reason>)")
    rec = {"model": a.model, "mode": a.mode, "judged": a.judged, "verdict": a.verdict,
           "date": time.strftime("%Y-%m-%d %H:%M %Z")}
    append_jsonl(Path(a.out) / "judgments.jsonl", rec)
    print(json.dumps(rec))
    return 0


def cmd_table(a) -> int:
    out = Path(a.out)
    rows = load_rows(out)
    proofs = by_current_name(json.loads(Path(a.proofs).read_text())
                             if a.proofs and Path(a.proofs).exists() else {})
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["mode"]] = r
    lines = ["| model | family | last proof | native | triton | triton-sequential |", "|---|---|---|---|---|---|"]
    for model in sorted(by):
        lp = (proofs.get(model) or {}).get("last_proof") or {}
        cells = []
        for mode in MODES:
            r = by[model].get(mode)
            if r is None:
                cells.append("not run")
            elif r["rc"] != 0:
                cells.append(f"rc {r['rc']}: {r.get('error', '')[:80]}")
            else:
                mech = r.get("mechanical") or {}
                cells.append(("DEGENERATE " + "; ".join(mech.get("reasons", []))[:80]) if mech.get("degenerate")
                             else f"ran {r['wall_s']} s, {r.get('verdict', 'pending')}: {r.get('judged', 'not judged')}")
        lines.append(f"| {model} | {by[model][next(iter(by[model]))]['family']} | "
                     f"{lp.get('date') or '—'} {lp.get('verdict') or ''} | " + " | ".join(cells) + " |")
    write_atomic(out / "table.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def catalogue_repo_ids(path: Path) -> dict:
    """cache directory -> repository id, from the catalogue's table (the repository id is the
    name; the directory is the 'cache directory if it differs' column, else the manifest's
    model_name, else the repository's basename — the first that exists in the cache)."""
    out = {}
    for line in path.read_text().splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 9 or not cols[0].isdigit():
            continue
        repo = cols[1].strip("`")
        for cand in (cols[8].strip("`"), cols[2], repo.split("/")[-1]):
            if cand and (CACHE / cand).is_dir():
                out[cand] = repo
                break
    return out


def cmd_export(a) -> int:
    """The joint row schema agreed with the Mac on the peer channel (2026-09-26 17:42 CEST): one
    JSON line per (model, stack, mode). 'native' is written 'compiled'; the container's sha256 is
    its manifest.json's; the verdict is the judge's, 'pending' until an outside judgment is written."""
    out = Path(a.out)
    repos = catalogue_repo_ids(Path(a.catalogue))
    proofs = by_current_name(json.loads(Path(a.proofs).read_text())
                             if a.proofs and Path(a.proofs).exists() else {})
    rows = load_rows(out)
    lines = []
    for r in rows:
        manifest = CACHE / r["model"] / "manifest.json"
        judged = r.get("judged")
        # A failed cell is 'pending' until judged: a harness cause (a timeout, an input not fed,
        # a card too small) is 'not-runnable-here(<reason>)', never 'broken' by its rc alone.
        verdict = r.get("verdict") or "pending"
        lp = (proofs.get(r["model"]) or {}).get("last_proof")
        if lp is not None and "stack" not in lp:
            # A proof whose python is unread is no regression baseline: the batteries ran on the
            # old venv until 2026-09-26 (the zoo brick), and an ATen DAG is not stable across torch.
            lp = {**lp, "stack": "unknown"}
        lines.append({
            "repo_id": repos.get(r["model"]),
            "container": r["model"],
            "container_sha256": sha256(manifest) if manifest.exists() else None,
            "stack": a.stack,
            "python": r.get("python"),
            "mode": "compiled" if r["mode"] == "native" else r["mode"],
            "request": " ".join(r["request"]),
            "today": {"date": r["date"], "engine": r["engine"], "rc": r["rc"],
                      "artifact": r.get("artefact"), "judged": judged, "verdict": verdict,
                      "error": r.get("error")},
            "last_proof": lp,
            "regression": r.get("regression"),
            "bisect": r.get("bisect"),
            "cause_class": r.get("cause_class"),
        })
    dest = Path(a.dest)
    write_atomic(dest, "".join(json.dumps(x) + "\n" for x in lines))
    unnamed = sorted({x["container"] for x in lines if x["repo_id"] is None})
    print(f"{len(lines)} rows -> {dest}; containers without a catalogue line: {unnamed or 'none'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", required=True)
    r.add_argument("--gpu", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--modes", default=",".join(MODES))
    r.add_argument("--gate-lists", nargs="+", default=None,
                   help="this run is a queue gate: the card lists of the whole gate; refused unless together they name every cell of the matrix")
    r.add_argument("--rerun", action="store_true",
                   help="run the listed cells although they have a row; the new row supersedes the old one")
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--src", default=str(REPO / "src"), help="the engine tree's src the runs import (a frozen worktree)")
    r.add_argument("--allow-certified-dir-override", action="store_true",
                   help=f"run although {CERTIFIED_DIR_ENV} relocates the certified directory (never a gate: "
                        f"a gate compares against the committed reference); the row records the override")
    t = sub.add_parser("table")
    t.add_argument("--out", required=True)
    t.add_argument("--proofs", default=None)
    j = sub.add_parser("judge", help="record a cell's outside judgment (R29)")
    j.add_argument("--out", required=True)
    j.add_argument("--model", required=True)
    j.add_argument("--mode", required=True, choices=list(MODES))
    j.add_argument("--judged", required=True, help="the instrument and what it saw")
    j.add_argument("--verdict", required=True)
    e = sub.add_parser("export", help="the joint row schema shared with the Mac's Metal half")
    e.add_argument("--out", required=True)
    e.add_argument("--catalogue", required=True, help="the Mac's CATALOGUE.md")
    e.add_argument("--proofs", default=None)
    e.add_argument("--dest", required=True)
    e.add_argument("--stack", required=True, choices=("cuda", "metal"),
                   help="the machine's stack, written in every row (the joint table has two halves)")
    a = ap.parse_args()
    return {"run": cmd_run, "table": cmd_table, "export": cmd_export, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
