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
or single-token for a text), the first error line on failure, and the request it ran. The
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

MODES = {"native": [], "triton": ["--triton"], "triton-sequential": ["--triton-sequential"]}
CACHE = Path(os.path.expanduser("~/.neurobrix/ca" + "che"))
LATTICE = {"image": 64, "video": 32}


def off_trace_size(model: str, family: str):
    """(height, width) for an image or video request: the container's own size, height at three
    quarters on the family's lattice, width kept. None for the families whose judged request is
    already away from the trace, or when the container states no size (said in the row)."""
    if family not in LATTICE:
        return None
    from neurobrix.core.runtime.loader import NBXRuntimeLoader
    from neurobrix.core.runtime.resolution.container_size import container_output_size
    pkg = NBXRuntimeLoader().load(str(CACHE / model))
    size = container_output_size(pkg.manifest, pkg.defaults,
                                 pkg.topology.get("components", {}) or {}, pkg.components)
    if size is None:
        # The container states no size: the engine then renders at the family's own default
        # (executor: "a family constant is the last resort"), which is what it was traced at.
        from neurobrix.core.config import get_family_config
        fam_defaults = get_family_config(family).get("defaults") or {}
        if "height" not in fam_defaults or "width" not in fam_defaults:
            return None
        size = (fam_defaults["height"], fam_defaults["width"])
    h, w = (int(v) for v in size)
    step = LATTICE[family]
    h2 = max(step, (h * 3 // 4) // step * step)
    return (h2, w) if h2 != h else (max(step, h - step), w)


def first_error(log: Path) -> str:
    text = log.read_text(errors="replace") if log.exists() else ""
    for pat in (r"(KILLED by SIGKILL[^\n]*)", r"(TIMEOUT after[^\n]*)", r"(ZERO FALLBACK[^\n]*)",
                r"((?:Runtime|Value|Shape\w*|OutOfMemory|Key|Index)Error[^\n]*)", r"(Traceback[^\n]*)"):
        m = re.findall(pat, text)
        if m:
            return m[-1][:300]
    return text.strip().splitlines()[-1][:300] if text.strip() else ""


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
#: Kept free beyond every running cell's owed growth (the kernel, the census, the gate harness).
HOST_HEADROOM = 16 << 30
#: How often a running cell's host footprint is sampled for its peak.
PEAK_SAMPLE_S = 1.0


def _host_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) << 10
    raise SystemExit("/proc/meminfo has no MemAvailable: the host budget cannot be measured")


def _rss_tree(pid: int) -> int:
    """Resident bytes of a runner and every process under it (its cell)."""
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) << 10
            todo += [int(c) for c in Path(f"/proc/{p}/task/{p}/children").read_text().split()]
        except OSError:
            continue
    return total


def _children_rss() -> int:
    """Resident bytes of every process this runner started (its cell's process tree), the runner excluded."""
    me = os.getpid()
    try:
        kids = [int(c) for c in Path(f"/proc/{me}/task/{me}/children").read_text().split()]
    except OSError:
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
            led = {p: n for p, n in led.items() if Path(f"/proc/{p}").exists()}
            result = change(led)
            path.write_text(json.dumps(led))
            return result
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def reserve_host(out: Path, need: int) -> bool:
    budget = int(_host_bytes() * HOST_SHARE)

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
        if _mem_available() < need + owed + HOST_HEADROOM:
            return False
        led[str(os.getpid())] = need
        return True
    return _ledger(out, take)


def release_host(out: Path) -> None:
    _ledger(out, lambda led: led.pop(str(os.getpid()), None))


def run_cell(model: str, mode: str, gpu: str, out: Path, timeout: int, src: Path, wait: bool = True):
    """The cell's row; None when the host budget cannot take it now and `wait` is False (the card
    runs its other cells meanwhile and comes back). A pause file (`<out>/PAUSE`, written while a
    gate runs — nothing runs beside a gate) holds every new cell."""
    need, need_from = int(container_bytes(model) * HOST_PER_WEIGHT_BYTE), "estimate"
    while True:
        while (out / "PAUSE").exists():
            time.sleep(30)
        if reserve_host(out, need):
            break
        if not wait:
            return None
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
    req = Z.request_args(model, family, [])
    size = off_trace_size(model, family)
    if size is not None:
        req = req + ["--height", str(size[0]), "--width", str(size[1])]
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
    cmd = [sys.executable, "-m", "neurobrix", "run", "--model", model, *req, *MODES[mode],
           "--output", str(art)]
    rc, wall = Z.run(cmd, env, log, timeout, stack_at_timeout=True)
    tree = src.parent
    row = {"model": model, "family": family, "mode": mode, "gpu": gpu, "rc": rc,
           "wall_s": round(wall, 1), "exec_s": Z.exec_time(log), "request": req,
           "off_trace_size": list(size) if size else None, "log": str(log),
           "python": sys.executable,
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "engine": subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
           "engine_tree": str(tree)}
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


def cmd_run(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"rows_card{a.gpu}.jsonl"
    # The matrix is one: a cell with a row on ANY card is done, so a model's remaining cells can be
    # handed to another card without running its finished ones twice.
    latest = {}
    for f in out.glob("rows_card*.jsonl"):
        for r in map(json.loads, f.read_text().splitlines()):
            latest[(r["model"], r["mode"])] = r if _row_time(r) >= _row_time(latest.get((r["model"], r["mode"]))) \
                else latest[(r["model"], r["mode"])]
    # --rerun: the listed cells run again although they have a row; the new row names the one it
    # supersedes (kept, never deleted — the supervisor's rule of 2026-09-27 02:57).
    done = set() if a.rerun else set(latest)
    todo = [(m.strip(), mode) for m in a.models.split(",") if m.strip() for mode in a.modes.split(",")
            if (m.strip(), mode) not in done]
    while todo:
        deferred = []
        for i, (model, mode) in enumerate(todo):
            # A cell the host budget cannot take now is deferred while this pass has cells left;
            # the pass's last cell waits for the budget (and the deferred ones come next pass).
            row = run_cell(model, mode, a.gpu, out, a.timeout, Path(a.src), wait=i == len(todo) - 1)
            if row is None:
                deferred.append((model, mode))
                continue
            prev = latest.get((model, mode))
            if a.rerun and prev is not None:
                row["supersedes"] = {k: prev.get(k) for k in ("date", "gpu", "rc", "wall_s", "error", "engine", "sha256")}
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[matrix] {model} {mode} rc={row['rc']} {row.get('wall_s')}s "
                  f"{row.get('error', '')[:120]}", flush=True)
        todo = deferred
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
        for r in map(json.loads, f.read_text().splitlines()):
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
        for l in jf.read_text().splitlines():
            j = json.loads(l)
            judged[(j["model"], j["mode"])] = j
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
    with open(Path(a.out) / "judgments.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return 0


def cmd_table(a) -> int:
    out = Path(a.out)
    rows = load_rows(out)
    proofs = json.loads(Path(a.proofs).read_text()) if a.proofs and Path(a.proofs).exists() else {}
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
    (out / "table.md").write_text("\n".join(lines) + "\n")
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
    proofs = json.loads(Path(a.proofs).read_text()) if a.proofs and Path(a.proofs).exists() else {}
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
    dest.write_text("".join(json.dumps(x) + "\n" for x in lines))
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
    r.add_argument("--rerun", action="store_true",
                   help="run the listed cells although they have a row; the new row supersedes the old one")
    r.add_argument("--timeout", type=int, default=900)
    r.add_argument("--src", default=str(REPO / "src"), help="the engine tree's src the runs import (a frozen worktree)")
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
