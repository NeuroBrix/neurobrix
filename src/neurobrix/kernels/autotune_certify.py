"""`neurobrix autotune certify --profile <profile>` — fills the certified autotune directory.

For one vendor profile, on the machine that carries it: every candidate
config of every autotuned kernel is run on every shape the zoo encountered,
its result compared to the fp64 oracle (the reference bank's definition of
the op in float64), the configs that diverge beyond the profile's tolerance
excluded with their deviation, the rest timed, and the winner written with its
proof into ``config/autotune/<vendor>/<profile>/<kernel>.<dtype>.json``.

The kernel-level harness is nbxtensor's (`nbxtensor.ops.autotune_harness`: the
operands drawn, the launch, the candidates run on a poisoned output, measured
and timed; `nbxtensor.ops.oracles.fp64`: the references and the deviation).
This module keeps the campaign: which keys, at which tolerance, under which
witness protocol, priced against which card and budget, written where.

The shapes come from the machine's replay cache — the accumulation of every
key the zoo's runs resolved on this machine — or from a census file. Nothing
here names a backend or a model.

Made to be run by us on our machines and by a contributor on theirs: the
directory's gate (`autotune_certified.validate`) refuses a file without a
proof or whose proof does not re-read.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import warnings
import platform

from neurobrix.core.paths import engine_home as _engine_home
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from nbxtensor.ops import autotune_certified as C
from nbxtensor.ops import autotune_harness as H
from nbxtensor.ops.oracles import fp64 as F
from nbxtensor.dtype import itemsize as _isz
import logging
_log = logging.getLogger(__name__)

ORACLE = "fp64: the op in float64 (numpy), the reference bank's definition"
BENCH_WARMUP_MS = 10
BENCH_REP_MS = 40
# Every candidate config runs once against the oracle, and that run is timed; only the
# CONTENDERS — the configs within this factor of the fastest run — go to the stopwatch.
# A register-spilling config on a 1024² convolution runs tens of times slower than the
# winners and cannot win: on 2026-09-07 the bench of all 18 took 557 s of a 672 s shape.
BENCH_CONTENDER_FACTOR = 2.0


def _contenders(results, factor=None):
    """(cfg, deviation, run_s) rows whose single run was within `factor` of the fastest;
    never fewer than two when two exist (the proof records a second-best time)."""
    factor = BENCH_CONTENDER_FACTOR if factor is None else factor
    if not results:
        return []
    fastest = min(r[2] for r in results)
    keep = [r for r in results if r[2] <= factor * fastest]
    if len(keep) < 2 and len(results) >= 2:
        keep = sorted(results, key=lambda r: r[2])[:2]
    return keep


def witnessed_bench(contenders, time_one, witness_ms, spec: Dict[str, Any], label: str):
    """Time `contenders` ((cfg, deviation, run_s) rows) under the witness regime; returns (timed, record):
    `timed` = [(cfg, deviation, ms)], best first, and the proof's `stability_witness` record.

    The witness brackets each CHUNK of `spec["chunk"]` contenders, not the whole sweep: a sweep long
    enough for the OS to move the GPU regime across it was refused every time, though every candidate
    pair inside a short span was comparable (the M4 Pro's baddbmm (23,4096,1024) key: 16.5 / 27.1 /
    17.4 / 25.3 % across its whole sweep, up and down, on a quiet host; the Mac, 2026-10-08). A chunk
    whose bracket drifts past `drift_tolerance` is timed again, up to `chunk_attempts` times, then the
    key is refused (WitnessDrift). Across chunks each time is ranked by its ratio to its own chunk's
    witness; the ranking's two best are then timed again together inside ONE bracket, and that bracket
    alone decides the winner and states best_ms and second_ms — the decision is never taken across two
    regimes. A protocol that does not state `chunk` and `chunk_attempts` is refused by name."""
    missing = [k for k in ("drift_tolerance", "chunk", "chunk_attempts") if k not in spec]
    if missing:
        raise ValueError(f"the witness protocol states no {', '.join(missing)}: the witness regime reads the "
                         f"tolerance, the contenders per bracket and the attempts per bracket from it")
    tol, size, attempts = float(spec["drift_tolerance"]), int(spec["chunk"]), int(spec["chunk_attempts"])
    if size < 2 or attempts < 1:
        raise ValueError(f"the witness protocol states chunk={size}, chunk_attempts={attempts}: a bracket "
                         f"compares at least two contenders and is timed at least once")
    brackets: List[Dict[str, Any]] = []

    def bracket(rows, what):
        drifts = []
        for attempt in range(1, attempts + 1):
            w_open = witness_ms()
            times = [time_one(cfg) for cfg, _dev, _run_s in rows]
            w_close = witness_ms()
            drift = abs(w_close - w_open) / max(w_open, 1e-9)
            drifts.append(round(drift, 4))
            if drift <= tol:
                rec = {"what": what, "open_ms": round(w_open, 4), "close_ms": round(w_close, 4),
                       "drift": round(drift, 4), "attempt": attempt, "contenders": len(rows)}
                brackets.append(rec)
                return times, (w_open + w_close) / 2.0, rec
        raise WitnessDrift(
            f"{label}: the witness drifted past {tol*100:.0f}% on every one of {attempts} timings of "
            f"{what} ({len(rows)} contenders; drifts {', '.join(f'{d*100:.1f}%' for d in drifts)}): the "
            f"GPU regime moved while those candidates were being compared. Refused — not a measurement.")

    rows = list(contenders)
    ranked = []                                   # (ratio to the chunk's witness, cfg, dev, ms)
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        times, w_mid, _rec = bracket(chunk, f"chunk {i // size + 1} of {-(-len(rows) // size)}")
        ranked += [(ms / w_mid, cfg, dev, ms) for (cfg, dev, _run_s), ms in zip(chunk, times)]
    ranked.sort(key=lambda r: r[0])
    if len(rows) <= size:
        decisive = brackets[-1]
        timed = [(cfg, dev, ms) for _r, cfg, dev, ms in ranked]
    else:
        head = [(cfg, dev, None) for _r, cfg, dev, _ms in ranked[:2]]
        times, _w_mid, decisive = bracket(head, "the final two")
        final = sorted(((cfg, dev, ms) for (cfg, dev, _), ms in zip(head, times)), key=lambda t: t[2])
        timed = final + [(cfg, dev, ms) for _r, cfg, dev, ms in ranked[2:]]
    record = {"open_ms": decisive["open_ms"], "close_ms": decisive["close_ms"], "drift": decisive["drift"],
              "tolerance": tol, "kernel": spec.get("kernel", "matmul"),
              "shape": [int(spec["M"]), int(spec["N"]), int(spec["K"])],
              "chunk": size, "brackets": brackets}
    return timed, record


# ---------------------------------------------------------------------------
# the census: which shapes
# ---------------------------------------------------------------------------
def keys_file(path: str) -> Dict[str, List[tuple]]:
    """{kernel qualname: [key, ...]} from a file of "<qualname>::<key repr>" lines or the local sweep store (its keys)
    — the keys a coverage check found missing, a launch record formed: certified exactly, no census table read (the
    supervisor, 2026-10-26 12:35: the coverage check's list suffices). An empty file, a malformed line or a
    kernel the certifier has no tuner for is refused by name: a list that certified nothing must not read green."""
    import ast
    out: Dict[str, List[tuple]] = {}
    lines = ([str(k) for k in json.loads(Path(path).read_text())] if str(path).endswith(".json")   # the sweep store
             else [l.strip() for l in open(path) if l.strip()])
    if not lines:
        raise RuntimeError(f"--keys {path}: the file names no key — nothing would be certified")
    for n, line in enumerate(lines, 1):
        qual, sep, key = line.partition("::")
        if not sep or not qual or not key:
            raise RuntimeError(f"--keys {path}:{n}: {line!r} is not '<kernel qualname>::<key repr>'")
        try:
            k = ast.literal_eval(key)
        except (ValueError, SyntaxError) as exc:
            raise RuntimeError(f"--keys {path}:{n}: the key {key!r} does not parse ({exc})") from exc
        if not isinstance(k, tuple):
            raise RuntimeError(f"--keys {path}:{n}: the key {key!r} is not a tuple")
        out.setdefault(qual, [])
        if k not in out[qual]:
            out[qual].append(k)
    return out


def census(vendor: str, profile: str, memory_class: int, models: Optional[List[str]] = None) -> Dict[str, List[tuple]]:
    """{kernel qualname: [key, ...]} from THE census table of this profile and memory class
    (`config/census/<vendor>/<profile>/<class>g.jsonl`, `kernels/census_table.py`) — and from nothing
    else. It used to default to the machine's replay cache (earlier runtime sweeps) or take any census
    JSON: the owner's method (2026-09-28) certifies exactly the committed table."""
    from neurobrix.kernels import census_table as T
    path = T.table_path(vendor, profile, memory_class)
    if not path.exists():
        raise RuntimeError(f"there is no census table {path}: the certifier certifies the table and nothing "
                           f"else — take the census for {vendor}/{profile} {memory_class} GB first")
    out: Dict[str, List[tuple]] = {}
    seen = set()
    want = set(models) if models else None
    for row in T.read(path):
        # `models`: only the rows of these containers — the keys a re-certification needs first (a kernel change
        # re-sweeps ~7 500 Apple conv keys at ~29 s: the video VAEs' first, the rest after)
        if want is not None and row.get("model") not in want:
            continue
        ident = T.key_line(row["kernel"], row["key"])
        if ident in seen:
            continue
        seen.add(ident)
        key = C.parse_key(row["key"])
        if key is not None:
            out.setdefault(row["kernel"], []).append(key)
    return out


# ---------------------------------------------------------------------------
# the size a key is drawn at: the card's share the campaign plans against
# ---------------------------------------------------------------------------
class WitnessDrift(RuntimeError):
    """The regime witness moved beyond its tolerance across a sweep or a re-prove: not a measurement.
    A key refused for this is neither certified nor swept — its retry pass re-proves it (inbox 60, 65)."""


def _room(card_bytes) -> "H.Room":
    """The capacity a key on a DISCRETE card is drawn within: the card's bytes at the share Prism plans against
    (`PRISM_DEFAULTS["safety_margin"]`) — never the card's whole bytes: Sana 4K's 16g rms_norm key (16 252 928 rows of
    128 fp32, in and out 15.5 GiB of a 16 GiB card) passed a door at the whole card, then failed its output's allocation
    at every pass (2026-10-23); a key no plan of the class can form is one no run of it presents. None: unbounded (a
    unified-memory device is bounded by its phase price, `_refuse_unified`)."""
    if card_bytes is None:
        return H.Room()
    from neurobrix.core.config.system import PRISM_DEFAULTS
    return H.Room(int(card_bytes), int(int(card_bytes) * float(PRISM_DEFAULTS["safety_margin"])))


def refused_path(vendor: str, profile: str, memory_class: int, root: Optional[Path] = None) -> Path:
    """The class's register of keys refused by size (`refused.<cls>g.jsonl`), in the certified directory beside the
    proofs, committed by the same checkpointer; one line per (kernel, key)."""
    return C.file_for(vendor, profile, "x", "x", root=root).parent / f"refused.{int(memory_class)}g.jsonl"


def read_refused(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(kernel qual, key repr): its refusal}; an absent register is an empty one. A line that does not parse is
    refused by name — a register that silently dropped a line would retry, or skip, the wrong key."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            out[(r["kernel"], r["key"])] = r
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}:{n}: not a refusal line ({exc}): {line[:120]}") from exc
    return out


def record_refusal(path: Path, qual: str, ktext: str, asked: int, usable: int, card: int, reason: str) -> None:
    """The refusal written (replacing the key's earlier one), atomically and under a lock: certifiers of one class
    run on several cards at once."""
    import fcntl
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            rows = read_refused(path)
            rows[(qual, ktext)] = {"kernel": qual, "key": ktext, "asked_bytes": int(asked), "usable_bytes": int(usable),
                                   "card_bytes": int(card), "reason": reason,
                                   "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for k in sorted(rows):
                    fh.write(json.dumps(rows[k], sort_keys=True) + "\n")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


#: the kernels whose key carries a padded row length (`launch_keys.softmax_feat_bucket`), drawn at the class's top where
#: the card holds it and else at the shortest row that forms the key (`H.Room.row_shape`)
_PADDED_ROW = ("softmax_forward_kernel", "softmax_forward_looped_kernel", "mean_forward_kernel", "sum_forward_kernel",
               "amax_forward_kernel", "group_norm_forward_kernel")


def _unified_padded_row(qual: str, tuner, key: tuple, card_bytes, short: str) -> tuple:
    """On a unified-memory device, the key a padded-row kernel is DRAWN at: the class's top where its phase price fits
    the card, else the shortest row that forms the key — `Room.row_shape`'s rule, which a unified device skipped: its
    bound is the phase price, so it was called with no card and drew every softmax at its top; the Mac's census on
    00e31765 refused HAT-L's 1 572 864 - 2 621 440 rows of 1 024 at 21-27 GB of a 19 GB card while the shortest row of
    the class (513) holds them (2026-10-24 07:18). The key filed is the census's; only the drawn row length moves."""
    if card_bytes is None or short not in _PADDED_ROW + ("ew_launch_kernel",):
        return key
    pr = price_key(qual, tuner, key)
    if short == "ew_launch_kernel":
        # `Room.ew_count`'s rule on the phase price: the class's top where it fits, else its floor. The open class's key is
        # its label (68 719 476 736 elements on the four-row ladder), no count a run reaches: priced there, every Apple
        # profile refused it at 2 TB and canary / granite-speech missed it under certified-only (2026-10-24 13:43)
        if pr is None or pr["peak"] <= card_bytes:
            return key
        from nbxtensor.ops.autotune_bucket import class_floor
        return (class_floor("elements", int(key[0])),) + tuple(key[1:])

    if pr is None or pr["peak"] <= card_bytes:
        return key
    shortest = H.shortest_row(int(key[1]))
    if shortest is None:
        return key                              # no shorter row forms the key: priced, and refused, at its top
    return (key[0], shortest) + tuple(key[2:])


def _refuse_unified(qual: str, tuner, key: tuple, card_bytes) -> None:
    """On a device whose memory IS the host's, the host copies live in the same pool as the
    device buffers (a conv sized at 3 GB of device bytes took the Mac's certifier past 14 GB,
    2026-09-27), so the bound is the key's phase price — host draws, device copies, oracle —
    against the card. Checked before any draw.

    It was a constant, 10 bytes per element of every operand and of the output (a float32 host
    draw, the device copy, a float32 readback). The draws are held at the kernel's width since
    2026-10-05 and the output is read back on the oracle's windows only, and the constant refused
    eight keys of the Apple table — convolutions at 4224-5120 px, a VAE conv at batch 162, an
    fp32 addmm at M=163840 — that the planner places whole on the same 18 GB profile (the rack,
    2026-10-05 13:40)."""
    if card_bytes is None:
        return
    pr = price_key(qual, tuner, key)
    if pr is None or pr["peak"] <= card_bytes:
        return
    mib = lambda b: f"{b / 2**20:.0f} MiB"                                 # noqa: E731
    raise H.KeyTooLargeForClass(
        int(pr["peak"]), int(card_bytes),
        f"{C.kernel_short(qual)} on unified memory, priced at its peak phase: draws "
        f"{mib(pr['draws'])} + device {mib(pr['device'])} + max(oracle {mib(pr['oracle'])}, "
        f"deviation {mib(pr['deviation'])}) = {mib(pr['peak'])}")



# ---------------------------------------------------------------------------
# the price of a key: what a certification holds at each of its phases
# ---------------------------------------------------------------------------
def unit_owed(qual: str, key: tuple, stored: Optional[Dict[str, Any]]) -> bool:
    """A served FMA certificate of an `mm` / `addmm` key the matrix unit can take, made before the unit was a
    candidate beside the FMA configurations (1.8): its proof names no unit measurement, so the runtime serves the FMA
    kernel where, until 1.8, the unit ran keyless by threshold (2026-11-04: ~880 keys a class, 41 models). The unit's
    tiles are owed — measured beside the stored configuration, kept (the owner's rule 2026-11-03 17:48: only the new
    candidate is measured, the certificate stays). Read at the operands `synthesize` draws (`H.gemm_rows`)."""
    if not stored or not stored.get("config") or "unit_measured" in (stored.get("proof") or {}):
        return False
    if C.kernel_short(qual) not in ("matmul_kernel", "addmm_kernel"):
        return False
    from nbxtensor.ops import launch_keys as LK
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    nbx = {"fp16": NBXDtype.float16, "bf16": NBXDtype.bfloat16, "fp32": NBXDtype.float32}
    dts = C.key_dtypes(key)
    if len(dts) < 2 or dts[0] not in nbx or dts[1] not in nbx:
        return False
    try:
        m = H.gemm_rows(key, dts, unit_keyed=True)
    except ValueError:
        return False
    return LK.matrix_unit_operands(nbx[dts[0]], nbx[dts[1]], m) is not None


def _refuse_unpriced(qual: str, tuner, key: tuple, budget_bytes: int, floor_bytes: int) -> None:
    """Refuse a key whose priced peak, over the process's floor, exceeds the working-set budget —
    before a value is drawn, naming the phases and the budget."""
    pr = price_key(qual, tuner, key)
    if pr is None:
        return
    asked = int(floor_bytes) + int(pr["peak"])
    if asked > budget_bytes:
        mib = lambda b: f"{b / 2**20:.0f} MiB"                             # noqa: E731
        raise H.KeyTooLargeForClass(
            asked, int(budget_bytes),
            f"{C.kernel_short(qual)} priced at its peak phase: draws {mib(pr['draws'])} + device "
            f"{mib(pr['device'])} + max(oracle {mib(pr['oracle'])}, deviation {mib(pr['deviation'])}) "
            f"over a {mib(floor_bytes)} floor = {mib(asked)}, against a {mib(budget_bytes)} working-set "
            f"budget (the figure the run's guard kills at)")

def _draw_bytes(dt: str) -> int:
    """What one element of a drawn operand costs on the host for the whole key: its own width
    (`H.draw` draws through a bounded chunk into the kernel's dtype, bf16 as its bits)."""
    return _isz.itemsize(_isz.from_autotune_key(dt))


def price_key(qual: str, tuner, key: tuple) -> Optional[Dict[str, int]]:
    """The bytes a certification of `key` holds, phase by phase, from what the code allocates —
    never a constant per element. Measured on the Mac 2026-09-28 (`NBX_CERTIFY_PHASES=1`,
    results/certifier_price): a depthwise key's peak is the windowed fp64 conv oracle (three
    output windows, each an fp64 input slab, its padded copy and a kept fp64 block), a matmul
    key's is its draws and the row-windowed launch oracle, and none of it is 10 bytes per element.

    Phases, each held in full while the next one is built:
      draws     — every input held on the host at its own width (`_draw_bytes`), alive for the
                  whole key (the oracle closures read them);
      device    — every operand and the output on the device at the key's width, plus `do_bench`'s
                  256 MiB flush buffer; on a unified-memory device these bytes are the host's;
      oracle    — the fp64 reference: whole under F.ORACLE_MAX_MACS, windowed above it
                  (`F.row_windows` / `F.conv_windows`), with the transient of one window's build on
                  top of the blocks kept;
      deviation — the kept oracle plus one candidate's readback of what is compared (fp32 read,
                  float64 cast), whole or on the windows.
    `peak` = draws + device + max(oracle, deviation). The process's own floor (its memory before
    the key) is not in it: the caller adds what it measures."""
    ops = H.key_operands(qual, tuner, key)
    if ops is None:
        return None
    E = lambda shape: int(np.prod(shape, dtype=np.int64))                # noqa: E731
    draws = sum(E(shape) * _draw_bytes(dt) for _, shape, dt in ops["inputs"])
    out_shape, out_dt = ops["out"]
    device = sum(E(shape) * _isz.itemsize(_isz.from_autotune_key(dt)) for _, shape, dt in ops["inputs"]) + E(out_shape) * _isz.itemsize(_isz.from_autotune_key(out_dt))
    device += 256 * 2 ** 20                                              # do_bench's flush buffer
    e_out = E(out_shape)
    if ops["family"] == "mm":
        M, N, K = ops["mnk"]; B = ops["batch"]
        ea = sum(E(shape) for name, shape, _ in ops["inputs"] if name == "a")
        eb = sum(E(shape) for name, shape, _ in ops["inputs"] if name == "b")
        ebias = sum(E(shape) for name, shape, _ in ops["inputs"] if name == "bias")
        wins = F.row_windows(M * B, N, K)
        if wins is None:                                                 # whole: a, b, bias, out in float64
            oracle = 8 * (ea + eb + ebias + e_out)
            kept = 8 * e_out
            deviation = kept + e_out * (4 + 8)
        else:
            rows = wins[0][1] - wins[0][0]
            if ops["row_windowed_launch"]:
                # the launch oracle (`screen_oracle._mm`): the rows cut on the device, read as
                # fp32 and cast to float64; b whole the same way; the window's product kept
                per_win = rows * K * (4 + 8) + eb * (4 + 8) + rows * N * 8
                kept = len(wins) * rows * N * 8
                oracle = kept + per_win
            else:
                # the synthesized oracle (`F.matmul_oracle`): a's rows cast to float64 from
                # the host draw, b whole in float64, the window's product kept
                wins_b = F.row_windows(M, N, K, cap=max(1, F.ORACLE_MAX_MACS // B)) or [(0, M)]
                rows = wins_b[0][1] - wins_b[0][0]
                per_win = B * rows * K * 8 + eb * 8 + B * rows * N * 8
                kept = len(wins_b) * B * rows * N * 8
                oracle = kept + per_win
            deviation = kept + kept // 8 * (4 + 8)                       # the windows' readback and cast
        return {"draws": draws, "device": device, "oracle": oracle, "deviation": deviation,
                "peak": draws + device + max(oracle, deviation)}
    if ops["family"] == "gemv":
        # `F.gemv_oracle`: the matrix's rows cast to float64 a block at a time (`F.gemv_rows`), the vector in
        # float64, the whole fp64 result kept
        _, N, K = ops["mnk"]
        rows = F.gemv_rows(N, K)
        oracle = N * 8 + rows * K * 8 + K * 8
        deviation = N * 8 + N * (4 + 8)
        return {"draws": draws, "device": device, "oracle": oracle, "deviation": deviation,
                "peak": draws + device + max(oracle, deviation)}
    if ops["family"] == "elementwise":
        # identity: the draw cast to float64 is the oracle, the result read back and cast beside it
        n = ops["mnk"][0]
        return {"draws": draws, "device": device, "oracle": n * 8, "deviation": n * 8 + n * (4 + 8),
                "peak": draws + device + n * 8 + n * (4 + 8)}
    if ops["family"] in ("softmax", "rownorm"):
        # `F.softmax_oracle`: per row window, the window cast to float64 plus the fp64 arrays it computes
        # (`F.ROW_F64_ARRAYS`), the windows' results kept (`RowWindowedOracle`); whole under the cap.
        # Priced with the mm and conv families: without it a softmax key read the conv geometry and every Apple
        # softmax certification failed on 'geometry' before its first launch (Mac, 2026-10-21).
        m, n, _ = ops["mnk"]
        wins = F.row_windows(m, n, F.ROW_F64_ARRAYS * 8)          # the oracle's own windows (bytes)
        rows = m if wins is None else wins[0][1] - wins[0][0]
        kept = (m if wins is None else len(wins) * rows) * n * 8           # the windows' fp64 results (whole: all)
        if wins is not None:
            # the draw is on the device once and the host keeps only the windows' rows (`F.oracle_rows`), as numbers
            _, xs, xdt = ops["inputs"][0]
            draws += len(wins) * rows * n * (4 if xdt == "bf16" else _draw_bytes(xdt)) - E(xs) * _draw_bytes(xdt)
        oracle = kept + rows * n * 8 * (1 + F.ROW_F64_ARRAYS)
        deviation = kept + kept // 8 * (4 + 8)                             # the windows' readback and cast
        return {"draws": draws, "device": device, "oracle": oracle, "deviation": deviation,
                "peak": draws + device + max(oracle, deviation)}
    if ops["family"] == "attention":
        # `F.attention_oracle`: per query-row window, q's rows, k and v whole and the window's scores in float64
        # (scores, their exp and the quotient), the window's result kept (`RowWindowedOracle`); whole under the cap.
        # Priced with the other families: without it a flash key read the conv geometry and every Apple flash
        # certification failed on 'geometry' before its first launch (C3, Mac 2026-10-23).
        q = next(shape for name, shape, _ in ops["inputs"] if name == "q")
        k = next(shape for name, shape, _ in ops["inputs"] if name == "k")
        bh, tq, hd, tk = q[0] * q[1], q[2], q[3], k[2]
        wins = F.row_windows(tq, hd, 2 * tk)
        rows = tq if wins is None else wins[0][1] - wins[0][0]
        kept = bh * tq * hd * 8
        oracle = kept + bh * (rows * hd + 2 * tk * hd + 3 * rows * tk) * 8
        deviation = kept + kept // 8 * (4 + 8)
        return {"draws": draws, "device": device, "oracle": oracle, "deviation": deviation,
                "peak": draws + device + max(oracle, deviation)}
    if ops["family"] == "moe":
        # `F.moe_oracle`: per row window of the pairs (whole under the cap), the window's activations in float64 and
        # their gather per expert, one expert's weights in float64 at a time, the window's product and its result kept;
        # the device beside it also holds the alignment's tables (the padded worst case, int64)
        nv, N, K = ops["mnk"]
        device += 8 * 2 * (nv + ops["experts"] * 16)
        wins = F.row_windows(nv, N, K) or [(0, nv)]
        rows = wins[0][1] - wins[0][0]
        kept = len(wins) * rows * N * 8
        oracle = kept + rows * K * 8 * 2 + N * K * 8 + rows * N * 8
        deviation = kept + kept // 8 * (4 + 8)
        return {"draws": draws, "device": device, "oracle": oracle, "deviation": deviation,
                "peak": draws + device + max(oracle, deviation)}
    n, ci, h, w, co, oh, ow, ci_g, kh, kw, ph, pw = ops["geometry"]
    ew = sum(E(shape) for name, shape, _ in ops["inputs"] if name == "w")
    wins = F.conv_windows(n, oh, ow, ci_g, co, kh, kw)
    if wins is None:                                                     # whole: x and its padded copy, w, out in float64
        ex = n * ci * h * w
        ex_pad = n * ci * (h + 2 * ph) * (w + 2 * pw)
        oracle = 8 * (ex + ex_pad + ew + e_out)
        kept = 8 * e_out
        deviation = kept + e_out * (4 + 8)
    else:
        # per window (`conv2d_reference_window`): the input slab of its receptive field in
        # float64, the slab padded, w in float64, the output block kept
        kept = 0; per_win = 0
        for (ni, r0, r1, c0, c1) in wins:
            rh, rw = r1 - r0, c1 - c0
            slab = ci * (rh + kh - 1) * (rw + kw - 1)
            block = co * rh * rw
            kept += 8 * block
            per_win = max(per_win, 8 * (2 * slab + ew))
        oracle = kept + per_win
        deviation = kept + kept // 8 * (4 + 8)
    return {"draws": draws, "device": device, "oracle": oracle, "deviation": deviation,
            "peak": draws + device + max(oracle, deviation)}

def synthesize(qual: str, tuner, key: tuple, rng, card_bytes: Optional[int] = None, unified: bool = False,
               values: bool = True, budget_bytes: Optional[int] = None,
               floor_bytes: int = 0) -> Optional[Tuple[Callable[[], Any], Optional[Callable[[], Any]], str]]:
    """(call, oracle_fn, output_arg_name) for `key` — the library's synthesis (`nbxtensor.ops.autotune_harness.
    synthesize`: the wrapper run on operands of the key's shape and dtypes, and its LAZY fp64 oracle) behind the
    campaign's doors, all before any draw:

    - `budget_bytes`: THE PRICE BEFORE ANY DRAW (supervisor, 2026-09-28 05:41): what this key's certification will hold
      at its peak phase, over the process's measured floor, against the budget the run is gated on — the same figure
      its guard kills at; refused by name, with the phases (`_refuse_unpriced`). The key DRAWN is priced, not the key
      filed: on a unified device the open element class (its label, 68 719 476 736 elements) asked 2.2 TB of an 18 GB
      Mac at its filed top and was refused at every re-proof (2026-11-07).
    - `unified`: the host draws and the device copies share one pool — a padded-row or element-wise class is drawn at
      the size its phase price fits (`_unified_padded_row`) and the key is refused over the card by its phase price
      (`_refuse_unified`);
    - else `card_bytes`: each operand and the output against the card's share Prism plans against (`_room`).

    `values=False`: the arguments only, no draw and no oracle (the Mac's `msl_census.py` compiles without running)."""
    if budget_bytes is not None:
        drawn = _unified_padded_row(qual, tuner, key, card_bytes, C.kernel_short(qual)) if unified else key
        _refuse_unpriced(qual, tuner, drawn, budget_bytes, floor_bytes)
    out_dt = C.output_dtype(tuner, key) if card_bytes is not None else None   # sized only to refuse
    filed = key
    if unified:
        key = _unified_padded_row(qual, tuner, key, card_bytes, C.kernel_short(qual))
        _refuse_unified(qual, tuner, key, card_bytes)
        card_bytes = None                       # the phase price is the bound on a unified device
    return H.synthesize(qual, tuner, key, rng, room=_room(card_bytes), values=values, filed_key=filed, out_dt=out_dt)


# ---------------------------------------------------------------------------
# one key: every config against the oracle, the survivors against the clock
# ---------------------------------------------------------------------------
def _machine() -> Dict[str, Any]:
    # no host name: a committed proof never names the machine that made it (the owner, 2026-11-07); the device, the
    # target and the clocks are what a certificate proves on
    info: Dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        from nbxtensor.ops import wrappers as W
        hw = W.get_hardware_profile()
        if hw is not None:
            info["hardware_profile"] = str(getattr(hw, "id", hw))
            info["has_native_bf16"] = bool(W.has_native_bf16())
    except Exception:
        pass
    try:
        from nbxtensor.backend.launcher import arch_name, target
        t = target()
        info["target"] = f"{t.backend}-{arch_name(t)}"
    except Exception:
        pass
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        for name in ("device_name", "driver_version"):
            fn = getattr(DeviceAllocator, name, None)
            if callable(fn):
                info[name] = fn()
    except Exception:
        pass
    info["device"] = _certifying_device()
    info["clocks_mhz"] = _clocks_mhz()
    return info


_CERTIFYING_DEVICE: list = []          # read once per run, like the clocks


def _certifying_device() -> Optional[Dict[str, Any]]:
    """Read ONCE for the run (a query per key is the anti-pattern `_clocks_mhz` names)."""
    if not _CERTIFYING_DEVICE:
        _CERTIFYING_DEVICE.append(_read_certifying_device())
    return _CERTIFYING_DEVICE[0]


def _read_certifying_device() -> Optional[Dict[str, Any]]:
    """The card the certifying run executes on — the device `from_numpy`
    places the synthesized inputs on (`DeviceAllocator.get_device()`), read in
    the Prism hardware profile in force: index, name, memory. The proof says
    which card's memory it was made on, and an entry is served only to that
    memory class (register 56). None when the card is not in the profile —
    the certifier then REFUSES to write the entry (`file_certification`)."""
    try:
        from nbxtensor.memory.allocator import DeviceAllocator
        idx = int(DeviceAllocator.get_device())
    except Exception:
        return None
    try:
        from nbxtensor.ops import wrappers as W
        prof = W.get_hardware_profile()
        devices = getattr(prof, "devices", None) if prof is not None else None
    except Exception:
        devices = None
    dev = next((d for d in (devices or []) if getattr(d, "index", None) == idx), None)
    if dev is None:
        return None
    # `ordinal` is the CUDA ordinal in the visible set (0 under a pin to any physical card),
    # not the physical card; the visible set is recorded beside it so the pair says which.
    return {"ordinal": idx, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": str(getattr(dev, "name", "?")), "memory_mb": int(getattr(dev, "memory_mb", 0) or 0),
            # `DeviceSpec.has_unified_memory` is a PROPERTY. It was called as a method, which
            # raised "'bool' object is not callable" on every profile that lists its devices
            # (the Mac's first certify on 2026-09-27); the test's stub had made it a method.
            "unified": bool(dev.has_unified_memory) if hasattr(dev, "has_unified_memory") else False}


def _clocks_mhz():
    """The application clock of every card, or None when it cannot be read.

    A certification picks a configuration BY TIMING candidates, so the frequency
    the cards ran at is a condition of the result exactly as the platform and the
    hardware profile already recorded here are. Without it a proof cannot say
    what regime produced its `best_ms`, and the question becomes unanswerable the
    moment the machine reboots — which is how the 2026-09-11 entries came to
    carry timings whose clock is unrecoverable.

    It records what it read and never asserts a protocol: whether a reading is
    the right one is the workshop's question (`tools/rig_clock.py`), not the
    engine's. `None` means the reading failed and is written as such, because a
    field quietly absent is indistinguishable from a machine that had no clocks.

    READ ONCE for the whole run, not once per shape. `_machine()` is called for
    every certified key — 7,137 of them in this directory — and a driver query
    per key is the anti-pattern this project has already paid for once. So this
    is the reading at the run's START, and it says nothing about whether the
    clock HELD: holding is the sampler's job (`ClockWatch` in bench_row.py),
    which watches `clocks.sm` throughout and marks an excursion. Entry condition
    here, ongoing condition there; neither substitutes for the other.
    """
    if _clocks_mhz.cached is not _UNREAD:
        return _clocks_mhz.cached
    _clocks_mhz.cached = _read_clocks_mhz()
    return _clocks_mhz.cached


class UnreachableCensusKey(RuntimeError):
    """The census holds a key the engine can no longer ask for.

    NOT a failure of the certification, and the distinction is the whole point of
    the class. The census accumulates across engine versions; when a wrapper
    changes how it computes its autotune key, every entry recorded under the old
    rule becomes unreachable — no run will ever present that key again, so there
    is nothing to certify and refusing is correct.

    It is separated from a real break because an exit code that conflates the two
    stops being read. On 2026-09-12 the certification refused 184 such keys, all
    of them the known debt D-CENSUS-HOLDS-KEYS-THE-ENGINE-CANNOT-PRODUCE, and
    exited 1 — the same 1 a genuine break would produce. A status that says
    "something is wrong" on every run of a directory that is in fact healthy will
    be ignored, and the day it tells the truth nobody will listen.
    """


_UNREAD = object()


def _read_clocks_mhz():
    """Both application clocks of every card.

    Both, because `nvidia-smi -ac <mem>,<gfx>` sets both and a protocol names
    both. A reader that returns only the graphics clock lets a rig satisfy this
    door while failing the workshop's (`tools/rig_clock.py`), and two doors that
    disagree about the same protocol are a false green waiting for its occasion.

    Every card, because `nvidia-smi` speaks real indices and ignores
    CUDA_VISIBLE_DEVICES: a card at the wrong frequency beside a measurement is
    a fact about the rig whether or not this job was pinned away from it.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.applications.graphics,"
             "clocks.applications.memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None
        out = {}
        for line in r.stdout.splitlines():
            if line.strip():
                idx, gfx, mem = [c.strip() for c in line.split(",")]
                out[idx] = {"graphics": int(gfx), "memory": int(mem)}
        return out or None
    except Exception:
        return None


_clocks_mhz.cached = _UNREAD

_PROTOCOL_ENV = "NEUROBRIX_RIG_PROTOCOL"
OFF_PROTOCOL_OPT = "--allow-off-protocol-clock"


def _current_backend() -> Optional[str]:
    """The backend this machine targets (`cuda`, `metal`, `hip`), or None.

    A protocol is scoped to a backend, so selecting the file needs the backend
    name — and it must be the ENGINE's name for the backend, not the Triton
    target's.

    This used to read `launcher.target().backend`. Those two agree on NVIDIA
    (both say `cuda`), which is why the difference stayed invisible, and they
    diverge on Apple: `gpu_backend()` says `metal` while the Triton
    target says whatever the installed Triton backend calls itself — `metal`
    under the archived fork, `mps` under triton-ext.

    So swapping the Triton backend silently renamed the thing the protocol is
    scoped by, `rig_protocol.metal.json` stopped being found, and certify ran on
    2026-09-17 reporting "this machine declares no measurement protocol for
    backend 'mps' — the regime is RECORDED in every proof but checked against
    nothing". Six shapes were certified that way. **The witness is the whole
    reason Apple certification is trustworthy** (the clocks cannot be locked, so
    stability is proven by re-timing a reference kernel), and it was not being
    checked.

    A measurement regime is a property of the MACHINE AND ITS GPU — "Apple GPU
    clocks are OS-managed" is true whatever Triton calls the target — so it is
    scoped by the engine's own vendor name, which is stable across Triton
    backend swaps. The old proofs agree: they record `backend.name = "metal"`
    and `target = "metal-apple-m4-pro"`.
    """
    try:
        from nbxtensor.backend.runtime import gpu_backend
        return gpu_backend()
    except Exception:
        return None


def _protocol_file() -> Optional[Path]:
    """The machine's declared measurement protocol FOR ITS BACKEND, or None.

    A protocol is a property of the MACHINE AND ITS BACKEND, not of the engine:
    it records the regime this rack's numbers are taken under (a clock lock on
    an NVIDIA rack, a witness on Apple Silicon where the clock cannot be
    locked). It is DISCOVERED, never shipped with a built-in value.

    SCOPED BY BACKEND so it cannot LEAK across machines: the file is
    `rig_protocol.<backend>.json`, and a Mac (metal) never reads the Dell's
    `rig_protocol.cuda.json`. The un-suffixed `rig_protocol.json` was a leak by
    construction — the checkout carried one machine's V100 protocol and every
    other machine inherited it (2026-09-16: a Mac refused certification because
    it could not read the NVIDIA clocks the Dell's committed protocol named).

    Order: an explicit env pointer, then the backend file in a source checkout,
    then the machine's dotfile. A legacy un-suffixed file is honoured ONLY if it
    declares this backend, never blindly — the leak does not come back.
    """
    backend = _current_backend()
    env = os.environ.get(_PROTOCOL_ENV)
    if env:
        return Path(env)                      # named, so a missing one refuses below
    names = [f"rig_protocol.{backend}.json"] if backend else []
    for name in names:
        for parent in Path(__file__).resolve().parents:
            cand = parent / "tools" / name
            if cand.is_file():
                return cand
        cand = _engine_home() / name
        if cand.is_file():
            return cand
    # A legacy un-suffixed file counts only if it names THIS backend.
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "rig_protocol.json"
        if cand.is_file() and _protocol_backend(cand) == backend:
            return cand
    cand = _engine_home() / "rig_protocol.json"
    if cand.is_file() and _protocol_backend(cand) == backend:
        return cand
    return None


def _protocol_backend(path: Path) -> Optional[str]:
    """The backend a protocol file declares (`_backend`), or None if unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("_backend")
    except Exception:
        return None


def rig_protocol_refusal(allow_off_protocol: bool = False, say=print) -> None:
    """Refuse to certify unless EVERY card sits at the machine's protocol clock.

    WHY A CERTIFICATION TAKES A CAMPAIGN'S ENTRY CONDITION
    -----------------------------------------------------
    Certification picks a configuration BY TIMING candidates. That makes it a
    measurement exactly as a benchmark row is one, and its `best_ms` values carry
    a regime whether or not anyone recorded which. On 2026-09-11 this machine lost
    mains twice in nine minutes, came back at 1312/1312/1290/1290 — each SKU at
    its OWN factory default — and a certification ran across all four cards with
    no harness reading the clocks at entry.

    WHY A DOOR AND NOT A CENSUS
    ---------------------------
    Application clocks do not survive a reboot, so the harmful state recurs on a
    schedule nobody controls. A census says the clocks were right this time; a
    refusal at entry says a number produced behind it was produced on protocol,
    which is a claim about every future run.

    WHY IT NAMES EVERY DIVERGING CARD
    ---------------------------------
    The rack is heterogeneous: 16 GB V100s default to 1312 MHz, 32 GB ones to
    1290, and the protocol value is 1290. After a reboot half the rig is already
    at the protocol value by pure manufacturer coincidence. A check that samples
    one card goes green on a rig that is half wrong, and one that samples card 2
    or 3 here goes green ALWAYS. Both SKUs advertise byte-identical supported-clock
    lists, so no capability query reveals the disagreement — only the reading does.

    A machine that declares NO protocol is not refused: there is nothing for it to
    diverge from, and the proof still records the clocks it read. But it is told
    so in the run's own output, because a silence here would be indistinguishable
    from a door that held.
    """
    path = _protocol_file()
    if path is None:
        be = _current_backend()
        say(f"[certify] this machine declares no measurement protocol for backend "
            f"{be!r} (no {_PROTOCOL_ENV}, no tools/rig_protocol.{be}.json, no "
            f"~/.neurobrix/rig_protocol.{be}.json) — the regime is RECORDED in "
            f"every proof but checked against nothing.")
        return

    try:
        proto = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the protocol from {path} ({exc}). The protocol is not "
            f"optional and has no default: restore the file rather than "
            f"certifying without it.")

    # The stability regime: a recorded LOCK (NVIDIA, clocks held to a protocol
    # and read at entry) or a recorded WITNESS (Apple, where the clock cannot be
    # locked, so stability is PROVEN per-sweep by a reference kernel's drift —
    # checked in `certify_key`, not here). The contract stays whole and single:
    # both are proofs, both are recorded, an entry without a recorded regime is
    # not served.
    regime = proto.get("regime", "clock_lock")
    if regime == "witness":
        _witness_entry_refusal(proto, path, say)
        return
    if regime != "clock_lock":
        raise RuntimeError(
            f"protocol {path} declares an unknown regime {regime!r}: the regimes "
            f"are 'clock_lock' (a held clock, read at entry) and 'witness' (a "
            f"reference kernel's drift, proven per sweep).")

    try:
        clock = proto["clock"]
        want_gfx = int(clock["application_graphics_mhz"])
        want_mem = int(clock["application_memory_mhz"])
    except Exception as exc:
        raise RuntimeError(
            f"cannot read the clock protocol from {path} ({exc}). The protocol "
            f"value is not optional and has no default: restore the file rather "
            f"than certifying without it.")

    cards = _clocks_mhz()
    if cards is None:
        raise RuntimeError(
            "cannot read the rig's clocks, and this machine declares a protocol "
            f"({want_gfx}/{want_mem} MHz) to hold them to. A certification whose "
            "conditions cannot be read is not a measurement.")
    if not cards:
        raise RuntimeError(
            "the driver reported ZERO cards. This check examined nothing, and a "
            "check that examined nothing must not be read as one that found "
            "nothing wrong.")

    off = {i: c for i, c in cards.items()
           if c["graphics"] != want_gfx or c["memory"] != want_mem}
    if not off:
        say(f"[certify] {len(cards)} card(s) read, all at the protocol clock "
            f"{want_gfx}/{want_mem} MHz ({path})")
        return

    lines = [f"the rig is NOT at the protocol clock: {len(off)} of {len(cards)} "
             f"card(s) diverge (protocol {want_gfx}/{want_mem} MHz, from {path})"]
    for i in sorted(cards, key=lambda s: int(s)):
        c = cards[i]
        lines.append(f"    card {i}  {c['graphics']}/{c['memory']} MHz"
                     + ("" if i in off else "  (at protocol)"))
    lines.append("")
    lines.append("  Application clocks do not survive a reboot and each SKU returns "
                 "to its OWN factory default, so half a heterogeneous rack can sit "
                 "at the protocol value by coincidence.")
    lines.append("")
    lines.append("  Restore every card, then re-run:")
    lines.append("      for i in " + " ".join(sorted(cards, key=lambda s: int(s)))
                 + f"; do sudo nvidia-smi -i $i -ac {want_mem},{want_gfx}; done")

    if allow_off_protocol:
        for line in ["WARNING, " + lines[0]] + lines[1:]:
            say("[certify] " + line)
        say(f"[certify] proceeding anyway on {OFF_PROTOCOL_OPT}: every timing "
            f"produced by this run was taken off protocol and may not be compared "
            f"with one that was not.")
        return

    raise RuntimeError("\n".join(lines) +
                       f"\n\n  To certify off protocol deliberately, pass "
                       f"{OFF_PROTOCOL_OPT} — the run then says so in its own output.")


_REGIME = _UNREAD


def _regime():
    """(regime_str, protocol_dict) for this machine's backend, read once.

    ('clock_lock', proto) on an NVIDIA rack, ('witness', proto) on Apple,
    (None, None) when the machine declares no protocol for its backend.
    """
    global _REGIME
    if _REGIME is not _UNREAD:
        return _REGIME
    path = _protocol_file()
    if path is None:
        _REGIME = (None, None)
        return _REGIME
    try:
        proto = json.loads(Path(path).read_text(encoding="utf-8"))
        _REGIME = (proto.get("regime", "clock_lock"), proto)
    except Exception:
        _REGIME = (None, None)
    return _REGIME


def _witness_time_ms(proto: Dict[str, Any]) -> float:
    """Time the protocol's witness kernel — a fixed-shape matmul, always the
    same — with the sweep's own timer (`L.do_bench`). The one number whose only
    variable across a sweep's open and close is the GPU regime: same shape, same
    inputs, same timer, so a change in it is a change in the machine (`autotune_harness.witness_ms`)."""
    spec = proto["witness"]
    if spec.get("kernel", "matmul") != "matmul":
        raise RuntimeError(f"witness kernel {spec.get('kernel')!r} not supported "
                           f"(only 'matmul')")
    return H.witness_ms(int(spec["M"]), int(spec["N"]), int(spec["K"]), spec.get("dtype", "fp16"),
                        BENCH_WARMUP_MS, BENCH_REP_MS)


def _witness_entry_refusal(proto: Dict[str, Any], path: Path, say) -> None:
    """Entry condition for the witness regime: the reference kernel must RUN, so
    the per-sweep drift check has a baseline it can take. It does NOT read or
    hold a clock — Apple's clock is OS-managed and unlockable; stability is
    proven per sweep in `certify_key`, not asserted here."""
    if "witness" not in proto:
        raise RuntimeError(
            f"protocol {path} declares regime 'witness' but carries no 'witness' "
            f"spec (shape, dtype, drift_tolerance): a witness regime with no "
            f"witness is a door with no hinge.")
    try:
        t = _witness_time_ms(proto)
    except Exception as exc:
        raise RuntimeError(
            f"the witness kernel from {path} could not be timed ({exc}). A "
            f"certification whose stability cannot be measured is not a "
            f"measurement — the same rule the clock lock enforces, by its own "
            f"means on this backend.")
    spec = proto["witness"]
    say(f"[certify] regime WITNESS ({path}): reference {spec.get('kernel','matmul')} "
        f"{spec['M']}x{spec['N']}x{spec['K']} {spec.get('dtype','fp16')} opens at "
        f"{t:.4f} ms; each sweep is bracketed and refused if it drifts past "
        f"{float(spec['drift_tolerance'])*100:.0f}%.")


def _backend() -> Dict[str, Any]:
    """The proof's generator stamp — C.generator_identity(), the ONE DOOR the
    gate reads back. Writing a different spelling here is how ec938641's gate
    refused all 945 Apple entries once, and how a pin move stayed invisible to
    it once more; both are documented on the door itself.

    ZERO FALLBACK: when the door cannot identify the generator it RAISES, and
    that must propagate — a certification stamped `{"name": "?"}` (the old
    silent fallback) mis-certifies under an unknown generator, exactly the
    mis-spelling this docstring warns against. A run that cannot say which
    compiler it is has no business writing a proof."""
    return dict(C.generator_identity())


def _phase(label: str) -> None:
    """NBX_CERTIFY_PHASES=1: one line per phase of a key with the process's memory now and at
    its peak so far (macOS `footprint`-style figures from `getrusage` and `mach`-free reading:
    resident bytes now, peak resident bytes). Written for the price door (2026-09-28): the
    certifier's peak is host memory in one phase, and a price that does not know which phase
    prices the wrong bytes."""
    if os.environ.get("NBX_CERTIFY_PHASES") != "1":
        return
    import resource, time as _t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss           # bytes on macOS
    try:
        import psutil                                                    # optional: resident now
        now = psutil.Process().memory_info().rss
    except Exception:                                                    # noqa: BLE001
        now = -1
    _log.debug(f"[phase] {_t.strftime('%H:%M:%S')} {label}: rss_now={now/2**20:.0f}MiB peak_rss={peak/2**20:.0f}MiB")


def certify_key(qual: str, tuner, key: tuple, tolerance: float, rng, bench=None,
                card_bytes: Optional[int] = None, unified: bool = False,
                num_stages: Optional[Tuple[int, ...]] = None, budget_bytes: Optional[int] = None,
                floor_bytes: int = 0, only_config: Optional[Dict[str, Any]] = None,
                reproven_from: Optional[str] = None) -> Dict[str, Any]:
    """Run every candidate of `tuner` on this key's inputs, compare each to the
    fp64 oracle, exclude beyond `tolerance`, time the rest; the entry (config,
    proof, excluded) for the directory. Raises when nothing survives.

    With `only_config` (a stored configuration, as the directory records it) the
    candidates are THAT one configuration: the oracle and one timing under the
    witness — a certificate made under a retired code generator is re-PROVEN, not
    re-swept (the owner, 2026-09-29 01:37, release-decisions); the proof then
    carries `reproven_from`, the generator label it was first proven under."""
    from nbxtensor.ops import autotune_cache as atc
    made = synthesize(qual, tuner, key, rng, card_bytes=card_bytes, unified=unified,
                      budget_bytes=budget_bytes, floor_bytes=floor_bytes)
    if made is None:
        raise RuntimeError(f"no synthesizer for {qual}")
    call, oracle_fn, out_name = made
    _phase("synthesized")
    oracle_box: Dict[str, Any] = {}
    bench = bench or (lambda fn: H.time_launch(fn, BENCH_WARMUP_MS, BENCH_REP_MS))
    state: Dict[str, Any] = {}
    saved_run = tuner.run

    def certifying_run(*args, **kwargs):
        # The seam the autotuner itself uses: its own key must be this one.
        tuner.nargs = dict(zip(tuner.arg_names, args))
        seen = atc.key_of(tuner, args, kwargs)
        if tuple(seen) != tuple(key):
            raise UnreachableCensusKey(
                f"the wrapper computed key {seen!r} for inputs synthesized from {key!r}: the census "
                f"and the kernel disagree — nothing certified for this key")
        names0 = list(tuner.arg_names)
        out_idx0 = next((i for i, n in enumerate(names0) if n == out_name), None)
        out0 = args[out_idx0] if out_idx0 is not None and out_idx0 < len(args) else None
        if "config" in state:
            # A LATER launch of the same key inside one wrapper call — a band of a product the
            # wrapper streams (see `H.launch_oracle`). It runs the configuration chosen on the
            # first launch, exactly as the runtime serves it, AND IS MEASURED against its own
            # oracle: the last band has another M, another grid and another edge mask, and an
            # entry that answers for N launch shapes is proven on each of them. A count is not
            # a comparison (the guardian, 2026-09-26).
            oracle_k = H.launch_oracle(qual, tuner, kwargs, out0)
            if oracle_k is None:
                raise RuntimeError(f"{qual} at {key!r}: a second launch of this key inside one "
                                   f"wrapper call, and no launch oracle covers it — nothing can "
                                   f"vouch for that band; not certified")
            ret, dev_k = H.launch_and_measure(tuner, args, kwargs, state["best_config"], out0, oracle_k)
            state["launches"] = int(state.get("launches", 1)) + 1
            state.setdefault("launch_deviations", [state["deviation"]]).append(dev_k)
            if not (dev_k <= state["tolerance"]):
                raise RuntimeError(f"{qual} at {key!r}: launch {state['launches']} of this key "
                                   f"({oracle_k.describe}) diverges from the fp64 oracle: "
                                   f"{dev_k:.3g} against {state['tolerance']:g} — the chosen "
                                   f"configuration does not hold on every band; not certified")
            state["deviation"] = max(float(state["deviation"]), float(dev_k))
            return ret
        t_or = time.time()
        # The oracle is THIS LAUNCH's, from its own operands, wherever the brick covers it; the
        # synthesized oracle for the families whose key names a single launch.
        oracle = H.launch_oracle(qual, tuner, kwargs, out0)
        if oracle is None:
            oracle = oracle_box.get("v")
            if oracle is None:
                oracle = oracle_box["v"] = oracle_fn()
        _phase("oracle built")
        state["t_oracle"] = round(time.time() - t_or, 3)
        state["oracle"] = ORACLE + (" " + oracle.describe if hasattr(oracle, "describe") else "")
        if only_config is not None:
            configs = [H.stored_config(only_config)]          # re-prove: the stored configuration, alone
        else:
            configs = restrict_num_stages(H.candidate_configs(tuner, kwargs), num_stages)
        # every config run once on a poisoned output, measured against the oracle on its windows, its run timed
        launch = H.LaunchUnderTest(tuner, args, kwargs, out_name, qual)
        t_runs = time.time()
        results, excluded, unrun = launch.sweep(configs, oracle, tolerance)
        _phase("deviation measured")
        if not results:
            if not excluded and unrun:
                raise RuntimeError(f"{qual} at {key!r}: no config could run ({len(unrun)} of {len(configs)}; "
                                   f"first: {unrun[0]['error']})")
            # The deviations are the whole diagnosis and were being thrown away: "beyond 0.04"
            # cannot distinguish a kernel that misses by a hair from one that is wrong, and the
            # two have nothing to do with each other. The BEST config's deviation says which
            # (2026-09-22: 21 depthwise bf16 keys refused, every config excluded, and the
            # message named no number to act on).
            best = min((e["deviation"] for e in excluded), default=float("nan"))
            # `tolerance` of 0 is legitimate — a profile may demand an exact match — and
            # `_tolerance` accepts it (it refuses a MISSING one, not a zero one). Dividing by
            # it would raise ZeroDivisionError INSIDE this error path and replace the
            # diagnosis with a traceback: exactly the failure this message was added to fix
            # (the rack found it on a277dd39, 2026-09-23; latent, no profile declares 0 today).
            ratio = f", {best / tolerance:.1f}x tolerance" if tolerance else " against an EXACT-match tolerance"
            raise RuntimeError(f"{qual} at {key!r}: every config diverges from the fp64 oracle beyond {tolerance:g} "
                               f"(best {best:.3g}{ratio}; "
                               f"{len(excluded)} excluded, {len(unrun)} could not run"
                               + (f"; first error: {unrun[0]['error']}" if unrun else "") + ")")
        state["t_runs"] = round(time.time() - t_runs, 3)
        t_bench = time.time()
        contenders = _contenders(results)
        # STABILITY REGIME. The candidate timings below decide `best_ms` by
        # comparison, so they are only comparable if the machine held still
        # across them. On an NVIDIA rack that is the clock lock, read at entry.
        # On Apple the clock is OS-managed and unlockable, so stability is
        # PROVEN, not asserted: a fixed reference kernel is timed just before and
        # just after this sweep, and if it drifts past the profile's tolerance
        # the regime moved while we compared and the sweep is refused.
        _regime_kind, _proto = _regime()

        def _witness_ms():
            """Time the witness on the UNPATCHED path.

            `certifying_run` IS the sweep, and it is installed AS `tuner.run`.
            The witness is itself a matmul (`_nbx_linalg_wrappers.mm`), so on a matmul
            certification it re-enters this very function: the nested call
            computes the witness's own key and compares it to the candidate's.

            Measured 2026-09-17, the first run after the protocol became
            findable at all: every fp16 matmul shape came back "UNREACHABLE —
            the wrapper computed key (512, 512, 512)", which is the witness's
            shape and not the census's, and the (512,512,512) shape itself
            FAILED with all ten configs "diverging from the fp64 oracle"
            because the witness's own launches were being taken for candidate
            runs. 0 certified where the unwitnessed run had certified 6.

            Suspending the patch keeps the witness on the real path — it must
            measure the same machine the candidates are measured on — while
            stopping it from being mistaken for one of them.
            """
            tuner.run = saved_run
            try:
                return _witness_time_ms(_proto)
            finally:
                tuner.run = certifying_run

        def _time_one(cfg):
            return launch.time(cfg, bench)

        if _regime_kind == "witness":
            timed, state["stability_witness"] = witnessed_bench(
                contenders, _time_one, _witness_ms, _proto["witness"], f"{qual} at {key!r}")
        else:
            timed = sorted(((cfg, dev, _time_one(cfg)) for cfg, dev, _run_s in contenders),
                           key=lambda t: t[2])
        state["t_bench"] = round(time.time() - t_bench, 3)
        best, dev, ms = timed[0]
        # THE MATRIX UNIT, a candidate beside the FMA configurations (the list of what remains 1.8, the owner's rule
        # 2026-11-03 17:48): a GEMM the unit can take is run at each of the profile's unit tiles, proven against the
        # same oracle and timed with the same bench; the faster route is the one the runtime serves.
        units = launch.sweep_units(H.unit_runs(tuner, args, kwargs), oracle, tolerance, bench, excluded, unrun)
        if units:
            utile, udev, ums = min(units, key=lambda t: t[2])
            state["unit_measured"] = {"tiles": [{"tile": t, "deviation": d, "ms": m} for t, d, m in units],
                                      "fma_best_ms": ms, "chosen": "unit" if ums < ms else "fma"}
            if ums < ms:
                state["unit"] = {"tile": utile, "deviation": udev, "best_ms": ums, "fma_best_ms": ms}
        state.update({"config": H.config_dict(best), "deviation": dev, "best_ms": ms,
                      "second_ms": timed[1][2] if len(timed) > 1 else None,
                      "candidates": len(configs), "accepted": len(results), "benched": len(contenders),
                      "excluded": excluded, "unrun": unrun,
                      "timings": [{"config": H.config_dict(c), "deviation": d, "ms": m} for c, d, m in timed]})
        tuner.cache[key] = best
        state["best_config"] = best
        state["launches"] = 1
        state["tolerance"] = float(tolerance)
        return launch.final(best)

    # The wrappers decide the kernel's dtypes and flags (PROMOTE_B, IEEE_PRECISION,
    # a conv's output dtype) from the hardware profile and the component's compute
    # dtype the engine hands them — reproduce that context, or the wrapper computes
    # another key than the census's (the first dry run: every matmul key mismatched
    # because a bare process reads `has_native_bf16` as True).
    from nbxtensor.ops import wrappers as W
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    _prev_dt = W.get_compute_dtype()
    _out_dt = C.output_dtype(tuner, key)
    _nbx = {"fp16": NBXDtype.float16, "bf16": NBXDtype.bfloat16, "fp32": NBXDtype.float32}.get(_out_dt)
    tuner.run = certifying_run
    fell_back = []
    C.certifying(True)          # a GEMM the matrix unit could take runs its FMA kernel here: its sweep is the point
    try:
        if _nbx is not None:
            W.set_compute_dtype(_nbx)
        # Did the kernel actually BUILD on the device, or did the backend fail
        # to compile it and fall back to the CPU?
        #
        # The screen cannot answer that. A CPU fallback COMPUTES CORRECTLY, so
        # its deviation against the fp64 oracle is excellent — 1e-6 like any
        # sound path — and an entry certified on it would record a
        # configuration chosen for a path that never runs. Measured
        # 2026-09-11: 30 of 30 entries do build, but the proof did not say so,
        # and a reader six months from now could not redo the check.
        #
        # What is measured and not written does not exist.
        # The other context a wrapper reads: the component's fp16 precision contract
        # (`wrappers._NBX_ACTIVATIONS_FP16_SAFE`, the calibration's verdict) — under it an fp16 GEMM
        # stores fp16 on hardware without native bf16 (`launch_keys.matmul_out_dtype(contract_half)`),
        # and the census keys such a component's GEMMs so. The key does not name the flag: it is
        # tried off first (the plain context), then on — the context that forms THIS key is the one
        # certified, and the proof says which (2026-10-26: 178 fp16-store keys read unreachable with
        # the flag never set, the census right and the certifier blind to the contract).
        _prev_safe = W.get_activations_fp16_safe()
        for _contract in (False, True):
            W.set_activations_fp16_safe(_contract)
            try:
                with warnings.catch_warnings(record=True) as _caught:
                    warnings.simplefilter("always")
                    call()
                    fell_back = [str(w.message).splitlines()[0][:120] for w in _caught
                                 if "fall back to CPU" in str(w.message)
                                 or "Metal compilation failed" in str(w.message)]
                state["contract"] = _contract
                break
            except UnreachableCensusKey:
                if _contract or "config" in state:
                    raise
    finally:
        C.certifying(False)
        W.set_compute_dtype(_prev_dt)
        try:
            W.set_activations_fp16_safe(_prev_safe)
        except NameError:
            pass
        tuner.run = saved_run
        tuner.nargs = None
    if "config" not in state:
        raise RuntimeError(f"{qual} at {key!r}: the wrapper never reached the autotuner")
    proof = {"date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
             "engine_version": _engine_version(), "backend": _backend(), "shape": list(key),
             "launches": int(state.get("launches", 1)),
             "fp16_contract": bool(state.get("contract", False)),   # the context the key was formed under
             "launch_deviations": [float(d) for d in state.get("launch_deviations", [state.get("deviation")])],
             "deviation": state["deviation"], "tolerance": tolerance, "oracle": state.get("oracle", ORACLE), "machine": _machine(),
             "best_ms": state["best_ms"], "second_ms": state["second_ms"], "candidates": state["candidates"],
             "accepted": state["accepted"], "benched": state["benched"], "could_not_run": len(state["unrun"]),
             "seconds": {"oracle": state.get("t_oracle"), "runs": state.get("t_runs"), "bench": state.get("t_bench")},
             # The stability regime this sweep was measured under: the machine's
             # clock is in `machine.clocks_mhz` (a held lock, NVIDIA), and the
             # witness is here (a proven drift, Apple). One of the two is present
             # whenever a protocol was declared; `proof_records_regime` reads
             # either, and an entry with neither is not served.
             "stability_witness": state.get("stability_witness"),
             **({"unit_measured": state["unit_measured"]} if state.get("unit_measured") else {}),
             **({"reproven_from": reproven_from} if reproven_from else {}),
             "built": {"gpu": not fell_back,
                       "how": "no backend compilation fallback was raised during "
                              "the certifying run",
                       "fallback": fell_back or None}}
    return {"config": state["config"], "proof": proof, "excluded": state["excluded"],
            "could_not_run": state["unrun"], "timings": state["timings"],
            **({"unit": state["unit"]} if state.get("unit") else {})}



def _bind_hardware_profile() -> str:
    """Hand Prism's hardware profile to the wrappers exactly as the CLI does
    before a request (`set_hardware_profile`): the flag `has_native_bf16` and
    the per-device VRAM the wrappers read for their dtype decisions."""
    from neurobrix.core.prism import load_profile
    from neurobrix.core.prism.autodetect import get_or_create_default_profile
    from nbxtensor.ops import wrappers as W
    hw_id = get_or_create_default_profile()
    from neurobrix.triton.dtype import declare_hardware_profile
    declare_hardware_profile(load_profile(hw_id))
    return str(hw_id)


def _has_native_bf16() -> bool:
    from nbxtensor.ops import wrappers as W
    return bool(W.has_native_bf16())


def _engine_version() -> str:
    try:
        from neurobrix import __version__
        return str(__version__)
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# the run: a profile, its shapes, its files
# ---------------------------------------------------------------------------
def _num_stages_space(vendor: str, profile: str) -> Optional[Tuple[int, ...]]:
    """The software-pipelining depths the profile allows certification to try
    (`autotune.certify_num_stages`), or None when it declares none (every depth the kernel's
    configurations carry, as before). Read from the device whole (the library's device file merged with the engine's
    vendor file, `loader.vendor_file`): `autotune` is the library's half — read from the engine's file alone it would
    vanish and every depth be tried (and a missing or unreadable file is a refusal, not a space of every depth)."""
    from neurobrix.core.config.loader import vendor_file
    cfg = vendor_file(Path(__file__).resolve().parents[1] / "config" / "vendors" / vendor / f"{profile}.yml")
    space = (cfg.get("autotune") or {}).get("certify_num_stages")
    if space is None:
        return None
    depths = tuple(sorted({int(d) for d in space}))
    if not depths or min(depths) < 1:
        raise RuntimeError(f"{vendor}/{profile}: autotune.certify_num_stages {space!r} names no valid depth")
    return depths


def restrict_num_stages(configs: list, depths: Optional[Tuple[int, ...]]) -> list:
    """Every tile shape of `configs`, at each allowed pipelining depth — not the configurations
    that happen to carry an allowed depth. Filtering would drop a tile the kernel only listed at
    depth 3 or 4; the depth is a separate axis of the space, so each tile is re-offered at each
    allowed depth, duplicates removed, order kept. None leaves the list untouched."""
    if not depths:
        return configs
    import triton
    out, seen = [], set()
    for cfg in configs:
        for d in depths:
            ident = (tuple(sorted(cfg.kwargs.items())), cfg.num_warps, d, cfg.num_ctas, cfg.maxnreg)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(triton.Config(dict(cfg.kwargs), num_warps=cfg.num_warps, num_stages=d,
                                     num_ctas=cfg.num_ctas, maxnreg=cfg.maxnreg))
    return out


def _tolerance(vendor: str, profile: str, dtype: str) -> float:
    tol = C._tolerance_for(vendor, profile, dtype)
    if tol is None:
        raise RuntimeError(f"the profile {vendor}/{profile} declares no `autotune_screen_rtol` for {dtype}: "
                           f"certification will not invent a tolerance")
    return tol


def _write_file(path: Path, vendor: str, profile: str, qual: str, dtype: str, fresh: Dict[str, Dict],
                cache: Optional[Dict[str, Dict]] = None) -> None:
    """Place this writer's FRESH certifications (key -> config, proof, excluded) into the kernel's
    file as it stands on disk now, each by the memory class its proof names, and write it back.

    Two certifiers write one kernel's file at once — one per memory class, pinned to two cards
    of one profile. The first repair (2026-09-21) merged by KEY: the disk's entries were added
    to the writer's own only where the writer had none. That kept disjoint keys, but a key both
    classes hold is ONE entry (a primary and its class variants): the writer's copy, read at the
    start of its pass, won whole — and erased every variant the other card had filed since. On
    2026-09-28/29 the 16 GB GEMM pass rewrote `baddbmm_kernel.fp32.json` after each of its keys
    for two hours; the 32 GB card's variant of eleven shared keys was absent from every
    checkpoint in that window, and the 32 GB card certified the same eleven again on every pass.
    Now nothing a writer read earlier is written back: the file under the lock is the base, and
    only what this writer just proved is placed into it (`file_certification`, the one placement
    rule). `cache`, the caller's view of the file, is refreshed to what was written."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".json.lock")
    with open(lock, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            entries = _read_file(path)
            for ktext, cert in fresh.items():
                C.file_certification(entries, ktext, cert)
            doc = {"format": C.format_for(entries), "vendor": vendor, "profile": profile, "kernel": qual, "dtype": dtype,
                   "entries": dict(sorted(entries.items()))}     # the stamp is what every entry satisfies, never the writer's era
            tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    if cache is not None:
        cache.clear()
        cache.update(entries)


def _file_entries(cache: Dict[str, Dict[str, Dict]], dtype: str, path: Path) -> Dict[str, Dict]:
    """The certify loop's view of one kernel file, READ ONCE per dtype and pass. It was
    `cache.setdefault(dtype, _read_file(path))`, whose default is evaluated on every call: every
    key of the census re-read and re-parsed its whole kernel file (up to 21 MB, 0.25 s) and threw
    it away — the 16 and 32 GB GEMM certifiers sat 8-10 minutes in it before their first key, the
    card idle (py-spy, 2026-09-29), which the supervisor saw as the cards waiting between passes. The view is refreshed by `_write_file` after
    each certification, so reading it once loses nothing."""
    entries = cache.get(dtype)
    if entries is None:
        entries = cache[dtype] = _read_file(path)
    return entries


def _read_file(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(doc.get("entries") or {}) if isinstance(doc, dict) else {}


_STICKY_MARKS = ("error 700", "rc=700", "illegal memory access", "STICKY")


def sticky_cuda_error(exc: BaseException) -> bool:
    """True when a key's failure says the CUDA context itself is dead. After an
    illegal memory access every later launch and malloc in the process fails
    with the same error, so nothing after it is a measurement: on 2026-09-14
    07:38 a certifier met one such key (a 2.19e9-element matmul on a kernel
    that still wrapped) and went on to report 227 further keys FAILED for
    "GPU malloc failed (error 700) for 256 bytes" — 227 shapes lost to a
    fault that happened once."""
    text = str(exc)
    return any(m in text for m in _STICKY_MARKS)


_OVERSIZE = re.compile(r"GPU malloc failed \(error 2\) for (\d+) bytes.*?driver_total=(\d+)MB", re.S)


def oversize_for_class(exc: BaseException):
    """The bytes asked and the card's total when a key's tensors cannot FIT the memory class,
    else None.

    This is not a certification failure, it is a CENSUS defect arriving late: Wan2.1-T2V's
    video VAE projects 81 frames of 722x1282x96 to RGB, whose input alone is 28.8 GB, and the
    16 GB census recorded it because the shadow planned the op at its graph shape. A real
    16 GB run never forms that key — Prism tiles the decode long before it — so no certified
    entry is owed for it and reporting it beside genuine failures buries both. Named, counted
    apart, and left for the census to stop recording (2026-09-22)."""
    if isinstance(exc, H.KeyTooLargeForClass):
        return (exc.asked, exc.card)
    m = _OVERSIZE.search(str(exc))
    if not m:
        return None
    asked, total_mb = int(m.group(1)), int(m.group(2))
    return (asked, total_mb * 1024 * 1024) if asked > total_mb * 1024 * 1024 else None


def _alloc_state() -> str:
    """Live bytes, pool-cached bytes and the pool's flush/evict counters, for the per-key line.

    Certification on this Mac reached a 42 GB physical footprint on keys whose operands are
    about 8 GB (M_BUCKET=163840 addmm), driving swap to 27.9 GB of 28.7 and taking jetsam
    kills. Whether the excess is live, pool-cached or neither is not readable from outside the
    process, and a footprint alone cannot say. Printed per key so a run answers it instead of
    another round of reading the code. Never raises: a probe that can end a sweep is worse than
    no probe."""
    try:
        from nbxtensor.memory.allocator import DeviceAllocator as A
        parts = [f"live {A.memory_allocated() / 2**20:.0f}MB", f"pool {A.pool_cached_bytes() / 2**20:.0f}MB"]
        if not A.pool_enabled():
            parts.append("pool OFF")
        st = A.pool_stats()
        if st["flushes"] or st["evictions"]:
            parts.append(f"flush {st['flushes']}/evict {st['evictions']}")
        return ", ".join(parts)
    except Exception:  # noqa: BLE001
        return "alloc ?"


def _release_between_keys() -> None:
    """Give the device back between keys: the operands collected, the pool's free blocks returned
    (`release_flow_memory`).

    The certifier held nothing deliberately and freed nothing either: a key's synthetic
    operands are released when Python collects them, which had not happened by the time the
    next key asked for its own. Measured 2026-09-22 on the 16 GB class — a matmul needing
    11.0 GiB was refused on a 15.8 GiB card with `live_tracked=5632MB` still resident from
    the key before it. The same day, a drain looked up by guessed names resolved to nothing
    and a whole Apple campaign ran with it inert: the library's release is called by name."""
    from nbxtensor.memory.memory_pool import release_flow_memory
    release_flow_memory()


def after_key_failure(exc: BaseException, summary: Dict[str, Any], key_text: str, log) -> bool:
    """Count a key's failure; return True when the run must STOP because the
    context is poisoned (the summary then names the key and the reason)."""
    big = oversize_for_class(exc)
    if big is not None:
        asked, card = big
        summary["oversize"] = summary.get("oversize", 0) + 1
        usable = getattr(exc, "usable", card)
        log(f"[certify] TOO LARGE FOR THIS CLASS at {key_text}: the key's tensors ask "
            f"{asked / 2**30:.1f} GiB of the {usable / 2**30:.1f} GiB a {card / 2**30:.1f} GiB card plans against. "
            f"No entry is owed — a run "
            f"of this class never forms this key, because the plan tiles before it. The CENSUS "
            f"recorded a shape this class cannot reach; that is where it is fixed.")
        return False
    summary["failed"] = summary.get("failed", 0) + 1
    if sticky_cuda_error(exc):
        summary["aborted"] = {"key": key_text, "reason": str(exc)[:300]}
        log(f"[certify] ABORTED at {key_text}: the CUDA context is poisoned ({str(exc)[:160]}) — "
            f"every launch after this would fail the same way; nothing after it is a measurement. "
            f"Re-run from this key once the cause is fixed.")
        return True
    return False


def census_tuners(shapes) -> Dict[str, Any]:
    """{kernel qual: its Autotuner} for every kernel the census names, its module loaded first: a kernel module the
    wrappers import lazily (inside the call that launches it) is not in the process until a launch, and
    `autotune_cache._autotuners` only sees what is loaded — every key of `gemv_vec_kernel` read "not an autotuner"
    and was skipped (2026-10-22)."""
    import importlib
    from nbxtensor.ops import autotune_cache as atc
    for qual in shapes:
        importlib.import_module(qual.rsplit(".", 1)[0])
    return {qual: t for qual, t in atc._autotuners()}


def certify(profile: str, vendor: Optional[str] = None, census_path: Optional[str] = None, bench=None,
            out: Optional[str] = None, kernels: Optional[List[str]] = None, limit: Optional[int] = None,
            only_missing: bool = False, log=None,
            allow_off_protocol: bool = False, reprove_unclocked: bool = False,
            reprove_generator: bool = False, working_set_mb: Optional[int] = None,
            models: Optional[List[str]] = None, skip_proven_since: Optional[str] = None,
            skip_keys: Optional[str] = None, keys_path: Optional[str] = None) -> Dict[str, Any]:
    """Certify every census shape for `profile` on this machine; write the files.

    `working_set_mb`: the budget this run may hold — every key is priced by its phases before any
    draw and refused by name over it (`price_key`, `_refuse_unpriced`). It is the figure the run's
    guard kills at, passed by the same script, so the door and the guard agree on one number."""
    if log is None:
        def log(*a):                      # a run of hours, read while it runs: never buffered
            print(*a, flush=True)
    # The entry condition, before anything is timed. It lives HERE and not in the
    # CLI because this is the narrowest point: the door then holds for every
    # caller, not only for the one that types the documented command.
    rig_protocol_refusal(allow_off_protocol=allow_off_protocol, say=log)
    active = C.active_profile()
    if active is None:
        raise RuntimeError("no vendor profile is in force on this machine (the launcher resolved none)")
    if vendor is None:
        vendor = active[0]
    if (vendor, profile) != active:
        raise RuntimeError(f"this machine carries the profile {active[0]}/{active[1]}, not {vendor}/{profile}: a "
                           f"certification is measured on the profile it names")
    root = Path(out) if out else C.directory()
    hw_id = _bind_hardware_profile()
    log(f"[certify] hardware profile {hw_id}: has_native_bf16={_has_native_bf16()} — the wrappers' dtype policy in force")
    from nbxtensor.ops import autotune_cache as atc
    tuners = {qual: t for qual, t in atc._autotuners()}
    # The card this run certifies on, read once (register 56): every proof
    # names it, `--only-missing` asks per its memory class, and a card the
    # profile in force does not describe is refused at entry — an entry made
    # on it could never say its class and would be served to no card.
    certifying_device = _certifying_device()
    certifying_class = C.memory_gb((certifying_device or {}).get("memory_mb"))
    # The working-set budget (`--working-set-mb`): what this run may hold, the figure its guard
    # kills at; None = no price door (the operand bound alone). The floor is the process now.
    budget_bytes = int(working_set_mb) * 2**20 if working_set_mb else None
    from neurobrix.core.prism.host_footprint import process_footprint_now
    floor_bytes = process_footprint_now() if working_set_mb else 0
    if budget_bytes is not None:
        log(f"[certify] working-set budget {budget_bytes / 2**20:.0f} MiB; the process's floor now "
            f"{floor_bytes / 2**20:.0f} MiB, read again before every key; every key is priced by its "
            f"phases before any draw")
    if certifying_class is None:
        raise RuntimeError("the certifying card is not described by the hardware profile in force "
                           f"(device {certifying_device}): refused — a proof must say which card's memory it was made on")
    log(f"[certify] certifying on {certifying_device['name']} ordinal {certifying_device['ordinal']} "
        f"(CUDA_VISIBLE_DEVICES={certifying_device['visible_devices']}), "
        f"{certifying_device['memory_mb']} MB = memory class {certifying_class} GB; entries serve that class only")
    if census_path:
        raise RuntimeError(f"--census {census_path}: REFUSED — the certifier certifies the committed census table of "
                           f"the profile and the card's memory class, and nothing else (the owner's method, 2026-09-28)")
    if models:
        from neurobrix.kernels import census_table as _T
        _known = {r.get("model") for r in _T.read(_T.table_path(vendor, profile, certifying_class))}
        _unknown = sorted(set(models) - _known)
        if _unknown:
            raise RuntimeError(f"--models {', '.join(_unknown)}: no row of the census table names it — a filter that "
                               f"matched nothing would certify nothing and report success")
    shapes = keys_file(keys_path) if keys_path else census(vendor, profile, certifying_class, models)
    tuners = census_tuners(shapes)
    log(f"[certify] census table {vendor}/{profile} {certifying_class} GB: "
        f"{sum(len(v) for v in shapes.values())} distinct key(s) over {len(shapes)} kernel(s)")
    if kernels:
        want = set(kernels)
        shapes = {q: ks for q, ks in shapes.items() if q in want or C.kernel_short(q) in want}
    rng = np.random.default_rng(20260907)   # one draw for every certification: a key's witness is reproducible
    summary: Dict[str, Any] = {"vendor": vendor, "profile": profile, "directory": str(root), "kernels": {},
                               "certified": 0, "skipped": 0, "failed": 0, "unreachable": 0,
                               "excluded_configs": 0, "started": time.time()}
    done = 0
    attempts = 0                                  # `limit` bounds the shapes TRIED, failures included
    # Certifications reach their file at a BOUNDED SHARE of the pass (`_BoundedWriter`); every
    # pending one is written before this function returns, on every path.
    writer = _BoundedWriter(vendor, profile)
    try:
        return _certify_loop(shapes, tuners, vendor, profile, root, limit, only_missing, reprove_unclocked,
                             reprove_generator, certifying_device, certifying_class, budget_bytes,
                             floor_bytes, rng, summary, log, writer, bench, skip_proven_since,
                             _read_skip_keys(skip_keys))
    finally:
        writer.flush_all()


_WRITE_SHARE = 0.10     # a certified file's rewrites may cost at most a tenth of the certifier's time


class _BoundedWriter:
    """When a certification reaches its kernel file. A kernel file is 13-21 MB of JSON, and
    rewriting it after every key took 36.7 % of a 16 GB conv pass (py-spy, 2026-09-29: `_write_file`
    beside 36.2 % of bench waits) while the card sat idle. A file is written when the time since its
    last write reaches that write's own duration over `_WRITE_SHARE` — the rewrites then cost at most
    that share of the pass — and `flush_all` writes whatever is pending. Each write still places only
    what this writer proved into the file as it stands, under the file's lock (`_write_file`)."""

    def __init__(self, vendor: str, profile: str, share: float = _WRITE_SHARE, write=None, clock=time.time):
        self.vendor, self.profile, self.share, self.clock = vendor, profile, share, clock
        self.write = write or _write_file
        self.pending: Dict[Path, Dict[str, Dict]] = {}
        self.where: Dict[Path, Tuple[str, str, Dict[str, Dict]]] = {}
        self.last: Dict[Path, Tuple[float, float]] = {}

    def add(self, path: Path, qual: str, dtype: str, cache: Dict[str, Dict], ktext: str, entry: Dict) -> None:
        self.pending.setdefault(path, {})[ktext] = entry
        self.where[path] = (qual, dtype, cache)
        at, cost = self.last.get(path, (float("-inf"), 0.0))
        if self.clock() - at >= cost / self.share:
            self.flush(path)

    def flush(self, path: Path) -> None:
        fresh = self.pending.pop(path, None)
        if not fresh:
            return
        qual, dtype, cache = self.where[path]
        t0 = self.clock()
        self.write(path, self.vendor, self.profile, qual, dtype, fresh, cache=cache)
        t1 = self.clock()
        self.last[path] = (t1, t1 - t0)

    def flush_all(self) -> None:
        for path in list(self.pending):
            self.flush(path)


def _read_skip_keys(path: Optional[str]):
    """{(kernel qualname, key repr)} from a file of `<qualname> <key repr>` lines — the keys a resumed sweep has done
    (when a re-proof since has re-dated every entry, a date cannot tell them apart). None without a file; a missing file
    is refused by name."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"--skip-keys {path}: no such file")
    out = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            qual, _, key = line.partition(" ")
            out.add((qual, key))
    return out


def _certify_loop(shapes, tuners, vendor, profile, root, limit, only_missing, reprove_unclocked,
                  reprove_generator, certifying_device, certifying_class, budget_bytes, floor_bytes,
                  rng, summary, log, writer, bench, skip_proven_since=None, skip_keys=None):
    done = 0
    attempts = 0
    # keys this class refused by size: a pass that only fills what is missing does not draw them again while the
    # bound they were refused at holds (Sana 4K's two 16g rms keys were retried, and failed, at every pass)
    from neurobrix.core.config.system import PRISM_DEFAULTS
    card_bytes = int(certifying_device["memory_mb"]) * 2**20
    usable_now = int(card_bytes * float(PRISM_DEFAULTS["safety_margin"]))
    refusals = refused_path(vendor, profile, certifying_class, root=root)
    refused = read_refused(refusals)
    for qual, keys in shapes.items():
        tuner = tuners.get(qual)
        if tuner is None:
            log(f"[certify] {qual}: not an autotuner in this engine — skipped ({len(keys)} shapes)")
            summary["skipped"] += len(keys)
            continue
        per_dtype: Dict[str, Dict[str, Dict]] = {}
        for key in keys:
            if limit is not None and attempts >= limit:
                break
            dtype = C.output_dtype(tuner, key)
            path = C.file_for(vendor, profile, qual, dtype, root=root)
            entries = _file_entries(per_dtype, dtype, path)
            ktext = C.key_repr(key)
            if skip_keys and (qual, ktext) in skip_keys:
                continue                      # listed done by the caller (a resumed sweep: `--skip-keys`)
            if skip_proven_since is not None:
                # resume a sweep: a key this card's class re-proved at or after the given time (ISO, UTC like the
                # proofs) is done — a stopped re-sweep restarts where it was instead of from the first key
                if C.class_proven_since(C.entry_for_memory_gb(entries.get(ktext), certifying_class), skip_proven_since):
                    continue
            owed = None
            if C.coverage_asked(entries, ktext, certifying_class, only_missing, reprove_unclocked, reprove_generator,
                                C.proof_backend({"backend": _backend()})):
                # certified FOR THIS CARD's memory class already, under the running code generator (the one the
                # runtime serves; --only-missing asks it too), and, with --reprove-unclocked, at a recorded clock —
                # unless the matrix unit's measurement is still owed to it (`unit_owed`): then the stored
                # configuration is kept, proven and timed once, and the unit's tiles are measured beside it
                owed = (C.certification_for_generator(C.entry_for_memory_gb(entries.get(ktext), certifying_class),
                                                      C.proof_backend({"backend": _backend()})) if only_missing else None)
                if not unit_owed(qual, key, owed):
                    continue
            _r = refused.get((qual, ktext))
            if only_missing and _r is not None and int(_r.get("usable_bytes", -1)) == usable_now:
                summary["refused_known"] = summary.get("refused_known", 0) + 1
                continue                      # refused by size at this very bound: named in the register, not owed
            # RE-PROVE, NOT RE-SWEEP (the owner, 2026-09-29 01:37): a certificate this class holds under
            # another code generator keeps its configuration — proven again here by the oracle and one
            # timing under the witness. A sweep only where that configuration fails the oracle, or where
            # there is no entry at all; the summary says how many of each and why.
            # The class head's configuration is re-proven; its proof is ADDED beside the other generators' (`file_certification`).
            stored = C.entry_for_memory_gb(entries.get(ktext), certifying_class) if reprove_generator else None
            stored_label = C.reproof_source(stored, C.proof_backend({"backend": _backend()}))
            reprove = stored_label is not None
            attempts += 1
            t0 = time.time()
            if budget_bytes is not None:
                # the floor is the process NOW, before this key — never the one read at the start
                from neurobrix.core.prism.host_footprint import process_footprint_now
                floor_bytes = process_footprint_now()
            try:
                tol = _tolerance(vendor, profile, dtype)
                common = dict(card_bytes=int(certifying_device["memory_mb"]) * 2**20,
                              unified=bool(certifying_device.get("unified")),
                              num_stages=_num_stages_space(vendor, profile),
                              budget_bytes=budget_bytes, floor_bytes=floor_bytes, bench=bench)
                if owed is not None:
                    entry = certify_key(qual, tuner, key, tol, rng, only_config=owed["config"],
                                        reproven_from=f"the unit measured beside the FMA configuration proven "
                                                      f"{owed.get('proof', {}).get('date', '?')}", **common)
                    summary["unit_owed"] = summary.get("unit_owed", 0) + 1
                elif reprove:
                    try:
                        entry = certify_key(qual, tuner, key, tol, rng, only_config=stored["config"],
                                            reproven_from=stored_label, **common)
                        summary["reproven"] = summary.get("reproven", 0) + 1
                    except (UnreachableCensusKey, H.KeyTooLargeForClass, WitnessDrift):
                        raise                         # a drift is a refusal for the retry pass, never a sweep
                    except Exception as exc:          # the stored configuration fails the oracle under
                        log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: the stored "
                            f"configuration does not re-prove under the running generator ({exc}); swept")
                        entry = certify_key(qual, tuner, key, tol, rng, **common)
                        summary["swept"] = summary.get("swept", 0) + 1
                        summary.setdefault("swept_why", {}).setdefault("stored configuration failed the oracle", 0)
                        summary["swept_why"]["stored configuration failed the oracle"] += 1
                else:
                    entry = certify_key(qual, tuner, key, tol, rng, **common)
                    if reprove_generator:
                        summary["swept"] = summary.get("swept", 0) + 1
                        why = "no entry for this class" if not stored else "no configuration in the stored entry"
                        summary.setdefault("swept_why", {})[why] = summary.get("swept_why", {}).get(why, 0) + 1
            except UnreachableCensusKey as exc:
                # Known debt, not a break: no run will ever present this key
                # again. Counted apart so the exit code can still mean something.
                summary["unreachable"] += 1
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: UNREACHABLE — {exc}")
                continue
            except Exception as exc:
                _release_between_keys()
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: FAILED — {exc}")
                _big = oversize_for_class(exc)
                if _big is not None:
                    record_refusal(refusals, qual, ktext, _big[0], getattr(exc, "usable", usable_now), _big[1],
                                   str(exc)[:300])
                if after_key_failure(exc, summary, ktext, log):
                    summary["seconds"] = round(time.time() - summary["started"], 1)
                    return summary                    # a poisoned context: stop, say it, exit non-zero
                continue
            _release_between_keys()
            try:
                C.file_certification(entries, ktext, entry)   # by the class its proof names; refused without one
            except ValueError as exc:
                summary["failed"] += 1
                log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: REFUSED — {exc}")
                continue
            writer.add(path, qual, dtype, entries, ktext, entry)
            done += 1
            summary["certified"] += 1
            summary["excluded_configs"] += len(entry["excluded"])
            k = summary["kernels"].setdefault(qual, {"certified": 0, "excluded_configs": 0})
            k["certified"] += 1; k["excluded_configs"] += len(entry["excluded"])
            p = entry["proof"]
            log(f"[certify] {C.kernel_short(qual)} {dtype} {C.describe_key(tuner, key)}: {entry['config']['kwargs']} "
                f"warps={entry['config']['num_warps']} stages={entry['config']['num_stages']} — deviation {p['deviation']:.2e} "
                f"(tol {tol:g}), {p['best_ms']:.4f} ms, {p['accepted']}/{p['candidates']} accepted, "
                f"{len(entry['excluded'])} excluded, {time.time() - t0:.1f} s "
                f"(oracle {p['seconds']['oracle']}, runs {p['seconds']['runs']}, bench {p['seconds']['bench']}; "
                f"{_alloc_state()})")
    summary["seconds"] = round(time.time() - summary["started"], 1)
    return summary
