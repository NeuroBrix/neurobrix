#!/usr/bin/env python3
"""Attribute a request's wall clock, phase by phase, against an nsys timeline.

Inputs: a daemon/CLI log carrying `NBX_PHASE_TRACE=1` stamps
(`[phase] <name> wall=<epoch s> ...`, see core/runtime/phase_trace.py) and
the sqlite export of the nsys profile that wrapped the SAME process
(`nsys export --type sqlite`). nsys stamps its events relative to the
session start whose UTC epoch is in TARGET_INFO_SESSION_START_TIME, so the
two clocks meet on the epoch axis.

For every consecutive pair of stamps of one request the tool prints the
wall time, the GPU-busy time (union of kernel + memcpy + memset intervals
on the device), the busy %, the kernel count, the host time inside CUDA
runtime API calls (by API name), the H2D/D2H bytes, and the top kernels
by time. Requests are split at `server.recv.*` stamps (serve) or taken as
one block (CLI).

Usage:
  python3 tools/phase_attrib.py --log daemon.log --sqlite run.sqlite [--request N] [--top 6]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict

STAMP = re.compile(r"\[phase\] (\S+) wall=([0-9.]+)")


def read_stamps(path: str):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = STAMP.search(line)
            if m:
                out.append((m.group(1), float(m.group(2))))
    return out


def split_requests(stamps):
    """Group stamps into requests: a `server.recv.*` opens one; without
    any server stamp the whole log is one request."""
    if not any(n.startswith("server.recv") for n, _ in stamps):
        return [stamps]
    reqs, cur = [], []
    for n, t in stamps:
        if n.startswith("server.recv") and cur:
            reqs.append(cur)
            cur = []
        cur.append((n, t))
    if cur:
        reqs.append(cur)
    return reqs


def union_busy(intervals):
    """Total length of the union of [s, e) intervals (ns)."""
    if not intervals:
        return 0
    intervals.sort()
    total, cs, ce = 0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + (ce - cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--request", type=int, default=None,
                    help="1-based request index to attribute (default: every request with > 2 stamps)")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    stamps = read_stamps(a.log)
    reqs = split_requests(stamps)
    db = sqlite3.connect(a.sqlite)
    epoch0 = db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    names = dict(db.execute("select id, value from StringIds"))

    kern = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    memc = db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    mems = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET").fetchall()
    api = db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()

    def to_ns(wall):  # epoch seconds -> nsys ns
        return int(wall * 1e9) - epoch0

    print(f"requests stamped: {len(reqs)}   (nsys session epoch {epoch0})")
    for i, r in enumerate(reqs, 1):
        span = r[-1][1] - r[0][1]
        print(f"  request {i}: {r[0][0]} -> {r[-1][0]}  {span:.3f} s  ({len(r)} stamps)")
    print()
    targets = [a.request] if a.request else [i for i, r in enumerate(reqs, 1) if len(r) > 2]
    for i in targets:
        r = reqs[i - 1]
        print(f"=== request {i} phase attribution ===")
        print(f"{'phase window':46s} {'wall s':>8s} {'gpu busy':>9s} {'busy%':>6s} {'kernels':>8s} {'api s':>7s} {'H2D MB':>8s} {'D2H MB':>8s}")
        for (n0, t0), (n1, t1) in zip(r, r[1:]):
            s, e = to_ns(t0), to_ns(t1)
            ks = [(ks_, ke_, nm) for ks_, ke_, nm in kern if ks_ >= s and ks_ < e]
            ms = [(a_, b_, by, ck) for a_, b_, by, ck in memc if a_ >= s and a_ < e]
            mm = [(a_, b_) for a_, b_ in mems if a_ >= s and a_ < e]
            busy = union_busy([(x, y) for x, y, _ in ks] + [(x, y) for x, y, _, _ in ms] + list(mm))
            wall = (e - s)
            apis = [(x, y, nm) for x, y, nm in api if x >= s and x < e]
            api_by = defaultdict(lambda: [0, 0])
            for x, y, nm in apis:
                api_by[names.get(nm, str(nm))][0] += 1
                api_by[names.get(nm, str(nm))][1] += (y - x)
            api_s = sum(v[1] for v in api_by.values()) / 1e9
            h2d = sum(by for _, _, by, ck in ms if ck == 1) / 2**20
            d2h = sum(by for _, _, by, ck in ms if ck == 2) / 2**20
            print(f"{n0 + ' -> ' + n1:46s} {wall/1e9:8.3f} {busy/1e9:9.3f} {100*busy/max(wall,1):6.1f} {len(ks):8d} {api_s:7.2f} {h2d:8.1f} {d2h:8.1f}")
            if wall / 1e9 >= 0.05:
                top_api = sorted(api_by.items(), key=lambda kv: -kv[1][1])[:a.top]
                print("     api: " + ", ".join(f"{k} x{v[0]} {v[1]/1e9:.2f}s" for k, v in top_api))
                kb = defaultdict(lambda: [0, 0])
                for x, y, nm in ks:
                    kn = names.get(nm, str(nm))
                    kb[kn][0] += 1
                    kb[kn][1] += (y - x)
                top_k = sorted(kb.items(), key=lambda kv: -kv[1][1])[:a.top]
                print("     kernels: " + ", ".join(f"{k} x{v[0]} {v[1]/1e9:.2f}s" for k, v in top_k))
        print(f"TOTAL {r[0][0]} -> {r[-1][0]} {r[-1][1] - r[0][1]:46.3f}")
        print()


if __name__ == "__main__":
    main()
