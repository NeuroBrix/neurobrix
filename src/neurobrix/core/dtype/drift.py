"""Drift-site detector — the op where the Triton engine departs from the ATen
oracle on one request.

Both engines already write the same per-op record under ``NBX_DUMP_TIDS``
(component, tid, op_uid, op_type, dtype, shape, head10, last_pos10, l2_norm)
— ``TritonSequence.nbx_tid_stats`` and the ATen loop's twin — so a drift
walk is a matter of running the request twice and reading the two files in
the oracle's op order. The detector names the FIRST op whose window deviates
beyond a relative bound, in the oracle's order, with the deviation of every
op behind it: the site to open, not a verdict on the output.

The bound is relative to the window's own scale (max |value| of both sides),
so a large activation is judged by its own magnitude; the first op over it is
the site, and the ops after it are cascade until proven otherwise (the
three-class discipline of the numerical chantiers: root, cascade, common
baseline). Matching is by (component, tid): op uids restart per component.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FIELDS = ("head10", "last_pos10")


def load_dump(path) -> Tuple[Dict[Tuple[str, str], dict], List[Tuple[str, str]]]:
    """Records keyed by (component, tid), in file order, first write kept."""
    recs: Dict[Tuple[str, str], dict] = {}
    order: List[Tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            r = d.get("record", d)
            tid = r.get("tid")
            if tid is None:
                continue
            key = (str(r.get("component", "?")), str(tid))
            if key in recs:
                continue
            recs[key] = r
            order.append(key)
    return recs, order


def deviation(a, b) -> Optional[float]:
    """max |a-b| over the window, relative to the window's own scale."""
    if not a or not b or len(a) != len(b):
        return None
    try:
        scale = max(max(abs(float(x)) for x in a), max(abs(float(y)) for y in b), 1e-12)
        return max(abs(float(x) - float(y)) for x, y in zip(a, b)) / scale
    except (TypeError, ValueError):
        return None


@dataclass
class DriftSite:
    component: str
    tid: str
    op_uid: Optional[str]
    op_type: Optional[str]
    shape: Any
    dtype_a: Optional[str]
    dtype_b: Optional[str]
    field: str
    rel_dev: float
    index: int                       # position in the oracle's op order
    window_a: list = field(default_factory=list)
    window_b: list = field(default_factory=list)


@dataclass
class DriftReport:
    ops_a: int
    matched: int
    missing_in_b: int
    bound: float
    first: Optional[DriftSite]
    top: List[DriftSite]
    over_bound: int
    #: the first over-bound op whose output dtype is the SAME on both sides —
    #: a kernel drift, not the two engines' precision policies disagreeing
    #: (the Triton engine keeps fp32 where the ATen oracle rounds to fp16:
    #: that deviation is the oracle's rounding and is reported apart).
    first_same_dtype: Optional[DriftSite] = None
    policy_sites: int = 0

    def to_dict(self) -> Dict[str, Any]:
        def site(s: Optional[DriftSite]):
            return None if s is None else {**s.__dict__}
        return {"ops_a": self.ops_a, "matched": self.matched, "missing_in_b": self.missing_in_b,
                "bound": self.bound, "over_bound": self.over_bound, "policy_sites": self.policy_sites,
                "first": site(self.first), "first_same_dtype": site(self.first_same_dtype),
                "top": [site(s) for s in self.top]}


def detect(dump_a, dump_b, *, bound: float = 0.02, top: int = 12, fields=FIELDS) -> DriftReport:
    """The drift walk: A is the oracle's dump (its op order rules), B the
    engine under test. An op deviates when any of `fields` deviates."""
    A, order = load_dump(dump_a)
    B, _ = load_dump(dump_b)
    rows: List[DriftSite] = []
    missing = 0
    for i, key in enumerate(order):
        ra, rb = A[key], B.get(key)
        if rb is None:
            missing += 1
            continue
        best = None
        for f in fields:
            d = deviation(ra.get(f), rb.get(f))
            if d is None:
                continue
            if best is None or d > best[0]:
                best = (d, f)
        if best is None:
            continue
        d, f = best
        rows.append(DriftSite(component=key[0], tid=key[1], op_uid=ra.get("op_uid"), op_type=ra.get("op_type"),
                              shape=ra.get("shape"), dtype_a=ra.get("dtype"), dtype_b=rb.get("dtype"),
                              field=f, rel_dev=float(d), index=i,
                              window_a=list(ra.get(f) or []), window_b=list(rb.get(f) or [])))
    first = next((r for r in rows if r.rel_dev > bound), None)
    over = sum(1 for r in rows if r.rel_dev > bound)
    same = [r for r in rows if r.rel_dev > bound and _same_dtype(r.dtype_a, r.dtype_b)]
    return DriftReport(ops_a=len(order), matched=len(rows), missing_in_b=missing, bound=bound,
                       first=first, top=sorted(rows, key=lambda r: -r.rel_dev)[:top], over_bound=over,
                       first_same_dtype=(same[0] if same else None), policy_sites=over - len(same))


def _same_dtype(a: Optional[str], b: Optional[str]) -> bool:
    """`torch.float16` and `fp16` name the same dtype across the two dumps."""
    def norm(x):
        x = str(x or "").replace("torch.", "").lower()
        return {"float16": "fp16", "half": "fp16", "bfloat16": "bf16", "float32": "fp32", "float": "fp32",
                "float64": "fp64", "double": "fp64"}.get(x, x)
    return norm(a) == norm(b)


def describe(report: DriftReport) -> str:
    lines = [f"ops in the oracle {report.ops_a}, matched {report.matched} (missing in the engine {report.missing_in_b}), "
             f"bound {report.bound:g}: {report.over_bound} op(s) over it"]
    s = report.first
    if s is None:
        lines.append("no drift site: every matched op is within the bound")
    else:
        lines.append(f"FIRST over the bound {s.component}/{s.op_uid or s.tid} ({s.op_type}) shape={s.shape} "
                     f"dtypes {s.dtype_a}->{s.dtype_b} {s.field} rel_dev={s.rel_dev:.4f} at op #{s.index}"
                     + ("" if _same_dtype(s.dtype_a, s.dtype_b) else "  [precision policy: the dtypes differ]"))
        lines.append(f"   oracle {s.field}: {[round(v, 5) for v in s.window_a]}")
        lines.append(f"   engine {s.field}: {[round(v, 5) for v in s.window_b]}")
        k = report.first_same_dtype
        if k is None:
            lines.append(f"no KERNEL drift site: every over-bound op ({report.policy_sites}) carries different "
                         f"dtypes on the two sides — the engines' precision policies, not a kernel")
        else:
            lines.append(f"KERNEL DRIFT SITE {k.component}/{k.op_uid or k.tid} ({k.op_type}) shape={k.shape} "
                         f"dtype {k.dtype_b} {k.field} rel_dev={k.rel_dev:.4f} at op #{k.index} "
                         f"({report.policy_sites} policy site(s) before or around it)")
            lines.append(f"   oracle {k.field}: {[round(v, 5) for v in k.window_a]}")
            lines.append(f"   engine {k.field}: {[round(v, 5) for v in k.window_b]}")
    lines.append("largest deviations:")
    for r in report.top:
        lines.append(f"   {r.rel_dev:8.4f}  {r.component}/{r.op_uid or r.tid}  ({r.op_type}) {r.shape}")
    return "\n".join(lines)
