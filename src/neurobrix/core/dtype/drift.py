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


def abs_deviation(a, b) -> Optional[float]:
    """max |a-b| over the window, in the window's own units."""
    if not a or not b or len(a) != len(b):
        return None
    try:
        return max(abs(float(x) - float(y)) for x, y in zip(a, b))
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
    abs_dev: float = 0.0             # max |a-b| over the same window, absolute
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
    #: the first over-bound op whose output dtype is the SAME on both sides
    #: and that computes (a cast, view, slice or copy only carries its input's
    #: deviation) — a kernel drift candidate, not the two engines' precision
    #: policies disagreeing (the Triton engine keeps fp32 where the ATen
    #: oracle rounds to fp16: that deviation is the oracle's rounding and is
    #: reported apart).
    first_same_dtype: Optional[DriftSite] = None
    policy_sites: int = 0
    #: How the FIRST over-bound op came to deviate: "kernel" (same dtype, arithmetic — read
    #: that kernel), "policy" (the dtypes differ — the two engines' precision policies),
    #: "discrete" (an integer tensor — indices, codes, tokens: a decision that flipped on a
    #: float deviation below the bound, named by `float_before`), "carrier" (a view, cast,
    #: slice or copy whose deviation is its input's), or None when nothing is over the bound.
    origin_class: Optional[str] = None
    #: The largest float deviation among the matched ops BEFORE the first over-bound op —
    #: what a discrete decision or a carrier actually amplified.
    float_before: Optional[DriftSite] = None
    #: The largest ABSOLUTE deviation among the float ops before the origin. A "kernel" origin
    #: whose absolute deviation is not larger than this (within 25 %) added no error of its
    #: own: the relative bound was crossed because the values SHRANK (a relu, a gate, a
    #: normalisation) — class "scale", the error is inherited from upstream.
    abs_before: Optional[float] = None
    #: The oracle's op just before the origin has no record on the engine side — the engine
    #: fused it or skipped it, so the origin shows its PRODUCER's deviation (read the fusion).
    producer_missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        def site(s: Optional[DriftSite]):
            return None if s is None else {**s.__dict__}
        return {"ops_a": self.ops_a, "matched": self.matched, "missing_in_b": self.missing_in_b,
                "bound": self.bound, "over_bound": self.over_bound, "policy_sites": self.policy_sites,
                "first": site(self.first), "first_same_dtype": site(self.first_same_dtype),
                "origin_class": self.origin_class, "float_before": site(self.float_before),
                "producer_missing": self.producer_missing, "abs_before": self.abs_before,
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
                              abs_dev=float(abs_deviation(ra.get(f), rb.get(f)) or 0.0),
                              window_a=list(ra.get(f) or []), window_b=list(rb.get(f) or [])))
    first = next((r for r in rows if r.rel_dev > bound), None)
    over = sum(1 for r in rows if r.rel_dev > bound)
    same = [r for r in rows if r.rel_dev > bound and _same_dtype(r.dtype_a, r.dtype_b)
            and not _carries_only(r.op_type)]
    origin_class = None
    float_before = None
    producer_missing = False
    abs_before = None
    if first is not None:
        if _is_integer(first.dtype_a) or _is_integer(first.dtype_b):
            origin_class = "discrete"
        elif not _same_dtype(first.dtype_a, first.dtype_b):
            origin_class = "policy"
        elif _carries_only(first.op_type):
            origin_class = "carrier"
        else:
            origin_class = "kernel"
        before = [r for r in rows if r.index < first.index and not _is_integer(r.dtype_a) and not _is_integer(r.dtype_b)]
        float_before = max(before, key=lambda r: r.rel_dev) if before else None
        abs_before = max((r.abs_dev for r in before if r.field == first.field), default=None)
        if origin_class == "kernel" and abs_before is not None and first.abs_dev <= 1.25 * abs_before:
            origin_class = "scale"
        producer_missing = first.index > 0 and order[first.index - 1] not in B
    return DriftReport(ops_a=len(order), matched=len(rows), missing_in_b=missing, bound=bound,
                       first=first, top=sorted(rows, key=lambda r: -r.rel_dev)[:top], over_bound=over,
                       first_same_dtype=(same[0] if same else None), policy_sites=over - len(same),
                       origin_class=origin_class, float_before=float_before, abs_before=abs_before,
                       producer_missing=bool(first is not None and producer_missing))


def _is_integer(dtype: Optional[str]) -> bool:
    """`torch.int64`, `int64`, `int32`, `bool`, `long`: a tensor of indices, codes or tokens."""
    name = str(dtype or "").replace("torch.", "").lower()
    return name.startswith(("int", "uint", "bool", "long", "short", "byte", "char"))


#: Ops that move or re-view a tensor without arithmetic: a cast, a view, a
#: slice, a copy. Whatever deviation they show is their INPUT's, so they can
#: never be the kernel site — they are the cascade behind it.
_CARRIERS = ("_to_copy", "to", "copy", "clone", "contiguous", "view", "_unsafe_view", "reshape", "transpose",
             "permute", "slice", "select", "unsqueeze", "squeeze", "expand", "narrow", "alias", "detach",
             "flatten", "unflatten", "t", "split", "chunk", "unbind", "index_select", "gather", "cat", "stack")


def _carries_only(op_type: Optional[str]) -> bool:
    name = str(op_type or "").split("::")[-1].split(".")[-1]
    return name in _CARRIERS


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
        if report.origin_class == "discrete":
            fb = report.float_before
            lines.append("ORIGIN: a DISCRETE decision (an integer tensor: indices, codes or tokens) — the flip came from a float "
                         "deviation below the bound"
                         + (f", largest before it {fb.component}/{fb.op_uid or fb.tid} ({fb.op_type}) rel_dev={fb.rel_dev:.4f} at op #{fb.index}"
                            if fb else ""))
        elif report.origin_class == "carrier":
            fb = report.float_before
            lines.append("ORIGIN: a carrier op (view/cast/slice/copy) — its deviation is its input's"
                         + (f"; largest float deviation before it {fb.component}/{fb.op_uid or fb.tid} ({fb.op_type}) rel_dev={fb.rel_dev:.4f}" if fb else ""))
        if report.origin_class == "scale":
            lines.append(f"ORIGIN: the values SHRANK here (abs deviation {s.abs_dev:.4g} vs {report.abs_before:.4g} before it) — "
                         "no new error at this op, the relative bound was crossed by an inherited one; read the largest deviation before it")
        if report.producer_missing:
            lines.append("   the oracle's op just before the origin has NO record on the engine side: the engine fused or "
                         "skipped it, so the origin shows its producer's deviation — read the fusion")
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
