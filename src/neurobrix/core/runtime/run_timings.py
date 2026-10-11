"""The four timings every `neurobrix run` states, on one parseable line — both engines.

    [Timings] load_s=<s> ttft_s=<s> tpot_ms=<ms|na> tok_s=<r|na> tokens=<n|na> first=<token|output>

Nothing is timed here: the line is READ from the hooks the engines already
carry — the run ledger of `phase_trace` (the `run.container.open`,
`run.exec.begin` and `run.exec.end` marks the CLI stamps, the
`flow.first_token` mark stamped at the per-token emit site, the load spans
around executor setup and every component weight load) and the token tally
of `decode_progress`. stdlib only: R33 and R34 safe, the same module for the
ATen and the Triton branch.

Definitions (monotonic clock; L = the union of the load spans):

* load_s  — container open to the request's start, plus every load span
  inside the request: the time the run spent becoming ready.
* ttft_s  — request start to the first token, L removed (`first=token`); a run
  that emitted no token is timed to its output, the request's end
  (`first=output`).
* tpot_ms — decode only: first to last token over (tokens - 1), L removed;
  `na` below two tokens.
* tok_s   — tokens over the generation wall (request start to the last
  token, L removed); `na` without a token.
* tokens  — tokens the request emitted; `na` without a token.

A value that is not defined for the run is written `na`, never a number.
`parse_line` is the one reader (the gate harness uses it): the producer and
the parser are one function pair, held together by tests.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from neurobrix import decode_progress
from neurobrix.core.runtime import phase_trace

TAG = "[Timings]"
NA = "na"
#: The line's fields, in order; every one present on every line.
FIELDS = ("load_s", "ttft_s", "tpot_ms", "tok_s", "tokens", "first")

OPEN = "run.container.open"
BEGIN = "run.exec.begin"
END = "run.exec.end"
FIRST_TOKEN = "flow.first_token"


def reset() -> None:
    """A fresh ledger and tally: a long-lived process (the daemon) opens one per request,
    so neither grows across requests nor carries a previous request's first token."""
    phase_trace.reset_ledger()
    decode_progress.reset_tally()


def open_container() -> None:
    """Stamp the container open: the start of the run's load time (a fresh ledger)."""
    reset()
    phase_trace.mark(OPEN)


def begin_request() -> None:
    """Stamp the request's start, the engine's device already drained."""
    decode_progress.reset_tally()
    phase_trace.mark(BEGIN)


def end_request() -> None:
    """Stamp the request's end, the engine's device already drained."""
    phase_trace.mark(END)


def _union(spans: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _inside(spans: List[Tuple[float, float]], a: float, b: float) -> float:
    """Seconds of `spans` (already a union) that fall inside [a, b]."""
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans)


def compute(t_open: float, t_begin: float, t_end: float, load_spans, t_first: Optional[float],
            t_last: Optional[float], n_tokens: int) -> dict:
    """The four timings from the ledger's instants. Pure: the tests feed it."""
    spans = _union(load_spans)
    out = {"load_s": (t_begin - t_open) + _inside(spans, t_begin, t_end)}
    if n_tokens > 0:
        if t_first is None or t_last is None:
            raise RuntimeError(
                f"ZERO FALLBACK: the run emitted {n_tokens} token(s) but its ledger holds no "
                f"'{FIRST_TOKEN}' mark or no last-token instant; the timings would be invented.")
        out["ttft_s"] = (t_first - t_begin) - _inside(spans, t_begin, t_first)
        gen = (t_last - t_begin) - _inside(spans, t_begin, t_last)
        out["tok_s"] = n_tokens / gen if gen > 0 else None
        if n_tokens >= 2:
            dec = (t_last - t_first) - _inside(spans, t_first, t_last)
            out["tpot_ms"] = 1000.0 * dec / (n_tokens - 1)
        else:
            out["tpot_ms"] = None
        out["tokens"] = n_tokens
        out["first"] = "token"
    else:
        out["ttft_s"] = (t_end - t_begin) - _inside(spans, t_begin, t_end)
        out.update(tpot_ms=None, tok_s=None, tokens=None, first="output")
    return out


def measured() -> dict:
    """The four timings of the request just ended, read from the hooks' ledgers."""
    t_open, t_begin, t_end = (phase_trace.first(n) for n in (OPEN, BEGIN, END))
    missing = [n for n, t in ((OPEN, t_open), (BEGIN, t_begin), (END, t_end)) if t is None]
    if missing:
        raise RuntimeError(f"ZERO FALLBACK: the run ledger lacks the mark(s) {missing}; "
                           f"open_container / begin_request / end_request were not all called.")
    tally = decode_progress.tally()
    return compute(t_open, t_begin, t_end, phase_trace.load_spans(), phase_trace.first(FIRST_TOKEN),
                   tally["last"], tally["n"])


#: Printed resolution of the numeric fields (ms resolution on the second-valued ones).
_DECIMALS = {"load_s": 3, "ttft_s": 3, "tpot_ms": 2, "tok_s": 2}


def _fmt(key: str, value) -> str:
    if value is None:
        return NA
    if key in _DECIMALS:
        return f"{value:.{_DECIMALS[key]}f}"
    return str(value)


def format_line(timings: dict) -> str:
    """The one line, every field present, `na` where the run does not define it."""
    return TAG + " " + " ".join(f"{k}={_fmt(k, timings[k])}" for k in FIELDS)


_LINE = re.compile(re.escape(TAG) + r" (.*)$", re.M)


def parse_line(text: str) -> Optional[dict]:
    """The LAST timings line of a log as {field: float | int | str | None} (`na` -> None);
    None when the log holds no such line. A line missing a field is refused by name."""
    found = _LINE.findall(text)
    if not found:
        return None
    pairs = dict(kv.split("=", 1) for kv in found[-1].split())
    absent = [k for k in FIELDS if k not in pairs]
    if absent:
        raise ValueError(f"{TAG} line lacks {absent}: {found[-1]!r}")
    row: dict = {}
    for k in FIELDS:
        v = pairs[k]
        if v == NA:
            row[k] = None
        elif k == "first":
            row[k] = v
        elif k == "tokens":
            row[k] = int(v)
        else:
            row[k] = float(v)
    return row
