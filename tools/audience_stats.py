#!/usr/bin/env python3
"""Audience measurement: PyPI installs, GitHub traffic, hub downloads.

The project has been public since February and nobody is measuring whether
anyone arrives. No marketing action can be judged without a baseline, so
this takes one — dated today — and one snapshot per week after it.

    python3 tools/audience_stats.py snapshot     # append one dated record
    python3 tools/audience_stats.py table        # render the weekly table
    python3 tools/audience_stats.py snapshot --table

Snapshots land in ``docs/internal/audience/snapshots.jsonl`` (one JSON
object per line, appended, never rewritten) and the rendered table beside
it. Both are internal business data: that directory is outside the public
repository and reaches the private corpus mirror through
``tools/sync_internal_corpus.sh``.

Two rules the collector keeps:

* **A number is measured or it is null.** A source that fails records
  ``null`` with the reason. Nothing is carried forward, interpolated or
  estimated — a fabricated growth curve is worse than no curve.
* **Writes are atomic** (temp file, then rename). The rack is on one
  breaker with no UPS; a snapshot interrupted mid-write must not corrupt
  the series it is appending to.

GitHub traffic needs a token with push access to the repo (``GITHUB_PAT``
in ``.env``); it only ever exposes the trailing 14 days, which is exactly
why a weekly snapshot is the thing that preserves it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "NeuroBrix/neurobrix"
PYPI_PACKAGE = "neurobrix"
HUB = "https://neurobrix.es"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "internal" / "audience"
SNAPSHOTS = OUT_DIR / "snapshots.jsonl"
TABLE = OUT_DIR / "AUDIENCE.md"
UA = "neurobrix-audience-stats"


def _get(url: str, token: str | None = None, timeout: int = 30, retries: int = 3):
    """GET with backoff on 429. pypistats throttles the anonymous API hard,
    and a throttled read must not be recorded as a missing metric when
    waiting a few seconds would have got the number."""
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def _source(fn, *args, **kwargs):
    """Run a collector; on failure return the reason instead of a number."""
    try:
        return fn(*args, **kwargs), None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, KeyError) as exc:
        return None, f"{type(exc).__name__}: {exc}"[:200]


# --- collectors -------------------------------------------------------------

def github_repo(token: str | None) -> dict:
    d = _get(f"https://api.github.com/repos/{REPO}", token)
    return {
        "stars": d["stargazers_count"],
        "forks": d["forks_count"],
        "watchers": d["subscribers_count"],
        "open_issues": d["open_issues_count"],
    }


def github_traffic(token: str | None) -> dict:
    """Views and clones over GitHub's trailing 14-day window."""
    views = _get(f"https://api.github.com/repos/{REPO}/traffic/views", token)
    clones = _get(f"https://api.github.com/repos/{REPO}/traffic/clones", token)
    return {
        "views_14d": views["count"], "unique_visitors_14d": views["uniques"],
        "clones_14d": clones["count"], "unique_cloners_14d": clones["uniques"],
    }


def pypi_downloads() -> dict:
    d = _get(f"https://pypistats.org/api/packages/{PYPI_PACKAGE}/recent")["data"]
    return {"pypi_day": d["last_day"], "pypi_week": d["last_week"],
            "pypi_month": d["last_month"]}


def hub_downloads() -> dict:
    models = _get(f"{HUB}/api/models?limit=500")["models"]
    per_model = {m["slug"]: int(m.get("downloadCount") or 0) for m in models}
    public = sum(1 for m in models if m.get("visibility") == "PUBLIC")
    return {
        "hub_models": len(models),
        "hub_public": public,
        "hub_downloads_total": sum(per_model.values()),
        "hub_per_model": per_model,
    }


# --- snapshot ---------------------------------------------------------------

def snapshot() -> dict:
    token = os.environ.get("GITHUB_PAT")
    record: dict = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "errors": {}}
    for name, fn, args in (
        ("github", github_repo, (token,)),
        ("traffic", github_traffic, (token,)),
        ("pypi", pypi_downloads, ()),
        ("hub", hub_downloads, ()),
    ):
        value, error = _source(fn, *args)
        if value is None:
            record["errors"][name] = error or "no data"
        else:
            record.update(value)
    return record


def append(record: dict) -> None:
    """Append one record; temp-file-then-rename so a power cut cannot
    truncate the series."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = SNAPSHOTS.read_text() if SNAPSHOTS.exists() else ""
    tmp = SNAPSHOTS.with_suffix(".jsonl.tmp")
    tmp.write_text(existing + json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(SNAPSHOTS)


# --- table ------------------------------------------------------------------

FIELDS = [
    ("stars", "GitHub stars"),
    ("forks", "forks"),
    ("watchers", "watchers"),
    ("open_issues", "open issues"),
    ("unique_visitors_14d", "unique visitors (14 d)"),
    ("views_14d", "page views (14 d)"),
    ("unique_cloners_14d", "unique cloners (14 d) [CI/bots — not audience]"),
    ("pypi_week", "PyPI installs (7 d)"),
    ("pypi_month", "PyPI installs (30 d)"),
    ("hub_downloads_total", "hub downloads (total)"),
    ("hub_public", "models public"),
]


def _iso_week(ts: str) -> str:
    d = datetime.fromisoformat(ts)
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"


def render() -> str:
    if not SNAPSHOTS.exists():
        return "No snapshot yet. Run: python3 tools/audience_stats.py snapshot\n"
    records = [json.loads(l) for l in SNAPSHOTS.read_text().splitlines() if l.strip()]
    # one row per ISO week — the last snapshot of each week wins
    by_week: dict[str, dict] = {}
    for r in records:
        by_week[_iso_week(r["ts"])] = r
    weeks = sorted(by_week)

    out = [
        "# Audience — one line per week",
        "",
        "Generated by `tools/audience_stats.py`. Every cell is measured at the",
        "date in its row; a source that failed is `—`, never a carried-forward",
        "or interpolated value. GitHub only exposes a trailing 14-day traffic",
        "window, so these snapshots are the only record of it that survives.",
        "",
        "**Read the clone count with care.** GitHub counts every `git clone`,",
        "including CI runners, package mirrors and scrapers. On the baseline",
        "snapshot it read 715 clones / 180 unique against **6 unique human",
        "visitors** — the clone line is infrastructure, not demand, and must",
        "never be quoted as audience. The honest attention metrics here are",
        "unique visitors, PyPI installs and hub downloads.",
        "",
        f"**Baseline: {weeks[0]}** ({by_week[weeks[0]]['ts']}).",
        "",
        "| metric | " + " | ".join(weeks) + " |",
        "|---|" + "---|" * len(weeks),
    ]
    for key, label in FIELDS:
        cells = []
        for w in weeks:
            v = by_week[w].get(key)
            cells.append("—" if v is None else str(v))
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    last = by_week[weeks[-1]]
    out += ["", "## Most-downloaded models (latest snapshot)", ""]
    per_model = last.get("hub_per_model") or {}
    top = sorted(per_model.items(), key=lambda kv: kv[1], reverse=True)[:10]
    if top:
        out += ["| model | downloads |", "|---|---|"]
        out += [f"| {slug} | {n} |" for slug, n in top if n > 0]
    if last.get("errors"):
        out += ["", "## Sources that failed in the latest snapshot", ""]
        out += [f"- `{k}`: {v}" for k, v in last["errors"].items()]
    return "\n".join(out) + "\n"


def write_table() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TABLE.with_suffix(".md.tmp")
    tmp.write_text(render())
    tmp.replace(TABLE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "table"])
    ap.add_argument("--table", action="store_true",
                    help="with `snapshot`: also re-render the table")
    args = ap.parse_args()

    if args.command == "snapshot":
        record = snapshot()
        append(record)
        measured = [k for k, _ in FIELDS if record.get(k) is not None]
        print(f"snapshot {record['ts']}: {len(measured)}/{len(FIELDS)} metrics measured")
        for key, label in FIELDS:
            v = record.get(key)
            print(f"  {label:26s} {'—' if v is None else v}")
        for name, err in record.get("errors", {}).items():
            print(f"  ! {name} failed: {err}", file=sys.stderr)
        if args.table:
            write_table()
            print(f"table -> {TABLE}")
        return 0

    write_table()
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
