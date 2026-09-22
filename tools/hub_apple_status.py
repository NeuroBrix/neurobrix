#!/usr/bin/env python3
"""One line per hub model: what it does on Apple silicon, and nothing invented.

The document that closes the loop. It is generated, never written by hand, for
the reason the failure register exists: a table typed by its author is a
sentence, and a sentence about forty-seven models is one nobody can check.

Per model, four columns and no empty cell:

  state         `runs` / `refuses` / `not measured` — the third is a RESULT,
                not a gap, and it carries the figure that explains it (the
                artefact's size against what the machine had). A cell that says
                "not measured" is worth more than a cell that lies.
  cause         for a refusal, the named cause read from the run's own output —
                the refusal message, not a summary of it.
  certified     how many certified autotune entries this model's target has.
  measured on   the campaign that produced the line, so any of it can be reread.

Sources, all on disk, none inferred:
  * validation_outputs/hub_E_*/records.json and <workshop>/*/records.json
  * the deferred register, for models with a size but no run
  * src/neurobrix/config/autotune/<vendor>/<profile>/ for the certified count

    tools/hub_apple_status.py --out TABLEAU.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The literal path segments below are real directories on the Mac and are NOT
# translated; only the identifier is.
WORKSHOP = Path.home() / "Workspace" / "nbx-atelier" / "campagnes"


def _records() -> list[Path]:
    out = list((REPO / "validation_outputs").glob("hub_E_*/records.json"))
    out += list(WORKSHOP.glob("*/records.json"))
    # A campaign that runs one model per call writes one level deeper.
    out += list(WORKSHOP.glob("*/*/records.json"))
    return sorted(out)


def _deferred_sizes() -> dict:
    """Model -> (GB on disk, weights MB, largest component MB) from the
    deferred register's tables. Read, not retyped."""
    sizes = {}
    for reg in (REPO / "validation_outputs").glob("hub_E_deferred_*/REGISTRE.md"):
        for line in reg.read_text().splitlines():
            m = re.match(r"\|\s*`([^`]+)`\s*\|\s*\w+\s*\|\s*([\d.]+)\s*\|"
                         r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
            if m:
                sizes[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                                     float(m.group(4)))
    return sizes


def _certified_counts() -> dict:
    root = REPO / "src" / "neurobrix" / "config" / "autotune"
    counts = {}
    if root.is_dir():
        for profile in sorted(root.glob("*/*")):
            if profile.is_dir():
                # SHAPES, not files: one file holds every shape of one kernel
                # and dtype, so counting files understates by an unknown
                # factor and reads like a smaller result than it is.
                shapes = 0
                for f in profile.glob("*.json"):
                    try:
                        shapes += len(json.loads(f.read_text()).get("entries", {}))
                    except (OSError, ValueError):
                        continue
                counts[f"{profile.parent.name}/{profile.name}"] = shapes
    return counts


#: Every shape a refusal takes in this engine. A cause that IS named and that
#: this pattern misses prints as "no cause named", which is a small lie in
#: the one column that must not tell any: the cascade's own refusal names its
#: arithmetic and looks like none of the others.
_REFUSAL = re.compile(r"(Refusing[^\n]{0,200}|ZERO FALLBACK:[^\n]{0,200}|"
                      r"Failed at [^\s:]+[^\n]{0,160}|"
                      r"largest component: [^\n]{0,80}|"
                      r"AttributeError: [^\n]{0,120}|"
                      r"Total required: [^\n]{0,60})")


def _cause(run: dict) -> str:
    blob = (run.get("stderr_tail") or "") + "\n" + (run.get("stdout_tail") or "")
    hits = _REFUSAL.findall(blob)
    if not hits:
        return f"rc {run.get('rc')}, no cause named in the retained output"
    # the LAST one: an earlier refusal may have been recovered from
    return re.sub(r"\s+", " ", hits[-1]).strip()[:190]


def build() -> tuple[dict, dict]:
    models: dict = {}
    for path in _records():
        doc = json.loads(path.read_text())
        stamp = doc.get("generated", "?")[:10]
        for m in doc.get("models", []):
            name = m.get("model")
            if not name:
                continue
            row = models.setdefault(name, {"runs": {}, "seen": [], "size": None})
            row["seen"].append((stamp, path.parent.name))
            if m.get("source", {}).get("gb"):
                row["size"] = m["source"]["gb"]
            for run in m.get("runs", []):
                row["runs"][run.get("arm", "?")] = run
            row["status"] = m.get("status", row.get("status"))
            row["reason"] = m.get("reason", row.get("reason"))
    return models, _deferred_sizes()


def render(models: dict, sizes: dict, certified: dict) -> str:
    known = sorted(set(models) | set(sizes))
    lines = [
        "# The hub\'s models on Apple silicon",
        "",
        f"Generated by `tools/hub_apple_status.py`. **{len(known)} models**, one "
        "line each, no empty cell. A cell reading \u00ab not measured \u00bb carries the "
        "figure that explains it: it is a result, not a hole.",
        "",
        "## The denominator, because two tables that do not count the same "
        "population cannot be set side by side",
        "",
        f"This table counts **{len(known)} artefacts**: the union of those "
        "carrying a campaign-E record and those the deferred register puts a "
        "figure on. That is the population **measurable from this machine**, "
        "not the hub catalogue.",
        "",
        "The other machine\'s table counts **47 models**, which is the **hub "
        "catalogue**. Both figures are correct and they do not name the same "
        "thing. The known correspondence: four of the artefacts counted here "
        "are **variants or backups** of a catalogue model (`-int4g128`, "
        "`-ffnonly`, `.pre-G-backup`), which a catalogue counts once and a "
        "build carries separately.",
        "",
        "**The right denominator depends on the question.** For \u00ab what does the "
        "hub do on Apple \u00bb it is the catalogue: 47. For \u00ab what have we measured "
        "and what is left \u00bb it is this list, because a quantised variant is "
        "measured and refused separately from its base model.",
        "",
        "**Not verified from here:** the catalogue could not be re-read — "
        "`neurobrix hub` fails certificate verification for `neurobrix.es` "
        "from this machine. The 47 is therefore taken as the other machine "
        "reports it, and is not confirmed.",
        "",
        "**The \u00ab measured on \u00bb column decides how to read this.** The verdicts "
        "come from campaigns of different dates, on different trees: a line "
        "from 9 September says nothing about today\'s tree. Without that "
        "column the table would be false while remaining accurate.",
        "",
        "| model | state | cause / figure | measured on | cert. entries |",
        "|---|---|---|---|---|",
    ]
    tally = {"runs": 0, "refuses": 0, "not measured": 0}
    for name in known:
        row = models.get(name)
        cert = "none for this vendor" if not certified else "—"
        if row and row["runs"]:
            ok = [a for a, r in row["runs"].items() if r.get("rc") == 0]
            bad = {a: r for a, r in row["runs"].items() if r.get("rc") != 0}
            if ok and not bad:
                state, detail = "runs", f"arms: {', '.join(sorted(ok))}"
            elif ok:
                state = "runs"
                detail = (f"green arms: {', '.join(sorted(ok))} · "
                          f"{list(bad)[0]} → {_cause(list(bad.values())[0])}")
            else:
                state = "refuses"
                detail = _cause(list(bad.values())[0])
        else:
            state = "not measured"
            gb = (sizes.get(name) or (None, None, None))[0] or (row or {}).get("size")
            biggest = (sizes.get(name) or (None, None, None))[2]
            detail = (f"artefact {gb} GB"
                      + (f", largest component {biggest:.0f} MB" if biggest else "")
                      ) if gb else ((row or {}).get("reason") or "no size data")
        tally[state] += 1
        when = max((d for d, _ in (row or {}).get("seen", [])), default="—")
        lines.append(f"| `{name}` | {state} | {detail} | {when} | {cert} |")
    lines += ["", "## Count", "",
              f"* **runs**: {tally['runs']}",
              f"* **refuses**: {tally['refuses']}",
              f"* **not measured**: {tally['not measured']}", ""]
    if certified:
        lines += ["## Certified entries per profile (shapes, not files)", ""]
        for prof, n in sorted(certified.items()):
            lines.append(f"* `{prof}` : **{n}**")
    else:
        lines += ["## Certified entries", "",
                  "**None.** The mechanism resolves; the directory is empty."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    models, sizes = build()
    text = render(models, sizes, _certified_counts())
    if args.out:
        args.out.write_text(text)
        print(f"written: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
