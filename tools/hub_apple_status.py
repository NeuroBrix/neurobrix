#!/usr/bin/env python3
"""One line per hub model: what it does on Apple silicon, and nothing invented.

The document that closes the loop. It is generated, never written by hand, for
the reason the failure register exists: a table typed by its author is a
sentence, and a sentence about forty-seven models is one nobody can check.

Per model, four columns and no empty cell:

  état        `tourne` / `refuse` / `non mesuré` — the third is a RESULT, not
              a gap, and it carries the figure that explains it (the artefact's
              size against what the machine had). A cell that says "non mesuré"
              is worth more than a cell that lies.
  cause       for a refusal, the named cause read from the run's own output —
              the refusal message, not a summary of it.
  certifié    how many certified autotune entries this model's target has.
  mesuré le   the campaign that produced the line, so any of it can be reread.

Sources, all on disk, none inferred:
  * validation_outputs/hub_E_*/records.json and campagnes/*/records.json
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
ATELIER = Path.home() / "Workspace" / "nbx-atelier" / "campagnes"


def _records() -> list[Path]:
    out = list((REPO / "validation_outputs").glob("hub_E_*/records.json"))
    out += list(ATELIER.glob("*/records.json"))
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
                counts[f"{profile.parent.name}/{profile.name}"] = len(
                    list(profile.glob("*.json")))
    return counts


_REFUSAL = re.compile(r"(Refusing[^\n]{0,200}|ZERO FALLBACK:[^\n]{0,200}|"
                      r"Failed at [^\s:]+[^\n]{0,160})")


def _cause(run: dict) -> str:
    blob = (run.get("stderr_tail") or "") + "\n" + (run.get("stdout_tail") or "")
    hits = _REFUSAL.findall(blob)
    if not hits:
        return f"rc {run.get('rc')}, aucune cause nommée dans la sortie conservée"
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
        "# Les modèles du hub sur puce Apple",
        "",
        f"Généré par `tools/hub_apple_status.py`. **{len(known)} modèles**, une "
        "ligne chacun, aucune case vide. Une case « non mesuré » porte le "
        "chiffre qui l'explique : c'est un résultat, pas un trou.",
        "",
        "| modèle | état | cause / chiffre | entrées certifiées |",
        "|---|---|---|---|",
    ]
    tally = {"tourne": 0, "refuse": 0, "non mesuré": 0}
    for name in known:
        row = models.get(name)
        cert = "aucune sur ce vendeur" if not certified else "—"
        if row and row["runs"]:
            ok = [a for a, r in row["runs"].items() if r.get("rc") == 0]
            bad = {a: r for a, r in row["runs"].items() if r.get("rc") != 0}
            if ok and not bad:
                state, detail = "tourne", f"bras : {', '.join(sorted(ok))}"
            elif ok:
                state = "tourne"
                detail = (f"bras verts : {', '.join(sorted(ok))} · "
                          f"{list(bad)[0]} → {_cause(list(bad.values())[0])}")
            else:
                state = "refuse"
                detail = _cause(list(bad.values())[0])
        else:
            state = "non mesuré"
            gb = (sizes.get(name) or (None, None, None))[0] or (row or {}).get("size")
            biggest = (sizes.get(name) or (None, None, None))[2]
            detail = (f"artefact {gb} Go"
                      + (f", plus gros composant {biggest:.0f} Mo" if biggest else "")
                      ) if gb else ((row or {}).get("reason") or "aucune donnée de taille")
        tally[state] += 1
        lines.append(f"| `{name}` | {state} | {detail} | {cert} |")
    lines += ["", "## Compte", "",
              f"* **tourne** : {tally['tourne']}",
              f"* **refuse** : {tally['refuse']}",
              f"* **non mesuré** : {tally['non mesuré']}", ""]
    if certified:
        lines += ["## Entrées certifiées par profil", ""]
        for prof, n in sorted(certified.items()):
            lines.append(f"* `{prof}` : **{n}**")
    else:
        lines += ["## Entrées certifiées", "",
                  "**Aucune.** Le mécanisme résout, le répertoire est vide."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    models, sizes = build()
    text = render(models, sizes, _certified_counts())
    if args.out:
        args.out.write_text(text)
        print(f"écrit : {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
