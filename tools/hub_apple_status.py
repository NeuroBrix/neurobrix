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
    # A campaign that runs one model per call writes one level deeper.
    out += list(ATELIER.glob("*/*/records.json"))
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
#: this pattern misses prints as "aucune cause nommée", which is a small lie in
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
        "## Le dénominateur, parce que deux tableaux qui ne comptent pas la "
        "même population ne se posent pas côte à côte",
        "",
        f"Ce tableau compte **{len(known)} artefacts** : l'union de ceux qui "
        "portent un enregistrement de campagne E et de ceux que le registre "
        "des différés chiffre. C'est la population **mesurable depuis ce "
        "poste**, pas le catalogue du hub.",
        "",
        "Le tableau de l'autre machine compte **47 modèles**, qui est le "
        "**catalogue du hub**. Les deux chiffres sont justes et ne désignent "
        "pas la même chose. La correspondance connue : quatre des artefacts "
        "comptés ici sont des **variantes ou sauvegardes** d'un modèle du "
        "catalogue (`-int4g128`, `-ffnonly`, `.pre-G-backup`), qu'un "
        "catalogue compte une fois et qu'un montage porte séparément.",
        "",
        "**Le bon dénominateur dépend de la question.** Pour « que fait le "
        "hub sur Apple », c'est le catalogue : 47. Pour « qu'a-t-on mesuré "
        "et que reste-t-il », c'est cette liste, parce qu'une variante "
        "quantifiée se mesure et se refuse séparément de son modèle de base.",
        "",
        "**Non vérifié d'ici :** le catalogue n'a pas pu être relu — "
        "`neurobrix hub` échoue sur la vérification du certificat de "
        "`neurobrix.es` depuis ce poste. Le 47 est donc repris tel que "
        "l'autre machine le rapporte, et non confirmé.",
        "",
        "**La colonne « mesuré le » décide de la lecture.** Les verdicts "
        "viennent de campagnes de dates différentes, sur des arbres "
        "différents : une ligne du 9 septembre ne dit rien de l'arbre "
        "d'aujourd'hui. Sans cette colonne le tableau serait faux tout en "
        "étant exact.",
        "",
        "| modèle | état | cause / chiffre | mesuré le | entrées cert. |",
        "|---|---|---|---|---|",
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
        when = max((d for d, _ in (row or {}).get("seen", [])), default="—")
        lines.append(f"| `{name}` | {state} | {detail} | {when} | {cert} |")
    lines += ["", "## Compte", "",
              f"* **tourne** : {tally['tourne']}",
              f"* **refuse** : {tally['refuse']}",
              f"* **non mesuré** : {tally['non mesuré']}", ""]
    if certified:
        lines += ["## Entrées certifiées par profil (formes, pas fichiers)", ""]
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
