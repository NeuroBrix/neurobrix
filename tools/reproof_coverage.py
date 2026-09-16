#!/usr/bin/env python
"""How much of a certified directory has been re-proven, and under which code generator.

WHY THIS EXISTS
---------------
A setting is proven for ONE code generator; raising Triton re-proves the directory. The pass
takes a night across four cards, and nothing said how far it had got or whether it had finished:

* `autotune check` is the directory GATE. It reads every file and refuses one whose entries do
  not satisfy their own format — and it passes a 3.6.0 proof and a 3.8.0 proof alike, because
  that is not the question it asks. Measured 2026-09-16 on a directory 62 % re-proven: 12 files,
  9 866 shapes, **0 refused**.
* `autotune status` DOES report "proven under", and it is the right instrument — but it resolves
  the hardware profile first, so it answers nothing on a machine with no GPU visible, and on a
  busy rig asking it means putting a CUDA context beside a measurement in flight.

So the completeness of a re-proof had no instrument that runs anywhere. This reads the files and
answers from them: no profile, no driver, no card.

    python tools/reproof_coverage.py --dir <certified dir> [--under 3.8.0] [--json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def generators_of(entry: dict) -> set:
    """Every code generator this entry's proofs name — its own and its class variants'.

    A variant is a proof for another memory class, carried in the same entry, and it is proven
    separately: an entry whose primary is re-proven and whose 32 GB variant is not is PARTLY
    done, and saying otherwise would hide half the work.
    """
    out = set()
    for proof in ([entry.get("proof")] +
                  [v.get("proof") for v in (entry.get("variants") or {}).values() if v]):
        v = ((proof or {}).get("backend") or {}).get("triton")
        if v:
            out.add(str(v))
    return out


def class_of(proof: dict) -> str:
    """The memory class a proof was made on, from the proof itself.

    An entry's primary names its machine; a variant names its class in its own key. A setting
    serves only the class it was proven on, so "is this directory re-proven" has one answer per
    class and not one answer overall.
    """
    mb = (((proof or {}).get("machine") or {}).get("device") or {}).get("memory_mb")
    if isinstance(mb, (int, float)) and mb > 0:
        return f"{int(round(mb / 1024))}g"
    return "?"


def coverage(directory: Path, under: str) -> dict:
    files = []
    seen = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)
    for path in sorted(directory.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = doc.get("entries")
        if not isinstance(entries, dict):
            continue
        per = collections.Counter()
        for entry in entries.values():
            e = entry or {}
            pr = e.get("proof") or {}
            if ((pr.get("backend") or {}).get("triton")):
                cls = class_of(pr)
                per_class[cls]["total"] += 1
                per_class[cls]["done" if str((pr.get("backend") or {}).get("triton")) == under
                                else "behind"] += 1
            for name, var in (e.get("variants") or {}).items():
                vp = (var or {}).get("proof") or {}
                v = ((vp.get("backend") or {}).get("triton"))
                if v:
                    per_class[str(name)]["total"] += 1
                    per_class[str(name)]["done" if str(v) == under else "behind"] += 1
            gens = generators_of(e)
            for g in gens:
                seen[g] += 1
            if not gens:
                per["unproven"] += 1
            elif gens == {under}:
                per["done"] += 1
            elif under in gens:
                # Some proof of this entry is at the target and some is not — its primary is
                # re-proven and a memory-class variant is not, or the reverse. Counting it as
                # done would hide half the work, and counting it as behind would hide the half
                # that is finished, so it has its own name.
                per["partly"] += 1
            else:
                per["behind"] += 1
        files.append({"file": str(path.relative_to(directory)), "entries": len(entries),
                      "done": per["done"], "partly": per["partly"],
                      "behind": per["behind"], "unproven": per["unproven"]})
    tot = {k: sum(f[k] for f in files) for k in ("entries", "done", "partly", "behind", "unproven")}
    tot["percent_done"] = round(100.0 * tot["done"] / tot["entries"], 1) if tot["entries"] else 0.0
    by_class = {k: {"total": v["total"], "done": v["done"], "behind": v["behind"],
                    "percent_done": round(100.0 * v["done"] / v["total"], 1) if v["total"] else 0.0}
                for k, v in sorted(per_class.items())}
    return {"schema": "neurobrix.reproof.coverage/1", "directory": str(directory),
            "under": under, "generators_seen": dict(sorted(seen.items())),
            "by_memory_class": by_class, "files": files, "total": tot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--under", default="3.8.0", help="the generator version the pass is re-proving to")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    if not d.is_dir():
        print(f"[coverage] not a directory: {d}", file=sys.stderr)
        return 2
    rec = coverage(d, a.under)
    if a.json:
        print(json.dumps(rec, indent=1))
        return 0
    print(f"{'file':40s}{'entries':>8}{'done':>7}{'partly':>8}{'behind':>8}{'unproven':>10}")
    for f in rec["files"]:
        print(f"{f['file']:40s}{f['entries']:>8}{f['done']:>7}{f['partly']:>8}"
              f"{f['behind']:>8}{f['unproven']:>10}")
    t = rec["total"]
    print(f"{'TOTAL':40s}{t['entries']:>8}{t['done']:>7}{t['partly']:>8}"
          f"{t['behind']:>8}{t['unproven']:>10}")
    print(f"\nfully re-proven under {a.under}: {t['done']}/{t['entries']} = {t['percent_done']} %"
          + (f"   (+{t['partly']} partly: one class re-proven, the other not)" if t["partly"] else ""))
    print(f"generators seen: {', '.join(f'{k} ({v})' for k, v in rec['generators_seen'].items())}")
    print("\nby memory class — a setting serves ONLY the class it was proven on, so this is the")
    print("number that says how far the pass has got; an entry reads fully done only once EVERY")
    print("class it carries is at the target, which for a two-tree pass means after the merge:")
    for k, v in rec["by_memory_class"].items():
        print(f"   {k:>6}: {v['done']:>6}/{v['total']:<6} = {v['percent_done']:>5} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
