#!/usr/bin/env python3
"""Does the memory class change what a certification finds? — the measurement
register 56 does not contain.

    tools/memory_class_sample.py draft <dir>      # reads a draft directory written by
                                                  # `autotune certify --out <dir>` on the OTHER
                                                  # memory class, compares each key against the
                                                  # engine directory's entry for the first class

For every key the draft certified, prints the config chosen on each class,
whether they agree, and the two deviations against the fp64 oracle; ends with
the counts. Same GV100 die, same locked clock, only the HBM differs — the
expectation is "identical", and this is how the expectation becomes a number.
A key certified on both classes with different configs is not a defect: the
directory serves each class its own; it is a finding for the record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "draft":
        print(__doc__); return 2
    from neurobrix.kernels import autotune_certified as C
    draft = Path(argv[1])
    engine = C.directory()
    agree = differ = missing = 0
    rows = []
    for path in sorted(draft.glob("*/*/*.json")):
        rel = path.relative_to(draft)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            e = json.loads((engine / rel).read_text(encoding="utf-8")) if (engine / rel).exists() else {"entries": {}}
        except (OSError, ValueError) as exc:
            print(f"unreadable {path}: {exc}"); continue
        for ktext, cert in (d.get("entries") or {}).items():
            base = (e.get("entries") or {}).get(ktext)
            cls_new = C.proof_memory_class(cert.get("proof"))
            if base is None:
                missing += 1
                rows.append((rel.name, ktext, cls_new, None, "no engine entry", None, None))
                continue
            classes = C.covered_memory_classes(base)
            other = next((c for c in classes if c != cls_new), None)
            ref = C.entry_for_memory_class(base, other) if other is not None else None
            if ref is None:
                missing += 1
                rows.append((rel.name, ktext, cls_new, None, "engine entry has no other class", None, None))
                continue
            same = (cert["config"]["kwargs"] == ref["config"]["kwargs"]
                    and cert["config"]["num_warps"] == ref["config"]["num_warps"]
                    and cert["config"]["num_stages"] == ref["config"]["num_stages"])
            agree += same; differ += (not same)
            rows.append((rel.name, ktext, cls_new, other, "same config" if same else "DIFFERENT config",
                         (cert["proof"].get("deviation"), ref["proof"].get("deviation")),
                         (cert["proof"].get("best_ms"), ref["proof"].get("best_ms"))))
    print("| file | key | new class | ref class | verdict | deviation new / ref | best ms new / ref |")
    print("|---|---|---:|---:|---|---|---|")
    for f, k, cn, co, v, dev, ms in rows:
        dv = f"{dev[0]:.2e} / {dev[1]:.2e}" if dev and None not in dev else "?"
        mv = f"{ms[0]:.4f} / {ms[1]:.4f}" if ms and None not in ms else "?"
        print(f"| {f} | `{k[:70]}` | {cn} | {co if co is not None else '—'} | {v} | {dv} | {mv} |")
    print(f"\n{agree} same config, {differ} different, {missing} without a counterpart "
          f"(over {agree + differ + missing} keys in the draft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
