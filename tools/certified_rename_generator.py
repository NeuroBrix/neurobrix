#!/usr/bin/env python3
"""Rename the generator identity a set of certified entries carries, from a date-based backend hash to the content
identity of the SAME bytes (autotune_certified._backend_content_hash). For the migration only: an entry proves a code;
triton's own backend hash folded in file dates, so a byte-identical backend restored or copied carried a new hash and
its certificates stopped serving (the Mac, 2026-10-20: 42ebffa -> 62109df8 over the same files).

The caller states the old hashes and the content identity they correspond to, and the evidence of that identity
(--evidence, recorded in every renamed proof). The tool changes nothing else in an entry and refuses an entry whose
hash is not one of --from.

usage: certified_rename_generator.py --dir <vendor/profile dir> --from H1,H2 --to content-XXXX --evidence "..." [--dry-run]
"""
import argparse, json, sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    old = {h.strip() for h in a.old.split(",") if h.strip()}
    if not a.to.startswith("content-"):
        raise SystemExit(f"--to {a.to!r}: the target is a content identity (content-...), never another date-based hash")
    total = 0
    for f in sorted(Path(a.dir).glob("*.json")):
        doc = json.loads(f.read_text())
        n = 0
        for key, e in doc.get("entries", {}).items():
            for proof in [e.get("proof")] + [v.get("proof") for v in (e.get("variants") or {}).values()
                                             if isinstance(v, dict)]:
                b = (proof or {}).get("backend") or {}
                if b.get("backend_hash") in old:
                    b["backend_hash_was"] = b["backend_hash"]
                    b["backend_hash"] = a.to
                    b["renamed_because"] = a.evidence
                    n += 1
        if n and not a.dry_run:
            f.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
        print(f"{f.name}: {n} proof(s) renamed{' (dry run)' if a.dry_run else ''}")
        total += n
    print(f"total {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
