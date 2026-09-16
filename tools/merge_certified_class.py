#!/usr/bin/env python
"""Merge the proofs of ONE memory class from a side directory into the engine's.

Two certifiers may not write one file (2026-09-16: cards 0 and 2 shared
`matmul_kernel.fp32.json` and the second died on a rename the first had
replaced). But the two card classes of this rack must be re-proven in parallel
or the pass takes twice the night. So each class writes its OWN tree
(`certify --out <dir>`), and this tool carries one class's proofs back:

For every shape key present in BOTH trees, the source entry's proof FOR THE
GIVEN CLASS (its primary proof or its variant for that class) is written into
the destination entry's slot for that class. Nothing else moves: the
destination's other class, its excluded configs and its own config stay as
they are. A key only in the source is copied whole; a key only in the
destination is untouched.

    python tools/merge_certified_class.py --from <dir> --into <dir> --class 32 [--apply]

Without `--apply` it says what it would do and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neurobrix.kernels import autotune_certified as C  # noqa: E402


def class_proof(entry: dict, cls: int):
    """The (config, proof, excluded) this entry carries FOR that class, or None."""
    cert = C.entry_for_memory_class(entry, cls)
    if cert is None:
        return None
    return {k: v for k, v in cert.items() if k in ("config", "proof", "excluded")}


def merge_file(src: Path, dst: Path, cls: int, apply: bool) -> dict:
    s = json.loads(src.read_text(encoding="utf-8"))
    d = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else dict(s, entries={})
    s_entries, d_entries = s.get("entries") or {}, d.get("entries") or {}
    slot = C._variant_slot(cls)
    moved = added = skipped = 0
    for key, s_entry in s_entries.items():
        cert = class_proof(s_entry, cls)
        if cert is None:
            skipped += 1
            continue
        if key not in d_entries:
            d_entries[key] = json.loads(json.dumps(s_entry))
            added += 1
            continue
        d_entry = d_entries[key]
        d_cls = C.proof_memory_class((d_entry.get("proof") or {}))
        if d_cls == cls:                       # the destination's PRIMARY is this class
            d_entry.update(cert)
        else:
            d_entry.setdefault("variants", {})[slot] = cert
        moved += 1
    d["entries"] = d_entries
    if apply:
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        tmp.replace(dst)
    return {"file": dst.name, "moved": moved, "added": added,
            "source_without_that_class": skipped, "entries_after": len(d_entries)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--into", dest="dst", required=True)
    ap.add_argument("--class", dest="cls", type=int, required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    rows = [merge_file(p, dst / p.name, a.cls, a.apply) for p in sorted(src.glob("*.json"))]
    print(json.dumps({"class_gb": a.cls, "applied": a.apply, "files": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
