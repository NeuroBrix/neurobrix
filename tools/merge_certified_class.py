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


def generator_of(proof: dict) -> tuple:
    """The code generator a proof was made under, as a comparable tuple. () when absent."""
    v = (((proof or {}).get("backend") or {}).get("triton"))
    if not isinstance(v, str):
        return ()
    out = []
    for part in v.split("."):
        out.append(int(part) if part.isdigit() else 0)
    return tuple(out)


def destination_proof_for(d_entry: dict, cls: int, slot: str):
    """What the destination already holds for this class, or None."""
    if C.proof_memory_class((d_entry.get("proof") or {})) == cls:
        return d_entry.get("proof")
    return ((d_entry.get("variants") or {}).get(slot) or {}).get("proof")


def merge_file(src: Path, dst: Path, cls: int, apply: bool) -> dict:
    s = json.loads(src.read_text(encoding="utf-8"))
    d = json.loads(dst.read_text(encoding="utf-8")) if dst.exists() else dict(s, entries={})
    s_entries, d_entries = s.get("entries") or {}, d.get("entries") or {}
    slot = C._variant_slot(cls)
    moved = added = skipped = kept_newer = 0
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
        # A source tree SEEDED from a live one carries a snapshot of it, so a key the live tree
        # re-proved after the seed is OLDER here. Writing it back would regress a proof and count
        # as a success: `moved` cannot tell a fresher proof from a staler one, and nothing else
        # looks. So the generators are compared, and a move that would go BACKWARDS is refused
        # and counted under its own name (2026-09-17, before the two 32 GB trees were merged).
        have = destination_proof_for(d_entry, cls, slot)
        if have is not None and generator_of(have) > generator_of(cert.get("proof")):
            kept_newer += 1
            continue
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
            "kept_newer_in_destination": kept_newer,
            "source_without_that_class": skipped, "entries_after": len(d_entries)}


def refuse_a_destination_that_holds_no_kernel_files(src: Path, dst: Path) -> None:
    """Refuse `--into` pointed at the directory ABOVE the kernel files.

    The certified directory is `autotune/<vendor>/<profile>/<kernel>.<dtype>.json` and a side
    tree written by `certify --out` is FLAT. So `--into .../config/autotune` is a plausible
    thing to type and a silently wrong thing to do: every key reads as absent, the summary says
    `moved: 0, added: 4641` — which looks like a successful first merge — and the tool writes a
    flat `matmul_kernel.fp32.json` at the autotune ROOT that serves no card, while the real
    directory is left untouched. No error, a plausible report, and a directory that looks fuller
    than it is (2026-09-16, caught by a dry run before the real merge).

    The discriminant is what the destination HOLDS: kernel files, or directories of them.
    """
    if any(dst.glob("*.json")):
        return
    nested = sorted(d for d in dst.glob("*/*") if d.is_dir() and any(d.glob("*.json")))
    names = sorted(p.name for p in src.glob("*.json"))
    raise SystemExit(
        f"[merge] --into {dst} holds no kernel file. The certified directory is\n"
        f"   autotune/<vendor>/<profile>/<kernel>.<dtype>.json\n"
        f"and a side tree is flat, so pointing --into one level too high writes new flat files\n"
        f"that serve no card and reports them as a successful first merge.\n"
        + (f"   Meant one of: {', '.join(str(d) for d in nested)}\n" if nested else
           f"   Nothing below it holds kernel files either — check the path.\n")
        + f"   The source carries {len(names)} kernel file(s): {', '.join(names[:4])}"
        + (" ..." if len(names) > 4 else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--into", dest="dst", required=True)
    ap.add_argument("--class", dest="cls", type=int, required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    refuse_a_destination_that_holds_no_kernel_files(src, dst)
    rows = [merge_file(p, dst / p.name, a.cls, a.apply) for p in sorted(src.glob("*.json"))]
    print(json.dumps({"class_gb": a.cls, "applied": a.apply, "files": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
