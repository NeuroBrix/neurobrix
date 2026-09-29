#!/usr/bin/env python3
"""Migrate conv2d_forward_kernel keys to the right `fp16` flag, in place — the owner's decision of
2026-09-29 02:49, after the IR proof.

The wrapper computed the key's `fp16` flag as `x.dtype == NBXDtype.float16`, a Triton element type
compared to an IntEnum: False for every launch. Fixed at the source, an fp16 input now keys True.
The flag's only effect in the kernel casts the loaded blocks to fp16 before tl.dot — a no-op on an
fp16 input: TTIR, TTGIR, LLIR and PTX are IDENTICAL with the flag False and True (sm_70, three
shapes; the same comparison sees the real cast on fp32 pointers). So a certification made under the
wrong key proves the right one: every entry whose INPUT tag is 'fp16' and whose flag reads False is
re-keyed with the flag True — its configuration and proof kept, the proof's `shape` re-keyed with
it — and every census-table row the same. Nothing is swept or re-proven.

    python tools/migrate_conv_fp16_key.py --directory src/neurobrix/config/autotune/nvidia/volta \\
        --tables src/neurobrix/config/census/nvidia/volta [--dry-run]

Each file is rewritten under its own lock (the certifier's `.json.lock`, the census table's
`.jsonl.lock`). Idempotent: a second run finds nothing to migrate. One tool for every vendor
directory (the same kernel runs on Apple).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

KERNEL = "neurobrix.kernels.ops.conv2d.conv2d_forward_kernel"
FLAG, INPUT_TAG = 16, 17          # positions in the key: 'fp16', then the input's dtype tag


def migrated(key: tuple):
    """The key with the flag right, or None when it needs nothing."""
    if len(key) > INPUT_TAG and key[FLAG] is False and key[INPUT_TAG] == "fp16":
        return key[:FLAG] + (True,) + key[FLAG + 1:]
    return None


def _rekey_proof(entry: dict, new_key: tuple) -> None:
    for cert in [entry] + list((entry.get("variants") or {}).values()):
        proof = cert.get("proof") or {}
        if isinstance(proof.get("shape"), list):
            proof["shape"] = list(new_key)


def migrate_directory(directory: Path, dry: bool) -> int:
    from neurobrix.kernels import autotune_certified as C
    n = 0
    for path in sorted(directory.glob("conv2d_forward_kernel.*.json")):
        with open(path.with_suffix(".json.lock"), "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                entries, moved = dict(doc.get("entries") or {}), 0
                for ktext, entry in list(entries.items()):
                    new = migrated(C.parse_key(ktext))
                    if new is None:
                        continue
                    ntext = C.key_repr(new)
                    if ntext in entries:
                        raise SystemExit(f"{path}: {ntext} already exists beside {ktext} — refused, nothing written")
                    _rekey_proof(entry, new)
                    entries[ntext] = entries.pop(ktext)
                    moved += 1
                if moved and not dry:
                    doc["entries"] = dict(sorted(entries.items()))
                    tmp = path.with_suffix(".json.migrate.tmp")
                    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
                    tmp.replace(path)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
        print(f"[migrate] {path.name}: {moved} entr{'y' if moved == 1 else 'ies'} re-keyed"
              f"{' (dry run)' if dry else ''}", flush=True)
        n += moved
    return n


def migrate_tables(tables: Path, dry: bool) -> int:
    from neurobrix.kernels import autotune_certified as C
    from neurobrix.kernels import census_table as T
    n = 0
    for path in sorted(tables.glob("*g.jsonl")):
        with open(str(path) + ".lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                rows, moved = T.read(path), 0
                for r in rows:
                    if r["kernel"] != KERNEL:
                        continue
                    new = migrated(C.parse_key(r["key"]))
                    if new is not None:
                        r["key"] = C.key_repr(new)
                        moved += 1
                if moved and not dry:
                    T.write(path, rows)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
        print(f"[migrate] {path.name}: {moved} row(s) re-keyed{' (dry run)' if dry else ''}", flush=True)
        n += moved
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--directory", type=Path, help="a certified directory (<vendor>/<profile>)")
    ap.add_argument("--tables", type=Path, help="a census table directory (<vendor>/<profile>)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.directory and not a.tables:
        ap.error("name --directory, --tables or both")
    total = 0
    if a.directory:
        total += migrate_directory(a.directory, a.dry_run)
    if a.tables:
        total += migrate_tables(a.tables, a.dry_run)
    print(f"[migrate] {total} re-keyed in all", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
