#!/usr/bin/env python3
"""release_files_check.py --manifest FILE --dist DIR: the files about to be published ARE the files the release gate
tested.

The gate writes the manifest at the release tag (`release/<tag>.sha256`, one `<sha256>  <file name>` line per
distribution file it trialled); the publish workflow downloads the release's files into DIR and runs this check before
its upload step, which runs only when this exits 0.

CONTRACT
  * exit 0 when DIR holds exactly the manifest's files, each with the manifest's sha256;
  * exit 1 naming each difference: a file whose sha256 differs, a manifest file absent from DIR, a file in DIR the
    manifest does not list (nothing unlisted is ever uploaded);
  * exit 2, refused by name: a manifest that is missing, empty or has a malformed line, or a DIR with no file — a check
    that read nothing proves nothing.
  The verdict is printed; nothing is written.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

LINE = re.compile(r"^([0-9a-f]{64})  ([^/\s][^/]*)$")


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"no manifest at {path}")
    entries = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        m = LINE.match(line)
        if m is None:
            raise ValueError(f"{path}:{n}: not '<sha256>  <file name>': {line!r}")
        entries[m.group(2)] = m.group(1)
    if not entries:
        raise ValueError(f"the manifest {path} lists no file")
    return entries


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def differences(manifest: dict, dist: Path) -> list:
    found = {p.name: p for p in dist.iterdir() if p.is_file()} if dist.is_dir() else {}
    if not found:
        raise ValueError(f"no file in {dist}")
    out = []
    for name, want in sorted(manifest.items()):
        if name not in found:
            out.append(f"{name}: listed by the manifest, absent from {dist}")
        elif sha256(found[name]) != want:
            out.append(f"{name}: sha256 {sha256(found[name])} is not the tested file's {want}")
    out += [f"{name}: in {dist}, not listed by the manifest" for name in sorted(set(found) - set(manifest))]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the files to publish are the tested files")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dist", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        manifest = read_manifest(a.manifest)
        diff = differences(manifest, a.dist)
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if diff:
        print("\n".join(f"DIFFERS {d}" for d in diff))
        return 1
    print(f"SAME {len(manifest)} file(s): " + ", ".join(sorted(manifest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
