#!/usr/bin/env python3
"""Artifact integrity audit: local containers and runtime-cache entries vs the hub manifest.

Answers three questions that a half-written file makes look like a broken model:

  1. does every local ``.nbx`` whose name matches a hub artifact still carry that
     artifact's byte size, or has a local re-trace silently overwritten it?
  2. does every runtime-cache entry hold the files the loader needs
     (``manifest.json`` + ``topology.json`` + ``components/``), or was it cut
     mid-restore?
  3. are there zero-byte or obviously truncated files under the paths the engine
     reads?
  4. was any of them last written just before the machine came back up — i.e.
     possibly mid-write when the rack lost power?

The rack sits on a single breaker with no UPS (see the infrastructure dossier):
a cut drops this node and the storage server together, and **any file being
written at that instant can be left truncated**. A half-written container reads
exactly like a broken model, so this check runs BEFORE any engine diagnosis,
never after.

Read-only. Sizes only by default: the model store is on NFS and hashing it whole
costs hours; ``--sha SLUG`` hashes one named container when a size match is not
enough.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import re
import subprocess
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REGISTRY = "https://neurobrix.es"
CACHE = Path.home() / ".neurobrix" / "cache"
REQUIRED_CACHE_ENTRIES = ("manifest.json", "topology.json", "components")
# A file last written within this long before a boot was in flight close enough
# to the outage to be suspect. Heuristic, deliberately generous: it selects what
# to verify by checksum, it does not by itself condemn a file.
CUT_PROXIMITY = timedelta(minutes=20)


def boot_times() -> list[datetime]:
    """Start of every retained boot. An unplanned boot marks a power cut."""
    try:
        out = subprocess.run(
            ["journalctl", "--list-boots", "-o", "short-iso"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    stamps = []
    for line in out.splitlines():
        match = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", line)
        if match:
            stamps.append(datetime.fromisoformat(match.group(1).replace("T", " ")))
    return sorted(stamps)


def near_a_cut(mtime: float, boots: list[datetime]) -> datetime | None:
    """The boot this file was written just before, if any."""
    written = datetime.fromtimestamp(mtime)
    for boot in boots:
        if boot - CUT_PROXIMITY <= written <= boot:
            return boot
    return None


def normalise(name: str) -> str:
    """Store directories carry build suffixes the hub name does not (`-diffusers`)."""
    key = name.lower()
    for suffix in ("-diffusers", "-hf", "-nbx"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def hub_manifest(timeout: int = 30) -> dict[str, dict]:
    """Map normalised model name -> hub record (the cache and the store key by name)."""
    request = urllib.request.Request(
        f"{REGISTRY}/api/models?limit=500",
        headers={"User-Agent": "neurobrix-integrity-audit"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as r:
        payload = json.load(r)
    models = payload["models"] if isinstance(payload, dict) else payload
    return {normalise(m["name"]): m for m in models}


def scan_store(root: Path, hub: dict[str, dict], boots: list[datetime]) -> tuple[list[tuple[str, str, str]], int, int]:
    findings = []
    seen = matched = 0
    for nbx in sorted(root.rglob("*.nbx")):
        seen += 1
        stat = nbx.stat()
        size = stat.st_size
        boot = near_a_cut(stat.st_mtime, boots)
        if boot is not None:
            findings.append((
                "WRITTEN-AT-A-CUT", str(nbx),
                f"mtime {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M:%S} "
                f"is within {CUT_PROXIMITY} of the boot at {boot:%Y-%m-%d %H:%M:%S} "
                f"— verify by checksum before trusting it",
            ))
        if size == 0:
            findings.append(("EMPTY", str(nbx), "0 bytes"))
            continue
        # the store names a container by its parent directory, the hub by `name`
        record = hub.get(normalise(nbx.parent.name))
        if record is None:
            findings.append(("UNMATCHED", str(nbx), f"{size:,} bytes — no hub artifact of this name"))
            continue
        matched += 1
        expected = record.get("fileSize")
        expected = int(expected) if expected is not None else None
        if expected and size != expected:
            findings.append((
                "SIZE-MISMATCH", str(nbx),
                f"local {size:,} != hub {expected:,} ({size - expected:+,}) "
                f"[{record['slug']}, hub updatedAt {record.get('updatedAt')}]",
            ))
    return findings, seen, matched


def scan_cache(cache: Path) -> list[tuple[str, str, str]]:
    findings = []
    if not cache.is_dir():
        return findings
    for entry in sorted(p for p in cache.iterdir() if p.is_dir()):
        missing = [n for n in REQUIRED_CACHE_ENTRIES if not (entry / n).exists()]
        if missing:
            findings.append(("CACHE-INCOMPLETE", str(entry), f"missing {', '.join(missing)}"))
            continue
        try:
            created = json.loads((entry / "manifest.json").read_text()).get("created_at")
        except (OSError, ValueError) as exc:
            findings.append(("CACHE-UNREADABLE", str(entry), f"manifest.json: {exc}"))
            continue
        findings.append(("CACHE-OK", entry.name, f"created_at {created}"))
    return findings


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="models", help="model store root (default: models)")
    ap.add_argument("--sha", metavar="PATH", help="hash one container instead of scanning")
    args = ap.parse_args()

    if args.sha:
        path = Path(args.sha)
        print(f"{sha256(path)}  {path}  ({path.stat().st_size:,} bytes)")
        return 0

    hub = hub_manifest()
    print(f"hub manifest: {len(hub)} artifacts")

    boots = boot_times()
    print(f"boot table: {len(boots)} boots retained")
    store, seen, matched = scan_store(Path(args.store), hub, boots)
    cache = scan_cache(CACHE)

    print("\n== model store vs hub ==")
    print(f"  {seen} local .nbx scanned, {matched} matched a hub artifact by name")
    if not store:
        print("  no empty or size-mismatched container")
    for kind, path, detail in store:
        print(f"  {kind:14s} {path}\n                 {detail}")

    print("\n== runtime cache ==")
    incomplete = [f for f in cache if f[0] != "CACHE-OK"]
    print(f"  {len(cache) - len(incomplete)} complete, {len(incomplete)} incomplete")
    for kind, path, detail in incomplete:
        print(f"  {kind:18s} {path}  — {detail}")

    return 1 if (store or incomplete) else 0


if __name__ == "__main__":
    sys.exit(main())
