"""Where model data lives, read from configuration. Nothing downloads a copy.

A snapshot is downloaded ONCE, into the configured snapshots root, and every
consumer reads it THERE — the build toolchain traces from it, vendor tools and
oracles load from it, the engine's own third-party fallbacks resolve against it.

The rule exists because a bare hub id handed to `from_pretrained` silently
fetches a second copy into `~/.cache/huggingface/hub`. On 2026-09-17 that cache
held 4.0 G across 25 entries — every single one of which already had its
snapshot on the NAS — on a local disk that had reached 0 MB the same day.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

SYSTEM_YML = Path(__file__).resolve().parent.parent / "config" / "system.yml"

_root: Optional[Path] = None


class SnapshotNotPresent(Exception):
    """A snapshot is not in the configured root, and nothing fetches it here."""


def snapshots_root() -> Path:
    """The configured snapshots root. `NEUROBRIX_HF_SNAPSHOTS` overrides it."""
    global _root
    env = os.environ.get("NEUROBRIX_HF_SNAPSHOTS")
    if env:
        return Path(env)
    if _root is None:
        cfg = yaml.safe_load(SYSTEM_YML.read_text()) or {}
        declared = (cfg.get("paths") or {}).get("snapshots")
        if not declared:
            raise SnapshotNotPresent(
                f"no snapshots root is configured: looked at "
                f"$NEUROBRIX_HF_SNAPSHOTS and {SYSTEM_YML} under "
                f"`paths.snapshots`. There is deliberately no default — a "
                f"default here downloads model data onto whichever disk the "
                f"process happens to run on.")
        _root = Path(declared)
    return _root


def snapshot_path(*names: str) -> Path:
    """The first of `names` present under the root, or a refusal naming them all.

    Several names because a directory on disk is not always spelled like the hub
    id it came from.
    """
    root = snapshots_root()
    for n in names:
        candidate = root / n
        if candidate.is_dir():
            return candidate
    raise SnapshotNotPresent(
        f"none of {list(names)} is in {root}. Download it THERE once; this "
        f"engine reads snapshots in place and does not fetch a second copy "
        f"into the Hugging Face hub cache, which must stay empty.")
