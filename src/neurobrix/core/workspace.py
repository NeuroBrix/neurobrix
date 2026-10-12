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


def io_workers() -> int:
    """How many shard reads the weight loaders keep in flight: `NBX_IO_WORKERS`, else `io.num_workers`
    in config/system.yml — read by every loader AND by Prism's host estimate (each read in flight holds
    a shard in host memory), so the figure has one source. Until 2026-09-27 four modules each wrote
    `8` beside a comment saying it "matches system.yml io.num_workers", and none read the file."""
    env = os.environ.get("NBX_IO_WORKERS")
    if env:
        return int(env)
    cfg = yaml.safe_load(SYSTEM_YML.read_text()) or {}
    n = (cfg.get("io") or {}).get("num_workers")
    if not isinstance(n, int) or n < 1:
        raise RuntimeError(
            f"ZERO FALLBACK: no I/O worker count is configured: looked at $NBX_IO_WORKERS and "
            f"{SYSTEM_YML} under `io.num_workers` (got {n!r}).")
    return n


def pinned_uploads_in_flight() -> int:
    """How many non-blocking uploads one loader worker keeps outstanding (`io.pinned_uploads_in_flight`
    in config/system.yml) — read by the compiled loader AND by Prism's host estimate, which prices the
    pinned bytes of a load as workers x this x the largest tensor."""
    cfg = yaml.safe_load(SYSTEM_YML.read_text()) or {}
    n = (cfg.get("io") or {}).get("pinned_uploads_in_flight")
    if not isinstance(n, int) or n < 1:
        raise RuntimeError(
            f"ZERO FALLBACK: no pinned upload depth is configured: looked at {SYSTEM_YML} under "
            f"`io.pinned_uploads_in_flight` (got {n!r}).")
    return n


_mmap_threshold_set = False


def _libc():
    import ctypes
    import ctypes.util
    return ctypes.CDLL(ctypes.util.find_library("c"))


def apply_host_mmap_threshold() -> None:
    """Pin glibc's mmap threshold to `io.host_mmap_threshold_bytes` (config/system.yml), once per process.

    A weight loader's large temporaries must go back to the OS when freed; with glibc's dynamic threshold
    they stay in the heap (measured figures beside the value). Not glibc (macOS, musl, Windows): there is
    no such threshold and nothing is done."""
    global _mmap_threshold_set
    if _mmap_threshold_set:
        return
    _mmap_threshold_set = True
    import platform
    if platform.libc_ver()[0] != "glibc":
        return
    cfg = yaml.safe_load(SYSTEM_YML.read_text()) or {}
    n = (cfg.get("io") or {}).get("host_mmap_threshold_bytes")
    if not isinstance(n, int) or n < 1:
        raise RuntimeError(
            f"ZERO FALLBACK: no host mmap threshold is configured: {SYSTEM_YML} under "
            f"`io.host_mmap_threshold_bytes` (got {n!r}).")
    M_MMAP_THRESHOLD = -3  # glibc <malloc.h>
    if _libc().mallopt(M_MMAP_THRESHOLD, n) != 1:
        raise RuntimeError(f"mallopt(M_MMAP_THRESHOLD, {n}) refused by glibc")


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
