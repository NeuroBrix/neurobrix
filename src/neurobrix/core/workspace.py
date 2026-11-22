"""Where model data lives, read from configuration. Nothing downloads a copy.

A snapshot is downloaded ONCE, into the configured snapshots root, and every
consumer reads it THERE — the build toolchain builds from it, vendor tools and
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
import logging
_log = logging.getLogger(__name__)

SYSTEM_YML = Path(__file__).resolve().parent.parent / "config" / "system.yml"

_root: Optional[Path] = None


#: the reads in flight a weight load may grow to, as the plan in force priced host memory for them (`declare_load_ceiling`,
#: Prism's host estimate); None: no plan priced them in this process
_LOAD_CEILING: Optional[int] = None
#: (pid, reader): the process's weight reader (`weight_reader`); a forked child builds its own
_READER: Optional[tuple] = None


def declare_load_ceiling(n: int) -> None:
    """The reads in flight Prism's host estimate priced for the plan in force (`PrismSolver`, step 7.9): the ceiling the
    weight reader may grow to (`weight_reader`)."""
    global _LOAD_CEILING
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"ZERO FALLBACK: a load ceiling is a count of reads >= 1, not {n!r}")
    _LOAD_CEILING = n


def processors() -> int:
    """The processors this process may run on (its affinity where the system states one, else the machine's count): the
    most reads and conversions that can work at once. Refused by name when the system states neither."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    n = os.cpu_count()
    if not n:
        raise RuntimeError("ZERO FALLBACK: the system states no processor count: no bound for parallel reads")
    return n


def load_ceiling() -> int:
    """The most reads a weight load keeps in flight: what the plan in force priced host memory for
    (`declare_load_ceiling`); with no plan in this process (a probe child, a tool), the processors it may run on — the
    bound of what can work at once, and nothing is priced against it there."""
    return _LOAD_CEILING if _LOAD_CEILING is not None else processors()


def weight_reader(source=None):
    """The process's weight reader (`nbxtensor.io.reader.Reader`): parallel reads that start at one and grow by their own
    measure up to `load_ceiling()`, or at the width the machine kept from an earlier load (`kept_launch_facts.kept_read_width`,
    written by `kept_launch_facts.note_read_width` after a load). No up-front trial (the launch-time cure, 2026-11-19). A new ceiling
    (another plan) starts a new reader, and the one it replaces is closed. `source` (a weight file the load reads):
    the kept width is the one kept for its storage, the library and the kernel (`kept_launch_facts.read_key`); a
    reader made without one starts at one."""
    global _READER
    ceiling = load_ceiling()
    if _READER is None or _READER[0] != os.getpid() or _READER[1].ceiling != ceiling:
        from nbxtensor.io.reader import Reader
        from neurobrix.core.prism import autodetect as _ad, kept_launch_facts as _kept
        if _READER is not None and _READER[0] == os.getpid():
            _READER[1].close()               # the reader a new ceiling replaces ends its threads (a forked child owns none)
        # the width this machine kept from an earlier load (row 167: measured once, never paid twice), else one
        _READER = (os.getpid(), Reader(ceiling, start=_kept.kept_read_width(_ad.ACTIVE_HARDWARE_ID, source)))
    return _READER[1]


def weight_reader_measured() -> Optional[dict]:
    """What this process's weight reader measured (`Reader.measured()`), for the engine to keep; None before any read."""
    if _READER is None or _READER[0] != os.getpid() or not _READER[1].bytes:
        return None
    return _READER[1].measured()


def weight_reader_said() -> Optional[str]:
    """What this process's weight loads measured, worded for the launch line; None when nothing was read here."""
    if _READER is None or _READER[0] != os.getpid() or not _READER[1].bytes:
        return None
    return _READER[1].said()


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


_huge_pages_set = False


def apply_host_huge_pages() -> None:
    """Tell numpy whether its large arrays may ask for transparent huge pages (`io.host_huge_pages`,
    config/system.yml), once per process: a weight loader's read and cast buffers are numpy arrays, and a
    huge-page request on a fragmented host compacts memory synchronously (measured figures beside the
    value). numpy's own switch (`_set_madvise_hugepage`, the in-process form of NUMPY_MADVISE_HUGEPAGE);
    where numpy never madvises (not Linux) the switch changes nothing."""
    global _huge_pages_set
    if _huge_pages_set:
        return
    _huge_pages_set = True
    cfg = yaml.safe_load(SYSTEM_YML.read_text()) or {}
    v = (cfg.get("io") or {}).get("host_huge_pages")
    if not isinstance(v, bool):
        raise RuntimeError(
            f"ZERO FALLBACK: no host huge-page policy is configured: {SYSTEM_YML} under "
            f"`io.host_huge_pages` (got {v!r}).")
    from numpy._core import multiarray as _ma
    _ma._set_madvise_hugepage(v)


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
        _root = Path(declared).expanduser()      # `~/...`: the shipped value names no machine's home
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
