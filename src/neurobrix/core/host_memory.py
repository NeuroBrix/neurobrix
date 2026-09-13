"""What the machine actually has free, right now.

A budget that describes what the hardware *recommends* is not a measure of
what is *available*. On unified memory the two are unrelated: this M4 Pro
reports `recommendedMaxWorkingSetSize` = 18 186 MB and does not lower it by
one byte when another process holds a third of the machine. A user never has
an empty machine — a browser, a mail client, sometimes a container or a VM —
so a plan sized against the recommendation is accepted and then killed by the
system under memory pressure, mid-render, without a word.

This module measures the other number. It has two consumers and they want
the same reading:

  * the planner, which must refuse loudly and name BOTH figures rather than
    start and be killed;
  * the measurement protocol, because a green obtained on a half-occupied
    machine is not the same green as one obtained on a free machine, and
    without this line written at the time nobody can tell them apart later.

Nothing is guessed. Where the platform is not one this module knows how to
read, `available_mb` is None and `source` says so — the caller then announces
that it could not be measured, which is honest, rather than substituting a
number, which is not.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

_MB = 1024 * 1024


@dataclass(frozen=True)
class MemoryState:
    """One reading of the machine, at one instant."""

    total_mb: Optional[int] = None
    available_mb: Optional[int] = None
    swap_used_mb: Optional[int] = None
    swap_total_mb: Optional[int] = None
    #: (command, resident MB), largest first — this is what names the VM or
    #: the browser holding the memory, without hardcoding either.
    largest_residents: Tuple[Tuple[str, int], ...] = field(default=())
    #: How it was read, or why it could not be.
    source: str = "not measured"

    @property
    def measured(self) -> bool:
        return self.available_mb is not None

    def describe(self) -> str:
        """The one line every measurement cell writes beside its result."""
        if not self.measured:
            return f"machine memory: NOT MEASURED ({self.source})"
        parts = [f"{self.available_mb} MB free"]
        if self.total_mb:
            parts.append(f"of {self.total_mb} MB")
        if self.swap_total_mb:
            parts.append(f"swap {self.swap_used_mb}/{self.swap_total_mb} MB")
        if self.largest_residents:
            biggest = ", ".join(f"{name} {mb} MB"
                                for name, mb in self.largest_residents)
            parts.append(f"largest resident: {biggest}")
        return "machine memory: " + ", ".join(parts)


def _run(cmd) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _largest_residents(count: int = 3) -> Tuple[Tuple[str, int], ...]:
    """The biggest resident processes, by RSS. Named, never acted on."""
    out = _run(["ps", "-Ao", "rss,comm", "-r"])
    if not out:
        return ()
    found = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        rss, _, comm = line.partition(" ")
        if not rss.isdigit():
            continue
        mb = int(rss) // 1024
        if mb <= 0:
            continue
        found.append((comm.strip().rsplit("/", 1)[-1], mb))
        if len(found) >= count:
            break
    return tuple(found)


def _macos_state() -> MemoryState:
    """`vm_stat` for pages, `sysctl` for the totals and for swap.

    Available is free + inactive + purgeable + speculative pages. Inactive
    and purgeable pages are reclaimable on demand, which is why macOS itself
    counts them as available; wired and compressed pages are not, and are
    deliberately excluded.
    """
    vm = _run(["vm_stat"])
    if not vm:
        return MemoryState(source="vm_stat unavailable")
    page = re.search(r"page size of (\d+) bytes", vm)
    page_size = int(page.group(1)) if page else 4096

    def pages(label: str) -> int:
        m = re.search(rf"{label}:\s+(\d+)", vm)
        return int(m.group(1)) if m else 0

    available_pages = (pages("Pages free") + pages("Pages inactive")
                       + pages("Pages purgeable") + pages("Pages speculative"))
    available_mb = available_pages * page_size // _MB

    total_mb = None
    memsize = _run(["sysctl", "-n", "hw.memsize"])
    if memsize and memsize.strip().isdigit():
        total_mb = int(memsize.strip()) // _MB

    swap_used = swap_total = None
    swap = _run(["sysctl", "-n", "vm.swapusage"])
    if swap:
        tot = re.search(r"total\s*=\s*([\d.]+)M", swap)
        use = re.search(r"used\s*=\s*([\d.]+)M", swap)
        if tot:
            swap_total = int(float(tot.group(1)))
        if use:
            swap_used = int(float(use.group(1)))

    return MemoryState(total_mb=total_mb, available_mb=available_mb,
                       swap_used_mb=swap_used, swap_total_mb=swap_total,
                       largest_residents=_largest_residents(),
                       source="vm_stat + sysctl")


def _linux_state() -> MemoryState:
    """`MemAvailable` is the kernel's own estimate of what a new allocation
    can have without swapping — exactly the question, already answered."""
    try:
        text = open("/proc/meminfo").read()
    except OSError:
        return MemoryState(source="/proc/meminfo unreadable")

    def kb(label: str) -> Optional[int]:
        m = re.search(rf"^{label}:\s+(\d+) kB", text, re.MULTILINE)
        return int(m.group(1)) if m else None

    total, avail = kb("MemTotal"), kb("MemAvailable")
    swap_total, swap_free = kb("SwapTotal"), kb("SwapFree")
    return MemoryState(
        total_mb=total // 1024 if total else None,
        available_mb=avail // 1024 if avail is not None else None,
        swap_total_mb=swap_total // 1024 if swap_total else None,
        swap_used_mb=((swap_total - swap_free) // 1024
                      if swap_total is not None and swap_free is not None
                      else None),
        largest_residents=_largest_residents(),
        source="/proc/meminfo")


def memory_state() -> MemoryState:
    """Read the machine now. Never cached: the whole point is that it moves."""
    if sys.platform == "darwin":
        return _macos_state()
    if sys.platform.startswith("linux"):
        return _linux_state()
    return MemoryState(source=f"no reader for platform {sys.platform!r}")
