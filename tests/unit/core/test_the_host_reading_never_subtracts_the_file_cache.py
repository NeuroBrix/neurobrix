"""The host's available memory on macOS never subtracts the file cache (inbox 68, 2026-09-29).

orpheus-3b was refused at placement one second into a certified-only run, right after its 14.5 GB in-flow copy:
`mps:0: 17277 MB recommended by the device, 10242 MB actually usable right now`. The reading counted free + inactive
+ purgeable + speculative pages and so dropped every ACTIVE file-backed page — the weights just copied, which the
kernel gives back the moment an allocation asks. A user who has just downloaded a model gets the same refusal.

The reclaimable set is free + purgeable + file-backed pages (vm_stat's `File-backed pages` already holds the
speculative queue: active + inactive + speculative == file-backed + anonymous, measured on this M4 Pro). Anonymous
pages, active or inactive, are NOT counted: they come back only through the compressor and swap.
"""
import os
import sys
import time
from pathlib import Path

import pytest

from neurobrix.core import host_memory as H

# vm_stat as it read at 12:26:xx on 2026-09-29, the orpheus moment (page size 16384): 330 MB free, 10 GB inactive,
# 14.4 GB file-backed — the copied container in the cache.
VM_STAT_AFTER_THE_COPY = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    21203.
Pages active:                                 519186.
Pages inactive:                               639686.
Pages speculative:                               174.
Pages throttled:                                   0.
Pages wired down:                             181379.
Pages purgeable:                                5893.
"Translation faults":                     1234567890.
Pages copy-on-write:                        12345678.
Pages zero filled:                         123456789.
Pages reactivated:                          12345678.
Pages purged:                               95311314.
File-backed pages:                            924473.
Anonymous pages:                              234573.
Pages stored in compressor:                    98765.
Pages occupied by compressor:                  31148.
Decompressions:                              1234567.
Compressions:                                2345678.
Pageins:                                    34567890.
Pageouts:                                     456789.
Swapins:                                    44669081.
Swapouts:                                   75168098.
"""
PAGE = 16384
MB = 1024 * 1024


def test_the_reading_after_a_large_copy_holds_the_file_cache():
    """Parsing the captured counters: the file-backed 14 445 MB are reclaimable and are counted."""
    available = H._available_mb_from_vm_stat(VM_STAT_AFTER_THE_COPY)
    file_backed_mb = 924473 * PAGE // MB          # 14 445 MB
    assert available >= file_backed_mb, (
        f"{available} MB available but {file_backed_mb} MB of file cache is reclaimable — the reading subtracts it")
    # and not the anonymous pages the compressor would have to take back
    assert available <= (21203 + 5893 + 924473) * PAGE // MB


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS reader; Linux reads MemAvailable, the kernel's own estimate")
def test_a_large_file_read_just_before_the_plan_does_not_lower_the_reading(tmp_path):
    """The orpheus case on this machine: a 2 GB file written and read back (its pages now file-backed and active),
    then the reading the plan takes. It must not fall by the file's size."""
    size = 2 * 1024 * MB
    anon0 = _anonymous_mb()
    before = H.memory_state().available_mb         # the host before the copy
    f = tmp_path / "weights.bin"
    with open(f, "wb") as fh:                      # the copy: the pages are written into the cache
        chunk = os.urandom(MB)
        for _ in range(size // MB):
            fh.write(chunk)
    os.sync()
    for _ in range(2):                             # the load: read twice, the pages move to the ACTIVE file queue
        with open(f, "rb") as fh:
            while fh.read(64 * MB):
                pass
    time.sleep(1)
    after = H.memory_state().available_mb
    anon_growth = max(0, _anonymous_mb() - anon0)  # other processes' anonymous pages taken meanwhile: not the file's
    f.unlink()                                     # 2 GB: not left to pytest's three-run retention
    assert before is not None and after is not None
    assert after >= before - 0.25 * (size // MB) - anon_growth, (
        f"the reading fell from {before} to {after} MB after reading a {size // MB} MB file "
        f"(anonymous pages grew {anon_growth} MB meanwhile): the file cache is subtracted")


def _anonymous_mb() -> int:
    """`Anonymous pages` from vm_stat — what other processes take between the two readings, discounted so the
    test judges the FILE's pages and not the host's churn (it went red once beside a 9 GB copy, 2026-09-29)."""
    import re, subprocess
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = re.search(r"page size of (\d+) bytes", text); anon = re.search(r"Anonymous pages:\s+(\d+)", text)
    return int(anon.group(1)) * int(page.group(1)) // MB if anon and page else 0
