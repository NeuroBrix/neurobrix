#!/usr/bin/env python
"""Name the origin of a device address the launcher refused, from a malloc
trace (`NBX_MALLOC_TRACE=<tsv>`; row: event_id, M|F, ptr, nbytes, site).

Given the address, prints every allocation whose range covered it, in event
order, and whether that allocation was FREED before the end of the trace —
the discriminator between "never ours" (no M row covers it: a foreign or
host pointer) and "ours, then freed" (an M row covers it and its F row comes
before the refusal: a dangling tensor the door caught). Ming's embedding
weight, 2026-09-16.

Usage: address_origin.py <trace.tsv> <hex address>
"""
from __future__ import annotations

import sys


def main() -> int:
    path, addr = sys.argv[1], int(sys.argv[2], 16)
    live: dict = {}
    hits = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[1] not in ("M", "F"):
                continue
            ev, kind, ptr, nbytes = parts[0], parts[1], int(parts[2], 0), int(parts[3] or 0)
            site = parts[4] if len(parts) > 4 else ""
            if kind == "M":
                live[ptr] = (ev, nbytes, site)
                if ptr <= addr < ptr + nbytes:
                    hits.append({"alloc_event": ev, "base": ptr, "nbytes": nbytes, "site": site,
                                 "offset": addr - ptr, "freed_event": None})
            else:
                rec = live.pop(ptr, None)
                if rec is not None:
                    for h in hits:
                        if h["base"] == ptr and h["freed_event"] is None:
                            h["freed_event"] = ev
    if not hits:
        print(f"{addr:#x}: NEVER inside an allocation this process's DeviceAllocator made "
              f"(a foreign, host or arena-external pointer)")
        return 1
    for h in hits:
        state = f"FREED at event {h['freed_event']}" if h["freed_event"] else "still live at the end"
        print(f"{addr:#x}: inside block {h['base']:#x} (+{h['offset']} of {h['nbytes']} bytes), "
              f"allocated at event {h['alloc_event']} by {h['site']} — {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
