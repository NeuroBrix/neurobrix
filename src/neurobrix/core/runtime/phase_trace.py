"""Per-request phase timestamps — the serve TTFT reconciliation instrument.

`NBX_PHASE_TRACE=1` stamps one stderr line per phase boundary of a request
(server receive, engine execute, session creation, tokenization, prefill,
first token, engine end, server send) with the wall-clock epoch, so a
serve-side time-to-first-token can be attributed phase by phase against
a GPU timeline (nsys stamps its events with the same epoch in the
session-start table). Off by default: the check is one module-level
bool, nothing is stamped, no sync is issued.

A phase mark can carry a device `sync` callable: the stamp then means
"the GPU finished this phase", not "the host finished launching it".
Only the diagnostic pays that sync — with the flag off the callable is
never invoked.

Engine-neutral by construction: no torch, no NBXTensor — the compiled
flow passes `torch.cuda.synchronize`, the triton flow passes
`DeviceAllocator.sync_device` (R33: the module itself imports neither).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional

_ENABLED = os.environ.get("NBX_PHASE_TRACE") == "1"
_T0 = time.monotonic()


def enabled() -> bool:
    return _ENABLED


def mark(name: str, sync: Optional[Callable[[], object]] = None,
         note: Optional[Callable[[], str]] = None) -> None:
    """Stamp `name` with the wall clock (after `sync`, when given). `note`
    is a callable returning a short string appended to the stamp (the
    flows pass their allocator's live/cached/free bytes so a serve-vs-CLI
    differential reads the memory baseline at each boundary); it is only
    invoked when the flag is on."""
    if not _ENABLED:
        return
    if sync is not None:
        sync()
    now = time.time()
    extra = ""
    if note is not None:
        try:
            extra = " " + note()
        except Exception as exc:  # a diagnostic never breaks the run
            extra = f" note_error={type(exc).__name__}"
    sys.stderr.write(f"[phase] {name} wall={now:.6f} t={time.monotonic() - _T0:.3f}{extra}\n")
    sys.stderr.flush()
