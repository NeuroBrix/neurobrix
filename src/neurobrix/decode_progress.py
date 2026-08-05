"""In-process per-token decode progress sink — observability, both engines.

Grows the NBX_DECODE_PROGRESS file diagnostic (src/neurobrix/CLAUDE.md §8)
into a first-class in-process channel: a consumer (the serving daemon's
streaming RPC) registers a listener for the duration of one request, and
the autoregressive generators emit one event per sampled token at the same
site as the file diagnostic. This is what makes real TTFT measurable over
the daemon RPC instead of estimated from wall-clock.

Doctrine notes:
  - Observability only, never compute — this module does NOT bridge the
    compiled and triton compute paths (they share the NBX container, the
    Prism plan, the flow contract, and observability surfaces; never
    compute code).
  - stdlib-only: R33-safe (zero torch) and R34-safe (zero vendor import),
    importable from core/ and triton/ alike.
  - Listener exceptions PROPAGATE (no silent failure): a broken consumer
    (e.g. client disconnected mid-stream) aborts the request cleanly at
    the daemon boundary instead of decoding into the void.
  - Thread-local: a listener registered by one request thread is invisible
    to any other thread. Default state is "no listener" — the emit site is
    a getattr + None-check, zero hot-path cost when unused.
"""

import threading
from typing import Callable, Optional

# Listener signature: fn(step_idx, n_generated, token_id, is_done) -> None
TokenListener = Callable[[int, int, int, bool], None]

_local = threading.local()


def set_listener(fn: TokenListener) -> None:
    """Register the per-token listener for the current thread."""
    _local.listener = fn


def clear_listener() -> None:
    """Remove the current thread's listener (always pair with set_listener)."""
    _local.listener = None


def emit(step_idx: int, n_generated: int, token_id: int, is_done: bool) -> None:
    """Emit one per-token event to the current thread's listener, if any."""
    fn: Optional[TokenListener] = getattr(_local, "listener", None)
    if fn is not None:
        fn(step_idx, n_generated, token_id, is_done)
