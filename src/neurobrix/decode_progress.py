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

import os
import threading
import time
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


def record(step_idx: int, n_generated: int, token_id: int, is_done: bool) -> None:
    """One per-token decode event on BOTH channels: the buffer-immune
    `NBX_DECODE_PROGRESS` file (src/neurobrix/CLAUDE.md section 8) and the
    in-process listener above.

    A flow that keeps its own decode loop calls this instead of writing the line
    itself, so every flow's trajectory has one shape and one site. The
    autoregressive and encoder_decoder flows emit their trajectory; the audio_llm
    flow kept its own loop and emitted none, so a decode rate could not be measured
    on any audio row at all (2026-09-08) — the harness saw a run that succeeded and
    left no trajectory, and reported no rate.

    Observability only: no numerical effect, default-off for the file, and a getattr
    plus a None test when nobody listens.
    """
    path = os.environ.get("NBX_DECODE_PROGRESS")
    if path:
        with open(path, "a") as pf:
            pf.write(f"t={time.time():.3f} step={step_idx} "
                     f"n={n_generated} last={token_id} done={is_done}\n")
    emit(step_idx, n_generated, token_id, is_done)
