"""The per-token stop seam — read by both engines' decode loops (Studio request 3).

A consumer that runs one generation on a thread (the serving daemon's worker)
binds a plain ``threading.Event`` to that thread for the duration of the
request. Every autoregressive decode loop — the compiled generator, the triton
generator, the vlm, audio_llm, encoder_decoder and tts_llm loops of both
engines — asks ``requested()`` once per token, after the token of the step is
sampled and recorded (where a loop emits ``decode_progress``, at that site). When the event is set, the loop ends
as if its length bound were reached: the flow finishes, its outputs carry the
tokens produced so far, and nothing is aborted mid-step (a step is the unit).

The thread records the step count at which the stop was first honoured
(``honoured_at()``), so the consumer can tell a generation the stop actually
ended from one that finished on its own before reading the flag.

Doctrine notes:
  - stdlib-only (``threading``): R33-safe (zero torch), importable from core/
    and triton/ alike; it carries a flag, never compute, so it bridges nothing
    between the two engines (they share no compute code).
  - Thread-local, like ``decode_progress``'s listener: an event bound by one
    request thread is invisible to every other thread. Default state is
    "nothing bound": the read is a getattr and a None test.
  - The event is SET from another thread (the daemon's control path); setting
    a ``threading.Event`` is thread-safe and the loop only reads it.
"""

import threading
from typing import Optional

_local = threading.local()


def bind(event: threading.Event) -> None:
    """Bind the stop event of the generation this thread is about to run."""
    _local.event = event
    _local.honoured_at = None


def unbind() -> None:
    """Remove this thread's stop event and its record (always pair with
    ``bind``; read ``honoured_at()`` before)."""
    _local.event = None
    _local.honoured_at = None


def requested(n_generated: int) -> bool:
    """True when this thread's generation has been asked to stop.

    Called by a decode loop once per token, with the number of tokens produced
    so far (the token of this step included). The first True records that
    count as the point where the stop was honoured."""
    event: Optional[threading.Event] = getattr(_local, "event", None)
    if event is None or not event.is_set():
        return False
    if getattr(_local, "honoured_at", None) is None:
        _local.honoured_at = int(n_generated)
    return True


def honoured_at() -> Optional[int]:
    """The token count at which this thread's decode loop honoured the stop,
    or None when no loop read a set flag since the last ``bind``."""
    return getattr(_local, "honoured_at", None)
