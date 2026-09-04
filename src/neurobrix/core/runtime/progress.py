"""Periodic progress for long runs — so a healthy run is distinguishable from a hung one.

A 100-step Allegro render at 720p takes about fourteen hours on a V100, and
until 2026-09-04 it printed **nothing at all** for the whole of it. There is no
way to tell that from a hang, and the operator cannot know whether to wait or
to kill it. The engine has already paid for this lesson elsewhere: a timeout is
not a diagnosis, and neither is silence.

It is a product defect before it is a tooling one. Anyone rendering video with
this engine sees the same blank terminal.

Design, and why each part is the way it is:

* **Time-based, not step-based.** A 20-step image render finishes in seconds
  and must stay silent; a 100-step video render must speak. One rule covers
  both: emit at most once per `interval`, so a run shorter than the interval
  prints nothing and a long one prints a steady heartbeat.
* **stderr, flushed.** stdout carries the model's OUTPUT and is parsed by the
  harness (`_parse_llm_text`) — progress on stdout would corrupt what callers
  read. Flushed every line because a block-buffered pipe discards the buffer
  when a run is killed, which is exactly when the last line matters most; the
  regression harness learned that in 2026-08 with `Partial stdout: b''`.
* **Torch-free.** The triton engine needs the same heartbeat and may not import
  torch (R33), so this is plain Python and both modes share ONE implementation
  rather than drifting apart (R30).
"""

from __future__ import annotations

import os
import sys
import time

# How long a run may stay silent. Not a hardware parameter — it is a human
# attention span, and the point is that a person watching a terminal learns
# within one interval whether anything is happening.
_DEFAULT_INTERVAL_S = 30.0
_ENV_INTERVAL = "NBX_PROGRESS_EVERY"
_ENV_DISABLE = "NBX_NO_PROGRESS"


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


class StepProgress:
    """Heartbeat for a bounded loop: step, elapsed, and a projected remainder.

    The projection is deliberately a simple mean over completed steps. A
    diffusion step is near-constant in cost, so a mean is honest here; anything
    cleverer would imply a confidence the number does not have.
    """

    def __init__(self, total: int, label: str = "render",
                 interval_s: float | None = None, stream=None):
        self.total = max(0, int(total or 0))
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        if interval_s is None:
            try:
                interval_s = float(os.environ.get(_ENV_INTERVAL, "")
                                   or _DEFAULT_INTERVAL_S)
            except ValueError:
                interval_s = _DEFAULT_INTERVAL_S
        self.interval = max(0.0, interval_s)
        self.disabled = os.environ.get(_ENV_DISABLE) == "1"
        self.start = time.monotonic()
        self._last_emit = self.start
        self.emitted = 0

    def step(self, index: int) -> None:
        """Call once per completed step, 0-based. Emits only when due."""
        if self.disabled or not self.total:
            return
        now = time.monotonic()
        if now - self._last_emit < self.interval:
            return
        self._last_emit = now
        done = index + 1
        elapsed = now - self.start
        line = (f"[progress] {self.label} step {done}/{self.total} · "
                f"{_fmt(elapsed)} elapsed")
        if done:
            remaining = (elapsed / done) * (self.total - done)
            if remaining > 0:
                line += f" · ~{_fmt(remaining)} remaining"
        self._write(line)

    def done(self) -> None:
        """Final line — only if this run ever spoke, so short runs stay silent."""
        if self.disabled or not self.emitted:
            return
        self._write(f"[progress] {self.label} complete · "
                    f"{_fmt(time.monotonic() - self.start)} total")

    def _write(self, line: str) -> None:
        try:
            self.stream.write(line + "\n")
            self.stream.flush()
            self.emitted += 1
        except (OSError, ValueError):        # closed pipe: never fatal
            pass
