"""The compiled branch's release fence on MPS (state engine step 2, docs/design/state-engine.md §4.3).

MPS reuses a buffer only once the command buffer that used it completed (§3.1), so a released state page waits in the
release queue (core/state/pages.py) until the work enqueued before its release has run. Each `record` places a
`torch.mps.Event` on the current stream and numbers it; `completed` queries the events in order without waiting and
returns the highest number reached (events on one stream complete in order).
"""
from __future__ import annotations

from collections import deque


class MpsStateFence:
    def __init__(self, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self._torch = torch_module
        self._events = deque()
        self._recorded = 0
        self._completed = 0

    def record(self) -> int:
        event = self._torch.mps.Event(enable_timing=False)
        event.record()
        self._recorded += 1
        self._events.append((self._recorded, event))
        return self._recorded

    def completed(self) -> int:
        while self._events and self._events[0][1].query():
            self._completed = self._events.popleft()[0]
        return self._completed

    def drain(self, pool) -> int:
        return pool.fence_completed(self.completed())
