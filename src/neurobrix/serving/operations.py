"""The daemon's operation table — one record per work request (Studio requests 3-4).

Every work request (`generate`, `chat`, `complete`) is an OPERATION with an id the
client may choose (`op_id` in the request) or the daemon assigns. The table is the
one shared state between the connection threads: the work path writes the step
count and the terminal state, the control path (`status`, `cancel`, `shutdown`)
reads them and sets the stop flag. Every read-modify-write happens under the
table's own lock — never the inference lock, so control never waits on a
generation.

States: `queued` (waiting for the inference lock) -> `running` -> one of
  - `completed`  the generation ended on its own (eos or its length bound);
  - `cancelled`  a client's `cancel` stopped it: its decode loop read the flag,
                 returned, and the devices were synced;
  - `orphaned`   its client vanished; the daemon stopped it (never `cancelled`);
  - `stopped`    a daemon shutdown stopped it;
  - `failed`     the engine raised.
A terminal state is final and set exactly once, after the native work has
stopped and the device has synced — `done` is set at that moment, and only then
may a cancel be acknowledged.

stdlib-only besides the config read: the module carries state, never compute.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import yaml

#: Why a stop was requested — the first reason wins and names the terminal state.
STOP_REASONS = {"cancel": "cancelled", "orphan": "orphaned", "shutdown": "stopped"}
TERMINAL = ("completed", "cancelled", "orphaned", "stopped", "failed")

#: The keys of system.yml `serving:`, each read by name (no default in code).
_SETTINGS = {
    "cancel_ack_timeout_s": (int, float),
    "operations_retained": (int,),
    "listen_backlog": (int,),
    "orphan_poll_s": (int, float),
    "drain_timeout_s": (int, float),
}


def serving_settings() -> Dict[str, Any]:
    """The dispatcher's limits and timeouts, from `serving:` in config/system.yml
    (each value's provenance is written beside it there). A missing or
    non-positive value is refused by name — no figure lives in this code."""
    from neurobrix.core.workspace import SYSTEM_YML
    cfg = (yaml.safe_load(SYSTEM_YML.read_text()) or {}).get("serving") or {}
    out = {}
    for key, types in _SETTINGS.items():
        v = cfg.get(key)
        if isinstance(v, bool) or not isinstance(v, types) or v <= 0:
            raise RuntimeError(
                f"ZERO FALLBACK: the serving dispatcher needs `serving.{key}` in {SYSTEM_YML} "
                f"(a positive number), got {v!r}.")
        out[key] = v
    return out


class Operation:
    """One work request's record. Mutated only under the table's lock, except
    `stop_event` (a threading.Event, set from any thread, read by the decode
    loop through `neurobrix.decode_stop`) and `done` (an Event)."""

    def __init__(self, op_id: str, method: str) -> None:
        self.op_id = op_id
        self.method = method
        self.state = "queued"
        self.steps = 0
        self.stop_event = threading.Event()
        self.stop_reason: Optional[str] = None
        self.cancel_requests = 0
        self.done = threading.Event()
        self.started_at = time.time()
        self.detail: Optional[str] = None

    def record(self) -> Dict[str, Any]:
        rec = {"op_id": self.op_id, "method": self.method, "state": self.state,
               "steps": self.steps}
        if self.stop_reason is not None:
            rec["stop_reason"] = self.stop_reason
        if self.detail is not None:
            rec["detail"] = self.detail
        return rec


class OperationTable:
    """The shared state of the dispatcher, under its own lock."""

    def __init__(self, retained: int) -> None:
        self._lock = threading.Lock()
        self._ops: "OrderedDict[str, Operation]" = OrderedDict()
        self._retained = int(retained)
        self._seq = itertools.count(1)
        self._prefix = f"{os.getpid()}-"

    # -- the work path ----------------------------------------------------
    def open(self, method: str, requested_id: Optional[str]) -> Operation:
        """Register a new operation. A client-chosen id that names a LIVE
        operation is refused by name (two requests cannot share a stop flag);
        one that names a finished operation replaces its record."""
        with self._lock:
            if requested_id is not None:
                if not isinstance(requested_id, str) or not requested_id:
                    raise ValueError(f"op_id must be a non-empty string, got {requested_id!r}")
                old = self._ops.get(requested_id)
                if old is not None and old.state not in TERMINAL:
                    raise ValueError(f"op_id {requested_id!r} names a live operation ({old.state})")
                op_id = requested_id
                self._ops.pop(op_id, None)
            else:
                op_id = f"{self._prefix}{next(self._seq)}"
            op = Operation(op_id, method)
            self._ops[op_id] = op
            self._evict_locked()
            return op

    def start(self, op: Operation) -> bool:
        """The work thread holds the inference lock: mark the op running.
        False when a stop was requested while it was queued — the op is then
        finished here, no native work having started."""
        with self._lock:
            if op.stop_reason is not None:
                self._finish_locked(op, STOP_REASONS[op.stop_reason], None)
                return False
            op.state = "running"
            return True

    def step(self, op: Operation, n_generated: int) -> None:
        with self._lock:
            op.steps = int(n_generated)

    def finish(self, op: Operation, honoured_at: Optional[int], error: Optional[str]) -> str:
        """Called AFTER the generation returned (or raised) and the devices
        were synced. `honoured_at` is the decode loop's own record that it read
        the stop flag (decode_stop); a stop flag set after the last token was
        read never turns a completed generation into a cancelled one."""
        with self._lock:
            if honoured_at is not None and op.stop_reason is not None:
                op.steps = int(honoured_at)
                state = STOP_REASONS[op.stop_reason]
            elif error is not None:
                state = "failed"
            else:
                state = "completed"
            self._finish_locked(op, state, error)
            return state

    def _finish_locked(self, op: Operation, state: str, detail: Optional[str]) -> None:
        op.state = state
        op.detail = detail
        op.done.set()   # the one moment a cancel may be acknowledged
        self._evict_locked()

    def _evict_locked(self) -> None:
        while len(self._ops) > self._retained:
            for key, old in self._ops.items():
                if old.state in TERMINAL:
                    del self._ops[key]
                    break
            else:
                return   # every retained op is live: nothing may be forgotten

    # -- the control path -------------------------------------------------
    def request_stop(self, op_id: str, reason: str):
        """Set an op's stop flag. Returns (op, prior) where prior names what
        was already true: "unknown", "finished", "already_<reason>",
        "stopping:<other reason>", or None for the first request."""
        with self._lock:
            op = self._ops.get(op_id)
            if op is None:
                return None, "unknown"
            if reason == "cancel":
                op.cancel_requests += 1
            if op.state in TERMINAL:
                return op, "finished"
            if op.stop_reason is not None:
                prior = ("already_" + reason) if op.stop_reason == reason else f"stopping:{op.stop_reason}"
                return op, prior
            op.stop_reason = reason
            op.stop_event.set()
            return op, None

    def stop_all(self, reason: str) -> List[Operation]:
        """Stop every live operation (daemon shutdown)."""
        with self._lock:
            live = [op for op in self._ops.values() if op.state not in TERMINAL]
            for op in live:
                if op.stop_reason is None:
                    op.stop_reason = reason
                    op.stop_event.set()
            return live

    def snapshot(self, op: Operation) -> Dict[str, Any]:
        with self._lock:
            return op.record()

    def live(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [op.record() for op in self._ops.values() if op.state not in TERMINAL]
