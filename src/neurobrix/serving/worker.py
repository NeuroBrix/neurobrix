"""The daemon's one inference thread.

Every engine call of a daemon — the load, every work request from either door (the Unix socket and the HTTP
door), the device sync that acknowledges a stop, `new_chat`, the unload — runs on ONE long-lived thread, in the
order it was submitted. The connection threads only parse, submit, wait, and send back what the work produced.

Why one thread and not "whichever connection thread holds the lock": a device runtime may keep per-thread state
that a thread hop leaks. On Metal a Python thread has no autorelease pool of its own, so what the driver autoreleases
outside a pool is never released (nbxtensor/src/nbxtensor/backend/triton_ext_driver.py, the note before its
`objc.autorelease_pool()`: arenas kept for good). One long-lived thread makes the engine's view of threads the same
as `neurobrix run`'s: one.

The order is FIFO: a job submitted while another runs waits behind it (the operation's state is `queued`). The
worker carries no policy (no bound, no priority): the dispatcher decides what may be submitted.

stdlib only: it carries jobs, never compute.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Optional

_log = logging.getLogger(__name__)


class Job:
    """One submitted call: its result or its exception, and the moment it ended."""

    def __init__(self, fn: Callable[[], Any], name: str) -> None:
        self.fn = fn
        self.name = name
        self.done = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self.done.wait(timeout)

    def outcome(self) -> Any:
        """The call's return value; its exception re-raised in the caller's thread."""
        if not self.done.is_set():
            raise RuntimeError(f"job {self.name} has not ended")
        if self.error is not None:
            raise self.error
        return self.result


_STOP = object()


class InferenceWorker:
    """One thread, one FIFO of jobs."""

    def __init__(self, name: str = "nbx-inference") -> None:
        self._name = name
        self._jobs: "queue.Queue[Any]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._pending = 0
        self._pending_lock = threading.Lock()

    @property
    def thread(self) -> Optional[threading.Thread]:
        return self._thread

    def start(self) -> None:
        """Start the thread once; a second call is a no-op."""
        with self._start_lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
            self._thread.start()

    def submit(self, fn: Callable[[], Any], name: str = "job") -> Job:
        """Queue `fn`; it runs on the worker thread after every job queued before it."""
        self.start()
        job = Job(fn, name)
        with self._pending_lock:
            self._pending += 1
            self._idle.clear()
        self._jobs.put(job)
        return job

    def call(self, fn: Callable[[], Any], name: str = "job") -> Any:
        """Run `fn` on the worker and return its result (its exception re-raised here). Called FROM the
        worker thread itself, it runs inline: a job never waits on a job queued behind it."""
        if threading.current_thread() is self._thread:
            return fn()
        job = self.submit(fn, name)
        job.wait()
        return job.outcome()

    def wait_idle(self, timeout: float) -> bool:
        """True once no job is queued or running, within `timeout` seconds."""
        return self._idle.wait(timeout)

    def stop(self) -> None:
        """End the thread after the jobs already queued."""
        if self._thread is not None:
            self._jobs.put(_STOP)

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is _STOP:
                return
            try:
                job.result = job.fn()
            except BaseException as e:   # handed to the caller, which decides; never swallowed here
                job.error = e
            finally:
                job.done.set()
                with self._pending_lock:
                    self._pending -= 1
                    if self._pending == 0:
                        self._idle.set()
