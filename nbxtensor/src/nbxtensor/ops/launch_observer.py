"""An optional observer of kernel launches, and a dry-run mode behind it.

An application may install ONE observer (`set_launch_observer`). While its `active()` answers True the library runs dry:
an autotuned kernel launches nothing — its wrapper forms the launch's key as always, hands it to `record(tuned, key)`,
and calls `shadow_run(...)` in place of the benchmark and the launch — and a count the library would read back from
the device is taken from the operation's own contract instead (`index_wrappers.nonzero_1d_wrapper`). This is how an
application enumerates the kernel keys a model forms without a device (NeuroBrix: its certification census).

With no observer installed, or one whose `active()` answers False, nothing changes: every launch runs. The observer is
read at call time, never cached, so one installed after the kernels were defined is the one consulted.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = ["set_launch_observer", "launch_observer"]


class _NoObserver:
    """No observer: never dry, records nothing."""

    @staticmethod
    def active() -> bool:
        return False

    @staticmethod
    def record(tuned, key) -> None:
        raise RuntimeError("nbxtensor: a launch was recorded with no launch observer installed — record() is reached "
                           "only during a dry run")

    @staticmethod
    def shadow_run(*args, **kwargs):
        raise RuntimeError("nbxtensor: a dry run was asked with no launch observer installed")


_OBSERVER: Any = _NoObserver


def set_launch_observer(observer: Optional[Any]) -> None:
    """Install `observer` (an object with `active()`, `record(tuned, key)` and `shadow_run(*args, **kwargs)`), or remove
    it (None). Installing one with a missing method is refused by name."""
    global _OBSERVER
    if observer is None:
        _OBSERVER = _NoObserver
        return
    missing = [m for m in ("active", "record", "shadow_run") if not callable(getattr(observer, m, None))]
    if missing:
        raise TypeError(f"set_launch_observer: the observer has no callable {', '.join(missing)}")
    _OBSERVER = observer


def launch_observer():
    """The observer in force (the no-observer when none is installed)."""
    return _OBSERVER
