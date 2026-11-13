"""Counts the `.to(<device>)` calls that name another device than the one the tensor is on.

Why it exists. `NBXTensor.to("cuda:N")` / `.to(device=...)` moves nothing: it answers "already on device" whatever
it is given (the built graphs' own `aten::to(device)` rely on that). A caller that MEANT a placement gets a tensor
left where it was, silently (found 2026-11-10; five uploads were written that way). Before that contract changes —
a refusal by name, or a real move — the agents want a measure: which callers name a device the tensor is not on,
and how often. This module is that measure and nothing else: it changes no behaviour.

Contract.
- Off unless the environment names a file (`NBX_TO_DEVICE_CENSUS=<path>`, read once at import): then `tensor.to`
  hands every device it is given to `count_device_naming`, and at process exit one JSON line is appended to the file:
  {"pid", "argv", "calls": every device-naming call seen, "other": {"<file>:<line> <where the tensor is> -> <device
  named>": count}}. `calls` proves the process counted; an empty `other` is a result.
- `count_device_naming(location, target)` counts a call when `target` names a device ("cpu", "cuda:N", an object with `.type` /
  `.index`) different from `location` ("cpu" or "cuda:N"); a bare "cuda" names no index and cannot differ.
- It refuses nothing and raises nothing: a target it cannot read is not a device naming, and is not counted.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from typing import Dict, Optional

ENV = "NBX_TO_DEVICE_CENSUS"
PATH: Optional[str] = os.environ.get(ENV) or None
calls = 0
other: Dict[str, int] = {}


def device_named(target) -> Optional[str]:
    """The device `target` names — "cpu" or "cuda:N" — or None when it names none (a bare "cuda", a dtype, anything
    else)."""
    if isinstance(target, str):
        text = target
    elif getattr(target, "type", None) in ("cpu", "cuda"):
        index = getattr(target, "index", None)
        text = "cpu" if target.type == "cpu" else ("cuda" if index is None else f"cuda:{index}")
    else:
        return None
    if text == "cpu":
        return "cpu"
    head, _, tail = text.partition(":")
    return f"cuda:{int(tail)}" if head == "cuda" and tail.isdigit() else None


def count_device_naming(location: str, target, depth: int = 1) -> None:
    """One `.to(target)` on a tensor at `location`; `depth`: the frames from this function up to the caller of
    `.to` (1: this function was called by that caller itself)."""
    global calls
    want = device_named(target)
    if want is None:
        return
    calls += 1
    if want != location:
        frame = inspect.currentframe()
        for _ in range(depth):
            frame = frame.f_back
        key = f"{frame.f_code.co_filename}:{frame.f_lineno} {location} -> {want}"
        other[key] = other.get(key, 0) + 1


def append_counts(path: str) -> None:
    """Append this process's line to `path`."""
    with open(path, "a") as f:
        f.write(json.dumps({"pid": os.getpid(), "argv": sys.argv[:6], "calls": calls, "other": other},
                           sort_keys=True) + "\n")


if PATH:
    import atexit
    atexit.register(append_counts, PATH)
