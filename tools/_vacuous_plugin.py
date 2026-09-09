"""pytest plugin: which passing tests executed no assertion at all.

A test that passes without ever running an `assert` (or entering a `pytest.raises` block) cannot
have failed for the reason it was written. That is the shape behind every empty gate this engine
has shipped: the R33 chain that was empty, the upscaler cell that only checked dimensions, the
directories that never collected. The suite reports what it ran; this reports what it checked.

Line-level, so a helper that asserts on the test's behalf counts as an assertion of the file it
lives in, not of the test — the report names the test whose OWN file executed none.
"""
from __future__ import annotations

import ast
import json
import os
import sys

import pytest
from pathlib import Path

_ASSERTING = ("fail", "xfail", "raises", "warns", "deprecated_call", "approx")
_LINES: dict = {}
_COUNTS: dict = {}


def _assert_lines(path: str) -> set:
    if path in _LINES:
        return _LINES[path]
    lines = set()
    try:
        tree = ast.parse(Path(path).read_text(), filename=path)
    except Exception:
        _LINES[path] = lines
        return lines
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            lines.add(node.lineno)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                fn = getattr(item.context_expr, "func", None)
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name in _ASSERTING:
                    lines.add(node.lineno)
    _LINES[path] = lines
    return lines


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Trace the call phase and count the assertion lines of the test's own file that executed.

    A WRAPPER, not a plain hook: `pytest_runtest_call` is not first-result, so an implementation
    that called `item.runtest()` itself would run every test twice — which is its own way of
    making a suite lie about what it did.
    """
    path = str(item.fspath)
    wanted = _assert_lines(path)
    hits = [0]

    def trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == path and frame.f_lineno in wanted:
            hits[0] += 1
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        yield
    finally:
        sys.settrace(previous)
        _COUNTS[item.nodeid] = hits[0]


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("NBX_VACUOUS_REPORT")
    if out:
        Path(out).write_text(json.dumps(_COUNTS, indent=1))
