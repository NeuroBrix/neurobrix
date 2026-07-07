"""P-ZERO-FALLBACK-SWEEP — CFG batch-expansion port resolution.

Pins the MED finding of engine audit #2 (2026-07-05): the CFG engines'
extra-conditioning-input batching loop did a blanket
`except Exception: continue` per port. A genuinely-absent optional port
is a legitimate skip, but a resolver BUG on a port the denoiser
consumes was silently swallowed too — feeding batch-1 conditioning to a
batch-2 forward.

Fixed boundary (`_resolve_optional_port`, one copy per engine — R30
symmetric, total path separation):
  - KeyError (the documented VariableResolver.get absence signal)
    → None → the port is skipped;
  - any OTHER exception propagates.

Run: PYTHONPATH=src python -m pytest tests/unit/runtime/test_cfg_optional_port.py
"""
from __future__ import annotations

import pytest

import neurobrix.core.runtime  # noqa: F401  (pre-resolve the cfg<->runtime import cycle)
from neurobrix.core.cfg.engine import (
    _resolve_optional_port as _resolve_compiled,
)
from neurobrix.triton.cfg.engine import (
    _resolve_optional_port as _resolve_triton,
)

_ENGINES = pytest.mark.parametrize(
    "resolve_port",
    [_resolve_compiled, _resolve_triton],
    ids=["compiled", "triton"],
)


class _Resolver:
    def __init__(self, values=None, error=None):
        self._values = values or {}
        self._error = error

    def get(self, port):
        if self._error is not None:
            raise self._error
        if port not in self._values:
            raise KeyError(port)
        return self._values[port]


@_ENGINES
def test_present_port_resolves(resolve_port):
    r = _Resolver(values={"encoder.output_0": "value"})
    assert resolve_port(r, "encoder.output_0") == "value"


@_ENGINES
def test_absent_port_returns_none(resolve_port):
    r = _Resolver(values={})
    assert resolve_port(r, "encoder.output_0") is None


@_ENGINES
def test_resolver_bug_propagates(resolve_port):
    r = _Resolver(error=ValueError("resolver bug"))
    with pytest.raises(ValueError, match="resolver bug"):
        resolve_port(r, "encoder.output_0")


@_ENGINES
def test_engines_carry_separate_helpers(resolve_port):
    """Path separation: the two engines must not share the helper."""
    assert _resolve_compiled is not _resolve_triton
    assert _resolve_compiled.__module__ == "neurobrix.core.cfg.engine"
    assert _resolve_triton.__module__ == "neurobrix.triton.cfg.engine"
