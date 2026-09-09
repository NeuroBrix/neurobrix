"""What a harness metric may assume about the payload its brick returned.

A metric that reads a key its brick never emits does not fail — it takes the
default and computes a verdict on it. `m_psnr_db` read `d.get("psnr", 0.0)`
against a brick emitting `psnr_db`, so every image comparison scored 0.0 dB and
no image cell could EVER report AGREES. It then reported DIVERGES on two renders
a human eyeballed as correct.

The static half of the rule is `tools/harness_metric_key_audit.py`, which reads
every campaign harness against what its bricks actually emit. This is the
runtime half: a missing key must be LOUD at the moment of the read, because a
verdict computed on a default is worse than no verdict — it aims the next lever.

    value = require_key(d, "psnr_db", produced_by="tools/image_fidelity.py")

Use it wherever a harness reads a brick's payload to decide something. Do NOT
use `.get(key, default)` there: a default is only honest for a field that is
genuinely optional, and a metric's input never is.
"""
from __future__ import annotations

from typing import Any, Mapping


class HarnessContractError(RuntimeError):
    """A brick did not emit what its consumer was written against."""


def require_key(payload: Mapping[str, Any], key: str, produced_by: str = "") -> Any:
    """The value at `key`, or a loud failure naming what the brick did emit.

    Never returns a default. A metric that cannot read its input has no verdict
    to give, and saying so is the whole point.
    """
    if not isinstance(payload, Mapping):
        raise HarnessContractError(
            f"expected a mapping from {produced_by or 'the brick'}, got "
            f"{type(payload).__name__}: {str(payload)[:200]}")
    if key not in payload:
        raise HarnessContractError(
            f"{produced_by or 'the brick'} emitted no {key!r}. It emitted "
            f"{sorted(payload)}. A metric reading an absent key returns a verdict "
            f"about nothing — fix the key, or fix the brick, but do not default it."
        )
    return payload[key]


def require_keys(payload: Mapping[str, Any], *keys: str, produced_by: str = "") -> tuple:
    """Several at once, reported together rather than one failure at a time."""
    if not isinstance(payload, Mapping):
        raise HarnessContractError(
            f"expected a mapping from {produced_by or 'the brick'}, got "
            f"{type(payload).__name__}: {str(payload)[:200]}")
    missing = [k for k in keys if k not in payload]
    if missing:
        raise HarnessContractError(
            f"{produced_by or 'the brick'} emitted none of {missing}. It emitted "
            f"{sorted(payload)}.")
    return tuple(payload[k] for k in keys)
