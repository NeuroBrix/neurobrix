"""What would have made it fit.

Addition 4 of `docs/reference/adaptive-memory-a-runtime-controller.md`: a refusal
that only says "out of memory" costs the reader the whole diagnosis. The allocator
knows what it asked for and what the driver had; between those two numbers there
is a concrete answer — how many pieces this work would have to be cut into, and
how much larger a card would have taken it whole.

This module says only what the numbers support. It does not guess at a remedy it
cannot compute: with no shape it talks about bands and card size, and it mentions
a spatial extent only when it is given the shape to derive one from.

It lives under `kernels/` because both execution modes raise the same
`DeviceOOMError` and both need the same sentence — and because `kernels/` is
already free of torch, so the triton branch can import it without an R33
question.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

_MB = 1024 * 1024


def _mb(n: int) -> str:
    return f"{n / _MB:,.0f} MB"


def what_would_have_fit(oom, shape: Optional[Sequence] = None) -> str:
    """One sentence naming what would have made the refused allocation fit.

    `oom` is a `DeviceOOMError` carrying `requested` and `driver_free`. Returns
    an empty string when it carries neither — a sentence built on a figure the
    driver never gave would be a guess wearing the clothes of a measurement.

    `shape` is the allocation's shape when the caller knows it. Its leading
    dimension is the one a band split cuts, so it is the only one this reports
    on; nothing here assumes NCHW or any other layout beyond "the first axis is
    divisible", which is what band streaming actually needs.
    """
    requested = getattr(oom, "requested", None)
    free = getattr(oom, "driver_free", None)
    if not requested or free is None:
        return ""

    parts = []
    shortfall = getattr(oom, "shortfall", None)
    if shortfall:
        parts.append(f"short by {_mb(shortfall)}")

    if free > 0:
        bands = math.ceil(requested / free)
        # `free` is what the driver has RIGHT NOW, with this run's own live set
        # already on the card, so a band of `requested / bands` is a size that
        # fits beside what is already there — not a hypothetical on an empty card.
        parts.append(
            f"it would fit in {bands} bands of about {_mb(requested / bands)}")
        if shape:
            lead = shape[0] if len(shape) else None
            if isinstance(lead, int) and lead >= bands > 1:
                parts.append(
                    f"cutting the leading axis of {tuple(shape)} from {lead} to "
                    f"{lead // bands} per band")
    else:
        parts.append("the card reports nothing free: no band split can help here")

    total = getattr(oom, "driver_total", None)
    if total and requested > total:
        parts.append(
            f"and no band split alone reaches it — the request of {_mb(requested)} "
            f"exceeds the card's entire {_mb(total)}")

    return "; ".join(parts)


def annotate(message: str, cause, shape: Optional[Sequence] = None) -> str:
    """Append the advice to an op-failure message when the cause is an OOM.

    Any other cause passes through untouched: this must not turn an unrelated
    failure into one that reads as a memory problem.
    """
    try:
        from neurobrix.kernels.nbx_tensor import DeviceOOMError
    except Exception:          # pragma: no cover - import cycle safety
        return message
    if not isinstance(cause, DeviceOOMError):
        return message
    advice = what_would_have_fit(cause, shape)
    return f"{message} | what would have fit: {advice}" if advice else message
