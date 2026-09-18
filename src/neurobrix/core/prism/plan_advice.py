"""What would have made the plan fit — computed, not named in words.

`_fail_error` ends with a list that includes *"3. A smaller input (resolution,
batch, context)"*. That sentence is right and it is not an answer: it names the
remedy and does not take it, and it does not say how much smaller.

The case that engraved this is the Mac's, measured 2026-09-18 and handed over:
`real-esrgan-x8` at 1024x1024 — the reproducer of
`docs/reference/adaptive-memory-a-runtime-controller.md` — refused at PLAN TIME
in one second, before any allocation, **17237 MB planned against 12598 MB
available, 16.4 GB of it activations against 32 MB of weights**. A controller that
waits for `DeviceOOMError` never fires there: nothing is ever allocated. And the
remedy the refusal declines to take was then proven by hand on that machine: four
tiles, 194 s, an 8192x8192 artefact judged, seam at +2.07 sigma and not
distinguishable from its surroundings.

So the refusal has to carry the same thing the allocator's now does
(`kernels/oom_advice.py`): the number of pieces the work must be cut into, and
whether any cut can reach it at all.

**One measured constraint shapes every figure here.** Prism's reading of available
device memory moved between **12598 MB and 15248 MB minutes apart on an idle
machine** — a 21 % swing. A band count divided straight out of that reading moves
by 20 % with it, so a plan that is exactly critical at one reading is short at the
next. Every budget below is therefore taken against the reading REDUCED by that
volatility, and the figure is stated so a reader can see the margin rather than
discover it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

#: The measured swing in Prism's own reading of available memory on an idle
#: machine (12598 -> 15248 MB, 2026-09-18, Apple). Budgets are taken against the
#: reading reduced by this, so a band count does not sit on a knife edge.
READING_VOLATILITY = 0.21

_MB = 1024 * 1024


@dataclass(frozen=True)
class TilingVerdict:
    """Whether the plan's overflow can be reshaped, and into how many pieces."""

    component: str
    #: Bytes the component's weights need. They are NOT reshapable: every band
    #: reads all of them.
    weight_bytes: int
    activation_bytes: int
    overhead_bytes: int
    #: The device budget actually used, after the volatility margin.
    budget_bytes: int
    #: None when no band count can fit — see `refusal_reason`.
    bands: Optional[int]
    refusal_reason: Optional[str] = None

    @property
    def reshapable(self) -> bool:
        return self.bands is not None and self.bands > 1

    @property
    def per_band_bytes(self) -> Optional[int]:
        if not self.bands:
            return None
        return self.weight_bytes + self.overhead_bytes + self.activation_bytes // self.bands


def _mb(n: float) -> str:
    return f"{n / _MB:,.0f} MB"


def assess(component: str, weight_bytes: int, activation_bytes: int,
           overhead_bytes: int, available_bytes: int,
           volatility: float = READING_VOLATILITY) -> TilingVerdict:
    """Can this component's overflow be cut into bands, and into how many?

    The arithmetic is deliberately the simplest thing that is true: weights and
    overhead are paid by EVERY band, activations divide across them. So the
    smallest band still costs `weights + overhead`, and if that alone exceeds the
    budget no number of bands reaches it — which is the honest refusal, and a
    different sentence from "your input is too large".
    """
    budget = int(available_bytes * (1.0 - volatility))
    fixed = weight_bytes + overhead_bytes

    if fixed >= budget:
        return TilingVerdict(
            component, weight_bytes, activation_bytes, overhead_bytes, budget, None,
            refusal_reason=(
                f"its weights and overhead alone are {_mb(fixed)} against a "
                f"{_mb(budget)} budget, and every band pays those in full — no "
                f"number of bands reaches it"))

    room_for_activations = budget - fixed
    if activation_bytes <= room_for_activations:
        # Nothing to reshape: this component fits. The refusal came from
        # elsewhere (another component, or the sum across a placement).
        return TilingVerdict(
            component, weight_bytes, activation_bytes, overhead_bytes, budget, 1)

    # A MINIMUM, and said so where it is produced. The arithmetic divides
    # activations exactly by the band count, which no real tiling does: bands
    # overlap by a halo, and the per-band peak is therefore somewhat above
    # `activations / bands`. The Mac's hand-run of this very case took FOUR tiles
    # where this returns two — both are right about different things, and the
    # difference is the halo plus the margin a human left. So this figure is the
    # floor below which no band count can work, not a recommendation; the rung
    # that executes it sizes its own halo and may need more.
    bands = math.ceil(activation_bytes / room_for_activations)
    return TilingVerdict(
        component, weight_bytes, activation_bytes, overhead_bytes, budget, bands)


def dominated_by_activations(weight_bytes: int, activation_bytes: int,
                             ratio: float = 4.0) -> bool:
    """True when activations are what overflows, so tiling is the right rung.

    A model whose WEIGHTS do not fit needs a different rung entirely — sharding,
    offload, a bigger card — and tiling its activations would be answering a
    question nobody asked. The Mac's reproducer is 16.4 GB of activations against
    32 MB of weights, a ratio of about 500; the default of 4 is far below that and
    is meant to separate classes, not to be tuned.
    """
    return activation_bytes > ratio * max(1, weight_bytes)


def side_scale_for(bands: int) -> float:
    """The factor each SPATIAL SIDE would shrink by to cut activations `bands` times.

    Stated with its assumption, because it only holds where it holds: a
    convolutional upscaler's activations scale with the spatial area, so dividing
    them by `bands` divides each side by `sqrt(bands)`. For a sequence model the
    same division is linear in the context, not square-root, and this figure must
    not be offered there.
    """
    return 1.0 / math.sqrt(max(1, bands))


def what_would_have_fit(verdicts: Sequence[TilingVerdict],
                        spatial: bool = False) -> List[str]:
    """The lines to put in a plan-stage refusal, one per binding component.

    Returns an empty list when there is nothing concrete to say — silence beats a
    sentence built on a figure that was not computed.
    """
    lines: List[str] = []
    for v in verdicts:
        if v.refusal_reason:
            lines.append(f"  {v.component}: {v.refusal_reason}")
            continue
        if not v.reshapable:
            continue
        piece = v.per_band_bytes or 0
        line = (f"  {v.component}: {_mb(v.activation_bytes)} of activations against "
                f"{_mb(v.weight_bytes)} of weights — it fits in {v.bands} tiles of "
                f"about {_mb(piece)} each, inside a {_mb(v.budget_bytes)} budget "
                f"(a floor: a real tiling's halo costs more, and the same case took "
                f"four tiles by hand on Apple where this arithmetic says two)")
        if spatial and v.bands:
            f = side_scale_for(v.bands)
            line += (f"; the same reduction from the input side is each spatial "
                     f"dimension at {f:.2f}x")
        lines.append(line)
    return lines
