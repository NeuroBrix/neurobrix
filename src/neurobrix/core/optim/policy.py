"""Optimization policy — the doctrine piece of the OptimizationEngine.

Every pass class declares, before any transformation exists:
- its gate class: EXACT passes are byte-preserving by construction and
  may default on; FLOATING passes change floating-point results within
  a drift budget and are opt-in, always behind the drift gate;
- its provenance: the module where the pass lives (or will live), so
  every reported finding and every applied transformation traces back
  to one owning file;
- its version: bumped on any behavior change — part of the
  optimization-cache invalidation key (graph fingerprint + pass
  versions), so a stale cached plan can never survive a pass upgrade.

The registry is the single source of truth consumed by the analyzer
(to label findings), the report (to print gate columns), and — from
Phase 2 on — the pass driver (to decide what runs at each level).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateClass(Enum):
    """How a pass must be proven before it lands or runs."""

    # Byte-preserving by construction; gate = byte-identical full-zoo
    # optimizer-ON vs optimizer-OFF, on both engines. May default on.
    EXACT = "exact"

    # Changes floating-point results (reassociation, cancellation).
    # Gate = the drift-gate policy vs the unoptimized oracle. Opt-in
    # only, never enabled by a default level.
    FLOATING = "floating"


@dataclass(frozen=True)
class PassPolicy:
    name: str
    gate_class: GateClass
    provenance: str  # owning module (relative to neurobrix.core.optim)
    version: int  # bump on ANY behavior change (cache invalidation)
    default_on: bool  # eligible for the default "exact" level
    summary: str


# One entry per pass class; one file per pass class (the architecture
# law). Phase 1 declares them; Phase 2+ lands them one by one, each
# behind its declared gate.
PASS_REGISTRY: dict[str, PassPolicy] = {
    p.name: p
    for p in (
        PassPolicy(
            name="algebraic",
            gate_class=GateClass.EXACT,
            provenance="passes/algebraic.py",
            version=1,
            default_on=True,
            summary=(
                "Exact identities (x*1, x+0, x-0, x/1, pow 1, "
                "transpose-of-transpose, same-shape view/expand, "
                "full-range slice, same-dtype cast) and exact "
                "cancelling motifs (+c..-c, *v../v) on integer and "
                "shape algebra."
            ),
        ),
        PassPolicy(
            name="cancellation",
            gate_class=GateClass.FLOATING,
            provenance="passes/cancellation.py",
            version=1,
            default_on=False,
            summary=(
                "Cancelling motifs on floating-point values — "
                "mathematically exact, numerically reassociating; "
                "separated from `algebraic` because the gate differs."
            ),
        ),
        PassPolicy(
            name="const_fold",
            gate_class=GateClass.EXACT,
            provenance="passes/const_fold.py",
            version=1,
            default_on=True,
            summary=(
                "Values known in advance: ops whose inputs are all "
                "parameters/constants (e.g. weight transposes) folded "
                "once at load instead of every forward."
            ),
        ),
        PassPolicy(
            name="cse",
            gate_class=GateClass.EXACT,
            provenance="passes/cse.py",
            version=1,
            default_on=True,
            summary=(
                "Common-subexpression elimination: identical op + "
                "identical inputs + identical attributes computed more "
                "than once."
            ),
        ),
        PassPolicy(
            name="dead_code",
            gate_class=GateClass.EXACT,
            provenance="passes/dead_code.py",
            version=1,
            default_on=True,
            summary="Ops whose outputs never reach a graph output.",
        ),
        PassPolicy(
            name="layout",
            gate_class=GateClass.EXACT,
            provenance="passes/layout.py",
            version=1,
            default_on=True,
            summary=(
                "Transpose/permute sinking and elimination; adjacent "
                "composition chains."
            ),
        ),
        PassPolicy(
            name="fusion_vertical",
            gate_class=GateClass.EXACT,
            provenance="fusion/matcher.py",
            version=1,
            default_on=False,  # lands with its measured benchmark delta
            summary=(
                "Vertical fusion: matmul-class anchor + single-consumer "
                "elementwise/activation epilogue chain, lowered to fused "
                "Triton templates on the Triton branch."
            ),
        ),
        PassPolicy(
            name="fusion_horizontal",
            gate_class=GateClass.EXACT,
            provenance="fusion/matcher.py",
            version=1,
            default_on=False,  # lands with its measured benchmark delta
            summary=(
                "Horizontal fusion: same-op same-shape siblings sharing "
                "an input (e.g. q/k/v projections) batched into one "
                "launch."
            ),
        ),
        PassPolicy(
            name="replay",
            gate_class=GateClass.EXACT,
            provenance="replay.py",
            version=1,
            default_on=False,  # lands with its measured benchmark delta
            summary=(
                "Frozen execution plan per shape bucket — kills the "
                "per-op launch tax; CUDA-Graph capture evaluated on the "
                "compiled branch."
            ),
        ),
    )
}


# --optim levels (engraved in the Phase 0 scoping doc):
#   off    — the optimizer never touches the load path.
#   report — analysis only, artifact written, zero transformation.
#   exact  — every EXACT pass with default_on=True.
#   full   — "exact" + explicitly opted-in FLOATING passes (each named
#            on the CLI; the level alone never enables them).
OPTIM_LEVELS: dict[str, tuple[str, ...]] = {
    "off": (),
    "report": (),
    "exact": tuple(
        p.name
        for p in PASS_REGISTRY.values()
        if p.gate_class is GateClass.EXACT and p.default_on
    ),
    "full": tuple(
        p.name
        for p in PASS_REGISTRY.values()
        if p.gate_class is GateClass.EXACT and p.default_on
    ),  # FLOATING passes join only by explicit opt-in flags
}
