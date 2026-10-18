"""The state engine's policies, chosen by score and always stated (step 5; docs/design/state-engine.md §4.4, owner
2026-10-15 18:00 point 1).

When the full-precision, whole-context state does not fit the node, the engine considers policies, scored like Prism's
strategies on memory, speed and quality cost:
- an EXACT policy that fits always wins (the output is the same): `pages` (the paged cache, step 2), then `offload` (cold
  pages to the host: exact, slower);
- a policy that CHANGES the output (`quantize_kv`, `window`) is chosen only if no exact policy fits. Then the plan carries
  its quality cost, MEASURED against the vendor at the same request, and says so to the user. A changing policy without
  a measured cost is never chosen: it is refused by name, so no unmeasured degradation is ever silent;
- `refuse`, by name, with the figures, when nothing fits.

Pure arithmetic over the candidates the engine prices, torch-free (both engines and Prism import it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class StatePolicyRefused(RuntimeError):
    pass


#: The exact policies in the order the engine prefers them when several fit (the faster first).
EXACT_ORDER = ("pages", "offload")


@dataclass
class PolicyCandidate:
    """One policy as the engine priced it for this request on this node."""
    name: str
    device_bytes: int                       # what it holds on the node's device
    exact: bool                             # the output is the vendor's (no change)
    quality_cost: Optional[float] = None    # measured against the vendor at the same request (None = not measured)
    cost_source: str = ""                   # where that measurement is recorded
    slowdown: float = 1.0                   # measured or derived time factor against `pages` (1.0 = as fast)


@dataclass
class PolicyDecision:
    name: str
    reason: str
    quality_cost: Optional[float] = None
    cost_source: str = ""
    considered: List[str] = field(default_factory=list)


def choose_policy(candidates: List[PolicyCandidate], room_bytes: int) -> PolicyDecision:
    """The policy taken for a state of this node, and why. `room_bytes` is what the plan leaves for state on the node."""
    fits = [c for c in candidates if c.device_bytes <= room_bytes]
    considered = [f"{c.name} {c.device_bytes / 2**20:,.0f} MB{' exact' if c.exact else ''}" for c in candidates]
    exact = [c for c in fits if c.exact]
    if exact:
        rank = {n: i for i, n in enumerate(EXACT_ORDER)}
        best = min(exact, key=lambda c: (rank.get(c.name, len(rank)), c.slowdown, c.device_bytes))
        return PolicyDecision(best.name, f"{best.name}: exact and fits ({best.device_bytes / 2**20:,.0f} of "
                                         f"{room_bytes / 2**20:,.0f} MB)", considered=considered)
    changing = [c for c in fits if not c.exact]
    unmeasured = [c.name for c in changing if c.quality_cost is None]
    measured = [c for c in changing if c.quality_cost is not None]
    if measured:
        best = min(measured, key=lambda c: (c.quality_cost, c.slowdown, c.device_bytes))
        return PolicyDecision(best.name, f"{best.name}: no exact policy fits {room_bytes / 2**20:,.0f} MB; it changes "
                                         f"the output by {best.quality_cost:g} against the vendor ({best.cost_source})",
                              quality_cost=best.quality_cost, cost_source=best.cost_source, considered=considered)
    smallest = min((c.device_bytes for c in candidates), default=0)
    raise StatePolicyRefused(
        f"ZERO FALLBACK: the state does not fit: {room_bytes / 2**20:,.0f} MB left for it on the node; "
        f"considered {'; '.join(considered) or 'nothing'}"
        + (f"; {', '.join(unmeasured)} would fit but change the output with no measured quality cost "
           f"against the vendor, so they are not taken" if unmeasured else "")
        + f" (the smallest needs {smallest / 2**20:,.0f} MB)")
