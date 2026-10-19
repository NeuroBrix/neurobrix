"""The invariant state of a repeated component — evaluated once, kept between runs (state engine step 4).

docs/design/state-engine.md §4.1 (`invariant`, `cross_kv`): the ops `core/optim/passes/invariant.plan_invariant` derives
from the container's declared invariant inputs run at the first run of a request; their outputs (the frontier) are kept,
and every later run whose invariant inputs are BIT-identical to the kept ones skips those ops and reads the kept frontier.
Exact by construction: the same ops on the same bits give the same bits, and anything else is a miss that re-evaluates.

ONE entry per sequence: a run whose invariant inputs differ (the other branch of a guidance split into two passes, the
next window of an encoder-decoder) replaces it. That bound is what Prism prices — the frontier and the kept inputs held
for a whole run (`core/prism/profiler.estimate_peak_memory(held_tids=...)`) — and it holds no matter how a flow calls.

Torch-free: each engine passes its own three tensor primitives (R33 — the Triton engine never sees torch):
  same_bits(a, b)        exact bitwise equality of two tensors (shape, dtype and every bit)
  keep(t)                an owned copy of t
  shares_storage(a, b)   whether a and b view one allocation
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class InvariantState:
    def __init__(self, plan: Dict[str, Any], same_bits: Callable[[Any, Any], bool], keep: Callable[[Any], Any],
                 shares_storage: Callable[[Any, Any], bool]):
        self.op_indices = tuple(plan["op_indices"])
        self.frontier_slots = tuple(plan["frontier_slots"])
        self.input_slots = tuple(plan["input_slots"])
        self._same_bits, self._keep, self._shares = same_bits, keep, shares_storage
        self._kept_inputs: Optional[Dict[int, Any]] = None
        self._kept_frontier: Optional[Dict[int, Any]] = None
        self._pending: Optional[Dict[int, Any]] = None
        self._live_inputs = ()
        self.reuses = 0
        self.evaluations = 0

    def skip_flags(self, n_ops: int):
        """A per-op tuple of booleans (True = skipped when the kept state is reused) — one index read per op."""
        flags = [False] * n_ops
        for i in self.op_indices:
            flags[i] = True
        return tuple(flags)

    def begin(self, arena) -> bool:
        """Before a run: True when the kept state serves it (the frontier slots are filled and the invariant ops are
        skipped), False when the run evaluates them (its invariant inputs are kept for the next run)."""
        values = [arena[s] for s in self.input_slots]
        if any(v is None for v in values):
            self.release()
            return False
        if self._kept_inputs is not None and all(
                self._same_bits(arena[s], self._kept_inputs[s]) for s in self.input_slots):
            for s in self.frontier_slots:
                arena[s] = self._kept_frontier[s]
            self.reuses += 1
            return True
        self.release()
        self._pending = {s: self._keep(arena[s]) for s in self.input_slots}
        self._live_inputs = values          # the run's own input tensors: a frontier viewing one is copied at end()
        return False

    def end(self, arena) -> None:
        """After an evaluating run: keep its frontier. A frontier value that views an input's allocation is copied — the
        caller may refill that input buffer before the next run. A run that left a frontier slot empty keeps nothing."""
        pending, self._pending = self._pending, None
        input_values, self._live_inputs = self._live_inputs, ()
        if pending is None:
            return
        frontier = {}
        for s in self.frontier_slots:
            v = arena[s]
            if v is None:
                return
            if any(x is not None and self._shares(v, x) for x in input_values):
                v = self._keep(v)
            frontier[s] = v
        self._kept_frontier, self._kept_inputs = frontier, pending
        self.evaluations += 1

    def release(self) -> None:
        """Drop the kept state (the end of a request, or a miss)."""
        self._kept_inputs = self._kept_frontier = self._pending = None
        self._live_inputs = ()

    @property
    def holds(self) -> bool:
        return self._kept_frontier is not None
