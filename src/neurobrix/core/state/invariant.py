"""The invariant state of a repeated component — evaluated once, kept between runs (state engine step 4).

docs/design/state-engine.md §4.1 (`invariant`, `cross_kv`): the ops `core/optim/passes/invariant.plan_invariant` derives
from the container's declared invariant inputs run at the first run of a request; their outputs (the frontier) are kept,
and every later run whose invariant inputs are BIT-identical to the kept ones skips those ops and reads the kept frontier.
Exact by construction: the same ops on the same bits give the same bits, and anything else is a miss that re-evaluates.

ONE entry per sequence: a run whose invariant inputs differ (the other branch of a guidance split into two passes, the
next window of an encoder-decoder) replaces it. That bound is what Prism prices — the frontier and the kept inputs held
for a whole run (`core/prism/profiler.estimate_peak_memory(held_tids=...)`) — and it holds no matter how a flow calls.

A run that binds the VERY SAME tensor objects as the evaluating run (same object, same storage address — the state holds
them, so neither can be recycled) reuses the state without comparing bits: their bits can change only by an in-place
write, and none reaches an invariant input — the planner excludes any input an in-place op of the component touches, and
no flow writes into a tensor it binds (2026-10-19). Any other binding is compared bit for bit. The comparison is what the
fast path saves: whisper-large-v3-turbo's decoder binds the same encoder states at every token, and comparing their
3.84 MB cost 1.28 ms per token in the decode (37 ms over 29 tokens), more than the 48 ops it skips on the compiled engine
(tpot 6.7 -> 7.8 ms).

Torch-free: each engine passes its own tensor primitives (R33 — the Triton engine never sees torch):
  same_bits(a, b)        exact bitwise equality of two tensors (shape, dtype and every bit)
  address(t)             the start of t's storage (the fast path's second key beside the object itself)
  keep(t)                an owned copy of t
  shares_storage(a, b)   whether a and b view one allocation
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class InvariantState:
    def __init__(self, plan: Dict[str, Any], same_bits: Callable[[Any, Any], bool], keep: Callable[[Any], Any],
                 shares_storage: Callable[[Any, Any], bool], address: Callable[[Any], int]):
        self.op_indices = tuple(plan["op_indices"])
        self.frontier_slots = tuple(plan["frontier_slots"])
        self.input_slots = tuple(plan["input_slots"])
        self._same_bits, self._keep, self._shares, self._address = same_bits, keep, shares_storage, address
        self._kept_live: Optional[Dict[int, Any]] = None   # the evaluating run's own input objects, held
        self.fast_reuses = 0
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
        if self._kept_inputs is not None:
            fast = self._kept_live is not None and all(
                arena[s] is self._kept_live[s][0] and self._address(arena[s]) == self._kept_live[s][1]
                for s in self.input_slots)
            if fast or all(self._same_bits(arena[s], self._kept_inputs[s]) for s in self.input_slots):
                for s in self.frontier_slots:
                    arena[s] = self._kept_frontier[s]
                self.reuses += 1
                self.fast_reuses += int(fast)
                if not fast:          # equal bits in new objects: they are the objects the next run is checked against
                    self._kept_live = {s: (arena[s], self._address(arena[s])) for s in self.input_slots}
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
        self._kept_live = {s: (v, self._address(v)) for s, v in zip(self.input_slots, input_values)}
        self.evaluations += 1

    def release(self) -> None:
        """Drop the kept state (the end of a request, or a miss)."""
        self._kept_inputs = self._kept_frontier = self._pending = self._kept_live = None
        self._live_inputs = ()

    @property
    def holds(self) -> bool:
        return self._kept_frontier is not None
