"""A decoding sequence's step slab, stabilized over its first runs (P-REPLAY-KV-DECODE B1; moved from the engine's
replayer, the owner's red line 2026-11-13).

Each sequence owns one `StepStabilizer` on the slab interceptor in force (`state`: a `SlabInterceptor`, the replayer's
`frozen_plan.STATE`) and ticks it at each run's start. Runs 1-2 pass through (prefill and warm-up); run 3 measures the
bytes the step carves; runs 4+ carve every allocation from the sequence's recycled `StepSlab`, sized from that
measurement (pure bump: the slab holds the step's TOTAL carved bytes, no reuse inside a step). The slab in force is
switched at each run's start: runs are synchronous, so the slab serving a run is the running sequence's.

A measurement is banked into the sequence that opened it, at the next tick of ANY sequence: another sequence's tick
must never clobber the measuring window (that once sized the decode slab from the sampler's tiny window and starved it
by 627 MB).
"""
from __future__ import annotations

import logging
from typing import Optional

from nbxtensor.memory.slab import StepSlab

_log = logging.getLogger(__name__)

#: the run whose step's carve sequence is compared with the previous one, once (a diagnosis at debug level)
SAY_AT_RUN = 6


class StepStabilizer:
    def __init__(self, state, label: str):
        self.state = state
        self.label = label                       # how the debug lines name the sequence
        self.runs = 0
        self.total: Optional[int] = None         # the bytes run 3 carved, until its slab is cut
        self.slab: Optional[StepSlab] = None
        self._said = False

    def tick(self, dev_idx: Optional[int]) -> None:
        """One run starts on `dev_idx`: the step slab its lifecycle puts in force (None while it passes through or
        measures)."""
        self._bank_pending()
        self.runs += 1
        if self.runs <= 3:
            self.state.stab_slab = None
            if self.runs == 3:
                self._open_window()
            return
        if self.total is not None and self.slab is None:
            self.slab = StepSlab.for_carved(self.total, dev_idx)
            self.total = None
            _log.debug(f"[StepSlab] {self.label}: stabilizing decode allocations ({self.slab.size / 1e6:.0f} MB slab)")
        if self.slab is not None:
            if not self._said and self.runs == SAY_AT_RUN:
                self._say_served(self.slab)
                self._said = True
            self.slab.reset()
        self.state.stab_slab = self.slab

    def _bank_pending(self) -> None:
        owner = getattr(self.state, "measure_owner", None)
        if owner is not None:
            owner.total = self.state.total_measured
            self.state.measure_owner = None
            self.state.measuring = False

    def _open_window(self) -> None:
        s = self.state
        s.measuring = True
        s.live = s.high_water = 0
        s.measured.clear()
        s.total_measured = 0
        s.measure_owner = self

    def _say_served(self, slab: StepSlab) -> None:
        _log.debug(f"[StepSlab] {self.label}: served {len(slab.allocs) + len(slab.stale)} allocations last step "
                   f"(live {len(slab.allocs)}, stale {len(slab.stale)}, shortfall {slab.shortfall})")
        a, b = slab.prev_carve or [], slab.carve_sizes
        if not (a and b):
            return
        if len(a) != len(b):
            _log.debug(f"[StepSlab] {self.label}: carve COUNT varies ({len(a)} vs {len(b)})")
            return
        div = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if div:
            i = div[0]
            _log.debug(f"[StepSlab] {self.label}: carve sizes diverge at {len(div)} indices; first at #{i}: "
                       f"{a[max(0, i - 2):i + 3]} vs {b[max(0, i - 2):i + 3]}")
        else:
            _log.debug(f"[StepSlab] {self.label}: carve size sequence IDENTICAL ({len(a)} allocs)")
