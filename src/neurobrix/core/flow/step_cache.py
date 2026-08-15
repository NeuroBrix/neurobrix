"""F1 step cache — delayed-signal whole-output reuse (drift tier).

Shared flow brick consumed by every compiled-engine step loop that
denoises iteratively (iterative_process driver loop, image_gen leg).
Design + the six-clause drift discipline: optimization scoping doc
"F1". R30 mirror: triton/flow/step_cache.py.

Opt-in only: activation requires BOTH threshold and
max_consecutive_skips from registry-driven defaults
(defaults.step_cache), CLI --set global.step_cache_*, or the
NBX_STEP_CACHE_* env pins used by benchmark rows. ABSENT = inactive —
no engine-side constants for a drift-tier pass (thresholds are data,
R15). Flow-specific eligibility (single loop component, no
dual-denoiser) stays in the calling flow.
"""

import os
from typing import Optional


class StepCache:
    """Delayed-signal skip decision over consecutive step predictions.

    The signal is the relative L1 between the last two EXECUTED
    predictions; a skip reuses the previous fully-processed prediction
    while the scheduler still advances the timestep normally.
    """

    def __init__(self, threshold: float, max_skips: int, total: int):
        self.threshold = threshold
        self.max_skips = max_skips
        self.total = total
        self.prev_pred = None
        self.signal: Optional[float] = None
        self.consec = 0
        self.skipped = 0
        self.executed = 0

    @classmethod
    def setup(cls, ctx, num_steps: int) -> Optional["StepCache"]:
        """Config channels, highest precedence first: --set
        global.step_cache_* > NBX_STEP_CACHE_* env > registry defaults.
        Explicit `is not None` checks — a --set value of 0/0.0 is a
        legitimate override, never a fall-through. Returns None when no
        channel opts in; raises on a partial config (ZERO FALLBACK)."""
        cfg = ctx.pkg.defaults.get("step_cache")
        resolved = ctx.variable_resolver.resolved
        thr_override = resolved.get("global.step_cache_threshold")
        if thr_override is None:
            thr_override = os.environ.get("NBX_STEP_CACHE_THRESHOLD")
        if cfg is None and thr_override is None:
            return None
        cfg = dict(cfg or {})
        if thr_override is not None:
            cfg["threshold"] = float(thr_override)
        mcs_override = resolved.get("global.step_cache_max_skips")
        if mcs_override is None:
            mcs_override = os.environ.get("NBX_STEP_CACHE_MAX_SKIPS")
        if mcs_override is not None:
            cfg["max_consecutive_skips"] = int(mcs_override)
        if "threshold" not in cfg or "max_consecutive_skips" not in cfg:
            raise RuntimeError(
                "ZERO FALLBACK: step_cache needs BOTH threshold and "
                "max_consecutive_skips (registry defaults or --set "
                "global.step_cache_*) — no engine-side constants for a "
                "drift-tier pass")
        return cls(float(cfg["threshold"]),
                   int(cfg["max_consecutive_skips"]), int(num_steps))

    def should_skip(self, step_idx: int) -> bool:
        """Skip iff the LAST EXECUTED step's signal sat below the
        threshold; never first/last step; never more than
        max_consecutive_skips in a row (re-arms on each executed
        step)."""
        if step_idx == 0 or step_idx >= self.total - 1:
            return False
        if self.prev_pred is None or self.signal is None:
            return False
        if self.consec >= self.max_skips:
            return False
        if self.signal < self.threshold:
            self.skipped += 1
            self.consec += 1
            return True
        return False

    def observe(self, model_output) -> None:
        """Relative-L1 signal between consecutive EXECUTED predictions
        (one scalar sync per executed step at the flow boundary)."""
        prev = self.prev_pred
        if (prev is not None and hasattr(prev, "shape")
                and tuple(prev.shape) == tuple(model_output.shape)):
            den = float(prev.float().abs().sum().item()) or 1.0
            num = float((model_output.float()
                         - prev.float()).abs().sum().item())
            self.signal = num / den
        else:
            self.signal = None
        self.prev_pred = model_output
        self.executed += 1
        self.consec = 0

    def report(self) -> None:
        """F1 rates line (drift-discipline clause 4)."""
        print(f"[StepCache] skipped {self.skipped}/{self.total} "
              f"steps (threshold={self.threshold}, "
              f"max_consecutive={self.max_skips})")
