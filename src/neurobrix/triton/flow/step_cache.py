"""F1 step cache — delayed-signal whole-output reuse (drift tier).

R30 mirror of core/flow/step_cache.py for the triton engine, consumed
by every NBXTensor step loop that denoises iteratively
(iterative_process driver loop, image_gen leg). R33: no torch — the
signal is computed at the numpy flow boundary (latent-sized D2H once
per executed step, same boundary class as the driver's own syncs).

Opt-in only: BOTH threshold and max_consecutive_skips from registry
defaults (defaults.step_cache), --set global.step_cache_*, or the
NBX_STEP_CACHE_* env pins. ABSENT = inactive (thresholds are data,
R15). Flow-specific eligibility stays in the calling flow.
"""

import os
from typing import Optional

import numpy as np


from neurobrix.core.flow.step_cache import StepCache as _SharedStepCache   # the steps both branches share (torch-free)

class StepCache(_SharedStepCache):
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
        self.prev_np = None
        self.signal: Optional[float] = None
        self.consec = 0
        self.skipped = 0
        self.executed = 0


    def observe(self, model_output) -> None:
        """Relative-L1 signal between consecutive EXECUTED predictions
        (numpy at the flow boundary — latents are small; one D2H per
        executed step). The prediction is an NBXTensor by the R33 flow
        contract — anything else is a flow bug, crash loudly (ZERO
        FALLBACK; a silent None here would freeze the skip budget)."""
        cur = model_output.float().numpy()
        prev = self.prev_np
        if prev is not None and prev.shape == cur.shape:
            den = float(np.abs(prev).sum()) or 1.0
            self.signal = float(np.abs(cur - prev).sum()) / den
        else:
            self.signal = None
        self.prev_np = cur
        self.prev_pred = model_output
        self.executed += 1
        self.consec = 0

