"""Says, once a phase, what a launch is measuring on this machine.

Why it exists. A launch that measures the machine for its generated hardware profile is silent for seconds to minutes
(0.6.2: 570 s without a line); it says what it does, once. And what it says must be what it does: the reload rate of a
stored dtype is written only from three reloads in a row within 5 % (`core/strategies/layer_streaming._note_reload`),
so on a machine that is not quiet a LATER launch makes those reloads again — said with the first launch's line, that
launch claimed "First launch on this machine ... once" at every launch until the rate settled (2026-11-11, a Mac with
a virtual machine running: launches 2 and 3).

Contract. `say(phase, what="")` logs the phase's line at INFO the first time the phase is said in this process
(`SAID`), with `what` (the expected time, already worded) placed in it; a phase not in `LINES` is a KeyError.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

LINES = {"detect": "First launch on this machine: detecting its hardware once...",
         "measure": "First launch on this machine: measuring it once for its hardware profile{what}...",
         # a later launch that owes a reload rate: a stored dtype this machine never reloaded (another model), or one
         # whose reloads were not steady enough to keep last time — the line claims neither, it says what is done
         "reload": "Measuring this machine's weight reload rate: none is kept yet for this model's stored "
                   "weights{what}..."}
SAID: set = set()


def say(phase: str, what: str = "") -> None:
    """The phase's line, once in this process."""
    if phase not in SAID:
        SAID.add(phase)
        _log.info(LINES[phase].format(what=what))
