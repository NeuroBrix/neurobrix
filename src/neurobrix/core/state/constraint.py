"""The constraint a request's decode follows (constrained decoding, reste_apres_064 point 9a) — the seam between a
grammar engine and the generators, the same for every grammar engine.

A constraint is made by the serving layer from the request (a JSON schema, a tool-call envelope) and handed to the
flow (`FlowContext.constraint`); both generators ask it, before each step, for the tokens allowed next and tell it the
token chosen after. A request without one pays nothing. The mask is int32 words, ceil(vocab / 32), bit v % 32 of word
v // 32 set when token v is allowed (the layout nbxtensor.ops.token_bitmask applies); a constraint allows the end of
sequence only where its output may end."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Constraint(Protocol):
    def mask(self, vocab: int) -> np.ndarray:
        """int32 [ceil(vocab / 32)]: the tokens allowed at the next step."""

    def advance(self, token_id: int) -> None:
        """The token chosen at this step; refused by name when the mask did not allow it."""
