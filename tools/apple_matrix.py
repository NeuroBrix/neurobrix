#!/usr/bin/env python3
"""The Apple support matrix: every catalogue model x three modes, verified.

"Complete" (the loop's deliverable): each of the catalogue's models runs on
Apple silicon in `compiled`, `triton`, and `triton-sequential`, its output
VERIFIED against the reference arm -- never a return code. `compiled` (the
default, ATen-backed) is the reference; `triton` and `triton-sequential` are
checked against it.

This module is the CORE that decides a cell -- classify(reference, arm) ->
a verdict -- kept apart from the run loop so it is testable without a GPU. The
loop merely calls `neurobrix run` per (model, mode) with owned caches and
feeds the outputs here.

A cell is one of:
  * VERIFIED   -- ran, output matches the reference within tolerance;
  * CAUSE      -- refused/failed with a named cause (origin, IR-verdict);
  * NOT_MEASURED -- with the number that explains it (artefact size, OOM,
                    memory window), never a bare blank.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/tools/test_apple_matrix.py
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


MODES = ("compiled", "triton", "triton-sequential")
REFERENCE_MODE = "compiled"


class Verdict(Enum):
    VERIFIED = "verified"
    CAUSE = "cause"
    NOT_MEASURED = "not_measured"


@dataclass
class Cell:
    verdict: Verdict
    detail: str          # PSNR/deviation for VERIFIED; the named cause; the number
    number: Optional[float] = None


def _image_metric(ref: np.ndarray, arm: np.ndarray):
    """(psnr_db, same_shape). PSNR is the defensible-output measure for
    upscalers -- the whisper/swin2SR comparisons used it; >40 dB is a match,
    fp/order difference, not a wrong output."""
    if ref.shape != arm.shape:
        return None, False
    d = (ref.astype(np.float64) - arm.astype(np.float64))
    mse = float((d * d).mean())
    psnr = float("inf") if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)
    return psnr, True


def classify_image(ref: np.ndarray, arm: np.ndarray, psnr_floor: float = 40.0) -> Cell:
    """A cell for an image-output model, arm vs reference.

    Shape disagreement is a CAUSE, not a soft pass: a 128x128 where 896x896 is
    owed is the real-esrgan tiling defect, and must not be scored as "close".
    """
    psnr, same = _image_metric(ref, arm)
    if not same:
        return Cell(Verdict.CAUSE,
                    f"output shape {arm.shape} != reference {ref.shape} "
                    f"(a size disagreement is a defect, not a near-match)")
    if psnr >= psnr_floor:
        return Cell(Verdict.VERIFIED, f"PSNR {psnr:.1f} dB vs reference", psnr)
    return Cell(Verdict.CAUSE,
                f"PSNR {psnr:.1f} dB below the {psnr_floor:.0f} dB floor: the "
                f"arm computes a visibly different image", psnr)


def classify_text(ref: str, arm: str) -> Cell:
    """A cell for a text-output model (whisper). Byte-exact is the strong bar;
    the whisper run met it. Anything else names the divergence."""
    if ref.strip() == arm.strip():
        return Cell(Verdict.VERIFIED, "byte-identical to reference")
    return Cell(Verdict.CAUSE,
                f"transcription differs from reference: "
                f"{arm[:40]!r} vs {ref[:40]!r}")


def classify_failure(reason: str, origin: str = "?", ir_verdict: str = "?") -> Cell:
    """A cell for an arm that did not produce output: a named cause carries its
    origin (upstream/us) and whether an IR backend would remove it."""
    return Cell(Verdict.CAUSE, f"{reason} [origine: {origin}; IR: {ir_verdict}]")


def not_measured(number_reason: str) -> Cell:
    """A cell deliberately not measured, with the number that explains it --
    an artefact size over the memory budget, an OOM, a required memory window.
    Never a bare blank: a blank is the vacuous cell."""
    return Cell(Verdict.NOT_MEASURED, number_reason)
