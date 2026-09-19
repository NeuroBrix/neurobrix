"""A certified setting proven under another compiler must not be served in silence.

Every certified entry carries a proof, and the proof carries the backend it was
measured under:

    "proof": {"engine_version": "0.5.3",
              "backend": {"triton": "3.6.0", "name": "cuda"}, ...}

Measured on this rack, 2026-09-18: 9,792 certified entries, 9,655 proven under
triton 3.6.0 and 137 under 3.7.0. The production stack carries triton 3.6.0 and
the candidate stack carries 3.8.0, so moving to the candidate would serve EVERY
ONE of those entries under a compiler none of them was proven against.

`autotune_certified.validate()` checks the format, the vendor/profile/kernel/dtype
consistency, the key arity and the presence of each proof field. It does not
compare `backend` or `engine_version` with what is running, and no comparison of
`engine_version` exists anywhere under `src/`. So the mismatch is invisible.

A kernel's compiled arithmetic is part of its bytes: the configuration retained
for a shape is a property of the compiler that produced it, not of the shape
alone. The doctrine already says so -- a proof "that does not re-read is refused",
and "a runtime exclusion contradicting a certification is a reported finding,
never a silence".

WHAT THIS GATE ASKS FOR, and what it deliberately does not. It asks that the
engine can SAY a served entry was proven under a different backend. It does NOT
ask that the entry be refused: refusing would make the candidate stack sweep all
9,792 shapes at runtime, which is a production behaviour change and the owner's
call, not a test's.

Run: PYTHONPATH=src python -m pytest tests/unit/kernels/test_a_certified_config_names_the_compiler_that_proved_it.py
"""
from __future__ import annotations

import glob
import json

import pytest


def _entries():
    for f in glob.glob("src/neurobrix/config/autotune/**/*.json", recursive=True):
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        items = doc.get("entries", doc)
        if isinstance(items, dict):
            for key, entry in items.items():
                if isinstance(entry, dict) and isinstance(entry.get("proof"), dict):
                    yield f, key, entry["proof"]


def test_every_certified_proof_records_the_backend_that_produced_it():
    """The datum has to be there before anything can compare it."""
    rows = list(_entries())
    assert rows, "no certified entries found — this cell would pass vacuously"
    missing = [(f, k) for f, k, p in rows
               if not isinstance(p.get("backend"), dict)
               or not p["backend"].get("triton")]
    assert not missing, (
        f"{len(missing)} of {len(rows)} certified entries carry no backend "
        f"version in their proof; the first is {missing[0]}")


def test_the_directory_can_be_asked_which_backend_it_was_proven_under():
    """A census the engine can take of itself, not a fact in a comment.

    The numbers in this file's docstring are only trustworthy if something
    recomputes them. This does.
    """
    from collections import Counter
    c = Counter(p["backend"]["triton"] for _, _, p in _entries())
    assert c, "no backend versions to count"
    # Not asserting WHICH versions: that changes as certifications are redone.
    # Asserting that the question has a definite answer, which is the point.
    assert all(isinstance(v, str) and v for v in c), c


def test_the_engine_reports_a_backend_mismatch_rather_than_serving_in_silence():
    """The capability this gate exists for.

    `autotune_certified` must expose a way to learn that a served entry was
    proven under a different backend than the one running. Absent that, a stack
    move re-uses 9,792 measurements taken under another compiler and says
    nothing -- which is the silence the doctrine forbids.
    """
    from neurobrix.kernels import autotune_certified as ac

    fn = getattr(ac, "backend_mismatch", None)
    assert callable(fn), (
        "autotune_certified exposes no `backend_mismatch(proof)`: a certified "
        "entry proven under another compiler is served with nothing said. "
        "Measured 2026-09-18: 9,792 entries proven under triton 3.6.0/3.7.0, "
        "candidate stack runs 3.8.0.")

    proven = {"engine_version": "0.5.3", "backend": {"triton": "3.6.0", "name": "cuda"}}
    assert fn(proven, running={"triton": "3.8.0", "name": "cuda"}), \
        "a different triton must be reported as a mismatch"
    assert not fn(proven, running={"triton": "3.6.0", "name": "cuda"}), \
        "the same triton must NOT be reported, or every run cries wolf"
