"""A certified entry says whether its kernel BUILT, and an older one is read.

The screen cannot answer "did this compile on the device": a CPU fallback
COMPUTES CORRECTLY, so its deviation against the fp64 oracle is excellent —
1e-6 like any sound path. An entry certified on a fallback would record a
configuration chosen for a path that never runs, and nothing in the proof
would say so. Measured 2026-09-11: 30 of 30 Apple entries do build, and the
proof did not record it, so a reader six months later could not redo the
check. What is measured and not written does not exist.

The field is required from format `/2`. It is NOT required of `/1`, and that
is deliberate: the other machine certified 5628 shapes before the field
existed, and refusing them over a schema they could not have known would be
destroying a measurement to tidy a format.

Runnable: PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_certified_proof_says_it_built.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurobrix.kernels import autotune_certified as C

_APPLE = (Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "config"
          / "autotune" / "apple" / "apple_m4_pro")


def _doc(fmt, with_built=True):
    proof = {
        "date": "2026-09-11T00:00:00+00:00", "engine_version": "0.5.3",
        "backend": {"triton": "3.7.0", "name": "metal"},
        "shape": [4, 4, 4, False, False, "fp32", "fp32", "fp32"],
        "deviation": 1e-6, "tolerance": 1e-4, "machine": {"hostname": "x"},
        "oracle": "fp64",
    }
    if with_built:
        proof["built"] = {"gpu": True, "how": "test", "fallback": None}
    return {
        "format": fmt, "vendor": "apple", "profile": "apple_m4_pro",
        "kernel": "neurobrix.kernels.ops.matmul.matmul_kernel", "dtype": "fp32",
        "entries": {"(4, 4, 4, False, False, 'fp32', 'fp32', 'fp32')":
                    {"config": {"kwargs": {"BLOCK_M": 32}, "num_warps": 4,
                                "num_stages": 2},
                     "proof": proof, "excluded": []}},
    }


_PATH = Path("/x/apple/apple_m4_pro/matmul_kernel.fp32.json")


def test_a_current_entry_without_built_is_refused():
    problems = C.validate(_doc(C.FORMAT, with_built=False), _PATH)
    assert any("built" in p for p in problems), (
        "an entry that does not say whether it built is not a proof")


def test_a_current_entry_with_built_passes():
    assert C.validate(_doc(C.FORMAT), _PATH) == []


def test_an_older_entry_without_built_is_still_read():
    """5628 proven shapes on the other machine pre-date the field."""
    assert C.validate(_doc("nbx-autotune-certified/1", with_built=False), _PATH) == []


def test_an_unknown_format_is_still_refused():
    assert C.validate(_doc("nbx-autotune-certified/99"), _PATH)


# ---------------------------------------------------------------------------
# The directory as it stands
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _APPLE.is_dir(), reason="no Apple certified directory")
def test_every_apple_entry_records_that_it_built():
    seen = 0
    for f in sorted(_APPLE.glob("*.json")):
        doc = json.loads(f.read_text())
        assert doc.get("format") == C.FORMAT, (
            f"{f.name}: written before the field existed; re-measure rather "
            f"than hand-edit")
        for ktext, entry in doc["entries"].items():
            built = entry["proof"].get("built")
            assert built is not None, f"{f.name} {ktext}: no `built`"
            assert built["gpu"] is True, (
                f"{f.name} {ktext}: certified on a path that did NOT build — "
                f"{built.get('fallback')}")
            seen += 1
    assert seen, "the directory is empty"


@pytest.mark.skipif(not _APPLE.is_dir(), reason="no Apple certified directory")
def test_the_whole_directory_re_reads():
    for f in sorted(_APPLE.glob("*.json")):
        assert C.validate(json.loads(f.read_text()), f) == [], f.name
