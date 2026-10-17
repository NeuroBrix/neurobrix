"""A model Prism REFUSES on a class in a mode (its single refusal, `_fail_error`: the exact ladder exhausted — Sana 4K
under the Triton engine on 16 GB, which has no host backend) forms no key there: the census writes that as the model's
knowledge (zero rows, the table's keyless mark), not as a defect of the derivation that keeps stale rows. Any other
failure to read the plan stays a refusal of the derivation, named.

What it would do if the code were wrong: the refusal reads as a defect (SystemExit, not PlanRefused) — red."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import derived_census as D  # noqa: E402


def _run(stderr):
    return lambda *a, **k: SimpleNamespace(returncode=1, stderr=stderr, stdout="")


def test_prisms_refusal_is_no_key_and_another_failure_is_a_defect(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _run("ZERO FALLBACK\n" + D.CANNOT_RUN + ".\n\nComponents: ..."))
    with pytest.raises(D.PlanRefused, match=D.CANNOT_RUN):
        D.plan_record("Sana_1600M_4Kpx_BF16", ["--prompt", "x"], "triton", "v100-16g", 16160)
    monkeypatch.setattr(subprocess, "run", _run("Traceback ... KeyError: 'vae'"))
    with pytest.raises(SystemExit, match="could not be read") as e:
        D.plan_record("Sana_1600M_4Kpx_BF16", ["--prompt", "x"], "triton", "v100-16g", 16160)
    assert not isinstance(e.value, D.PlanRefused)
