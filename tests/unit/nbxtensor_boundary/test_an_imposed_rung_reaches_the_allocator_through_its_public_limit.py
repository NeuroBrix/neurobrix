"""The rung imposed on a process (`NBX_PRISM_BUDGET_MB`: the census, the rung proofs, the gate's class cells) is the
engine's; the library knows only a generic limit (`nbxtensor.memory.set_limit`), which the engine applies when it starts
(Hocine 2026-10-29 16:28: nbxtensor holds nothing of NeuroBrix; `imposed_rung_mb` leaves nbxtensor.host).

What this test would do if the code were wrong: an engine started under a rung that leaves the allocator at the
physical card (red); a library that still reads NeuroBrix's variable or asks the host for the rung (red); a limit that
stays after the rung is lifted (red)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.unit.child_env import child_env

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "nbxtensor" / "src" / "nbxtensor"

_PROBE = r"""
import neurobrix
from nbxtensor.memory.allocator import DeviceAllocator as D
print("LIMIT", D._imposed_bytes(0))
"""


def _limit_seen(**env):
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, capture_output=True, text=True, timeout=300,
                         env=child_env(PYTHONPATH=f"{REPO / 'src'}:{REPO / 'nbxtensor' / 'src'}", PYTHONNOUSERSITE="1",
                                       CUDA_VISIBLE_DEVICES="", **env))
    return out.stdout + out.stderr


def test_the_engine_started_under_a_rung_holds_the_allocator_to_it():
    assert f"LIMIT {2048 << 20}" in _limit_seen(NBX_PRISM_BUDGET_MB="2048")
    assert "LIMIT None" in _limit_seen()


def test_the_applier_follows_the_variable(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator as D
    from neurobrix.core.prism.memory_budget import apply_imposed_rung
    monkeypatch.setattr(D, "_limit_bytes", {})
    monkeypatch.setattr(D, "_rung_limit_bytes", {})
    monkeypatch.setattr(D, "_pool_cap_bytes", {})
    monkeypatch.setenv("NBX_PRISM_BUDGET_MB", "1024")
    apply_imposed_rung()
    assert D._imposed_bytes(0) == 1024 << 20
    monkeypatch.delenv("NBX_PRISM_BUDGET_MB")
    apply_imposed_rung()
    assert D._imposed_bytes(0) is None


def test_the_library_knows_no_rung():
    from nbxtensor import host
    assert "imposed_rung_mb" not in host.SERVICES
    readers = [str(p.relative_to(LIB)) for p in LIB.rglob("*.py") if "NBX_PRISM_BUDGET_MB" in p.read_text()]
    assert not readers, f"the library reads NeuroBrix's rung variable in {readers}"
