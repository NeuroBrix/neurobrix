"""The system's own share of a unified machine's RAM, measured calm (`host_measured.system_share_mb`), comes off the
pool a plan may fill (the Mac, 2026-10-26: 6 994 MB of 24 576 held by macOS and its apps; the door decided Ming and
Qwen3-VL by luck at the calm-free edge). What this test would do if the code were wrong: the pool reads the declared
working set less the base (18 013 MB) — RED; with the share it is 24 576 - 6 994 - the engine's base."""
from neurobrix.core.prism import PrismSolver
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, profile


def test_the_measured_share_bounds_the_pool():
    p = profile(APPLE_M4_PRO)
    assert (p.host_measured or {}).get("system_share_mb") == 6994, p.host_measured
    s = PrismSolver(); s._mode = "triton"
    base = float((p.cpu.runtime_base_mb or {}).get("triton", 0) or 0) if p.cpu else 0.0
    assert s._unified_host_pool_mb(p) == min(24576 - 6994, 18186) - base, s._unified_host_pool_mb(p)
