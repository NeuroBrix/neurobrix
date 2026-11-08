"""The compiled engine on Apple's MPS records what its allocator's cache grows to during a model's run — the driver's
allocation less the allocator's live bytes (`torch.mps.driver_allocated_memory() - current_allocated_memory()`), the
"other allocations" of an MPS out-of-memory — as a measured fact per model (core/runtime/runtime_growth.py), under a
method of its own, so Prism prices the measured cache instead of a vendor constant (the Dell + Mac, 2026-11-08 14:50).

Why: mochi-1-preview native on the M4 Pro failed at its VAE's `pad_time` with "MPS allocated 11.97 GiB + other
4.71 GiB > 17.76 GiB" (RUN 78, 2026-11-08) while Prism held `compiled_private_reserve_mb` 3 900 for that cache.

No card: the driver and the allocator are functions here. What each cell would do if the code were wrong: a sampler that
keeps no cache peak, or keeps the last reading instead of the largest (red); a compiled record read under the Triton
method, or a Triton record read for the compiled engine (red); a run that never records it (red)."""
import time

import pytest

from neurobrix.core.device_utils import SampledDevicePeak
from neurobrix.core.runtime import runtime_growth as RG


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "_root", lambda: tmp_path)
    return tmp_path


def test_the_sampler_keeps_the_largest_cache_and_the_reading_before_the_run():
    driver = iter([500] + [900] * 10_000)
    cache = iter([120, 300, 1800, 700] + [50] * 10_000)     # driver - live, as the run's allocator frees and caches
    s = SampledDevicePeak(lambda: next(driver), period_s=0.0005, cache_read=lambda: next(cache)).start()
    time.sleep(0.05)
    s.stop()
    assert s.cache_start == 120                       # read once at start, before the executor
    assert s.cache_peak == 1800                       # the largest seen over the span, not the last


def test_a_compiled_record_is_read_for_the_compiled_engine_only(store):
    assert RG.record("M", "compiled", "apple_silicon", {0: 120.0}, {0: 9300.0}) == 9180.0
    assert RG.load("M", "compiled", "apple_silicon") == 9180.0
    assert RG.load("M", "triton", "apple_silicon") is None
    RG.record("M", "triton", "volta", {0: 300.0}, {0: 1172.5})
    assert RG.load("M", "triton", "volta") == 872.5 and RG.load("M", "compiled", "volta") is None
    data = __import__("json").loads(RG.path("M").read_text())
    assert data["compiled"]["apple_silicon"]["method"] == RG.METHODS["compiled"] != RG.METHODS["triton"]


def test_a_compiled_run_on_mps_records_its_cache_growth():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[3] / "src" / "neurobrix" / "cli" / "commands" / "run.py").read_text()
    assert '_record_compiled_growth(' in src
    assert src.count('_record_compiled_growth(') >= 3     # defined, after a run, and after a failed one
