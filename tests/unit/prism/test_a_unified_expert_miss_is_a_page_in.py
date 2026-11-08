"""On a unified device an expert store's miss is priced as a page-in from the container's file, never as free.

2026-11-08 (the Mac, RUN 79-81): DeepSeek-Coder-V2-Lite-Instruct on the Triton engine of an M4 Pro was planned
lazy_sequential + an in-place expert store, estimated at 18 s, and chosen over layer_streaming (79-90 s); the run went
past 2 400 s. `time_estimate._miss_bps` returned None for a unified device ("nothing crosses the host link"), so every
routed expert the cache missed cost nothing — but the store reads the expert in place in the mapped file, and a miss is
a page-in from storage. It is now priced at the engine's measured COLD device load rate of the stored dtype
(`weight_load_gbps`), refused by name when not measured.

What this test does if the code is wrong: the first cell fails when a unified device's miss rate is None (red on main);
the second when the rate is not the measured cold device load rate of the stored dtype; the third when an unmeasured
rate is guessed instead of refused.

Run: python -m pytest tests/unit/prism/test_a_unified_expert_miss_is_a_page_in.py
"""
import pytest

from neurobrix.core.prism import time_estimate as T

RATES = {"triton": {"BF16": {"device": {"cold": 2.38, "warm": 12.28}, "host": {"cold": 1.55, "warm": 3.0}}}}


def _facts(unified=True, rates=RATES):
    import dataclasses
    fields = {f.name for f in dataclasses.fields(T.DeviceFacts)}
    base = {n: None for n in fields}
    base.update(unified=unified, load_rates=rates)
    for n in fields:
        if base[n] is None and n not in ("load_rates", "gemv_rates", "expert_copy_bps"):
            base[n] = 1.0
    return T.DeviceFacts(**base)


def test_a_unified_miss_is_priced():
    assert T._miss_bps(_facts(), "triton", "model", "BF16", "mps:0") is not None


def test_it_is_the_measured_cold_device_load_rate():
    assert T._miss_bps(_facts(), "triton", "model", "BF16", "mps:0") == pytest.approx(2.38e9)


def test_an_unmeasured_rate_is_refused_by_name():
    with pytest.raises(T.MissingFact, match="weight_load_gbps.triton.BF16.device.cold"):
        T._miss_bps(_facts(rates={}), "triton", "model", "BF16", "mps:0")
