"""A temporal stream's chunk on the compiled engine is priced at what the device's DRIVER holds for it — the chunk's
price times its component family's measured ratio of driver memory to price (`memory.compiled_stream_driver_over_price`
of the vendor file, measured on the Mac M4 Pro 2026-10-26: mochi-1-preview's VAE chunk, 17 714 MB of driver for a
10 159 MB price) — against the device's own limit less the engine's base, the allocator's other allocations being
inside the ratio (counted once: decision C1, the supervisor 2026-10-27 07:30).

Before: the chunk was priced at its tensors against the rung, and mochi native on a calm Mac planned chunks [1,2,2]
(13 089 MB priced) that MPS refused at native_group_norm::32 (14.88 GiB allocated + 2.38 asked + 1.40 other > 17.76 GiB,
2026-10-27 03:0x); 1-frame chunks ran (2026-10-24, 2026-10-26 23:4x). RED: the schedule [1, 2, 2]. A vendor that states
no ratio for a family (NVIDIA: null) is priced as before — asserted on the V100-16GB profile. No card."""
import pytest

from tests.unit.prism._pinned_machine import pin_host
from tests.unit.prism.test_a_component_over_the_rung_streams_inside_itself import _plan, no_door


def test_mochi_native_on_a_calm_mac_streams_its_vae_one_frame_at_a_time(monkeypatch):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17582, "the calm Mac (its measured system share)")
    p, _ = _plan("mochi-1-preview", "compiled")
    sched = (p.component_stream or {}).get("vae", {}).get("schedule")
    assert sched == [1, 1, 1, 1, 1], sched


def test_the_ratio_is_a_vendor_fact_with_its_source_and_null_where_unmeasured():
    from neurobrix.core.config.loader import vendor_config_for_device
    apple = (vendor_config_for_device("apple", "apple_silicon", "0.0", model="Apple M4 Pro").get("memory") or {})
    table = apple.get("compiled_stream_driver_over_price")
    assert isinstance(table, dict) and table.get("vae", {}).get("ratio") and table["vae"].get("source"), table
    nvidia = (vendor_config_for_device("nvidia", "volta", "7.0").get("memory") or {})
    assert "compiled_stream_driver_over_price" in nvidia and nvidia["compiled_stream_driver_over_price"] is None
