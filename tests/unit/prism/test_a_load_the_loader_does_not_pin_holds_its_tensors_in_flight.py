"""A compiled load onto a device the loader does not pin for holds its tensors in flight, not the
component read and pinned.

Measured 2026-10-12 on the M4 Pro (mq25 46b0a9b7, gate chain `results/apple_gate_mq25_retrace8_2026_10_12/`):
PixArt-XL-2-1024-MS retraced stores its text encoder in float32 (18 GB). The native plan (mps, bf16,
eager) priced 50 170 MB of host, 32 740 MB of it "transient" = every component as read plus a pinned
copy at the plan's width, summed under eager — the CUDA rule (the caching host allocator keeps every
pinned block, pytorch#134332). The harness withheld the cell for hours; Sana-4K and Sana MultiLing
were refused at 31/32 GiB while their earlier native rows measured 2.6-7.2 GB of host peak.

The loader (core/io/weight_loader.py `_load_with_pinned_dma`) pins only when the device is cuda/hip;
elsewhere it reads one tensor per worker through `safe_open`, converts it, `.to(device)` and drops it.
What a load holds on the host is the tensors in flight: one per worker, the largest that many, as read
and as converted. Its file pages are clean and counted as available by core/host_memory.py (the M4,
PixArt-XL-2's T5 read so by 8 workers: peak RSS 15.6 GB, peak phys_footprint 0.59 GB), so not priced.
Nothing outlives the load, so the passes do not add up under eager. The device copy is the device
plan's (the unified `device` term).

Run: PYTHONPATH=src python -m pytest tests/unit/prism/test_a_load_the_loader_does_not_pin_holds_its_tensors_in_flight.py
"""
from types import SimpleNamespace as NS

import pytest

from neurobrix.core.prism import host_footprint as H

MB = 1 << 20
DT = {"float16": 2, "bfloat16": 2, "float32": 4}


def _plan(loading, **comps):
    return NS(components={n: NS(device=d, dtype=t, shard_map={}) for n, (d, t) in comps.items()},
              component_memory={n: NS(weight_bytes=0, activation_bytes=0, overhead_bytes=0) for n in comps},
              loading_mode=loading)


def _block(k):
    return k.startswith("block.")


# text encoder: 4 shards of 400 MB, its largest tensors 250, 200, 100 MB; transformer: one 300 MB shard
KEYS = {"te": {"embed": 250 * MB, "block.0.w": 200 * MB, "block.1.w": 100 * MB,
               **{f"block.{i}.x": 50 * MB for i in range(2, 23)}},
        "tr": {"block.0.w": 120 * MB, "block.1.w": 180 * MB}}
SHARDS = {"te": {f"s{i}": 400 * MB for i in range(4)}, "tr": {"s0": 300 * MB}}
STORED = {"te": {"float32"}, "tr": {"bfloat16"}}


def _price(device, loading="eager", workers=2):
    plan = _plan(loading, te=(device, "bfloat16"), tr=(device, "bfloat16"))
    return H.host_footprint(plan, KEYS, SHARDS, "compiled", None, DT, _block, STORED,
                            load_workers=workers, pinned_in_flight=4)


def test_an_mps_load_holds_its_tensors_in_flight_not_its_file_pages():
    # te: 2 workers -> the 2 largest tensors, each as read and converted (fp32 -> bf16: 250 + 125,
    #     200 + 100); tr: 180 + 120, no conversion. No shard is priced.
    te = (250 + 125) * MB + (200 + 100) * MB
    tr = 180 * MB + 120 * MB
    f = _price("mps:0")
    assert f["transient_bytes"] == max(te, tr), (f["transient_bytes"] / MB, te / MB)


def test_nothing_outlives_an_mps_load_so_eager_takes_the_largest():
    assert _price("mps:0", "eager")["transient_bytes"] == _price("mps:0", "lazy")["transient_bytes"]


def test_the_cuda_rule_is_unchanged():
    # read whole + pinned at the plan's width, summed under eager (the caching host allocator).
    te = 1600 * MB + 800 * MB
    tr = 300 * MB + 300 * MB
    assert _price("cuda:0")["transient_bytes"] == te + tr


def test_an_mps_load_without_its_workers_is_refused_by_name():
    with pytest.raises(ValueError, match="te.*loader workers"):
        _price("mps:0", workers=0)


def test_a_streamed_cuda_component_still_adds_up_under_eager():
    """The Dell, 2026-10-12 15:26: the change must leave every pinning load's price as it was. Under eager two
    streamed CUDA components' passes were summed; they stay summed (only a load the loader does not pin for
    leaves the sum). Injection: streamed pinning passes priced as max(...) -> RED."""
    keys = {"lm": {"block.0.w": 100 * MB, "block.1.w": 100 * MB}, "te": {"block.0.w": 50 * MB, "block.1.w": 50 * MB}}
    shards = {"lm": {"s0": 200 * MB}, "te": {"s0": 100 * MB}}
    loads = {"lm": [{"block.0.w"}, {"block.1.w"}], "te": [{"block.0.w"}, {"block.1.w"}]}
    kw = dict(load_workers=1, pinned_in_flight=1)

    def price(*names):
        plan = _plan("eager", **{n: ("cuda:0", "float16") for n in names})
        return H.host_footprint(plan, {n: keys[n] for n in names}, {n: shards[n] for n in names}, "compiled", None,
                                DT, _block, {n: {"float16"} for n in names},
                                streamed_loads={n: loads[n] for n in names}, **kw)["transient_bytes"]
    assert price("lm", "te") == price("lm") + price("te"), (price("lm", "te") / MB, price("lm") / MB, price("te") / MB)
