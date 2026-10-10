"""A component the compiled engine STREAMS is priced by what one load holds, not by the whole component.

deepseek-moe-16b-chat forced to layer_streaming on V100 (2026-10-10): the whole-component rule priced
61 670 MB of loading; the run held RssFile 9.4 GB (16 GB card, 12.9 GB segment) and 14.7 GB (32 GB
card, a 27.5 GB segment read through 2 GB shards by 8 workers), RssShmem flat at 0.8 GB once the loader
bounds its pinned uploads. The rule: file pages = min(largest load, the shards the workers hold open)
+ the workers x (1 + uploads in flight) largest tensors, as read, converted and pinned.
Injection: `_streams_through` bypassed (the whole-component `_passes_through` for every component) ->
the first test goes red.
"""
from types import SimpleNamespace as NS

import pytest

from neurobrix.core.prism import host_footprint as H

MB = 1 << 20
DT = {"float16": 2, "bfloat16": 2, "float32": 4}


def _plan():
    return NS(components={"lm": NS(device="cuda:0", dtype="float16", shard_map={})},
              component_memory={"lm": NS(weight_bytes=0, activation_bytes=0, overhead_bytes=0)},
              loading_mode="lazy")


# 10 blocks of 4 tensors of 10 MB, an embedding of 50 MB, in 5 shards of 90 MB
KEYS = {f"block.{b}.w{i}": 10 * MB for b in range(10) for i in range(4)}
KEYS["embed.w"] = 50 * MB
SHARDS = {f"s{i}": 90 * MB for i in range(5)}
SEGMENTS = [{f"block.{b}.w{i}" for b in range(0, 6) for i in range(4)} | {"block.0.inv_freq"},
            {f"block.{b}.w{i}" for b in range(6, 10) for i in range(4)}]


def _price(workers, in_flight, stored="float16", shards=SHARDS):
    return H.host_footprint(_plan(), {"lm": KEYS}, {"lm": shards}, "compiled", None, DT, lambda k: True,
                            {"lm": {stored}}, streamed_loads={"lm": SEGMENTS},
                            load_workers=workers, pinned_in_flight=in_flight)["transient_bytes"]


def test_the_largest_segment_and_the_tensors_in_flight():
    # largest load 240 MB (6 blocks; the rest is the 50 MB embedding); 1 worker x (1 + 1) in flight:
    # the embedding and one 10 MB tensor, each as read and pinned (same width: no converted copy)
    assert _price(1, 1, shards={"s0": 400 * MB}) == min(240 * MB, 400 * MB) + 2 * (50 + 10) * MB


def test_the_mapped_pages_never_exceed_the_shards_held_open():
    # 2 workers hold two 90 MB shards open: 180 MB < the 240 MB segment
    assert _price(2, 1) == 180 * MB + 2 * (50 + 3 * 10) * MB


def test_a_converted_tensor_is_priced_three_times():
    # stored bf16 -> plan fp16: as read, as converted, pinned
    assert _price(1, 1, stored="bfloat16", shards={"s0": 400 * MB}) == 240 * MB + 3 * (50 + 10) * MB


def test_a_segment_in_another_key_space_is_refused():
    with pytest.raises(ValueError, match="another key space"):
        H.host_footprint(_plan(), {"lm": KEYS}, {"lm": SHARDS}, "compiled", None, DT, lambda k: True,
                         {"lm": {"float16"}}, streamed_loads={"lm": [{"model.layers.0.w"}]},
                         load_workers=1, pinned_in_flight=1)


def test_a_streamed_component_without_the_loader_figures_is_refused():
    with pytest.raises(ValueError, match="no loader workers"):
        _price(0, 1)
