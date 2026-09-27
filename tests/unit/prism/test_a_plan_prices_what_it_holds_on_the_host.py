"""Prism prices a plan's host footprint from the plan it chose — per engine, from the runtime's own
rules — so a host ledger reserves what the plan says and a measured peak can judge it.

The owner's rule of 2026-09-27 14:27: never a per-model table; the plan's placement on this profile.
Each case below is one rule of core/prism/host_footprint.py, read from the runtime; before this branch
the module does not exist and every case fails.
"""
from types import SimpleNamespace as NS

import pytest

from neurobrix.core.prism import host_footprint as H

MB = 1 << 20
DT = {"float16": 2, "bfloat16": 2, "float32": 4}


def _plan(loading="lazy", **comps):
    components, memory = {}, {}
    for name, (device, dtype, shard_map, w, a, o) in comps.items():
        components[name] = NS(device=device, dtype=dtype, shard_map=shard_map)
        memory[name] = NS(weight_bytes=w, activation_bytes=a, overhead_bytes=o)
    return NS(components=components, component_memory=memory, loading_mode=loading)


def _block(k):
    return k.startswith("block.")


def test_compiled_host_compute_holds_fp32_weights_and_activations():
    plan = _plan(vae=("cpu", "float16", {"s0": "cpu"}, 100 * MB, 30 * MB, 5 * MB))
    f = H.host_footprint(plan, {"vae": {"w": 100 * MB}}, {"vae": {"s0": 100 * MB}}, "compiled",
                         None, 8, DT, _block)
    assert f["steady"]["vae"] == 200 * MB + 30 * MB + 5 * MB


def test_compiled_zero3_holds_the_share_of_weights_on_host_shards():
    plan = _plan(lm=("cuda:0", "float16", {"s0": "cpu", "s1": "cpu", "s2": "cuda:0", "s3": "cuda:0"},
                     400 * MB, 50 * MB, 10 * MB))
    f = H.host_footprint(plan, {"lm": {}}, {"lm": {f"s{i}": 100 * MB for i in range(4)}}, "compiled",
                         None, 8, DT, _block)
    assert f["steady"]["lm"] == 200 * MB


def test_lazy_holds_the_largest_eager_holds_them_all():
    comps = dict(a=("cpu", "float32", {"s": "cpu"}, 10 * MB, 0, 0), b=("cpu", "float32", {"s": "cpu"}, 30 * MB, 0, 0))
    sizes, shards = {"a": {}, "b": {}}, {"a": {"s": 1}, "b": {"s": 1}}
    assert H.host_footprint(_plan("lazy", **comps), sizes, shards, "compiled", None, 8, DT, _block)["steady_bytes"] == 30 * MB
    assert H.host_footprint(_plan("eager", **comps), sizes, shards, "compiled", None, 8, DT, _block)["steady_bytes"] == 40 * MB


def test_triton_pins_only_block_weights_and_loads_a_tensor_at_a_time():
    plan = _plan(lm=("cuda:0", "float16", {"s0": "cpu"}, 999 * MB, 0, 0))
    keys = {"lm": {"block.0.w": 60 * MB, "block.1.w": 40 * MB, "embed.w": 80 * MB}}
    f = H.host_footprint(plan, keys, {"lm": {"s0": 180 * MB}}, "triton", None, 8, DT, _block)
    assert f["steady"]["lm"] == 100 * MB
    assert f["transient_bytes"] == H.TRITON_COPIES_PER_TENSOR * 80 * MB


def test_the_compiled_transient_is_the_shards_in_flight_and_a_conversion_adds_a_copy():
    plan = _plan(lm=("cuda:0", "float16", {}, 0, 0, 0))
    shards = {"lm": {f"s{i}": (i + 1) * MB for i in range(20)}}
    same = H.host_footprint(plan, {"lm": {}}, shards, "compiled", None, 8, DT, _block, {"lm": {"float16"}})
    assert same["transient_bytes"] == 8 * H.COMPILED_COPIES_PER_SHARD * 20 * MB and same["steady_bytes"] == 0
    conv = H.host_footprint(plan, {"lm": {}}, shards, "compiled", None, 8, DT, _block, {"lm": {"bfloat16"}})
    assert conv["transient_bytes"] == 8 * (H.COMPILED_COPIES_PER_SHARD + H.COMPILED_CONVERSION_COPIES) * 20 * MB


def test_the_base_is_the_profiles_measurement_or_is_said_to_be_missing():
    plan = _plan(lm=("cuda:0", "float16", {}, 0, 0, 0))
    f = H.host_footprint(plan, {"lm": {}}, {"lm": {}}, "triton", 512, 8, DT, _block)
    assert f["base_bytes"] == 512 * MB and f["base_measured"] and f["total_bytes"] == 512 * MB
    f = H.host_footprint(plan, {"lm": {}}, {"lm": {}}, "triton", None, 8, DT, _block)
    assert f["base_bytes"] == 0 and not f["base_measured"]


def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="no host rules"):
        H.host_footprint(_plan(), {}, {}, "metal-magic", None, 8, DT, _block)
    assert H.engine_of("triton_sequential") == "triton" and H.engine_of("compiled") == "compiled"
