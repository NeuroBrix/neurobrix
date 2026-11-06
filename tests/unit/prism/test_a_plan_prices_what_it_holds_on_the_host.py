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
                         None, DT, _block, load_workers=1)
    assert f["steady"]["vae"] == 200 * MB + 30 * MB + 5 * MB


def test_compiled_zero3_holds_the_share_of_weights_on_host_shards():
    plan = _plan(lm=("cuda:0", "float16", {"s0": "cpu", "s1": "cpu", "s2": "cuda:0", "s3": "cuda:0"},
                     400 * MB, 50 * MB, 10 * MB))
    f = H.host_footprint(plan, {"lm": {f"w{i}": 100 * MB for i in range(4)}}, {"lm": {f"s{i}": 100 * MB for i in range(4)}},
                         "compiled", None, DT, _block, load_workers=1, pinned_in_flight=1)
    assert f["steady"]["lm"] == 200 * MB


def test_lazy_holds_the_largest_eager_holds_them_all():
    comps = dict(a=("cpu", "float32", {"s": "cpu"}, 10 * MB, 0, 0), b=("cpu", "float32", {"s": "cpu"}, 30 * MB, 0, 0))
    sizes, shards = {"a": {"w": 1}, "b": {"w": 1}}, {"a": {"s": 1}, "b": {"s": 1}}
    assert H.host_footprint(_plan("lazy", **comps), sizes, shards, "compiled", None, DT, _block,
                            load_workers=1)["steady_bytes"] == 30 * MB
    assert H.host_footprint(_plan("eager", **comps), sizes, shards, "compiled", None, DT, _block,
                            load_workers=1)["steady_bytes"] == 40 * MB


def test_triton_pins_only_block_weights_and_loads_a_tensor_at_a_time():
    plan = _plan(lm=("cuda:0", "float16", {"s0": "cpu"}, 999 * MB, 0, 0))
    keys = {"lm": {"block.0.w": 60 * MB, "block.1.w": 40 * MB, "embed.w": 80 * MB}}
    f = H.host_footprint(plan, keys, {"lm": {"s0": 180 * MB}}, "triton", None, DT, _block)
    assert f["steady"]["lm"] == 100 * MB
    assert f["transient_bytes"] == H.TRITON_COPIES_PER_TENSOR * 80 * MB


def test_a_compiled_whole_load_is_priced_per_tensor_and_a_host_held_component_unpinned():
    # Every compiled load reads one tensor at a time per worker, at most `pinned_in_flight` uploads outstanding, and
    # empties the pinned host cache when the component ends (core/io/weight_loader.py); a component HELD on the host
    # loads there unpinned and is copied, one weight at a time, into one block registered once filled
    # (core/strategies/pinned_block.py). What this does if the price were wrong: the whole component read and pinned
    # (2 x its bytes: Qwen3-Coder-30B priced 115 GB of loading for 57.6 GB of weights, measured 62.5 GB peak) — red.
    keys = {"lm": {f"w{i}": (i + 1) * MB for i in range(20)}}
    shards = {"lm": {f"s{i}": 21 * MB for i in range(10)}}
    lm = sum(range(1, 21)) * MB
    on_card = _plan(lm=("cuda:0", "float16", {}, 0, 0, 0))
    f = H.host_footprint(on_card, keys, shards, "compiled", None, DT, _block, {"lm": {"float16"}},
                         load_workers=2, pinned_in_flight=1)
    # 2 workers x (1 + 1 upload) = 4 tensors in flight (20, 19, 18, 17 MB), each as read and as pinned, beside the
    # file pages of the 2 shards the workers hold open
    assert f["transient_bytes"] == 2 * 21 * MB + 2 * (20 + 19 + 18 + 17) * MB < 2 * lm
    held = _plan(lm=("cuda:0", "float16", {f"s{i}": "cpu" for i in range(10)}, lm, 0, 0))
    g = H.host_footprint(held, keys, shards, "compiled", None, DT, _block, {"lm": {"float16"}},
                         load_workers=2, pinned_in_flight=1)
    # unpinned per tensor: one tensor per worker, as read (no conversion: stored at the plan's width); file pages
    # are reclaimable and not priced where the loader does not pin (the streamed rule's)
    assert g["transient_bytes"] == (20 + 19) * MB and g["steady"]["lm"] == lm


def test_the_base_is_the_profiles_measurement_or_is_said_to_be_missing():
    plan = _plan(lm=("cuda:0", "float16", {}, 0, 0, 0))
    f = H.host_footprint(plan, {"lm": {}}, {"lm": {}}, "triton", 512, DT, _block)
    assert f["base_bytes"] == 512 * MB and f["base_measured"] and f["total_bytes"] == 512 * MB
    f = H.host_footprint(plan, {"lm": {}}, {"lm": {}}, "triton", None, DT, _block)
    assert f["base_bytes"] == 0 and not f["base_measured"]


def test_the_planning_process_resident_memory_is_part_of_the_figure():
    # The parsed container is held by the process that plans (1.5-4.8x its JSON bytes, not a constant):
    # it is read from the process, then the engine's measured addition and the plan's holdings go on top.
    plan = _plan(lm=("cuda:0", "float16", {}, 0, 0, 0))
    f = H.host_footprint(plan, {"lm": {}}, {"lm": {}}, "triton", 100, DT, _block, resident_bytes=300 * MB)
    assert f["resident_bytes"] == 300 * MB and f["total_bytes"] == 400 * MB
    assert "resident 300 MB" in H.summary(f)
    assert H.process_footprint_now() > 0


def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="no host rules"):
        H.host_footprint(_plan(), {}, {}, "metal-magic", None, DT, _block)
    assert H.engine_of("triton_sequential") == "triton" and H.engine_of("compiled") == "compiled"


def test_a_unified_device_s_plan_is_held_on_the_host():
    """The Mac, 2026-10-03: on a unified device the plan's device bytes are host bytes, so the host
    figure a ledger reserves carries them; on a discrete card it carries nothing more.
    Injection: `device_bytes` left out of `total_bytes` -> the unified total equals the discrete one, RED."""
    plan = _plan(lm=("mps", "float16", {}, 0, 0, 0))
    discrete = H.host_footprint(plan, {"lm": {"w": 10 * MB}}, {"lm": {"s0": 10 * MB}}, "triton", None, DT, _block)
    unified = H.host_footprint(plan, {"lm": {"w": 10 * MB}}, {"lm": {"s0": 10 * MB}}, "triton", None, DT, _block,
                               device_bytes=18085 * MB)
    assert discrete["device_bytes"] == 0
    assert unified["total_bytes"] == discrete["total_bytes"] + 18085 * MB
    assert "device plan 18085 MB (unified)" in H.summary(unified)
    assert "device plan" not in H.summary(discrete)


def test_the_solver_counts_the_device_plan_only_for_a_unified_profile():
    """Injection: `unified_device_bytes` returning the plan on a discrete card, or 0 on a unified
    one -> RED."""
    from neurobrix.core.prism.solver import unified_device_bytes
    plan = NS(total_memory_mb=18085.0)
    assert unified_device_bytes(plan, NS(devices=[NS(has_unified_memory=True)])) == 18085 * MB
    assert unified_device_bytes(plan, NS(devices=[NS(has_unified_memory=False)] * 4)) == 0
