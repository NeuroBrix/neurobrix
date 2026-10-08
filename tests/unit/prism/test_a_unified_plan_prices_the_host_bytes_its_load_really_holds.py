"""On a unified profile the compiled host figure prices what the load really holds — no pinned copy, no file read,
no component twice — and a discrete profile prices exactly what it did.

Measured on the Mac (M4 Pro, 24 GB unified, 2026-10-08), `/usr/bin/time -l` peak memory footprint of the native
run against the figure Prism priced for it (campagnes/2026_09_22_apple/results/native_2026_10_08/measure/):

  canary-qwen-2.5b     single_gpu       priced 16 241 MB, peak  5 352 MiB (loading 9 666 = stored + "pinned")
  Voxtral-Mini-3B-2507 cpu_execution    priced 39 800 MB, peak 16 110 MiB (held 14 934 + loading 15 313 + device 9 440)
  PixArt-XL-2-1024-MS  cpu_streaming    priced 55 973 MB, peak 11 227 MiB, max RSS 16 619 MiB (held 18 815 + device 9 732)

and, in one process (the same day): safetensors' `load_file` of 1 GiB of random bf16 to "cpu" adds 0 MiB of footprint
after every byte is read (the tensors are the file's own mapped pages), `.to("mps")` adds 1 101 MiB.

Three over-counts, each decided by the profile's memory model (`has_unified_memory`), never by a device name:
  (2) a unified device takes no pinned copy (core/io/weight_loader.py: `.to(device)`, "no pinned DMA — unified
      memory"), and the read is the mapped file: a load holds only a converted copy where the plan's dtype differs
      from the stored one, freed per component;
  (3) a component computed on the host: its converted copy IS its held weights (steady), not a second pass;
      and its bytes are not in the device term too (`unified_device_bytes`, `host_bytes`).
"""
from types import SimpleNamespace as NS

from neurobrix.core.prism import host_footprint as H
from neurobrix.core.prism.solver import unified_device_bytes

MB = 1 << 20
DT = {"float16": 2, "bfloat16": 2, "float32": 4}


def _plan(loading, **comps):
    return NS(components={n: NS(device=d, dtype=t, shard_map=s) for n, (d, t, s, w, a, o) in comps.items()},
              component_memory={n: NS(weight_bytes=w, activation_bytes=a, overhead_bytes=o)
                                for n, (d, t, s, w, a, o) in comps.items()},
              loading_mode=loading)


def _never(k):
    return False


def test_a_unified_load_at_the_stored_dtype_holds_no_host_copy():
    """canary-qwen-2.5b: three bf16 components planned bf16 on the device; the old figure priced each pass as read +
    pinned (9 666 MB eager) where the run's whole footprint was 5 352 MiB.
    Injection: the unified branch removed (the discrete pass kept) -> transient 2 x stored, RED."""
    plan = _plan("eager", perception=("mps:0", "bfloat16", {}, 0, 0, 0), llm=("mps:0", "bfloat16", {}, 0, 0, 0))
    shards = {"perception": {"s0": 1551 * MB}, "llm": {"s0": 2048 * MB, "s1": 640 * MB}}
    stored = {"perception": {"bfloat16"}, "llm": {"bfloat16"}}
    f = H.host_footprint(plan, {}, shards, "compiled", None, DT, _never, stored, unified=True)
    assert f["transient_bytes"] == 0


def test_a_unified_load_that_converts_holds_the_converted_copy_of_one_component():
    """The converted copy is anon memory, freed when its component's load returns: the dearest one, eager or lazy.
    Injection: summed over components (the pinned allocator's rule) -> 1 500 MB, RED."""
    for loading in ("eager", "lazy"):
        plan = _plan(loading, te=("mps:0", "float16", {}, 0, 0, 0), vae=("mps:0", "float16", {}, 0, 0, 0))
        shards = {"te": {"s0": 2000 * MB}, "vae": {"s0": 1000 * MB}}
        f = H.host_footprint(plan, {}, shards, "compiled", None, DT, _never,
                             {"te": {"float32"}, "vae": {"float32"}}, unified=True)
        assert f["transient_bytes"] == 1000 * MB, loading


def test_a_host_computed_component_on_a_unified_profile_is_held_once():
    """Voxtral-Mini-3B-2507 cpu_execution: the language model (7 656 MB bf16) computes on the host in fp32; the old
    figure held 14 934 MB AND passed 15 313 MB through the host for it; the run's peak was 16 110 MiB.
    Injection: a host component's pass priced as on a card -> transient = its converted copy, RED."""
    plan = _plan("lazy", language_model=("cpu", "bfloat16", {}, 7656 * MB, 7000 * MB, 300 * MB))
    shards = {"language_model": {f"s{i}": 1914 * MB for i in range(4)}}
    f = H.host_footprint(plan, {}, shards, "compiled", None, DT, _never, {"language_model": {"bfloat16"}},
                         unified=True)
    assert f["steady_bytes"] == 2 * 7656 * MB + 7300 * MB
    assert f["transient_bytes"] == 0


def test_a_host_placed_component_is_not_also_in_the_unified_device_term():
    """PixArt-XL-2-1024-MS cpu_streaming: every component on the host; the old figure held 18 815 MB for the text
    encoder AND counted a 9 732 MB device plan; max RSS 16 619 MiB.
    Injection: `host_bytes` ignored -> the device term keeps the host components, RED."""
    unified = NS(devices=[NS(has_unified_memory=True)])
    eager = NS(total_memory_mb=16166.0, device_window_mb=None, loading_mode="eager")
    assert unified_device_bytes(eager, unified, host_bytes=16166 * MB) == 0
    assert unified_device_bytes(eager, unified, host_bytes=6000 * MB) == 10166 * MB
    lazy = NS(total_memory_mb=16166.0, device_window_mb=None, loading_mode="lazy")
    assert unified_device_bytes(lazy, unified, 0, host_bytes=16166 * MB) == 0      # the caller's peak, host excluded
    assert unified_device_bytes(eager, NS(devices=[NS(has_unified_memory=False)]), host_bytes=6000 * MB) == 0


def test_a_discrete_profile_prices_byte_identically():
    """The figure a discrete card's plan priced before this change (frozen from the code at 0eee3f5b): host compute,
    a zero3 share, a converting and a same-dtype component, eager and lazy.
    Injection: the unified rules applied without the declaration -> another number, RED."""
    comps = {"te": ("cpu", "float16", {}, 300 * MB, 40 * MB, 8 * MB),
             "lm": ("cuda:0", "float16", {"s0": "cuda:0", "s1": "cpu"}, 600 * MB, 0, 0),
             "vae": ("cuda:0", "float16", {}, 80 * MB, 0, 0)}
    shards = {"te": {"s0": 600 * MB}, "lm": {"s0": 300 * MB, "s1": 300 * MB}, "vae": {"s0": 160 * MB}}
    stored = {"te": {"float32"}, "lm": {"bfloat16"}, "vae": {"float32"}}
    frozen = {"eager": (4096786432, 994050048, 2453667840), "lazy": (2586836992, 679477248, 1258291200)}
    for loading, (total, steady, transient) in frozen.items():
        f = H.host_footprint(_plan(loading, **comps), {"te": {}, "lm": {}, "vae": {}}, shards, "compiled", 512, DT,
                             _never, stored, resident_bytes=100 * MB, output_bytes=7 * MB)
        assert (f["total_bytes"], f["steady_bytes"], f["transient_bytes"]) == (total, steady, transient), loading
