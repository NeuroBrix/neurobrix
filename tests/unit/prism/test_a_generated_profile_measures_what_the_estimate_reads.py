"""S2 (the supervisor, 2026-10-27 22:20 and the decision of 23:00): a machine whose GENERATED profile and architecture
vendor file state no measured facts is probed at its first launch (`autodetect.ensure_measured_facts`, through
`facts_probe`, the measuring tool's own primitives) for the device and host facts; the engine's LOADER rate is measured
by the loader itself at its first real load (`note_loader_rate`), cold then warm — until then Prism ranks without an
estimate and says so.

What each test would do if the code were wrong: a generated profile left unprobed (red); a committed profile written
(red); a fact the profile or its vendor file already states measured again (red); a loader rate kept after its warm
reading, or written for a load too small to time (red)."""
from __future__ import annotations

import json

import pytest
import yaml

from neurobrix.pytorch.prism import autodetect as A
from neurobrix.pytorch.prism import owed_facts as OF
from neurobrix.pytorch.prism import facts_probe as P
from nbxtensor.backend import host_measure as H
from neurobrix.pytorch.prism.kept_launch_facts import card_key

CALLS = []


@pytest.fixture
def hw(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "HARDWARE_DIR", tmp_path)
    CALLS.clear()
    monkeypatch.setattr(P, "matmul_tflops", lambda t, *a, **k: CALLS.append(("mm", t)) or {"best": 10.0})
    monkeypatch.setattr(P, "copies", lambda *a, **k: CALLS.append(("copies",)) or
                        {"memory_gbps": 500.0, "h2d_gbps": 10.0, "d2h_gbps": 9.0, "h2d_pageable_gbps": 5.0})
    monkeypatch.setattr(H, "host_matmul_gflops", lambda *a, **k: CALLS.append(("hmm",)) or 300.0)
    monkeypatch.setattr(H, "host_memory_gbps", lambda *a, **k: CALLS.append(("hmem",)) or 20.0)
    monkeypatch.setattr(P, "sweep_chunk", lambda *a, **k: CALLS.append(("vmm",)) or (2 << 20, None))
    monkeypatch.setattr(P, "ruled_chunk", lambda *a, **k: None)   # a class the library's rule does not answer: swept
    monkeypatch.setattr(P, "vmm_first_replay", lambda *a, **k: None)      # no VMM: nothing replayed ahead
    # the runtime's total and own context on each card (255bc661d), read in a child on the real card: mocked like the
    # other probes, so the expected keys do not depend on whether this host has a card
    monkeypatch.setattr(P, "card_facts_in_a_child", lambda index: CALLS.append(("card", index)) or
                        {"runtime_total_mb": 16000, "runtime_context_mb": 300})
    # the launched engine's runtime base (line 3 of the ruling of 22/11/2026 17:43): its own child, not this test's
    monkeypatch.setattr(A, "_measure_runtime_base_mb", lambda engine: 800)
    return tmp_path


def _profile(hw, name, device):
    (hw / f"{name}.yml").write_text(yaml.safe_dump({"id": name, "devices": [device, dict(device)]}))


DEV = {"brand": "nvidia", "model": "Unknown-Card", "architecture": "unknownarch", "supports_dtypes": ["float16", "float32"]}


def test_a_generated_profile_without_facts_is_probed_once_and_written(hw):
    _profile(hw, "default-abc", DEV)
    # the host's GEMM and copy wait for a host-held plan's pricing (line 5 of the ruling of 22/11/2026 17:43)
    assert set(A.ensure_measured_facts("default-abc")) == {"measured", "vmm_chunk_bytes", "runtime_card_facts"}
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    for d in data["devices"]:
        assert d["runtime_total_mb"] == 16000 and d["runtime_context_mb"] == 300
        assert d["measured"]["matmul_tflops"] == {"float16": 10.0, "float32": 10.0} and d["measured"]["h2d_gbps"] == 10.0
        assert "first launch" in d["measured"]["source"] and "weight_load_gbps" not in d["measured"]
        assert d["measured"]["vmm_chunk_bytes"] == 2 << 20
    assert "host_measured" not in data and not any(c[0] in ("hmm", "hmem") for c in CALLS)
    assert OF.measure_host_facts("default-abc") == ["host_matmul_gflops", "host_memory_gbps"]
    assert yaml.safe_load((hw / "default-abc.yml").read_text())["host_measured"]["host_memory_gbps"] == 20
    CALLS.clear()
    assert A.ensure_measured_facts("default-abc") == {} and CALLS == []


def test_a_committed_profile_is_never_written(hw, tmp_path):
    _profile(hw, "c4140-something", DEV)
    before = (hw / "c4140-something.yml").read_text()
    assert A.ensure_measured_facts("c4140-something") == {}
    assert not A.note_loader_rate("c4140-something", "triton", "bfloat16", "device", 1 << 30, 1.0, "m:c")
    assert (hw / "c4140-something.yml").read_text() == before and CALLS == []


def test_what_the_vendor_file_states_is_not_probed_again(hw):
    """A V100 (volta.yml states the device facts): its device facts are not probed again (the host's, which volta.yml
    does not state, wait for a host-held plan's pricing since the ruling of 22/11/2026 17:43, line 5)."""
    _profile(hw, "default-v100", {"brand": "nvidia", "model": "Tesla V100-SXM2-16GB", "architecture": "volta",
                                  "compute_capability": "7.0", "supports_dtypes": ["float16", "float32"]})
    # the expandable pool's chunk is the machine's measure, never the vendor file's (R16)
    # and so is the runtime's own context on the card, which no vendor file states
    assert set(A.ensure_measured_facts("default-v100")) == {"vmm_chunk_bytes", "runtime_card_facts"}
    assert not any(c[0] in ("mm", "copies") for c in CALLS), CALLS


def test_the_loader_measures_itself_cold_then_warm_then_stops(hw):
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc")
    assert not A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 1 << 20, 0.01, "m:c"), "too small"
    assert A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 2 * 10 ** 9, 2.0, "m:c")
    assert A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 4 * 10 ** 9, 2.0, "m:c")
    assert not A.note_loader_rate("default-abc", "triton", "bfloat16", "device", 4 * 10 ** 9, 1.0, "m:c")
    d = yaml.safe_load((hw / "default-abc.yml").read_text())["devices"][1]["measured"]
    assert d["weight_load_gbps"]["triton"]["BF16"]["device"] == {"cold": 1.0, "warm": 2.0}
    assert d["matmul_tflops"]["float16"] == 10.0, "the probe's facts kept beside the loader's"


def test_the_engine_measures_its_per_token_reload_once_on_a_generated_profile(hw):
    """The supervisor 2026-11-02 06:40 (B): the per-token reload rate a layer_streaming plan pays is a profile fact the
    engine measures itself at a real reload (`note_reload_rate`), per engine and stored dtype, written once into a
    GENERATED profile's host_measured; a committed profile is never written, a reload too small to time is not kept.
    What this test would do if the code were wrong: a second measure overwriting the first, a committed profile written,
    or the rate filed under another key than the estimator reads (time_estimate.reload_bps) — RED."""
    from neurobrix.pytorch.prism import time_estimate as T
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc")
    assert not A.note_reload_rate("default-abc", "compiled", "bfloat16", 1 << 20, 0.01, "m:c"), "too small"
    assert A.note_reload_rate("default-abc", "compiled", "bfloat16", 2 * 10 ** 9, 2.0, "m:c")
    assert not A.note_reload_rate("default-abc", "compiled", "bfloat16", 8 * 10 ** 9, 2.0, "m:c"), "written once"
    assert A.note_reload_rate("default-abc", "triton", "bfloat16", 3 * 10 ** 9, 2.0, "m:c")
    host = yaml.safe_load((hw / "default-abc.yml").read_text())["host_measured"]
    assert host["weight_reload_gbps"] == {"compiled": {"BF16": 1.0}, "triton": {"BF16": 1.5}}
    assert abs(T.reload_bps(T.host_facts(host), "compiled", "bfloat16") - 1e9) < 1
    _profile(hw, "c4140-something", DEV)
    assert not A.note_reload_rate("c4140-something", "compiled", "bfloat16", 2 * 10 ** 9, 2.0, "m:c")


def test_an_unsteady_reload_is_kept_at_its_low_draw_until_a_steady_one(hw, monkeypatch):
    """The supervisor 2026-11-02 09:50: a profile fact a neighbour can disturb is no fact — so three runs that disagree
    were never written, and every later launch on a busy machine probed again (the catalogue's kept homes, 2026-11-19:
    TinyLlama owed BF16 three launches in a row). The orchestrator's ruling (2026-11-19): the planner needs the figure
    before the run, so the measure is KEPT with its spread and date and priced at its low draw; the first steady three
    replace it; an unsteady one never replaces a kept figure. What this test would do if the code were wrong: nothing
    kept after an unsteady three (red: the probe owed again), the high or median draw priced (red), a steady measure
    not replacing it (red), a second unsteady measure moving it (red)."""
    from neurobrix.core.strategies import layer_streaming as LS
    _profile(hw, "default-abc", DEV)
    A.ensure_measured_facts("default-abc")
    monkeypatch.setattr(A, "ACTIVE_HARDWARE_ID", "default-abc")
    monkeypatch.setattr(A, "stored_major", lambda *a: "bfloat16")
    monkeypatch.setattr(LS, "_RELOAD_DRAWS", {})
    ex = type("E", (), {"mode": "compiled"})()

    def host():
        return yaml.safe_load((hw / "default-abc.yml").read_text()).get("host_measured") or {}

    def written():
        return (host().get("weight_reload_gbps") or {}).get("compiled", {}).get("BF16")

    for b, s in ((10 ** 9, 1.0), (10 ** 9, 2.0), (10 ** 9, 1.0)):       # 1, 0.5, 1 GB/s: not steady
        LS._note_reload(ex, "/x", "lm", b, s)
    assert written() == 0.5, written()
    spread = host()["weight_reload_spread"]["compiled"]["BF16"]
    assert spread["low"] == 0.5 and spread["high"] == 1.0 and spread["measured"]
    LS._note_reload(ex, "/x", "lm", 10 ** 9, 4.0)                        # 0.5, 1, 0.25: unsteady again, kept as is
    assert written() == 0.5
    for b, s in ((10 ** 9, 1.0), (10 ** 9, 1.02), (10 ** 9, 0.99)):      # steady around 1 GB/s: replaces it
        LS._note_reload(ex, "/x", "lm", b, s)
    assert written() is not None and abs(written() - 0.98) < 0.03, written()
    assert "weight_reload_spread" not in host()
    # the launch's check finds it kept: no reload probe owed for this engine and stored dtype
    assert A.STORED_DTYPES["bfloat16"] in host()["weight_reload_gbps"]["compiled"]


def _decide(reload_gbps):
    """The faster of two plans for one generating component, as Prism ranks them (`time_estimate.plan_seconds`): it
    streamed (a quarter of it reloaded at the second use, at `reload_gbps`) or loaded on demand by each of the three
    phases that use it (lazy: 4 GB cold, then twice warm)."""
    from neurobrix.pytorch.prism import time_estimate as T
    from neurobrix.pytorch.prism.work import Work
    gb = 10 ** 9
    dev = T.DeviceFacts(matmul_flops={"float16": 100e12}, memory_bps=800 * gb, h2d_bps=10 * gb,
                        load_rates={"compiled": {"BF16": {"device": {"cold": 1.0, "warm": 2.0}}}})
    host = T.HostFacts(matmul_flops=1e12, memory_bps=10 * gb, reload_gbps={"compiled": {"BF16": reload_gbps}})
    works, uses = {"lm": Work(flops=10 ** 9, weight_bytes=4 * gb)}, {"lm": 2}
    kw = dict(engine="compiled", stored={"lm": "bfloat16"}, held={"lm": 0.75}, phases={"lm": 3})
    plans = {s: T.plan_seconds(s, {"lm": "cuda:0"}, works, uses, {"cuda:0": dev}, host, "float16", **kw)
             for s in ("layer_streaming", "lazy_sequential")}
    return min(plans, key=plans.get)


def test_a_plan_from_the_kept_figure_holds_across_its_spread():
    """The ruling's proof: plans made from the low draw and from the high draw take the same decision, or differ — and
    then the kept figure (the low draw, by rule) takes the safe one, never a streamed plan only the fast draw ranks
    first. Two spreads: one where both draws stream, one where only the high draw would."""
    assert _decide(4.0) == _decide(6.0) == "layer_streaming"            # same decision across the spread
    low, high = 0.2, 6.0
    assert _decide(high) == "layer_streaming" and _decide(low) == "lazy_sequential"
    kept = min(low, high)                                              # what note_reload_rate(spread=) writes
    assert _decide(kept) == _decide(low) == "lazy_sequential"          # the safe one: no reload promised at the fast draw


def test_a_profile_s_host_facts_lie_over_its_vendor_s_key_by_key():
    """The Dell's 1.12 rule (2026-11-03 22:54): the vendor's host_measured (the architecture's) under the profile's (this
    machine's, measured by the engine), key by key into its tables. What this would do if the profile's block replaced the
    vendor's (the rule before): the vendor's host_slab_gbps and its triton reload rate lost — RED."""
    from neurobrix.pytorch.prism.loader import _over
    vendor = {"host_slab_gbps": 3.0, "host_matmul_gflops": 100, "weight_reload_gbps": {"triton": {"BF16": 2.1}}}
    prof = {"host_matmul_gflops": 300, "weight_reload_gbps": {"compiled": {"BF16": 1.0}, "triton": {"F32": 0.5}}}
    assert _over(vendor, prof) == {"host_slab_gbps": 3.0, "host_matmul_gflops": 300,
                                   "weight_reload_gbps": {"triton": {"BF16": 2.1, "F32": 0.5}, "compiled": {"BF16": 1.0}}}
    assert _over(None, prof) == prof and _over(vendor, None) == vendor


def _container(root, name, comp, dtype, nbytes):
    c = root / name / "components" / comp
    c.mkdir(parents=True)
    (root / name / "manifest.json").write_text("{}")
    (c / "graph.json").write_text("{}")
    (c / "weights_index.json").write_text(json.dumps({"tensors": {"w": {"dtype": dtype, "size_bytes": nbytes}}}))
    with open(c / "w.safetensors", "wb") as f:
        f.truncate(nbytes)                       # sparse: the size is what the probe reads


def test_a_streamed_pricing_reloads_the_smallest_component_of_each_unstated_stored_dtype(hw, tmp_path, monkeypatch):
    """The supervisor 2026-11-03 22:50: the per-piece reload rate is measured by the engine (since the ruling of
    22/11/2026 17:43, line 4: when Prism prices a streamed candidate, not at the first launch), per
    stored dtype of the cache neither block states, on the smallest component stored so (at least the loader-rate floor)
    — before, only a streamed run measured it and Prism refused to plan one without it. What this would do if the code
    were wrong: no probe (the fact never measured), a component under the floor or not the smallest, or a stated rate
    measured again — RED."""
    from neurobrix.pytorch.prism import arena_probe as AP
    import neurobrix.core.paths as paths
    cache = tmp_path / "cache"
    floor = A.LOADER_RATE_MIN_BYTES
    _container(cache, "big", "lm", "bfloat16", 4 * floor)
    _container(cache, "small", "lm", "bfloat16", 2 * floor)
    _container(cache, "tiny", "lm", "bfloat16", floor // 2)
    _container(cache, "f32", "net", "float32", 3 * floor)
    _container(cache, "huge", "lm", "float16", 64 << 30)        # larger than the card: never reloaded whole
    monkeypatch.setattr(paths, "cache_dir", lambda: cache)
    monkeypatch.setattr(P, "gemv_gbps", lambda *a, **k: 100.0)          # the other first-launch probes: not this test's
    probed = []
    monkeypatch.setattr(AP, "child", lambda d, c, h, e: probed.append((d.rsplit("/", 1)[1], c, e)) or
                        {"container": d, "component": c, "draws": []})
    _profile(hw, "default-abc", {**DEV, "memory_mb": 16384})
    A.ensure_measured_facts("default-abc", "compiled")
    # line 4 of the ruling of 22/11/2026 17:43: the first launch owes no reload; Prism asks for it when it prices a
    # streamed candidate (`owed_facts.measure_owed` -> `measure_reload_rates`)
    assert probed == [], "reloaded at the first launch, before any streamed candidate was priced"
    OF.measure_reload_rates("default-abc", "compiled")
    assert sorted(probed) == [("f32", "net", "compiled"), ("small", "lm", "compiled")], probed
    data = yaml.safe_load((hw / "default-abc.yml").read_text())
    data.setdefault("host_measured", {})["weight_reload_gbps"] = {"compiled": {"BF16": 2.0, "F32": 1.0}}
    # kept under this driver and library (an unkeyed rate is of an older launch and measured again: drop_stale)
    data["host_measured"]["weight_reload_under"] = card_key()
    (hw / "default-abc.yml").write_text(yaml.safe_dump(data))
    probed.clear()
    OF.measure_reload_rates("default-abc", "compiled")
    assert probed == [], "a stated rate is not measured again"


def test_the_reload_counts_an_nbxtensor_s_bytes():
    """Flex.1-alpha triton, 2026-11-04: `_weights_bytes` took `nbytes` first, a METHOD on NBXTensor (ATen's API), and died
    on int(method) at the first streamed reload. What this would do if the order came back: TypeError — RED."""
    from neurobrix.core.strategies import layer_streaming as LS
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import parse_dtype
    ex = type("E", (), {"_weights": {"w": NBXTensor.zeros((4, 8), parse_dtype("float32"), "cpu"), "none": None}})()
    assert LS._weights_bytes(ex) == 4 * 8 * 4


def test_the_reload_counts_a_zero_size_nbxtensor_as_zero():
    """Sana's text encoder holds zero-size constants (shape (0,), `_nbytes` 0): `_nbytes or nbytes` fell through to the
    METHOD and 0.6.3's first launch from PyPI died on int(method) in the arena probe's reload (2026-11-09). Red if a
    count of 0 is read as absent."""
    from neurobrix.core.strategies import layer_streaming as LS
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import parse_dtype
    f32 = parse_dtype("float32")
    ex = type("E", (), {"_weights": {"w": NBXTensor.zeros((4, 8), f32, "cpu"), "c": NBXTensor.zeros((0,), f32, "cpu"),
                                     "_arenas": {0: object()}}})()       # the arenas the weights are views of: not counted
    assert LS._weights_bytes(ex) == 4 * 8 * 4


def test_a_probe_in_a_child_reads_the_facts_measured_before_it(hw, monkeypatch, tmp_path):
    """A child probe (the reload rate, the arena factor) reads the profile from disk, and the engine it starts refuses by
    name a fact the profile lacks: on a clean machine (the stable's install test, 2026-11-05) the first child's Triton
    pool refused the vmm_chunk_bytes measured in the parent and not yet written. Red if a child is started before the
    in-process facts are on disk."""
    from neurobrix.pytorch.prism import arena_probe as AP
    from neurobrix.core import paths
    cache = tmp_path / "cache"
    comp = cache / "M" / "components" / "c"
    comp.mkdir(parents=True)
    (comp / "w.safetensors").write_bytes(b"\0" * (A.LOADER_RATE_MIN_BYTES + 1))
    monkeypatch.setattr(paths, "cache_dir", lambda: cache)
    monkeypatch.setattr(A, "stored_major", lambda root, name: next(iter(A.STORED_DTYPES)))
    monkeypatch.setattr(P, "gemv_gbps", lambda *a, **k: 100.0)
    seen = []

    def child(root, component, hardware_id, engine):
        seen.append(yaml.safe_load((hw / f"{hardware_id}.yml").read_text())["devices"][0].get("measured", {}))
        return {"container": "M", "component": "c", "draws": [(1, 1.0)]}
    monkeypatch.setattr(AP, "child", child)
    _profile(hw, "default-abc", {**DEV, "memory_mb": 16384})
    A.ensure_measured_facts("default-abc", "compiled")
    OF.measure_reload_rates("default-abc", "compiled")        # asked by Prism when it prices a streamed candidate
    assert seen and all(m.get("vmm_chunk_bytes") == 2 << 20 for m in seen), seen
