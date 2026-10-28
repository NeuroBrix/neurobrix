"""A streamed LLM's plan — segments, what stays beside them, their constants AND the KV cache —
fits the usable part of the rung it was cut against.

Three defects, one plan. On main (fa6e13d2), the Mac's profile, host idle at 18 186 MB free:

* the post-scoring KV check judged `layer_streaming` against the SUM of every device's raw
  capacity, not the figure its segments were cut against, and subtracted the KV estimate from a
  streamed segment peak that never contained it. The cache was then sized from memory the rung
  does not grant:
      DeepSeek-Coder-V2-Lite, serve   13 598.8 + KV 3 846.4 = 17 445 MB
      Qwen3-Coder-30B-A3B,   serve   11 966.3 + KV 8 394.4 = 20 361 MB
  against a 16 384 MB rung, 15 073.3 usable, 17 276.7 capacity — past the device itself;
* the segment reserve was the RUN-mode KV estimate while a serve plan demands two turns
  (`_kv_min_tokens`), so closing the first defect alone REFUSED Qwen3 in serve mode;
* the reserve's dtype was `getattr(self, "_target_dtype_str", "float16")` — invented when
  absent — and after the fp32 fallback `solve` updated a local, not the attribute, so the
  re-evaluated strategies reserved an fp32 cache at 2 bytes.

The machine is built (register 102): the Mac's profile and its idle reading, no door.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism import InputConfig, PrismSolver
from neurobrix.core.prism.solver import PlanNotComputable, _graph_constant_bytes
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import (APPLE_M4_PRO, V100_16GB, container_root, no_door,
                                              pin_host, pin_no_calibration, pin_shared_card, profile)


@pytest.fixture(autouse=True)
def _no_machine_calibration(monkeypatch):
    """Every cell plans without the machine's calibration store (`pin_no_calibration`): a record there moves the
    compiled widths, and this file went red on the rack only (2026-10-28)."""
    pin_no_calibration(monkeypatch)

MB = 1024 * 1024
LLMS = ["DeepSeek-Coder-V2-Lite-Instruct", "Qwen3-Coder-30B-A3B-Instruct"]


def _solve(monkeypatch, model, serve):
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    s = PrismSolver()
    seen, evals = {}, []
    real_mem, real_eval = s._compute_memory, s._evaluate_all_strategies

    def spy_mem(*a, **k):
        out = real_mem(*a, **k)
        seen.update(out)
        return out

    def spy_eval(*a, **k):
        evals.append(a)
        return real_eval(*a, **k)

    s._compute_memory = spy_mem
    s._evaluate_all_strategies = spy_eval
    c = NBXContainer.load(str(container_root(model)))
    p = s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1), serve_mode=serve,
                      mode="triton")
    return p, s, seen, c, evals


def _judge_expert_store(p, s, seen, c, model, dev):
    """A routed LM held as an expert store on the unified device (`_place_component` Strategy 2.5):
    its dense part and its expert cache as granted, its graph constants, at least its profiled activations (without the
    KV estimate the check sizes itself) and its overhead, every other component whole, the cache as
    planned — within the usable part of the rung the store was placed against. A lower bound on what
    the plan holds, read from the plan and the component memory, never from the solver's own sums."""
    lm = s._lm_component_name
    a = p.components[lm]
    m = seen[lm]
    kv_est = s._estimate_kv_cache_bytes(c, s._target_dtype_str)
    store = (m.activation_bytes - kv_est + m.overhead_bytes
             + (a.resident_weight_mb + a.expert_cache_mb) * MB)
    beside = sum(q.total_bytes for n, q in seen.items() if n != lm)
    consts = _graph_constant_bytes(next(x.graph for x in c.get_neural_components() if x.name == lm),
                                   s._constants_store_fp64(dev))
    held = (store + beside + consts + p.kv_cache_plan.memory_bytes) / MB
    assert a.expert_cache_mb > 0 and a.resident_weight_mb > 0, (a.resident_weight_mb, a.expert_cache_mb)
    assert held <= s._usable_mb(dev) + 1e-6, (
        f"{model}: dense {a.resident_weight_mb:.1f} + expert cache {a.expert_cache_mb:.1f} + activations, "
        f"beside and KV = {held:.1f} MB over the usable {s._usable_mb(dev):.1f} MB of the rung")


@pytest.mark.parametrize("serve", [False, True], ids=["run", "serve"])
@pytest.mark.parametrize("model", LLMS)
def test_segments_resident_constants_and_cache_fit_the_usable_rung(monkeypatch, model, serve):
    """Whichever plan the rung yields — the routed LM streamed, or held as an expert store where its
    dense part, its widest `top_k` experts and the cache fit the usable rung (0a90722c, 53eeaec5) —
    what it holds fits the usable part of the rung it was cut or placed against."""
    p, s, seen, c, _ = _solve(monkeypatch, model, serve)
    dev = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    if p.strategy == "lazy_sequential" and p.components[s._lm_component_name].strategy == "expert_store":
        _judge_expert_store(p, s, seen, c, model, dev)
        return
    assert p.strategy == "layer_streaming", (
        f"{model} ({'serve' if serve else 'run'}) planned {p.strategy!r}; this cell judges a "
        f"streamed plan or an expert store — a refusal here is the second defect back")
    parts = s._layer_stream_partitions
    # Precondition, so this oracle never has to copy the solver's subtraction rule: the LM is the
    # streamed component, whose segment peak never held the cache; what stays beside it is whole.
    assert s._lm_component_name in parts, (s._lm_component_name, list(parts))
    beside = sum(m.total_bytes for n, m in seen.items() if n not in parts)
    held = (sum(q.peak_resident_bytes for q in parts.values()) + beside
            + s._layer_stream_constant_bytes + p.kv_cache_plan.memory_bytes) / MB
    assert held <= s._usable_mb(dev) + 1e-6, (
        f"{model}: segments + resident + constants + KV = {held:.1f} MB over the usable "
        f"{s._usable_mb(dev):.1f} MB of the rung the segments were cut against")


@pytest.mark.parametrize("model", LLMS)
def test_a_served_expert_store_is_held_at_the_rung_it_fits(monkeypatch, model):
    """A served cache takes every byte the KV check leaves, and the store's expert cache is granted
    after it (`_zero3_resident_budgets`). The check must price the store as its placement and its
    grant do — the usable part of the rung, its activations as the arena holds them — or the cache
    takes the difference out of the store's floor and the grant refuses the plan at every rung.
    DeepSeek-Coder-V2-Lite served on the Mac's idle profile descended 16 384 -> 4 096 and streamed
    (6e7dbff0: "usable 14120 MB - held 13154 MB", then 51.6 MB short at the arena's activations).

    Injection (seen red, then restored green): either figure put back — the check at the summed
    rung, or the store at its profiled activations — and the plan descends the ladder."""
    p, s, seen, c, _ = _solve(monkeypatch, model, True)
    refused = [r for r in (getattr(p, "unified_rungs_tried", None) or []) if r[1] == "refused"]
    if p.strategy == "layer_streaming" and not refused:
        pytest.skip(f"{model}: the store's floor and the served cache do not fit the first rung; streamed there")
    assert not refused, f"{model}: the served plan descended the ladder: {p.unified_rungs_tried}"
    assert p.components[s._lm_component_name].strategy == "expert_store", p.strategy


def test_a_served_llm_on_a_shared_card_sizes_its_cache_on_the_rung_not_the_capacity(monkeypatch):
    """The KV check's non-streaming branches judged against the SUM of raw capacity. On a card
    another process holds 3 000 MB of, the rung sits well under the capacity, and a served cache —
    which takes every byte of `remaining` — was sized from the difference.

    The bound is independent of the solver's own cost arithmetic: whatever else it holds, every
    resident component holds at least its activations (the LM's without the KV estimate the check
    sizes itself), so cache + those activations must fit the rung."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 257530, 200000, "an idle rack host")
    pin_shared_card(monkeypatch, 16151, 3000, 306, "a V100-16GB another process holds 3 000 MB of")
    s = PrismSolver()
    c = NBXContainer.load(str(container_root("DeepSeek-Coder-V2-Lite-Instruct")))
    p = s.solve_smart(c, profile(V100_16GB), InputConfig(batch_size=1), serve_mode=True, mode="triton")
    # The branch under test is the KV check's GENERAL one (sum of the strategy's device rungs);
    # lazy_sequential is what wins here. One device, so the sum over devices IS this card's rung —
    # a multi-card host (where the sum overstates the LM's own card) is not covered by this cell.
    # Seen failing with raw capacity injected into that branch (2026-09-24, injection round 4).
    assert p.strategy == "lazy_sequential", p.strategy
    dev = s._prepare_devices(profile(V100_16GB))[0]
    rung = s._effective_capacity_mb(dev)
    # Since 2026-10-25 (the owner, 15:16) the other process's 3 000 MB do not shrink the budget — the card's
    # declared capacity sizes the plan and the moment is the execution door, which names the hold when the
    # plan does not fit what is free now. The bound below is the same: cache + activations inside the rung.
    # (the door itself — the plan's bytes on the card against what is free now — is pinned in
    # test_a_unified_plan_fits_its_host_side_in_the_free_memory.py and test_a_whole_plan_is_budgeted_against_the_pools_fact.py)
    kv_est = s._estimate_kv_cache_bytes(c, s._target_dtype_str)
    floor = sum(m.activation_bytes - (kv_est if n == s._lm_component_name else 0)
                for n, m in p.component_memory.items())
    held = (floor + p.kv_cache_plan.memory_bytes) / MB
    assert held <= rung + 1e-6, (f"{p.strategy}: cache {p.kv_cache_plan.memory_bytes / MB:.1f} MB + "
                                 f"activations {floor / MB:.1f} MB = {held:.1f} over the {rung:.1f} rung")


def test_the_reserve_refuses_to_invent_a_dtype(monkeypatch):
    """With no dtype decided, the reserve is REFUSED — it was sized at an invented float16."""
    _, s, _, c, evals = _solve(monkeypatch, LLMS[0], False)
    sorted_comps, comp_mem, devices, shard_sizes, prof, container = evals[0][1:7]
    del s._target_dtype_str
    with pytest.raises(RuntimeError, match="ZERO FALLBACK"):
        s._try_layer_streaming(sorted_comps, comp_mem, devices, shard_sizes, prof, container)


def test_after_the_fp32_fallback_the_reserve_and_the_plan_read_fp32(monkeypatch):
    """Force the fallback path: the first evaluation yields nothing, the fallback applies. Every
    KV figure taken after it must be taken at float32.

    "After" is read from the fallback itself, not from an evaluation count: the rungs between the
    first evaluation and the fallback (the token-axis slices, `_split_token_axis`) re-evaluate at
    the dtype that just failed, as they must."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    # the planning process's own resident memory, pinned: it enters the plan's host side, and after the file's earlier
    # cells loaded their containers the rack's test process held enough to fire the execution door — a busy re-plan
    # this stub of the fallback (it sets no dtype) was never meant to reach; red in file order on the rack only
    import neurobrix.core.prism.host_footprint as _HF
    monkeypatch.setattr(_HF, "process_footprint_now", lambda: 200 << 20)
    s = PrismSolver()
    calls = {"n": 0}
    real_eval = s._evaluate_all_strategies

    def first_empty(*a, **k):
        calls["n"] += 1
        return [] if calls["n"] == 1 else real_eval(*a, **k)

    dtypes = []
    fell = {"back": False}
    real_est, real_min, real_plan = s._estimate_kv_cache_bytes, s._kv_min_bytes, s._compute_kv_cache_plan
    s._evaluate_all_strategies = first_empty

    def fall_back(*a, **k):
        fell["back"] = True
        return True
    s._try_fp32_fallback = fall_back
    s._estimate_kv_cache_bytes = lambda c, d: (dtypes.append(("estimate", fell["back"], d)), real_est(c, d))[1]
    s._kv_min_bytes = lambda c, d: (dtypes.append(("minimum", fell["back"], d)), real_min(c, d))[1]
    s._compute_kv_cache_plan = lambda c, d, r: (dtypes.append(("plan", fell["back"], d)), real_plan(c, d, r))[1]
    s.solve_smart(NBXContainer.load(str(container_root(LLMS[0]))), profile(APPLE_M4_PRO),
                  InputConfig(batch_size=1), mode="triton")
    assert fell["back"], "the fallback was never reached"
    after = [(what, d) for what, back, d in dtypes if back]
    assert any(w == "minimum" for w, _ in after) and any(w == "plan" for w, _ in after), dtypes
    assert all(d == "float32" for _, d in after), after


def test_every_rung_starts_unsliced(monkeypatch):
    """`solve` re-enters `_solve_at_rung` down the unified ladder; a token-axis slicing a previous
    rung kept (`_split_token_axis` sets `_token_split`) must not reach the next rung's first
    evaluation, which tries the cuts that cost nothing first.

    Injection (seen red, then restored green): the reset before the first evaluation removed ->
    the first evaluation runs sliced."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    s = PrismSolver()
    s._token_split = True                       # what a previous rung that sliced left behind
    seen = []
    real_eval = s._evaluate_all_strategies

    def record(*a, **k):
        seen.append(bool(getattr(s, "_token_split", False)))
        return real_eval(*a, **k)
    s._evaluate_all_strategies = record
    s.solve_smart(NBXContainer.load(str(container_root(LLMS[0]))), profile(APPLE_M4_PRO),
                  InputConfig(batch_size=1), mode="triton")
    assert seen and seen[0] is False, seen


@pytest.mark.parametrize("serve", [False, True], ids=["run", "serve"])
def test_a_package_without_its_decoder_facts_is_refused_not_planned_without_a_cache(
        monkeypatch, tmp_path, serve):
    """The Mac, 4c29e0e7: its Qwen3-Coder-30B-A3B package had no runtime/defaults.json. The plan
    still went out — layer_streaming, `kv_cache_plan=None`, in run and serve — because
    `_compute_kv_cache_plan` answers None when it reads no decoder facts, and every check that
    subtracts the KV estimate read 0. A decoder whose cache cannot be sized has no plan: refused,
    naming the file. The package is the rack's own, every file linked but that one.

    Injection (seen red, then restored green): the refusal removed -> a plan with no cache."""
    src = container_root(LLMS[1])
    pkg = tmp_path / LLMS[1]
    pkg.mkdir()
    for entry in src.iterdir():
        if entry.name != "runtime":
            (pkg / entry.name).symlink_to(entry)
    (pkg / "runtime").mkdir()
    for entry in (src / "runtime").iterdir():
        if entry.name != "defaults.json":
            (pkg / "runtime" / entry.name).symlink_to(entry)
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    s = PrismSolver()
    c = NBXContainer.load(str(pkg))
    try:
        p = s.solve_smart(c, profile(APPLE_M4_PRO), InputConfig(batch_size=1), serve_mode=serve,
                          mode="triton")
    except PlanNotComputable as refusal:
        assert "defaults.json" in str(refusal) and "absent" in str(refusal), str(refusal)
        return
    pytest.fail(f"planned {p.strategy!r} with kv_cache_plan={p.kv_cache_plan!r} from a package whose "
                f"decoder facts are absent")
