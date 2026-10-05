"""No arena is planned above the largest single allocation the device grants, and a piece change
is priced with the arena the allocator keeps parked beside the next one.

ONE ALLOCATION PER ARENA. The Triton engine loads the weights of a component held whole, of one
streamed piece, and of a streamed component's base each as ONE buffer (`triton/weight_loader.py`:
`ComponentArena(total, dev)`, one `malloc_cuda(total)`). A Metal device holds `memory_mb` in all
and refuses any ONE buffer above `MTLDevice.maxBufferLength`, however much is free: 13 639 MB
against 18 186 MB on the M4 Pro. At 17.1 GB free Prism cut deepseek-moe-16b-chat in 3 pieces whose
first arena asked 14 661 558 272 bytes and the device refused it by SIZE: "GPU malloc failed
(error 1) ... driver_free=17785MB / driver_total=18186MB" (the Mac, 2026-10-04). The profile
publishes the fact (`DeviceSpec.max_allocation_mb`); the solver now holds every arena to it — a
component held whole is declined by name (`_arena_over_allocation`, said by the refusal), a piece is
cut below it (`LayerPartitioner.partition(max_arena_bytes=)`), and one door in the ranked loop
(`_arena_door`) makes a rung that forgot unable to win. The compiled engine loads one tensor per
weight and has no arena: it is not bounded.

THE PARKED ARENA. A freed arena at most the pool's cap (`memory.alloc_pool_parked_cap_fraction` of
the device, 0.25 on Apple) stays allocated in the Triton allocator's pool, and the next piece takes
it back only when it lies in [w, 2w] of that request. A discrete card's driver refuses what it cannot
serve and the pool is flushed and retried; a UNIFIED device serves the request beside it. Measured on
the Mac (same run): the last piece's 4.5 GB parked while the first piece's arena allocated at the
step's wrap-around, 13.5 GB live against an 11.3 GB plan. The cut now holds every change — the
incoming arena, the outgoing one where it parks and is not taken back, and the activations live
across the cut, the last piece to the first included — to the budget (`partition(parked_cap_bytes=)`,
`PrismSolver._parked_cap_mb`).

SEEN RED (each cell, on the commit before this rule and under its injection; the numbers are in the
assertion messages): deepseek-moe's capped cut was [14 048, 13 455, 3 332] MB — the first piece over
13 639 — and its 3 332 MB last piece parked beside the 14 048 MB first one; Janus-Pro-7B's language
model was held whole under a bound below its arena.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. python -m pytest -q \\
     tests/unit/prism/test_no_arena_is_planned_above_the_devices_largest_allocation.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from neurobrix.core.prism import PrismSolver
from neurobrix.core.prism.layer_partition import LayerPartitioner
from neurobrix.nbx import NBXContainer
from tests.unit.prism._pinned_machine import APPLE_M4_PRO, V100_16GB, container_root, no_door, pin_host, profile
from tests.unit.prism.test_layer_partition import _chain

TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS))
import trace_request as TR  # noqa: E402

MB = 1024 * 1024
#: this M4 Pro's MTLDevice.maxBufferLength in MB, as its profile publishes it (d441a237, 2026-10-04)
MAX_ALLOCATION_MB = 13639
APPLE_CAPPED = copy.deepcopy(APPLE_M4_PRO)
APPLE_CAPPED["id"] = "scenario-apple-m4-pro-18g-max-allocation"
APPLE_CAPPED["devices"][0]["max_allocation_mb"] = MAX_ALLOCATION_MB


def _plan(model: str, spec: dict, mode: str = "triton"):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    flag = {"triton": "--triton", "triton_sequential": "--triton-sequential", "compiled": "--compiled"}[mode]
    args = create_parser().parse_args(["run", "--model", model, *TR.derived_request(model), flag])
    c = NBXContainer.load(str(container_root(model)))
    man = c.get_manifest() or {}
    s = PrismSolver()
    p = s.solve_smart(c, profile(spec), request_input_config(args, man, man.get("family"), c.cache_path), mode=mode)
    return p, s


def _pieces(s) -> dict:
    """{streamed component: [each piece's weight arena, MB]} (bf16 on Apple: cost multiplier 1)."""
    return {n: [round(sg.weight_bytes / MB) for sg in part.segments]
            for n, part in (getattr(s, "_layer_stream_partitions", {}) or {}).items()}


def _changes(segments, parked_cap: float) -> int:
    """The allocator's own rule, written from `DeviceAllocator.free` / `_pool_take`, not from the
    partitioner: at each change the outgoing arena parks when it is at most the cap, and the incoming
    request takes it back only if it lies in [w, 2w]; the device then holds both and the activations
    live across the cut."""
    peak = 0
    for i, out in enumerate(segments):
        inc = segments[(i + 1) % len(segments)]
        parks = out.weight_bytes <= parked_cap
        taken = inc.weight_bytes <= out.weight_bytes <= 2 * inc.weight_bytes
        peak = max(peak, inc.weight_bytes + (out.weight_bytes if parks and not taken else 0)
                   + out.live_bytes_at_exit)
    return peak if len(segments) > 1 else 0


# ─────────────────────────────── the facts ───────────────────────────────

def test_the_profile_carries_the_bound_and_only_the_triton_engine_reads_it():
    assert profile(APPLE_CAPPED).devices[0].max_allocation_mb == MAX_ALLOCATION_MB
    assert profile(APPLE_M4_PRO).devices[0].max_allocation_mb is None
    s = PrismSolver()
    s._mode = "triton"
    capped = s._prepare_devices(profile(APPLE_CAPPED))[0]
    plain = s._prepare_devices(profile(APPLE_M4_PRO))[0]
    assert s._max_allocation_mb(capped) == MAX_ALLOCATION_MB and s._max_allocation_mb(plain) is None
    # the parked cap: a quarter of the unified device's memory (apple_silicon yml), none on a discrete card
    assert s._parked_cap_mb(plain, profile(APPLE_M4_PRO)) == 0.25 * 18186
    v100 = s._prepare_devices(profile(V100_16GB))[0]
    assert s._max_allocation_mb(v100) is None and s._parked_cap_mb(v100, profile(V100_16GB)) is None
    s._mode = "compiled"   # one tensor per weight: no arena, no Triton pool
    assert s._max_allocation_mb(capped) is None and s._parked_cap_mb(plain, profile(APPLE_M4_PRO)) is None


def test_the_cut_holds_every_piece_under_the_arena_bound():
    """Eight 10 MB blocks against a 100 MB budget are one piece; bounded at 25 MB, none is over it."""
    g = _chain(8, weight_mb=10)
    free = LayerPartitioner(g).partition(100 * MB)
    bound = LayerPartitioner(g).partition(100 * MB, max_arena_bytes=25 * MB)
    assert free.fits and len(free.segments) == 1, [s.weight_bytes / MB for s in free.segments]
    assert bound.fits and all(sg.weight_bytes <= 25 * MB for sg in bound.segments), \
        [sg.weight_bytes / MB for sg in bound.segments]
    one_op = LayerPartitioner(g).partition(100 * MB, max_arena_bytes=5 * MB)
    assert not one_op.fits and "largest" in one_op.refusal, one_op.refusal


def test_the_cut_holds_every_change_with_the_parked_arena():
    """Seven 10 MB blocks at 32 MB (each op holds its 1 MB input and its 1 MB output while it runs,
    `op_peak_curve`, + 30 MB of weights): the greedy cut is 30/30/10, and the 10 MB last piece parks
    beside the 30 MB first one at the wrap-around — over the budget. Priced, every change fits."""
    g = _chain(7, weight_mb=10)
    budget = 32 * MB
    greedy = LayerPartitioner(g).partition(budget)
    assert greedy.fits and _changes(greedy.segments, math.inf) > budget, \
        ([s.weight_bytes / MB for s in greedy.segments], _changes(greedy.segments, math.inf) / MB)
    priced = LayerPartitioner(g).partition(budget, parked_cap_bytes=math.inf)
    assert priced.fits and _changes(priced.segments, math.inf) <= budget, \
        ([s.weight_bytes / MB for s in priced.segments], _changes(priced.segments, math.inf) / MB)
    assert priced.peak_resident_bytes <= budget
    # a cap below the outgoing piece: it is freed to the driver, nothing parks, the greedy cut stands
    capped = LayerPartitioner(g).partition(budget, parked_cap_bytes=5 * MB)
    assert [s.weight_bytes for s in capped.segments] == [s.weight_bytes for s in greedy.segments]


def _weighted(weights_mb, act_mb):
    """One weighted op per entry; the first also writes `act_mb` of activation that is a graph
    output (live to the end, so carried across the wrap-around like a model's). The chain between
    the ops is empty, so the activations held WHILE an op runs (`op_peak_curve`) are the ones a cut
    carries: `act_mb`."""
    tensors = {"x0": {"shape": [0], "dtype": "bfloat16", "is_parameter": False},
               "carry": {"shape": [act_mb * MB // 2], "dtype": "bfloat16", "is_parameter": False}}
    ops, order, prev = {}, [], "x0"
    for i, w in enumerate(weights_mb):
        tensors[f"p{i}"] = {"shape": [w * MB // 2], "dtype": "bfloat16", "is_parameter": True, "weight_name": f"w{i}"}
        tensors[f"x{i+1}"] = {"shape": [0], "dtype": "bfloat16", "is_parameter": False}
        ops[f"o{i}"] = {"input_tensor_ids": [prev, f"p{i}"],
                        "output_tensor_ids": [f"x{i+1}"] + (["carry"] if i == 0 else [])}
        order.append(f"o{i}"); prev = f"x{i+1}"
    return {"tensors": tensors, "ops": ops, "execution_order": order, "output_tensor_ids": [prev, "carry"]}


def test_the_change_search_does_not_step_past_the_cuts_floor():
    """Weights 20/1/19 MB under 1 000 MB of activations, at a 1 021 MB budget: the greedy cut is
    [20+1][19], and the wrap-around parks the 19 MB piece beside the 21 MB one, 19 MB over. Stepping by
    that overshoot lands at 1 002 MB, under the 1 020 MB floor (activations + the 20 MB op), a refusal;
    AT the floor the cut is [20][1+19], each arena taken back by the next, 1 020 MB at every change.
    The class: Janus-Pro-7B's language model at the Mac's 4096 rung, 1 508 -> 1 427 MB under a 1 461 MB
    floor, where 1 461-1 470 MB fit (2026-10-04)."""
    g = _weighted([20, 1, 19], act_mb=1000)
    lp = LayerPartitioner(g)
    assert lp._cut_floor() == 1020 * MB
    greedy = lp.partition(1021 * MB)
    assert [s.weight_bytes // MB for s in greedy.segments] == [21, 19]
    assert _changes(greedy.segments, math.inf) == 1040 * MB
    priced = LayerPartitioner(g).partition(1021 * MB, parked_cap_bytes=math.inf)
    assert priced.fits, priced.refusal
    assert [s.weight_bytes // MB for s in priced.segments] == [20, 20]
    assert _changes(priced.segments, math.inf) <= 1021 * MB and priced.peak_resident_bytes <= 1021 * MB


def test_the_arena_door_refuses_a_plan_holding_one_component_whole_over_the_bound():
    s = PrismSolver()
    s._mode = "triton"
    s._get_component_dtype = lambda container, name: "bfloat16"
    dev = s._prepare_devices(profile(APPLE_CAPPED))[0]
    mem = {"big": SimpleNamespace(weight_mb=MAX_ALLOCATION_MB + 1.0), "small": SimpleNamespace(weight_mb=10.0)}
    whole = {"big": (dev.device_string, {}), "small": (dev.device_string, {})}
    why = s._arena_door(None, "lazy_sequential", whole, [dev], mem)
    assert why and "'big'" in why and "max_allocation_mb" in why, why
    # host-held weights and a streamed component are no arena of the device
    assert s._arena_door(None, "zero3", {"big": ("zero3:" + dev.device_string, {})}, [dev], mem) is None
    s._layer_stream_partitions = {"big": object()}
    assert s._arena_door(None, "layer_streaming", whole, [dev], mem) is None


# ─────────────────────────────── the measured cells ───────────────────────────────

def test_deepseek_moe_at_17_gb_free_cuts_every_piece_under_the_largest_allocation(monkeypatch):
    """The Mac's failure, the arena bound alone (pool off, so the parked rule moves nothing): 17.1 GB
    free, 3 pieces, the first one's arena refused by size."""
    no_door(monkeypatch)
    monkeypatch.setenv("NBX_ALLOC_POOL", "0")
    pin_host(monkeypatch, 24576, 17510, "the Mac at 17.1 GB free")
    p, s = _plan("deepseek-moe-16b-chat", APPLE_M4_PRO)
    assert p.strategy == "layer_streaming" and max(_pieces(s)["model"]) > MAX_ALLOCATION_MB, \
        f"the scenario's precondition: unbounded, a piece over the device's largest allocation {_pieces(s)}"
    p, s = _plan("deepseek-moe-16b-chat", APPLE_CAPPED)
    assert p.strategy == "layer_streaming", p.strategy
    assert max(_pieces(s)["model"]) <= MAX_ALLOCATION_MB, _pieces(s)
    assert "'model'" in (s._arena_declined.get("model") or ""), s._arena_declined


def test_deepseek_moe_prices_the_arena_parked_at_the_wrap_around(monkeypatch):
    """The Mac's second measurement: the last piece parked beside the first. With the pool on, every
    change of the cut — by the allocator's rule, written here — fits the window the plan announces."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 17510, "the Mac at 17.1 GB free")
    cap = 0.25 * 18186 * MB
    monkeypatch.setenv("NBX_ALLOC_POOL", "0")
    p0, s0 = _plan("deepseek-moe-16b-chat", APPLE_M4_PRO)
    part0 = s0._layer_stream_partitions["model"]
    assert _changes(part0.segments, cap) > part0.peak_resident_bytes, \
        f"the precondition: unpriced, a change above the announced peak {_pieces(s0)}"
    monkeypatch.setenv("NBX_ALLOC_POOL", "1")
    p, s = _plan("deepseek-moe-16b-chat", APPLE_M4_PRO)
    part = s._layer_stream_partitions["model"]
    assert p.strategy == "layer_streaming"
    assert _changes(part.segments, cap) <= part.peak_resident_bytes, (_pieces(s), _changes(part.segments, cap) / MB)
    assert p.device_window_mb <= s._usable_mb(s._prepare_devices(profile(APPLE_M4_PRO))[0]) + 1e-6


def test_a_component_whole_over_the_largest_allocation_is_streamed_and_named(monkeypatch):
    """Janus-Pro-7B on the idle Mac holds its language model whole. Under a bound one MB below that
    model's weights — the scenario derived from the container, not a figure — it cannot be one arena:
    the whole rungs decline it by name and it is streamed, every piece under the bound."""
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p, s = _plan("Janus-Pro-7B", APPLE_M4_PRO)
    lm = ((s._flow_topology(NBXContainer.load(str(container_root("Janus-Pro-7B")))).get("flow") or {})
          .get("generation") or {}).get("lm_component")
    assert p.strategy != "layer_streaming", p.strategy
    lm_mb = p.component_memory[lm].weight_bytes / MB
    spec = copy.deepcopy(APPLE_M4_PRO)
    spec["devices"][0]["max_allocation_mb"] = int(lm_mb) - 1
    p, s = _plan("Janus-Pro-7B", spec)
    assert p.strategy == "layer_streaming" and lm in _pieces(s), (p.strategy, _pieces(s))
    assert max(_pieces(s)[lm]) <= int(lm_mb) - 1, _pieces(s)
    assert f"'{lm}'" in (s._arena_declined.get(lm) or "") and "max_allocation_mb" in s._arena_declined[lm]


def test_the_door_alone_keeps_a_rung_that_forgot_from_winning(monkeypatch):
    """Every rung made to forget the bound (`_arena_over_allocation` answers None to all but the door):
    the door in the ranked loop still rejects each plan holding the language model whole over it — the
    plan refuses naming the bound, or does not hold the model whole on the card."""
    import sys as _sys
    no_door(monkeypatch)
    pin_host(monkeypatch, 24576, 18186, "the Mac, idle")
    p, s = _plan("Janus-Pro-7B", APPLE_M4_PRO)
    lm = ((s._flow_topology(NBXContainer.load(str(container_root("Janus-Pro-7B")))).get("flow") or {})
          .get("generation") or {}).get("lm_component")
    spec = copy.deepcopy(APPLE_M4_PRO)
    spec["devices"][0]["max_allocation_mb"] = int(p.component_memory[lm].weight_bytes / MB) - 1
    real = PrismSolver._arena_over_allocation

    def forgetful(self, *a, **k):
        return real(self, *a, **k) if _sys._getframe(1).f_code.co_name == "_arena_door" else None
    monkeypatch.setattr(PrismSolver, "_arena_over_allocation", forgetful)
    try:
        p, s = _plan("Janus-Pro-7B", spec)
    except RuntimeError as refusal:
        assert "max_allocation_mb" in str(refusal), str(refusal)[-1500:]
        return
    dev = p.components[lm].device
    assert (p.strategy == "layer_streaming" and lm in p.layer_stream_plan) or not str(dev).startswith("mps"), \
        (p.strategy, dev)
