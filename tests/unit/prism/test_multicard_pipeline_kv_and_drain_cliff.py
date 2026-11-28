"""Multi-card pipeline_parallel KV cache sizing, drain-cliff clamping, and graph-constant device preservation.

Covers three interacting defects that caused OOMs on multi-GPU pipeline_parallel decoders:
1. Per-card KV budget: summing free memory across cards let an imbalanced heavier card (holding
   embeddings / lm_head / auxiliary components beside its layers) borrow headroom from a lighter
   card, over-sizing the KV cache for the heavier card's layers.
2. Multi-card Triton drain cliff: `_assign_drain_cliffs` and `_kv_card_room` skipped multi-device
   allocations (`len(devices) != 1`), leaving the heavier card with the full profile drain cliff
   (e.g. 2 GB) on top of weights + KV cache.
3. Unsharded graph constants (`constant: True`, e.g. 0-element `aten::lift_fresh` constants bound
   on the primary card): included in `weight_input_slots`, they reset `current_activation_device`
   to the primary card inside remote-card layers, pulling downstream weightless ops (including
   SDPA and its per-layer KV cache allocation) onto the primary card.
"""
from __future__ import annotations

from types import SimpleNamespace
import pytest

MB = 1024 * 1024


class _FakeContainer:
    def __init__(self, block_mb: float, non_block_mb: float, n_blocks: int = 4):
        self._block_mb = block_mb
        self._non_block_mb = non_block_mb
        self._n_blocks = n_blocks

    def get_component(self, name):
        return None


def _make_mem(weight_mb: float, act_mb: float = 100.0, overhead_mb: float = 0.0):
    w = int(weight_mb * MB)
    a = int(act_mb * MB)
    o = int(overhead_mb * MB)
    return SimpleNamespace(
        weight_bytes=w,
        activation_bytes=a,
        overhead_bytes=o,
        total_bytes=w + a + o,
        weight_mb=weight_mb,
        activation_mb=act_mb,
        overhead_mb=overhead_mb,
        total_mb=weight_mb + act_mb + overhead_mb,
    )


@pytest.mark.parametrize("branch", ["pytorch", "triton"])
def test_per_card_kv_accounting_and_drain_cliff_respect_heavier_card(branch):
    if branch == "pytorch":
        from neurobrix.pytorch.prism.solver import (
            PrismSolver, DeviceState, ComponentAllocation, ExecutionPlan, KVCachePlan,
        )
    else:
        from neurobrix.triton.prism.solver import (
            PrismSolver, DeviceState, ComponentAllocation, ExecutionPlan, KVCachePlan,
        )

    solver = PrismSolver()
    solver._mode = "triton"
    solver._lm_component_name = "lm"
    solver._serve_requested = False
    solver._serve_cold_fallback = False
    solver._input_config = None
    solver._load_copies_mb = 0.0

    mem_cfg = {
        "deferred_drain_cliff_bytes": 2000 * MB,
        "deferred_drain_floor_bytes": 256 * MB,
        "deferred_drain_reserve_bytes": 128 * MB,
    }
    solver._drain_profile_memory = lambda dev: mem_cfg
    solver._runtime_growth_fact = lambda spec: 0.0

    spec0 = SimpleNamespace(index=0, name="GPU0", memory_mb=10000, has_unified_memory=False, supports_dtypes=["float32", "float16"])
    spec1 = SimpleNamespace(index=1, name="GPU1", memory_mb=10000, has_unified_memory=False, supports_dtypes=["float32", "float16"])
    d0 = DeviceState(device_string="cuda:0", capacity_mb=10000.0, used_mb=0.0, components=[], spec=spec0)
    d1 = DeviceState(device_string="cuda:1", capacity_mb=10000.0, used_mb=0.0, components=[], spec=spec1)
    cards = [d0, d1]

    blocks_dict = {
        "blocks": {i: [f"block.{i}.w"] for i in range(4)},
        "block_sizes": {i: 2000.0 for i in range(4)},
        "non_block_keys": ["embed.w"],
        "non_block_mb": 1000.0,
    }
    solver._parse_blocks = lambda container, comp_name: blocks_dict
    solver._unloaded_by_request = lambda container: set()
    solver._get_component_dtype = lambda container, comp_name: "float16"
    solver._triton_arena_factor = lambda on=None: 1.0

    shard_map = {
        "embed.w": "cuda:0",
        "block.0.w": "cuda:0",
        "block.1.w": "cuda:0",
        "block.2.w": "cuda:1",
        "block.3.w": "cuda:1",
    }
    strat_allocs = {
        "lm": ("fgp:cuda:0,cuda:1", shard_map),
        "aux": ("cuda:0", {"aux.w": "cuda:0"}),
    }
    comp_mem = {
        "lm": _make_mem(weight_mb=9000.0, act_mb=200.0),
        "aux": _make_mem(weight_mb=2900.0, act_mb=100.0),
    }

    per_card = solver._per_card_kv_accounting(
        "pipeline_parallel", strat_allocs, comp_mem, cards, None,
        kv_already_counted=0, container=_FakeContainer(2000.0, 1000.0, 4),
    )
    assert per_card is not None
    held_by_card, kv_share = per_card
    assert kv_share == {"cuda:0": 0.5, "cuda:1": 0.5}
    assert held_by_card["cuda:0"] == int(8200 * MB)
    assert held_by_card["cuda:1"] == int(4200 * MB)

    cap0 = solver._kv_capacity_bytes(d0, set())
    cap1 = solver._kv_capacity_bytes(d1, set())
    max_kv = min(
        int(max(cap0 - held_by_card["cuda:0"], 0) / kv_share["cuda:0"]),
        int(max(cap1 - held_by_card["cuda:1"], 0) / kv_share["cuda:1"]),
    )
    assert abs(max_kv / MB - 3088.0) < 1.0

    kv_plan = KVCachePlan(
        max_cache_len=1000, num_layers=4, num_kv_heads=8, k_head_dim=128, v_head_dim=128,
        dtype="float16", memory_bytes=max_kv, per_token_bytes=max_kv // 1000,
    )
    solver._kv_card_left = solver._kv_card_room(cards, sum(held_by_card.values()), kv_plan, per_card=per_card)
    assert solver._kv_card_left is not None
    assert abs(solver._kv_card_left["cuda:0"] / MB - 256.0) < 1.0
    assert solver._kv_card_left["cuda:1"] > 2000 * MB

    plan = SimpleNamespace(
        strategy="pipeline_parallel",
        components={
            "lm": SimpleNamespace(
                device="cuda:0", devices=["cuda:0", "cuda:1"], dtype="float16",
                sharded=True, shard_map=shard_map, strategy="pipeline_parallel",
                deferred_drain_cliff_bytes=None,
            ),
            "aux": SimpleNamespace(
                device="cuda:0", devices=["cuda:0"], dtype="float16",
                sharded=False, shard_map={"aux.w": "cuda:0"}, strategy="single_gpu",
                deferred_drain_cliff_bytes=None,
            ),
        },
        component_memory=comp_mem,
        kv_cache_plan=kv_plan,
        layer_stream_plan=None,
        component_tiling=None,
        component_stream=None,
        deferred_drain_cliffs={},
    )
    solver._assign_drain_cliffs(_FakeContainer(2000.0, 1000.0, 4), plan, cards)
    cliff0 = plan.deferred_drain_cliffs["cuda:0"]
    cliff1 = plan.deferred_drain_cliffs["cuda:1"]
    assert abs(cliff0 / MB - 256.0) < 1.0
    assert abs(cliff1 / MB - 2000.0) < 1.0
    assert plan.components["lm"].deferred_drain_cliff_bytes == cliff0
    for d in cards:
        dn = d.device_string
        total_mb = (held_by_card[dn] + kv_plan.memory_bytes * kv_share[dn] + plan.deferred_drain_cliffs[dn]) / MB
        assert total_mb <= solver._rung_part_mb(d) + 1e-3


def test_graph_constant_does_not_hijack_op_device_compiled():
    """CompiledSequence.compute_op_devices: a constant:True tensor bound on device 0 must not
    pull remote-card layer ops (like SDPA) back onto device 0."""
    import torch
    from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence, CompiledOp

    class _FakeTensor:
        def __init__(self, dev_str: str, n: int):
            self.device = torch.device(dev_str)
            self._n = n
        def numel(self):
            return self._n

    dag = {
        "tensors": {
            "param::block.0.w": {"is_parameter": True, "weight_name": "block.0.w"},
            "param::block.1.w": {"is_parameter": True, "weight_name": "block.1.w"},
            "param::constant_T_0": {"is_parameter": True, "weight_name": "constant_T_0", "constant": True},
        },
        "ops": {},
        "execution_order": [],
    }
    seq = CompiledSequence.__new__(CompiledSequence)
    seq.dag = dag
    seq.device = torch.device("cuda:0")
    seq._num_weights = 3
    seq._num_inputs = 0
    seq._slot_to_tensor_id = {0: "param::block.0.w", 1: "param::block.1.w", 2: "param::constant_T_0"}
    seq._arena = [
        _FakeTensor("cuda:0", 1024),
        _FakeTensor("cuda:1", 1024),
        _FakeTensor("cuda:0", 0),
        None, None, None, None,
    ]
    noop = lambda *a, **k: None
    op0 = CompiledOp("mm::0", "aten::mm", noop, noop, noop, output_slots=(3,), weight_input_slots=(0,), all_input_slots=(0,))
    op1 = CompiledOp("mm::1", "aten::mm", noop, noop, noop, output_slots=(4,), weight_input_slots=(1,), all_input_slots=(3, 1))
    op_const = CompiledOp("lift_fresh::1", "aten::lift_fresh", noop, noop, noop, output_slots=(5,), weight_input_slots=(), all_input_slots=(2,))
    op_sdpa = CompiledOp("sdpa::1", "aten::scaled_dot_product_attention", noop, noop, noop, output_slots=(6,), weight_input_slots=(), all_input_slots=(4, 5))
    seq._ops = [op0, op1, op_const, op_sdpa]

    seq.compute_op_devices()
    assert seq._is_multi_device is True
    assert op0.device == torch.device("cuda:0")
    assert op1.device == torch.device("cuda:1")
    assert op_const.device == torch.device("cuda:1")
    assert op_const.needs_transfer is True
    assert op_sdpa.device == torch.device("cuda:1")


def test_graph_constant_does_not_hijack_op_device_triton():
    """TritonSequence._extract_weight_slots & compute_op_devices: twin mirror of compiled."""
    from neurobrix.triton.sequence import TritonSequence, CompiledOp

    class _FakeNBX:
        def __init__(self, dev_idx: int, n: int):
            self._device = "cuda"
            self._device_idx = dev_idx
            self._numel = n

    dag = {
        "tensors": {
            "param::block.0.w": {"is_parameter": True, "weight_name": "block.0.w"},
            "param::block.1.w": {"is_parameter": True, "weight_name": "block.1.w"},
            "param::constant_T_0": {"is_parameter": True, "weight_name": "constant_T_0", "constant": True},
        },
        "ops": {},
        "execution_order": [],
    }
    seq = TritonSequence.__new__(TritonSequence)
    seq.dag = dag
    seq.device_idx = 0
    seq._num_weights = 3
    seq._num_inputs = 0
    seq._tid_to_slot = {"param::block.0.w": 0, "param::block.1.w": 1, "param::constant_T_0": 2}
    w_slots_const = seq._extract_weight_slots({"attributes": {"args": [{"type": "tensor", "tensor_id": "param::constant_T_0"}]}})
    w_slots_real = seq._extract_weight_slots({"attributes": {"args": [{"type": "tensor", "tensor_id": "param::block.1.w"}]}})
    assert w_slots_const == []
    assert w_slots_real == [1]

    seq._arena = [
        _FakeNBX(0, 1024),
        _FakeNBX(1, 1024),
        _FakeNBX(0, 0),
        None, None, None, None,
    ]
    noop = lambda *a, **k: None
    op0 = CompiledOp("mm::0", "aten::mm", noop, noop, noop, output_slots=(3,), weight_input_slots=(0,), all_input_slots=(0,))
    op1 = CompiledOp("mm::1", "aten::mm", noop, noop, noop, output_slots=(4,), weight_input_slots=(1,), all_input_slots=(3, 1))
    op_const = CompiledOp("lift_fresh::1", "aten::lift_fresh", noop, noop, noop, output_slots=(5,), weight_input_slots=tuple(w_slots_const), all_input_slots=(2,))
    op_sdpa = CompiledOp("sdpa::1", "aten::scaled_dot_product_attention", noop, noop, noop, output_slots=(6,), weight_input_slots=(), all_input_slots=(4, 5))
    seq._ops = [op0, op1, op_const, op_sdpa]

    seq.compute_op_devices()
    assert seq._is_multi_device is True
    assert op0.device_idx == 0
    assert op1.device_idx == 1
    assert op_const.device_idx == 1
    assert op_const.needs_transfer is True
    assert op_sdpa.device_idx == 1
