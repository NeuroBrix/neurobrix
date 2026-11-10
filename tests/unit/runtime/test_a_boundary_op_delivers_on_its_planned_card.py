"""A device-boundary op delivers its outputs on the card the plan gave it (2026-11-10). The fused MoE op reads the arena
itself, so the boundary path's moved arguments never reach it; its experts held on another card, it returned on the
card its input came from, and the next op — fast-pathed on the planned card — added tensors of two cards
(Qwen3-Coder-30B-A3B, weight_sharding on two V100-32GB: aten.add::71, 'cuda:3 and cuda:2').

What this test would do if the code were wrong: the add after the arena-reading op raises the two-device error — RED."""
import pytest

torch = pytest.importorskip("torch")

from neurobrix.core.runtime.graph.compiled_sequence import CompiledOp, CompiledSequence, TensorArena  # noqa: E402


def test_an_arena_reading_boundary_op_hands_its_result_to_its_planned_card():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two cards are needed: the op's planned card and the card its result comes back on")
    home, other = torch.device("cuda:0"), torch.device("cuda:1")
    arena = TensorArena(3)
    arena[0] = torch.ones(4, device=home)

    def reads_the_arena_and_returns_elsewhere(a):          # the fused MoE's shape: the arena itself is its argument
        return a[0].to(other) * 2

    moe = CompiledOp(op_uid="moe_fused::block.0", op_type="custom::moe_fused", func=reads_the_arena_and_returns_elsewhere,
                     args_resolver=lambda a: [a], kwargs_resolver=lambda a: {}, output_slots=(1,),
                     all_input_slots=(0,), device=home, needs_transfer=True)
    add = CompiledOp(op_uid="aten.add::0", op_type="aten::add", func=torch.add,
                     args_resolver=lambda a: [a[0], a[1]], kwargs_resolver=lambda a: {}, output_slots=(2,),
                     all_input_slots=(0, 1), device=home, needs_transfer=False)

    seq = CompiledSequence.__new__(CompiledSequence)
    seq._ops = [moe, add]
    seq.device = home
    seq._kill = lambda arena, slots, uid: None
    CompiledSequence._run_inner_multi_device(seq, arena)

    assert arena[1].device == home
    assert torch.equal(arena[2].cpu(), torch.full((4,), 3.0))
