"""Prism prices a Triton arena at the library's layout of the weights the loader places in it (M3, COST.md section 2):
each weight at the dtype the Triton loader holds it in (`expert_files.held_dtype(..., "triton")`), each on the arena's
256-byte boundary (`nbxtensor.memory.layout.arena_layout`) — the very bump allocation `ComponentArena` performs. It
priced the component's stored bytes times one dtype multiplier, with no alignment: a figure the allocator never asks
for.

What this test would do if the code were wrong: an arena priced from the stored bytes or without the boundary differs
from the layout (red); an arena whose bump allocation lands elsewhere than the layout says (red); an expert store's
dense arena counting its routed experts (red). Card-less: the allocator's two device calls are replaced."""
from __future__ import annotations

from types import SimpleNamespace

from neurobrix.core.prism.solver import _consumed_weights, _held_weight_sizes
from nbxtensor.memory.layout import ARENA_ALIGNMENT, arena_layout


def _component():
    """Three weights read by one op, stored fp32 and bf16 at sizes off the 256-byte boundary, one of them a routed
    expert's; the weights index states what is stored."""
    tensors = {"x": {"shape": [1], "dtype": "float16", "is_parameter": False}}
    index = {}
    for name, shape, dtype in (("layers.0.norm.weight", [3], "float32"), ("layers.0.proj.weight", [5, 7], "bfloat16"),
                               ("layers.0.mlp.experts.0.w", [9, 11], "bfloat16")):
        tensors["param::" + name] = {"shape": shape, "dtype": dtype, "is_parameter": True, "weight_name": name}
        n = 1
        for d in shape:
            n *= d
        index[name] = {"shape": shape, "dtype": dtype, "size_bytes": n * (4 if dtype == "float32" else 2)}
    ops = {"op0": {"input_tensor_ids": ["x", *[t for t in tensors if t.startswith("param::")]],
                   "output_tensor_ids": ["y"]}}
    graph = {"tensors": tensors, "ops": ops, "execution_order": ["op0"]}
    return SimpleNamespace(graph=graph, weights_index={"tensors": index})


def test_the_arena_is_the_layout_of_the_held_weights():
    comp = _component()
    held = _held_weight_sizes(comp, "float16")
    # fp32 under a half compute is held half (the loader's one rule), bf16 under fp16 compute is held fp16
    assert held == [3 * 2, 5 * 7 * 2, 9 * 11 * 2], held
    assert arena_layout(held).total_bytes == 3 * ARENA_ALIGNMENT
    stored = sum(e[1] for e in _consumed_weights(comp))
    assert arena_layout(held).total_bytes != stored, "the arena is priced at what is stored, not at what is held"


def test_an_expert_stores_dense_arena_leaves_its_routed_experts_out():
    held = _held_weight_sizes(_component(), "float16", exclude=frozenset({"layers.0.mlp.experts.0.w"}))
    assert held == [3 * 2, 5 * 7 * 2]
    assert arena_layout(held).total_bytes == 2 * ARENA_ALIGNMENT


def test_the_arena_allocates_where_the_layout_says(monkeypatch):
    from nbxtensor.memory.allocator import DeviceAllocator
    from nbxtensor.memory.memory_pool import ComponentArena
    asked = []
    monkeypatch.setattr(DeviceAllocator, "set_device", staticmethod(lambda i: None))
    monkeypatch.setattr(DeviceAllocator, "malloc_cuda",
                        staticmethod(lambda n, dev_idx=None, pooled=True: asked.append((n, pooled)) or 1 << 30))
    monkeypatch.setattr(DeviceAllocator, "free_cuda", staticmethod(lambda p: None))
    sizes = [6, 0, 70, 512, 1, 257]
    layout = arena_layout(sizes)
    arena = ComponentArena(layout.total_bytes, 0)
    try:
        got = [arena.alloc(n) for n in sizes]
        used = arena.used
    finally:
        arena.free()                     # under the replaced device calls, never at a later collection
    assert asked == [(layout.total_bytes, False)], asked
    assert [p - (1 << 30) if p else 0 for p in got] == [o if n else 0 for o, n in zip(layout.offsets, sizes)]
    assert used == layout.total_bytes
