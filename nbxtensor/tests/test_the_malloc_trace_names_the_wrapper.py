"""The malloc trace (NBX_MALLOC_TRACE) names the frame that asked for the memory: a wrapper of the library or the engine
that calls it, never the allocator, the pool or the tensor themselves. It keyed on 'neurobrix' alone until the library
moved out of the engine (M1, 2026-10-28): every wrapper frame was skipped and each site named the engine loop instead.

What this test would do if the code were wrong: a library wrapper not taken for a site, the allocator taken for one,
or an engine file named by a tree that happens to be called nbxtensor (red)."""
from nbxtensor.memory import allocator as A

TREE = "/home/u/worktrees/nbxtensor"


def test_a_wrapper_of_the_library_is_a_site():
    f = TREE + "/nbxtensor/src/nbxtensor/ops/conv_wrappers.py"
    assert A._is_caller_frame(f) and A._rel_site(f) == "nbxtensor/ops/conv_wrappers.py"


def test_an_engine_frame_is_named_from_its_package_even_in_a_tree_named_nbxtensor():
    f = TREE + "/src/neurobrix/triton/sequence.py"
    assert A._is_caller_frame(f) and A._rel_site(f) == "neurobrix/triton/sequence.py"


def test_the_allocator_the_pool_and_the_tensor_are_never_sites():
    for own in ("memory/allocator.py", "memory/memory_pool.py", "memory/slab.py", "tensor.py"):
        assert not A._is_caller_frame(TREE + "/nbxtensor/src/nbxtensor/" + own), own
    assert not A._is_caller_frame("/usr/lib/python3/dist-packages/triton/runtime/jit.py")
