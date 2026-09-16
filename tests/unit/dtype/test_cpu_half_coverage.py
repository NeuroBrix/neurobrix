"""The CPU backend has narrower fp16 coverage than CUDA, and the engine knows it.

`engine.py` describes its op sets as "the CUDA autocast rules — universally
applicable". They are not. PyTorch's CPU backend refuses fp16 for ops CUDA
accepts, and its own CPU autocast defaults to bfloat16 for exactly that reason.

Any Prism plan that places compute on the host — `lazy_sequential`,
`cpu_execution`, `cpu_streaming`, zero3 offload — therefore runs ops on a
backend that may refuse them. Kokoro-82M reaches this on a single 16 GB card,
where Prism picks `lazy_sequential` and puts `decoder` on the host. On three
cards it picks `single_gpu` and the op stays on CUDA, which is why the full-zoo
battery — pinned to 0,1,3 — has never seen it. Path coverage, not model
coverage.

The second test is the one that matters over time: it RE-MEASURES every entry
against THIS torch and ties the engine's own decision (`cpu_lacks_half_kernel`)
to what the op actually does here. On a fleet of two torches — the cert rig's
2.5.1 (both ops raise) and Apple's 2.14 (both implemented) — the same entry is
needed on one machine and free on the other, so the set does not shrink; it is
measured. An entry that no torch lacks any more is dead weight to delete, but an
entry one machine still needs is not stale just because another implemented it.
"""

from __future__ import annotations

import pytest
import torch

from neurobrix.core.dtype.engine import CPU_NO_HALF_OPS


def _probe(op_name: str, device: str, dtype: torch.dtype):
    """Call the op the way the failing model does, and report the outcome."""
    if op_name == "_weight_norm_interface":
        v = torch.randn(4, 4, device=device, dtype=dtype)
        g = torch.randn(4, 1, device=device, dtype=dtype)
        return torch._weight_norm_interface(v, g, 0)
    if op_name == "reflection_pad1d":
        x = torch.randn(1, 4, 16, device=device, dtype=dtype)
        return torch.nn.functional.pad(x, (2, 2), mode="reflect")
    raise AssertionError(
        f"no probe for '{op_name}' — an entry in CPU_NO_HALF_OPS without a "
        f"probe here is an unverified claim, which is what this file exists "
        f"to prevent"
    )


def test_the_set_is_not_empty_by_accident():
    """If it empties, the branch in the engine is dead and should be removed
    deliberately rather than left as decoration."""
    assert CPU_NO_HALF_OPS, (
        "CPU_NO_HALF_OPS is empty — either PyTorch fixed every entry (delete "
        "the branch in engine.py too) or an edit removed them silently"
    )


@pytest.mark.parametrize("op_name", sorted(CPU_NO_HALF_OPS))
def test_the_engine_agrees_with_this_torch(op_name):
    """A candidate, not a verdict — and the engine measures it, per torch.

    `CPU_NO_HALF_OPS` names ops that SOME supported torch lacks a CPU half
    kernel for. Whether THIS torch lacks it is a property of this torch: the
    cert rig's 2.5.1 still raises for both; Apple's 2.14 implemented both. A
    static "still lacks" assertion is therefore right on one machine and wrong
    on the other — the exact defect the engine's own comment calls out, which
    is why the engine does not trust the list but probes with
    `cpu_lacks_half_kernel`.

    This test pins that probe to observed reality: the engine's decision must
    equal what the raw op actually does here. True where it raises (the wrapper
    is needed), False where a newer build implemented it (the wrapper is not
    applied, and the entry simply stops costing anything). Either way the entry
    is safe to keep, so the set does not have to shrink to stay honest — it has
    to be measured, which is what this asserts."""
    from neurobrix.core.dtype.engine import (
        cpu_lacks_half_kernel, _CPU_HALF_MEASURED,
    )
    _CPU_HALF_MEASURED.pop(op_name, None)   # force a fresh, uncached measurement
    try:
        _probe(op_name, "cpu", torch.float16)
        raw_raises = False
    except RuntimeError as exc:
        assert "not implemented for 'Half'" in str(exc)
        raw_raises = True
    assert cpu_lacks_half_kernel(op_name) == raw_raises, (
        f"the engine's CPU-half decision for {op_name} disagrees with what "
        f"torch {torch.__version__} actually does: the fp32 wrapper would be "
        + ("missing where the op needs it" if raw_raises
           else "applied to an op this torch runs natively")
    )


@pytest.mark.parametrize("op_name", sorted(CPU_NO_HALF_OPS))
def test_every_entry_works_in_fp32_on_cpu(op_name):
    """The remedy has to be the remedy: if fp32 also fails, upcasting is not
    the fix and the entry is misfiled."""
    _probe(op_name, "cpu", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("op_name", sorted(CPU_NO_HALF_OPS))
def test_the_same_op_is_fine_in_fp16_on_cuda(op_name):
    """This is why the wrapper decides per CALL and per DEVICE rather than
    upcasting everywhere: on CUDA the op is fine, and forcing fp32 there would
    cost throughput to work around a limitation that is not present."""
    _probe(op_name, "cuda", torch.float16)


def test_the_wrapper_leaves_cuda_tensors_alone():
    from neurobrix.core.dtype.engine import DtypeEngine

    engine = object.__new__(DtypeEngine)
    seen = {}

    def spy(*args, **kwargs):
        seen["dtype"] = args[0].dtype
        return args[0]

    wrapped = DtypeEngine._make_cpu_fp32_wrapper(engine, spy)
    wrapped(torch.zeros(2, 2, dtype=torch.float16))          # host -> upcast
    assert seen["dtype"] == torch.float32
    if torch.cuda.is_available():
        wrapped(torch.zeros(2, 2, dtype=torch.float16, device="cuda"))
        assert seen["dtype"] == torch.float16, "CUDA inputs must pass through"


def test_the_result_comes_back_in_the_graphs_dtype():
    """The op must be invisible downstream: fp16 in, fp16 out, whatever
    happened inside."""
    from neurobrix.core.dtype.engine import DtypeEngine

    engine = object.__new__(DtypeEngine)
    wrapped = DtypeEngine._make_cpu_fp32_wrapper(engine, lambda x: x * 2)
    out = wrapped(torch.ones(2, 2, dtype=torch.float16))
    assert out.dtype == torch.float16


@pytest.mark.parametrize("op_name", sorted(CPU_NO_HALF_OPS))
def test_the_sequential_oracle_honours_the_set(op_name):
    """The ATen oracle (`--sequential`) dispatches the graph's ops directly,
    without the compiled engine's per-op wrappers — so a host-placed component
    reached the CPU backend in fp16 and the oracle died where the compiled
    engine had already learned to upcast (Kokoro-82M's decoder under
    `lazy_sequential` on a 16 GB card, the drift table of 2026-09-06). The
    hardware contract is one rule with one owner; both engines apply it."""
    from neurobrix.core.runtime.graph.sequential_dispatcher import NativeATenDispatcher

    d = NativeATenDispatcher(device="cpu")
    if op_name == "_weight_norm_interface":
        v = torch.randn(4, 4, dtype=torch.float16)
        g = torch.randn(4, 1, dtype=torch.float16)
        out = d.dispatch("aten::_weight_norm_interface", [v, g, 0], {"kwargs": {}})
        assert out[0].dtype == torch.float16 and out[0].device.type == "cpu"
    elif op_name == "reflection_pad1d":
        x = torch.randn(1, 4, 16, dtype=torch.float16)
        out = d.dispatch("aten::reflection_pad1d", [x, [2, 2]], {"kwargs": {}})
        assert out.dtype == torch.float16 and out.shape[-1] == 20
    else:
        raise AssertionError(f"no oracle probe for '{op_name}'")


def test_a_host_placement_decides_the_compute_dtype_and_the_weight_dtype_alike():
    """The plan-time decision — a host-placed component computes in fp32 —
    must reach the weights too. On 2026-09-06 the engine computed in fp32
    while the loader still narrowed the same component's weights to the
    plan's fp16, and the ATen oracle met fp32 activations with fp16 weights
    at its first matmul (Kokoro-82M's decoder under lazy_sequential)."""
    from neurobrix.core.dtype.engine import compute_dtype_for_placement

    assert compute_dtype_for_placement("cpu", torch.float16) is torch.float32
    assert compute_dtype_for_placement("cuda:0", torch.float16) is torch.float16
    assert compute_dtype_for_placement("cuda:1", torch.bfloat16) is torch.bfloat16
    assert compute_dtype_for_placement(None, torch.float16) is torch.float16
