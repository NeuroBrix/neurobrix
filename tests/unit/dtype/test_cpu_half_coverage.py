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

The second test is the one that matters over time: it RE-MEASURES every entry,
so the set shrinks by construction when upstream implements a kernel and can
never fill up with ops that were once broken.
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
def test_every_entry_still_lacks_a_cpu_half_kernel(op_name):
    """The list shrinks by construction.

    An entry PyTorch has since implemented is costing an upcast for nothing.
    Delete it — do not leave it because it was once true."""
    with pytest.raises(RuntimeError, match="not implemented for 'Half'"):
        _probe(op_name, "cpu", torch.float16)


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
