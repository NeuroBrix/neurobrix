"""Writing into the buffer you just read is safe ONLY where nothing else reads it.

An element-wise activation reads one tensor and writes another of the same shape.
Where the tensor it reads has no other consumer and is not a graph output, the
second buffer exists for one kernel and is waste. Where it has ANY other reader,
overwriting it is silent corruption: the other reader gets activated values where
it expected the convolution's.

This gate is about the LIVENESS PROOF, not the kernel. It is CPU-only and
deterministic, because the failure it guards is the one that produces a plausible
wrong picture rather than a crash -- the class this project calls a contaminated
oracle, where a byte gate cannot see an error present on both sides.

The motivating case is admitted and pinned: `real-esrgan-x8` at 1024x1024 has
`aten.convolution::349` -> [1, 64, 8192, 8192] fp16 = 8,589,934,592 bytes, read by
`aten.leaky_relu::278` and nothing else, and the run dies asking for a second
buffer of exactly that size.

SEEN RED: with the `consumers.get(src, []) != [op_uid]` clause removed,
`test_a_second_reader_forbids_the_reuse` fails; with `src in output_ids` removed,
`test_a_graph_output_is_never_overwritten` fails; with the shape equality check
removed, `test_a_shape_change_forbids_the_reuse` fails.

Run: PYTHONPATH=src python -m pytest tests/unit/module/test_an_activation_reuses_only_a_buffer_that_dies.py
"""
from __future__ import annotations

import pytest

from neurobrix.core.module.tiling_engine import OpLevelTilingEngine

BIG = [1, 64, 8192, 8192]      # 8,589,934,592 bytes at fp16 -- the real shape
SMALL = [1, 8, 16, 16]         # under any sane threshold


class _Executor:
    """The detector reads `_dag` and nothing else."""
    def __init__(self, dag):
        self._dag = dag


def _dag(*, consumers_of_conv=("act",), conv_is_graph_output=False,
         act_out_shape=None, act_type="aten::leaky_relu", shape=None):
    shape = shape or BIG
    act_out_shape = act_out_shape or shape
    ops = {
        "conv": {"op_uid": "conv", "op_type": "aten::convolution",
                 "input_tensor_ids": ["x"], "output_tensor_ids": ["t"],
                 "input_shapes": [shape], "output_shapes": [shape]},
        "act": {"op_uid": "act", "op_type": act_type,
                "input_tensor_ids": ["t"], "output_tensor_ids": ["y"],
                "input_shapes": [shape], "output_shapes": [act_out_shape]},
        "other": {"op_uid": "other", "op_type": "aten::convolution",
                  "input_tensor_ids": ["y"], "output_tensor_ids": ["z"],
                  "input_shapes": [act_out_shape], "output_shapes": [act_out_shape]},
    }
    # a second reader of the conv's output, when the case asks for one
    if "second" in consumers_of_conv:
        ops["second"] = {"op_uid": "second", "op_type": "aten::convolution",
                         "input_tensor_ids": ["t"], "output_tensor_ids": ["w"],
                         "input_shapes": [shape], "output_shapes": [shape]}
    order = ["conv", "act", "other"] + (["second"] if "second" in consumers_of_conv else [])
    return {"ops": ops, "execution_order": order,
            "output_tensor_ids": (["t"] if conv_is_graph_output else ["z"])}


def _detect(dag, threshold=1024 ** 3):
    return OpLevelTilingEngine._detect_inplace_unary_candidates(_Executor(dag), threshold)


def test_the_esrgan_case_is_admitted():
    """The 8,589,934,592-byte activation that kills the run."""
    assert _detect(_dag()) == [("act", "aten::leaky_relu")]


def test_a_second_reader_forbids_the_reuse():
    """Overwriting `t` would hand the other consumer activated values."""
    assert _detect(_dag(consumers_of_conv=("act", "second"))) == []


def test_a_graph_output_is_never_overwritten():
    """A tensor handed back to the caller is still needed after the op."""
    assert _detect(_dag(conv_is_graph_output=True)) == []


def test_a_shape_change_forbids_the_reuse():
    """A different element count would not fit the buffer it read."""
    assert _detect(_dag(act_out_shape=[1, 64, 8192, 4096])) == []


def test_an_op_outside_the_safe_map_is_not_admitted():
    """The map is the admission list, and softmax is not element-wise.

    `aten::softmax` reads every element of a row to produce each one, so its
    kernel cannot read and write the same buffer. It is the case that proves
    the map is consulted rather than the op merely having one input.
    """
    assert _detect(_dag(act_type="aten::softmax")) == []
    assert "aten::softmax" not in OpLevelTilingEngine.INPLACE_SAFE_UNARY


def test_size_is_deliberately_not_decided_here():
    """The detector admits a SMALL tensor, and that is the fix, not a bug.

    The DAG carries the extents the model was TRACED at. `real-esrgan-x8`'s
    `aten.leaky_relu::278` declares [1, 64, 896, 640] -- 140 MB -- and the
    1024x1024 request runs that same op at [1, 64, 8192, 8192], 8 GiB, which is
    the allocation that kills it. A byte threshold applied to `output_shapes`
    rejected precisely the op it existed to catch, and said nothing.

    So liveness is settled here and size is settled in the interceptor, which
    holds the real tensor. This test pins the SPLIT: if a size filter ever comes
    back to the detector, it goes red.
    """
    assert _detect(_dag(shape=SMALL)) == [("act", "aten::leaky_relu")]
    # and the argument is accepted-and-ignored rather than removed, so the
    # signature still matches the adds' detector
    assert _detect(_dag(shape=SMALL), threshold=10 ** 15) == [("act", "aten::leaky_relu")]


class _CapturingExecutor(_Executor):
    """Catches the interceptor dict instead of installing it."""
    def __init__(self, dag):
        super().__init__(dag)
        self.captured = {}
        self._op_uid_interceptors = {}

    def register_op_uid_interceptors(self, interceptors):
        self.captured.update(interceptors)
        return len(interceptors)


def _interceptor_for(dag, op_uid="act"):
    from neurobrix.core.module.tiling_engine import OpLevelTilingPlan
    plan = OpLevelTilingPlan("model")
    engine = OpLevelTilingEngine(plan)
    ex = _CapturingExecutor(dag)
    engine.register_into_graph_executor(ex)
    return ex.captured.get(op_uid), plan


def test_the_interceptor_is_registered_for_the_admitted_op():
    fn, plan = _interceptor_for(_dag())
    assert plan.inplace_unary == [("act", "aten::leaky_relu")]
    assert fn is not None, "liveness was proved but no interceptor was installed"


def test_the_interceptor_decides_size_at_call_time():
    """Small tensors take the ordinary path; large ones write in place.

    This is the half the DAG cannot answer, so it is the half worth pinning.
    Driven with a real NBXTensor when a card is present -- the threshold is
    lowered by the environment variable rather than allocating a gigabyte.
    """
    pytest.importorskip("neurobrix.kernels.nbx_tensor")
    from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
    if DeviceAllocator.device_count() == 0:
        pytest.skip("needs a CUDA device")
    import numpy as np, os

    x = np.linspace(-2, 2, 4096).astype(np.float32)

    os.environ["NBX_INPLACE_MIN_BYTES"] = str(10 ** 12)     # nothing qualifies
    fn, _ = _interceptor_for(_dag())
    t = NBXTensor.from_numpy(x.copy())
    out = fn(t)
    DeviceAllocator.sync_device()
    assert out.data_ptr() != t.data_ptr(), (
        "above the threshold-as-configured nothing should be written in place")

    os.environ["NBX_INPLACE_MIN_BYTES"] = "0"               # everything qualifies
    fn, _ = _interceptor_for(_dag())
    t2 = NBXTensor.from_numpy(x.copy())
    out2 = fn(t2)
    DeviceAllocator.sync_device()
    assert out2.data_ptr() == t2.data_ptr(), (
        "below the threshold-as-configured the input buffer should carry it")
    os.environ.pop("NBX_INPLACE_MIN_BYTES", None)


def test_every_op_in_the_map_names_a_real_wrapper():
    """A map entry pointing at nothing would raise only when the op is met."""
    from neurobrix.kernels import wrappers
    import inspect
    for op_type, fn_name in OpLevelTilingEngine.INPLACE_SAFE_UNARY.items():
        fn = getattr(wrappers, fn_name, None)
        assert fn is not None, f"{op_type} -> wrappers.{fn_name} does not exist"
        assert "out" in inspect.signature(fn).parameters, (
            f"wrappers.{fn_name} has no `out` parameter, so the interceptor "
            f"cannot make it write in place and would silently allocate")
