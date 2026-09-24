"""An op that would produce an impossible tensor stops, by name, instead of continuing.

Measured 2026-09-24 (Allegro-TI2V census, container without its `pad_image_to_num_frames`
flag): the VAE encoder received a 1-frame clip. `aten.convolution::14` (temporal kernel 2,
stride 2, no padding) produced T = 0 and nothing stopped. In triton-sequential `aten.cat::16`
of two `(1, 128, 0, 448, 448)` tensors returned `(0,)`, rank 5 collapsed to rank 1, and the
failure surfaced 40 ops later as an IndexError in a slice. In triton mode the next temporal
conv computed T = 0 - 3 + 1 = -2, and the failure surfaced as a division by zero in a group norm.

The two refusals:
* a convolution whose output extent is <= 0 raises `ImpossibleExtentError` naming the input
  and weight shapes and the arithmetic, at the convolution;
* a `cat` whose inputs are all empty raises instead of returning a rank-1 `(0,)`.
The executors add the op uid and the bound symbol values (pinned in the executor tests).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")

from neurobrix.kernels.nbx_tensor import NBXDtype, NBXTensor  # noqa: E402


def _gpu_or_skip():
    try:
        from neurobrix.kernels.nbx_tensor import _detect_gpu_backend
        _detect_gpu_backend()
    except Exception:
        pytest.skip("no GPU backend the engine can resolve")


def _t(shape):
    return NBXTensor.from_numpy(np.zeros(shape, dtype=np.float32))


def test_a_temporal_conv_with_no_frame_to_produce_stops_by_name():
    _gpu_or_skip()
    from neurobrix.kernels.nbx_tensor import ImpossibleExtentError
    from neurobrix.kernels.wrappers import conv2d_wrapper
    x = _t((1, 4, 1, 8, 8))               # one frame
    w = _t((4, 4, 2, 1, 1))               # temporal kernel 2
    with pytest.raises(ImpossibleExtentError) as e:
        conv2d_wrapper(x, w, None, [2, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1)
    msg = str(e.value)
    assert "(1, 4, 1, 8, 8)" in msg and "(4, 4, 2, 1, 1)" in msg and "T_out=0" in msg, msg


def test_a_spatial_conv_larger_than_its_input_stops_by_name():
    _gpu_or_skip()
    from neurobrix.kernels.nbx_tensor import ImpossibleExtentError
    from neurobrix.kernels.wrappers import conv2d_wrapper
    x = _t((1, 2, 2, 2))
    w = _t((2, 2, 3, 3))
    with pytest.raises(ImpossibleExtentError) as e:
        conv2d_wrapper(x, w, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    assert "(1, 2, 2, 2)" in str(e.value) and "out_h=0" in str(e.value), str(e.value)


def test_a_valid_conv_is_unchanged():
    _gpu_or_skip()
    from neurobrix.kernels.wrappers import conv2d_wrapper
    out = conv2d_wrapper(_t((1, 2, 5, 5)), _t((3, 2, 3, 3)), None, [1, 1], [0, 0], [1, 1],
                         False, [0, 0], 1)
    assert tuple(out.shape) == (1, 3, 3, 3)


def test_a_cat_of_empty_tensors_does_not_collapse_to_rank_one():
    _gpu_or_skip()
    from neurobrix.kernels.nbx_tensor import ImpossibleExtentError
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    d = TritonSequentialDispatcher.__new__(TritonSequentialDispatcher)
    a = NBXTensor.empty((1, 8, 0, 4, 4), dtype=NBXDtype.float32)
    with pytest.raises(ImpossibleExtentError) as e:
        d._cat_inputs_or_refuse([[a, a], 2])
    assert "(1, 8, 0, 4, 4)" in str(e.value), str(e.value)


def test_a_cat_with_one_empty_operand_keeps_the_other():
    """The legitimate case (an empty KV cache concatenated with the first token) is unchanged."""
    _gpu_or_skip()
    from neurobrix.triton.sequential import TritonSequentialDispatcher
    d = TritonSequentialDispatcher.__new__(TritonSequentialDispatcher)
    empty = NBXTensor.empty((1, 8, 0, 4), dtype=NBXDtype.float32)
    full = _t((1, 8, 3, 4))
    kind, value = d._cat_inputs_or_refuse([[empty, full], 2])
    assert kind == "single" and tuple(value.shape) == (1, 8, 3, 4)


def test_the_executor_names_the_input_shapes_and_the_bound_symbols():
    """The suffix both mode-2 executors append to an ImpossibleExtentError (and only to it)."""
    from neurobrix.kernels.nbx_tensor import ImpossibleExtentError
    from neurobrix.triton.symbols import SymbolResolver, impossible_extent_context
    r = SymbolResolver({"symbols": {"s0": {"name": "batch"}, "s1": {"name": "time"}}})
    r._bindings.update({"s0": 1, "s1": 1})

    class _T:
        shape = (1, 128, 1, 448, 448)

    msg = impossible_extent_context(ImpossibleExtentError("x"), [_T(), [1, 1, 1], 2], r)
    assert "(1, 128, 1, 448, 448)" in msg and "s0=1 (batch)" in msg and "s1=1 (time)" in msg, msg
    assert impossible_extent_context(ValueError("other"), [_T()], r) == ""
