"""A weight whose aten::t is eliminated has its graph shape swapped ONCE per graph, however many compiles see it.

The elimination rewires the transpose's consumers onto the weight and stamps it `pretransposed`, but the aten::t
op itself stays in the op table. A streamed piece compiled again for the next pass rediscovered it and swapped the
recorded shape back — the metadata flipped on every compile. The census shadow shapes its weights from that
metadata, so from the second pass on it handed the bind a weight already transposed, and the bind's .t() made it
(10240, 4096) again: PixArt-XL-2-1024-MS's T5 streamed in 29 pieces died at aten.mm::4 on its negative-prompt pass,
and with it every T5-family text encoder of the catalogue at the 6-8 GB rungs (census of 2026-09-26).
"""
from __future__ import annotations

import copy

import pytest


def _dag():
    tensors = {
        "param::w": {"shape": [10240, 4096], "dtype": "float32", "weight_name": "w"},
        "input::x": {"shape": [120, 4096], "dtype": "float32"},
    }
    ops = {
        "aten.t::0": {"op_type": "aten::t", "input_tensor_ids": ["param::w"], "output_tensor_ids": ["aten.t::0::out_0"],
                      "attributes": {"args": [{"type": "tensor", "tensor_id": "param::w"}], "kwargs": {}}},
        "aten.mm::0": {"op_type": "aten::mm", "input_tensor_ids": ["input::x", "aten.t::0::out_0"],
                       "output_tensor_ids": ["aten.mm::0::out_0"],
                       "attributes": {"args": [{"type": "tensor", "tensor_id": "input::x"},
                                               {"type": "tensor", "tensor_id": "aten.t::0::out_0"}], "kwargs": {}}},
    }
    return tensors, ops, ["aten.t::0", "aten.mm::0"]


def _compile_twice(cls):
    tensors, ops, order = _dag()
    shapes = []
    for _ in range(2):                                   # two compiles over the SAME graph objects
        seq = cls.__new__(cls)
        seq._pretranspose_weights = set()
        # A streamed piece recompiles from its segment graph: the op table is shared, the order is fresh.
        seq._eliminate_weight_transpose_ops(tensors, ops, list(order))
        shapes.append(list(tensors["param::w"]["shape"]))
        assert "param::w" in seq._pretranspose_weights
    return shapes, tensors


@pytest.mark.parametrize("which", ["triton", "compiled"])
def test_the_recorded_shape_is_swapped_once(which):
    if which == "triton":
        from neurobrix.triton.sequence import TritonSequence as cls
    else:
        from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence as cls
    shapes, tensors = _compile_twice(cls)
    assert shapes == [[4096, 10240], [4096, 10240]], (
        f"{which}: the recorded shape went {shapes} over two compiles — it flipped back on the second")
    assert tensors["param::w"]["pretransposed"] is True


def test_the_shadow_shapes_a_stamped_weight_as_its_file():
    from neurobrix.kernels import census
    tensors, _, _ = _dag()
    tensors["param::w"].update({"shape": [4096, 10240], "pretransposed": True})   # after a compile
    captured = {}

    class _FakeNBX:
        @staticmethod
        def empty(shape, dtype=None, device=None):
            captured["shape"] = tuple(shape); return object()

    class _Exec:
        _dag = {"tensors": {"param::w": tensors["param::w"]}}
        dtype = "float32"
        device = "mps:0"

    import neurobrix.kernels.nbx_tensor as nt
    real = nt.NBXTensor
    nt.NBXTensor = _FakeNBX
    try:
        census._shadow_params_for(_Exec(), "/nowhere", "c")
    finally:
        nt.NBXTensor = real
    assert captured["shape"] == (10240, 4096), (
        f"the shadow shaped the weight {captured['shape']}: the bind's .t() will transpose it twice")
