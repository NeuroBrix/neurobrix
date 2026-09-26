"""The compiled mode runs the spatial promotion even when every seq_len symbol collides with a weight dim.

`CompiledSequence._promote_seq_len_scalars_to_symbolic` returned early when no seq_len trace value was
collision-safe, and the spatial pass (height/width rebinding, and the correction of a slice end bound to a
spatial symbol by trace coincidence) sits after that point. Two containers of the catalogue meet it:
PixArt-XL-2-1024-MS's transformer (text length 120, also a weight extent) and mochi-1-preview's (256). PixArt
at 768x1024 then died in --compiled at aten.addmm::0 on (2x160 @ 256x1152) — its timestep embedding's
`emb[:, :128]` had been traced with the end bound to the latent HEIGHT (128 at the traced 1024 px) — while
both Triton modes, which call the spatial pass themselves, rendered the image (2026-09-26).

The graph below is that shape in miniature: a seq_len symbol whose trace value is a weight dim, spatial
symbols, and a slice of a non-spatial tensor whose end is the height symbol.
"""
from __future__ import annotations

from neurobrix.core.runtime.graph.compiled_sequence import CompiledSequence


def _graph():
    symbols = {
        "s1": {"name": "seq_len", "trace_value": 120, "source": "input::encoder_hidden_states::dim_1"},
        "s4": {"name": "height", "trace_value": 128, "source": "input::hidden_states::dim_2"},
        "s5": {"name": "width", "trace_value": 128, "source": "input::hidden_states::dim_3"},
    }
    tensors = {
        "param::caption.weight": {"shape": [1152, 120], "weight_name": "caption.weight"},   # 120 is a weight extent
        "param::time.weight": {"shape": [1152, 256], "weight_name": "time.weight"},
    }
    ops = {
        "aten.slice::5": {
            "op_type": "aten::slice",
            "input_tensor_ids": ["aten.cat::0::out_0"],        # the timestep embedding: not spatial
            "output_tensor_ids": ["aten.slice::5::out_0"],
            "attributes": {"args": [
                {"type": "tensor", "tensor_id": "aten.cat::0::out_0"},
                {"type": "scalar", "value": 1}, {"type": "scalar", "value": 0},
                {"type": "symbol", "id": "s4", "trace": 128}]},
        },
    }
    dag = {"symbolic_context": {"symbols": symbols}, "ops": ops, "tensors": tensors}
    return dag, tensors, ops


def test_a_coincidental_spatial_slice_end_is_corrected_in_compiled_mode():
    dag, tensors, ops = _graph()
    seq = CompiledSequence.__new__(CompiledSequence)
    seq.dag = dag
    seq._config_constants = ()
    seq._promote_seq_len_scalars_to_symbolic(tensors, ops)
    end = ops["aten.slice::5"]["attributes"]["args"][3]
    assert end == {"type": "scalar", "value": 128}, (
        f"the slice end stayed {end}: with every seq_len symbol colliding, the compiled mode skipped the "
        f"spatial pass, so at a latent height of 96 this slice keeps 96 of 256 channels")
