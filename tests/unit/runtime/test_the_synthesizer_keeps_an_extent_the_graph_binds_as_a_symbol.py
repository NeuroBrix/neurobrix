"""The input synthesizer's shape transform cut a 213-token `inputs_embeds` down to the trace's 23
because the topology's `shapes` carry the trace's concrete extents — while the graph binds that
dimension as a symbol (`s1`, source `input::inputs_embeds::dim_1`) and runs at any length
(Qwen3-Omni, 2026-09-21: `shape '[-1, 1, 23]' is invalid for input of size 639`).

A dimension the graph binds as a symbol is never sliced. A dimension the graph carries as a
literal (a frozen extent, a trace defect at the source) is still cut to the literal — the only
way that graph runs — and the cut is said in clear.

Shapes: 213 is Omni's real prompt length here (196 image tokens plus text), 23 the trace's
prime; 2048 the hidden size, unchanged on both sides so the old rule would slice.
"""
from __future__ import annotations

import torch

from neurobrix.core.runtime.resolution.input_synthesizer import InputSynthesizer


class _Executor:
    def __init__(self, dag):
        self._dag = dag


def _dag(symbolic_seq: bool):
    dims = [{"type": "symbol", "id": "s0", "trace": 10},
            {"type": "symbol", "id": "s1", "trace": 23} if symbolic_seq else 23,
            2048]
    symbols = {"s0": {"name": "batch", "trace_value": 10, "source": "input::inputs_embeds::dim_0"}}
    if symbolic_seq:
        symbols["s1"] = {"name": "seq_len", "trace_value": 23, "source": "input::inputs_embeds::dim_1"}
    return {
        "input_tensor_ids": ["t0", "t1"],
        "tensors": {
            "t0": {"shape": [10, 23, 2048], "symbolic_shape": {"dims": dims, "concrete": [10, 23, 2048]}},
            "t1": {"shape": [10, 23, 2048], "symbolic_shape": {"dims": dims, "concrete": [10, 23, 2048]}},
        },
        "symbolic_context": {"symbols": symbols, "expressions": {}},
    }


def _synth(symbolic_seq: bool) -> InputSynthesizer:
    topology = {"components": {"thinker.model": {"shapes": {"inputs_embeds": [10, 23, 2048], "visual_pos_masks": [10, 23, 2048]}}}}
    return InputSynthesizer(topology, variable_resolver=None, plan=None, modules={},  # type: ignore[arg-type]
                            executors={"thinker.model": _Executor(_dag(symbolic_seq))})  # type: ignore[arg-type]


def test_a_symbolic_sequence_extent_is_handed_to_the_graph_whole():
    synth = _synth(symbolic_seq=True)
    inputs = {"inputs_embeds": torch.zeros(1, 213, 2048), "visual_pos_masks": torch.zeros(1, 213, 2048, dtype=torch.bool)}
    out = synth.apply_shape_transforms("thinker.model", inputs)
    assert tuple(out["inputs_embeds"].shape) == (1, 213, 2048)
    assert tuple(out["visual_pos_masks"].shape) == (1, 213, 2048)


def test_a_literal_sequence_extent_is_still_cut_to_the_literal_and_said(capsys):
    synth = _synth(symbolic_seq=False)
    out = synth.apply_shape_transforms("thinker.model", {"inputs_embeds": torch.zeros(1, 213, 2048)})
    assert tuple(out["inputs_embeds"].shape) == (1, 23, 2048)
    said = capsys.readouterr().out
    assert "thinker.model" in said and "inputs_embeds" in said and "213" in said and "23" in said
