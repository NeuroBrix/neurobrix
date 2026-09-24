"""A seam that carries only bytes is not a seam.

`build_segment_graph` aliases each tensor crossing a segment boundary to `input::<tid>` and
rewrites the ops to read the alias. It rewrote two places — `input_tensor_ids` and
`attributes["args"]` — and dropped everything else the next segment needs to make sense of
what it received. Three defects, each hidden behind the one before it, measured 2026-09-22 on
DeepSeek-Coder-V2-Lite-Instruct (16 GB V100, `NBX_FORCE_STRATEGY=layer_streaming`,
`--max-tokens 8`) by fixing them one at a time and re-running:

1. **A tensor id in an attribute was not aliased.** A fused op does not take all its inputs
   positionally: `custom::moe_fused` names its hidden states, gate scores and pre-computed
   routing by tid in `attributes`, and the runtime resolves those to arena SLOTS. The value
   arrived under `input::<tid>` while the attribute still asked for `<tid>`, so the slot was
   empty:

       RuntimeError: MoE fused: hidden_states is None (slot N).
       Killed by liveness analysis before fused op.

   The message names the wrong cause — nothing killed it, it was never bound. All four class-1
   MoE models failed here, and it was recorded as a `hidden_states-None` column for a whole
   day before anyone read the seam.

2. **The alias dropped `symbolic_shape`.** Every dim crossing the boundary arrived as a
   literal, so no later segment could bind a symbol:

       UnboundSymbolError: symbol 's0' (batch, binds from input::input_ids::dim_0)
       is not bound at runtime. Bound: ['s2', 's3']

   `s2`/`s3` bound because they source from `position_ids`, a graph input every segment gets;
   `s0`/`s1` source from `input_ids`, which the embedding reads in segment 0 and nothing reads
   after it. Principle 1 is not suspended at a seam.

3. **A symbol's SOURCE still named an input the segment does not have.** Carrying the
   symbolic shape is not enough — the binding must point at a tensor this segment receives.
   The first repair RE-SOURCED the symbol to a seam dim with the same id; it could not serve a
   trace with two ids for one extent (PixArt's T5: `s1` from attention_mask, `s3` from
   input_ids; the hidden states carry `s3`, a later piece uses `s1` — the Mac's render died
   there, 8e786e70). Now the segment CARRIES the component input the symbol binds from: the
   source is unchanged, the input is declared, the strategy hands it in. The executed binding
   is gated at three sizes in `test_every_streamed_piece_binds_every_symbol_it_uses.py`.

After all three, the same command ran all three segments. The cells below assert the
invariants — no op name, no attribute name, no symbol id in an assertion — because a fused op
written next year is covered by the invariant and not by a list.
"""
from __future__ import annotations

import pytest

from neurobrix.core.prism.layer_partition import Segment, build_segment_graph


def _graph():
    """Two ops across one boundary. `b` is a fused-style op: it names its hidden-state
    input in an ATTRIBUTE, the way `custom::moe_fused` does, not only positionally."""
    return {
        "tensors": {
            # A component input is named `input::<name>` — the engine's convention, and the one
            # a symbol's source (`input::input_ids::dim_0`) names.
            "input::input_ids": {"tensor_id": "input::input_ids", "is_input": True, "shape": [1, 23],
                          "dtype": "int64",
                          "symbolic_shape": {"dims": [{"type": "symbol", "id": "s0", "trace": 1},
                                                      {"type": "symbol", "id": "s1", "trace": 23}],
                                             "concrete": [1, 23]}},
            "h": {"tensor_id": "h", "producer_op_uid": "a", "shape": [1, 23, 8],
                  "dtype": "float16",
                  "symbolic_shape": {"dims": [{"type": "symbol", "id": "s0", "trace": 1},
                                              {"type": "symbol", "id": "s1", "trace": 23}, 8],
                                     "concrete": [1, 23, 8]}},
            "out": {"tensor_id": "out", "producer_op_uid": "b", "shape": [1, 23, 8],
                    "dtype": "float16", "output_name": "logits"},
        },
        "ops": {
            "a": {"op_type": "aten::embedding", "input_tensor_ids": ["input::input_ids"],
                  "output_tensor_ids": ["h"]},
            "b": {"op_type": "custom::moe_fused", "input_tensor_ids": ["h"],
                  "output_tensor_ids": ["out"],
                  "attributes": {"args": [{"tensor_id": "h"}],
                                 "hidden_states_tid": "h",
                                 "gate_group": ["h"],
                                 "top_k": 6, "num_experts": 64}},
        },
        "execution_order": ["a", "b"],
        "input_tensor_ids": ["input::input_ids"],
        "output_tensor_ids": ["out"],
        "symbolic_context": {
            "symbols": {
                "s0": {"name": "batch", "trace_value": 1, "source": "input::input_ids::dim_0"},
                "s1": {"name": "seq_len", "trace_value": 23, "source": "input::input_ids::dim_1"},
            },
            "expressions": {},
        },
    }


@pytest.fixture()
def seg1():
    """The SECOND segment — the one that receives `h` across the seam."""
    return build_segment_graph(_graph(), Segment(index=1, first_op="b", last_op="b",
                                                 op_count=1, weight_bytes=0))


def _alias(seg):
    a = [t for t in seg["input_tensor_ids"] if t.startswith("input::")]
    assert a, "the segment received nothing across the seam — the fixture is vacuous"
    return a[0]


# ───────────────────────── 1. every tensor id in an attribute ─────────────────────────

def test_an_id_in_a_SCALAR_attribute_is_aliased(seg1):
    """`hidden_states_tid`. Left un-aliased it pointed at a slot the seam never fills."""
    assert seg1["ops"]["b"]["attributes"]["hidden_states_tid"] == _alias(seg1)


def test_an_id_in_a_LIST_attribute_is_aliased(seg1):
    """`gate_group`, `topk_*_tid` — a fused op carries lists of ids too."""
    assert seg1["ops"]["b"]["attributes"]["gate_group"] == [_alias(seg1)]


def test_a_NON_tensor_attribute_is_untouched(seg1):
    """The rewrite knows only 'this value IS a tensor id this seam aliases'. It must not
    reach a count, a flag or a name, or it corrupts the op it was fixing."""
    attrs = seg1["ops"]["b"]["attributes"]
    assert attrs["top_k"] == 6 and attrs["num_experts"] == 64


def test_the_positional_rewrite_that_already_worked_still_works(seg1):
    assert seg1["ops"]["b"]["input_tensor_ids"] == [_alias(seg1)]
    assert seg1["ops"]["b"]["attributes"]["args"][0]["tensor_id"] == _alias(seg1)


def test_the_partition_does_not_damage_the_graph_it_was_cut_from():
    """Ops are shared with the full graph; the rewrite must copy."""
    g = _graph()
    build_segment_graph(g, Segment(index=1, first_op="b", last_op="b",
                                   op_count=1, weight_bytes=0))
    assert g["ops"]["b"]["attributes"]["hidden_states_tid"] == "h"
    assert g["ops"]["b"]["attributes"]["gate_group"] == ["h"]


# ───────────────────────── 2. the seam carries the SYMBOLIC shape ─────────────────────

def test_the_seam_tensor_keeps_its_symbolic_shape(seg1):
    """Without it every crossing dim is a literal and Principle 1 stops at the boundary."""
    dims = (seg1["tensors"][_alias(seg1)].get("symbolic_shape") or {}).get("dims")
    assert dims, "the seam tensor arrived with no symbolic shape"
    assert [d.get("id") for d in dims if isinstance(d, dict)] == ["s0", "s1"]


# ───────────────────────── 3. a symbol binds from what the segment HAS ────────────────

def test_a_symbol_binds_where_the_whole_component_binds_it(seg1):
    """`input_ids` is read by segment 0 and by nothing after it: the segment CARRIES it, and the
    symbol keeps its original source — the same tensor the whole component binds from."""
    for sid in ("s0", "s1"):
        assert seg1["symbolic_context"]["symbols"][sid]["source"].startswith("input::input_ids::dim_")
    assert "input::input_ids" in seg1["input_tensor_ids"]
    assert seg1["symbol_carriers"] == ["input::input_ids"]
    assert seg1["tensors"]["input::input_ids"]["symbol_carrier"] is True


def test_the_carrier_is_named_the_way_the_caller_hands_it_in(seg1):
    """`segment_input_names` is the key set the strategy passes; the carrier is the component's
    own input, so it is looked up under the name the component was called with."""
    assert "input_ids" in seg1["segment_input_names"]
    assert "input::input_ids" not in seg1["segment_input_names"]


def test_a_symbol_the_segment_already_binds_carries_nothing():
    """Segment 0 holds `input_ids` itself: nothing to carry, nothing declared twice."""
    seg0 = build_segment_graph(_graph(), Segment(index=0, first_op="a", last_op="a",
                                                 op_count=1, weight_bytes=0))
    assert seg0["symbolic_context"]["symbols"]["s0"]["source"] == "input::input_ids::dim_0"
    assert seg0["symbol_carriers"] == []
    assert seg0["input_tensor_ids"].count("input::input_ids") == 1


def _with_s9(source):
    """`s9` USED in segment 1 (the output tensor carries it), sourced from `source`."""
    g = _graph()
    g["symbolic_context"]["symbols"]["s9"] = {"name": "other", "trace_value": 4, "source": source}
    g["tensors"]["out"]["symbolic_shape"] = {"dims": [{"type": "symbol", "id": "s9", "trace": 4}, 23, 8],
                                             "concrete": [4, 23, 8]}
    return g


def test_a_symbol_whose_source_is_not_a_component_input_is_not_carried_and_stays_unbound():
    """Nothing is invented. A dim that genuinely does not cross must refuse at binding — the
    engine's own resolver leaves it unbound — or the seam would paper over a coverage hole."""
    from types import SimpleNamespace
    from neurobrix.triton.symbols import SymbolResolver
    seg = build_segment_graph(_with_s9("input::absent::dim_0"),
                              Segment(index=1, first_op="b", last_op="b", op_count=1, weight_bytes=0))
    assert seg["symbolic_context"]["symbols"]["s9"]["source"] == "input::absent::dim_0"
    assert "input::absent" not in seg["symbol_carriers"]
    r = SymbolResolver(seg["symbolic_context"])
    r.bind_from_inputs({t: SimpleNamespace(shape=(1, 23, 8))
                        for t in seg["input_tensor_ids"]}, seg["input_tensor_ids"], seg["tensors"])
    assert "s9" not in r.bindings and {"s0", "s1"} <= set(r.bindings)


def test_a_source_form_the_partitioner_cannot_carry_is_refused_at_partition_time():
    with pytest.raises(ValueError, match="cannot carry"):
        build_segment_graph(_with_s9("weights::x::dim_0"),
                            Segment(index=1, first_op="b", last_op="b", op_count=1, weight_bytes=0))


def test_a_seam_tensor_keeps_its_dtype_at_every_engines_entry(seg1):
    """The seam rule where the engines apply it (register 105): the triton entry cast returns no
    target for a seam tensor, and a component input of the same dtype string IS cast."""
    from neurobrix.core.prism.layer_partition import is_seam_tensor
    from neurobrix.triton.dtype import TritonDtypeEngine
    seam = seg1["tensors"][_alias(seg1)]
    assert is_seam_tensor(seam) and not is_seam_tensor(seg1["tensors"]["input::input_ids"])
    eng = TritonDtypeEngine.__new__(TritonDtypeEngine)
    from neurobrix.kernels.nbx_tensor import parse_dtype
    eng.compute_dtype = parse_dtype("bfloat16")
    assert eng._target_dtype_for_input(_alias(seg1), seg1["tensors"]) is None
    plain = {"x": {"dtype": "float32"}}
    assert eng._target_dtype_for_input("x", plain) == parse_dtype("bfloat16")
