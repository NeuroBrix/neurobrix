"""Unit test for the sequential-mode RoPE cache-slice re-slicing policy.

Regression guard for the op-by-op ORACLE decode crash (D5.3): TinyLlama-style
models register a `cos_cached`/`sin_cached` buffer of shape
[max_position_embeddings, head_dim] (e.g. [2048, 64]) and, in the traced
graph, slice it to [seq_len, head_dim] then index it by `position_ids`:

    cos = cos_cached[:seq_len]      # aten::slice, end == seq_len symbol
    cos = cos[position_ids]         # aten::index

The slice `end` is captured as a SYMBOL (`{"type": "symbol", "id": "s1",
"trace": 23}`). During autoregressive decode the runtime seq_len is 1, so the
symbol resolves to 1 and narrows the table to [1, head_dim]. The decode-step
`position_ids` is ABSOLUTE (=[[cache_len]]), so indexing [1, head_dim] with
position `cache_len >= 1` walks off the end -> CUDA device-side assert
(index out of bounds) at `aten.index::0`.

Compiled mode never hit this because
`CompiledSequence._promote_seq_len_scalars_to_symbolic` unconditionally
rewrites the RoPE cache-slice `end` to the FULL table size. The sequential
mirror `GraphExecutor._patch_seq_len_in_ops` only rewrote SCALAR ends and
silently skipped SYMBOL ends. This test pins the corrected contract: the
RoPE cache-slice `end` is forced to the full table dim regardless of the
original arg type (scalar OR symbol), so absolute-position indexing during
decode always hits valid rows.

CPU-only (no torch tensors, no GPU) -> always runnable.

Runnable:  PYTHONPATH=src python3 -m pytest \
    tests/unit/runtime/test_rope_cache_slice_sequential.py -v
"""
from __future__ import annotations

import neurobrix.core.runtime  # noqa: F401  (pre-resolve cfg<->runtime import cycle)
from neurobrix.core.runtime.graph_executor import GraphExecutor

TABLE_LEN = 2048
HEAD_DIM = 64
TRACE_SEQ = 23


def _make_executor(end_arg):
    """Bare GraphExecutor with a single RoPE cos_cached slice op.

    `end_arg` is the args[3] (slice end) node under test (symbol or scalar).
    """
    param = "param::block.0.attn.rotary_embed.cos_cached"
    ops = {
        "aten.slice::0": {
            "op_uid": "aten.slice::0",
            "op_type": "aten::slice",
            "input_tensor_ids": [param],
            "output_tensor_ids": ["aten.slice::0::out_0"],
            "attributes": {
                "args": [
                    {"type": "tensor", "tensor_id": param},
                    {"type": "scalar", "value": 0},   # dim
                    {"type": "scalar", "value": 0},   # start
                    end_arg,                            # end (under test)
                ],
                "kwargs": {},
            },
        },
    }
    dag = {
        "symbolic_context": {
            "symbols": {"s1": {"name": "seq_len", "trace_value": TRACE_SEQ}},
        },
        "tensors": {param: {"shape": [TABLE_LEN, HEAD_DIM]}},
        "ops": ops,
    }
    ex = object.__new__(GraphExecutor)
    ex._dag = dag
    return ex, ops


def _end_of(ops):
    return ops["aten.slice::0"]["attributes"]["args"][3]


def test_symbol_end_forced_to_full_table():
    """SYMBOL end (the TinyLlama crash case): a decode bind of seq_len=1 must
    NOT narrow the table; end is rewritten to the full table size."""
    ex, ops = _make_executor(
        {"type": "symbol", "id": "s1", "trace": TRACE_SEQ}
    )
    # Decode step: runtime seq_len bound to 1.
    ex._patch_seq_len_in_ops({"s1": 1})

    end = _end_of(ops)
    assert end.get("type") == "scalar", f"end must be a concrete scalar: {end}"
    assert end["value"] == TABLE_LEN, (
        f"RoPE cache-slice end must be the full table dim {TABLE_LEN}, not the "
        f"runtime seq_len — else absolute position_ids index out of bounds "
        f"during decode. Got {end['value']}."
    )


def test_scalar_end_forced_to_full_table():
    """SCALAR end (pre-existing behavior) is preserved: still forced to the
    full table size, never left at the runtime seq_len."""
    ex, ops = _make_executor({"type": "scalar", "value": TRACE_SEQ})
    ex._patch_seq_len_in_ops({"s1": 1})

    end = _end_of(ops)
    assert end.get("type") == "scalar" and end["value"] == TABLE_LEN


def test_ar_safe_across_decode_steps():
    """Autoregressive re-invocation must re-derive the full size from the
    restored ORIGINAL args each call — never compound or drift to seq_len."""
    ex, ops = _make_executor(
        {"type": "symbol", "id": "s1", "trace": TRACE_SEQ}
    )
    # Prefill (seq_len=22), then several decode steps (seq_len=1).
    for bound in ({"s1": 22}, {"s1": 1}, {"s1": 1}, {"s1": 1}):
        ex._patch_seq_len_in_ops(bound)
        end = _end_of(ops)
        assert end.get("type") == "scalar" and end["value"] == TABLE_LEN, (
            f"end drifted across AR steps: {end}"
        )
