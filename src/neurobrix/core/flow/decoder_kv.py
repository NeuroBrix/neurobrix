"""Decoder self-attention plan for encoder-decoder flows — graph-derived.

An encoder-decoder decoder (Whisper-class) carries two attention kinds per
layer: SELF-attention over the generated tokens, whose keys and values grow
by one token per step and belong in the KV cache, and CROSS-attention over
the encoder states, whose keys and values are the same at every step and
must NOT be concatenated. The graph tells them apart by dataflow: a
cross-attention's K/V derive from the `encoder_hidden_states` input. This
module is pure dict walking (no torch, no NBXTensor) so both engines share
one analysis (R30 / R33).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

_SDPA_TYPES = {
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_attention_math",
}


def _derived_from(dag: Dict[str, Any], roots: Set[str]) -> Set[str]:
    """Tensor ids reachable from `roots` through the graph's ops."""
    ops = dag.get("ops") or {}
    producers: Dict[str, List[str]] = {}
    for uid, op in ops.items():
        for tid in op.get("input_tensor_ids") or []:
            producers.setdefault(tid, []).append(uid)
    reached = set(roots)
    frontier = list(roots)
    while frontier:
        tid = frontier.pop()
        for uid in producers.get(tid, []):
            for out in ops[uid].get("output_tensor_ids") or []:
                if out not in reached:
                    reached.add(out)
                    frontier.append(out)
    return reached


def decoder_self_attention_plan(dag: Dict[str, Any],
                                encoder_input: str = "input::encoder_hidden_states",
                                ) -> Optional[Dict[str, Any]]:
    """Return the decoder's self-attention geometry and op ids, or None when
    the graph has no attention op.

    keys: num_layers (self-attention ops), num_heads, head_dim (from the
    traced q shape [B, H, S, D]), self_attn_uids, cross_attn_uids,
    arange_uids (the positional arange the cache offsets during decode),
    position_slice_uids (the other positional form: a slice of a parameter
    table by [0 : token-length symbol] — whisper-class decoders traced by
    older vendors; the cache offsets its window during decode). A decoder
    with neither cannot be offset one token at a time: the caller refuses
    the cache and keeps the recompute path (D-STT-KV-WHISPER-LARGE).
    """
    ops = dag.get("ops") or {}
    inputs = list(dag.get("input_tensor_ids") or [])
    # The residual stream carries encoder information after the first
    # cross-attention, so "derived from the encoder" alone is not the test:
    # a cross-attention's K/V derive from the encoder states and NOT from the
    # decoder's own token inputs; a self-attention's K/V derive from the tokens.
    from_tokens = _derived_from(dag, {t for t in inputs if t != encoder_input})
    self_uids: List[str] = []
    cross_uids: List[str] = []
    num_heads = head_dim = None
    for uid, op in ops.items():
        if op.get("op_type") not in _SDPA_TYPES:
            continue
        ins = op.get("input_tensor_ids") or []
        kv = ins[1:3]
        if kv and not any(t in from_tokens for t in kv):
            cross_uids.append(uid)
            continue
        self_uids.append(uid)
        shapes = op.get("input_shapes") or []
        if shapes and len(shapes[0]) == 4 and num_heads is None:
            num_heads, head_dim = int(shapes[0][1]), int(shapes[0][3])
    if not self_uids:
        return None
    if num_heads is None:
        raise RuntimeError("ZERO FALLBACK: decoder self-attention has no traced q shape")
    arange_uids = [uid for uid, op in ops.items() if op.get("op_type") == "aten::arange"]
    position_slice_uids = _positional_table_slices(dag, {t for t in inputs if t != encoder_input})
    return {
        "num_layers": len(self_uids), "num_heads": num_heads, "head_dim": head_dim,
        "self_attn_uids": self_uids, "cross_attn_uids": cross_uids,
        "arange_uids": arange_uids, "position_slice_uids": position_slice_uids,
    }


def _positional_table_slices(dag: Dict[str, Any], token_inputs: Set[str]) -> List[str]:
    """`aten::slice(param, dim, 0, <symbol>)` where the symbol is a dimension
    of a token input: the rows [0, seq_len) of a positional table. Pure
    graph data — no module name takes part."""
    symbols = (dag.get("symbolic_context") or {}).get("symbols") or {}
    token_syms = {sid for sid, sym in symbols.items()
                  if any(str(sym.get("source", "")).startswith(t + "::") for t in token_inputs)}
    found: List[str] = []
    for uid, op in (dag.get("ops") or {}).items():
        if op.get("op_type") != "aten::slice":
            continue
        args = (op.get("attributes") or {}).get("args") or []
        if len(args) < 4 or not str(args[0].get("tensor_id", "")).startswith("param::"):
            continue
        start, end = args[2], args[3]
        if start.get("type") == "scalar" and start.get("value") == 0 \
                and end.get("type") == "symbol" and end.get("id") in token_syms:
            found.append(uid)
    return found
