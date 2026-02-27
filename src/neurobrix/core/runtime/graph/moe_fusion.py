"""
MoE Fusion Pass — DAG Rewrite for Mixture-of-Experts Models

Detects MoE routing subgraphs (topk → sort → bincount → per-expert FFN → scatter_reduce)
and replaces them with a single fused moe_dispatch op that executes dynamically at runtime.

This eliminates hardcoded slice boundaries burned into the graph during tracing,
enabling correct routing for any input at runtime.

ZERO HARDCODE: All parameters (num_experts, top_k, hidden_dim, intermediate_dim)
are extracted from the graph tensors and op attributes.

Called BEFORE CompiledSequence.compile() to transform the DAG in-place.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import re
import os


def detect_and_fuse_moe(dag: Dict[str, Any], family: str, norm_topk_prob: bool = True) -> Dict[str, Any]:
    """
    Detect MoE patterns in DAG and replace with fused ops.

    Guards:
        - family must be "llm" (eliminates image/audio/video)
        - Must find topk ops with k > 1 (eliminates dense LLMs)

    Args:
        dag: The TensorDAG dict (mutated in-place)
        family: Model family from manifest ("llm", "image", etc.)
        norm_topk_prob: Whether to normalize routing weights after topk selection.
            DeepSeek: False (raw softmax scores). Qwen3/Mixtral: True (default).

    Returns:
        The DAG (same reference, possibly mutated)
    """
    if os.environ.get("NBX_DISABLE_MOE_FUSION"):
        return dag
    if family != "llm":
        return dag

    ops = dag.get("ops", {})
    execution_order = dag.get("execution_order", [])
    tensors = dag.get("tensors", {})

    # Single O(N) pass: find all topk ops with k > 1
    moe_topk_uids = []
    for op_uid in execution_order:
        op_data = ops.get(op_uid)
        if op_data is None:
            continue
        if op_data.get("op_type") != "aten::topk":
            continue
        # Extract k from attributes.args[1] (positional arg)
        k = _extract_topk_k(op_data)
        if k is not None and k > 1:
            moe_topk_uids.append(op_uid)

    if not moe_topk_uids:
        return dag  # Not MoE — unchanged

    # Build maps ONCE, then update incrementally after each fusion
    consumer_map = _build_consumer_map(ops, execution_order)
    producer_map = _build_producer_map(ops, execution_order)

    fused_count = 0
    ops_removed_total = 0

    for topk_uid in moe_topk_uids:
        result = _fuse_one_moe_layer(
            dag, ops, execution_order, tensors,
            consumer_map, producer_map, topk_uid,
            norm_topk_prob=norm_topk_prob,
        )
        if result is not None:
            removed_count, fused_uid, fused_op = result
            fused_count += 1
            ops_removed_total += removed_count

            # Incremental map update: add fused op's inputs/outputs
            for in_tid in _collect_input_tids(fused_op):
                if in_tid not in consumer_map:
                    consumer_map[in_tid] = []
                consumer_map[in_tid].append(fused_uid)
            for out_tid in fused_op.get("output_tensor_ids", []):
                producer_map[out_tid] = fused_uid

    # Update DAG
    dag["ops"] = ops
    dag["execution_order"] = execution_order

    return dag


def _fuse_one_moe_layer(
    dag: Dict[str, Any],
    ops: Dict[str, Any],
    execution_order: List[str],
    tensors: Dict[str, Any],
    consumer_map: Dict[str, List[str]],
    producer_map: Dict[str, str],
    topk_uid: str,
    norm_topk_prob: bool = True,
) -> Optional[Tuple[int, str, Dict[str, Any]]]:
    """
    Fuse one MoE layer starting from a topk op.

    Returns (ops_removed, fused_uid, fused_op_data) or None if fusion failed/skipped.

    COMPATIBILITY:
    - DeepSeek v1 (deepseek-moe-16b-chat): ~1300 ops per layer, mm ops in subgraph → FUSE
    - DeepSeek v2 (DeepSeek-Coder-V2): ~11 ops per layer, mm ops NOT in subgraph → SKIP
      V2 uses a different architecture where expert computation is decoupled from routing.
    """
    topk_data = ops[topk_uid]

    # --- Extract MoE parameters from graph (ZERO HARDCODE) ---

    # topk input = gate_scores (softmax output)
    gate_scores_tid = _get_input_tensor_id(topk_data, 0)
    if gate_scores_tid is None:
        raise RuntimeError(f"[MoE Fusion] Cannot find gate_scores input for {topk_uid}")

    # k from topk attributes
    top_k = _extract_topk_k(topk_data)

    # topk outputs: scores and indices
    topk_scores_tid = topk_data["output_tensor_ids"][0]
    topk_indices_tid = topk_data["output_tensor_ids"][1]

    # --- Trace forward from topk to find the complete MoE subgraph ---

    # Collect ALL ops that belong to this MoE layer
    moe_op_uids: Set[str] = set()
    moe_op_uids.add(topk_uid)

    # Find the hidden_states input: trace back from the index ops
    # The gate_scores come from softmax, which comes from router mm
    # The hidden_states are the input to both the router AND the expert index ops

    # Step 1: Find sort, bincount, floor_divide, view ops after topk
    _trace_routing_ops(ops, execution_order, consumer_map,
                       topk_scores_tid, topk_indices_tid, moe_op_uids)

    # num_experts from topk input shape (gate_scores last dim) — ZERO HARDCODE
    num_experts = _count_total_experts(tensors, topk_uid, ops)

    # Step 2: Extract expert weight IDs BEFORE removing boundary ops
    # Boundary removal would exclude mm ops whose outputs exit the subgraph (down projection),
    # but we need those mm ops to extract the expert weight tensor IDs.
    expert_weight_ids, num_experts_found, hidden_states_tid = \
        _trace_expert_blocks(ops, execution_order, tensors, consumer_map,
                             producer_map, moe_op_uids, topk_uid, num_experts)

    # Step 3: Remove boundary ops — ops whose outputs have consumers OUTSIDE the MoE subgraph.
    # These ops must stay in execution_order. This happens AFTER weight extraction.
    boundary_ops = set()
    for op_uid in list(moe_op_uids):
        op_data = ops.get(op_uid, {})
        for out_tid in op_data.get("output_tensor_ids", []):
            for c_uid in consumer_map.get(out_tid, []):
                if c_uid not in moe_op_uids:
                    boundary_ops.add(op_uid)
                    break
            if op_uid in boundary_ops:
                break
    moe_op_uids -= boundary_ops

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPATIBILITY CHECK: Skip fusion if expert weights not found in MoE subgraph
    # ═══════════════════════════════════════════════════════════════════════════
    # DeepSeek v2 architecture decouples routing from expert execution.
    # The mm ops with expert weights are NOT in the topk-derived subgraph.
    # In this case, we SKIP fusion — the model runs correctly without it.
    if num_experts_found == 0:
        return None

    # Step 3: Find MoE output tensor — DATA-DRIVEN (works for v1 scatter_reduce AND v2 index_put+sum)
    last_scatter_tid = _find_moe_output(
        ops, execution_order, tensors, consumer_map, producer_map,
        moe_op_uids, boundary_ops
    )

    if last_scatter_tid is None:
        return None

    if hidden_states_tid is None:
        return None

    # --- Create fused op ---
    # Extract block identifier from parent_module of topk
    parent = topk_data.get("parent_module", "")
    # e.g. "block.1.ffn.router" -> "block.1"
    block_match = re.match(r"(block\.\d+)", parent)
    block_id = block_match.group(1) if block_match else topk_uid

    fused_uid = f"moe_fused::{block_id}"

    # Build args list for liveness analysis — all input tensors must be declared
    # so _extract_input_slots_from_dag can track them and prevent premature GC
    liveness_args = [
        {"type": "tensor", "tensor_id": hidden_states_tid},
        {"type": "tensor", "tensor_id": gate_scores_tid},
    ]
    # All expert weight tensors (64 × 3 = 192 tensors)
    for i in range(num_experts):
        liveness_args.append({"type": "tensor", "tensor_id": expert_weight_ids["gate"][i]})
        liveness_args.append({"type": "tensor", "tensor_id": expert_weight_ids["up"][i]})
        liveness_args.append({"type": "tensor", "tensor_id": expert_weight_ids["down"][i]})

    # input_tensor_ids for native mode liveness tracking (_compute_last_use scans this)
    all_input_tids = [hidden_states_tid, gate_scores_tid]
    for i in range(num_experts):
        all_input_tids.append(expert_weight_ids["gate"][i])
        all_input_tids.append(expert_weight_ids["up"][i])
        all_input_tids.append(expert_weight_ids["down"][i])

    fused_op = {
        "op_type": "custom::moe_fused",
        "output_tensor_ids": [last_scatter_tid],
        "input_tensor_ids": all_input_tids,
        "output_shapes": tensors.get(last_scatter_tid, {}).get("shape", []),
        "attributes": {
            "args": liveness_args,
            "kwargs": {},
            "gate_scores_tid": gate_scores_tid,
            "hidden_states_tid": hidden_states_tid,
            "expert_gate_weight_ids": expert_weight_ids["gate"],
            "expert_up_weight_ids": expert_weight_ids["up"],
            "expert_down_weight_ids": expert_weight_ids["down"],
            "top_k": top_k,
            "num_experts": num_experts,
            "norm_topk_prob": norm_topk_prob,
        },
    }

    # --- Remove old ops from execution_order, add fused op ---

    # Find the latest non-MoE op position that is BEFORE or WITHIN the MoE range.
    # The fused op must come AFTER all its input producers (hidden_states, gate_scores).
    # These producers may be interleaved with MoE ops in execution_order.

    # Find the position of the LAST MoE op in the original order
    last_moe_pos = 0
    for i, uid in enumerate(execution_order):
        if uid in moe_op_uids:
            last_moe_pos = i

    # Remove all MoE ops and build new order
    new_order = []
    for uid in execution_order:
        if uid in moe_op_uids:
            continue
        new_order.append(uid)

    # Insert fused op right after the last non-MoE op that was before or within
    # the MoE block range. This ensures all input producers have executed.
    insert_idx = 0
    for uid in execution_order[:last_moe_pos + 1]:
        if uid not in moe_op_uids:
            insert_idx += 1
    new_order.insert(insert_idx, fused_uid)

    # Post-fusion dead-op elimination: boundary ops that depend on removed
    # internal MoE ops will get None inputs → remove them.
    # The fused op produces last_scatter_tid; any other internal tensor is gone.
    #
    # PROTECTION: Never remove shared_expert ops — they have consumers outside
    # the MoE subgraph (residual connections). The parent_module heuristic
    # catches "shared_expert" or "shared" in the module path.
    fused_output_tids = set(fused_op.get("output_tensor_ids", []))
    removed_producers = set()
    for uid in moe_op_uids:
        for out_tid in ops.get(uid, {}).get("output_tensor_ids", []):
            if out_tid not in fused_output_tids:
                removed_producers.add(out_tid)

    # Iteratively remove ops whose inputs depend on removed tensors
    dead_ops: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for uid in new_order:
            if uid in dead_ops or uid == fused_uid:
                continue
            # PROTECT shared_expert paths from dead-op elimination
            op_data = ops.get(uid, {})
            parent_module = op_data.get("parent_module", "")
            if "shared_expert" in parent_module or "shared" in parent_module:
                continue
            for in_tid in _collect_input_tids(op_data):
                if in_tid in removed_producers:
                    dead_ops.add(uid)
                    # This dead op's outputs are also removed
                    for out_tid in op_data.get("output_tensor_ids", []):
                        removed_producers.add(out_tid)
                    changed = True
                    break

    if dead_ops:
        new_order = [uid for uid in new_order if uid not in dead_ops]

    # Update structures
    ops[fused_uid] = fused_op
    # Don't delete old ops from ops dict (tensors reference them), just remove from execution_order
    execution_order.clear()
    execution_order.extend(new_order)

    ops_removed = len(moe_op_uids)

    return ops_removed, fused_uid, fused_op


# ============================================================================
# TRACING HELPERS
# ============================================================================

def _trace_routing_ops(
    ops: Dict[str, Any],
    execution_order: List[str],
    consumer_map: Dict[str, List[str]],
    topk_scores_tid: str,
    topk_indices_tid: str,
    moe_op_uids: Set[str],
) -> None:
    """
    Trace routing ops after topk: view, sort, bincount, floor_divide,
    _to_copy, detach, zeros_like, etc.

    Phase 1: Forward BFS from topk outputs to find all MoE ops.
    Phase 2: Backward pass to collect ops whose outputs are ONLY consumed by MoE ops
             (e.g., zeros_like that feeds scatter_reduce).
    """
    # MoE op types — any op with these types that consumes a topk-derived tensor
    # is part of the MoE subgraph
    MOE_OP_TYPES = {
        # Routing infrastructure
        "aten::view", "aten::reshape", "aten::_unsafe_view",
        "aten::sort", "aten::bincount", "aten::floor_divide",
        "aten::_to_copy", "aten::detach",
        "aten::mul", "aten::div", "aten::sum",
        "aten::unsqueeze", "aten::squeeze", "aten::permute",
        # Expert selection (Qwen3-MoE pattern: gt → nonzero → unbind → per-expert paths)
        "aten::gt", "aten::lt", "aten::ge", "aten::le", "aten::eq", "aten::ne",
        "aten::nonzero", "aten::unbind", "aten::_local_scalar_dense",
        # Expert block ops
        "aten::select",   # Qwen3: select(routing_matrix, dim=0, index=expert_id) per expert
        "aten::slice", "aten::index", "aten::t", "aten::mm", "aten::silu",
        "aten::scatter_reduce", "aten::repeat",
        "aten::index_put", "aten::scatter_add", "aten::index_add",
        # V2 aggregation pattern: cat + scatter (NOT aten::add — escapes via residuals)
        "aten::cat", "aten::scatter",
        # Accumulation setup (buffers for expert outputs)
        "aten::zeros_like", "aten::zeros", "aten::full",
        "aten::empty_like", "aten::empty",
    }

    # Phase 1: Forward BFS from topk outputs
    visited_tids: Set[str] = set()
    frontier = [topk_scores_tid, topk_indices_tid]

    while frontier:
        tid = frontier.pop()
        if tid in visited_tids:
            continue
        visited_tids.add(tid)

        for consumer_uid in consumer_map.get(tid, []):
            if consumer_uid in moe_op_uids:
                continue
            consumer_data = ops.get(consumer_uid)
            if consumer_data is None:
                continue

            op_type = consumer_data.get("op_type", "")
            if op_type in MOE_OP_TYPES:
                moe_op_uids.add(consumer_uid)
                for out_tid in consumer_data.get("output_tensor_ids", []):
                    frontier.append(out_tid)



def _find_moe_output(
    ops: Dict[str, Any],
    execution_order: List[str],
    tensors: Dict[str, Any],
    consumer_map: Dict[str, List[str]],
    producer_map: Dict[str, str],
    moe_op_uids: Set[str],
    boundary_ops: Set[str],
) -> Optional[str]:
    """
    Find the MoE output tensor using DATA-DRIVEN detection.

    Instead of hardcoding op_type (scatter_reduce for v1, index_put for v2, etc.),
    we find the last MoE op whose output has consumers OUTSIDE the MoE subgraph.
    This is the definition of "MoE output" — the tensor that continues in the network.

    Strategy 1: Find op in moe_op_uids whose output exits the subgraph
    Strategy 2: If boundary_ops contains the exit op, find it there
    Strategy 3: Fallback to last op with [seq_len, hidden_dim] shaped output

    Returns:
        output tensor_id, or None if not found
    """
    # Strategy 1: Find op whose output exits the MoE subgraph
    for op_uid in reversed(execution_order):
        if op_uid not in moe_op_uids:
            continue
        op_data = ops.get(op_uid)
        if op_data is None:
            continue
        for out_tid in op_data.get("output_tensor_ids", []):
            consumers = consumer_map.get(out_tid, [])
            has_external_consumer = any(c not in moe_op_uids for c in consumers)
            if has_external_consumer:
                # Validate: output should be [seq_len, hidden_dim], not a routing scalar
                tdata = tensors.get(out_tid, {})
                shape = tdata.get("shape", [])
                if len(shape) >= 2 and shape[-1] > 1:
                    return out_tid

    # Strategy 2: boundary_ops were removed from moe_op_uids because they have external consumers.
    # One of them IS the MoE output — find the one whose input comes from moe_op_uids.
    for op_uid in reversed(execution_order):
        if op_uid not in boundary_ops:
            continue
        op_data = ops.get(op_uid)
        if op_data is None:
            continue
        # Check if this boundary op consumes a tensor from the MoE subgraph
        has_moe_input = False
        for in_tid in _collect_input_tids(op_data):
            producer_uid = producer_map.get(in_tid)
            if producer_uid in moe_op_uids:
                has_moe_input = True
                break
        if has_moe_input:
            out_tids = op_data.get("output_tensor_ids", [])
            if out_tids:
                tdata = tensors.get(out_tids[0], {})
                shape = tdata.get("shape", [])
                if len(shape) >= 2 and shape[-1] > 1:
                    return out_tids[0]

    # Strategy 3: Fallback — last MoE op with activation-sized output
    for op_uid in reversed(execution_order):
        if op_uid not in moe_op_uids:
            continue
        op_data = ops.get(op_uid)
        if op_data is None:
            continue
        out_tids = op_data.get("output_tensor_ids", [])
        if out_tids:
            tdata = tensors.get(out_tids[0], {})
            shape = tdata.get("shape", [])
            # Activation-sized: at least 2D with hidden_dim > routing_dim
            # e.g., [64, 2048] not [64, 6] or [64]
            if len(shape) >= 2 and shape[-1] > 64:
                return out_tids[0]

    return None


def _trace_expert_blocks(
    ops: Dict[str, Any],
    _execution_order: List[str],  # unused, kept for API compatibility
    tensors: Dict[str, Any],
    _consumer_map: Dict[str, List[str]],  # unused, kept for API compatibility
    producer_map: Dict[str, str],
    moe_op_uids: Set[str],
    _topk_uid: str,  # unused, kept for API compatibility
    num_experts: int = 0,
) -> Tuple[Dict[str, List[str]], int, Optional[str]]:
    """
    Extract expert weight tensor IDs from MM ops in the MoE subgraph.

    Returns:
        expert_weight_ids: {"gate": [...], "up": [...], "down": [...]} ordered by expert ID
        num_experts_found: number of distinct experts with ops in graph
        hidden_states_tid: tensor ID of the hidden_states input to experts

    NOTE: last_scatter_tid is now found by _find_moe_output() which uses data-driven
    detection instead of hardcoded op_type matching.
    """
    # Pattern: param::block.X.ffn.expert.Y.{gate,up,down}.weight
    # This pattern matches:
    #   - DeepSeek v1/v2: param::block.1.ffn.expert.0.gate.weight
    #   - Mixtral (future): param::model.layers.1.block_sparse_moe.experts.0.gate_proj.weight
    # Group 1 = expert_id, Group 2 = gate|up|down
    weight_pattern = re.compile(
        r"param::.*\.expert[s]?\.(\d+)\.(gate|up|down)(?:_proj)?\.weight"
    )

    # Collect weight tensor IDs from mm ops in the MoE subgraph
    expert_weights: Dict[int, Dict[str, str]] = {}  # expert_id -> {gate: tid, up: tid, down: tid}
    hidden_states_tid = None

    for op_uid in moe_op_uids:
        op_data = ops.get(op_uid)
        if op_data is None:
            continue

        # Find weight references in mm ops
        # mm consumes [activation, transposed_weight]. The transposed_weight comes from
        # an aten::t op whose input is param::block.X.ffn.expert.Y.{gate,up,down}.weight
        if op_data.get("op_type") == "aten::mm":
            # Check both inputs of mm for weight references
            attrs = op_data.get("attributes", {})
            for arg in attrs.get("args", []):
                if arg.get("type") != "tensor":
                    continue
                tid = arg.get("tensor_id", "")
                # The mm input is the t output (e.g. aten.t::12::out_0)
                # Trace back to find the param:: weight
                producer_uid = producer_map.get(tid)
                if producer_uid is None:
                    continue
                producer_data = ops.get(producer_uid, {})
                if producer_data.get("op_type") == "aten::t":
                    # The transpose input is the weight tensor
                    t_input = _get_input_tensor_id(producer_data, 0)
                    if t_input is None:
                        continue
                    m = weight_pattern.match(t_input)
                    if m:
                        expert_id = int(m.group(1))  # Group 1 = expert ID
                        proj_type = m.group(2)       # Group 2 = gate|up|down
                        if expert_id not in expert_weights:
                            expert_weights[expert_id] = {}
                        expert_weights[expert_id][proj_type] = t_input

        # Find hidden_states: input to an index op that gathers expert tokens
        # Must be float (not int64 routing tensors), large hidden_dim,
        # and produced OUTSIDE the MoE subgraph (not an intermediate routing tensor).
        if op_data.get("op_type") == "aten::index" and hidden_states_tid is None:
            input_tid = _get_input_tensor_id(op_data, 0)
            if input_tid is not None and not input_tid.startswith("param::"):
                tdata = tensors.get(input_tid, {})
                shape = tdata.get("shape", [])
                dtype = tdata.get("dtype", "")
                # Hidden states: 2D [seq, hidden] or 3D [batch, seq, hidden]
                # with large last dim (hidden_dim, e.g. 2048), float dtype,
                # and produced by a non-MoE op (not an intermediate routing tensor)
                if (len(shape) in (2, 3) and shape[-1] > 64
                        and "int" not in dtype
                        and producer_map.get(input_tid) not in moe_op_uids):
                    hidden_states_tid = input_tid

    # Build ordered weight ID lists (0..num_experts-1)
    num_experts_found = len(expert_weights)
    gate_ids = []
    up_ids = []
    down_ids = []

    # Use num_experts from topk input shape (authoritative), fall back to max_expert_id
    max_expert_id = max(expert_weights.keys()) if expert_weights else 0
    total_experts = num_experts if num_experts > 0 else (max_expert_id + 1)

    # Derive block/layer identifier from any weight to construct missing expert IDs
    # The weight pattern is e.g., "param::block.1.ffn.expert.0.gate.weight"
    # We need to extract "block.1" to reconstruct missing expert paths
    block_prefix = None
    block_pattern = re.compile(r"param::(.*?)\.expert[s]?\.\d+")
    for projs in expert_weights.values():
        for tid in projs.values():
            m = block_pattern.match(tid)
            if m:
                block_prefix = m.group(1)  # e.g., "block.1.ffn" or "model.layers.1.block_sparse_moe"
                break
        if block_prefix is not None:
            break

    # Get reference shapes from any existing expert (for missing expert tensor entries)
    ref_shapes: Dict[str, list] = {}
    for projs in expert_weights.values():
        for proj_type, tid in projs.items():
            if proj_type not in ref_shapes:
                tdata = tensors.get(tid, {})
                if "shape" in tdata:
                    ref_shapes[proj_type] = tdata["shape"]
        if len(ref_shapes) == 3:
            break

    for expert_id in range(total_experts):
        if expert_id in expert_weights:
            projs = expert_weights[expert_id]
            # Some experts may be partially traced (e.g., gate+up present but down
            # was not activated during trace). Fill missing projections from pattern.
            gate_tid = projs.get("gate") or (
                f"param::{block_prefix}.expert.{expert_id}.gate.weight"
                if block_prefix else "")
            up_tid = projs.get("up") or (
                f"param::{block_prefix}.expert.{expert_id}.up.weight"
                if block_prefix else "")
            down_tid = projs.get("down") or (
                f"param::{block_prefix}.expert.{expert_id}.down.weight"
                if block_prefix else "")
            gate_ids.append(gate_tid)
            up_ids.append(up_tid)
            down_ids.append(down_tid)

            # Ensure tensor entries exist for synthesized IDs
            for proj_type, tid in [("gate", gate_tid), ("up", up_tid), ("down", down_tid)]:
                if tid and tid not in tensors:
                    tensors[tid] = {
                        "shape": ref_shapes.get(proj_type, []),
                        "dtype": "float32",
                        "weight_name": tid[len("param::"):],
                    }
        elif block_prefix is not None:
            # Expert absent from graph (never activated during trace)
            # Construct tensor ID from pattern — weights ARE in the checkpoint
            gate_tid = f"param::{block_prefix}.expert.{expert_id}.gate.weight"
            up_tid = f"param::{block_prefix}.expert.{expert_id}.up.weight"
            down_tid = f"param::{block_prefix}.expert.{expert_id}.down.weight"
            gate_ids.append(gate_tid)
            up_ids.append(up_tid)
            down_ids.append(down_tid)

            # Ensure tensor entries exist so they get arena slots
            for proj_type, missing_tid in [("gate", gate_tid), ("up", up_tid), ("down", down_tid)]:
                if missing_tid not in tensors:
                    tensors[missing_tid] = {
                        "shape": ref_shapes.get(proj_type, []),
                        "dtype": "float32",
                        "weight_name": missing_tid[len("param::"):],  # strip param:: prefix
                    }

    return (
        {"gate": gate_ids, "up": up_ids, "down": down_ids},
        num_experts_found,
        hidden_states_tid,
    )


def _count_total_experts(
    tensors: Dict[str, Any],
    topk_uid: str,
    ops: Dict[str, Any],
) -> int:
    """
    Derive num_experts from the gate_scores tensor shape (dim -1).
    This is the softmax output shape's last dimension.

    Falls back to counting weight tensors if shape unavailable.
    """
    topk_data = ops.get(topk_uid, {})
    input_shapes = topk_data.get("input_shapes", [])
    if input_shapes and len(input_shapes[0]) >= 2:
        # gate_scores shape: [seq_len, num_experts]
        return input_shapes[0][-1]

    # Fallback: count from weight tensor IDs
    weight_pattern = re.compile(r"param::block\.\d+\.ffn\.expert\.(\d+)\.gate\.weight")
    expert_ids = set()
    for tid in tensors:
        m = weight_pattern.match(tid)
        if m:
            expert_ids.add(int(m.group(1)))
    return len(expert_ids) if expert_ids else 0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _extract_topk_k(op_data: Dict[str, Any]) -> Optional[int]:
    """Extract k value from topk op attributes."""
    attrs = op_data.get("attributes", {})
    args = attrs.get("args", [])
    # topk args: [input_tensor, k, dim, largest, sorted]
    for arg in args:
        if arg.get("type") == "scalar" and isinstance(arg.get("value"), int):
            return arg["value"]
    # Also check top-level attributes
    return attrs.get("k")


def _get_input_tensor_id(op_data: Dict[str, Any], idx: int) -> Optional[str]:
    """Get tensor_id from op's args at position idx."""
    attrs = op_data.get("attributes", {})
    args = attrs.get("args", [])
    tensor_idx = 0
    for arg in args:
        if arg.get("type") == "tensor":
            if tensor_idx == idx:
                return arg.get("tensor_id")
            tensor_idx += 1
        elif arg.get("type") == "tensor_tuple":
            # For index ops, tensor_tuple contains the index tensors
            if tensor_idx == idx:
                tids = arg.get("tensor_ids", [])
                return tids[0] if tids else None
            tensor_idx += 1
    # Also check input_tensor_ids
    input_tids = op_data.get("input_tensor_ids", [])
    if idx < len(input_tids):
        return input_tids[idx]
    return None


def _build_consumer_map(
    ops: Dict[str, Any],
    execution_order: List[str],
) -> Dict[str, List[str]]:
    """Build map: tensor_id -> list of op_uids that consume it."""
    consumer_map: Dict[str, List[str]] = {}
    for op_uid in execution_order:
        op_data = ops.get(op_uid)
        if op_data is None:
            continue
        # Collect all tensor_ids referenced in args
        for tid in _collect_input_tids(op_data):
            if tid not in consumer_map:
                consumer_map[tid] = []
            consumer_map[tid].append(op_uid)
    return consumer_map


def _build_producer_map(
    ops: Dict[str, Any],
    execution_order: List[str],
) -> Dict[str, str]:
    """Build map: tensor_id -> op_uid that produces it."""
    producer_map: Dict[str, str] = {}
    for op_uid in execution_order:
        op_data = ops.get(op_uid)
        if op_data is None:
            continue
        for out_tid in op_data.get("output_tensor_ids", []):
            producer_map[out_tid] = op_uid
    return producer_map


def _collect_input_tids(op_data: Dict[str, Any]) -> List[str]:
    """Collect all input tensor IDs from op's args."""
    tids = []
    attrs = op_data.get("attributes", {})
    args = attrs.get("args", [])
    for arg in args:
        if arg.get("type") == "tensor":
            tid = arg.get("tensor_id")
            if tid:
                tids.append(tid)
        elif arg.get("type") == "tensor_tuple":
            for tid in arg.get("tensor_ids", []):
                tids.append(tid)
    # Also from input_tensor_ids if present
    for tid in op_data.get("input_tensor_ids", []):
        if tid not in tids:
            tids.append(tid)
    return tids
