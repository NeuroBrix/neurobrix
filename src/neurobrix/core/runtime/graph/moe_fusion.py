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

from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
import re
import os

from neurobrix.nbx.neurotax import SynonymRegistry as _NeuroTax

# The expert weights' key tokens, AS THE PARSER NAMES THEM — the neurotaxe has one reader.
# Spelled here once, a renamed canonical token (the SwiGLU gate's `gate` -> `ffn_gate`,
# 2026-09-26) would leave this reader matching nothing and the fusion silently off.
_EXPERT = _NeuroTax.resolve("experts")
_EXPERT_PROJ = {role: _NeuroTax.resolve(vendor) for role, vendor in
                (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj"))}
_EXPERT_ROLE = {token: role for role, token in _EXPERT_PROJ.items()}


def _expert_tid(prefix: str, expert_id: int, role: str) -> str:
    return f"param::{prefix}.{_EXPERT}.{expert_id}.{_EXPERT_PROJ[role]}.weight"


def detect_and_fuse_moe(dag: Dict[str, Any], family: str, norm_topk_prob: bool = True,
                        declared: bool = False,
                        refusals: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Detect MoE patterns in DAG and replace with fused ops.

    Guards:
        - family "llm" (the historical fast path) OR an EXPLICIT MoE
          declaration (`declared=True` — the flow read num_experts > 1
          from lm_config, registry-driven). MoE LMs are packaged under
          other families too (Qwen3-Omni thinker → multimodal); a family
          branch alone on this universal primitive silently skipped them
          and the graph ran with only the trace-fired experts → garbage.
          The structural pattern-match below remains the correctness
          authority either way.
        - Must find topk ops with k > 1 (eliminates dense LLMs)

    Args:
        dag: The TensorDAG dict (mutated in-place)
        family: Model family from manifest ("llm", "image", etc.)
        norm_topk_prob: Whether to normalize routing weights after topk selection.
            DeepSeek: False (raw softmax scores). Qwen3/Mixtral: True (default).
        declared: The caller declared this component a MoE LM (lm_config
            num_experts > 1) — admits non-llm families to the pass.
        refusals: When given, receives {topk op_uid: reason} for every router
            the stacked-expert matcher declined (the per-expert matcher's
            refusals are silent by design: most of them are "no per-expert
            weights here", which the stacked matcher then answers).

    Returns:
        The DAG (same reference, possibly mutated)
    """
    if os.environ.get("NBX_DISABLE_MOE_FUSION"):
        return dag
    if family != "llm" and not declared:
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

    # Multi-gate routers (one MoE layer driven by N sibling gates whose
    # topk results are blended by per-token modality masks). Empty dict on
    # every single-gate model — the legacy path below is then bit-identical.
    blends = _detect_gate_blends(ops, execution_order, consumer_map,
                                 tensors, moe_topk_uids)

    fused_count = 0
    ops_removed_total = 0
    dead_outputs_total = 0
    declined: Dict[str, str] = {}

    for topk_uid in moe_topk_uids:
        blend = blends.get(topk_uid)
        if blend is not None and blend.representative != topk_uid:
            # Sibling gate of an already-handled multi-gate layer: its
            # routing tensors are consumed by the representative's fused op.
            continue
        result = _fuse_one_moe_layer(
            dag, ops, execution_order, tensors,
            consumer_map, producer_map, topk_uid,
            norm_topk_prob=norm_topk_prob,
            blend=blend,
        )
        if result is None and blend is None:
            # Stacked expert parameters (two [E, ., .] slabs, no per-expert
            # tensors): softmax after topk with a sorted dispatch, or softmax
            # before topk with the dense all-experts combine. Structurally
            # disjoint from the walk above, so it gets its own matcher — and
            # its refusals carry their reason.
            result = _fuse_one_stacked_layer(
                dag, ops, execution_order, tensors,
                consumer_map, producer_map, topk_uid,
                declared_norm=norm_topk_prob if declared else None)
            if isinstance(result, Declined):
                declined[topk_uid] = result.reason
                if refusals is not None:
                    refusals[topk_uid] = result.reason
                if os.environ.get("NBX_DEBUG") or os.environ.get("NBX_MOE_FUSION_LOG"):
                    print(f"[MoE Fusion] {topk_uid}: not fused — {result.reason}")
                result = None
        if result is not None:
            removed_count, fused_uid, fused_op, dead_outputs_count = result
            fused_count += 1
            ops_removed_total += removed_count
            dead_outputs_total += dead_outputs_count

            # Incremental map update: add fused op's inputs/outputs
            for in_tid in _collect_input_tids(fused_op):
                if in_tid not in consumer_map:
                    consumer_map[in_tid] = []
                consumer_map[in_tid].append(fused_uid)
            for out_tid in fused_op.get("output_tensor_ids", []):
                producer_map[out_tid] = fused_uid

    # A DECLARED MoE (the registry says num_experts > 1) whose router neither
    # matcher fuses would run as traced: trace-frozen routing, or every expert
    # for every token — in silence. Refused by name instead.
    if declared and declined:
        named = "; ".join(f"{u}: {r}" for u, r in list(declined.items())[:3])
        raise RuntimeError(
            f"ZERO FALLBACK: [MoE Fusion] {len(declined)} router(s) of a declared MoE "
            f"are fused by no matcher — {named}. Extend the matcher to this block "
            "rather than letting the traced routing run.")

    # Update DAG
    dag["ops"] = ops
    dag["execution_order"] = execution_order

    # Criterion (H): log output-sweep count. Expect 0 on dense LLMs (TinyLlama),
    # ~384 × num_blocks on MoE (e.g. Qwen3-30B-A3B: 18432 ± 10%).
    if os.environ.get("NBX_DEBUG") or os.environ.get("NBX_MOE_FUSION_LOG"):
        print(
            f"[MoE Fusion] fused_layers={fused_count} "
            f"ops_removed={ops_removed_total} "
            f"output_sweep_removed={dead_outputs_total}"
        )

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
    blend: Optional["GateBlend"] = None,
) -> Optional[Tuple[int, str, Dict[str, Any], int]]:
    """
    Fuse one MoE layer starting from a topk op.

    Returns (ops_removed, fused_uid, fused_op_data) or None if fusion failed/skipped.

    COMPATIBILITY:
    - DeepSeek v1 (deepseek-moe-16b-chat): ~1300 ops per layer, mm ops in subgraph → FUSE
    - DeepSeek v2 (DeepSeek-Coder-V2): ~11 ops per layer, mm ops NOT in subgraph → SKIP
      V2 uses a different architecture where expert computation is decoupled from routing.

    MULTI-GATE (`blend` not None): the layer has several sibling gates whose
    topk results are blended in-graph (see `_detect_gate_blends`). The gate
    and blend ops STAY in the DAG; the walk is seeded from the blended
    routing tensors and the fused op binds them directly instead of a single
    gate's raw scores.
    """
    topk_data = ops[topk_uid]

    # --- Extract MoE parameters from graph (ZERO HARDCODE) ---

    # k from topk attributes
    top_k = _extract_topk_k(topk_data)

    # Collect ALL ops that belong to this MoE layer
    moe_op_uids: Set[str] = set()

    if blend is not None:
        # Routing is computed by the graph (N gates + mask blend) and the
        # fused op consumes the RESULT. No single gate's scores are bound.
        gate_scores_tid = None
        seed_weights_tid = blend.weights_tid
        seed_indices_tid = blend.indices_tid
    else:
        # topk input = gate_scores (softmax output)
        gate_scores_tid = _get_input_tensor_id(topk_data, 0)
        if gate_scores_tid is None:
            raise RuntimeError(f"[MoE Fusion] Cannot find gate_scores input for {topk_uid}")
        # topk outputs: scores and indices
        seed_weights_tid = topk_data["output_tensor_ids"][0]
        seed_indices_tid = topk_data["output_tensor_ids"][1]
        moe_op_uids.add(topk_uid)

    # --- Trace forward from the routing tensors to find the MoE subgraph ---

    # Find the hidden_states input: trace back from the index ops
    # The gate_scores come from softmax, which comes from router mm
    # The hidden_states are the input to both the router AND the expert index ops

    # Step 1: Find sort, bincount, floor_divide, view ops after the routing
    _trace_routing_ops(ops, execution_order, consumer_map,
                       seed_weights_tid, seed_indices_tid, moe_op_uids)

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

    # ── Doomed-boundary absorption (gated-shared-expert motif) ─────────
    # A boundary op that consumes subgraph-INTERNAL tensors other than
    # the chosen exit tensor is guaranteed dead after the removal pass
    # (its producers vanish with the subgraph) and its death cascades
    # into the combine tail. Qwen3-Omni talker: the gated shared expert
    # interposes an aten::add (excluded from MOE_OP_TYPES — residual
    # escape) between the last index_add and the reshape, so the last
    # combine op lands on the boundary consuming removed internals —
    # measured effect: 1/20 layers fused, ~41k downstream ops killed,
    # the graph output left with no producer in execution_order.
    # Absorb exactly those ops back into the subgraph and rebind the
    # exit tensor to their output. The trigger condition is precisely
    # "this op would be declared dead by the removal pass", so clean
    # graphs (thinker/deepseek/qwen3: a view boundary consuming ONLY
    # the exit tensor) never match — fused sets stay bit-identical
    # (verified on thinker 48/48, deepseek-moe 27/27, V2-Lite 26/26,
    # Qwen3-30B 48/48, Ming multi-gate 28/28).
    _absorb_changed = True
    while _absorb_changed:
        _absorb_changed = False
        for b_uid in sorted(boundary_ops,
                            key=lambda u: execution_order.index(u)
                            if u in execution_order else -1):
            b_op = ops.get(b_uid, {})
            b_outs = b_op.get("output_tensor_ids", [])
            if len(b_outs) != 1:
                continue
            in_tids = _collect_input_tids(b_op)
            internal_ins = [t for t in in_tids
                            if producer_map.get(t) in moe_op_uids
                            or t == last_scatter_tid]
            doomed = any(t != last_scatter_tid for t in internal_ins)
            if not doomed:
                continue
            # Absorption requires every non-weight input to be subgraph-
            # internal (or the exit tensor) — an op with a live external
            # activation input (the shared-expert combine add) stays out.
            external = [t for t in in_tids
                        if t not in internal_ins
                        and producer_map.get(t) is not None
                        and producer_map.get(t) not in moe_op_uids]
            if external:
                continue
            moe_op_uids.add(b_uid)
            boundary_ops.discard(b_uid)
            last_scatter_tid = b_outs[0]
            _absorb_changed = True
            break

    # --- Create fused op ---
    # Extract block identifier from parent_module of topk
    parent = topk_data.get("parent_module", "")
    # e.g. "block.1.ffn.router" -> "block.1"
    block_match = re.match(r"(block\.\d+)", parent)
    block_id = block_match.group(1) if block_match else topk_uid

    fused_uid = f"moe_fused::{block_id}"

    # Routing tensors the fused op reads: a single gate's raw scores (legacy)
    # or the blended (indices, weights) pair produced in-graph (multi-gate).
    routing_tids = ([blend.indices_tid, blend.weights_tid] if blend is not None
                    else [gate_scores_tid])

    # Build args list for liveness analysis — all input tensors must be declared
    # so _extract_input_slots_from_dag can track them and prevent premature GC
    liveness_args = [{"type": "tensor", "tensor_id": hidden_states_tid}]
    liveness_args += [{"type": "tensor", "tensor_id": t} for t in routing_tids]
    # All expert weight tensors (64 × 3 = 192 tensors)
    for i in range(num_experts):
        liveness_args.append({"type": "tensor", "tensor_id": expert_weight_ids["gate"][i]})
        liveness_args.append({"type": "tensor", "tensor_id": expert_weight_ids["up"][i]})
        liveness_args.append({"type": "tensor", "tensor_id": expert_weight_ids["down"][i]})

    # input_tensor_ids for native mode liveness tracking (_compute_last_use scans this)
    all_input_tids = [hidden_states_tid] + list(routing_tids)
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

    if blend is not None:
        # Pre-computed routing: the engine reads these instead of running
        # topk on gate scores. The per-gate normalization already happened
        # in-graph, so the runtime NEVER re-normalizes in this mode
        # (norm_topk_prob is ignored — see the runtime dispatchers).
        fused_op["attributes"]["topk_indices_tid"] = blend.indices_tid
        fused_op["attributes"]["topk_weights_tid"] = blend.weights_tid
        fused_op["attributes"]["gate_group"] = list(blend.members)

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

    # DATA-DEPENDENCE CLAMP. `last_moe_pos` is the tail of the traced
    # subgraph, which is NOT always the op that produced the MoE result: a
    # trace can leave a DEAD routing by-product after the combine (Ming-Lite
    # -Omni emits `blended_topk_idx.view(batch, seq, k)` past the weighted
    # sum). The heuristic above then places the fused op AFTER the surviving
    # op that reads its output. Clamp to the true dependence window:
    #   latest producer of a fused input  <  fused op  <=  earliest surviving
    #                                                      consumer of its output
    new_pos = {uid: i for i, uid in enumerate(new_order)}

    earliest_consumer = len(new_order)
    for c_uid in consumer_map.get(last_scatter_tid, []):
        i = new_pos.get(c_uid)
        if i is not None and i < earliest_consumer:
            earliest_consumer = i

    latest_producer = -1
    for in_tid in all_input_tids:
        p_uid = producer_map.get(in_tid)
        if p_uid is None:
            continue
        i = new_pos.get(p_uid)
        if i is not None and i > latest_producer:
            latest_producer = i

    if insert_idx > earliest_consumer:
        insert_idx = earliest_consumer
    if insert_idx <= latest_producer:
        insert_idx = latest_producer + 1
    if insert_idx > earliest_consumer:
        raise RuntimeError(
            f"[MoE Fusion] Cannot place {fused_uid}: an input producer at "
            f"position {latest_producer} runs after the consumer of the MoE "
            f"output at position {earliest_consumer}. The traced subgraph is "
            "not a contiguous dependence window."
        )

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
                    # This dead op's outputs are also removed — EXCEPT the
                    # fused op's own outputs (same exemption as the seeding
                    # above): cascading a fused output tid into
                    # removed_producers self-poisons every consumer of the
                    # fused result.
                    for out_tid in op_data.get("output_tensor_ids", []):
                        if out_tid not in fused_output_tids:
                            removed_producers.add(out_tid)
                    changed = True
                    break

    if dead_ops:
        new_order = [uid for uid in new_order if uid not in dead_ops]

    # Pass 2 requires the fused_op to be discoverable via `ops` so its
    # inputs (hidden_states_tid, gate_scores_tid, expert weight tids) are
    # counted as live consumers during the local_consumer_map build below.
    # Inserting it here (before the sweep) rather than at the very end of
    # the function doesn't affect the rest of the flow — new_order still
    # references fused_uid, and the final execution_order swap still
    # happens atomically.
    ops[fused_uid] = fused_op

    # ═══════════════════════════════════════════════════════════════════════════
    # Pass 2: Output-side dead-op sweep (architectural fix for MoE weight retention)
    # ═══════════════════════════════════════════════════════════════════════════
    # After MoE fusion, ops like `aten::t` on expert weights lose their only
    # consumer (the aten::mm that was fused into custom::moe_fused). If left in,
    # they run at execution, store a .t() view in the arena whose `_base` pins
    # the CPU-offloaded expert weight to GPU memory under zero3 pipelining.
    # Pass 1 (input-side) misses them because their INPUTS are still weight
    # params (not in removed_producers) — only their OUTPUT consumer is gone.
    # Fixed-point iteration so chains (t → t → mm) collapse in one call.
    #
    # PROTECTS: fused_uid, shared_expert paths, DAG-level output tids.
    active_ops: Set[str] = set(new_order) - dead_ops
    graph_outputs: Set[str] = set(dag.get("output_tensor_ids", []))

    local_consumer_map: Dict[str, Set[str]] = {}
    for uid in active_ops:
        op_data = ops.get(uid, {})
        for in_tid in _collect_input_tids(op_data):
            local_consumer_map.setdefault(in_tid, set()).add(uid)

    dead_outputs_count = 0
    changed = True
    while changed:
        changed = False
        for uid in list(active_ops):
            if uid == fused_uid:
                continue
            op_data = ops.get(uid, {})
            parent_module = op_data.get("parent_module", "")
            if "shared_expert" in parent_module or "shared" in parent_module:
                continue
            out_tids = op_data.get("output_tensor_ids", [])
            if not out_tids:
                continue
            all_dead = True
            for out_tid in out_tids:
                if out_tid in graph_outputs:
                    all_dead = False
                    break
                live_consumers = local_consumer_map.get(out_tid, set()) & active_ops
                live_consumers.discard(uid)
                if live_consumers:
                    all_dead = False
                    break
            if all_dead:
                dead_ops.add(uid)
                active_ops.discard(uid)
                dead_outputs_count += 1
                for in_tid in _collect_input_tids(op_data):
                    local_consumer_map.get(in_tid, set()).discard(uid)
                changed = True

    if dead_outputs_count:
        new_order = [uid for uid in new_order if uid not in dead_ops]

    # Update structures (fused_op already inserted into `ops` before Pass 2).
    # Don't delete old ops from ops dict (tensors reference them), just remove from execution_order
    execution_order.clear()
    execution_order.extend(new_order)

    ops_removed = len(moe_op_uids)

    return ops_removed, fused_uid, fused_op, dead_outputs_count


# ============================================================================
# MULTI-GATE ROUTER BLEND DETECTION
# ============================================================================
#
# Some MoE layers are driven by SEVERAL sibling gates instead of one.
# Ming-Lite-Omni (BailingMoe) holds three BailingMoeGate routers per layer
# (text `router`, `image_gate`, `audio_gate`); every gate runs the full
# softmax → topk → normalize chain and the three (topk_idx, topk_weight)
# pairs are then blended by per-token modality masks through a chain of
# elementwise `mul`/`add` before the sort/scatter that drives dispatch.
#
# The blend is ordinary ATen compute over runtime inputs (the masks are
# graph inputs), so it STAYS in the DAG and the fused op binds the blend
# RESULT. Absorbing the blend into the fused op would force the universal
# MoE primitive to re-implement per-modality gate selection — model
# semantics inside a hardware primitive, forbidden.
#
# Structural criterion for "this op belongs to the routing blend" — no
# model name, no op count, no parent_module string:
#   1. op_type is a pure elementwise / dtype / shape op (no reduction, no
#      gather, no matmul) — `_BLEND_OP_TYPES`;
#   2. it has exactly ONE output and that output's shape is EXACTLY the
#      topk output shape (…, k). This is the bound that keeps the walk
#      from escaping through residual `add`s (activations are (…, hidden))
#      and through the post-dispatch combine ((…, k, hidden));
#   3. every tensor operand broadcasts into that routing shape (the
#      modality masks come in as (…, 1)).
# Two gates are siblings when their blend closures share a tensor — i.e.
# they converge into the same blended routing tensor.

_BLEND_OP_TYPES = {
    "aten::add", "aten::sub", "aten::mul", "aten::div",
    "aten::where", "aten::masked_fill",
    "aten::_to_copy", "aten::clone", "aten::detach",
    "aten::view", "aten::reshape", "aten::_unsafe_view",
}


class GateBlend(NamedTuple):
    """Resolved multi-gate routing blend for one MoE layer."""
    representative: str          # topk op_uid that carries the fusion
    members: Tuple[str, ...]     # every sibling topk op_uid (exec order)
    indices_tid: str             # blended topk indices (integer dtype)
    weights_tid: str             # blended topk weights (float dtype)


def _tensor_shape(tensors: Dict[str, Any], tid: str) -> Optional[List[Any]]:
    td = tensors.get(tid)
    if td is None:
        return None
    shape = td.get("shape")
    return list(shape) if shape is not None else None


def _broadcasts_into(shape: Optional[List[Any]],
                     target: Optional[List[Any]]) -> bool:
    """True when `shape` broadcasts into `target` (numpy right-aligned rules).

    Symbolic dims (strings) compare by identity — a symbol only broadcasts
    into itself or into a literal 1.
    """
    if shape is None or target is None:
        return False
    if len(shape) > len(target):
        return False
    for s, t in zip(reversed(shape), reversed(target)):
        if s == 1 or s == t:
            continue
        return False
    return True


def _blend_closure(
    ops: Dict[str, Any],
    consumer_map: Dict[str, List[str]],
    tensors: Dict[str, Any],
    seed_tids: List[str],
    routing_shape: List[Any],
) -> Tuple[Set[str], Set[str]]:
    """Forward closure of routing-shaped elementwise ops from one gate.

    Returns (op_uids, tensor_ids). The tensor set includes the seeds so
    that two gates that never converge produce disjoint sets.
    """
    closure_ops: Set[str] = set()
    closure_tids: Set[str] = set(seed_tids)
    frontier = list(seed_tids)

    while frontier:
        tid = frontier.pop()
        for consumer_uid in consumer_map.get(tid, []):
            if consumer_uid in closure_ops:
                continue
            op_data = ops.get(consumer_uid)
            if op_data is None:
                continue
            if op_data.get("op_type") not in _BLEND_OP_TYPES:
                continue
            out_tids = op_data.get("output_tensor_ids", [])
            if len(out_tids) != 1:
                continue
            if _tensor_shape(tensors, out_tids[0]) != routing_shape:
                continue
            operands = _collect_input_tids(op_data)
            if not all(_broadcasts_into(_tensor_shape(tensors, t), routing_shape)
                       for t in operands):
                continue
            closure_ops.add(consumer_uid)
            closure_tids.add(out_tids[0])
            frontier.append(out_tids[0])

    return closure_ops, closure_tids


def _detect_gate_blends(
    ops: Dict[str, Any],
    execution_order: List[str],
    consumer_map: Dict[str, List[str]],
    tensors: Dict[str, Any],
    topk_uids: List[str],
) -> Dict[str, GateBlend]:
    """Group sibling gates of multi-gate MoE layers.

    Returns {topk_uid: GateBlend} covering ONLY topk ops that belong to a
    group of size > 1. Single-gate models get an empty dict, so the legacy
    fusion path stays bit-identical.
    """
    closures: Dict[str, Tuple[Set[str], Set[str]]] = {}
    routing_shapes: Dict[str, List[Any]] = {}

    for topk_uid in topk_uids:
        op_data = ops.get(topk_uid, {})
        out_tids = op_data.get("output_tensor_ids", [])
        if len(out_tids) < 2:
            continue
        routing_shape = _tensor_shape(tensors, out_tids[0])
        if routing_shape is None:
            continue
        closures[topk_uid] = _blend_closure(
            ops, consumer_map, tensors, list(out_tids), routing_shape)
        routing_shapes[topk_uid] = routing_shape

    # Union-find: two gates are siblings when their closures share a tensor.
    parent: Dict[str, str] = {u: u for u in closures}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    tid_owner: Dict[str, str] = {}
    for topk_uid, (_, closure_tids) in closures.items():
        for tid in closure_tids:
            owner = tid_owner.get(tid)
            if owner is None:
                tid_owner[tid] = topk_uid
            else:
                _union(owner, topk_uid)

    grouped: Dict[str, List[str]] = {}
    for topk_uid in closures:
        grouped.setdefault(_find(topk_uid), []).append(topk_uid)

    order_index = {uid: i for i, uid in enumerate(execution_order)}
    blends: Dict[str, GateBlend] = {}

    for members in grouped.values():
        if len(members) < 2:
            continue  # single gate — legacy path
        members.sort(key=lambda u: order_index.get(u, 0))
        representative = members[0]

        shapes = {tuple(routing_shapes[u]) for u in members}
        if len(shapes) != 1:
            raise RuntimeError(
                "[MoE Fusion] Multi-gate router group with inconsistent "
                f"routing shapes {sorted(shapes)} at {members}. The gates of "
                "one MoE layer must produce identically shaped topk results."
            )

        union_ops: Set[str] = set()
        for u in members:
            union_ops |= closures[u][0]

        # Terminals: closure outputs with no consumer inside the closure —
        # the tensors the sort/scatter actually reads.
        terminals: List[str] = []
        for op_uid in union_ops:
            for out_tid in ops.get(op_uid, {}).get("output_tensor_ids", []):
                if not any(c in union_ops for c in consumer_map.get(out_tid, [])):
                    terminals.append(out_tid)

        int_terms = [t for t in terminals
                     if "int" in (tensors.get(t, {}).get("dtype") or "")]
        float_terms = [t for t in terminals
                       if "float" in (tensors.get(t, {}).get("dtype") or "")]

        if len(int_terms) != 1 or len(float_terms) != 1:
            raise RuntimeError(
                "[MoE Fusion] Multi-gate router group at "
                f"{members} does not resolve to exactly one blended index "
                f"tensor and one blended weight tensor "
                f"(int={sorted(int_terms)}, float={sorted(float_terms)}). "
                "The blend topology is unsupported — extend "
                "_detect_gate_blends (workstream P-MOE-MULTIGATE) rather than "
                "letting the trace-frozen routing run."
            )

        blend = GateBlend(
            representative=representative,
            members=tuple(members),
            indices_tid=int_terms[0],
            weights_tid=float_terms[0],
        )
        for u in members:
            blends[u] = blend

    return blends


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
    execution_order: List[str],
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
    # Pattern: param::<prefix>.<expert>.Y.<swiglu gate|up|down>.weight, the tokens the parser
    # emits (a container holds no raw vendor name). Group 1 = expert_id, group 2 = the token.
    weight_pattern = re.compile(
        rf"param::.*\.{re.escape(_EXPERT)}\.(\d+)\."
        rf"({'|'.join(re.escape(t) for t in _EXPERT_PROJ.values())})\.weight"
    )

    # Collect weight tensor IDs from mm ops in the MoE subgraph
    expert_weights: Dict[int, Dict[str, str]] = {}  # expert_id -> {gate: tid, up: tid, down: tid}
    hidden_states_tid = None

    # DETERMINISM: iterate in EXECUTION ORDER, never in `moe_op_uids` set
    # order. Set iteration over op_uid strings is randomized per process by
    # PYTHONHASHSEED, and the `hidden_states_tid` pick below is "first match
    # wins" — Qwen3-MoE offers 128 equivalent `unsqueeze(view)` candidates
    # per layer, so the bound tid (and hence the surviving op / arena slot /
    # execution_order) changed on every process. Values were unaffected (the
    # candidates are views of the same tensor) but the plan was not
    # reproducible, which breaks R28-style build/plan diffing.
    for op_uid in execution_order:
        if op_uid not in moe_op_uids:
            continue
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
                        proj_type = _EXPERT_ROLE[m.group(2)]  # the role: gate|up|down
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
    block_pattern = re.compile(rf"param::(.*?)\.{re.escape(_EXPERT)}\.\d+")
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
                _expert_tid(block_prefix, expert_id, "gate") if block_prefix else "")
            up_tid = projs.get("up") or (
                _expert_tid(block_prefix, expert_id, "up") if block_prefix else "")
            down_tid = projs.get("down") or (
                _expert_tid(block_prefix, expert_id, "down") if block_prefix else "")
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
            gate_tid = _expert_tid(block_prefix, expert_id, "gate")
            up_tid = _expert_tid(block_prefix, expert_id, "up")
            down_tid = _expert_tid(block_prefix, expert_id, "down")
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
    weight_pattern = re.compile(
        rf"param::{re.escape(_NeuroTax.resolve('layers'))}\.\d+\.{re.escape(_NeuroTax.resolve('mlp'))}"
        rf"\.{re.escape(_EXPERT)}\.(\d+)\.{re.escape(_EXPERT_PROJ['gate'])}\.weight")
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


# ============================================================================
# STACKED-EXPERT MoE (both routing orders, both slab layouts)
# ============================================================================
#
# Some MoE blocks keep their experts as TWO stacked parameters instead of
# E x 3 per-expert matrices: an input slab holding every expert's fused
# gate|up projection and an output slab holding every expert's down
# projection. Two traced forms of such a block are recognised by ONE matcher
# (`_fuse_one_stacked_layer`), each element read from the graph:
#
#   softmax AFTER topk, sorted-index dispatch (granitemoe, 24 layers):
#     topk(logits[T,E], k) -> softmax OVER THE TOP-K
#     view/sort/div/index  -> a sorted-assignment gather of hidden [T*k, H]
#     split_with_sizes     -> per-expert pieces, sizes BAKED AT TRACE TIME
#     E x (select(W_in[E,2F,H], e) -> t -> mm)          stacked input_linear
#     cat -> split -> silu -> mul                       fused gate|up halves
#     split_with_sizes -> E x (select(W_out[E,H,F], e) -> t -> mm) -> cat
#     mul by sorted routing weights -> zeros -> index_add[T,H]
#   The baked sizes are trace-time ROUTING — any other prompt routes
#   differently and the graph refuses (split_with_sizes sums mismatch).
#   granite's softmax(topk(logits)) equals softmax(logits) -> topk ->
#   renormalize EXACTLY (exp(x_i)/sum_top exp is invariant to the
#   full-softmax denominator), so the rewrite inserts one aten::_softmax on
#   the logits and sets norm_topk_prob=True — no dispatcher learns new math.
#
#   softmax BEFORE topk, dense all-experts combine (the inference form of
#   transformers' Qwen3VLMoeTextExperts, 4.57; 48 layers of
#   Qwen3-VL-30B-A3B-Thinking):
#     softmax(logits[T,E]) -> topk -> [sum + div: renormalise] -> [cast]
#     -> scatter into zeros[T,E]                        routing matrix
#     hidden -> repeat(E, 1) -> view[E,T,H] -> bmm(W_in[E,H,2F])
#     -> split(F, -1) -> silu(gate) * up -> bmm(W_out[E,F,H])
#     -> view -> mul by the routing matrix -> sum over the expert axis
#   Correct as traced — unrouted experts are multiplied by 0 — but every
#   expert is READ and COMPUTED for every token (E/k x the routed FLOPs,
#   9.94x the active bytes per decode token, measured 2026-10-04). Its
#   routing is already the fused op's own (topk of the softmax scores, then
#   the renormalisation if the graph divides by the top-k sum), so the
#   rewrite binds the softmax output as `gate_scores` and inserts nothing.
#
# Either way the fused op carries a `stacked_experts` spec and every
# dispatcher resolves its weight lists through `expert_weight_lists` below —
# zero-copy select/narrow/transpose views into the two stacked parameters.
# The spec names the slab GEOMETRY the graph read; the reader never assumes
# one (transformers itself flipped Qwen3-VL's slab from [E, H, 2F] in 4.57
# to [E, 2F, H] in 5.x).

# The spec fields every stacked fused op carries (written by the matcher,
# required by the reader — a spec without them is refused by name).
_STACKED_SPEC_FIELDS = ("input_linear_tid", "output_linear_tid", "ffn_dim",
                        "input_linear_in_axis", "gate_offset",
                        "output_linear_in_axis")


def expert_weight_lists(attrs: Dict[str, Any], fetch):
    """Resolve the fused op's per-expert (gate, up, down) weight lists.

    `fetch(tid)` returns the tensor for a tensor id (torch or NBXTensor —
    both carry select/narrow/t views). Every dispatcher consumes the
    nn.Linear layout: gate and up [F, H], down [H, F] (out x in).

    Stacked layout, from the spec the matcher read off the graph:
      * W_in[E, a, b] — per expert a 2-D matrix whose `input_linear_in_axis`
        (0 or 1) indexes the hidden (contraction) dim H; the other axis holds
        the 2F projection, the gate half at `gate_offset` (0 or F), the up
        half the other F;
      * W_out[E, c, d] — per expert, `output_linear_in_axis` indexes F (the
        down projection's contraction), the other axis H.
    A per-expert matrix stored (in x out) is returned transposed — a view.
    """
    st = attrs.get("stacked_experts")
    if not st:
        return ([fetch(t) for t in attrs["expert_gate_weight_ids"]],
                [fetch(t) for t in attrs["expert_up_weight_ids"]],
                [fetch(t) for t in attrs["expert_down_weight_ids"]])
    missing = [f for f in _STACKED_SPEC_FIELDS if f not in st]
    if missing:
        raise RuntimeError(
            f"ZERO FALLBACK: stacked expert spec lacks {missing} — the slab "
            "geometry is read from the graph by the matcher, never assumed "
            "by the reader.")
    w_in = fetch(st["input_linear_tid"])
    w_out = fetch(st["output_linear_tid"])
    if w_in is None or w_out is None:
        raise RuntimeError(
            "ZERO FALLBACK: stacked expert parameters "
            f"{st['input_linear_tid']!r} / {st['output_linear_tid']!r} not "
            "loaded — the fused op declared them as inputs; a loader that "
            "dropped them is a defect, not a fallback.")
    F = int(st["ffn_dim"])
    E = int(attrs["num_experts"])
    in_axis = int(st["input_linear_in_axis"])
    out_in_axis = int(st["output_linear_in_axis"])
    g_off = int(st["gate_offset"])
    if in_axis not in (0, 1) or out_in_axis not in (0, 1) or g_off not in (0, F):
        raise RuntimeError(
            f"ZERO FALLBACK: stacked expert spec out of range: in_axis="
            f"{in_axis}, output_in_axis={out_in_axis}, gate_offset={g_off} "
            f"(F={F}).")
    u_off = F - g_off
    proj_axis = 1 - in_axis

    def _linear(m):
        # (out x in) as stored when the contraction is axis 1; a view otherwise
        return m if in_axis == 1 else m.t()

    gate, up, down = [], [], []
    for e in range(E):
        m = w_in.select(0, e)
        gate.append(_linear(m.narrow(proj_axis, g_off, F)))
        up.append(_linear(m.narrow(proj_axis, u_off, F)))
        d = w_out.select(0, e)
        down.append(d if out_in_axis == 1 else d.t())
    return gate, up, down


class Declined:
    """A stacked-expert matcher's refusal, with its reason. Never a tuple, so
    it can never be mistaken for a fusion result."""
    __slots__ = ("reason",)

    def __init__(self, reason: str):
        self.reason = reason

    def __repr__(self) -> str:
        return f"Declined({self.reason!r})"


def _split_halves(ops, consumer_map, split_uid, tensors, F):
    """Read the gate|up halves off the graph's own split of the projection:
    returns (gate_offset, silu_uid, mul_uid) or a Declined. The gate half is
    the piece `silu` consumes; the up half is the other piece, multiplied by
    the silu output in one aten::mul."""
    sp = ops.get(split_uid) or {}
    if sp.get("op_type") != "aten::split":
        return Declined(f"the gate|up projection is not split by one aten::split "
                        f"(found {sp.get('op_type')!r})")
    outs = sp.get("output_tensor_ids", [])
    in_tid = _get_input_tensor_id(sp, 0)
    ish = _tensor_shape(tensors, in_tid)
    vals = [a.get("value") for a in sp.get("attributes", {}).get("args", [])
            if a.get("type") == "scalar"]
    size = vals[0] if vals else sp.get("attributes", {}).get("split_size")
    dim = vals[1] if len(vals) > 1 else sp.get("attributes", {}).get("dim", 0)
    if not ish or len(outs) != 2 or size != F or dim not in (-1, len(ish) - 1) \
            or ish[-1] != 2 * F:
        return Declined(f"the gate|up split is not two halves of F={F} on the "
                        f"last dim (outs={len(outs)}, size={size}, dim={dim}, "
                        f"input={ish})")
    silu = [(i, cu) for i, t in enumerate(outs) for cu in consumer_map.get(t, [])
            if ops.get(cu, {}).get("op_type") == "aten::silu"]
    if len(silu) != 1:
        return Declined(f"the gate|up halves feed {len(silu)} aten::silu (one "
                        "SwiGLU gate expected)")
    g_idx, silu_uid = silu[0]
    up_tid = outs[1 - g_idx]
    silu_out = ops[silu_uid]["output_tensor_ids"][0]
    muls = [cu for cu in consumer_map.get(silu_out, [])
            if ops.get(cu, {}).get("op_type") == "aten::mul"
            and set(_collect_input_tids(ops[cu])) == {silu_out, up_tid}]
    if len(muls) != 1 or len(consumer_map.get(silu_out, [])) != 1 \
            or consumer_map.get(up_tid, []) != muls:
        return Declined("silu(gate) and the up half do not meet in exactly one "
                        "aten::mul (not a SwiGLU)")
    return g_idx * F, silu_uid, muls[0]


def _stacked_fused_op(tensors, fused_uid, *, top_k, num_experts, F, H,
                      hidden_tid, gate_scores_tid, in_tid, out_tid, exit_tid,
                      in_axis, gate_offset, out_in_axis, norm_topk_prob,
                      extra_attrs):
    """The ONE emission of a stacked-expert custom::moe_fused op, whichever
    traced form the matcher recognised. The per-expert entries carry the
    slab's own dtype — a parameter the graph describes."""
    sdt = tensors[in_tid]["dtype"]
    syn = lambda half, e: f"{in_tid}::stacked::{half}::{e}"
    syn_down = lambda e: f"{out_tid}::stacked::down::{e}"
    gate_ids = [syn("gate", e) for e in range(num_experts)]
    up_ids = [syn("up", e) for e in range(num_experts)]
    down_ids = [syn_down(e) for e in range(num_experts)]
    for e in range(num_experts):
        tensors.setdefault(gate_ids[e], {
            "tensor_id": gate_ids[e], "shape": [F, H], "dtype": sdt,
            "is_parameter": True})
        tensors.setdefault(up_ids[e], {
            "tensor_id": up_ids[e], "shape": [F, H], "dtype": sdt,
            "is_parameter": True})
        tensors.setdefault(down_ids[e], {
            "tensor_id": down_ids[e], "shape": [H, F], "dtype": sdt,
            "is_parameter": True})
    all_input_tids = [hidden_tid, gate_scores_tid, in_tid, out_tid]
    attrs = {
        "args": [{"type": "tensor", "tensor_id": t} for t in all_input_tids],
        "kwargs": {},
        "gate_scores_tid": gate_scores_tid,
        "hidden_states_tid": hidden_tid,
        "expert_gate_weight_ids": gate_ids,
        "expert_up_weight_ids": up_ids,
        "expert_down_weight_ids": down_ids,
        "stacked_experts": {"input_linear_tid": in_tid,
                            "output_linear_tid": out_tid,
                            "ffn_dim": F,
                            "input_linear_in_axis": in_axis,
                            "gate_offset": gate_offset,
                            "output_linear_in_axis": out_in_axis},
        "top_k": top_k,
        "num_experts": num_experts,
        "norm_topk_prob": norm_topk_prob,
    }
    attrs.update(extra_attrs)
    return {
        "op_uid": fused_uid,
        "op_type": "custom::moe_fused",
        "output_tensor_ids": [exit_tid],
        "input_tensor_ids": all_input_tids,
        "output_shapes": [tensors.get(exit_tid, {}).get("shape", [])],
        "attributes": attrs,
    }


def _fuse_one_stacked_layer(dag, ops, execution_order, tensors,
                            consumer_map, producer_map, topk_uid,
                            declared_norm=None):
    """Match and fuse one stacked-expert MoE layer, either routing order.
    Returns like `_fuse_one_moe_layer`, or a `Declined` naming why this is
    not such a block — every check REFUSES rather than guessing."""
    topk_data = ops[topk_uid]
    outs = topk_data.get("output_tensor_ids", [])
    if len(outs) < 2:
        return Declined("the top-k has no (scores, indices) pair")
    # The routing order is read from the graph: a softmax that CONSUMES the
    # top-k scores (softmax after topk), or a softmax the top-k READS.
    sm = [u for u in consumer_map.get(outs[0], [])
          if ops.get(u, {}).get("op_type") == "aten::_softmax"]
    if len(sm) == 1:
        return _fuse_softmax_after_topk_layer(
            dag, ops, execution_order, tensors, consumer_map, producer_map,
            topk_uid)
    return _fuse_softmax_first_dense_layer(
        dag, ops, execution_order, tensors, consumer_map, producer_map,
        topk_uid, declared_norm=declared_norm)


def _fuse_softmax_after_topk_layer(dag, ops, execution_order, tensors,
                                   consumer_map, producer_map, topk_uid):
    """Match and fuse one softmax-after-topk, sorted-dispatch stacked block
    (granite). Returns like `_fuse_one_moe_layer`, or a `Declined` when this
    is not that block — every check below REFUSES to fuse rather than
    guessing, and an unfused granite graph then fails loudly at its baked
    split sizes."""
    D = Declined
    topk_data = ops[topk_uid]
    k = _extract_topk_k(topk_data)
    if k is None or k <= 1:
        return D("the top-k selects at most one expert")
    logits_tid = _get_input_tensor_id(topk_data, 0)
    lsh = _tensor_shape(tensors, logits_tid)
    if not lsh or len(lsh) != 2:
        return D(f"the top-k input is not a rank-2 [tokens, experts] tensor ({lsh})")
    num_experts = int(lsh[1])
    outs = topk_data.get("output_tensor_ids", [])
    if len(outs) < 2:
        return D("the top-k has no (scores, indices) pair")
    scores_tid, idx_tid = outs[0], outs[1]

    # The marker of this order: softmax CONSUMES the topk scores.
    sm = [u for u in consumer_map.get(scores_tid, [])
          if ops.get(u, {}).get("op_type") == "aten::_softmax"]
    if len(sm) != 1:
        return D("no single softmax consumes the top-k scores")

    # Forward walk from the routing outputs to the index_add join.
    interior = {topk_uid}
    join_uid = None
    frontier = [scores_tid, idx_tid]
    seen = set()
    while frontier:
        tid = frontier.pop()
        if tid in seen:
            continue
        seen.add(tid)
        for cu in consumer_map.get(tid, []):
            if cu in interior:
                continue
            cop = ops.get(cu)
            if cop is None:
                return D(f"the routing walk meets an op with no record ({cu})")
            interior.add(cu)
            if cop.get("op_type") == "aten::index_add":
                if join_uid is not None and join_uid != cu:
                    return D(f"two index_add joins ({join_uid}, {cu})")
                join_uid = cu
                continue
            for ot in cop.get("output_tensor_ids", []):
                frontier.append(ot)
            if len(interior) > 4000:
                return D("the routing walk exceeds 4000 ops without a join")
    if join_uid is None:
        return D("the sorted dispatch reaches no aten::index_add join")

    # Backward absorption: producers of interior inputs that are pure
    # weight-side chains (select/t of a parameter) plus the zeros seed.
    hidden_tid = None
    stacked_tids = []
    changed = True
    while changed:
        changed = False
        for uid in list(interior):
            for tid in _collect_input_tids(ops[uid]):
                pu = producer_map.get(tid)
                if pu is None or pu in interior:
                    continue
                pop_ = ops.get(pu, {})
                pt = pop_.get("op_type")
                if pt in ("aten::t", "aten::select", "aten::zeros"):
                    interior.add(pu)
                    changed = True
                    if pt == "aten::select":
                        src = _get_input_tensor_id(pop_, 0)
                        if src is not None and src not in stacked_tids:
                            stacked_tids.append(src)

    # The gather that brings hidden states in names the block's activation
    # input: an aten::index whose first input is rank-2 and NOT interior.
    for uid in interior:
        if ops[uid].get("op_type") != "aten::index":
            continue
        cand = _get_input_tensor_id(ops[uid], 0)
        csh = _tensor_shape(tensors, cand)
        if csh and len(csh) == 2 and producer_map.get(cand) not in interior:
            hidden_tid = cand
            break
    if hidden_tid is None:
        return D("no rank-2 gather brings the hidden states in")

    # Two stacked parameters: [E, 2F, H] (input_linear) and [E, H, F].
    if len(stacked_tids) != 2:
        return D(f"{len(stacked_tids)} stacked parameters selected (two expected)")
    sh = {t: _tensor_shape(tensors, t) for t in stacked_tids}
    if any(v is None or len(v) != 3 or v[0] != num_experts
           for v in sh.values()):
        return D(f"a stacked parameter is not [E={num_experts}, ., .]: {sh}")
    hsh = _tensor_shape(tensors, hidden_tid)
    H = int(hsh[1])
    in_tid = out_tid_p = None
    for t, v in sh.items():
        if v[2] == H:
            in_tid = t
        elif v[1] == H:
            out_tid_p = t
    if in_tid is None or out_tid_p is None or in_tid == out_tid_p:
        return D(f"the stacked parameters do not resolve to an input and an "
                 f"output slab against H={H}: {sh}")
    F = int(sh[out_tid_p][2])
    if int(sh[in_tid][1]) != 2 * F:
        return D(f"the input slab {sh[in_tid]} does not hold 2F={2 * F} rows")
    if not tensors.get(in_tid, {}).get("dtype"):
        return D(f"the input slab {in_tid} carries no dtype in the graph")

    # The gate|up halves, read from the graph's own split of the projection.
    _splits = [u for u in interior
               if ops[u].get("op_type") == "aten::split"
               and (_tensor_shape(tensors, _get_input_tensor_id(ops[u], 0))
                    or [None])[-1] == 2 * F]
    if len(_splits) != 1:
        return D(f"{len(_splits)} aten::split of the 2F={2 * F} projection "
                 "(one expected)")
    _halves = _split_halves(ops, consumer_map, _splits[0], tensors, F)
    if isinstance(_halves, Declined):
        return _halves
    gate_offset = _halves[0]

    # Every remaining external input must be the hidden states or the
    # routing logits — anything else means this is not the block we know.
    for uid in interior:
        for tid in _collect_input_tids(ops[uid]):
            pu = producer_map.get(tid)
            if pu in interior:
                continue
            if tid in (hidden_tid, logits_tid, in_tid, out_tid_p):
                continue
            if pu is None and tensors.get(tid, {}).get("is_parameter"):
                return D(f"an unexpected parameter {tid} enters the block")
            if pu is not None:
                return D(f"a live external activation {tid} enters the block")

    join_out = ops[join_uid]["output_tensor_ids"][0]

    # --- the inserted softmax (granite's exact math, see header) ---
    sm_uid = f"moe_softmax::{topk_uid}"
    sm_out = f"{sm_uid}::out_0"
    ldt = tensors.get(logits_tid, {}).get("dtype", "bfloat16")
    ops[sm_uid] = {
        "op_uid": sm_uid, "op_type": "aten::_softmax",
        "input_tensor_ids": [logits_tid],
        "output_tensor_ids": [sm_out],
        "input_shapes": [list(lsh)], "output_shapes": [list(lsh)],
        "input_dtypes": [ldt], "output_dtypes": [ldt],
        "device": ops[topk_uid].get("device", "cuda"),
        "attributes": {"args": [
            {"type": "tensor", "tensor_id": logits_tid},
            {"type": "scalar", "value": -1},
            {"type": "scalar", "value": False},
        ], "kwargs": {}, "dim": -1, "half_to_float": False},
    }
    tensors[sm_out] = {
        "tensor_id": sm_out, "shape": list(lsh), "dtype": ldt,
        "device": tensors.get(logits_tid, {}).get("device", "cuda"),
        "producer_op_uid": sm_uid, "output_index": 0,
        "consumer_op_uids": [],
    }

    # --- the fused op, stacked spec ---
    fused_uid = f"moe_fused::{topk_uid}"
    fused_op = _stacked_fused_op(
        tensors, fused_uid, top_k=k, num_experts=num_experts, F=F, H=H,
        hidden_tid=hidden_tid, gate_scores_tid=sm_out, in_tid=in_tid,
        out_tid=out_tid_p, exit_tid=join_out,
        # select(W_in, e) -> t -> mm: the per-expert matrix is (out x in)
        in_axis=1, gate_offset=gate_offset, out_in_axis=1,
        # The rewrite's own math: softmax(topk(x)) == renormalised topk(softmax(x)),
        # so this op REQUIRES the renormalisation; it is owned by the rewrite, not
        # by the registry's declaration, and the executor's patch loop leaves it.
        norm_topk_prob=True,
        extra_attrs={"routing_rewritten": "softmax_after_topk"})
    ops[fused_uid] = fused_op

    # --- rebuild order: softmax + fused op sit where the topk sat ---
    new_order = []
    for uid in execution_order:
        if uid == topk_uid:
            new_order.append(sm_uid)
            new_order.append(fused_uid)
            continue
        if uid in interior:
            continue
        new_order.append(uid)
    removed = [u for u in interior]
    for u in removed:
        ops.pop(u, None)
    execution_order[:] = new_order
    return (len(removed), fused_uid, fused_op, 0)


# Pure re-indexing ops a dense stacked block threads its tensors through.
_VIEW_LIKE = {"aten::view", "aten::reshape", "aten::_unsafe_view",
              "aten::unsqueeze", "aten::squeeze", "aten::transpose",
              "aten::permute", "aten::expand"}
_RESHAPE = {"aten::view", "aten::reshape", "aten::_unsafe_view"}


def _scalar_args(op: Dict[str, Any]) -> List[Any]:
    return [a.get("value") for a in op.get("attributes", {}).get("args", [])
            if a.get("type") in ("scalar", "list")]


def _is_param(tensors: Dict[str, Any], tid: str) -> bool:
    return bool(tensors.get(tid, {}).get("is_parameter")) or tid.startswith("param::")


def _same_shape(tensors: Dict[str, Any], a: str, b: str) -> bool:
    ta, tb = tensors.get(a, {}), tensors.get(b, {})
    if ta.get("shape") is None or ta.get("shape") != tb.get("shape"):
        return False
    sa, sb = ta.get("symbolic_shape"), tb.get("symbolic_shape")
    return sa is None or sb is None or sa == sb


def _fuse_softmax_first_dense_layer(dag, ops, execution_order, tensors,
                                    consumer_map, producer_map, topk_uid,
                                    declared_norm=None):
    """Match and fuse one softmax-before-topk stacked block traced in its dense
    all-experts form (see the section header). Returns like
    `_fuse_one_moe_layer`, or a `Declined` naming the first structural
    mismatch. The fused op's routing is the graph's own — topk of the softmax
    scores, renormalised exactly when the graph divides by the top-k sum.
    `declared_norm` is the registry's norm_topk_prob when the caller declares
    it (a declared pass), None otherwise; a contradiction is raised by name."""
    D = Declined
    topk = ops[topk_uid]
    k = _extract_topk_k(topk)
    if k is None or k <= 1:
        return D("the top-k selects at most one expert")
    gs_tid = _get_input_tensor_id(topk, 0)
    gsh = _tensor_shape(tensors, gs_tid)
    if not gsh or len(gsh) != 2:
        return D(f"the top-k input is not a rank-2 [tokens, experts] tensor ({gsh})")
    E = int(gsh[1])
    tk_args = _scalar_args(topk)
    if len(tk_args) > 1 and tk_args[1] not in (-1, 1):
        return D(f"the top-k does not select over the expert axis (dim={tk_args[1]})")
    sm_op = ops.get(producer_map.get(gs_tid), {})
    if sm_op.get("op_type") != "aten::_softmax":
        return D("the top-k neither feeds a softmax nor reads one: the routing "
                 f"order is not recognised (its input comes from "
                 f"{sm_op.get('op_type')!r})")
    sm_args = _scalar_args(sm_op)
    if not sm_args or sm_args[0] not in (-1, 1):
        return D(f"the softmax before the top-k is not over the expert axis ({sm_args})")
    scores_tid, idx_tid = topk["output_tensor_ids"][0], topk["output_tensor_ids"][1]
    interior = {topk_uid}

    def only_consumer(tid):
        cs = consumer_map.get(tid, [])
        return cs[0] if len(cs) == 1 else None

    # --- routing weights: [sum + div] renormalisation, [cast] ------------
    sc = consumer_map.get(scores_tid, [])
    sc_types = sorted(ops.get(u, {}).get("op_type") for u in sc)
    if sc_types == ["aten::div", "aten::sum"]:
        sum_uid = next(u for u in sc if ops[u]["op_type"] == "aten::sum")
        div_uid = next(u for u in sc if ops[u]["op_type"] == "aten::div")
        sum_out = ops[sum_uid]["output_tensor_ids"][0]
        sargs = _scalar_args(ops[sum_uid])
        dims = sargs[0] if sargs else None
        keep = sargs[1] if len(sargs) > 1 else False
        if dims not in ([-1], [1], -1, 1) or keep is not True:
            return D(f"the top-k scores are summed over {dims} keepdim={keep}, "
                     "not renormalised over the top-k axis")
        if _collect_input_tids(ops[div_uid])[:2] != [scores_tid, sum_out] \
                or consumer_map.get(sum_out, []) != [div_uid]:
            return D("the top-k scores are not divided by their own sum")
        interior |= {sum_uid, div_uid}
        renorm = True
        w_tid = ops[div_uid]["output_tensor_ids"][0]
    elif len(sc) == 1:
        renorm = False
        w_tid = scores_tid
    else:
        return D(f"the top-k scores feed {sc_types}: not a (renormalised) "
                 "routing weight")
    cu = only_consumer(w_tid)
    if cu is not None and ops[cu].get("op_type") == "aten::_to_copy":
        interior.add(cu)
        w_tid = ops[cu]["output_tensor_ids"][0]
        cu = only_consumer(w_tid)

    # --- the routing matrix: scatter(zeros[T,E], 1, indices, weights) ------
    if cu is None or ops[cu].get("op_type") != "aten::scatter":
        return D("the routing weights are not scattered into a [tokens, experts] "
                 f"matrix (they meet {ops.get(cu, {}).get('op_type')!r})")
    sc_uid = cu
    st_ins = [a for a in ops[sc_uid]["attributes"].get("args", [])]
    st_t = [a.get("tensor_id") for a in st_ins if a.get("type") == "tensor"]
    st_s = [a.get("value") for a in st_ins if a.get("type") == "scalar"]
    if len(st_t) != 3 or st_t[1] != idx_tid or st_t[2] != w_tid \
            or not st_s or st_s[0] not in (1, -1):
        return D("the scatter is not (zeros, dim=1, top-k indices, routing weights)")
    if consumer_map.get(idx_tid, []) != [sc_uid]:
        return D("the top-k indices are read beyond the routing scatter "
                 "(a dispatch, not the dense combine)")
    base_uid = producer_map.get(st_t[0])
    if ops.get(base_uid, {}).get("op_type") not in ("aten::zeros_like", "aten::zeros") \
            or consumer_map.get(st_t[0], []) != [sc_uid]:
        return D("the routing scatter does not start from a zeros tensor")
    interior |= {sc_uid, base_uid}
    shape_only_ins = set(_collect_input_tids(ops[base_uid]))

    # --- the routing matrix reaches the weighted combine through views -----
    t = ops[sc_uid]["output_tensor_ids"][0]
    while True:
        cu = only_consumer(t)
        cop = ops.get(cu, {})
        if cop.get("op_type") in _VIEW_LIKE:
            interior.add(cu)
            t = cop["output_tensor_ids"][0]
            continue
        break
    if cop.get("op_type") != "aten::mul":
        return D("the routing matrix meets "
                 f"{cop.get('op_type')!r} before a weighted combine")
    mul_uid = cu
    route_tid = t
    mins = _collect_input_tids(cop)
    if len(mins) != 2 or route_tid not in mins:
        return D("the weighted combine is not a binary aten::mul")
    xo_tid = mins[0] if mins[1] == route_tid else mins[1]
    mul_out = cop["output_tensor_ids"][0]
    msh = _tensor_shape(tensors, mul_out)
    rsh = _tensor_shape(tensors, route_tid)
    if not msh or not rsh or msh[0] != E or rsh[0] != E:
        return D(f"the weighted combine does not carry the expert axis first "
                 f"(product {msh}, routing {rsh}, E={E})")
    red = only_consumer(mul_out)
    rop = ops.get(red, {})
    rargs = _scalar_args(rop)
    rdims = rargs[0] if rargs else None
    rkeep = rargs[1] if len(rargs) > 1 else False
    if rop.get("op_type") != "aten::sum" or rdims not in ([0], 0) or rkeep:
        return D("the weighted expert outputs are not summed over the expert axis "
                 f"({rop.get('op_type')!r} dims={rdims} keepdim={rkeep})")
    exit_tid = rop["output_tensor_ids"][0]
    interior |= {mul_uid, red}
    esh = _tensor_shape(tensors, exit_tid)
    if not esh or list(esh) != list(msh[1:]):
        return D(f"the combine's output {esh} is not the product {msh} without "
                 "its expert axis")
    H = int(esh[-1])

    # --- the expert chain, backwards: view <- bmm(act, W_out) --------------
    t = xo_tid
    while ops.get(producer_map.get(t), {}).get("op_type") in _RESHAPE \
            and len(consumer_map.get(t, [])) == 1 \
            and consumer_map[t][0] in interior:
        pu = producer_map[t]
        interior.add(pu)
        t = _get_input_tensor_id(ops[pu], 0)
    down_uid = producer_map.get(t)
    dop = ops.get(down_uid, {})
    if dop.get("op_type") != "aten::bmm":
        return D(f"the expert outputs do not come from a batched matmul over the "
                 f"experts ({dop.get('op_type')!r})")
    act_tid, w_out = (_collect_input_tids(dop) + [None, None])[:2]
    wsh_out = _tensor_shape(tensors, w_out)
    if not _is_param(tensors, w_out) or not wsh_out or len(wsh_out) != 3 \
            or wsh_out[0] != E or wsh_out[2] != H:
        return D(f"the down projection does not read a stacked [E={E}, F, H={H}] "
                 f"parameter ({w_out}: {wsh_out})")
    F = int(wsh_out[1])
    interior.add(down_uid)
    act_pu = producer_map.get(act_tid)
    if ops.get(act_pu, {}).get("op_type") != "aten::mul":
        return D("the down projection's input is not silu(gate) * up")
    silu_in = [ti for ti in _collect_input_tids(ops[act_pu])
               if ops.get(producer_map.get(ti), {}).get("op_type") == "aten::silu"]
    if len(silu_in) != 1:
        return D("the down projection's input is not silu(gate) * up")
    silu_uid = producer_map[silu_in[0]]
    split_uid = producer_map.get(_get_input_tensor_id(ops[silu_uid], 0))
    halves = _split_halves(ops, consumer_map, split_uid, tensors, F)
    if isinstance(halves, Declined):
        return halves
    gate_offset, h_silu, h_mul = halves
    if h_silu != silu_uid or h_mul != act_pu:
        return D("the SwiGLU read off the split is not the down projection's input")
    interior |= {act_pu, silu_uid, split_uid}

    # --- the gate|up projection: bmm(repeat(hidden), W_in[E, H, 2F]) -------
    up_uid = producer_map.get(_get_input_tensor_id(ops[split_uid], 0))
    uop = ops.get(up_uid, {})
    if uop.get("op_type") != "aten::bmm":
        return D(f"the gate|up projection is not a batched matmul over the experts "
                 f"({uop.get('op_type')!r})")
    hrep_tid, w_in = (_collect_input_tids(uop) + [None, None])[:2]
    wsh_in = _tensor_shape(tensors, w_in)
    if not _is_param(tensors, w_in) or not wsh_in or len(wsh_in) != 3 \
            or wsh_in[0] != E or wsh_in[1] != H or wsh_in[2] != 2 * F:
        return D(f"the gate|up projection does not read a stacked "
                 f"[E={E}, H={H}, 2F={2 * F}] parameter ({w_in}: {wsh_in})")
    if not tensors.get(w_in, {}).get("dtype"):
        return D(f"the input slab {w_in} carries no dtype in the graph")
    # The dispatchers weight each expert's output by its routing score cast to the
    # weight dtype: faithful only when the graph's combine runs in that dtype too.
    combine_dt = {tensors.get(x, {}).get("dtype") for x in (route_tid, xo_tid)}
    if combine_dt != {tensors[w_in].get("dtype")} or \
            tensors.get(w_out, {}).get("dtype") != tensors[w_in].get("dtype"):
        return D(f"the weighted combine runs in {sorted(map(str, combine_dt))}, not the "
                 f"slabs' dtype {tensors[w_in].get('dtype')}")
    interior.add(up_uid)
    t = hrep_tid
    while ops.get(producer_map.get(t), {}).get("op_type") in _RESHAPE \
            and len(consumer_map.get(t, [])) == 1:
        pu = producer_map[t]
        interior.add(pu)
        t = _get_input_tensor_id(ops[pu], 0)
    rep_uid = producer_map.get(t)
    rp = ops.get(rep_uid, {})
    reps = (_scalar_args(rp) or [None])[0]
    if rp.get("op_type") != "aten::repeat" or not isinstance(reps, list) \
            or not reps or reps[0] != E or any(r != 1 for r in reps[1:]) \
            or len(consumer_map.get(t, [])) != 1:
        return D("the experts do not read the hidden states repeated once per "
                 f"expert ({rp.get('op_type')!r} {reps})")
    interior.add(rep_uid)

    # The hidden states: back from the repeat through views that feed only this
    # chain, to the first tensor shaped like the block's output — the fused op
    # returns its result in its input's shape, which the residual reads.
    t = _get_input_tensor_id(rp, 0)
    hidden_tid = None
    while True:
        if _same_shape(tensors, t, exit_tid):
            hidden_tid = t
            break
        pu = producer_map.get(t)
        if ops.get(pu, {}).get("op_type") not in _RESHAPE \
                or len(consumer_map.get(t, [])) != 1:
            break
        interior.add(pu)
        t = _get_input_tensor_id(ops[pu], 0)
    if hidden_tid is None:
        return D(f"no tensor on the hidden-state chain has the block output's "
                 f"shape {esh}")

    # --- closure: nothing escapes, nothing unexpected enters ---------------
    allowed_in = {hidden_tid, gs_tid, w_in, w_out}
    for uid in interior:
        for tid in _collect_input_tids(ops[uid]):
            if producer_map.get(tid) in interior or tid in allowed_in:
                continue
            if tid in shape_only_ins and uid == base_uid:
                continue        # zeros_like reads only the shape
            return D(f"{tid} enters the block at {uid} from outside it")
        for ot in ops[uid].get("output_tensor_ids", []):
            if ot == exit_tid:
                continue
            esc = [c for c in consumer_map.get(ot, []) if c not in interior]
            if esc:
                return D(f"{ot} escapes the block to {esc[0]}")

    # A registry that DECLARES the renormalisation and a trace that computes the
    # other one is a contradiction in the data, not "another block": refused
    # loudly, never overruled in silence, never left to run unfused unnoticed.
    if declared_norm is not None and bool(declared_norm) != renorm:
        raise RuntimeError(
            f"ZERO FALLBACK: [MoE Fusion] {topk_uid}: the registry declares "
            f"norm_topk_prob={bool(declared_norm)} but the traced graph computes "
            f"norm_topk_prob={renorm} (it {'divides' if renorm else 'does not divide'} "
            "the top-k scores by their sum) — fix the data at its source.")

    # --- the fused op -------------------------------------------------------
    parent = topk.get("parent_module", "") or ""
    bm = re.match(rf"({re.escape(_NeuroTax.resolve('layers'))}\.\d+)", parent)
    fused_uid = f"moe_fused::{bm.group(1) if bm else topk_uid}"
    if fused_uid in ops:
        fused_uid = f"moe_fused::{topk_uid}"
    fused_op = _stacked_fused_op(
        tensors, fused_uid, top_k=k, num_experts=E, F=F, H=H,
        hidden_tid=hidden_tid, gate_scores_tid=gs_tid, in_tid=w_in,
        out_tid=w_out, exit_tid=exit_tid,
        # bmm(x[E,T,H], W_in[E,H,2F]): per expert (in x out); the same for W_out
        in_axis=0, gate_offset=gate_offset, out_in_axis=0,
        norm_topk_prob=renorm,
        # The renormalisation is what the GRAPH computes (sum + div present or
        # not) — the traced vendor code, not the registry flag, decides it.
        extra_attrs={"routing_from_graph": "softmax_before_topk"})

    # --- placement: inside the dependence window ---------------------------
    new_order = [u for u in execution_order if u not in interior]
    pos = {u: i for i, u in enumerate(new_order)}
    latest = max((pos[p] for p in (producer_map.get(hidden_tid),
                                   producer_map.get(gs_tid)) if p in pos),
                 default=-1)
    earliest = min((pos[c] for c in consumer_map.get(exit_tid, []) if c in pos),
                   default=len(new_order))
    if latest + 1 > earliest:
        return D("an input of the block is produced after its output is read")
    new_order.insert(latest + 1, fused_uid)
    ops[fused_uid] = fused_op
    for u in interior:
        ops.pop(u, None)
    execution_order[:] = new_order
    return (len(interior), fused_uid, fused_op, 0)
