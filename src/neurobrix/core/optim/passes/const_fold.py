"""const_fold — shared planner for load-time constant folding (Phase 2).

EXACT-gate pass (policy.py). This module is the ENGINE-NEUTRAL half of
the pass (scoping doc D2): pure-dict analysis over the shared in-memory
graph at the D1 hook (`GraphExecutor._init_from_dag`, after MoE
fusion). It computes the foldable partition and annotates the dag; it
never computes a value and never imports torch (R33-compatible).

The PER-ENGINE half executes the partition ONCE through the mode's own
op machinery at first bind-with-weights and stores the frontier values
as constants — byte-identical to the per-forward computation by
construction, because the same op callables run in the same order on
the same inputs (see the Phase 2 design section of
docs/internal/optimization_engine_scoping.md).

Deliberate exclusions (evidence-driven, recorded in the design doc):
  - transpose/permute/detach class: owned by the existing per-engine
    view-based passes (`pretransposed` contract) — folding would
    materialize weight-sized copies where a view is free.
  - symbolic ops: an op whose attributes or output shape carry a
    symbol depends on a runtime extent; folding it would freeze the
    trace value (the slice-attrs doctrine).
  - custom:: consumers: fused ops (e.g. custom::moe_fused) consume
    tensors through attribute contracts the planner does not model —
    any tensor they touch stays live and unfolded.
"""

from __future__ import annotations

# Mirrors analyzer.py — ops whose output is not a pure function of
# their inputs. Never foldable.
_NONDETERMINISTIC = {
    "aten::rand", "aten::randn", "aten::rand_like", "aten::randn_like",
    "aten::randint", "aten::multinomial", "aten::bernoulli",
    "aten::normal", "aten::uniform", "aten::exponential",
}

# Owned by the existing per-engine view-based passes; folding these
# would trade a free view for a materialized copy.
_VIEW_OWNED = {"aten::t", "aten::transpose", "aten::permute",
               "aten::detach"}

_MUTATING_SUFFIX = "_"

PLAN_KEY = "_optim_const_fold"
PASS_NAME = "const_fold"


def _has_symbol(node) -> bool:
    if isinstance(node, dict):
        if node.get("type") == "symbol":
            return True
        return any(_has_symbol(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_symbol(v) for v in node)
    return False


def plan_const_fold(dag: dict) -> dict | None:
    """Compute the foldable partition of `dag` and return the plan.

    Returns None when nothing is foldable. Otherwise a dict:
      op_uids            ordered op uids to execute once at bind time
      frontier_tids      output tensor ids that become load constants
      releasable_tids    parameter tensor ids consumed ONLY by the
                         partition (memory credit once folded)
    The caller stores the plan under dag[PLAN_KEY]; the dag's ops and
    execution_order are NOT rewritten here — each engine excludes the
    partition from its hot sequence at compile (per-engine lowering).
    """
    ops: dict = dag.get("ops", {})
    tensors: dict = dag.get("tensors", {})
    order = [u for u in dag.get("execution_order", []) if u in ops]
    graph_outputs = set(dag.get("output_tensor_ids") or [])

    def tensor_inputs(op: dict) -> list[str]:
        return list(op.get("input_tensor_ids") or [])

    def consumers(tid: str) -> list[str]:
        t = tensors.get(tid)
        cons = list(t.get("consumer_op_uids") or []) if t else []
        return [c for c in cons if c in ops]

    def out_symbolic(op: dict) -> bool:
        for tid in op.get("output_tensor_ids") or []:
            ss = (tensors.get(tid) or {}).get("symbolic_shape")
            if isinstance(ss, dict) and _has_symbol(ss.get("dims")):
                return True
        return False

    # Tensors a custom:: op touches (inputs AND attribute-referenced
    # ids) are consumed through contracts we do not model — pin them.
    custom_pinned: set[str] = set()

    def collect_ids(node) -> None:
        if isinstance(node, dict):
            tid = node.get("tensor_id")
            if isinstance(tid, str):
                custom_pinned.add(tid)
            for v in node.values():
                collect_ids(v)
        elif isinstance(node, list):
            for v in node:
                collect_ids(v)
            if all(isinstance(x, str) for x in node):
                # id-list attributes (e.g. expert_*_weight_ids)
                custom_pinned.update(x for x in node if "::" in x)

    for op in ops.values():
        if not str(op.get("op_type", "")).startswith("aten::"):
            custom_pinned.update(tensor_inputs(op))
            collect_ids(op.get("attributes", {}))
        elif str(op.get("op_type", "")).endswith(_MUTATING_SUFFIX):
            # In-place aliasing guard (gardien M1): a hot-path mutator
            # writing into a const-closure tensor would corrupt the
            # folded constant on forward N>=2 — pre-fold, the producer
            # re-ran every forward and regenerated a fresh value. Pin
            # every tensor a mutating op touches out of the closure.
            custom_pinned.update(tensor_inputs(op))
            custom_pinned.update(op.get("output_tensor_ids") or [])

    # ---- const closure (concrete only) -------------------------------
    # Pinned tensors are excluded from the SEEDS too: a param a mutator
    # writes into is not a constant — a const op reading it re-read the
    # mutated value at its position in the trace order every forward.
    const_tids: set[str] = {
        tid for tid, t in tensors.items()
        if t.get("is_parameter") and tid not in custom_pinned
    }
    const_ops: set[str] = set()
    for uid in order:
        op = ops[uid]
        op_type = op.get("op_type", "")
        if (
            op_type in _NONDETERMINISTIC
            or op_type in _VIEW_OWNED
            or not op_type.startswith("aten::")
            or op_type.endswith(_MUTATING_SUFFIX)
            or _has_symbol(op.get("attributes", {}))
            or out_symbolic(op)
        ):
            continue
        outs = op.get("output_tensor_ids") or []
        tin = tensor_inputs(op)
        if not tin or not all(t in const_tids for t in tin):
            continue
        if any(o in graph_outputs or o in custom_pinned for o in outs):
            continue
        const_ops.add(uid)
        const_tids.update(outs)

    if not const_ops:
        return None

    # ---- frontier + ancestors-of-frontier ----------------------------
    frontier_tids: list[str] = []
    frontier_ops: set[str] = set()
    for uid in const_ops:
        for o in ops[uid].get("output_tensor_ids") or []:
            if any(c not in const_ops for c in consumers(o)):
                frontier_ops.add(uid)
                frontier_tids.append(o)

    if not frontier_ops:
        return None  # fully dead const chains belong to dead_code

    # Keep only the partition that feeds a frontier (reverse walk);
    # dead const interior is dead_code's business, not ours.
    needed: set[str] = set(frontier_ops)
    producer = {
        o: uid for uid in const_ops
        for o in ops[uid].get("output_tensor_ids") or []
    }
    stack = list(frontier_ops)
    while stack:
        uid = stack.pop()
        for tid in tensor_inputs(ops[uid]):
            p = producer.get(tid)
            if p and p not in needed:
                needed.add(p)
                stack.append(p)

    plan_ops = [u for u in order if u in needed]

    # ---- releasable params (all consumers inside the partition) ------
    releasable: list[str] = []
    for tid, t in tensors.items():
        if not t.get("is_parameter"):
            continue
        cons = consumers(tid)
        if cons and all(c in needed for c in cons) \
                and tid not in custom_pinned:
            releasable.append(tid)

    return {
        "pass": PASS_NAME,
        "op_uids": plan_ops,
        "frontier_tids": sorted(set(frontier_tids)),
        "releasable_tids": sorted(releasable),
    }
