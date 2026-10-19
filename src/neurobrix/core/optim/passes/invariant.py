"""invariant — the ops a repeated component computes from its invariant inputs alone (state engine step 4).

docs/design/state-engine.md §4.1: a loop component's ops whose inputs are all loop-invariant (the same at every run of a
request) are invariant; the runtime evaluates them once and keeps their outputs as state. The build declares WHICH inputs
are invariant (topology.json `state`, kind `invariant`: the outputs of a component computed before this one — the text
condition a denoiser reads at every step, the encoder states a decoder reads at every token). This module derives the
ops from that declaration, on the op list an engine actually runs — after its own fusions — at the slot level, so both
engines plan with the same function. Pure Python, torch-free (R33).

An op is invariant when:
  - it is an `aten::` op that is neither random nor in place,
  - every slot it reads is a weight or an invariant slot, and at least one is invariant (an op of weights alone is
    const_fold's or the pre-transpose passes' business, not this state's),
  - no slot it reads or writes is touched by an in-place op anywhere in the sequence (the value could change mid-run),
  - every symbol its attributes or output shapes carry is bound by an invariant input (a symbol bound by a varying
    input — the latents' token count — makes its value vary),
  - the caller does not exclude it (an intercepted op keeps per-run state; a per-run constant, the rotary tables sliced
    to the run's length, is not a weight here).
Its outputs become invariant. The frontier is the invariant slots a non-invariant op reads: the state kept between runs.

What this test would do if the code were wrong: a varying op marked invariant gives a stale value at the second run —
the unit tests change the varying input between runs and require the output to follow it.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

PASS_NAME = "invariant"

_NONDETERMINISTIC = {
    "aten::rand", "aten::randn", "aten::rand_like", "aten::randn_like",
    "aten::randint", "aten::multinomial", "aten::bernoulli",
    "aten::normal", "aten::uniform", "aten::exponential", "aten::dropout",
    "aten::native_dropout",
}


class InvariantPlanRefused(RuntimeError):
    pass


def symbol_ids(node, acc: Optional[Set[str]] = None) -> Set[str]:
    """Every symbol id a serialized dim, expression or attribute tree references."""
    acc = set() if acc is None else acc
    if isinstance(node, dict):
        if node.get("type") == "symbol":
            sid = node.get("id", node.get("symbol_id"))
            if sid:
                acc.add(str(sid))
        for v in node.values():
            if isinstance(v, (dict, list)):
                symbol_ids(v, acc)
        for f in node.get("factors", []) or []:
            if isinstance(f, str):
                acc.add(f)
    elif isinstance(node, list):
        for v in node:
            symbol_ids(v, acc)
    return acc


def invariant_input_names(topology, component: str) -> List[str]:
    """The inputs the container declares invariant for `component` (topology.json `state`, kind `invariant`)."""
    out: List[str] = []
    for e in ((topology or {}).get("state") or {}).get(component) or []:
        if e.get("kind") == "invariant":
            out.extend(e.get("inputs") or [])
    return out


def plan_invariant(
    ops: Sequence,                       # the engine's compiled ops, in run order: .op_uid .op_type .all_input_slots .output_slots
    weight_slots: Iterable[int],
    invariant_slots: Iterable[int],
    graph_output_slots: Iterable[int] = (),
    symbols_of: Optional[Callable[[str], Set[str]]] = None,
    invariant_symbols: Iterable[str] = (),
    excluded_uids: Iterable[str] = (),
    excluded_slots: Iterable[int] = (),
) -> Optional[Dict[str, object]]:
    """{op_indices, frontier_slots, input_slots} of the invariant partition of `ops`, or None when nothing is
    invariant. Refused by name when a slot is written by two ops (the partition is a slot-level analysis)."""
    weights = set(weight_slots) - set(excluded_slots)
    seeds = set(invariant_slots)
    outs = set(graph_output_slots)
    excluded = set(excluded_uids)
    inv_syms = set(invariant_symbols)

    writer: Dict[int, int] = {}
    for i, op in enumerate(ops):
        for s in op.output_slots:
            if s in writer:
                raise InvariantPlanRefused(
                    f"slot {s} is written by two ops ({ops[writer[s]].op_uid}, {op.op_uid}) — the invariant "
                    f"partition needs one writer per slot")
            writer[s] = i

    touched: Set[int] = set()
    for op in ops:
        if str(op.op_type).endswith("_"):
            touched.update(op.all_input_slots)
            touched.update(op.output_slots)
    seeds -= touched
    if not seeds:
        return None

    inv: Set[int] = set(seeds)
    chosen: List[int] = []
    for i, op in enumerate(ops):
        t = str(op.op_type)
        if (not t.startswith("aten::") or t.endswith("_") or t in _NONDETERMINISTIC
                or op.op_uid in excluded):
            continue
        ins = list(op.all_input_slots)
        if not ins or not any(s in inv for s in ins) or not all(s in inv or s in weights for s in ins):
            continue
        if any(s in outs or s in touched for s in op.output_slots) or not op.output_slots:
            continue
        if symbols_of is not None and not symbols_of(op.op_uid) <= inv_syms:
            continue
        chosen.append(i)
        inv.update(op.output_slots)
    if not chosen:
        return None
    chosen_set = set(chosen)
    produced = {s for i in chosen for s in ops[i].output_slots}
    frontier: Set[int] = set()
    for i, op in enumerate(ops):
        if i in chosen_set:
            continue
        frontier.update(s for s in op.all_input_slots if s in produced)
    if not frontier:
        return None                     # an invariant chain nothing reads is dead code's business
    used_inputs = sorted(s for s in seeds if any(s in ops[i].all_input_slots for i in chosen)
                         or any(s in op.all_input_slots for op in ops))
    return {"pass": PASS_NAME, "op_indices": chosen, "frontier_slots": sorted(frontier),
            "input_slots": used_inputs}


DAG_KEY = "_state_invariant_inputs"     # the declared invariant input tensor ids, stamped on the dag at load


def plan_for_sequence(dag, ops: Sequence, tid_to_slot: Dict[str, int], num_weights: int,
                      excluded_slots: Iterable[int] = (), excluded_uids: Iterable[str] = ()) -> Optional[Dict[str, object]]:
    """The invariant partition of a compiled sequence (either engine): its declared invariant inputs read from the dag
    (`DAG_KEY`), the weights its arena lays out first ([0, num_weights)), the symbols those inputs bind."""
    tids = [t for t in (dag.get(DAG_KEY) or []) if t in tid_to_slot]
    if not tids:
        return None
    tensors = dag.get("tensors") or {}
    dag_ops = dag.get("ops") or {}
    inv_syms: Set[str] = set()
    for t in tids:
        symbol_ids(((tensors.get(t) or {}).get("symbolic_shape") or {}).get("dims"), inv_syms)

    def symbols_of(uid: str) -> Set[str]:
        op = dag_ops.get(uid) or {}
        acc = symbol_ids(op.get("attributes") or {})
        for o in op.get("output_tensor_ids") or []:
            symbol_ids(((tensors.get(o) or {}).get("symbolic_shape") or {}).get("dims"), acc)
        return acc

    outs = [tid_to_slot[t] for t in (dag.get("output_tensor_ids") or []) if t in tid_to_slot]
    return plan_invariant(ops, range(num_weights), [tid_to_slot[t] for t in tids], outs, symbols_of, inv_syms,
                          excluded_uids, excluded_slots)


def stamp_dag(dag, topology, component: str) -> List[str]:
    """dag[DAG_KEY] = the input tensor ids the container declares invariant for `component` and its graph reads — the
    one reading of the declaration, for the engines (GraphExecutor at load) and for Prism (the container reader)."""
    tensors = dag.get("tensors") or {}
    tids = [f"input::{n}" for n in invariant_input_names(topology, component) if f"input::{n}" in tensors]
    if tids:
        dag[DAG_KEY] = tids
    return tids


class _DagOp:
    __slots__ = ("op_uid", "op_type", "all_input_slots", "output_slots")

    def __init__(self, uid, op):
        self.op_uid, self.op_type = uid, op.get("op_type", "")
        self.all_input_slots = tuple(op.get("input_tensor_ids") or ())
        self.output_slots = tuple(op.get("output_tensor_ids") or ())


def plan_for_dag(dag) -> Optional[Dict[str, object]]:
    """The invariant partition of a graph as the build wrote it (tensor ids stand for slots) — what Prism prices: the
    frontier the state holds through a whole run, and the inputs it keeps a copy of. The engines plan the same rule on
    their own compiled op lists."""
    tids = list(dag.get(DAG_KEY) or [])
    if not tids:
        return None
    tensors = dag.get("tensors") or {}
    ops = dag.get("ops") or {}
    order = [_DagOp(u, ops[u]) for u in dag.get("execution_order") or [] if u in ops]
    weights = [t for t, m in tensors.items() if m.get("is_parameter") or str(t).startswith(("param::", "buffer::"))]
    inv_syms: Set[str] = set()
    for t in tids:
        symbol_ids(((tensors.get(t) or {}).get("symbolic_shape") or {}).get("dims"), inv_syms)

    def symbols_of(uid: str) -> Set[str]:
        op = ops.get(uid) or {}
        acc = symbol_ids(op.get("attributes") or {})
        for o in op.get("output_tensor_ids") or []:
            symbol_ids(((tensors.get(o) or {}).get("symbolic_shape") or {}).get("dims"), acc)
        return acc

    try:
        return plan_invariant(order, weights, tids, dag.get("output_tensor_ids") or (), symbols_of, inv_syms)
    except InvariantPlanRefused:
        return None
