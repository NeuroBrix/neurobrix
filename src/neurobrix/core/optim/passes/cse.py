"""cse — shared planner for common-subexpression elimination (Phase 3).

EXACT-gate pass (policy.py). Engine-neutral half: pure-dict analysis
over the shared in-memory graph at the D1 hook
(`GraphExecutor._init_from_dag`, after MoE fusion). It computes a
redundancy plan — "op B recomputes what op A already produced, read A's
output instead" — and annotates the dag. It never computes a value and
never imports torch (R33-compatible). Each engine lowers the plan by
skipping the duplicate ops and aliasing their output slots.

WHY THIS IS WORTH LANDING NOW. It was deferred for a measured reason:
the per-op dispatch tax dominated, so deleting an op removed a
computation already drowned in its own launch toll. Frozen-plan replay
plus per-bucket CUDA-graph capture crushed that floor, so a removed op
is now removed WORK *and* one fewer node in the captured graph — the
two levers multiply instead of masking each other.

=== THE EQUIVALENCE CONTRACT ===

Identical tensor ids are NOT identical values in a functionalised DAG,
for the same reason tid-reachability is not liveness (R19, 2026-08-19).
Two ops with the same op_type and the same input tids are
interchangeable ONLY if nothing between them could have changed what
those tids hold. Three exclusions, each with its evidence:

1. MUTATION BARRIERS. `aten::copy` is executed as `x.copy_(src)` by all
   three dispatch paths (compiled_ops, sequential_dispatcher,
   kernels/dispatch `_meta_copy`), and `aten::uniform`'s wrapper writes
   into its input. Such an op can rewrite the storage behind a tid
   whose id never changes — and because views (slice/select/…) alias
   their base, deciding *which* tids it reached is an aliasing analysis
   this planner deliberately does not attempt. The sound, cheap rule:
   every mutating op is a FULL BARRIER — all available expressions are
   killed at that point in execution order. Cost is negligible (a few
   hundred such ops zoo-wide); soundness is total.
   Note `_MUTATING_SUFFIX` alone is NOT a sufficient test here: the
   tracer strips the trailing underscore, so `aten::copy_` is recorded
   as `aten::copy` and a suffix check matches nothing. The check is a
   named set, audited against the engine's own dispatch tables.

2. RANDOM DRAWS ARE NEVER VALUE-NUMBERED. Two identical `randn` ops
   draw different values by design. Worse, under `NBX_FORCE_RAND_SEED`
   every draw comes from ONE sequential stream (kernels/rng_pin.py), so
   merging two draws changes the draw COUNT and shifts every later
   draw — breaking the seeded battery baselines.

3. EQUALITY IS THE WHOLE RECORD. The key hashes op_type, the ORDERED
   input tids, the attributes (args AND kwargs — scalar and shape
   arguments live there), and the output dtype/device. It excludes the
   op's own uid and output tids. Deterministic by R28. Ops are merged
   only within one device.

Further exclusions, matching const_fold's evidence:
  - non-`aten::` ops (custom:: fusions consume tensors through
    attribute contracts this planner does not model);
  - any op whose output is a graph output (the positional contract is
    frozen; aliasing it away is not this pass's business);
  - ops with no tensor inputs — a nullary op (`arange`, `full`) is
    cheap and its inputs carry no ordering evidence, so merging buys
    nothing and complicates the barrier argument.
"""

from __future__ import annotations

import json

PLAN_KEY = "_optim_cse"
PASS_NAME = "cse"

# Ops the ENGINES execute as a write into an input's storage. Audited
# 2026-08-19 against all three dispatch paths; see the module docstring.
# A mutating op is a full barrier, never a merge candidate.
_MUTATING = {
    "aten::copy",
    "aten::uniform",
}

# Ops whose output is not a pure function of their inputs. Never merged
# AND never barriers — they simply cannot be value-numbered.
_NONDETERMINISTIC = {
    "aten::rand", "aten::randn", "aten::rand_like", "aten::randn_like",
    "aten::randint", "aten::randint_like", "aten::multinomial",
    "aten::bernoulli", "aten::normal", "aten::uniform",
    "aten::exponential", "aten::randperm",
}

# VIEW CLASS — excluded, and the exclusion is the whole point of this
# pass's measured yield.
#
# A view op computes nothing: it returns metadata over its base's
# storage. Merging two of them removes a launch but zero work, which is
# the "two levers" argument in reverse — and it is not even free of
# risk, because both engines run dedicated view passes
# (_eliminate_detach_ops, _eliminate_weight_transpose_ops, the
# pretransposed contract, the seq-dependent-constant narrowing that
# rewrites a weight slot in place) that assume each view op is its own
# node over its own base. const_fold already excludes this class for
# the sibling reason (folding trades a free view for a materialized
# copy).
#
# MEASURED, 2026-08-19: with views eligible, the full-zoo byte gate
# went RED on sana1024_triton and vibevoice_triton — deterministic
# (pass1 == pass2) but different from the OFF baselines, i.e. a real
# numeric change, triton-only. The op-by-op fingerprint put the first
# divergence at aten.cat::3 in the RoPE neighbourhood, and the plan
# dump showed the merges were 100% view class (view 59, slice 21,
# unsqueeze 19) with zero compute ops among them. Excluding the class
# removes the hazard AND costs no real work.
_VIEW_CLASS = {
    "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::slice", "aten::select", "aten::unsqueeze", "aten::squeeze",
    "aten::expand", "aten::expand_as", "aten::permute",
    "aten::transpose", "aten::t", "aten::detach", "aten::alias",
    "aten::narrow", "aten::unfold", "aten::flatten", "aten::unflatten",
    "aten::split", "aten::chunk", "aten::movedim", "aten::swapaxes",
}


def rename_tensor_refs(op: dict, rename: dict) -> bool:
    """Re-point every tensor reference in `op` through `rename`.

    Shared by both engines' lowering so they cannot drift. Tensor ids
    live in `input_tensor_ids` AND inside `attributes` — capture.py
    emits {"type": "tensor", "tensor_id": …} and
    {"type": "tensor_tuple", "tensor_ids": [...]} in both `args` and
    `kwargs`. A lowering that renames only the id list leaves an
    attribute pointing at a dropped output, whose slot is never
    written: the op then reads None at run time (measured on
    aten.mul::278, Qwen3-Coder int4 decode, 2026-08-19). Same lesson as
    R19's collect_input_tids.

    Returns True when anything was rewritten.
    """
    touched = False
    ins = op.get("input_tensor_ids")
    if ins and any(t in rename for t in ins):
        op["input_tensor_ids"] = [rename.get(t, t) for t in ins]
        touched = True

    def walk(node):
        nonlocal touched
        if isinstance(node, dict):
            tid = node.get("tensor_id")
            if isinstance(tid, str) and tid in rename:
                node["tensor_id"] = rename[tid]
                touched = True
            tids = node.get("tensor_ids")
            if isinstance(tids, list) and any(
                    isinstance(t, str) and t in rename for t in tids):
                node["tensor_ids"] = [
                    rename.get(t, t) if isinstance(t, str) else t
                    for t in tids]
                touched = True
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(op.get("attributes", {}))
    return touched


def _canon(node) -> object:
    """Canonical, order-stable form of an attribute subtree."""
    if isinstance(node, dict):
        return {k: _canon(node[k]) for k in sorted(node)}
    if isinstance(node, (list, tuple)):
        return [_canon(v) for v in node]
    return node


def plan_cse(dag: dict) -> dict | None:
    """Compute the redundancy plan of `dag`, or None when nothing merges.

    Returns:
      merges       list of {"keep": uid, "drop": uid, "alias": [[from, to], ...]}
                   in execution order; `drop` recomputes what `keep`
                   already produced, and each of drop's output tids
                   aliases the positionally-matching output of keep.
      n_ops        number of ops the plan removes (== len(merges))
      barriers     count of mutation barriers that reset the table
                   (recorded so the census explains its own yield)
    The dag's ops and execution_order are NOT rewritten here; each
    engine skips the dropped ops and binds the alias map at compile.
    """
    ops: dict = dag.get("ops", {})
    tensors: dict = dag.get("tensors", {})
    order = [u for u in dag.get("execution_order", []) if u in ops]
    graph_outputs = set(dag.get("output_tensor_ids") or [])

    # A tensor produced by a non-aten op, or touched by one through an
    # attribute contract, is outside the model this planner reasons in.
    opaque: set[str] = set()
    for op in ops.values():
        if not str(op.get("op_type", "")).startswith("aten::"):
            opaque.update(op.get("input_tensor_ids") or [])
            opaque.update(op.get("output_tensor_ids") or [])

    available: dict[str, str] = {}     # equivalence key -> keeping uid
    merges: list[dict] = []
    n_barriers = 0

    for uid in order:
        op = ops[uid]
        op_type = str(op.get("op_type", ""))

        if op_type in _MUTATING:
            # FULL BARRIER: this op may have rewritten storage behind
            # tids whose ids did not change. Everything computed before
            # it is no longer known to be reproducible.
            available.clear()
            n_barriers += 1
            continue

        if (
            op_type in _NONDETERMINISTIC
            or op_type in _VIEW_CLASS
            or not op_type.startswith("aten::")
        ):
            continue

        ins = list(op.get("input_tensor_ids") or [])
        outs = list(op.get("output_tensor_ids") or [])
        if not ins or not outs:
            continue
        if any(o in graph_outputs for o in outs):
            continue
        if any(t in opaque for t in ins) or any(o in opaque for o in outs):
            continue

        out_meta = []
        for o in outs:
            t = tensors.get(o) or {}
            out_meta.append((t.get("dtype"), t.get("device")))

        key = json.dumps(
            [op_type, ins, _canon(op.get("attributes", {})), out_meta],
            sort_keys=True, separators=(",", ":"),
        )

        keep = available.get(key)
        if keep is None:
            available[key] = uid
            continue

        keep_outs = list(ops[keep].get("output_tensor_ids") or [])
        if len(keep_outs) != len(outs):
            # Same key, different arity: refuse rather than guess.
            continue
        merges.append({
            "keep": keep,
            "drop": uid,
            "alias": [[o, k] for o, k in zip(outs, keep_outs)],
        })

    if not merges:
        return None

    return {
        "pass": PASS_NAME,
        "merges": merges,
        "n_ops": len(merges),
        "barriers": n_barriers,
    }
