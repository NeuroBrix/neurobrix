"""fusion_horizontal — shared planner for sibling-matmul batching.

Engine-neutral half: pure-dict analysis over the shared in-memory graph
at the D1 hook (`GraphExecutor._init_from_dag`, after MoE fusion). It
identifies groups of `aten::mm` ops that read the SAME source tensor
with DIFFERENT constant weights — the q/k/v (and gate/up) projection
class — and annotates the dag. It never computes a value and never
imports torch.

The lowering each engine performs: concatenate the group's weights
along the output dimension ONCE at bind time (they are load-time
constants), run ONE wider mm in the hot loop, and hand each original
consumer its column band of the fused output. N launches become 1.

WHY IT PAYS HERE. Measured shape of the zoo (2026-08-19): **30,880
sibling groups, 30,860 of size 2 and 20 of size 4, with 100% of the
weights produced by `aten::t`** (i.e. transposes of parameters). A
typical decode-time site is `[2,2048] x [2048,1408]` twice — skinny
GEMMs where the launch dominates the arithmetic, which is exactly the
residual the frozen-plan replay and per-bucket graph capture left
behind. On Volta, where the mm gap versus cuBLAS is structural, giving
the kernel a wider N is also its own small win.

=== THE CONTRACT ===

1. SAME SOURCE, DIFFERENT WEIGHTS. Members must share input[0] by
   tensor id. Sharing "the same values through different ids" is not
   enough — that is CSE's job, and it runs first.

2. WEIGHTS MUST BE LOAD-TIME CONSTANTS. Every member's weight must be
   the output of an `aten::t` whose own input is a parameter. This is
   what makes the concatenation a bind-time act rather than a per-step
   one; a runtime-computed weight would turn the fusion into extra
   work every step. (Measured: 100% of sites match this shape, so the
   restriction costs nothing today and keeps the pass honest if a
   different shape appears.)

3. IDENTICAL CONTRACTION EXTENT. All members must agree on K (the
   shared dimension) and on output dtype/device. A group that does not
   is refused rather than reshaped.

4. NO MEMBER MAY PRODUCE A GRAPH OUTPUT. The positional output
   contract is frozen; a fused output band is not the same tensor
   object, and this pass does not touch that surface.

5. EXECUTION-ORDER ADJACENCY IS NOT REQUIRED, BUT MUTATION IS A
   BARRIER. Members are collected in execution order; if a mutating op
   (`aten::copy`, `aten::uniform` — the set audited for CSE against the
   engines' dispatch tables) executes between two candidates, the
   group is cut there. The source could have been rewritten in place
   between them, and views alias their base.

=== BIT-EXACTNESS IS NOT ASSUMED ===

Mathematically the fused GEMM computes each output element with the
same dot product over the same K. But the ACCUMULATION ORDER over K is
chosen by the kernel, and a wider N can make cuBLAS or the Triton
autotuner pick a different tiling or split-K. So this pass is declared
EXACT in policy.py yet must EARN that byte gate on the full zoo like
every other rung; if the gate says otherwise, the honest outcome is to
reclassify it as a drift-policy pass with its evidence, not to widen
the tolerance. The planner does not decide this — the measurement does.
"""

from __future__ import annotations

from collections import defaultdict

PLAN_KEY = "_optim_fusion_horizontal"
PASS_NAME = "fusion_horizontal"

# Same audited set as cse.py: ops the engines execute as a write into
# an input's storage. Cuts a group in execution order.
_MUTATING = {"aten::copy", "aten::uniform"}

_MIN_GROUP = 2


def plan_fusion_horizontal(dag: dict) -> dict | None:
    """Return the sibling-matmul grouping plan, or None when there is none.

    Plan shape:
      groups: [{"source": tid,
                "members": [{"uid", "weight_tid", "out_tid", "cols"}],
                "k": int, "total_cols": int}]
      n_ops_saved: launches removed (sum over groups of len(members)-1)
    The dag is NOT rewritten here; each engine lowers at its own
    compile/bind step.
    """
    ops: dict = dag.get("ops", {})
    tensors: dict = dag.get("tensors", {})
    order = [u for u in dag.get("execution_order", []) if u in ops]
    graph_outputs = set(dag.get("output_tensor_ids") or [])

    producer: dict[str, str] = {}
    for uid in order:
        for tid in ops[uid].get("output_tensor_ids") or []:
            producer[tid] = uid

    def weight_is_load_time_constant(tid: str) -> bool:
        """True when `tid` is t(param) — a transpose of a parameter."""
        puid = producer.get(tid)
        if puid is None:
            # No producer: it is a leaf. A parameter leaf qualifies.
            return bool((tensors.get(tid) or {}).get("is_parameter"))
        pop = ops.get(puid) or {}
        if pop.get("op_type") != "aten::t":
            return False
        ins = pop.get("input_tensor_ids") or []
        if len(ins) != 1:
            return False
        base = tensors.get(ins[0]) or {}
        return bool(base.get("is_parameter"))

    def cols_of(tid: str) -> int | None:
        shape = (tensors.get(tid) or {}).get("shape") or []
        return shape[-1] if len(shape) == 2 and isinstance(shape[-1], int) else None

    def rows_of(tid: str) -> int | None:
        shape = (tensors.get(tid) or {}).get("shape") or []
        return shape[0] if len(shape) == 2 and isinstance(shape[0], int) else None

    # Walk in execution order, cutting every candidate set at a mutation.
    # `open_groups` maps source tid -> list of member records collected
    # since the last barrier.
    open_groups: dict[str, list[dict]] = defaultdict(list)
    closed: list[tuple[str, list[dict]]] = []

    def flush() -> None:
        for src, members in open_groups.items():
            if len(members) >= _MIN_GROUP:
                closed.append((src, members))
        open_groups.clear()

    for uid in order:
        op = ops[uid]
        op_type = str(op.get("op_type", ""))

        if op_type in _MUTATING:
            flush()
            continue

        if op_type != "aten::mm":
            continue

        ins = list(op.get("input_tensor_ids") or [])
        outs = list(op.get("output_tensor_ids") or [])
        if len(ins) != 2 or len(outs) != 1:
            continue
        src, w = ins
        out = outs[0]
        if out in graph_outputs:
            continue
        if not weight_is_load_time_constant(w):
            continue
        cols, k = cols_of(w), rows_of(w)
        if cols is None or k is None:
            continue
        t_out = tensors.get(out) or {}
        open_groups[src].append({
            "uid": uid,
            "weight_tid": w,
            "out_tid": out,
            "cols": cols,
            "k": k,
            "dtype": t_out.get("dtype"),
            "device": t_out.get("device"),
        })

    flush()

    groups = []
    for src, members in closed:
        ks = {m["k"] for m in members}
        dts = {(m["dtype"], m["device"]) for m in members}
        if len(ks) != 1 or len(dts) != 1:
            # Refuse rather than reshape (contract clause 3).
            continue
        groups.append({
            "source": src,
            "members": [
                {k2: m[k2] for k2 in ("uid", "weight_tid", "out_tid", "cols")}
                for m in members
            ],
            "k": next(iter(ks)),
            "total_cols": sum(m["cols"] for m in members),
        })

    if not groups:
        return None

    return {
        "pass": PASS_NAME,
        "groups": groups,
        "n_groups": len(groups),
        "n_ops_saved": sum(len(g["members"]) - 1 for g in groups),
    }
