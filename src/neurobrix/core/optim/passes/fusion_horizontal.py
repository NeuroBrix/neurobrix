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

    # Consumer index, needed for the placement window below.
    pos = {uid: i for i, uid in enumerate(order)}
    first_reader: dict[str, int] = {}
    for i, uid in enumerate(order):
        for tid in ops[uid].get("input_tensor_ids") or []:
            if tid not in first_reader:
                first_reader[tid] = i

    groups = []
    for src, members in closed:
        ks = {m["k"] for m in members}
        dts = {(m["dtype"], m["device"]) for m in members}
        if len(ks) != 1 or len(dts) != 1:
            # Refuse rather than reshape (contract clause 3).
            continue

        # CLAUSE 6 — THE PLACEMENT WINDOW MUST EXIST.
        # The fused chain needs every member's weight AND the source
        # available, so it cannot run before the latest of those
        # producers; and it must produce every member's output before
        # anything reads one, so it cannot run after the earliest such
        # reader. Measured 2026-08-19: splicing at the first member's
        # position instead produced 6,144 topological violations on the
        # int4 decode graph alone, because a later member's weight
        # transpose sits after the first member's matmul. A group whose
        # window is empty is REFUSED, not reordered — moving real work
        # across a consumer is not this pass's licence.
        # Only NON-CONSTANT inputs constrain the start of the window.
        # A member's weight is `t(param)` — a view over a parameter,
        # available from bind — so its transpose imposes no real
        # ordering: the rewrite hoists those transposes into the fused
        # chain. Measured why this matters: the graph INTERLEAVES them
        # (t::5, mm::5, <reader of q>, t::6, mm::6), so counting them as
        # constraints made every window empty on the int4 decode graph.
        earliest = 0
        puid = producer.get(src)
        if puid is not None:
            earliest = max(earliest, pos[puid] + 1)
        # Only genuine CONSUMERS bound the window from above: the
        # members' own matmuls are what the chain replaces, so moving
        # the compute later than they sat is legitimate as long as no
        # reader of a member output has run yet.
        latest = min(first_reader.get(m["out_tid"], len(order) - 1)
                     for m in members)
        if earliest > latest:
            continue

        groups.append({
            "source": src,
            "members": [
                {k2: m[k2] for k2 in ("uid", "weight_tid", "out_tid", "cols")}
                for m in members
            ],
            "k": next(iter(ks)),
            "total_cols": sum(m["cols"] for m in members),
            "anchor": order[latest],
        })

    if not groups:
        return None

    return {
        "pass": PASS_NAME,
        "groups": groups,
        "n_groups": len(groups),
        "n_ops_saved": sum(len(g["members"]) - 1 for g in groups),
    }


def apply_fusion_horizontal(dag: dict, plan: dict) -> int:
    """Rewrite `dag` in place so each group runs as ONE wider matmul.

    Engine-neutral by construction: the rewrite emits only ATen ops the
    two engines already execute, so there is no new kernel, no new
    custom op and no per-engine lowering to keep in sync — the
    "reuse the brick" discipline. Per group:

      cat([w_1 .. w_n], dim=1) -> w_fused        (constant-weight concat)
      mm(source, w_fused)      -> wide_out
      slice(wide_out, dim=1, a_i, b_i) -> the member's ORIGINAL out tid

    Every consumer keeps reading the tid it always read, so nothing
    downstream needs rewiring, and the positional output contract is
    untouched (members producing a graph output were excluded by the
    planner).

    WHY THE CONCAT IS NOT A PER-STEP COST. Its inputs are `aten::t` of
    parameters, so the `cat` is computable from parameters alone —
    which is exactly const_fold's definition of a foldable op. Running
    this rewrite BEFORE const_fold plans lets that pass hoist every
    concat to bind time through machinery both engines already have.
    Run without const_fold the graph is still correct, merely paying a
    weight-sized concat per forward; that is why the D1 hook orders
    them.

    Returns the number of launches removed.
    """
    ops: dict = dag["ops"]
    tensors: dict = dag.setdefault("tensors", {})
    order: list = dag["execution_order"]
    producer = {t: u for u in order for t in (ops[u].get("output_tensor_ids") or [])}
    pos = {uid: i for i, uid in enumerate(order)}
    chain_uids: dict[int, list[str]] = {}
    hoisted_all: set[str] = set()
    removed = 0

    for gi, group in enumerate(plan["groups"]):
        members = group["members"]
        src = group["source"]
        k = group["k"]
        anchor = group["anchor"]

        w_fused = f"custom.fh_w::{gi}::out_0"
        wide_out = f"custom.fh_out::{gi}::out_0"
        cat_uid = f"custom.fh_cat::{gi}"
        mm_uid = f"custom.fh_mm::{gi}"

        first_out_meta = tensors.get(members[0]["out_tid"]) or {}
        dtype = first_out_meta.get("dtype")
        device = first_out_meta.get("device")
        rows = (first_out_meta.get("shape") or [None])[0]

        # Hoist the members' weight transposes into the chain: they are
        # views over parameters, so an earlier position is always legal.
        hoisted = []
        for m in members:
            wp = producer.get(m["weight_tid"])
            if wp is not None and wp in ops:
                hoisted.append(wp)

        ops[cat_uid] = {
            "op_uid": cat_uid,
            "op_type": "aten::cat",
            "input_tensor_ids": [m["weight_tid"] for m in members],
            "output_tensor_ids": [w_fused],
            "attributes": {"args": [{"type": "scalar", "value": 1}]},
        }
        tensors[w_fused] = {"shape": [k, group["total_cols"]],
                            "dtype": (tensors.get(members[0]["weight_tid"]) or {}).get("dtype"),
                            "device": device}

        ops[mm_uid] = {
            "op_uid": mm_uid,
            "op_type": "aten::mm",
            "input_tensor_ids": [src, w_fused],
            "output_tensor_ids": [wide_out],
            "attributes": {},
        }
        tensors[wide_out] = {"shape": [rows, group["total_cols"]],
                             "dtype": dtype, "device": device}

        band_uids = []
        offset = 0
        for mi, m in enumerate(members):
            slice_uid = f"custom.fh_band::{gi}_{mi}"
            ops[slice_uid] = {
                "op_uid": slice_uid,
                "op_type": "aten::slice",
                "input_tensor_ids": [wide_out],
                # dim, start, end, step — the decomposed-ATen form both
                # engines already dispatch.
                "attributes": {"args": [
                    {"type": "scalar", "value": 1},
                    {"type": "scalar", "value": offset},
                    {"type": "scalar", "value": offset + m["cols"]},
                    {"type": "scalar", "value": 1},
                ]},
                "output_tensor_ids": [m["out_tid"]],
            }
            band_uids.append(slice_uid)
            offset += m["cols"]

        # The chain replaces the members; the hoisted transposes and the
        # new ops are ordered by the topological rebuild below, which is
        # provably correct where incremental splicing is not (an
        # interleaved graph makes every hand-placed insertion a special
        # case — measured: the first splice attempt produced 6,144
        # topological violations, the second dropped ops out of the
        # order entirely).
        chain_uids[gi] = [*hoisted, cat_uid, mm_uid, *band_uids]
        for uid in {m["uid"] for m in members}:
            ops.pop(uid, None)
            hoisted_all.discard(uid)
        hoisted_all.update(hoisted)
        removed += len(members) - 1

    # ---- rebuild execution_order: stable topological sort ------------
    # Seeded from the ORIGINAL relative order so an untouched graph
    # keeps its exact sequence (R28: the emitted order must be a
    # function of the graph, not of insertion history).
    rank = {uid: i for i, uid in enumerate(order)}
    for uids in chain_uids.values():
        for j, uid in enumerate(uids):
            rank.setdefault(uid, len(order) + j)
    producer_now = {t: u for u, o in ops.items()
                    for t in (o.get("output_tensor_ids") or [])}
    indeg: dict[str, int] = {u: 0 for u in ops}
    dependents: dict[str, list[str]] = {u: [] for u in ops}
    for uid, op in ops.items():
        for tid in op.get("input_tensor_ids") or []:
            pu = producer_now.get(tid)
            if pu is not None and pu != uid:
                dependents[pu].append(uid)
                indeg[uid] += 1
    import heapq
    ready = [(rank.get(u, 0), u) for u, d in indeg.items() if d == 0]
    heapq.heapify(ready)
    new_order = []
    while ready:
        _, uid = heapq.heappop(ready)
        new_order.append(uid)
        for dep in dependents[uid]:
            indeg[dep] -= 1
            if indeg[dep] == 0:
                heapq.heappush(ready, (rank.get(dep, 0), dep))
    if len(new_order) != len(ops):
        raise RuntimeError(
            f"ZERO FALLBACK: fusion_horizontal rebuilt an order of "
            f"{len(new_order)} ops for {len(ops)} ops — the rewrite "
            f"introduced a cycle")

    order.clear()
    order.extend(new_order)
    return removed
