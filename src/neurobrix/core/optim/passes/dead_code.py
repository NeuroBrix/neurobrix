"""dead_code — unreachable-op elimination on the LOADED graph.

EXACT-gate pass (policy.py). Engine-neutral half: pure-dict analysis at
the D1 hook. It removes operations in the executed sequence that no
graph output depends on.

=== HOW THIS DIFFERS FROM R19 ===

R19 prunes dead subgraphs at BUILD time, born-at-source, and re-emits
graph.json. This pass runs on the graph the engine actually loaded, and
the two are not redundant:

  - R19 has only been applied to the six models rebuilt on 2026-08-19.
    Every other build on disk predates it — Open-Sora carries 10,784
    unreachable ops, Wan2.2 2,605, Wan VACE 1,495, swin2SR 972 each.
  - The loaded graph is not the stored graph: MoE fusion has run, so
    reachability is computed over what will really execute.
  - A build can arrive from the hub already published; the engine
    cannot re-trace it, but it can refuse to execute what nothing
    reads.

=== THE CONTRACT IS R19'S, AND IT WAS PAID FOR ===

Tid-reachability is NOT liveness. The tracer functionalises in-place
writes: `dst[:, k] = v` becomes slice -> temporary view -> `copy`, and
that copy gets a FRESH output tid nothing consumes. The engines execute
`aten::copy` as `x.copy_(src)` — a real mutation of the input's
storage, observable through the base tensor. Deleting it is the
documented silent-corruption class (OpenAudio DAC codebook all-zero ->
silence; the triton slice path records the same class as "washed-out
VACE/Wan2.2, 2026-06-28"). The R19 chantier shipped that bug and had to
withdraw it the same day.

So the roots are: graph outputs, every tensor referenced by
`computable_buffers`, AND every side-effecting op. Membership of that
last set is decided by reading the engines' dispatch tables, never by
the op's name — audited 2026-08-19: `aten::copy` mutates in all three
paths and `aten::uniform`'s wrapper writes into its input, while
`fill`, `scatter`, `index_put` and `masked_fill` are implemented
functionally and stay eligible. Random draws are retained too: the
seeded stream's draw ORDER is the cross-engine determinism contract.

The analyzer's own dead-code detector does NOT model this (its
mutating-suffix test never fires, because the tracer strips the
underscore), so its count is an upper bound and this planner is
deliberately stricter.
"""

from __future__ import annotations

PLAN_KEY = "_optim_dead_code"
PASS_NAME = "dead_code"

# Ops the ENGINES execute as a write into an input's storage. Roots.
_SIDE_EFFECTING = {"aten::copy", "aten::uniform"}

# Ops whose output is not a pure function of their inputs. Retained so
# the pinned draw order is unchanged for the live draws.
_NONDETERMINISTIC = {
    "aten::rand", "aten::randn", "aten::rand_like", "aten::randn_like",
    "aten::randint", "aten::randint_like", "aten::multinomial",
    "aten::bernoulli", "aten::normal", "aten::uniform",
    "aten::exponential", "aten::randperm",
}


def _tensor_refs(node, universe: set, out: list) -> None:
    if isinstance(node, str):
        if node in universe and node not in out:
            out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            _tensor_refs(v, universe, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _tensor_refs(v, universe, out)


def plan_dead_code(dag: dict) -> dict | None:
    """Return the set of unreachable ops to skip, or None when clean."""
    ops: dict = dag.get("ops", {})
    tensors: dict = dag.get("tensors", {})
    order = [u for u in dag.get("execution_order", []) if u in ops]
    outputs = list(dag.get("output_tensor_ids") or [])
    if not outputs:
        # Legacy graphs define outputs AS the consumerless tensors;
        # pruning there would delete them. Refuse loudly rather than
        # guess (same clause as the build-side pass).
        return None

    universe: set = set(tensors)
    for uid in order:
        universe.update(ops[uid].get("output_tensor_ids") or [])

    producer: dict = {}
    for uid in order:
        for tid in ops[uid].get("output_tensor_ids") or []:
            producer[tid] = uid

    def inputs_of(op: dict) -> list:
        tids = list(op.get("input_tensor_ids") or [])
        attrs = op.get("attributes") or {}
        for bucket in (attrs.get("args") or [], list((attrs.get("kwargs") or {}).values())):
            for ref in bucket:
                if isinstance(ref, dict):
                    t = ref.get("tensor_id")
                    if isinstance(t, str) and t not in tids:
                        tids.append(t)
                    for t2 in ref.get("tensor_ids") or []:
                        if isinstance(t2, str) and t2 not in tids:
                            tids.append(t2)
        return tids

    roots: list = list(outputs)
    _tensor_refs(dag.get("computable_buffers"), universe, roots)

    live: set = set()

    def close_over(seed_tids) -> None:
        stack = [producer[t] for t in seed_tids if t in producer]
        while stack:
            uid = stack.pop()
            if uid in live:
                continue
            live.add(uid)
            for tid in inputs_of(ops[uid]):
                p = producer.get(tid)
                if p is not None and p not in live:
                    stack.append(p)

    close_over(roots)

    # Side-effecting ops and random draws are roots in their own right.
    retained_writes, retained_rng = [], []
    for uid in order:
        if uid in live:
            continue
        t = ops[uid].get("op_type")
        if t in _SIDE_EFFECTING:
            retained_writes.append(uid)
            live.add(uid)
            close_over(inputs_of(ops[uid]))
    for uid in order:
        if uid in live:
            continue
        if ops[uid].get("op_type") in _NONDETERMINISTIC:
            retained_rng.append(uid)
            live.add(uid)
            close_over(inputs_of(ops[uid]))

    # A non-aten op is a contract this planner does not model: keep it
    # and everything it reads.
    for uid in order:
        if uid in live:
            continue
        if not str(ops[uid].get("op_type", "")).startswith("aten::"):
            live.add(uid)
            close_over(inputs_of(ops[uid]))

    dead = [uid for uid in order if uid not in live]
    if not dead:
        return None

    return {
        "pass": PASS_NAME,
        "dead_uids": dead,
        "n_ops": len(dead),
        "retained_writes": len(retained_writes),
        "retained_rng": len(retained_rng),
    }
