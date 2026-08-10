"""fusion_vertical — shared planner for matmul + unary-epilogue fusion.

EXACT-gate pass (policy.py). ENGINE-NEUTRAL half (scoping doc D2):
pure-dict analysis over the shared in-memory graph at the D1 hook
(`GraphExecutor._init_from_dag`, after MoE fusion). It identifies
anchor -> (layout-transparent)* -> unary-epilogue chains and annotates
the dag; it never rewrites ops and never imports torch.

PER-ENGINE lowering: the triton engine fuses each verified group at
compile time (TritonSequence Phase -0.28, after the swiglu/rope
fusions) into `custom::mm_epilogue` / `custom::addmm_epilogue`, backed
by the SAME `matmul_kernel` / `addmm_kernel` functions with an
`EPILOGUE` tl.constexpr — one launch for the pair, byte-identical to
the unfused pair via per-stage rounding emulation plus shared-autotune
config parity (both variants of one kernel function share one tuning
cache entry; two separate kernels would not — R16 research 2026-08-10).
The compiled engine deliberately ignores the annotation (D2-legit:
cuBLAS launch cost does not dominate there); the sequential paths are
oracles and never lower optimization annotations.

v1 scope (deposit-probe 2026-08-10, validation_outputs/
optim_fusion_vertical/deposit_probe): unary epilogues silu / gelu on
mm / addmm anchors. The scouted "linear+bias -> addmm" quick win
measured ~0 on the reference rows (bias'd linears trace as aten::addmm
already; aten::linear is absent from the zoo graphs) — the launches
live in the activation epilogues instead: Ming denoiser 51/step,
Qwen3-VL vision tower 31, MiniCPM voice path 17 direct. Binary
epilogues (residual add, mul) and the bmm anchor are the named next
increments.

Safety rules (all enforced here, re-verified by the emitter on the
live post-fusion op state):
  - single consumer at EVERY link of the chain;
  - intermediate hops restricted to LAYOUT-ONLY metadata ops — dtype
    casts (`_to_copy`, `to`) are NOT transparent: gelu(cast(x)) !=
    cast(gelu(x));
  - no chain tensor may be a graph output (the anchor's output tensor
    holds ACTIVATED data after fusion);
  - no chain tensor may be touched by a custom:: attribute contract or
    an in-place mutator (aliasing guard, const_fold M1 precedent).
"""

from __future__ import annotations

PLAN_KEY = "_optim_fusion_vertical"
PASS_NAME = "fusion_vertical"

_ANCHORS = ("aten::mm", "aten::addmm")

# Unary elementwise epilogues fused in v1. Maps op_type -> epilogue tag.
_EPILOGUES = {"aten::silu": "silu", "aten::gelu": "gelu"}

# Layout-only metadata ops the chain may traverse: they commute exactly
# with a unary elementwise epilogue (same storage, reinterpreted).
# `_to_copy` / `to` are deliberately ABSENT (they cast).
LAYOUT_TRANSPARENT = frozenset({
    "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::unsqueeze", "aten::squeeze", "aten::alias",
    "aten::detach", "aten::clone", "aten::contiguous",
})

_MUTATING_SUFFIX = "_"
_MAX_HOPS = 6


def _gelu_approximate(op: dict) -> str | None:
    """Parse aten::gelu's `approximate` attribute across the observed
    trace encodings: plain attr key, kwargs scalar dict, args scan.

    FAIL-CLOSED (gardien 2026-08-10): when an `approximate` attribute
    is PRESENT but not parseable to "tanh"/"none", return None and the
    planner SKIPS the group — a skipped fusion costs one launch, a
    mis-parsed one ships wrong numerics under the EXACT gate. When no
    approximate attribute exists anywhere, the ATen default "none"
    applies (that is the op's semantic, not a parse failure).
    """
    attrs = op.get("attributes") or {}
    v = attrs.get("approximate")
    if isinstance(v, str):
        return v if v in ("tanh", "none") else None
    kw = (attrs.get("kwargs") or {}).get("approximate")
    if isinstance(kw, dict):
        kw = kw.get("value")
    if kw is not None:
        return kw if kw in ("tanh", "none") else None
    for a in attrs.get("args") or []:
        if isinstance(a, str) and a in ("tanh", "none"):
            return a
        if isinstance(a, dict) and a.get("value") in ("tanh", "none"):
            return a.get("value")
    if v is not None:
        return None  # present in an unrecognized form
    return "none"  # attribute absent everywhere: ATen default


def plan_fusion_vertical(dag: dict) -> dict | None:
    """Identify fusable anchor->epilogue groups and return the plan.

    Returns None when no group is found. Otherwise:
      {"pass": "fusion_vertical",
       "groups": [{"anchor_uid", "epilogue_uid", "epilogue",
                   "gelu_tanh", "via_hops"}, ...]}
    The dag is NOT rewritten here; each engine lowers (or ignores) the
    annotation at its own compile step.
    """
    ops: dict = dag.get("ops", {})
    tensors: dict = dag.get("tensors", {})
    order = [u for u in dag.get("execution_order", []) if u in ops]
    graph_outputs = set(dag.get("output_tensor_ids") or [])

    def is_graph_output(tid: str) -> bool:
        # Output-ness is a UNION at the categorizers (gardien HIGH,
        # 2026-08-10): `tensors[tid]["is_output"]` is live and
        # independently propagated (detach elimination transfers it
        # without touching output_tensor_ids) — guarding on the id
        # list alone lets a chain tensor the arena protects as a
        # component output get silently redefined.
        if tid in graph_outputs:
            return True
        t = tensors.get(tid)
        return bool(t and t.get("is_output"))

    # Consumer index from the ops themselves (tensors' consumer lists
    # can be stale after MoE fusion rewrites).
    consumers: dict[str, list[str]] = {}
    for uid in order:
        for tid in ops[uid].get("input_tensor_ids") or []:
            consumers.setdefault(tid, []).append(uid)

    # Pinned tensors: referenced by custom:: attribute contracts, or
    # touched by an in-place mutator (aliasing guard).
    pinned: set[str] = set()

    def collect_ids(node) -> None:
        if isinstance(node, dict):
            tid = node.get("tensor_id")
            if isinstance(tid, str):
                pinned.add(tid)
            for v in node.values():
                collect_ids(v)
        elif isinstance(node, list):
            for v in node:
                collect_ids(v)
            if all(isinstance(x, str) for x in node):
                pinned.update(x for x in node if "::" in x)

    for op in ops.values():
        op_type = str(op.get("op_type", ""))
        if not op_type.startswith("aten::"):
            pinned.update(op.get("input_tensor_ids") or [])
            collect_ids(op.get("attributes", {}))
        elif op_type.endswith(_MUTATING_SUFFIX):
            pinned.update(op.get("input_tensor_ids") or [])
            pinned.update(op.get("output_tensor_ids") or [])

    groups: list[dict] = []
    claimed: set[str] = set()  # op uids already in a group (overlap dedup)

    for uid in order:
        op = ops[uid]
        if op.get("op_type") not in _ANCHORS or uid in claimed:
            continue
        outs = op.get("output_tensor_ids") or []
        if len(outs) != 1:
            continue

        cur = op
        hops = 0
        chain_tids: list[str] = []
        epilogue_uid = None
        while hops <= _MAX_HOPS:
            c_outs = cur.get("output_tensor_ids") or []
            if len(c_outs) != 1:
                break
            tid = c_outs[0]
            if is_graph_output(tid) or tid in pinned:
                break
            cons = consumers.get(tid, [])
            if len(cons) != 1:
                break
            nxt_uid = cons[0]
            nxt = ops.get(nxt_uid)
            if nxt is None or nxt_uid in claimed:
                break
            nxt_type = nxt.get("op_type")
            if nxt_type in _EPILOGUES:
                e_outs = nxt.get("output_tensor_ids") or []
                e_ins = nxt.get("input_tensor_ids") or []
                if (len(e_outs) == 1 and len(e_ins) == 1
                        and not is_graph_output(e_outs[0])
                        and e_outs[0] not in pinned):
                    epilogue_uid = nxt_uid
                    chain_tids.append(tid)
                break
            if nxt_type not in LAYOUT_TRANSPARENT:
                break
            chain_tids.append(tid)
            cur = nxt
            hops += 1

        if epilogue_uid is None:
            continue
        e_op = ops[epilogue_uid]
        gelu_tanh = False
        if e_op["op_type"] == "aten::gelu":
            approx = _gelu_approximate(e_op)
            if approx is None:
                continue  # fail-closed: unparseable approximate = skip
            gelu_tanh = approx == "tanh"
        groups.append({
            "anchor_uid": uid,
            "epilogue_uid": epilogue_uid,
            "epilogue": _EPILOGUES[e_op["op_type"]],
            "gelu_tanh": gelu_tanh,
            "via_hops": hops,
        })
        claimed.add(uid)
        claimed.add(epilogue_uid)

    if not groups:
        return None
    return {"pass": PASS_NAME, "groups": groups}
