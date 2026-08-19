"""algebraic — exact identity elimination on the executed sequence.

EXACT-gate pass (policy.py). Engine-neutral half: pure-dict analysis at
the D1 hook. It finds operations whose output IS their input — a
full-range slice, a view/expand/reshape to the shape the tensor already
has — and emits an alias plan so every consumer reads the input
directly and the operation leaves the sequence.

=== THE DIVIDING LINE, DECIDED BEFORE IMPLEMENTATION ===

The family the analyzer reports as `algebraic` is two families, and
only one of them belongs to an EXACT gate. Measured on the executed
surface (gains map v2.1, 2026-08-19):

  BIT-EXACT — NO ARITHMETIC HAPPENS AT ALL (25,829 ops, this pass)
    full-range slice (sentinel end)      15,600
    expand to identical shape             8,098
    view to identical shape                 921
    full-range slice (symbolic end==dim)    913
    _unsafe_view to identical shape         281
    full-range slice (concrete dim)          10
    alias to identical shape                  4
    reshape to identical shape                2
  These return the input unchanged. Removing one cannot alter a value
  because no value is computed; the output tid simply becomes another
  name for the input tid.

  FLOATING-POINT ARITHMETIC — NOT EXACT (1,632 ops, NOT this pass)
    mul by scalar 1   638
    div by scalar 1   559
    add of scalar 0   435
  `x * 1` looks like an identity and is not one in IEEE-754 as the
  engines actually run it: under flush-to-zero a denormal input becomes
  zero WITH the operation and stays denormal without it; `x + 0`
  turns -0.0 into +0.0; and a fp32 scalar operand can promote the
  result dtype, which is the DtypeEngine's business, not this pass's.
  These belong to `cancellation`, the FLOATING-gate pass already
  declared in the registry with default_on=False. They are recorded
  here with their number so the split is visible, and left alone.

That line is the mandate's, and it is drawn before the gate rather
than after it: a pass does not get reclassified because its byte gate
was inconvenient.

=== THE CONTRACT ===

1. IDENTITY MUST BE STRUCTURAL, not inferred from a runtime value. A
   slice qualifies only when its bounds cover the whole axis by
   construction (sentinel end, or end == the dim it slices); a
   view/expand/reshape qualifies only when the recorded output shape
   equals the recorded input shape.
2. NO GRAPH OUTPUT IS ALIASED AWAY. The positional output contract is
   frozen.
3. TENSORS TOUCHED BY A `custom::` OP ARE OPAQUE, as in cse and
   const_fold: those contracts are not modelled here.
4. MUTATION IS NOT A BARRIER HERE, and that is deliberate. This pass
   does not move or merge anything — it renames one tid to another
   over the whole graph. Whatever writes into that storage keeps
   writing into the same storage; there is no window in which the two
   names could disagree.
"""

from __future__ import annotations

PLAN_KEY = "_optim_algebraic"
PASS_NAME = "algebraic"

_VIEW_IDENTITY = {
    "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::expand", "aten::alias",
}

# Recorded, deliberately NOT eliminated — see the module docstring.
_FLOATING_IDENTITY = {"aten::mul", "aten::div", "aten::add"}

_SENTINEL_ENDS = (9223372036854775807, -1)


def _scalar(arg) -> object:
    if isinstance(arg, dict) and arg.get("type") == "scalar":
        return arg.get("value")
    return None


def plan_algebraic(dag: dict) -> dict | None:
    """Return the identity-elimination plan, or None when there is none.

    Plan shape mirrors cse's so the engines reuse one lowering:
      merges: [{"keep": producer_uid_or_None, "drop": uid,
                "alias": [[dead_tid, live_tid]]}]
    `keep` is informational here (the surviving value already exists);
    the lowering only needs `drop` and `alias`.
    """
    ops: dict = dag.get("ops", {})
    tensors: dict = dag.get("tensors", {})
    order = [u for u in dag.get("execution_order", []) if u in ops]
    graph_outputs = set(dag.get("output_tensor_ids") or [])

    opaque: set[str] = set()
    for op in ops.values():
        if not str(op.get("op_type", "")).startswith("aten::"):
            opaque.update(op.get("input_tensor_ids") or [])
            opaque.update(op.get("output_tensor_ids") or [])

    def shape_of(tid: str):
        return (tensors.get(tid) or {}).get("shape")

    merges: list[dict] = []
    n_floating = 0

    for uid in order:
        op = ops[uid]
        op_type = str(op.get("op_type", ""))
        ins = list(op.get("input_tensor_ids") or [])
        outs = list(op.get("output_tensor_ids") or [])
        if len(ins) < 1 or len(outs) != 1:
            continue
        src, out = ins[0], outs[0]
        if out in graph_outputs or out in opaque or src in opaque:
            continue

        identity = False

        if op_type in _VIEW_IDENTITY:
            si, so = shape_of(src), shape_of(out)
            identity = (si is not None and si == so)

        elif op_type == "aten::slice":
            args = (op.get("attributes") or {}).get("args") or []
            if len(args) >= 3:
                dim, start, end = (_scalar(a) for a in args[:3])
                step = _scalar(args[3]) if len(args) > 3 else 1
                if (isinstance(dim, int) and start == 0
                        and (step == 1 or step is None)):
                    si = shape_of(src)
                    if end in _SENTINEL_ENDS:
                        identity = True
                    elif (isinstance(end, int) and si
                          and -len(si) <= dim < len(si)
                          and si[dim] == end):
                        identity = True

        elif op_type in _FLOATING_IDENTITY:
            # Counted, never removed. See the docstring: these are
            # `cancellation`'s business, under a different gate.
            args = (op.get("attributes") or {}).get("args") or []
            v = _scalar(args[1]) if len(args) > 1 else None
            neutral = {"aten::mul": 1, "aten::div": 1, "aten::add": 0}
            if v is not None and v == neutral[op_type]:
                n_floating += 1
            continue

        if identity:
            merges.append({"keep": None, "drop": uid,
                           "alias": [[out, src]]})

    if not merges:
        return None

    return {
        "pass": PASS_NAME,
        "merges": merges,
        "n_ops": len(merges),
        "floating_identities_seen": n_floating,
    }
