"""Graph analyzer — value-flow analysis over the shared graph contract.

Pure-Python, engine-neutral, read-only: the analyzer consumes the
graph.json dict both engines already parse and maps every optimization
opportunity onto its declared pass class (policy.py). It never
transforms anything — Phase 1's whole delivery is this map.

Detector → pass mapping:
  const_fold        ops computable from parameters alone
  algebraic         exact identities + exact cancelling motifs
                    (integer/bool dtypes)
  cancellation      the same cancelling motifs on floating dtypes
  cse               duplicated (op, inputs, attrs) computations
  dead_code         ops that never reach a graph output
  layout            transpose/permute composition + sink candidates
  fusion_vertical   matmul-class anchor + elementwise epilogue chain
  fusion_horizontal same-op same-shape siblings on one source tensor

Every finding is conservative: a site is reported only when the
pattern is structurally certain from the graph alone. Sites involving
symbolic extents are flagged `symbolic` (the future pass must prove
them per-shape or at the symbol level).
"""

from __future__ import annotations

from collections import defaultdict

from .report import AnalysisReport, Finding, graph_fingerprint

# Ops whose output is NOT a pure function of inputs — never const-fold,
# never CSE.
_NONDETERMINISTIC = {
    "aten::rand",
    "aten::randn",
    "aten::rand_like",
    "aten::randn_like",
    "aten::randint",
    "aten::multinomial",
    "aten::bernoulli",
    "aten::normal",
    "aten::uniform",
    "aten::exponential",
}

# In-place / aliasing mutators — excluded from CSE and dead-code by
# conservatism (their effect is on storage, not only on their output).
_MUTATING_SUFFIX = "_"

_ELEMENTWISE_UNARY = {
    "aten::silu", "aten::gelu", "aten::relu", "aten::tanh",
    "aten::sigmoid", "aten::neg", "aten::exp", "aten::log",
    "aten::rsqrt", "aten::sqrt", "aten::abs", "aten::clamp",
    "aten::logical_not",
}
_ELEMENTWISE_BINARY = {
    "aten::add", "aten::sub", "aten::mul", "aten::div",
    "aten::maximum", "aten::minimum", "aten::pow",
}
_ELEMENTWISE = _ELEMENTWISE_UNARY | _ELEMENTWISE_BINARY

_MATMUL_ANCHORS = {
    "aten::mm", "aten::addmm", "aten::bmm", "aten::matmul",
    "aten::convolution", "aten::conv2d", "aten::linear",
}

# NOTE: aten::contiguous is deliberately ABSENT — whether it is a
# no-op depends on strides, which the graph cannot prove for symbolic
# extents (contiguous-guard doctrine, commit a862fe0).
_VIEW_LIKE_SAME_SHAPE = {
    "aten::view", "aten::_unsafe_view", "aten::reshape",
    "aten::expand", "aten::alias",
}

# Fusion-transparent ops: pure index-math bookkeeping a fused kernel
# absorbs into its addressing — fusion chains may pass through them,
# and findings on them are flagged `meta` (they save bookkeeping, not
# launches). Deliberately NARROWER than the METADATA class in
# kernels/classification.py, which also covers memory-copy ops
# (cat/stack/narrow/select/...) that a fusion chain must NOT traverse.
_FUSION_TRANSPARENT_OPS = {
    "aten::view", "aten::_unsafe_view", "aten::reshape", "aten::alias",
    "aten::slice", "aten::unsqueeze", "aten::squeeze", "aten::expand",
    "aten::t", "aten::transpose", "aten::permute",
}

_INT64_MAX_SENTINEL = 2 ** 62  # any slice end >= this means "to the end"

_INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "bool"}


def _arg_scalar(args: list, index: int):
    if index < len(args) and args[index].get("type") == "scalar":
        return args[index].get("value")
    return None


class GraphAnalyzer:
    """One instance per component graph; `run()` returns the report."""

    def __init__(self, model: str, component: str, graph: dict,
                 raw_bytes: bytes):
        self.model = model
        self.component = component
        self.g = graph
        self.ops: dict = graph["ops"]
        self.tensors: dict = graph["tensors"]
        # Containers can carry uids in execution_order / consumer lists
        # that were pruned from `ops` at build time — normalize once so
        # every detector sees a closed graph.
        self.order: list[str] = [
            u for u in graph["execution_order"] if u in self.ops
        ]
        self.report = AnalysisReport(
            model=model,
            component=component,
            graph_fingerprint=graph_fingerprint(raw_bytes),
            n_ops=len(self.ops),
            n_tensors=len(self.tensors),
        )

    # -- helpers -------------------------------------------------------

    def _producer(self, tensor_id: str) -> str | None:
        t = self.tensors.get(tensor_id)
        p = t.get("producer_op_uid") if t else None
        return p if p in self.ops else None

    def _consumers(self, tensor_id: str) -> list[str]:
        t = self.tensors.get(tensor_id)
        cons = list(t.get("consumer_op_uids") or []) if t else []
        return [c for c in cons if c in self.ops]

    def _is_param(self, tensor_id: str) -> bool:
        t = self.tensors.get(tensor_id)
        return bool(t and t.get("is_parameter"))

    def _tensor_inputs(self, op: dict) -> list[str]:
        return list(op.get("input_tensor_ids") or [])

    def _op_dtype_class(self, op: dict) -> str:
        outs = op.get("output_dtypes") or []
        if outs and all(d in _INT_DTYPES for d in outs):
            return "int"
        return "float"

    def _has_symbol(self, op: dict) -> bool:
        def walk(node) -> bool:
            if isinstance(node, dict):
                if node.get("type") == "symbol":
                    return True
                return any(walk(v) for v in node.values())
            if isinstance(node, list):
                return any(walk(v) for v in node)
            return False
        return walk(op.get("attributes", {}))

    def _sym_dims(self, tensor_id: str) -> list | None:
        """The tensor's symbolic dims ([symbol-node | int, ...]), or
        None when the graph carries no symbolic_shape for it."""
        t = self.tensors.get(tensor_id)
        ss = (t or {}).get("symbolic_shape")
        return ss.get("dims") if isinstance(ss, dict) else None

    @staticmethod
    def _dim_is_symbolic(entry) -> bool:
        def walk(node) -> bool:
            if isinstance(node, dict):
                if node.get("type") == "symbol":
                    return True
                return any(walk(v) for v in node.values())
            if isinstance(node, list):
                return any(walk(v) for v in node)
            return False
        return walk(entry)

    def _shape_key(self, tensor_id: str) -> str:
        """Grouping key honest to symbolic extents: the symbolic dims
        when the graph declares them, the concrete shape otherwise."""
        import json as _json

        dims = self._sym_dims(tensor_id)
        if dims is not None:
            return _json.dumps(dims, sort_keys=True, default=str)
        t = self.tensors.get(tensor_id) or {}
        return _json.dumps(t.get("shape"), default=str)

    # -- detectors -----------------------------------------------------

    def _detect_const_fold(self) -> set[str]:
        """Ops transitively computable from parameters alone."""
        const_tensors: set[str] = {
            tid for tid, t in self.tensors.items() if t.get("is_parameter")
        }
        const_ops: set[str] = set()
        for uid in self.order:
            op = self.ops[uid]
            if op["op_type"] in _NONDETERMINISTIC:
                continue
            tin = self._tensor_inputs(op)
            if tin and all(t in const_tensors for t in tin):
                const_ops.add(uid)
                const_tensors.update(op.get("output_tensor_ids") or [])
        # Report the FRONTIER (const op feeding a non-const consumer):
        # those outputs become the new load-time constants.
        for uid in sorted(const_ops):
            op = self.ops[uid]
            outs = op.get("output_tensor_ids") or []
            if any(
                c not in const_ops
                for o in outs
                for c in self._consumers(o)
            ):
                self.report.add(Finding(
                    category="const_fold",
                    op_uids=[uid],
                    detail=f"{op['op_type']} computable from parameters",
                    ops_removable=1,
                    symbolic=self._has_symbol(op),
                ))
        return const_ops

    def _detect_identities(self) -> None:
        for uid in self.order:
            op = self.ops[uid]
            t = op["op_type"]
            args = op.get("attributes", {}).get("args", [])
            detail = None
            if t in ("aten::mul", "aten::div"):
                if _arg_scalar(args, 1) in (1, 1.0):
                    detail = f"{t} by scalar 1"
            elif t in ("aten::add", "aten::sub"):
                if _arg_scalar(args, 1) in (0, 0.0):
                    detail = f"{t} of scalar 0"
            elif t == "aten::pow":
                if _arg_scalar(args, 1) in (1, 1.0):
                    detail = "pow 1"
            elif t == "aten::_to_copy":
                # Identity ONLY when nothing changes: same dtype AND no
                # device/memory_format request in kwargs (_to_copy also
                # encodes transfers, which are never identities).
                ind = op.get("input_dtypes") or []
                outd = op.get("output_dtypes") or []
                kw = set(op.get("attributes", {}).get("kwargs", {}))
                if ind and ind == outd and kw <= {"dtype"}:
                    detail = f"cast to same dtype {outd[0]}"
            elif t in _VIEW_LIKE_SAME_SHAPE:
                # Same-shape is only an identity if it holds SYMBOLICALLY
                # — concrete trace-shape equality on symbolic dims proves
                # nothing at other extents.
                tin = self._tensor_inputs(op)
                outs_t = op.get("output_tensor_ids") or []
                if tin and outs_t:
                    in_key = self._shape_key(tin[0])
                    out_key = self._shape_key(outs_t[0])
                    in_dims = self._sym_dims(tin[0])
                    out_dims = self._sym_dims(outs_t[0])
                    concrete_ok = (
                        in_dims is None and out_dims is None
                        and (op.get("input_shapes") or [None])[0]
                        == (op.get("output_shapes") or [None])[0]
                        and (op.get("input_shapes") or [None])[0] is not None
                    )
                    if (in_dims is not None and in_key == out_key) \
                            or concrete_ok:
                        detail = f"{t} to identical shape"
            elif t in ("aten::t", "aten::transpose"):
                # composition: input produced by the same transpose
                src = self._tensor_inputs(op)
                if src:
                    p = self._producer(src[0])
                    if p and self.ops[p]["op_type"] == t:
                        if t == "aten::t" or (
                            self.ops[p].get("attributes", {}).get("args")
                            == op.get("attributes", {}).get("args")
                        ):
                            detail = f"{t} of {t} (composes to identity)"
            elif t == "aten::slice":
                # Symbolic honesty (slice-attrs doctrine): a concrete
                # end that merely EQUALS the trace extent of a SYMBOLIC
                # dim is the signature of a trace-frozen bound — a
                # build-side coverage bug to surface, never an
                # optimization site. Only the to-the-end sentinel, a
                # concrete dim, or a proven-symbolic end qualify.
                start = _arg_scalar(args, 2)
                end = _arg_scalar(args, 3)
                dim = _arg_scalar(args, 1)
                ins = op.get("input_shapes") or []
                tin = self._tensor_inputs(op)
                if (
                    start == 0 and ins and tin
                    and isinstance(dim, int)
                    and -len(ins[0]) <= dim < len(ins[0])
                ):
                    dim_size = ins[0][dim]
                    sdims = self._sym_dims(tin[0])
                    dim_sym = (
                        sdims is not None
                        and -len(sdims) <= dim < len(sdims)
                        and self._dim_is_symbolic(sdims[dim])
                    )
                    end_arg = args[3] if len(args) > 3 else None
                    end_is_symbol = (
                        isinstance(end_arg, dict)
                        and end_arg.get("type") == "symbol"
                    )
                    if isinstance(end, (int, float)) \
                            and end >= _INT64_MAX_SENTINEL:
                        detail = "full-range slice (sentinel end)"
                    elif end_is_symbol and dim_sym:
                        detail = "full-range slice (symbolic end == dim)"
                    elif (
                        isinstance(end, (int, float))
                        and not dim_sym and sdims is not None
                        and end >= dim_size
                    ):
                        detail = "full-range slice (concrete dim)"
                    elif (
                        isinstance(end, (int, float))
                        and dim_sym and end >= dim_size
                    ):
                        self.report.add_suspect(
                            op_uids=[uid],
                            detail=(
                                f"slice end {end} == trace extent of a "
                                f"SYMBOLIC dim — frozen-bound signature "
                                f"(slice-attrs class); fix at the source"
                            ),
                        )
            if detail:
                self.report.add(Finding(
                    category="algebraic",
                    op_uids=[uid],
                    detail=detail,
                    ops_removable=1,
                    symbolic=self._has_symbol(op),
                ))

    def _detect_cancellations(self) -> None:
        """Adjacent exact inverses: add+c→sub+c, mul*v→div/v, neg∘neg."""
        inverse = {
            "aten::sub": "aten::add",
            "aten::div": "aten::mul",
        }
        for uid in self.order:
            op = self.ops[uid]
            t = op["op_type"]
            site = None
            if t in inverse:
                sc = _arg_scalar(op.get("attributes", {}).get("args", []), 1)
                if sc is None or sc in (0, 0.0, 1, 1.0):
                    continue  # identity cases handled elsewhere
                src = self._tensor_inputs(op)
                if not src:
                    continue
                p = self._producer(src[0])
                if p and self.ops[p]["op_type"] == inverse[t]:
                    psc = _arg_scalar(
                        self.ops[p].get("attributes", {}).get("args", []), 1
                    )
                    if psc == sc:
                        site = (p, uid, f"{inverse[t]} {sc} then {t} {sc}")
            elif t in ("aten::neg", "aten::logical_not"):
                src = self._tensor_inputs(op)
                if src:
                    p = self._producer(src[0])
                    if p and self.ops[p]["op_type"] == t:
                        site = (p, uid, f"{t} of {t}")
            if site:
                p, u, detail = site
                cat = (
                    "algebraic"
                    if self._op_dtype_class(op) == "int"
                    else "cancellation"
                )
                # Both ops die only if the intermediate has no other
                # consumer; otherwise only the second op is removable.
                p_outs = self.ops[p].get("output_tensor_ids") or []
                sole = all(
                    self._consumers(o) == [u] for o in p_outs
                )
                self.report.add(Finding(
                    category=cat,
                    op_uids=[p, u],
                    detail=detail
                    + ("" if sole else " (intermediate shared: 1 op)"),
                    ops_removable=2 if sole else 1,
                    symbolic=self._has_symbol(op),
                ))

    def _detect_cse(self) -> dict[str, str]:
        """Report duplicate computations; return a tensor alias map
        (duplicate op output -> representative op output) so later
        detectors can group through would-be-eliminated duplicates."""
        import json as _json

        alias: dict[str, str] = {}
        groups: dict[str, list[str]] = defaultdict(list)
        for uid in self.order:
            op = self.ops[uid]
            t = op["op_type"]
            if t in _NONDETERMINISTIC or t.endswith(_MUTATING_SUFFIX):
                continue
            args = op.get("attributes", {}).get("args", [])
            canon_args = [
                {**a, "tensor_id": alias.get(a["tensor_id"], a["tensor_id"])}
                if a.get("type") == "tensor" else a
                for a in args
            ]
            key = _json.dumps(
                [t, canon_args, op.get("attributes", {}).get("kwargs", {})],
                sort_keys=True, default=str,
            )
            groups[key].append(uid)
            rep_uid = groups[key][0]
            if rep_uid != uid:
                rep_outs = self.ops[rep_uid].get("output_tensor_ids") or []
                for mine, theirs in zip(
                    op.get("output_tensor_ids") or [], rep_outs
                ):
                    alias[mine] = theirs
        for uids in groups.values():
            if len(uids) > 1:
                self.report.add(Finding(
                    category="cse",
                    op_uids=uids,
                    detail=(
                        f"{self.ops[uids[0]]['op_type']} recomputed "
                        f"{len(uids)}x on identical inputs"
                    ),
                    ops_removable=len(uids) - 1,
                    symbolic=self._has_symbol(self.ops[uids[0]]),
                    meta=self.ops[uids[0]]["op_type"] in _FUSION_TRANSPARENT_OPS,
                ))
        return alias

    def _detect_dead_code(self) -> None:
        live_tensors = set(self.g.get("output_tensor_ids") or [])
        live_ops: set[str] = set()
        for uid in reversed(self.order):
            op = self.ops[uid]
            if op["op_type"].endswith(_MUTATING_SUFFIX):
                live_ops.add(uid)  # conservative: mutators stay
                live_tensors.update(self._tensor_inputs(op))
                continue
            if any(o in live_tensors for o in op.get("output_tensor_ids") or []):
                live_ops.add(uid)
                live_tensors.update(self._tensor_inputs(op))
        dead = [u for u in self.order if u not in live_ops]
        if dead:
            self.report.add(Finding(
                category="dead_code",
                op_uids=dead,
                detail=f"{len(dead)} ops unreachable from graph outputs",
                ops_removable=len(dead),
            ))

    def _detect_layout(self) -> None:
        for uid in self.order:
            op = self.ops[uid]
            if op["op_type"] not in ("aten::transpose", "aten::permute"):
                continue
            outs = op.get("output_tensor_ids") or []
            cons = [c for o in outs for c in self._consumers(o)]
            if cons and all(
                self.ops[c]["op_type"] in _ELEMENTWISE for c in cons
            ):
                self.report.add(Finding(
                    category="layout",
                    op_uids=[uid],
                    detail=(
                        f"{op['op_type']} consumed only by elementwise "
                        f"ops (sink candidate)"
                    ),
                    symbolic=self._has_symbol(op),
                ))

    def _detect_fusion_vertical(self) -> None:
        """Anchor + elementwise epilogue; the chain sees THROUGH
        single-consumer metadata ops (they cost no launch and fold into
        the fused kernel's index math)."""
        for uid in self.order:
            op = self.ops[uid]
            if op["op_type"] not in _MATMUL_ANCHORS:
                continue
            chain = [uid]
            epilogue = 0
            cur = op
            while True:
                outs = cur.get("output_tensor_ids") or []
                cons = [c for o in outs for c in self._consumers(o)]
                if len(cons) != 1:
                    break
                nxt = self.ops[cons[0]]
                if nxt["op_type"] in _ELEMENTWISE:
                    epilogue += 1
                elif nxt["op_type"] not in _FUSION_TRANSPARENT_OPS:
                    break
                chain.append(cons[0])
                cur = nxt
            if epilogue >= 1:
                self.report.add(Finding(
                    category="fusion_vertical",
                    op_uids=chain,
                    detail=(
                        f"{op['op_type']} + {epilogue}-op elementwise "
                        f"epilogue ({len(chain) - 1 - epilogue} metadata "
                        f"ops traversed)"
                    ),
                    ops_removable=epilogue,  # launches saved
                ))

    def _detect_fusion_horizontal(self, alias: dict[str, str]) -> None:
        """Same-op same-shape siblings on one source tensor. Sources
        are canonicalized through the CSE alias map AND through
        metadata-op chains, so q/k/v projections reading separate
        (duplicate) views of one hidden state group correctly."""
        def canon_source(tid: str) -> str:
            tid = alias.get(tid, tid)
            seen = 0
            while seen < 8:  # bounded walk through metadata producers
                p = self._producer(tid)
                if not p or self.ops[p]["op_type"] not in _FUSION_TRANSPARENT_OPS:
                    return tid
                pin = self._tensor_inputs(self.ops[p])
                if not pin:
                    return tid
                tid = alias.get(pin[0], pin[0])
                seen += 1
            return tid

        groups: dict[tuple, list[str]] = defaultdict(list)
        for uid in self.order:
            op = self.ops[uid]
            if op["op_type"] not in _MATMUL_ANCHORS:
                continue
            tin = self._tensor_inputs(op)
            outs_t = op.get("output_tensor_ids") or []
            if tin and outs_t:
                # Shape key honest to symbolic extents (concrete trace
                # shapes would over-group at coincident trace values).
                groups[(
                    op["op_type"],
                    canon_source(tin[0]),
                    self._shape_key(outs_t[0]),
                )].append(uid)
        for (op_type, _src, _shape), uids in groups.items():
            if len(uids) > 1:
                self.report.add(Finding(
                    category="fusion_horizontal",
                    op_uids=uids,
                    detail=(
                        f"{len(uids)}x {op_type} same-shape on one "
                        f"source tensor"
                    ),
                    ops_removable=len(uids) - 1,  # launches saved
                ))

    # -- entry ---------------------------------------------------------

    def run(self) -> AnalysisReport:
        self._detect_const_fold()
        self._detect_identities()
        self._detect_cancellations()
        alias = self._detect_cse()
        self._detect_dead_code()
        self._detect_layout()
        self._detect_fusion_vertical()
        self._detect_fusion_horizontal(alias)
        return self.report
