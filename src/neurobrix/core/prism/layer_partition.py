"""Cut a component into segments that each fit a memory budget.

WHY THIS EXISTS

The strategy cascade's last rung, `cpu_streaming`, streams one COMPONENT at
a time. That is the finest grain it has, and it is not fine enough: measured
2026-09-09, `DeepSeek-Coder-V2-Lite-Instruct` is a single `model` component
whose live weights are 17777 MB, and no rung below the component exists. A
model larger than the budget in ONE component has nowhere to go.

This module supplies the missing grain. It answers one question — where can
this graph be cut so that each piece's weights fit — and it answers it from
DATAFLOW.

HOW THE BOUNDARIES ARE FOUND

Not from a list of layer types. Not from `parent_module`, which every graph
carries and which would make this module a catalogue of other people's
naming conventions. The only inputs are:

  * `execution_order`  — the topological order the engine replays
  * each op's `input_tensor_ids` / `output_tensor_ids`
  * `tensors[tid]["is_parameter"]` and its shape/dtype

A cut between two ops costs whatever must survive it: the activations
produced at or before the cut and consumed after it. For a stack of repeated
blocks those minima fall between blocks — but nothing here needs to know
that, and a graph shaped some other way is cut wherever ITS minima are.

WHAT IT REFUSES

A single op whose own weights exceed the budget cannot be served by cutting,
because a cut cannot run half an op. That is a real impossibility and it is
reported with its arithmetic rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

_DTYPE_WIDTH = {
    "float64": 8, "float32": 4, "bfloat16": 2, "float16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
}


def tensor_bytes(tensor: Dict[str, Any]) -> Optional[int]:
    """Bytes for one tensor, or None when the graph does not say.

    A symbolic or negative dimension, or a dtype with no known width, means
    the size is not known. None says so; it never guesses a width.
    """
    n = 1
    for dim in (tensor.get("shape") or []):
        if not isinstance(dim, int) or dim < 0:
            return None
        n *= dim
    width = _DTYPE_WIDTH.get(str(tensor.get("dtype", "")).lower())
    if width is None:
        return None
    return n * width


@dataclass
class Segment:
    """One streamable piece of a component."""
    index: int
    first_op: str
    last_op: str
    op_count: int
    weight_names: Set[str] = field(default_factory=set)
    weight_bytes: int = 0
    live_bytes_at_exit: int = 0

    @property
    def weight_mb(self) -> float:
        return self.weight_bytes / (1024 * 1024)

    @property
    def live_mb_at_exit(self) -> float:
        return self.live_bytes_at_exit / (1024 * 1024)


@dataclass
class Partition:
    """What the whole component costs when streamed this way."""
    segments: List[Segment]
    total_weight_bytes: int
    peak_resident_bytes: int
    peak_live_bytes: int
    refusal: Optional[str] = None

    @property
    def fits(self) -> bool:
        return self.refusal is None

    @property
    def peak_resident_mb(self) -> float:
        return self.peak_resident_bytes / (1024 * 1024)


class LayerPartitioner:
    """Partition one component graph into budget-sized segments."""

    def __init__(self, graph: Dict[str, Any],
                 weight_sizes: Optional[Dict[str, int]] = None):
        self.tensors: Dict[str, Any] = graph.get("tensors") or {}
        self.ops: Dict[str, Any] = graph.get("ops") or {}
        self.order: List[str] = list(graph.get("execution_order") or [])
        # Authoritative sizes when the caller has the weights index; the
        # graph's own shape/dtype otherwise. The index wins because it
        # records what is STORED, which is what a load actually costs.
        self.weight_sizes = weight_sizes or {}

    # -- dataflow ---------------------------------------------------------

    def _last_use(self) -> Dict[str, int]:
        last: Dict[str, int] = {}
        for i, op_uid in enumerate(self.order):
            for tid in (self.ops.get(op_uid) or {}).get("input_tensor_ids") or []:
                last[tid] = i
        return last

    def live_activation_curve(self) -> List[int]:
        """Bytes of activation alive after each op in the order.

        This is the cost of cutting THERE, and its minima are where the
        graph comes apart.
        """
        last = self._last_use()
        curve, live = [], 0
        # Only what has been ADDED can be freed. Subtracting every input at
        # its last use freed the graph's own inputs too — tensors no op
        # produced — and drove the curve negative, reporting a peak of 0 on a
        # graph whose activations were 64 MB. A live set, not a running total.
        alive: Set[str] = set()
        for i, op_uid in enumerate(self.order):
            op = self.ops.get(op_uid) or {}
            for tid in op.get("output_tensor_ids") or []:
                t = self.tensors.get(tid)
                if t is not None and not t.get("is_parameter") and tid not in alive:
                    alive.add(tid)
                    live += tensor_bytes(t) or 0
            for tid in op.get("input_tensor_ids") or []:
                if tid in alive and last.get(tid) == i:
                    t = self.tensors.get(tid)
                    alive.discard(tid)
                    live -= tensor_bytes(t) or 0
            curve.append(live)
        return curve

    def _op_weight_names(self, op_uid: str) -> Set[str]:
        names: Set[str] = set()
        for tid in (self.ops.get(op_uid) or {}).get("input_tensor_ids") or []:
            t = self.tensors.get(tid)
            if t is not None and t.get("is_parameter"):
                names.add(t.get("weight_name") or tid)
        return names

    def _bytes_for(self, name: str, tid_hint: Optional[str] = None) -> int:
        if name in self.weight_sizes:
            return int(self.weight_sizes[name])
        # fall back to the graph's own description of that parameter
        for tid, t in self.tensors.items():
            if t.get("is_parameter") and (t.get("weight_name") or tid) == name:
                return tensor_bytes(t) or 0
        return 0

    # -- the partition ----------------------------------------------------

    def partition(self, budget_bytes: int) -> Partition:
        """Greedy left-to-right: extend while the segment's weights fit.

        Greedy is right here because the order is fixed — the engine replays
        it — so the only freedom is where to cut, and taking as much as fits
        before each cut minimises the number of loads. It is not an
        optimisation problem with a better answer hiding in it.
        """
        curve = self.live_activation_curve()
        peak_live = max(curve) if curve else 0

        # The weights get the budget MINUS what the activations will hold.
        # Sizing segments against the whole budget and then adding the
        # activations on top announced a peak ABOVE the budget — 521.5 MB
        # against 500, 9042.5 against 9000, measured on the first version of
        # this method. The number this returns is the number the strategy
        # promises, so it has to be the one that is actually held.
        weight_budget = budget_bytes - peak_live
        if weight_budget <= 0:
            return Partition(
                segments=[], total_weight_bytes=0, peak_resident_bytes=0,
                peak_live_bytes=peak_live,
                refusal=(
                    f"activations alone peak at "
                    f"{peak_live / (1024*1024):.1f} MB, at or over the "
                    f"{budget_bytes / (1024*1024):.1f} MB budget. Cutting "
                    f"between ops cannot help: no weight has been loaded "
                    f"yet at that peak. This needs a smaller batch or "
                    f"context, which is the caller's to choose."))

        segments: List[Segment] = []
        cur_names: Set[str] = set()
        cur_bytes = 0
        cur_first: Optional[str] = None
        cur_ops = 0
        total = 0

        for i, op_uid in enumerate(self.order):
            names = self._op_weight_names(op_uid)
            new = {n for n in names if n not in cur_names}
            add = sum(self._bytes_for(n) for n in new)

            if add > weight_budget:
                one = sum(self._bytes_for(n) for n in names)
                return Partition(
                    segments=[], total_weight_bytes=0,
                    peak_resident_bytes=0, peak_live_bytes=peak_live,
                    refusal=(
                        f"op {op_uid!r} reads {one / (1024*1024):.1f} MB of "
                        f"weights on its own, over the "
                        f"{weight_budget / (1024*1024):.1f} MB left for "
                        f"weights once activations are reserved "
                        f"({peak_live / (1024*1024):.1f} MB of a "
                        f"{budget_bytes / (1024*1024):.1f} MB budget). "
                        f"A cut cannot "
                        f"run half an op, so no partition of this graph "
                        f"fits. Serving it needs the op's own weights "
                        f"sharded, which is a different rung."))

            if cur_first is not None and cur_bytes + add > weight_budget:
                segments.append(Segment(
                    index=len(segments), first_op=cur_first,
                    last_op=self.order[i - 1], op_count=cur_ops,
                    weight_names=set(cur_names), weight_bytes=cur_bytes,
                    live_bytes_at_exit=curve[i - 1]))
                total += cur_bytes
                cur_names, cur_bytes, cur_first, cur_ops = set(), 0, None, 0
                new, add = names, sum(self._bytes_for(n) for n in names)

            if cur_first is None:
                cur_first = op_uid
            cur_names |= new
            cur_bytes += add
            cur_ops += 1

        if cur_first is not None:
            segments.append(Segment(
                index=len(segments), first_op=cur_first,
                last_op=self.order[-1], op_count=cur_ops,
                weight_names=set(cur_names), weight_bytes=cur_bytes,
                live_bytes_at_exit=curve[-1] if curve else 0))
            total += cur_bytes

        # What is resident at the worst moment: one segment's weights plus
        # the activations alive while it runs. This is the number the
        # strategy ANNOUNCES, and it is the number it holds.
        peak_resident = max(
            (s.weight_bytes for s in segments), default=0) + peak_live

        return Partition(segments=segments, total_weight_bytes=total,
                         peak_resident_bytes=peak_resident,
                         peak_live_bytes=peak_live)


def rewire_arg(arg: Any, rewire: Dict[str, str]) -> Any:
    """One op argument with every tensor id it references mapped through `rewire` — every form
    the engines resolve: `tensor` / `tensor_ref` (`tensor_id`), `tensor_tuple` (`tensor_ids`, the
    list `aten::cat` takes), and nested `list` (`value`); an untyped dict's `tensor_id` too. Returns
    the argument unchanged, or a copy.

    The ONE walk, shared by the triton sequence's in-place rewrites (`TritonSequence._rewire_arg`)
    and a streamed piece's seam aliasing (`build_segment_graph`). The second walked `tensor_id`
    only: a seam inside a `tensor_tuple` kept its raw id, the piece held it under `input::<tid>`,
    and triton-sequential met None — Flex.1-alpha's joint attention `aten.cat::19` concatenated
    512 text queries with nothing, and the next `mul` failed `(1, 24, 512, 128)` against the
    4 608-token RoPE table (the Mac's two Flex rows, df2588e7)."""
    if not isinstance(arg, dict):
        return arg
    arg_type = arg.get("type")
    # A `tensor_id` is rewired whatever the dict's `type` says (`tensor`, `tensor_ref`, or none —
    # the seam builder always rewired untyped ones, and the union of the two walks it replaces
    # is what this one must cover).
    if arg.get("tensor_id") in rewire:
        arg = dict(arg)
        arg["tensor_id"] = rewire[arg["tensor_id"]]
    elif arg_type == "tensor_tuple":
        tids = arg.get("tensor_ids", [])
        new_tids = [rewire.get(t, t) for t in tids]
        if new_tids != tids:
            arg = dict(arg)
            arg["tensor_ids"] = new_tids
    elif arg_type == "list":
        items = arg.get("value", [])
        new_items = [rewire_arg(item, rewire) for item in items]
        if new_items != items:
            arg = dict(arg)
            arg["value"] = new_items
    return arg


def flow_embeds_into(graph: Optional[Dict[str, Any]]) -> bool:
    """Whether the FLOW supplies this component's embeddings — its graph takes `inputs_embeds`,
    the convention both autoregressive flows read (`uses_embeds`) — and so reads the token
    embedding BY NAME from the component's executor, outside the graph. Such a component,
    streamed, keeps its non-block weights resident on its base executor
    (`LayerStreamingStrategy._ensure_flow_reads`) and Prism reserves them; any other component
    (a VAE, a DiT, an encoder fed token ids) is read by no flow and its pieces load exactly what
    they consume."""
    return "input::inputs_embeds" in ((graph or {}).get("input_tensor_ids") or [])


def is_seam_tensor(meta: Optional[Dict[str, Any]]) -> bool:
    """A streamed piece's SEAM input: an intermediate the previous piece produced, aliased to
    `input::<tid>` by `build_segment_graph`. It enters a piece in the dtype its producing op gave it
    — never cast to the compute dtype or re-aligned to the graph's recorded (trace) dtype, as a
    model input or a leaf would be: that narrowed fp32 islands at every piece's entry (PixArt T5
    pieces rel L2 0.46 % from whole, register 105). ONE rule, read by every site that casts a
    component input or re-aligns a leaf, in every engine (R30): TritonDtypeEngine, the torch
    sequential input resolver and leaf re-alignment, and the compiled input map. Pure Python — the
    triton branch may read it (R33)."""
    return bool((meta or {}).get("seam_alias_of"))


def build_segment_graph(graph: Dict[str, Any], segment: Segment,
                        order_index: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """One segment as a standalone, executable graph.

    The point of returning a GRAPH rather than an op range is that the whole
    engine already knows how to run a graph. A segment executed this way goes
    through the same executor, the same binding, the same dispatch as any
    component — the only difference is that it holds one segment's weights.

    Its inputs are the tensors produced BEFORE it and read INSIDE it; its
    outputs are the tensors produced inside it and read AFTER it, plus any of
    the component's own outputs it produces. Those two sets are exactly what
    must cross the seam, and they are computed here rather than assumed.
    """
    tensors: Dict[str, Any] = graph.get("tensors") or {}
    ops: Dict[str, Any] = graph.get("ops") or {}
    order: List[str] = list(graph.get("execution_order") or [])
    if order_index is None:
        order_index = {op_uid: i for i, op_uid in enumerate(order)}

    first = order_index[segment.first_op]
    last = order_index[segment.last_op]
    inside = order[first:last + 1]
    inside_set = set(inside)

    produced_here: Set[str] = set()
    for op_uid in inside:
        produced_here.update((ops.get(op_uid) or {}).get("output_tensor_ids") or [])

    # Inputs: read here, not produced here, not a parameter.
    seg_inputs: List[str] = []
    for op_uid in inside:
        for tid in (ops.get(op_uid) or {}).get("input_tensor_ids") or []:
            t = tensors.get(tid)
            if tid in produced_here or tid in seg_inputs:
                continue
            if t is not None and t.get("is_parameter"):
                continue
            seg_inputs.append(tid)

    # Outputs: produced here and read later, or an output of the component.
    component_outputs = set(graph.get("output_tensor_ids") or [])
    read_later: Set[str] = set()
    for op_uid in order[last + 1:]:
        read_later.update((ops.get(op_uid) or {}).get("input_tensor_ids") or [])
    seg_outputs = [tid for tid in produced_here
                   if tid in read_later or tid in component_outputs]

    keep = set(produced_here)
    for op_uid in inside:
        op = ops.get(op_uid) or {}
        for tid in op.get("input_tensor_ids") or []:
            if (tensors.get(tid) or {}).get("is_parameter"):
                keep.add(tid)
        # A tensor may be referenced ONLY from an op's attributes — a
        # constant an op reads without listing it as an input. Keeping just
        # what `input_tensor_ids` names dropped those, and the segment then
        # lowered differently from the same ops in the whole graph: measured
        # on TinyLlama, `aten.scaled_dot_product_attention::0` refused inside
        # a segment while lowering cleanly outside one.
        attrs = op.get("attributes")
        if isinstance(attrs, dict):
            for arg in (attrs.get("args") or []):
                if isinstance(arg, dict) and arg.get("tensor_id"):
                    keep.add(arg["tensor_id"])
            kw = attrs.get("kwargs")
            if isinstance(kw, dict):
                for arg in kw.values():
                    if isinstance(arg, dict) and arg.get("tensor_id"):
                        keep.add(arg["tensor_id"])

    # A seam tensor has to be BINDABLE. The executor finds its inputs by the
    # `input::` prefix and strips seven characters to get the name it looks up
    # in the caller's dict, so `aten.add::42::out_0` — which is what a seam
    # tensor is called — can never be handed in. Each one is therefore aliased
    # to `input::<tid>`: the prefix makes the executor see it, and stripping
    # the prefix gives back the tid, so the caller passes {tid: value} and
    # nothing has to invent or remember a second name.
    #
    # A tensor the component itself was given already carries the prefix and
    # is left exactly as it is.
    alias_of: Dict[str, str] = {}
    for tid in seg_inputs:
        if tid.startswith("input::"):
            keep.add(tid)
            continue
        alias = "input::" + tid
        src = tensors.get(tid) or {}
        alias_of[tid] = alias
        keep.add(alias)

    seg_tensors: Dict[str, Any] = {
        tid: tensors[tid] for tid in keep if tid in tensors}
    for tid, alias in alias_of.items():
        src = tensors.get(tid) or {}
        seg_tensors[alias] = {
            "tensor_id": alias,
            "shape": src.get("shape"),
            "dtype": src.get("dtype"),
            "device": src.get("device"),
            "producer_op_uid": None,
            "output_index": None,
            "consumer_op_uids": list(src.get("consumer_op_uids") or []),
            "is_parameter": False,
            "is_input": True,
            "weight_name": None,
            "input_name": tid,
            "output_name": None,
            "seam_alias_of": tid,
            # The seam carries the SYMBOLIC shape, not only the concrete one.
            # Without it every dim crossing a segment boundary arrives as a
            # literal, and a later segment has nothing to bind its symbols from
            # — which is not a shape bug that shows up as a wrong number, it is
            # the engine's own refusal:
            #   UnboundSymbolError: symbol 's0' (batch, binds from
            #   input::input_ids::dim_0) is not bound at runtime. Bound: ['s2','s3']
            # `input_ids` is read by the embedding in segment 0 and by nothing
            # after it, so s0 and s1 lost their source at the first boundary
            # while `position_ids` (a graph input throughout) kept s2 and s3.
            # Principle 1 is not suspended at a seam.
            "symbolic_shape": src.get("symbolic_shape"),
        }

    # Ops are shared with the full graph, so the rewrite copies rather than
    # mutating: a partition must not damage the graph it was cut from.
    seg_ops: Dict[str, Any] = {}
    for op_uid in inside:
        op = ops.get(op_uid)
        if op is None:
            continue
        if not alias_of:
            seg_ops[op_uid] = op
            continue
        new_op = dict(op)
        new_op["input_tensor_ids"] = [alias_of.get(t, t)
                                      for t in (op.get("input_tensor_ids") or [])]
        attrs = op.get("attributes")
        if isinstance(attrs, dict):
            new_attrs = dict(attrs)
            if attrs.get("args"):
                new_attrs["args"] = [rewire_arg(arg, alias_of) for arg in attrs["args"]]
            if isinstance(attrs.get("kwargs"), dict):
                new_attrs["kwargs"] = {k: rewire_arg(v, alias_of)
                                       for k, v in attrs["kwargs"].items()}
            # Every OTHER tensor id the op carries in its attributes. A fused op
            # does not take all of its inputs positionally: `custom::moe_fused`
            # names its hidden states, its gate scores and its pre-computed
            # routing by tid in `attributes`, and the runtime resolves those
            # attributes to arena SLOTS — so an id left un-aliased points at a
            # slot the seam never fills.
            #
            # Measured 2026-09-22, DeepSeek-Coder-V2-Lite-Instruct under
            # `layer_streaming`: segment 0 ran and returned its five outputs,
            # segment 1 then raised
            #     RuntimeError: MoE fused: hidden_states is None (slot N).
            #     Killed by liveness analysis before fused op.
            # which is not what happened — nothing killed it. The tensor was
            # present under `input::<tid>` while the attribute still asked for
            # `<tid>`. All four class-1 MoE models failed this way.
            #
            # Data-driven, and it has to be: the rewrite knows only "this value
            # IS a tensor id this seam is aliasing". It names no op, no
            # attribute and no family, so a fused op added later is covered the
            # day it is written. A weight id is never in `alias_of` — a seam
            # carries activations — so expert weight lists pass through
            # untouched.
            for key, val in attrs.items():
                if key in ("args", "kwargs"):
                    continue
                if isinstance(val, str) and val in alias_of:
                    new_attrs[key] = alias_of[val]
                elif isinstance(val, list) and any(
                        isinstance(v, str) and v in alias_of for v in val):
                    new_attrs[key] = [alias_of.get(v, v) if isinstance(v, str)
                                      else v for v in val]
            new_op["attributes"] = new_attrs
        seg_ops[op_uid] = new_op

    out = {k: v for k, v in graph.items()
           if k not in ("tensors", "ops", "execution_order",
                        "input_tensor_ids", "output_tensor_ids")}
    out["tensors"] = seg_tensors
    out["ops"] = seg_ops
    out["execution_order"] = inside
    # What the executor will bind, in its own vocabulary.
    out["input_tensor_ids"] = [alias_of.get(t, t) for t in seg_inputs]
    # What the caller must hand in, keyed as the executor will look it up.
    out["segment_input_names"] = [t[7:] for t in out["input_tensor_ids"]]
    # A seam tensor is an INTERMEDIATE, and must not look like one of the
    # component's own outputs. The executor narrows `output_tensor_ids` to the
    # entries whose `output_name` is a PRIMARY output name, keeping all of
    # them only when none matches — so a single seam tensor that inherited a
    # primary name from the component's metadata silently discarded every
    # other seam tensor. Measured: segment 0 declared 4 outputs and its `run`
    # returned 1, and the next segment then had no inputs.
    #
    # The component's REAL outputs keep their name — in the last segment that
    # is what the caller reads.
    for tid in seg_outputs:
        if tid in component_outputs:
            continue
        t = out["tensors"].get(tid)
        if t is not None and t.get("output_name"):
            t = dict(t)
            # REMOVE the key, do not set it to None. A reader using
            # `.get("output_name", tid)` gets its default only when the key is
            # absent; a present-but-null name returned None and collapsed
            # every seam output onto one key.
            t.pop("output_name", None)
            t["seam_intermediate"] = True
            out["tensors"][tid] = t
    out["output_tensor_ids"] = sorted(seg_outputs)
    out["segment_index"] = segment.index

    # EVERY SYMBOL A SEGMENT USES BINDS WHERE THE WHOLE COMPONENT BINDS IT.
    #
    # A symbol binds from a component INPUT (`input::attention_mask::dim_1`), and a segment's
    # inputs are the tensors crossing its seam. The first repair re-sourced each symbol to a seam
    # dim carrying the SAME symbol id, and kept the original source when none did — which then
    # refused. It could not serve a trace that minted two ids for one extent: PixArt's T5 binds
    # seq_len twice, `s1` from attention_mask and `s3` from input_ids; the hidden states carry
    # `s3`, a later piece uses `s1`, and the Mac's 2048x1024 render died after 1 821 s:
    #     UnboundSymbolError: symbol 's1' (seq_len, binds from input::attention_mask::dim_1)
    #     is not bound at runtime ... Bound: ['s0', 's3']          (the Mac, 8e786e70)
    #
    # So the segment CARRIES the component input its symbols bind from: that input is declared as
    # one of the segment's inputs, the symbol keeps its original source, and the streaming
    # strategy — which holds the component's inputs for the whole run — hands it in. Every piece
    # then binds every symbol from exactly the tensor the whole component binds it from: nothing
    # is equated, nothing re-sourced, a value-sourced symbol (`::val_`) reads the same data. A
    # carried input no op reads is a binding source only, and it is already resident (the strategy
    # holds the component's inputs for the whole run), so it adds a REFERENCE, not bytes. Its entry
    # handling is a model input's, once per piece: an integer carrier on the device (the T5 case) is
    # passed as is; a floating carrier not in the compute dtype is cast per piece, and a `::val_`
    # carrier is read to the host per piece — not measured, and not counted by
    # `live_activation_curve`, which has never counted component inputs. A symbol whose source is
    # NOT a component input keeps it and refuses by name, as before: that dim genuinely does not
    # cross. A source form this function does not know is refused HERE, at partition time.
    sym_ctx = graph.get("symbolic_context")
    if isinstance(sym_ctx, dict) and isinstance(sym_ctx.get("symbols"), dict):
        used: Set[str] = set()

        known = set(sym_ctx["symbols"])

        def _walk(obj):
            # Every form the resolvers accept: `{"type": "symbol", "id"}`, a `symbol_id` key
            # (scaled / derived nodes, shape_resolver + triton/sequence), and a bare string that
            # IS a symbol id (triton/symbols.resolve).
            if isinstance(obj, dict):
                if obj.get("type") == "symbol" and obj.get("id"):
                    used.add(str(obj["id"]))
                if isinstance(obj.get("symbol_id"), str):
                    used.add(obj["symbol_id"])
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, str):
                if obj in known:
                    used.add(obj)
            elif isinstance(obj, list):
                for v in obj:
                    _walk(v)

        _walk(out["tensors"])
        _walk(out["ops"])
        component_inputs = set(graph.get("input_tensor_ids") or [])
        carried: List[str] = []
        for sid, info in sym_ctx["symbols"].items():
            if sid not in used or not isinstance(info, dict):
                continue
            raw = info.get("source")
            if isinstance(raw, dict) and raw.get("tensor_id") is not None:
                src_tid = str(raw["tensor_id"])
            else:
                source = str(raw or "")
                sep = ("::dim_" if "::dim_" in source else "::val_" if "::val_" in source
                       else None)
                if not source.startswith("input::") or sep is None:
                    raise ValueError(
                        f"symbol {sid!r} ({info.get('name')}) used in segment {segment.index} has "
                        f"a source this partitioner cannot carry: {raw!r}. Refused at partition "
                        f"time rather than left to refuse at runtime, pointing at the wrong place.")
                src_tid = source.rsplit(sep, 1)[0]
            if (src_tid in component_inputs and src_tid not in out["input_tensor_ids"]
                    and src_tid not in carried):
                carried.append(src_tid)
        for tid in carried:
            meta = dict(tensors.get(tid) or {})
            meta["symbol_carrier"] = True
            out["tensors"][tid] = meta
            out["input_tensor_ids"].append(tid)
            out["segment_input_names"].append(tid[7:])
        out["symbol_carriers"] = carried
        # The segment's OWN copy: a later pass (promotion) mutates `symbols` in place per segment
        # executor, and a partition must not damage the graph it was cut from.
        out["symbolic_context"] = {**sym_ctx, "symbols": {k: (dict(v) if isinstance(v, dict) else v)
                                                          for k, v in sym_ctx["symbols"].items()}}

    return out


# The seam is bindable: each incoming tensor is aliased to `input::<tid>`, so
# `GraphExecutor._graph_input_tids` sees it and `run` strips the prefix back to
# the tid the caller passes. `segment_input_names` on the returned graph is
# exactly the key set the caller must provide, and outputs come back keyed by
# tensor id (graph_executor.py: `output_name if output_name else tid`), so a
# segment's outputs feed the next segment's inputs with no renaming at all.


