#!/usr/bin/env python3
"""The fusion census of a language container: kernel launches per decoded token today, and after each fused family.

One decoded token costs one run of the decoder's graph. The time of a small model's token is no longer the weight
read alone: it is also the NUMBER of kernels launched between the matrix products. This tool counts them without a
card and without a weight, from the container's own graph, and says what each family of fused layer kernels would
remove.

How the count is made (nothing here knows a model's name):

  * the decoder is the component whose state the container declares as a key/value cache (`topology.json`, `state`,
    kind `kv`); its head is the flow's `head_component` when the container names one;
  * the graph is prepared by the ENGINE'S OWN code, exactly as a `--triton` run prepares it and with the same opt-in
    switches read from the environment: the executor's load passes (attention scale, expert fusion when the
    container's `lm_config` declares experts), then `TritonSequence.compile()` (weight transposes, dead masks,
    SwiGLU, rotary fusion, common subexpressions when planned). Only the constants and the weights are left unread.
    The op list counted is therefore the list the engine runs, at whatever commit this tool is run from;
  * each op of that list is then one of: a KERNEL (the library's dispatch launches a Triton kernel), a COPY the
    graph names (a cast, a strided clone, a concatenation, a gather, a fill: one copy kernel), a DEVICE COPY (a
    clone of contiguous data: a device memory copy, no kernel), or HOST-SIDE (a view: pointer arithmetic, nothing
    launched). Two ops launch more than one kernel for one token and are counted as the library issues them: a
    self-attention under the paged key/value cache (append, length advance, split attention, reduce) and a fused
    expert block (routing top-k, its renormalisation, the three one-token bands);
  * what the count does NOT contain, and a measured profile does: the copies the library issues INSIDE an op's own
    call (a contiguity copy before a kernel, a matrix-vector row copied to its slot), the sampler's kernels, and
    whatever a flow launches around the graph. `--profile` puts a measured table beside the count and names the
    difference kernel by kernel.

The families (each group is a maximal set of adjacent launching ops that could become ONE launch):

  F-E  element-wise chains: add, sub, mul, div, scale by a scalar, negation, activations;
  F-S  the softmax / attention-score chain: the softmax or attention op with the scale, mask and cast ops feeding it;
  F-N  normalisation: the norm kernel (or its decomposed chain) with its scale and its casts;
  F-R  rotary positions on q and k: the rotary op with the position gathers and the layout copies around it.

A group obeys the census rules: every member but the last has ONE reader and that reader is in the group, every
member is on the same device, no output but the last member's is read outside the group, and no shape-changing op
lies between two members except those the family names. Three readings are counted, because the families'
definitions leave three points open:

  strict     only the family's own arithmetic ops are members; any op between two members breaks the group, even a
             host-side view;
  wide       (the default) a view that keeps the element order (view, reshape, unsqueeze, squeeze, flatten) is
             not "between": it launches nothing; a cast or a layout copy inside a chain is a member of the group it
             sits in (a fused kernel reads and stores the dtypes and layout it is given);
  anchored   wide, and a group whose only reader, or only source, is a matrix product folds into that product's
             launch as its prologue or epilogue (the group then costs no launch at all).

    python tools/fusion_census.py CONTAINER [CONTAINER ...] [--json OUT.json] [--markdown OUT.md]
    python tools/fusion_census.py CONTAINER --list              # the launch list of the decode graph, op by op
    python tools/fusion_census.py CONTAINER --profile TABLE.json --tokens N   # against a measured kernel table

A CONTAINER is a `.nbx` file or an unpacked container folder. Exit status: 0 counted, 2 refused (named).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "nbxtensor" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class Refused(RuntimeError):
    """The census cannot count this container; the message says what is missing."""


# ---------------------------------------------------------------------------------------------------------------
# 1. A container's language decoders, read from its own files.

class Container:
    """A `.nbx` file or an unpacked container folder, read member by member (never a weight)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_dir():
            self._zip = None
        elif self.path.is_file():
            self._zip = zipfile.ZipFile(self.path)
        else:
            raise Refused(f"{path}: neither a container file nor a container folder")

    def has(self, member: str) -> bool:
        if self._zip is None:
            return (self.path / member).is_file()
        return member in self._zip.namelist()

    def json(self, member: str) -> Any:
        if not self.has(member):
            raise Refused(f"{self.path}: the container holds no `{member}`")
        if self._zip is None:
            return json.loads((self.path / member).read_bytes())
        return json.loads(self._zip.read(member))

    @property
    def name(self) -> str:
        return self.path.parent.name if self.path.name.endswith(".nbx") and self.path.stem == "model" \
            else self.path.stem if self._zip is not None else self.path.name


def decoders_of(container: Container) -> List[Dict[str, Any]]:
    """The components the container declares a key/value cache for, each with the flow's head when one is named:
    [{"component", "head", "layers_declared"}]. Refused when the container declares none."""
    topology = container.json("topology.json")
    state = topology.get("state") or {}
    generation = (topology.get("flow") or {}).get("generation") or {}
    found = []
    for component, records in state.items():
        for record in records or []:
            if record.get("kind") == "kv":
                found.append({"component": component, "layers_declared": record.get("layers"),
                              "kv_heads": record.get("kv_heads"), "head": None})
    if not found:
        raise Refused(f"{container.path}: `topology.json` declares no key/value cache state: no decoder to count")
    head = generation.get("head_component")
    lm = generation.get("lm_component")
    if head and container.has(f"components/{head}/graph.json"):
        owners = [d for d in found if d["component"] == lm] or (found if len(found) == 1 else [])
        for d in owners:
            d["head"] = head
    return found


# ---------------------------------------------------------------------------------------------------------------
# 2. The op list the Triton engine runs for this graph: the engine's own passes, no device, no weight.

def engine_sequence(dag: Dict[str, Any], family: str, lm_config: Optional[Dict[str, Any]]):
    """`(TritonSequence compiled, dag as the engine rewrote it)` for a component's graph. The executor's load passes
    and the sequence's compile are the engine's; the constants embedded in the graph and the weights are not read
    (nothing here needs a device)."""
    from neurobrix.core.runtime.graph_executor import GraphExecutor
    from neurobrix.triton.sequence import TritonSequence
    from nbxtensor.dtype.nbx_dtype import parse_dtype

    class _GraphOnly(GraphExecutor):
        def _load_constants_from_graph(self):
            return None

        def _load_what_the_rewrite_added(self):
            return None

    dtype = dag.get("torch_dtype")
    if not dtype:
        raise Refused(f"component `{dag.get('component_name')}`: its graph records no `torch_dtype`")
    executor = _GraphOnly(family, vendor="", arch="", device="cpu", dtype=dtype, mode="triton")
    executor.load_graph_from_dict(dag)
    experts = (lm_config or {}).get("num_experts")
    if experts is not None and experts > 1:
        norm_topk = (lm_config or {}).get("norm_topk_prob")
        if norm_topk is None:
            raise Refused(f"component `{dag.get('component_name')}`: `lm_config` declares {experts} experts and no "
                          f"`norm_topk_prob`")
        executor.set_moe_config(norm_topk_prob=norm_topk)
    sequence = TritonSequence(executor._dag, device_idx=0, compute_dtype=parse_dtype(dtype), stores_fp64=False)
    sequence.compile()
    return sequence, executor._dag


# ---------------------------------------------------------------------------------------------------------------
# 3. What one op launches for one decoded token (read from the library's dispatch; see the module's docstring).

#: Views: pointer arithmetic on the host. ORDER-KEEPING ones leave the flat element order as it is.
ORDER_KEEPING_VIEWS = frozenset({"view", "_unsafe_view", "reshape", "flatten", "squeeze", "unsqueeze", "alias",
                                 "detach", "lift_fresh", "contiguous_view"})
LAYOUT_VIEWS = frozenset({"transpose", "permute", "t", "expand", "expand_as", "slice", "narrow", "select", "split",
                          "chunk", "unbind", "movedim", "swapaxes", "diagonal", "unfold", "split_with_sizes",
                          "numpy_T", "mT", "view_as_real", "view_as_complex", "unflatten"})
SHAPE_QUERIES = frozenset({"size", "dim", "numel", "stride", "is_contiguous", "sym_size"})
#: A change of dtype: one copy kernel when the dtype changes, the same tensor otherwise.
CASTS = frozenset({"_to_copy", "to", "type_as", "float", "half", "bfloat16", "int", "long", "double", "bool"})
#: A copy of the data as it is: a strided-copy kernel from a layout view, a device memory copy otherwise.
CLONES = frozenset({"clone", "contiguous", "copy"})
#: One copy / gather / fill kernel each (a `zeros` is a device memory set, no kernel).
COPY_KERNELS = frozenset({"cat", "stack", "repeat", "index", "ones", "full", "scalar_tensor", "arange", "linspace",
                          "ones_like", "full_like", "new_ones", "new_full", "slice_scatter", "select_scatter",
                          "pad", "constant_pad_nd", "tile", "roll", "index_put", "index_put_", "copy_", "fill"})
DEVICE_SETS = frozenset({"zeros", "zeros_like", "new_zeros", "empty", "empty_like", "new_empty"})
#: An inference graph's identity ops.
IDENTITIES = frozenset({"dropout", "native_dropout", "feature_dropout", "alpha_dropout"})
ATTENTION = frozenset({"scaled_dot_product_attention", "_scaled_dot_product_attention",
                       "_scaled_dot_product_efficient_attention", "_scaled_dot_product_flash_attention",
                       "_scaled_dot_product_cudnn_attention", "_scaled_dot_product_flash_attention_for_cpu"})
MATRIX_PRODUCTS = frozenset({"mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "mv", "addmv", "mm_epilogue",
                             "addmm_epilogue", "einsum"})
#: The kernels one self-attention launches for one token under the paged cache, and one fused expert block.
PAGED_REGROUP_COPIES = ("strided_copy (k, one head per group)", "strided_copy (v, one head per group)")
PAGED_DECODE_KERNELS = ("paged_kv_append_step", "paged_lens_advance", "paged_decode_attn_split", "flash_decode_reduce")
EXPERT_BLOCK_KERNELS = ("topk", "div (renormalisation)", "sum (renormalisation)", "moe_gateup", "moe_down_split",
                        "moe_part_reduce")
#: A positional arange the cache offsets at decode: the arange, and the add of the device length.
OFFSET_ARANGE_KERNELS = ("arange", "add (cache length)")


ACTIVATION_DTYPES = ("float32", "float16", "bfloat16")


def bare(op_type: str) -> str:
    """`aten::multiply` -> `mul`: the op's canonical bare name, as the engine's dispatch resolves it."""
    from neurobrix.kernels.classification import canonical_aten
    return canonical_aten(op_type.split("::")[-1])


def _device(recorded: Optional[str]) -> Optional[str]:
    """The device an op was recorded on, `cuda` and `cuda:0` being the same card."""
    if recorded is None or ":" in recorded:
        return recorded
    return f"{recorded}:0"


class Op:
    """One op of the engine's list, with what it launches for one decoded token."""
    __slots__ = ("index", "uid", "type", "name", "inputs", "outputs", "device", "kind", "launches", "kernels",
                 "module", "record", "plain_copy", "inner_copies")

    def __init__(self, index, uid, op_type, inputs, outputs, record):
        self.index, self.uid, self.type, self.name = index, uid, op_type, bare(op_type)
        self.inputs, self.outputs, self.record = tuple(inputs), tuple(outputs), record
        self.device = _device(record.get("device"))
        self.module = record.get("parent_module") or ""
        self.kind, self.launches, self.kernels = "host", 0, ()
        self.plain_copy = False     # a launch that only moves one value: a cast, a strided clone, a one-operand cat
        self.inner_copies = 0       # of its launches, the copies its own call issues before its kernels


class DecodeGraph:
    """The engine's op list of one component, as a dataflow over slots: who writes a slot, who reads it."""

    def __init__(self, sequence, dag, self_attention: Iterable[str] = (), offset_aranges: Iterable[str] = (),
                 offset_slices: Iterable[str] = (), offset_counts: Iterable[str] = (), keys_expanded: bool = False,
                 activation_dtype: Optional[str] = None):
        self.dag = dag
        self.activation_dtype = activation_dtype
        self.tensors = dag.get("tensors") or {}
        self.dtype_name = str(dag.get("torch_dtype") or "").replace("torch.", "")
        self.keys_expanded = bool(keys_expanded)
        records = dag.get("ops") or {}
        self.slot_tid: Dict[int, str] = {}
        for tid, slot in sequence._tid_to_slot.items():
            self.slot_tid.setdefault(slot, tid)
        self.ops: List[Op] = []
        self.writer: Dict[int, int] = {}
        self.readers: Dict[int, List[int]] = collections.defaultdict(list)
        for i, compiled in enumerate(sequence._ops):
            op = Op(i, compiled.op_uid, compiled.op_type, compiled.all_input_slots, compiled.output_slots,
                    records.get(compiled.op_uid) or {})
            # where the op's value lives is the record of its OUTPUT (an op with no tensor input, a fill, carries
            # no device of its own in its record)
            written = (self.tensors.get(self.slot_tid.get(op.outputs[0], "")) or {}).get("device") \
                if op.outputs else None
            op.device = _device(written or op.record.get("device"))
            self.ops.append(op)
            for slot in dict.fromkeys(op.inputs):
                self.readers[slot].append(i)
            for slot in op.outputs:
                self.writer[slot] = i
        self.run_dtypes: Optional[Dict[int, str]] = self._run_dtypes() if activation_dtype else None
        self.returned = {sequence._tid_to_slot[t] for t in sequence._output_ids if t in sequence._tid_to_slot}
        self.self_attention = frozenset(self_attention)
        self._offset = {**{u: "arange" for u in offset_aranges}, **{u: "slice" for u in offset_slices},
                        **{u: "count" for u in offset_counts}}
        for op in self.ops:
            op.kind, op.kernels = self._launch(op)
            op.launches = len(op.kernels)
            op.plain_copy = bool(op.launches) and (op.name in CASTS or op.name in CLONES or self._joins_one_value(op))
            op.inner_copies = sum(1 for kernel in op.kernels if kernel in PAGED_REGROUP_COPIES)

    # -- reading a slot's recorded facts ------------------------------------------------------------------------
    def dtype(self, slot: int) -> Optional[str]:
        return (self.tensors.get(self.slot_tid.get(slot, "")) or {}).get("dtype")

    def run_dtype(self, slot: int) -> Optional[str]:
        """The dtype the value at `slot` holds at run time: the activation dtype's propagation when the census was
        given one, else the graph's recorded dtype."""
        if self.run_dtypes is None:
            return self.dtype(slot)
        return self.run_dtypes.get(slot, self.dtype(slot))

    def _run_dtypes(self) -> Dict[int, str]:
        """Each slot's dtype at run time under the activation dtype, by the engine's precision rules
        (`nbxtensor.dtype.policy` op classes, `amp_fp32_output_dtype`): a half dtype recorded is held in the
        activation dtype; under a half activation dtype an op of the float32 class, and the matrix products that
        need float32 (mm, bmm, addmm, div), write float32, an op of the lower-precision class writes the activation
        dtype, a cast its target (a half target in the activation dtype);
        any other op the widest of its floating inputs (type promotion)."""
        from nbxtensor.dtype.outputs import amp_fp32_output_dtype
        from nbxtensor.dtype.policy import default_classes
        classes = default_classes()
        compute, graph = self.activation_dtype, self.dtype_name
        halves = ("float16", "bfloat16")
        width = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}
        half = compute in halves

        def held(name):
            return compute if name in halves else name
        out: Dict[int, str] = {}
        for slot, tid in self.slot_tid.items():
            recorded = (self.tensors.get(tid) or {}).get("dtype")
            if recorded in width and (self.is_weight(slot) or slot not in self.writer):
                out[slot] = compute if recorded != "float64" else recorded
            elif recorded:
                out[slot] = held(recorded)
        for op in self.ops:
            for slot in op.outputs:
                recorded = self.dtype(slot)
                if recorded not in width:
                    continue
                floats = [out[s] for s in op.inputs if out.get(s) in width]
                first = floats[0] if floats else held(recorded)
                if op.name in CASTS:
                    out[slot] = held(recorded)
                elif half and op.name in classes["float32"]:
                    out[slot] = amp_fp32_output_dtype(compute, graph, False, False)
                elif half and op.name in ("mm", "bmm", "addmm", "div") and compute == "float16":
                    out[slot] = "float32"
                elif half and op.name in classes["lower"]:
                    out[slot] = compute
                else:
                    # the widest floating input (type promotion), the first one's when there is no other
                    out[slot] = max(floats, key=width.get) if floats else first
        return out

    def is_weight(self, slot: int) -> bool:
        meta = self.tensors.get(self.slot_tid.get(slot, "")) or {}
        return bool(meta.get("is_parameter")) or slot not in self.writer and not meta.get("is_input")

    def _dim_at_decode(self, dim) -> Optional[int]:
        """A recorded dim when ONE sequence decodes ONE position: every symbol of the graph (batch, positions) is 1."""
        if isinstance(dim, int):
            return dim
        if not isinstance(dim, dict):
            return None
        kind = dim.get("type")
        if kind == "symbol":
            return 1
        left, right = self._dim_at_decode(dim.get("left")), self._dim_at_decode(dim.get("right"))
        if left is None or right is None:
            return None
        if kind == "mul":
            return left * right
        if kind == "add":
            return left + right
        if kind == "sub":
            return left - right
        if kind in ("floordiv", "div") and right:
            return left // right
        return None

    def shape_at_decode(self, slot: int) -> Optional[Tuple[Optional[int], ...]]:
        """The tensor's shape when one sequence decodes one position (its recorded symbolic shape, evaluated)."""
        meta = self.tensors.get(self.slot_tid.get(slot, "")) or {}
        dims = (meta.get("symbolic_shape") or {}).get("dims")
        if dims is None:
            return None
        return tuple(self._dim_at_decode(d) for d in dims)

    def _keeps_order(self, op: Op) -> bool:
        """Whether a layout view leaves the elements in the order they are stored, at a decode step: a transpose
        or permute that only moves axes of one element, an expand that repeats nothing, a slice of leading rows."""
        if not op.inputs or not op.outputs:
            return False
        before, after = self.shape_at_decode(op.inputs[0]), self.shape_at_decode(op.outputs[0])
        if before is None or after is None or None in before or None in after:
            return False
        scalars = [a.get("value") for a in (op.record.get("attributes") or {}).get("args") or []
                   if isinstance(a, dict) and a.get("type") == "scalar"]
        if op.name in ("transpose", "swapaxes", "t"):
            if op.name == "t":
                scalars = [0, 1]
            if len(scalars) < 2 or not all(isinstance(v, int) for v in scalars[:2]):
                return False
            a, b = sorted(v % len(before) for v in scalars[:2])
            return sum(1 for d in before[a:b + 1] if d != 1) <= 1
        if op.name == "permute":
            order = next((a.get("value") for a in (op.record.get("attributes") or {}).get("args") or []
                          if isinstance(a, dict) and a.get("type") == "list"), None)
            if not order or not all(isinstance(v, int) for v in order):
                return False
            kept = [v % len(before) for v in order if before[v % len(before)] != 1]
            return kept == sorted(kept)
        if op.name in ("expand", "expand_as"):
            return tuple(before) == tuple(after)[-len(before):] and all(d == 1 for d in after[:len(after) - len(before)])
        if op.name in ("slice", "narrow"):
            return tuple(before) == tuple(after) or (bool(scalars) and scalars[0] == 0 and (
                len(scalars) < 4 or scalars[3] in (1, None)))
        return False

    def _from_layout_view(self, slot: int) -> bool:
        """Whether the data read at `slot` comes through a view that reorders, repeats or skips stored elements
        at a decode step (its copy is then a strided-copy kernel, not a copy of contiguous memory)."""
        seen = 0
        while slot in self.writer and seen < 64:
            op = self.ops[self.writer[slot]]
            if op.name in LAYOUT_VIEWS:
                if not self._keeps_order(op):
                    return True
            elif op.name not in ORDER_KEEPING_VIEWS:
                return False
            if not op.inputs:
                return False
            slot, seen = op.inputs[0], seen + 1
        return False

    def _joins_one_value(self, op: Op) -> bool:
        """A concatenation whose operands but one are EMPTY (a cache's empty past joined to the step's keys): a
        copy of that one operand."""
        if op.name not in ("cat", "concat", "concatenate"):
            return False
        filled = 0
        for slot in dict.fromkeys(op.inputs):
            shape = self.shape_at_decode(slot)
            if shape is None or None in shape:
                return False
            filled += 0 not in shape
        return filled == 1

    def _reads_cached_attention(self, slot: int) -> bool:
        """Whether the value at `slot` is a cached self-attention's output, seen through views."""
        seen = 0
        while slot in self.writer and seen < 64:
            op = self.ops[self.writer[slot]]
            if op.uid in self.self_attention:
                return True
            if op.launches or not op.inputs or (op.name not in ORDER_KEEPING_VIEWS and op.name not in LAYOUT_VIEWS):
                return False
            slot, seen = op.inputs[0], seen + 1
        return False

    def _launch(self, op: Op) -> Tuple[str, Tuple[str, ...]]:
        from neurobrix.kernels.classification import OpExecution, get_execution_type
        name = op.name
        if op.uid in self.self_attention:
            # keys and values the graph repeated to the query heads are read back one head per group: a strided
            # view the cache copies before its append kernel
            return "kernel", (PAGED_REGROUP_COPIES if self.keys_expanded else ()) + PAGED_DECODE_KERNELS
        if op.uid in self._offset:
            how = self._offset[op.uid]
            if how == "arange":
                return "copy", OFFSET_ARANGE_KERNELS
            if how == "slice":
                return "copy", OFFSET_ARANGE_KERNELS + ("index_select",)
            return "kernel", ("cumsum", "add (cache length)")
        if op.type == "custom::moe_fused":
            routed_in_graph = (op.record.get("attributes") or {}).get("topk_indices_tid") is not None
            return "kernel", EXPERT_BLOCK_KERNELS[3:] if routed_in_graph else EXPERT_BLOCK_KERNELS
        if name in ORDER_KEEPING_VIEWS or name in LAYOUT_VIEWS or name in SHAPE_QUERIES or name in IDENTITIES:
            return "host", ()
        if name in CASTS:
            source = op.inputs[0] if op.inputs else None
            target = self.dtype(op.outputs[0]) if op.outputs else None
            if source is not None and self.run_dtype(source) == self.run_dtype(op.outputs[0]):
                return "host", ()
            if (source is not None and self.run_dtypes is None and target == self.dtype_name
                    and self._reads_cached_attention(source)):
                # the paged attention writes its output in the cache's dtype, the graph's: nothing left to cast
                return "host", ()
            if source is not None and self._from_layout_view(source):
                return "copy", (f"strided_copy (cast {self.run_dtype(source)} to {self.run_dtype(op.outputs[0])})",)
            return "copy", (f"copy ({self.run_dtype(source)} to {self.run_dtype(op.outputs[0])})",)
        if name in CLONES:
            if op.inputs and self._from_layout_view(op.inputs[0]):
                return "copy", ("strided_copy",)
            return ("host", ()) if name == "contiguous" else ("device_copy", ())
        if name in COPY_KERNELS:
            return "copy", ("cat (one operand and an empty one)" if self._joins_one_value(op) else name,)
        if name in DEVICE_SETS:
            return "device_copy", ()
        if op.type.startswith("custom::"):
            return "kernel", (name,)
        try:
            execution = get_execution_type(op.type)
        except KeyError:
            raise Refused(f"op `{op.uid}` ({op.type}): the engine's classification does not know this op, so the "
                          f"census cannot say whether it launches a kernel; add it to the census's tables") from None
        if execution is OpExecution.TRITON:
            return "kernel", (name,)
        raise Refused(f"op `{op.uid}` ({op.type}): a host-side op the census's tables do not name; say what it "
                      f"launches in tools/fusion_census.py")

    # -- the dataflow between LAUNCHING ops, through the views between them ---------------------------------------
    def launching(self) -> List[Op]:
        return [op for op in self.ops if op.launches]

    def sinks(self, op: Op, through: frozenset) -> Tuple[List[int], bool]:
        """`(ops that read op's outputs, clean)`, every output taken together (see `sinks_of_slot`)."""
        found: List[int] = []
        clean = True
        for slot in op.outputs:
            readers, slot_clean = self.sinks_of_slot(slot, through)
            found.extend(readers)
            clean = clean and slot_clean
        return list(dict.fromkeys(found)), clean

    def sinks_of_slot(self, slot: int, through: frozenset) -> Tuple[List[int], bool]:
        """`(ops that read the value, clean)`: readers found THROUGH the host-side views named in `through` (each
        followed to its own readers). `clean` is False when the value is returned by the graph, or is read by a
        host-side op that is not in `through` (it then reaches something this walk does not follow)."""
        found: List[int] = []
        clean = True
        stack = [slot]
        seen = set()
        while stack:
            slot = stack.pop()
            if slot in seen:
                continue
            seen.add(slot)
            if slot in self.returned:
                clean = False
            for r in self.readers.get(slot, ()):
                reader = self.ops[r]
                if reader.launches:
                    found.append(r)
                elif reader.name in through or (through and reader.kind == "device_copy"):
                    stack.extend(reader.outputs)          # a view, or a copy of contiguous memory: the same values
                elif reader.kind == "device_copy":
                    found.append(r)
                else:
                    clean = False
        return list(dict.fromkeys(found)), clean

    def sources(self, op: Op) -> List[int]:
        """The launching ops whose values `op` reads, through any host-side op."""
        found: List[int] = []
        stack = list(op.inputs)
        seen = set()
        while stack:
            slot = stack.pop()
            if slot in seen or slot not in self.writer:
                continue
            seen.add(slot)
            writer = self.ops[self.writer[slot]]
            if writer.launches:
                found.append(writer.index)
            else:
                stack.extend(writer.inputs)
        return list(dict.fromkeys(found))


# ---------------------------------------------------------------------------------------------------------------
# 4. The families.

ACTIVATIONS = frozenset({"relu", "silu", "gelu", "sigmoid", "tanh", "hardsigmoid", "hardswish", "leaky_relu", "elu",
                         "mish", "selu", "softplus", "celu", "log_sigmoid", "swiglu_fused", "glu", "erf",
                         "clamp", "clamp_min", "clamp_max", "threshold"})
ARITHMETIC = frozenset({"add", "sub", "rsub", "mul", "div", "neg"})
NORM_KERNELS = frozenset({"rms_norm", "native_layer_norm", "layer_norm", "native_group_norm", "group_norm",
                          "batch_norm"})
#: The ops of a norm the builder left decomposed (x * rsqrt(mean(x^2) + eps) * w and its LayerNorm sibling).
NORM_CHAIN = frozenset({"rsqrt", "var", "std", "linalg_vector_norm"})
NORM_CHAIN_COMPANIONS = frozenset({"pow", "square", "mean", "sqrt", "mul", "add", "div", "sub"})
SOFTMAXES = frozenset({"softmax", "_softmax", "_safe_softmax", "log_softmax", "_log_softmax"})
MASKS = frozenset({"masked_fill", "where", "tril", "triu", "logical_not", "ones", "full", "scalar_tensor",
                   "eq", "ne", "lt", "le", "gt", "ge", "bitwise_not", "logical_and", "logical_or"})
ROTARY = frozenset({"rope_fused", "rope"})
#: A rotation the engine's rotary pass left decomposed: the negated half (`neg`) joined back (`stack`, `cat`), the
#: cos and sin products and their sum. The negation is the mark: no other op of a decoder layer negates.
ROTATE_HALF = frozenset({"neg"})
ROTARY_CHAIN = frozenset({"cat", "stack", "mul", "add"})


class Family:
    """A family of fused kernels: the ops a group must hold one of (`anchors`), the ops that may join one
    (`companions`, and `wide_companions` under the wide reading), the layout views a group may cross besides the
    order-keeping ones (`views`), and how many values a group hands outside (`outputs`: one, or the rotary's two,
    q and k)."""

    def __init__(self, key: str, title: str, anchors: frozenset, companions: frozenset, wide_companions: frozenset,
                 views: frozenset = frozenset(), outputs: int = 1):
        self.key, self.title, self.outputs = key, title, outputs
        self.anchors, self.companions, self.wide_companions, self.views = anchors, companions, wide_companions, views

    def member(self, op: Op, wide: bool) -> bool:
        if op.name in self.anchors or op.name in self.companions:
            return True
        return wide and (op.name in self.wide_companions or op.plain_copy)

    def anchor(self, op: Op) -> bool:
        return op.name in self.anchors


FAMILIES: Tuple[Family, ...] = (
    Family("F-N", "normalisation with its scale and casts",
           anchors=NORM_KERNELS | NORM_CHAIN, companions=NORM_CHAIN_COMPANIONS,
           wide_companions=frozenset()),
    Family("F-R", "rotary positions on q and k",
           anchors=ROTARY | ROTATE_HALF, companions=ROTARY_CHAIN,
           wide_companions=frozenset({"index", "index_select"}),
           views=frozenset({"transpose", "permute", "expand", "slice"}), outputs=2),
    Family("F-S", "the softmax / attention-score chain",
           anchors=SOFTMAXES | ATTENTION, companions=frozenset({"mul", "div", "add"}) | MASKS,
           wide_companions=frozenset(), views=frozenset({"transpose", "expand"})),
    Family("F-E", "element-wise chains",
           anchors=ARITHMETIC | ACTIVATIONS, companions=frozenset(), wide_companions=frozenset()),
)
FAMILY = {f.key: f for f in FAMILIES}
READINGS = ("strict", "wide", "anchored")


class Group:
    __slots__ = ("family", "members", "last", "folds", "wide")

    def __init__(self, family: str, members: List[Op], last: Op, folds: bool, wide: bool):
        self.family, self.members, self.last, self.folds, self.wide = family, members, last, folds, wide

    @property
    def before(self) -> int:
        return sum(op.launches for op in self.members)

    @property
    def after(self) -> int:
        """The launches the fused group still costs: the most any ONE member launches (a multi-kernel member such
        as the paged attention keeps its own kernels; under the wide readings, not the copies its call issues
        before them), none when it folds into a matrix product."""
        if self.folds:
            return 0
        return max(op.launches - (op.inner_copies if self.wide else 0) for op in self.members)

    @property
    def pattern(self) -> str:
        return " > ".join(op.name for op in self.members)


def groups_of(graph: DecodeGraph, family: Family, reading: str, taken: frozenset = frozenset()) -> List[Group]:
    """The maximal groups `family` could fuse in `graph` under `reading`, ops in `taken` left out (they already
    belong to another family's group)."""
    wide = reading != "strict"
    through = (ORDER_KEEPING_VIEWS | family.views) if wide else frozenset()
    members = {op.index for op in graph.launching() if op.index not in taken and family.member(op, wide)}
    parent = {i: i for i in members}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in sorted(members):
        op = graph.ops[i]
        # every value the op writes has ONE reader, in the family, on the same device, and nothing else sees it;
        # an op writing several values may feed as many readers as the family hands values outside
        targets = []
        for slot in op.outputs:
            readers, clean = graph.sinks_of_slot(slot, through)
            if not readers and clean:
                continue                                  # an output nothing reads (a norm's unused statistics)
            if not clean or len(readers) != 1 or readers[0] not in members \
                    or graph.ops[readers[0]].device != op.device:
                targets = []
                break
            targets.append(readers[0])
        if targets and len(set(targets)) <= family.outputs:
            for reader in targets:
                parent[find(i)] = find(reader)
    by_root: Dict[int, List[int]] = collections.defaultdict(list)
    for i in members:
        by_root[find(i)].append(i)
    out: List[Group] = []
    for root, idxs in by_root.items():
        ops = [graph.ops[i] for i in sorted(idxs)]
        if not any(family.anchor(op) for op in ops):
            continue
        last = graph.ops[max(idxs)]
        folds = False
        if reading == "anchored" and not any(op.launches > 1 for op in ops):
            folds = _feeds_only_products(graph, last) or _ends_a_product(graph, ops, set(idxs))
        if len(ops) < 2 and not folds:
            continue
        out.append(Group(family.key, ops, last, folds, wide))
    return sorted(out, key=lambda g: g.members[0].index)


def _feeds_only_products(graph: DecodeGraph, last: Op) -> bool:
    """The group's value is read by matrix products and nothing else: it can be formed on their input read."""
    readers, clean = graph.sinks(last, ORDER_KEEPING_VIEWS)
    return bool(clean and readers and all(graph.ops[r].name in MATRIX_PRODUCTS for r in readers))


def _ends_a_product(graph: DecodeGraph, ops: List[Op], inside: set) -> bool:
    """A member reads a matrix product that nothing else reads: the group can be that product's epilogue."""
    for op in ops:
        for s in graph.sources(op):
            source = graph.ops[s]
            if s in inside or source.name not in MATRIX_PRODUCTS:
                continue
            readers, clean = graph.sinks(source, ORDER_KEEPING_VIEWS)
            if clean and readers == [op.index]:
                return True
    return False


def fuse_all(graph: DecodeGraph, reading: str) -> List[Group]:
    """Every family applied in turn, the specific ones first; an op belongs to one group."""
    taken: set = set()
    out: List[Group] = []
    for family in FAMILIES:
        for group in groups_of(graph, family, reading, frozenset(taken)):
            out.append(group)
            taken.update(op.index for op in group.members)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 5. The census of one component, and of one container.

def layer_split(graph: DecodeGraph) -> Dict[str, Any]:
    """Launches per layer: a layer is the run of ops from one self-attention to the next (the same ops as from one
    layer's start to the next one's, rotated). `per_layer` is the median run; `outside` is what the first and last
    partial runs hold beyond one layer (embedding, final norm)."""
    marks = [op.index for op in graph.ops if op.uid in graph.self_attention]
    total = sum(op.launches for op in graph.ops)
    if len(marks) < 2:
        return {"layers": len(marks), "per_layer": None, "per_layer_min": None, "per_layer_max": None,
                "outside": None, "uniform": None}
    runs = [sum(op.launches for op in graph.ops[a:b]) for a, b in zip(marks, marks[1:])]
    ordered = sorted(runs)
    median = ordered[len(ordered) // 2]
    return {"layers": len(marks), "per_layer": median, "per_layer_min": ordered[0], "per_layer_max": ordered[-1],
            "outside": total - sum(runs) - median, "uniform": ordered[0] == ordered[-1]}


def census_of_graph(graph: DecodeGraph) -> Dict[str, Any]:
    launching = graph.launching()
    today = sum(op.launches for op in launching)
    by_kernel: collections.Counter = collections.Counter()
    by_kind: collections.Counter = collections.Counter()
    for op in graph.ops:
        by_kind[op.kind] += max(op.launches, 1) if op.kind != "host" else 1
        for kernel in op.kernels:
            by_kernel[kernel] += 1
    out: Dict[str, Any] = {
        "ops": len(graph.ops), "launches": today, **layer_split(graph),
        "kernel_launches": sum(op.launches for op in launching if op.kind == "kernel"),
        "copy_launches": sum(op.launches for op in launching if op.kind == "copy"),
        "device_copies": by_kind.get("device_copy", 0), "host_side_ops": by_kind.get("host", 0),
        "by_kernel": dict(by_kernel.most_common()), "readings": {},
    }
    for reading in READINGS:
        row: Dict[str, Any] = {"after": {}, "groups": {}, "patterns": {}}
        for family in FAMILIES:
            groups = groups_of(graph, family, reading)
            row["after"][family.key] = today - sum(g.before - g.after for g in groups)
            row["groups"][family.key] = len(groups)
            row["patterns"][family.key] = dict(collections.Counter(g.pattern for g in groups).most_common())
        fused = fuse_all(graph, reading)
        row["after"]["all"] = today - sum(g.before - g.after for g in fused)
        in_a_group = {op.index for g in fused for op in g.members}
        left = collections.Counter()
        for op in launching:
            if op.index not in in_a_group:
                left[op.kernels[0] if op.launches == 1 else op.name] += op.launches
        row["left"] = dict(left.most_common())
        row["unfused_chains"] = unfused_chains(graph, in_a_group)
        out["readings"][reading] = row
    return out


def unfused_chains(graph: DecodeGraph, in_a_group: set) -> Dict[str, int]:
    """The op patterns no family took: maximal single-reader chains of the launching ops left outside every group
    (a matrix product may sit in such a chain: it is the next family to consider, not a member of these four)."""
    left = {op.index for op in graph.launching() if op.index not in in_a_group}
    nxt: Dict[int, int] = {}
    for i in left:
        readers, clean = graph.sinks(graph.ops[i], ORDER_KEEPING_VIEWS)
        if clean and len(readers) == 1 and readers[0] in left:
            nxt[i] = readers[0]
    heads = left - set(nxt.values())
    patterns: collections.Counter = collections.Counter()
    fan_in = collections.Counter(nxt.values())
    for head in heads:
        chain, i = [head], head
        while i in nxt and fan_in[nxt[i]] == 1:
            i = nxt[i]
            chain.append(i)
        if len(chain) >= 2:
            patterns[" > ".join(graph.ops[j].name for j in chain)] += 1
    return dict(patterns.most_common())


def decode_graph_of(container: Container, component: str, family: str, lm_config: Optional[Dict[str, Any]],
                    decoder: Optional[Dict[str, Any]], activation_dtype: Optional[str] = None) -> DecodeGraph:
    """The launch graph of a component: a cached decoder's (`decoder`: its entry of `decoders_of`), or a plain
    component's such as the head (`decoder` None)."""
    dag = container.json(f"components/{component}/graph.json")
    sequence, dag = engine_sequence(dag, family, lm_config)
    if decoder is None:
        return DecodeGraph(sequence, dag, activation_dtype=activation_dtype)
    from neurobrix.core.flow.decoder_kv import decoder_self_attention_plan
    plan = decoder_self_attention_plan(dag)
    if plan is None:
        raise Refused(f"{container.path}: component `{component}` is declared a cached decoder and its graph holds "
                      f"no self-attention op")
    kv_heads = decoder.get("kv_heads")
    if kv_heads is None:
        raise Refused(f"{container.path}: the key/value state of `{component}` declares no `kv_heads`")
    return DecodeGraph(sequence, dag, plan["self_attn_uids"], plan["arange_uids"], plan["position_slice_uids"],
                       plan["position_cumsum_uids"], keys_expanded=int(plan["kv_heads_at_op"]) > int(kv_heads),
                       activation_dtype=activation_dtype)


def lm_config_for(container: Container, component: str) -> Dict[str, Any]:
    """The decoder's `lm_config` as the engine's flows read it (`lm_facts.lm_config_of`: the container's defaults,
    else the component's recorded values, else its declared key/value state)."""
    from neurobrix.core.runtime.lm_facts import lm_config_of
    defaults = container.json("runtime/defaults.json") if container.has("runtime/defaults.json") else None
    return lm_config_of(defaults, container.json("topology.json"), component)


def census_of(path: Path, activation_dtype: Optional[str] = None) -> Dict[str, Any]:
    """The census of one container: one entry per cached decoder, its head's launches added to the token."""
    container = Container(path)
    family = container.json("manifest.json").get("family")
    if not family:
        raise Refused(f"{path}: `manifest.json` names no family")
    out = {"container": container.name, "path": str(path), "family": family, "decoders": []}
    for decoder in decoders_of(container):
        component = decoder["component"]
        graph = decode_graph_of(container, component, family, lm_config_for(container, component), decoder,
                                 activation_dtype)
        row = census_of_graph(graph)
        row["component"], row["layers_declared"] = component, decoder["layers_declared"]
        row["expert_blocks"] = sum(1 for op in graph.ops if op.type == "custom::moe_fused")
        row["one_row_products"] = sum(1 for op in graph.ops if op.name in ("mm", "addmm"))
        row["head"], row["head_launches"], row["head_one_row_products"] = decoder["head"], 0, 0
        if decoder["head"]:
            head = decode_graph_of(container, decoder["head"], family, None, None, activation_dtype)
            row["head_launches"] = sum(op.launches for op in head.ops)
            row["head_one_row_products"] = sum(1 for op in head.ops if op.name in ("mm", "addmm"))
        out["decoders"].append(row)
    return out


# ---------------------------------------------------------------------------------------------------------------
# 6. Output.

def launch_list(graph: DecodeGraph, groups: Sequence[Group]) -> str:
    """The decode graph's launching ops in order, each with its kernels and the group that would take it."""
    of_group = {op.index: f"{g.family}#{n}" for n, g in enumerate(groups) for op in g.members}
    lines = []
    for op in graph.ops:
        if not op.launches and op.kind != "device_copy":
            continue
        what = ", ".join(op.kernels) if op.kernels else "device copy"
        lines.append(f"{op.index:6d}  {op.uid:<44} {op.launches}  {what:<60} {of_group.get(op.index, '')}")
    return "\n".join(lines)


def _top(counter: Dict[str, int], n: int) -> str:
    return "; ".join(f"{k} x{v}" for k, v in list(counter.items())[:n]) or "-"


def table(rows: List[Dict[str, Any]], reading: str) -> str:
    head = ("| container | decoder | layers | per layer | launches per token today | after F-E | after F-S | after F-N "
            "| after F-R | after all | left unfused (launches) | chains no family takes |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for entry in rows:
        if "refused" in entry:
            lines.append(f"| {entry['container']} | - | - | - | refused: {entry['refused']} | | | | | | | |")
            continue
        for d in entry["decoders"]:
            r = d["readings"][reading]
            extra = d["head_launches"]
            lines.append(
                f"| {entry['container']} | {d['component']} | {d['layers']} | {d['per_layer']} | "
                f"{d['launches'] + extra} | " + " | ".join(str(r["after"][k] + extra) for k in
                                                           ("F-E", "F-S", "F-N", "F-R", "all"))
                + f" | {_top(r['left'], 6)} | {_top(r['unfused_chains'], 4)} |")
    return "\n".join(lines)


#: The library kernel a counted launch shows up as in a measured table of a one-token step, by a stem of its name.
MEASURED_AS = (
    ("mm", "gemv_vec"), ("addmm", "addmv"), ("bmm", "baddbmm"), ("copy (", "triton_copy_kernel"),
    ("strided_copy", "strided_copy_kernel"), ("index", "index_select"), ("rms_norm", "rms_norm"),
    ("ones", "fill_kernel"), ("full", "fill_kernel"), ("scalar_tensor", "fill_kernel"), ("tril", "tril"),
    ("where", "where_forward"), ("mul", "mul_scalar_kernel"), ("add", "add_forward"), ("rope_fused", "rope_forward"),
    ("swiglu_fused", "silu_mul"), ("paged_kv_append_step", "paged_kv_append_step"),
    ("paged_lens_advance", "paged_lens_advance"), ("paged_decode_attn_split", "paged_decode_attn_split"),
    ("flash_decode_reduce", "flash_decode_reduce"), ("embedding", "embedding"), ("_softmax", "softmax_forward"),
    ("softmax", "softmax_forward"), ("topk", "topk_stage1"), ("div (renormalisation)", "div_strided"),
    ("sum (renormalisation)", "sum_forward"), ("moe_gateup", "moe_gateup"), ("moe_down_split", "moe_down_split"),
    ("moe_part_reduce", "moe_part_reduce"), ("cat", "cat_copy"), ("cos", "cos_forward"), ("sin", "sin_forward"),
)


def against_profile(entry: Dict[str, Any], profile: Dict[str, Any], tokens: int) -> str:
    """The census beside a measured kernel table (`run --profile-kernels JSON`), kernel by kernel. Measured = the
    table's launches over `tokens`; the run's prefill is in the table, so a kernel launched once a pass measures
    its count plus one pass's share. A device copy is the row `blit` of a Metal table."""
    counted: collections.Counter = collections.Counter()
    device_copies = row_copies = 0
    for d in entry["decoders"]:
        counted.update(d["by_kernel"])
        device_copies += d["device_copies"]
        row_copies += d["one_row_products"] + d["head_one_row_products"]
        counted["mm"] += d["head_one_row_products"]
    per_measured: collections.Counter = collections.Counter()
    unplaced: collections.Counter = collections.Counter()
    names = [k["name"] for k in profile["kernels"]]
    for label, n in counted.items():
        stem = next((m for c, m in MEASURED_AS if label == c or (c.endswith("(") and label.startswith(c))
                     or label.startswith(c + " (")), None)
        name = next((x for x in names if stem and stem in x), None)
        if name is None:
            unplaced[label] += n
        else:
            per_measured[name] += n
    measured_kernels = sum(k["launches"] for k in profile["kernels"] if k["name"] != "blit") / tokens
    lines = [f"counted kernel launches a token: {sum(counted.values())}; measured: {measured_kernels:.1f} "
             f"(over {tokens} tokens, the prefill's share and the sampler's kernels included)", "",
             "| measured kernel | measured a token | counted a token |", "|---|---|---|"]
    for k in profile["kernels"]:
        if k["name"] == "blit":
            lines.append(f"| blit (device copies) | {k['launches'] / tokens:.2f} | {device_copies + row_copies} "
                         f"({device_copies} copies of contiguous data, {row_copies} one-row matrix products "
                         f"copied to their slot) |")
            continue
        lines.append(f"| {k['name']} | {k['launches'] / tokens:.2f} | {per_measured.get(k['name'], 0)} |")
    for label, n in unplaced.most_common():
        lines.append(f"| (no measured row) {label} | - | {n} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("containers", nargs="+", type=Path)
    parser.add_argument("--reading", choices=READINGS, default="wide")
    parser.add_argument("--json", type=Path, help="write the whole census here")
    parser.add_argument("--markdown", type=Path, help="write the table here (every reading)")
    parser.add_argument("--list", action="store_true", help="print each decoder's launch list")
    parser.add_argument("--activation-dtype", choices=ACTIVATION_DTYPES,
                        help="the dtype the run's hardware profile resolves for activations: casts whose value "
                             "already holds the target dtype at run time launch nothing (default: the graph's "
                             "recorded dtypes decide)")
    parser.add_argument("--profile", type=Path, help="a measured kernel table (run --profile-kernels JSON)")
    parser.add_argument("--tokens", type=int, help="the decoded tokens of the measured run")
    args = parser.parse_args(argv)
    if args.profile and not args.tokens:
        parser.error("--profile needs --tokens: the decoded tokens of the measured run")

    rows: List[Dict[str, Any]] = []
    refused = False
    for path in args.containers:
        try:
            if args.list:
                container = Container(path)
                family = container.json("manifest.json").get("family")
                for decoder in decoders_of(container):
                    graph = decode_graph_of(container, decoder["component"], family,
                                            lm_config_for(container, decoder["component"]), decoder,
                                            args.activation_dtype)
                    print(f"## {container.name} / {decoder['component']} ({args.reading})")
                    print(launch_list(graph, fuse_all(graph, args.reading)))
            entry = census_of(path, args.activation_dtype)
        except Refused as why:
            refused = True
            print(f"refused: {why}", file=sys.stderr)
            rows.append({"container": Container(path).name if Path(path).exists() else str(path), "refused": str(why)})
            continue
        rows.append(entry)
        if args.profile:
            print(against_profile(entry, json.loads(args.profile.read_text()), args.tokens))
    print(table(rows, args.reading))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))
    if args.markdown:
        args.markdown.write_text("\n\n".join(f"### reading: {r}\n\n{table(rows, r)}" for r in READINGS) + "\n")
    return 2 if refused and not any("decoders" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
