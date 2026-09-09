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
