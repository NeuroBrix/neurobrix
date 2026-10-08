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
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

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
    #: The stretches run in slices of a token axis (`LayerPartitioner._chunk_regions`), each one
    #: segment of its own: {"first_op", "last_op", "symbol", "slice", "count", "passes",
    #: "peak_bytes"}. Empty when every piece runs whole — every plan before this brick.
    chunks: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.refusal is None

    @property
    def peak_resident_mb(self) -> float:
        return self.peak_resident_bytes / (1024 * 1024)


class LayerPartitioner:
    """Partition one component graph into budget-sized segments."""

    def __init__(self, graph: Dict[str, Any],
                 weight_sizes: Optional[Dict[str, int]] = None,
                 symbol_map: Optional[Dict[str, int]] = None,
                 compute_dtype_bytes: Optional[int] = None,
                 widths: Optional[Dict[str, int]] = None,
                 token_split: bool = False):
        self._dag = graph
        # Whether a refusal on the activations may be answered by running a stretch of the graph in
        # slices of a token axis (`_chunk_regions`). The solver turns it on only once the cuts that
        # cost nothing — between ops, and the guidance branches one pass each — have refused: a
        # sliced stretch recomputes what its later passes read.
        self._token_split = bool(token_split)
        # Tensors the runtime keeps past their last reader (the strategy's protected ids): a sliced
        # stretch must hand them out whole.
        self.protected: FrozenSet[str] = frozenset()
        self._axes: Dict[str, Any] = {}
        self._op_syms: Optional[Dict[str, Any]] = None
        self._pass_peak_memo: Dict[Tuple[Tuple[str, ...], str, int], int] = {}
        self._regions_used: Optional[Tuple[List["_Region"], int]] = None
        self.tensors: Dict[str, Any] = graph.get("tensors") or {}
        # Activations are sized AT THE REQUEST when the caller gives the request's symbol map: the
        # profiler's resolver (symbolic_shape at the request's symbols) and its compute-dtype rule,
        # the same numbers Prism's placement estimate uses. Without it a tensor's `shape` is the
        # TRACE's — SANA-Video's VAE input [1,128,9,14,22] where a 1280x512 request makes
        # 21x64x160, ~77x more — and every segment was cut for the trace, not the run (the same
        # bug written twice: the profiler already resolved it). No map = the trace, said so by
        # the caller's absence of one, as before.
        self._symbol_map = symbol_map
        self._compute_dtype_bytes = compute_dtype_bytes
        # {tensor_id: bytes per element} each activation is EXECUTED at (core/prism/runtime_widths),
        # the width the placement estimate prices it at — a segment is cut at what the plan pays
        self._widths = widths or {}
        self._resolver = None
        if symbol_map is not None:
            if compute_dtype_bytes is None:
                raise ValueError("ZERO FALLBACK: sizing at a request needs the plan's compute dtype width")
            from neurobrix.core.prism.profiler import ActivationProfiler
            self._resolver = ActivationProfiler(graph)
        self.ops: Dict[str, Any] = graph.get("ops") or {}
        self.order: List[str] = list(graph.get("execution_order") or [])
        # Authoritative sizes when the caller has the weights index; the
        # graph's own shape/dtype otherwise. The index wins because it
        # records what is STORED, which is what a load actually costs.
        self.weight_sizes = weight_sizes or {}

    def _activation_bytes(self, tid: str, t: Dict[str, Any]) -> int:
        if self._resolver is None:
            return tensor_bytes(t) or 0
        shape = self._resolver._resolve_shape(t, self._symbol_map)
        # the width by the tensor's id in the graph — its key, which an entry need not repeat inside it
        if tid in self._widths:
            numel = 1
            for d in shape:
                numel *= int(d)
            return numel * int(self._widths[tid])
        return self._resolver._compute_size(shape, t, self._compute_dtype_bytes)

    # -- dataflow ---------------------------------------------------------

    def _last_use(self) -> Dict[str, int]:
        """tensor id -> index of the op after which it is dead: the PROFILER's liveness
        (`profiler.dag_last_uses`), one rule for both walks of the same graph. This copy recorded
        consumers only, so an output no op reads (a layer norm's statistics, an attention's
        log-sum-exp, a RoPE `copy_` result) was never freed and the curve climbed to the last op:
        Wan2.2-I2V-A14B's transformer at 480x832x81, 108 800 MB of "activations alone" (the Mac's
        refusal, 2026-10-04); this rule prices 7 737 MB."""
        from neurobrix.core.prism.profiler import dag_last_uses
        index = {uid: i for i, uid in enumerate(self.order)}
        return {tid: index[uid] for tid, uid in dag_last_uses(self._dag).items() if uid in index}

    def live_activation_curve(self) -> List[int]:
        """Bytes of activation alive after each op in the order.

        This is the cost of cutting THERE, and its minima are where the
        graph comes apart. The liveness is the profiler's (`dag_last_uses`): a graph output is
        never freed, and a stride-0 broadcast view allocates nothing (`ZERO_ALLOC_OP_TYPES`).

        The transpose of a WEIGHT keeps its bytes on this curve, though the placement estimate
        prices it at zero on the Triton engines (`profiler.weight_transposes_read_in_place`: the
        contraction reads the view in place). Here the figure is the cost of a CUT: between the
        transpose and the contraction that reads it, a seam would carry the weight out of the
        piece that loaded it. Its bytes on the curve are what keeps that seam from looking free.
        """
        if getattr(self, "_curve_memo", None) is not None:
            return list(self._curve_memo)
        self._walk_liveness()
        return list(self._curve_memo)

    def op_peak_curve(self) -> List[int]:
        """Bytes of activation alive WHILE each op runs: what was live before it plus the outputs it
        allocates, its inputs not yet freed — the profiler's rule (`estimate_peak_memory`: allocate,
        record the peak, then free). This is what a piece must RESERVE; `live_activation_curve` is
        what a CUT carries, and its maximum is not the peak.

        The two differ by the op's inputs at their last use: an FFN's activation reads the
        projection's output and writes one of the same size, both alive while it runs. Reserving the
        after-op figure, layer streaming priced Allegro's transformer at its derived request on a
        16 GB V100 (guidance batch 2, 720x1280, 88 frames) at 7 012 MB of activations where the
        profiler prices 12 541 — the gelu's [2, 79 200, 9 216] fp32 input and output, 5 569 MB each,
        counted once — and the Triton run died at aten.gelu::16 asking 5 569 MB with 11 050 MB live
        (split16, 2026-10-05).

        A view (`profiler.VIEW_OP_TYPES`, the profiler's own rule) allocates nothing while it runs:
        its output IS its input's storage, so its in-op figure is the live set before it. Counted as a new buffer, Ming-Lite-Omni-1.5's
        vision tower on the busy Mac read 9 878 MB at an `_unsafe_view` of a 4 420 MB bmm output —
        the same buffer twice — and the plan that streams it was refused."""
        if getattr(self, "_op_peak_memo", None) is None:
            self._walk_liveness()
        return list(self._op_peak_memo)

    def _walk_liveness(self) -> None:
        """One walk of the order, filling both curves (`live_activation_curve`, `op_peak_curve`)."""
        from neurobrix.core.prism.profiler import VIEW_OP_TYPES, ZERO_ALLOC_OP_TYPES
        last = self._last_use()
        outputs = set(self._dag.get("output_tensor_ids") or [])
        curve, during, live = [], [], 0
        # Only what has been ADDED can be freed. Subtracting every input at
        # its last use freed the graph's own inputs too — tensors no op
        # produced — and drove the curve negative, reporting a peak of 0 on a
        # graph whose activations were 64 MB. A live set, not a running total.
        alive: Dict[str, int] = {}
        for i, op_uid in enumerate(self.order):
            op = self.ops.get(op_uid) or {}
            zero = op.get("op_type") in ZERO_ALLOC_OP_TYPES
            before = live
            for tid in op.get("output_tensor_ids") or []:
                t = self.tensors.get(tid)
                if t is not None and not t.get("is_parameter") and tid not in alive:
                    alive[tid] = 0 if zero else self._activation_bytes(tid, t)
                    live += alive[tid]
            during.append(before if op.get("op_type") in VIEW_OP_TYPES else live)
            # Every tensor dead after this op: its inputs at their last use and its own outputs
            # no op reads.
            for tid in list(op.get("input_tensor_ids") or []) + list(op.get("output_tensor_ids") or []):
                if tid in alive and last.get(tid) == i and tid not in outputs:
                    live -= alive.pop(tid)
            curve.append(live)
        # the graph, the request and the widths are fixed at construction, and the parked-change
        # search (`partition`) cuts again many times over the same curve
        self._curve_memo = list(curve)
        self._op_peak_memo = list(during)

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

    def partition(self, budget_bytes: int, max_arena_bytes: Optional[int] = None,
                  parked_cap_bytes: Optional[float] = None) -> Partition:
        """The cut `_greedy` takes, held to `budget_bytes` at every piece CHANGE as well when the
        allocator parks a freed arena (`parked_cap_bytes`; None: it parks nothing the next load
        cannot take back, the plan is the greedy one).

        A piece's arena freed into the Triton allocator's pool stays allocated, parked, when it is
        at most the pool's cap (`DeviceAllocator.free`, `memory.alloc_pool_parked_cap_fraction` of
        the device; `math.inf` where the cap is unbounded), and the next piece's arena takes it back
        only when it lies in [w, 2w] of that request (`_pool_take`). A discrete card's driver
        refuses an allocation it cannot serve and the pool is flushed and the request retried; a
        unified device serves it beside the parked block. So at each change — the step's
        wrap-around included, the last piece to the first — the device holds the incoming arena,
        the outgoing one when it parks and is not taken back, and the activations live across the
        cut. Measured on the Mac (deepseek-moe-16b-chat, 2026-10-04): the last piece's 4.5 GB
        parked while the first one's arena allocated, 13.5 GB live against an 11.3 GB plan.

        Cut again lower until every change fits, or the cut refuses. Lower by the overshoot, but by
        at most a tenth of the cut at a time: the change peak is not monotonic in the cut (a change
        whose incoming arena takes the parked one back costs nothing), and one jump by the whole
        overshoot went from 3 768 MB to 486 MB on PixArt-XL-2-1024-MS's T5 at the Mac's 4096 rung —
        under its 502 MB embedding, a refusal — where pieces of about half the budget fit every
        change. Nor below the cut's own floor in one step (`_cut_floor`: the activations' peak plus
        the most weight one op reads — under it the greedy refuses whatever the change): a step
        that would cross it lands ON it. Janus-Pro-7B's language model at the Mac's 4096 rung
        (triton-sequential) stepped from 1 508 to 1 427 MB, under its 1 461 MB floor, and was
        refused, where every cut from 1 461 to 1 470 MB fits every change at 1 439 MB."""
        part = self._greedy(budget_bytes, max_arena_bytes)
        if parked_cap_bytes is None or not part.fits:
            return part
        cut_at = budget_bytes
        floor = self._cut_floor()
        while True:
            change = self._change_peak(part.segments, parked_cap_bytes)
            over = change - budget_bytes
            if over <= 0:
                part.peak_resident_bytes = max(part.peak_resident_bytes, change)
                return part
            step = max(min(over, cut_at // 10), 1024 * 1024)
            cut_at = floor if cut_at - step < floor < cut_at else cut_at - step
            again = self._greedy(cut_at, max_arena_bytes)
            if not again.fits:
                again.refusal = (
                    f"the piece changes peak at {change / (1024*1024):.1f} MB against the "
                    f"{budget_bytes / (1024*1024):.1f} MB budget once the outgoing arena stays "
                    f"parked in the allocator's pool beside the incoming one, and cutting smaller "
                    f"to make room refuses: " + again.refusal)
                return again
            part = again

    def _cut_floor(self) -> int:
        """The lowest cut the greedy can take: the activations' peak (reserved in every piece) plus
        the most weight a single op reads (an op is never split across pieces) — or, when stretches
        run in slices (`_chunk_regions`), the peak those slices leave plus the most weight one op
        or one stretch reads (a stretch is one piece). Below it `_greedy` refuses by construction;
        at or above it the op that reads the most fits in a piece."""
        if self._regions_used is not None:
            regions, reserve = self._regions_used
            return reserve + self._weights_floor(regions)
        peaks = self.op_peak_curve()
        one_op = max((sum(self._bytes_for(n) for n in self._op_weight_names(uid)) for uid in self.order),
                     default=0)
        return (max(peaks) if peaks else 0) + one_op

    @staticmethod
    def _change_peak(segments: List[Segment], parked_cap_bytes: float) -> int:
        """The most a piece change holds: the incoming arena, the outgoing one when the pool parks
        it (at most `parked_cap_bytes`) and the incoming request does not take it back (it lies in
        [w, 2w] of that request), and the activations live across the cut — every consecutive
        pair, the last piece to the first included. One piece: no change."""
        if len(segments) < 2:
            return 0
        peak = 0
        for i, out in enumerate(segments):
            inc = segments[(i + 1) % len(segments)]
            taken_back = inc.weight_bytes <= out.weight_bytes <= 2 * inc.weight_bytes
            parked = out.weight_bytes if (out.weight_bytes <= parked_cap_bytes and not taken_back) else 0
            peak = max(peak, inc.weight_bytes + parked + out.live_bytes_at_exit)
        return peak

    def _greedy(self, budget_bytes: int, max_arena_bytes: Optional[int] = None) -> Partition:
        """The cut between ops (`_greedy_on` over the activation curve); when it refuses and the
        solver allows it, the same cut over the curve that running some stretches in slices of a
        token axis leaves (`_chunk_regions`), each such stretch a piece of its own."""
        self._regions_used = None
        part = self._greedy_on(self.live_activation_curve(), budget_bytes, max_arena_bytes,
                               peaks=self.op_peak_curve())
        if part.fits or not self._token_split or self._resolver is None:
            return part
        regions, why = self._chunk_regions(budget_bytes, max_arena_bytes)
        if regions is None:
            part.refusal = (f"{part.refusal} Nor does running a stretch of it in slices of a token "
                            f"axis: {why}.")
            return part
        eff, reserve = self._effective_curve(regions)
        again = self._greedy_on(self.live_activation_curve(), budget_bytes, max_arena_bytes, regions,
                                reserve, peaks=eff)
        if again.fits:
            self._regions_used = (regions, reserve)
            again.chunks = [r.record() for r in regions]
        else:
            again.refusal = (f"{part.refusal} Running {len(regions)} stretch(es) in slices of a token "
                             f"axis still refuses: {again.refusal}")
        return again

    def _greedy_on(self, curve: List[int], budget_bytes: int, max_arena_bytes: Optional[int] = None,
                   regions: Sequence["_Region"] = (), reserve: Optional[int] = None,
                   peaks: Optional[List[int]] = None) -> Partition:
        """Greedy left-to-right: extend while the segment's weights fit.

        Greedy is right here because the order is fixed — the engine replays
        it — so the only freedom is where to cut, and taking as much as fits
        before each cut minimises the number of loads. It is not an
        optimisation problem with a better answer hiding in it.

        `max_arena_bytes` bounds one segment's weights on their own: they are
        ONE allocation when the piece loads (its arena), and a device grants a
        single allocation only up to its largest one (the profile's
        `max_allocation_mb`). None: the device grants its whole memory in one.

        `regions`: stretches run in slices (`_chunk_regions`). Each is one piece of its own — cut
        before its first op and after its last — and `reserve` is the activations' peak with them
        sliced (`_effective_curve`).

        `curve` is what each cut carries (`live_activation_curve`, a piece's `live_bytes_at_exit`);
        `peaks` what each op holds while it runs (`op_peak_curve`), the activations every piece
        reserves. None: `curve` stands for both (a caller with no op inside its ops' live sets).
        """
        peaks = curve if peaks is None else peaks
        peak_live = max(peaks) if peaks else 0
        if reserve is not None:
            peak_live = max(peak_live, reserve)
        starts = {r.first: r for r in regions}

        # The weights get the budget MINUS what the activations will hold.
        # Sizing segments against the whole budget and then adding the
        # activations on top announced a peak ABOVE the budget — 521.5 MB
        # against 500, 9042.5 against 9000, measured on the first version of
        # this method. The number this returns is the number the strategy
        # promises, so it has to be the one that is actually held.
        weight_budget = budget_bytes - peak_live
        arena_bound = ""
        if max_arena_bytes is not None and max_arena_bytes < weight_budget:
            weight_budget = int(max_arena_bytes)
            arena_bound = (f" (one segment's weights are one allocation, bounded by the "
                           f"device's largest: {max_arena_bytes / (1024*1024):.1f} MB)")
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

        def close(last_index: int) -> None:
            nonlocal cur_names, cur_bytes, cur_first, cur_ops, total
            segments.append(Segment(
                index=len(segments), first_op=cur_first,
                last_op=self.order[last_index], op_count=cur_ops,
                weight_names=set(cur_names), weight_bytes=cur_bytes,
                live_bytes_at_exit=curve[last_index]))
            total += cur_bytes
            cur_names, cur_bytes, cur_first, cur_ops = set(), 0, None, 0

        i = 0
        while i < len(self.order):
            op_uid = self.order[i]
            region = starts.get(i)
            if region is not None:
                # A sliced stretch is ONE piece: every pass re-reads its weights.
                if cur_first is not None:
                    close(i - 1)
                names = region.weight_names
                add = sum(self._bytes_for(n) for n in names)
                if add > weight_budget:
                    return Partition(
                        segments=[], total_weight_bytes=0, peak_resident_bytes=0,
                        peak_live_bytes=peak_live,
                        refusal=(
                            f"the stretch {region.first_op!r}..{region.last_op!r} run in slices of "
                            f"{region.symbol} reads {add / (1024*1024):.1f} MB of weights, over the "
                            f"{weight_budget / (1024*1024):.1f} MB left once activations are reserved "
                            f"({peak_live / (1024*1024):.1f} MB of a "
                            f"{budget_bytes / (1024*1024):.1f} MB budget){arena_bound}"))
                cur_first, cur_names, cur_bytes = op_uid, set(names), add
                cur_ops = region.last - region.first + 1
                close(region.last)
                i = region.last + 1
                continue

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
                        f"{budget_bytes / (1024*1024):.1f} MB budget){arena_bound}. "
                        f"A cut cannot "
                        f"run half an op, so no partition of this graph "
                        f"fits. Serving it needs the op's own weights "
                        f"sharded, which is a different rung."))

            if cur_first is not None and cur_bytes + add > weight_budget:
                close(i - 1)
                new, add = names, sum(self._bytes_for(n) for n in names)

            if cur_first is None:
                cur_first = op_uid
            cur_names |= new
            cur_bytes += add
            cur_ops += 1
            i += 1

        if cur_first is not None:
            close(len(self.order) - 1)

        # What is resident at the worst moment: one segment's weights plus
        # the activations alive while it runs. This is the number the
        # strategy ANNOUNCES, and it is the number it holds.
        peak_resident = max(
            (s.weight_bytes for s in segments), default=0) + peak_live

        return Partition(segments=segments, total_weight_bytes=total,
                         peak_resident_bytes=peak_resident,
                         peak_live_bytes=peak_live)

    # -- stretches run in slices of a token axis --------------------------

    def _bytes_at(self, tid: str, symbol_map: Dict[str, int]) -> int:
        """One activation's bytes at `symbol_map` (the request's, or a slice's)."""
        t = self.tensors.get(tid) or {}
        shape = self._resolver._resolve_shape(t, symbol_map)
        if tid in self._widths:
            numel = 1
            for d in shape:
                numel *= int(d)
            return numel * int(self._widths[tid])
        return self._resolver._compute_size(shape, t, self._compute_dtype_bytes)

    def _axis(self, sym: str):
        from neurobrix.core.prism.chunked_region import TokenAxis
        if sym not in self._axes:
            self._axes[sym] = TokenAxis(self._dag, sym)
        return self._axes[sym]

    def _op_symbols(self) -> Dict[str, Any]:
        from neurobrix.core.prism.chunked_region import op_symbols
        if self._op_syms is None:
            self._op_syms = op_symbols(self._dag)
        return self._op_syms

    def _op_weight_bytes(self, i: int) -> int:
        return sum(self._bytes_for(n) for n in self._op_weight_names(self.order[i]))

    def _weights_floor(self, regions: Sequence["_Region"]) -> int:
        """The most weight one piece must hold: one op's outside every stretch, or one stretch's."""
        inside: Set[int] = set()
        for r in regions:
            inside.update(range(r.first, r.last + 1))
        one_op = max((self._op_weight_bytes(i) for i in range(len(self.order)) if i not in inside),
                     default=0)
        return max([one_op] + [sum(self._bytes_for(n) for n in r.weight_names) for r in regions])

    def _effective_curve(self, regions: Sequence["_Region"]) -> Tuple[List[int], int]:
        """What each op holds while it runs (`op_peak_curve`) with these stretches sliced, and the
        peak to reserve. Inside a stretch the live set is its sliced peak (`_Region.peak_bytes`);
        outside, the op's own figure — the stretch hands out exactly the tensors the whole ops
        would."""
        eff = list(self.op_peak_curve())
        reserve = 0
        for r in regions:
            for i in range(r.first, r.last + 1):
                eff[i] = r.peak_bytes
            reserve = max(reserve, r.peak_bytes)
        return eff, max([reserve] + eff) if eff else reserve

    def _pass_peak(self, cp, sym: str, c: int, chunked: bool) -> int:
        """The live activations of one pass at slice `c`: its own curve (its ops alone, its targets
        kept) plus the contiguous slices of the stretch's inputs it is fed."""
        from neurobrix.core.prism.chunked_region import pass_graph
        key = (tuple(cp.ops), sym, c if chunked else 0)
        if key not in self._pass_peak_memo:
            sm = dict(self._symbol_map)
            if chunked:
                sm[sym] = c
            sub = LayerPartitioner(pass_graph(self._dag, cp), {}, symbol_map=sm,
                                   compute_dtype_bytes=self._compute_dtype_bytes, widths=self._widths)
            peaks = sub.op_peak_curve()
            self._pass_peak_memo[key] = max(peaks) if peaks else 0
        return self._pass_peak_memo[key]

    def _accumulator_bytes(self, tid: str, symbol_map: Dict[str, int]) -> int:
        """What summing a contraction's slices holds at once (`chunked_region.accumulate`): the
        running sum and the sum that replaces it, in the accumulator the engines' DtypeEngine
        names (`triton/dtype.contraction_accumulator_bytes`, the width form of
        `contraction_accumulator_dtype`), plus the partial cast to it when the op stores
        narrower. The store once summed (`chunked_region.store`) is no wider than one of them."""
        from neurobrix.triton.dtype import contraction_accumulator_bytes
        store = self._bytes_at(tid, symbol_map)
        numel = 1
        for d in self._resolver._resolve_shape(self.tensors.get(tid) or {}, symbol_map):
            numel *= int(d)
        if numel == 0:
            return 0
        width = store // numel
        acc = contraction_accumulator_bytes(width)
        return numel * (2 * acc + (acc if acc != width else 0))

    def _region_peak(self, ax, plan, c: int) -> int:
        """What a stretch run in slices of `c` holds at its worst: everything live before it (held
        until its last pass), its outputs assembled whole, each contraction's accumulator and the
        sum that replaces it (`_accumulator_bytes`, its store included), and the worst pass — its
        slice's activations plus the slices of the inputs it reads that carry the axis (a
        contiguous copy each, a broadcast's too)."""
        sm = self._symbol_map
        curve = self.live_activation_curve()
        before = curve[plan.first - 1] if plan.first > 0 else 0
        outs = sum(self._bytes_at(t, sm) for t in plan.outputs if t not in plan.contractions)
        acc = sum(self._accumulator_bytes(t, sm) for t in plan.contractions)
        sliced = dict(sm)
        sliced[plan.symbol] = c
        worst = 0
        for cp in plan.passes:
            fed = 0
            if cp.chunked:
                reads = set()
                inside = set()
                for uid in cp.ops:
                    op = self.ops.get(uid) or {}
                    reads.update(t for t in op.get("input_tensor_ids") or [] if t not in inside)
                    inside.update(op.get("output_tensor_ids") or [])
                # and the carriers of the stretch's symbols, sliced into every pass beside them
                reads.update(t for t in plan.layouts if t not in plan.outputs)
                for t in reads:
                    if t in plan.layouts and t not in plan.outputs:
                        fed += self._bytes_at(t, sliced)
            worst = max(worst, self._pass_peak(cp, plan.symbol, c, cp.chunked) + fed)
        return before + outs + acc + worst

    def _chunk_regions(self, budget_bytes: int, max_arena_bytes: Optional[int]
                       ) -> Tuple[Optional[List["_Region"]], str]:
        """The stretches to run in slices of a token axis so the cut between ops fits, or (None,
        why not). Activations first: the curve's hottest op not yet in a stretch is enclosed in
        one (`_new_region`), a stretch over the target is sliced finer, until every op's live set
        leaves room for the most weight one piece must hold."""
        from neurobrix.core.prism.chunked_region import token_symbols
        syms = token_symbols(self._dag, self._symbol_map)
        if not syms:
            return None, "no symbol of this graph is bound from a dimension of one of its inputs"
        regions: List[_Region] = []
        while True:
            for r in regions:
                w = sum(self._bytes_for(n) for n in r.weight_names)
                if max_arena_bytes is not None and w > max_arena_bytes:
                    return None, (f"the stretch {r.first_op!r}..{r.last_op!r} reads {w / _MB:.1f} MB "
                                  f"of weights, over the device's largest allocation "
                                  f"({max_arena_bytes / _MB:.1f} MB)")
            eff, _ = self._effective_curve(regions)
            target = budget_bytes - self._weights_floor(regions)
            h = max(range(len(eff)), key=lambda i: eff[i]) if eff else 0
            if not eff or eff[h] <= target:
                if not regions:
                    return None, "the activations are not what binds"
                return regions, ""
            owner = next((r for r in regions if r.first <= h <= r.last), None)
            if owner is not None:
                c = self._largest_slice(owner.axis, owner.plan, target, owner.slice - 1)
                if c is None:
                    return None, (f"the stretch {owner.first_op!r}..{owner.last_op!r} peaks at "
                                  f"{self._region_peak(owner.axis, owner.plan, 1) / _MB:.1f} MB in "
                                  f"slices of one {owner.symbol}, over the {target / _MB:.1f} MB its "
                                  f"activations may hold")
                owner.set_slice(c, self._region_peak(owner.axis, owner.plan, c), self._symbol_map)
                continue
            best: Optional[_Region] = None
            whys: List[str] = []
            for sym in syms:
                got, why = self._new_region(self._axis(sym), h, eff, target, regions)
                if got is None:
                    whys.append(f"{sym}: {why}")
                elif best is None or (got.count, got.peak_bytes) < (best.count, best.peak_bytes):
                    best = got
            if best is None:
                return None, (f"{self.order[h]!r} holds {eff[h] / _MB:.1f} MB of activations "
                              f"against {target / _MB:.1f} MB — " + "; ".join(whys))
            regions.append(best)
            regions.sort(key=lambda r: r.first)

    def _largest_slice(self, ax, plan, target: int, upper: int) -> Optional[int]:
        """The largest slice in [1, upper] whose stretch peak is at most `target`, or None."""
        if upper < 1 or self._region_peak(ax, plan, 1) > target:
            return None
        lo, hi = 1, upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._region_peak(ax, plan, mid) <= target:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _new_region(self, ax, h: int, eff: List[int], target: int, regions: Sequence["_Region"]
                    ) -> Tuple[Optional["_Region"], str]:
        """A stretch around op `h` run in slices of `ax.sym`: the ops over the target around it
        (none may mix the axis), widened to the seams where the least is live — before its first op
        and after its last — among the ops that do not mix the axis either."""
        from neurobrix.core.prism.chunked_region import MIX, plan_region, region_carriers
        if ax.verdicts[h].kind == MIX:
            return None, f"{self.order[h]} mixes the axis ({ax.verdicts[h].reason})"
        taken: Set[int] = set()
        for r in regions:
            taken.update(range(r.first, r.last + 1))

        def free(i: int) -> bool:
            return 0 <= i < len(self.order) and i not in taken and ax.verdicts[i].kind != MIX

        lo = hi = h
        while free(lo - 1) and eff[lo - 1] > target:
            lo -= 1
        while free(hi + 1) and eff[hi + 1] > target:
            hi += 1
        span_lo, span_hi = lo, hi
        while free(span_lo - 1):
            span_lo -= 1
        while free(span_hi + 1):
            span_hi += 1
        curve = self.live_activation_curve()

        def before(i: int) -> int:
            return curve[i - 1] if i > 0 else 0

        # A seam farther from the hot ops is worth taking only where less is live there than at
        # every seam nearer them: it adds ops (their weights, and recomputation in later passes)
        # for nothing otherwise. The record lows walking outward, under the target.
        def record_lows(seq: Sequence[int], value: Callable[[int], int]) -> List[int]:
            out: List[int] = []
            low: Optional[int] = None
            for i in seq:
                v = value(i)
                if v <= target and (low is None or v < low):
                    out.append(i)
                    low = v
            return out

        lefts = record_lows(range(lo, span_lo - 1, -1), before)
        rights = record_lows(range(hi, span_hi + 1), lambda i: curve[i])
        pairs = sorted(((L, R) for L in lefts for R in rights),
                       key=lambda p: (max(before(p[0]), curve[p[1]]), before(p[0]) + curve[p[1]],
                                      p[1] - p[0]))
        total = int(self._symbol_map[ax.sym])
        why = f"no seam around {self.order[h]} leaves the stretch under {target / _MB:.1f} MB"
        for L, R in pairs:
            plan, no = plan_region(ax, L, R, protected=self.protected,
                                   carriers=region_carriers(self._dag, self.order[L:R + 1],
                                                            self._op_symbols()))
            if plan is None:
                why = no
                continue
            c = self._largest_slice(ax, plan, target, total - 1)
            if c is None:
                why = (f"{self.order[L]}..{self.order[R]} peaks at "
                       f"{self._region_peak(ax, plan, 1) / _MB:.1f} MB in slices of one")
                continue
            names: Set[str] = set()
            for uid in self.order[L:R + 1]:
                names |= self._op_weight_names(uid)
            reg = _Region(axis=ax, plan=plan, weight_names=names)
            reg.set_slice(c, self._region_peak(ax, plan, c), self._symbol_map)
            return reg, ""
        return None, why


_MB = 1024 * 1024


@dataclass
class _Region:
    """One stretch run in slices of a token axis, as the partitioner sized it."""
    axis: Any
    plan: Any
    weight_names: Set[str]
    slice: int = 0
    count: int = 0
    peak_bytes: int = 0

    @property
    def first(self) -> int:
        return self.plan.first

    @property
    def last(self) -> int:
        return self.plan.last

    @property
    def first_op(self) -> str:
        return self.plan.first_op

    @property
    def last_op(self) -> str:
        return self.plan.last_op

    @property
    def symbol(self) -> str:
        return self.plan.symbol

    def set_slice(self, c: int, peak: int, symbol_map: Dict[str, int]) -> None:
        total = int(symbol_map[self.symbol])
        self.slice, self.count, self.peak_bytes = int(c), -(-total // int(c)), int(peak)

    def record(self) -> Dict[str, Any]:
        return {"first_op": self.first_op, "last_op": self.last_op, "symbol": self.symbol,
                "slice": self.slice, "count": self.count, "passes": len(self.plan.passes),
                "peak_bytes": self.peak_bytes}


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


def flow_reads_by_name(component: str, graph: Optional[Dict[str, Any]],
                       topology: Optional[Dict[str, Any]]) -> bool:
    """Whether a flow reads `component`'s weights BY NAME, outside its graph — the ONE rule a
    streamed base's resident set (`LayerStreamingStrategy._ensure_flow_reads`) and Prism's reserve
    for it (`PrismSolver._try_layer_streaming`) both read. Two readers: the component whose
    embeddings the flow supplies (`flow_embeds_into`), and the VLM handlers' `head_component`,
    whose 2-D weight both engines' `_compute_logits` project with directly — its graph takes hidden
    states, so the first rule never saw it, and a streamed head's base held nothing (GLM-4.1V and
    MiniCPM-o streamed, 2026-10-08: the logits came from the token embedding instead)."""
    if flow_embeds_into(graph):
        return True
    vlm = ((topology or {}).get("flow") or {}).get("vlm") or {}
    return bool(component) and component == vlm.get("head_component")


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
                        order_index: Optional[Dict[str, int]] = None,
                        pass_ops: Optional[Sequence[str]] = None,
                        outputs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """One segment as a standalone, executable graph.

    The point of returning a GRAPH rather than an op range is that the whole
    engine already knows how to run a graph. A segment executed this way goes
    through the same executor, the same binding, the same dispatch as any
    component — the only difference is that it holds one segment's weights.

    Its inputs are the tensors produced BEFORE it and read INSIDE it; its
    outputs are the tensors produced inside it and read AFTER it, plus any of
    the component's own outputs it produces. Those two sets are exactly what
    must cross the seam, and they are computed here rather than assumed.

    `pass_ops` / `outputs`: one PASS of a stretch run in slices (core/prism/chunked_region.py) — a
    subset of the segment's ops, in order, and the tensors it hands out — instead of the whole
    range and what is read after it. Everything else (seam aliasing, carried symbols) is the
    segment's rule, unchanged.
    """
    tensors: Dict[str, Any] = graph.get("tensors") or {}
    ops: Dict[str, Any] = graph.get("ops") or {}
    order: List[str] = list(graph.get("execution_order") or [])
    if order_index is None:
        order_index = {op_uid: i for i, op_uid in enumerate(order)}

    first = order_index[segment.first_op]
    last = order_index[segment.last_op]
    inside = list(pass_ops) if pass_ops is not None else order[first:last + 1]

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
    if outputs is not None:
        stray = [t for t in outputs if t not in produced_here]
        if stray:
            raise ValueError(f"build_segment_graph: outputs {stray[:3]} are not produced by its ops")
        seg_outputs = list(outputs)

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


