"""A streamed piece run in slices of a token axis — the runtime of `Partition.chunks`.

Prism plans a stretch of a streamed component in slices of a token axis when its activations alone
are over the rung (`LayerPartitioner._chunk_regions`, core/prism/chunked_region.py). The stretch is
one piece of the plan; this object stands where that piece's executor stood in
`LayerStreamingStrategy`, with the same surface (`_dag`, `protect_tensor_id`, `load_weights`, `run`,
`tensors_of_last_run`, `unload_weights`, `register_triton_interceptors`), so the strategy's loop
runs it as any other piece.

Inside, the stretch runs in PASSES (`plan_region`): each pass is an ordinary executor over the ops
it needs, run once per slice of the axis. A pass's inputs that carry the axis are sliced
(`take_slice`); the outputs that carry it are assembled whole (`put_slice`); a contraction over the
axis is summed over the slices and fed whole to the later passes. The weights are loaded ONCE per
run, on the piece's own executor, and every pass borrows them (`GraphExecutor._borrow`).

Both engines: the slicing is `reshape` / `narrow` / `contiguous` / `new_empty` / `copy_` / `+` on
whatever tensor type the engine hands out — torch.Tensor in compiled, NBXTensor in the Triton
modes. No import of either (R33), no conversion between them. A contraction's partials are summed
in the dtype the pass executor's DtypeEngine names (`accumulation_dtype`) and stored once in the
dtype the op's own output was given. None of these launches an autotuned kernel: the stitch holds
no key of the census (tests/unit/prism/test_the_stitch_launches_no_autotuned_kernel.py).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from neurobrix.core.prism.chunked_region import (
    TokenAxis, accumulate, dim_source, slice_reads, own, plan_region, put_slice, region_carriers, take_slice_cyclic,
    sliced_reads, store, take_slice, whole_shape)
from neurobrix.core.prism.layer_partition import Segment, build_segment_graph
from neurobrix.core.strategies.value_parking import to_device
from neurobrix.core.strategies.tiled_norm import group_norm_apply, norm_statistics
from neurobrix.core.runtime.graph_executor import output_key


class ChunkedPiece:
    """One piece of a streamed component, run in slices of `chunk["symbol"]`."""

    def __init__(self, dag: Dict[str, Any], holder: Any, chunk: Dict[str, Any],
                 make_executor: Callable[[Dict[str, Any], Any], Any],
                 axis: Optional[TokenAxis] = None):
        """`dag`: the component's graph as Prism cut it. `holder`: the piece's own executor (its
        segment graph), which loads the stretch's weights. `chunk`: the plan's record.
        `make_executor(graph, lender)`: an executor for a pass graph borrowing from `lender`, built
        as the strategy builds a piece."""
        self._full = dag
        self._holder = holder
        self._chunk = dict(chunk)
        self._make = make_executor
        self._sym = str(chunk["symbol"])
        self._slice = int(chunk["slice"])
        # the axis' unit (a window's rows): a slice of c is c * unit of the symbol; a plan before units is 1
        self._unit = int(chunk.get("unit", 1))
        if self._slice < 1 or self._unit < 1:
            raise ValueError(f"chunked piece: slice {self._slice} of {self._sym} in units of {self._unit}")
        self._axis = axis if axis is not None else TokenAxis(dag, self._sym, self._unit)
        if self._axis.unit != self._unit:
            raise ValueError(f"chunked piece: the axis is in units of {self._axis.unit}, the plan's of {self._unit}")
        index = {u: i for i, u in enumerate(self._axis.order)}
        missing = [u for u in (chunk["first_op"], chunk["last_op"]) if u not in index]
        if missing:
            raise RuntimeError(f"chunked piece: the plan's stretch bounds {missing} are not in the graph")
        self._first, self._last = index[chunk["first_op"]], index[chunk["last_op"]]
        self._protected: Set[str] = set()
        self._interceptors: Dict[str, Any] = {}
        self._passes: Optional[List[Any]] = None
        self._plan = None
        self._last_run: Dict[str, Any] = {}
        # contraction tid -> (accumulator dtype, store dtype): the dtype the engine gives the op's
        # output unpinned, read from its first partial; where the accumulator is wider, the op is
        # pinned at the accumulator's width on its pass (`GraphExecutor.pin_fp32_ops`)
        self._sums: Dict[str, Any] = {}
        self._loaded = False
        self._nbx = None
        # {norm op uid: (mean, rstd)} of the run in flight — the statistics of a group norm over the axis, taken once
        # over its whole input before the slices (`norm_statistics`) and read by its interceptor on every slice
        self._norm_stats: Dict[str, Any] = {}

    # -- the surface a piece shows its strategy ------------------------------------------------

    @property
    def _dag(self) -> Dict[str, Any]:
        return self._holder._dag

    @property
    def device(self) -> Any:
        """The device the piece runs on: its holder's."""
        return getattr(self._holder, "device", None)

    @property
    def _weights(self) -> Any:
        return getattr(self._holder, "_weights", None)

    def __getattr__(self, name: str) -> Any:
        # Read-only facts of the piece (`_component_name`, `mode`, `device` ...) are its holder's.
        if name.startswith("__") or name in ("_holder",):
            raise AttributeError(name)
        return getattr(self._holder, name)

    def protect_tensor_id(self, tid: str) -> None:
        if tid not in self._protected:
            self._protected.add(tid)
            if self._plan is not None and tid not in self._plan.outputs:
                self._passes = None        # re-planned with it among the outputs, at the next run
        self._holder.protect_tensor_id(tid)

    def register_triton_interceptors(self, interceptors: Dict[str, Any]) -> None:
        inside = {(self._axis.ops.get(u) or {}).get("op_type") for u in self._axis.order[self._first:self._last + 1]}
        hit = sorted(t for t in interceptors if t in inside)
        if hit:
            raise RuntimeError(
                f"chunked piece: an interceptor for {hit} would run on slices of {self._sym} inside "
                f"the stretch {self._chunk['first_op']}..{self._chunk['last_op']}; an interceptor "
                f"keeps its own state across calls, and a slice is not the call it was written for")
        self._interceptors.update(interceptors)
        self._holder.register_triton_interceptors(interceptors)
        for ex in self._passes or ():
            ex.register_triton_interceptors(interceptors)

    def register_op_interceptors(self, interceptors: Dict[str, Any]) -> None:
        """The compiled engine's op-type interceptors (its KV cache): refused inside the sliced
        stretch for the reason `register_triton_interceptors` gives; the holder runs the rest."""
        inside = {(self._axis.ops.get(u) or {}).get("op_type") for u in self._axis.order[self._first:self._last + 1]}
        hit = sorted(t for t in interceptors if t in inside)
        if hit:
            raise RuntimeError(
                f"chunked piece: an interceptor for {hit} would run on slices of {self._sym} inside "
                f"the stretch {self._chunk['first_op']}..{self._chunk['last_op']}; an interceptor "
                f"keeps its own state across calls, and a slice is not the call it was written for")
        self._holder.register_op_interceptors(interceptors)

    def register_op_uid_interceptors(self, interceptors: Dict[str, Any], groups=(),
                                     planned=()) -> None:
        """Op-level interceptors for ops of the stretch, which runs in PASSES over slices.

        An op the PLAN tiles is refused by name: Prism drops the op-level tiling of a sliced
        stretch (its ops are budgeted per slice), so one reaching here is a plan the budget was
        not accepted under. The interceptors the tiling engine derives from the whole graph
        itself — an in-place reuse proven on the whole graph's liveness, a proxy chain — are
        not registered on the passes: their proof does not hold for a slice (a pass's
        whole-fed contraction is read again by the next slice; an in-place add would have
        overwritten it). The passes run those ops as the graph writes them."""
        inside = set(self._axis.order[self._first:self._last + 1])
        hit = sorted(u for u in planned if u in inside and u in interceptors)
        if hit:
            raise RuntimeError(
                f"chunked piece: the plan tiles {hit[:4]}{'...' if len(hit) > 4 else ''} "
                f"inside the stretch {self._chunk['first_op']}..{self._chunk['last_op']}, which "
                f"runs in slices of {self._sym}; Prism drops a sliced stretch's op-level tiling, "
                f"so this plan is not the one the budget was accepted under")
        outside = {u: f for u, f in interceptors.items() if u not in inside}
        if outside:
            self._holder.register_op_uid_interceptors(
                outside, groups=[g for g in groups if g and g[0] not in inside],
                planned=[u for u in planned if u in outside])

    def load_weights(self, nbx_path: str, component: str, *args: Any, **kwargs: Any) -> None:
        self._nbx = (nbx_path, component)
        self._holder.load_weights(nbx_path, component, *args, **kwargs)
        self._loaded = True

    def unload_weights(self) -> None:
        for ex in self._passes or ():
            ex.unload_weights()
        self._holder.unload_weights()
        self._loaded = False
        # What the last run kept is read BEFORE the unload (`tensors_of_last_run`, the executor's contract: the
        # unload drops the arena and the capture both). Kept past it, every sliced stretch held its assembled output on
        # the device for the rest of the run: HAT-S at 2048 px on 32 GB, four stretches' 2 304 MB outputs beside the
        # ninth piece (2026-10-22).
        self._last_run = {}

    def sliced_input_names(self) -> Set[str]:
        """The fed names this stretch reads only in slices (`chunked_region.sliced_reads`): with the values between
        pieces parked, the strategy leaves them on the host and each pass uploads its slice."""
        if not self._chunk.get("host_reads"):
            return set()                  # the plan keeps this stretch's inputs whole on the device
        if self._plan is None:
            self._build()
        tids = sliced_reads(self._plan, self._axis.ops)
        return {n for n in (self._dag.get("segment_input_names") or []) if self._tid(n) in tids}

    def tensors_of_last_run(self, tids: Sequence[str]) -> Dict[str, Any]:
        return {t: self._last_run[t] for t in tids if t in self._last_run}

    # -- the passes ----------------------------------------------------------------------------

    def _build(self) -> None:
        ax = self._axis
        carriers = region_carriers(self._full, ax.order[self._first:self._last + 1])
        plan, why = plan_region(ax, self._first, self._last, protected=frozenset(self._protected),
                                carriers=carriers)
        if plan is None:
            raise RuntimeError(
                f"chunked piece: the plan's stretch {self._chunk['first_op']}..{self._chunk['last_op']} "
                f"does not run in slices of {self._sym} on this graph: {why}")
        if int(self._chunk["passes"]) != len(plan.passes):
            raise RuntimeError(
                f"chunked piece: the plan priced {self._chunk['passes']} pass(es) for "
                f"{self._chunk['first_op']}..{self._chunk['last_op']}, this graph needs "
                f"{len(plan.passes)} — not the stretch the budget was accepted under")
        seg = Segment(index=int(self._dag["segment_index"]), first_op=plan.first_op,
                      last_op=plan.last_op, op_count=plan.last - plan.first + 1)
        passes = []
        for cp in plan.passes:
            sub = build_segment_graph(self._full, seg, pass_ops=cp.ops, outputs=cp.targets)
            ex = self._make(sub, self._holder)
            pins = self._pins_of(cp, plan, sub)
            if pins:
                ex.pin_fp32_ops(pins)
            if self._interceptors:
                ex.register_triton_interceptors(self._interceptors)
            for tid in self._protected:
                if tid in cp.targets:
                    ex.protect_tensor_id(tid)
            norms = {u: self._norm_interceptor(u) for u in plan.norms if u in cp.ops}
            if norms:
                ex.register_op_uid_interceptors(norms)
            passes.append(ex)
        self._plan, self._passes = plan, passes

    def _norm_interceptor(self, uid: str) -> Callable[..., Any]:
        """`aten::native_group_norm` on a slice, normalised with the WHOLE input's statistics (`_norm_stats[uid]`).
        Returns what the op returns — (out, mean, rstd), the output in fp32 as the engine's fp32-class norm gives it."""
        def norm_slice(x, weight, bias, n, c, hw, group, eps, *rest, **kw):
            mean, rstd = self._norm_stats[uid]
            return group_norm_apply(x, weight, bias, int(group), mean, rstd)
        # the compiled engine's norm returns its fp32 output itself; on the Triton engine the DtypeEngine wraps the
        # interceptor as it wraps the op it replaces (its operand casts and its output contract)
        norm_slice.self_manages_dtype = not str(getattr(self._holder, "mode", "")).startswith("triton")
        return norm_slice

    def _pins_of(self, cp: Any, plan: Any, graph: Dict[str, Any]) -> Set[str]:
        """The producers of the contractions this chunked pass sums whose partial the engine gives
        narrower than its accumulator (learned at the first partial, `_sums`)."""
        if not cp.chunked:
            return set()
        wide = {t for t in cp.targets if t in plan.contractions
                and t in self._sums and self._sums[t][0] != self._sums[t][1]}
        return {u for u, op in (graph.get("ops") or {}).items()
                if wide.intersection(op.get("output_tensor_ids") or ())}

    def _tid(self, name: str) -> str:
        """The graph tensor a fed name stands for: a seam tid as is, a component input under
        its `input::` id."""
        tensors = self._axis.tensors
        if name in tensors and not name.startswith("input::"):
            return name
        return "input::" + name

    def _symbols(self, feed: Dict[str, Any]) -> Callable[[str], int]:
        syms = ((self._full.get("symbolic_context") or {}).get("symbols") or {})
        cache: Dict[str, int] = {}

        def value(sid: str) -> int:
            if sid not in cache:
                src = dim_source(syms.get(sid))
                if src is None:
                    raise RuntimeError(f"chunked piece: symbol {sid!r} binds from no input dimension")
                tid, dim = src
                name = tid[7:] if tid.startswith("input::") else tid
                x = feed.get(name)
                if x is None:
                    raise RuntimeError(
                        f"chunked piece: symbol {sid!r} binds from {tid}::dim_{dim}, which this "
                        f"piece is not fed")
                cache[sid] = int(x.shape[dim])
            return cache[sid]
        return value

    def run(self, feed: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if self._passes is None:
            self._build()
        if not self._loaded:
            raise RuntimeError("chunked piece: run before load_weights")
        plan = self._plan
        value = self._symbols(feed)
        rows = value(self._sym)
        total, rest = divmod(rows, self._unit)
        if rest:
            raise RuntimeError(f"chunked piece: {self._sym} = {rows} is not a whole number of its units of "
                               f"{self._unit} — the graph's own views of it in units would not hold either")
        count = -(-total // self._slice)
        inner = {t: (lay.inner.evaluate(self._axis.scaled(value)) if lay.inner is not None else None)
                 for t, lay in plan.layouts.items()}
        # a group norm over the axis: its statistics over its WHOLE input, once, before any slice (the first of the two
        # passes of an exact tiled norm; the second is each slice normalised with them)
        self._norm_stats = {}
        by_tid = {self._tid(n): n for n in feed}
        for uid, src in plan.norms.items():
            if src not in by_tid:
                raise RuntimeError(f"chunked piece: the norm {uid} reads {src}, which this stretch is not fed whole")
            op = self._axis.ops.get(uid) or {}
            scalars = [a.get("value") for a in ((op.get("attributes") or {}).get("args") or [])
                       if a.get("type") == "scalar"]
            group, eps = int(scalars[-2]), float(scalars[-1])
            self._norm_stats[uid] = norm_statistics(feed[by_tid[src]], group, eps)
        # what is sliced on the way in: the stretch's inputs and the carriers of its symbols
        sliced_in = set(plan.inputs) | {t for t in plan.layouts if t not in plan.outputs}
        done: Dict[str, Any] = {}          # tid -> whole value (contraction sums, assembled outputs)
        nbx_path, component = self._nbx
        for cp, ex in zip(plan.passes, self._passes):
            names = ex._dag.get("segment_input_names") or []
            ex.load_weights(nbx_path, component)
            try:
                while self._run_pass(cp, ex, names, feed, done, total, count, inner, sliced_in,
                                     args, kwargs):
                    for tid in cp.targets:        # pinned at the accumulator's width: the pass again
                        done.pop(tid, None)
            finally:
                ex.unload_weights()
        self._last_run = {t: done[t] for t in plan.outputs if t in done}
        self._norm_stats = {}
        meta = self._dag.get("tensors") or {}
        result = {}
        for tid in self._dag.get("output_tensor_ids") or []:
            if tid not in done:
                raise RuntimeError(f"chunked piece: the stretch did not make its output {tid!r}")
            result[output_key(meta.get(tid), tid)] = done[tid]
        return result

    def _run_pass(self, cp: Any, ex: Any, names: Sequence[str], feed: Dict[str, Any],
                  done: Dict[str, Any], total: int, count: int, inner: Dict[str, Any],
                  sliced_in: Set[str], args: Any, kwargs: Any) -> bool:
        """One pass over its slices, its targets written into `done`. True when the pass must run
        again: a contraction's first partial came narrower than the engine's accumulator, and its
        producer is now pinned at the accumulator's width (`GraphExecutor.pin_fp32_ops`) — a
        partial rounded once per slice would sum to what the whole op, rounding once, does not."""
        plan = self._plan
        m = plan.margin if cp.chunked else 0
        # each slice READS its rows widened by the stretch's halo, clipped at the axis' true ends
        for start, length, in_start, in_len in (slice_reads(total, count, m, plan.cyclic) if cp.chunked
                                                else [(0, total, 0, total)]):
            sub_feed = {}
            for n in names:
                tid = self._tid(n)
                if n in done:
                    v = done[n]
                elif tid in done:
                    v = done[tid]
                elif n in feed:
                    v = feed[n]
                else:
                    raise RuntimeError(f"chunked piece: pass input {n!r} is neither fed nor made")
                lay = plan.layouts.get(tid)
                if cp.chunked and lay is not None and tid in sliced_in:
                    v = (take_slice_cyclic if plan.cyclic else take_slice)(
                        v, lay, total, inner[tid], in_start, in_len)
                # an input kept on the host (`sliced_input_names`) comes to the pass's device a slice at a time
                sub_feed[n] = to_device(v, getattr(ex, "device", None))
            out = ex.run(sub_feed, *args, **kwargs) or {}
            meta = ex._dag.get("tensors") or {}
            for tid in cp.targets:
                key = output_key(meta.get(tid), tid)
                if key not in out:
                    raise RuntimeError(f"chunked piece: pass did not return {key!r}")
                part = out[key]
                lay = plan.layouts.get(tid)
                if tid in plan.contractions and cp.chunked:
                    # summed in the engine's accumulator, stored once in the op's own dtype
                    if tid not in self._sums:
                        self._sums[tid] = (ex.accumulation_dtype(part.dtype), part.dtype)
                        pins = self._pins_of(cp, plan, ex._dag)
                        if pins:
                            ex.pin_fp32_ops(pins)
                            return True
                    done[tid] = accumulate(done.get(tid), part, self._sums[tid][0])
                elif tid in plan.contractions:
                    done[tid] = own(part)
                elif cp.chunked and lay is not None:
                    if m:      # keep the slice's own rows: the margin's rows read padding, not the image
                        part = take_slice(part, lay, in_len, inner[tid], start - in_start, length)
                    if start == 0:
                        done[tid] = part.new_empty(whole_shape(part.shape, lay, total, length))
                    put_slice(done[tid], part, lay, total, inner[tid], start, length)
                else:
                    done[tid] = own(part)
        for tid in cp.targets:
            if tid in plan.contractions and cp.chunked:
                done[tid] = store(done[tid], self._sums[tid][1])
        return False
