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
    TokenAxis, accumulate, chunk_extents, dim_source, own, plan_region, put_slice, region_carriers,
    store, take_slice, whole_shape)
from neurobrix.core.prism.layer_partition import Segment, build_segment_graph
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
        if self._slice < 1:
            raise ValueError(f"chunked piece: slice {self._slice} of {self._sym}")
        self._axis = axis if axis is not None else TokenAxis(dag, self._sym)
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
        self._loaded = False
        self._nbx = None

    # -- the surface a piece shows its strategy ------------------------------------------------

    @property
    def _dag(self) -> Dict[str, Any]:
        return self._holder._dag

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
            if self._interceptors:
                ex.register_triton_interceptors(self._interceptors)
            for tid in self._protected:
                if tid in cp.targets:
                    ex.protect_tensor_id(tid)
            passes.append(ex)
        self._plan, self._passes = plan, passes

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
        total = value(self._sym)
        extents = chunk_extents(total, -(-total // self._slice))
        inner = {t: (lay.inner.evaluate(value) if lay.inner is not None else None)
                 for t, lay in plan.layouts.items()}
        # what is sliced on the way in: the stretch's inputs and the carriers of its symbols
        sliced_in = set(plan.inputs) | {t for t in plan.layouts if t not in plan.outputs}
        done: Dict[str, Any] = {}          # tid -> whole value (contraction sums, assembled outputs)
        dtypes: Dict[str, Any] = {}        # contraction tid -> (accumulator dtype, store dtype)
        nbx_path, component = self._nbx
        for cp, ex in zip(plan.passes, self._passes):
            names = ex._dag.get("segment_input_names") or []
            ex.load_weights(nbx_path, component)
            try:
                for start, length in (extents if cp.chunked else [(0, total)]):
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
                            v = take_slice(v, lay, total, inner[tid], start, length)
                        sub_feed[n] = v
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
                            if tid not in dtypes:
                                dtypes[tid] = (ex.accumulation_dtype(part.dtype), part.dtype)
                            done[tid] = accumulate(done.get(tid), part, dtypes[tid][0])
                        elif tid in plan.contractions:
                            done[tid] = own(part)
                        elif cp.chunked and lay is not None:
                            if start == 0:
                                done[tid] = part.new_empty(whole_shape(part.shape, lay, total, length))
                            put_slice(done[tid], part, lay, total, inner[tid], start, length)
                        else:
                            done[tid] = own(part)
                for tid in cp.targets:
                    if tid in dtypes:
                        done[tid] = store(done[tid], dtypes.pop(tid)[1])
            finally:
                ex.unload_weights()
        self._last_run = {t: done[t] for t in plan.outputs if t in done}
        meta = self._dag.get("tensors") or {}
        result = {}
        for tid in self._dag.get("output_tensor_ids") or []:
            if tid not in done:
                raise RuntimeError(f"chunked piece: the stretch did not make its output {tid!r}")
            result[output_key(meta.get(tid), tid)] = done[tid]
        return result
