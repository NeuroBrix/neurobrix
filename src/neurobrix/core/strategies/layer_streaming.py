"""Execute one component a segment at a time, holding one segment's weights.

The rung below every rung that keeps a component whole. `lazy_sequential`
reduces the requirement from sum(components) to max(component);
`cpu_streaming` does the same on the host. Neither helps a model that is ONE
component larger than the budget — measured 2026-09-09,
`DeepSeek-Coder-V2-Lite-Instruct` is a single `model` component of 17777 MB of
live weights. This rung cuts the component itself.

Prism decides WHERE to cut (`prism/layer_partition.py`, from dataflow, never
from a module name) and this executes what it decided. The segments are
carried in the plan rather than recomputed here, because the executor's graph
may have been transformed since Prism read it and a boundary recomputed on a
different graph is not the boundary the budget was accepted under.

ZERO SEMANTIC, and no vendor: it names no device, no backend and no brand. It
runs wherever the allocation points.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from neurobrix.core.strategies.base import ExecutionStrategy


class LayerStreamingStrategy(ExecutionStrategy):
    """One segment resident at a time, inside a single component."""

    #: One segment's weights resident at a time, loaded and released around
    #: each segment. That is managing its own residency, and it is why this
    #: strategy needs the same runtime hook zero3 uses.
    manages_weight_residency = True

    def __init__(self, context, strategy_name: str):
        super().__init__(context, strategy_name)
        self._segment_executors: Dict[str, List[Any]] = {}
        self._installed: set = set()

    # -- segment executors -------------------------------------------------

    def _segments_for(self, component_name: str) -> Optional[List[List[str]]]:
        raw = (getattr(self.context, "layer_segments", None) or {}).get(component_name)
        if not raw:
            return None
        return [list(pair) for pair in raw]

    def _nbx_path(self, component_name: str) -> str:
        """Delegates to the strategy brick — one resolver for every strategy."""
        return self.resolve_artifact_path(component_name)


    def _build_segment_executors(self, component_name: str) -> List[Any]:
        """One executor per segment, each carrying that segment's graph.

        `ExecutorFactory.create` takes a dag directly, so a segment goes
        through exactly the path a component does — same binding, same
        dispatch. And because `GraphExecutor.consumed_weight_names` reads the
        executor's OWN dag, a segment executor asks the loader for precisely
        its own weights with nothing further to arrange.
        """
        from neurobrix.core.prism.layer_partition import (
            LayerPartitioner, build_segment_graph, Segment)
        base = self.context.component_executors.get(component_name)
        if base is None:
            raise RuntimeError(
                f"ZERO FALLBACK: no executor for '{component_name}'")
        dag = getattr(base, "_dag", None)
        if not isinstance(dag, dict):
            raise RuntimeError(
                f"layer_streaming: '{component_name}' has no graph to cut")

        bounds = self._segments_for(component_name)
        if not bounds:
            raise RuntimeError(
                f"layer_streaming was chosen for '{component_name}' but the "
                f"plan carries no segments for it")

        order_index = {op_uid: i for i, op_uid
                       in enumerate(dag.get("execution_order") or [])}
        missing = [b for pair in bounds for b in pair if b not in order_index]
        if missing:
            # The graph moved under the plan. Refusing is the only honest
            # answer: executing different boundaries than the ones the budget
            # was accepted under is exactly the failure this rung exists to
            # avoid.
            raise RuntimeError(
                f"layer_streaming: the plan's segment boundaries are not in "
                f"'{component_name}'s graph ({len(missing)} of "
                f"{2 * len(bounds)} op ids absent, e.g. {missing[0]!r}). The "
                f"graph was transformed after Prism read it; re-plan rather "
                f"than execute boundaries that no longer mean what they meant.")

        nbx_path = self._nbx_path(component_name)

        # A segment executor is the COMPONENT's executor with a different
        # graph. Built from the base executor's own resolved configuration
        # rather than through the factory, which wants a ComponentAllocation
        # the strategy context does not carry — and because copying the
        # resolved values is the only way to guarantee a segment runs under
        # exactly the contract its component runs under: same family, same
        # vendor and arch, same device, same dtype, same mode.
        executors = []
        for index, (first_op, last_op) in enumerate(bounds):
            seg = Segment(index=index, first_op=first_op, last_op=last_op,
                          op_count=order_index[last_op] - order_index[first_op] + 1)
            sub = build_segment_graph(dag, seg, order_index)
            seg_exec = type(base)(
                family=getattr(base, "family", None),
                vendor=getattr(base, "vendor", None),
                arch=getattr(base, "arch", None),
                device=getattr(base, "device", None),
                dtype=getattr(base, "dtype", None),
                mode=getattr(base, "mode", "compiled"),
            )
            cache_path = getattr(base, "_cache_path", None)
            if cache_path is not None:
                seg_exec._cache_path = cache_path
            seg_exec.load_graph_from_dict(sub)
            seg_exec._component_name = component_name
            executors.append(seg_exec)
        return executors

    # -- the strategy API --------------------------------------------------

    def execute_component(
        self,
        component_name: str,
        phase: str = "loop",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the component one segment at a time.

        Values flow by NAME: a segment declares `segment_input_names`, which
        are the tensor ids the executor will look up after stripping the
        `input::` prefix, and its outputs come back keyed by tensor id. So a
        segment's outputs feed the next segment's inputs with no renaming.
        """
        # A component is streamed when the PLAN carries segments for it.
        # Not a device-string prefix: several places parse an allocation by
        # splitting on ":" and taking the index, and a three-part device
        # reached `int('mps')`.
        import os as _os
        if _os.environ.get("NBX_LAYER_DIAG") == "1":
            print(f"[LAYERDIAG] execute_component({component_name}) "
                  f"segments={self._segments_for(component_name)!r} "
                  f"ctx_keys={list((getattr(self.context,'layer_segments',None) or {}).keys())}",
                  flush=True)
        if not self._segments_for(component_name):
            executor = self.context.component_executors.get(component_name)
            if executor is None:
                raise RuntimeError(
                    f"ZERO FALLBACK: no executor for '{component_name}'")
            self.load_weights(component_name)
            return executor.run(inputs or {})

        if component_name not in self._segment_executors:
            self._segment_executors[component_name] = \
                self._build_segment_executors(component_name)
        executors = self._segment_executors[component_name]

        values: Dict[str, Any] = dict(inputs or {})
        last: Dict[str, Any] = {}
        nbx_path = self._nbx_path(component_name)

        for executor in executors:
            sub = executor._dag
            needed = sub.get("segment_input_names") or []
            missing = [n for n in needed if n not in values]
            if missing:
                raise RuntimeError(
                    f"layer_streaming: segment {sub.get('segment_index')} of "
                    f"'{component_name}' needs {missing[:3]} and nothing "
                    f"before it produced them")
            seg_inputs = {n: values[n] for n in needed}

            # This segment's weights, and only this segment's: the executor
            # asks its own dag what it consumes.
            executor.load_weights(nbx_path, component_name)
            try:
                out = executor.run(seg_inputs) or {}
            finally:
                # Released before the next segment is loaded, which is the
                # whole point: one segment resident at a time.
                executor.unload_weights()
            values.update(out)
            last = out

        return last

    def install_for_executor(self, component_name: str, executor) -> None:
        """Make this component's own executor run segment by segment.

        Called by `RuntimeExecutor._ensure_weights_loaded` for any strategy
        that declares `manages_weight_residency`. It exists because an
        autoregressive flow never enters `execute_component` at all — the
        flow handler calls `executor.run` directly, and `executor.py` says so
        itself. zero3 uses this same hook for the same reason.

        So the segmenting is installed ON the executor: `run` is replaced by
        one that walks the segments, holding one segment's weights at a time.
        Idempotent, and a component with no segments in the plan is left
        exactly as it was.
        """
        if not self._segments_for(component_name):
            return
        if component_name in self._installed:
            return
        self._installed.add(component_name)

        segments = self._build_segment_executors(component_name)
        nbx_path = self._nbx_path(component_name)

        def segmented_run(inputs=None, *args, **kwargs):
            values = dict(inputs or {})
            last = {}
            for seg_exec in segments:
                sub = seg_exec._dag
                needed = sub.get("segment_input_names") or []
                missing = [n for n in needed if n not in values]
                if missing:
                    raise RuntimeError(
                        f"layer_streaming: segment {sub.get('segment_index')} "
                        f"of '{component_name}' needs {missing[:3]} and "
                        f"nothing before it produced them")
                seg_exec.load_weights(nbx_path, component_name)
                try:
                    out = seg_exec.run({n: values[n] for n in needed},
                                       *args, **kwargs) or {}
                finally:
                    # Released before the next segment loads. This is the
                    # residency the plan was budgeted against.
                    seg_exec.unload_weights()
                import os as _os
                if _os.environ.get("NBX_LAYER_DIAG") == "1":
                    print(f"   [LAYERDIAG] seg{sub.get('segment_index')} "
                          f"type={type(out).__name__} repr={repr(out)[:160]}",
                          flush=True)
                    print(f"   [LAYERDIAG] seg{sub.get('segment_index')} "
                          f"déclare {len(sub.get('output_tensor_ids') or [])} sorties, "
                          f"rend {len(out)} clés; "
                          f"rendues={sorted(out)[:4]}; "
                          f"ctx_out={sorted(getattr(getattr(seg_exec,'_ctx',None),'output_tensor_ids',[]) or [])[:4]}",
                          flush=True)
                values.update(out)
                last = out
            return last

        executor.run = segmented_run
        print(f"   [layer_streaming] '{component_name}': {len(segments)} "
              f"segments, one resident at a time", flush=True)

    def prepare_inputs(self, component_name: str,
                       inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.transfer_dict(inputs, self.context.get_device(component_name))

    def handle_outputs(self, component_name: str, outputs: Dict[str, Any],
                       target_device: Optional[str] = None) -> Dict[str, Any]:
        if target_device is None:
            return outputs
        return self.transfer_dict(outputs, target_device)
