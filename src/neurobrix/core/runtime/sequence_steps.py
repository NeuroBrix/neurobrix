"""The steps the compiled sequence (core/runtime/graph/compiled_sequence.CompiledSequence) and the Triton one
(triton/sequence.TritonSequence) ran alike, body for body — written once. Torch-free at import: the Triton sequence
inherits it (R33); the arguments each compiles are its own types (`_compile_arg`).
"""
from __future__ import annotations

from typing import Any, Dict, List


class SequenceSteps:
    """What both sequences share; each inherits it."""

    def _extract_const_fold_partition(
        self,
        tensors: Dict[str, Any],
        execution_order: List[str],
    ) -> None:
        """Take the const_fold partition out of the hot sequence.

        The engine-neutral planner (core/optim/passes/const_fold.py)
        annotated dag["_optim_const_fold"] = {op_uids, frontier_tids,
        releasable_tids} at the load hook. Here: partition ops leave
        execution_order, and every frontier tensor is stamped
        `folded_const` so slot categorization treats it as a weight
        (its producer no longer runs — the value arrives at bind time
        from _execute_const_fold). Idempotent on recompile: the stamps
        and the plan survive in the shared DAG, removal is a no-op the
        second time (same contract as the pretranspose stamps).
        """
        plan = self.dag.get("_optim_const_fold")
        if not plan:
            return
        self._const_fold_plan = plan
        for tid in plan["frontier_tids"]:
            meta = tensors.setdefault(tid, {})
            meta["folded_const"] = True
        uids = set(plan["op_uids"])
        new_order = [u for u in execution_order if u not in uids]
        if len(new_order) != len(execution_order):
            execution_order.clear()
            execution_order.extend(new_order)

    def _compile_kwargs(self, kwargs: Dict[str, Any], tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-compile keyword arguments to typed objects."""
        return {k: self._compile_arg(v, tensors) for k, v in kwargs.items()}
