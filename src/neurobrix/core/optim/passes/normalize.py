"""The graph as the branch's sequence will actually run it — for anyone who must plan on it.

Prism partitions the RAW container graph. Each sequence then transforms that graph IN PLACE
before the executor runs it. So a `layer_streaming` segment boundary names an op id that the
fusions have already rewritten away, and the strategy refuses at execution:

    layer_streaming: the plan's segment boundaries are not in 'model's graph
    (2 of 4 op ids absent, e.g. 'aten.silu::843')

Measured 2026-09-22 on this rack, under the Mac's own unified profile (the census shadow
reads the PROFILE, not the card, so another machine's plan is reproducible here):
`_fuse_swiglu_ops` alone takes DeepSeek-Coder-V2-Lite from 11 722 ops to 10 745 in the
execution order, folding 977 silu+mul pairs into `custom.swiglu_fused` — and both
`aten.silu::843` and `aten.silu::890`, the two boundary ids the Mac measured at two different
rungs, are among what it removes.

Re-partitioning at execution is design-rejected (`solver.py`, the `Plan` dataclass): a
boundary recomputed on a different graph is not the boundary the budget was accepted under.
So the graph is normalized BEFORE the partition instead, and both sides then speak about the
same ops.

WHAT IS AND IS NOT APPLIED HERE
-------------------------------
Only the rewrites that are **pure over the graph** — they take `(tensors, ops, order)` and
touch nothing else. Verified by reading them: `_eliminate_dead_causal_mask_ops`,
`_fuse_swiglu_ops` and `_fuse_rope_ops` carry no `self.*` reference at all.

The others are deliberately NOT applied, and the reason is not tidiness:

* `_eliminate_weight_transpose_ops` calls `_pretranspose_weights` — it MUTATES WEIGHTS. At
  plan time there are no weights (a census shadow never loads any), so running it here would
  either fail or, worse, quietly plan against a graph the executor cannot reproduce.
* `_eliminate_detach_ops`, `_extract_const_fold_partition`, `_apply_cse_plan` and
  `_lower_fusion_vertical` read `self.dag` or planner annotations that exist only once a
  sequence is built.

WHICH GRAPH THE STRATEGY CUTS (2026-09-24). The header above assumed the executor's graph is
rewritten before `layer_streaming` reads it. A streamed component's base executor holds no
weights and never compiles, so its graph is the one loaded — and the Mac's 26 "boundaries not in
the graph" refusals were exactly the boundaries the partition had put ON a fused op
(`custom.swiglu_fused::15` and `::31` of granite-speech at 4 096 MB: 2 of 16, both absent, the
base graph holding no fused op at all). So the strategy now cuts `normalize_for_branch(base
graph)` — this function, both sides — and each piece's own sequence compiles what it receives.

ONLY `triton` REWRITES. The rewrites below are performed by `TritonSequence.compile`, which only
mode `triton` builds. `triton_sequential` executes the graph as loaded, op by op — the kernel
oracle — so its plan must cut that graph, not a fused one it never runs.

Two OPT-IN load-time passes rewrite the execution order and are NOT mirrored here:
`NBX_OPTIM_DEAD_CODE` and `NBX_OPTIM_FUSION_HORIZONTAL` (`GraphExecutor`). With either on, the
graph the strategy cuts differs from the one planned and `layer_streaming` refuses at its
fingerprint check — a refusal, not a wrong cut.

That is a real limit and it is stated rather than hidden: a boundary removed by one of those
would still go absent. The measured cases all fall to the swiglu fusion, and the gate asserts
the INVARIANT — every boundary present in the graph that will run — so if one of the others
ever takes a boundary, the gate says so instead of passing.

The env gates of the three rewrites below are honoured exactly as the sequence honours them: a fusion the run will not
perform must not be performed here either, or the two disagree in the other direction.
"""
from __future__ import annotations

import copy
import hashlib
import os
from typing import Any, Dict, Optional


def _triton_pure_passes():
    """The triton branch's graph-pure rewrites, in the order `sequence.py` applies them.

    Imported lazily: `core/prism` plans for both engines, and a module-level import of the
    Triton branch here would put it on the compiled path's import graph for no reason.
    """
    from neurobrix.triton.sequence import TritonSequence as _TS
    return [
        # Phase -0.4 — the flash kernel handles causality itself, so the mask chain is dead.
        ("dead_causal_mask", _TS._eliminate_dead_causal_mask_ops, None),
        # Phase -0.3 — silu + mul -> custom::swiglu_fused. THIS is the one that takes the
        # boundaries in every case measured so far.
        ("swiglu", _TS._fuse_swiglu_ops, "NBX_DISABLE_SWIGLU_FUSION"),
        # Phase -0.2 — the rotate_half chain -> custom::rope_fused.
        ("rope", _TS._fuse_rope_ops, "NBX_DISABLE_ROPE_FUSION"),
    ]


def _sequence_rewrites_graph(mode: str) -> bool:
    """The mode whose sequence performs the rewrites of `_triton_pure_passes` (`TritonSequence.
    compile`). `triton_sequential` runs the loaded graph op by op and performs none of them."""
    return str(mode or "").lower() == "triton"


def normalize_for_branch(graph: Dict[str, Any], mode: str, family: str = "",
                         declared_moe: Optional[bool] = None) -> Dict[str, Any]:
    """A COPY of `graph` rewritten the way `mode`'s sequence will rewrite it.

    Never mutates the caller's graph: the container's graph is read by other components and
    by the census, and a plan-time rewrite leaking into it would be a far worse defect than
    the one this exists to fix.

    `family` enables the MoE fusion, which is shared by both modes; without it that rewrite
    is skipped and a MoE model's boundaries will still move. `declared_moe` is the declaration
    a flow makes for a MoE LM packaged under another family (`GraphExecutor.set_moe_config`):
    None when there is none, else the `norm_topk_prob` the fused op carries. The runtime fuses
    such an LM, so a plan for it must cut the fused graph too — and the pieces must be cut from
    it, whatever order the flow declares in (the vlm flows declare AFTER the pieces exist).

    Any mode but `triton` (compiled, sequential, triton_sequential) returns a copy with the MoE fusion applied and nothing else — the compiled branch's two
    transforms both need weights or a built sequence, so there is nothing pure to apply.
    """
    out = copy.deepcopy(graph)

    # FIRST, and in every mode: the MoE fusion. It is not a branch rewrite — the doctrine is
    # that it runs in all modes — and it is by far the largest: measured on
    # DeepSeek-Coder-V2-Lite, 11 722 ops -> 2 678, folding the expert matmuls into 26
    # `custom::moe_fused` ops. `aten.mm::2648`, the boundary the executor could not find once
    # the swiglu fusion was accounted for, is among what it removes. The `Plan` dataclass has
    # named this rewrite as the reason boundaries move since before this pass existed.
    if family:
        try:
            from neurobrix.core.runtime.graph.moe_fusion import detect_and_fuse_moe
            if declared_moe is None:
                detect_and_fuse_moe(out, family)
            else:
                detect_and_fuse_moe(out, family, norm_topk_prob=bool(declared_moe), declared=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"normalize_for_branch: the MoE fusion failed on this graph "
                f"({type(exc).__name__}: {str(exc)[:160]}).\n"
                f"  Planning on the un-fused graph would hand the executor boundaries it "
                f"cannot find — a refusal rather than a silent fallback."
            ) from exc

    if not _sequence_rewrites_graph(mode):
        return out

    tensors = out.get("tensors")
    ops = out.get("ops")
    order = out.get("execution_order")
    if not isinstance(tensors, dict) or not isinstance(ops, dict) or not isinstance(order, list):
        return out                      # not a shape we can normalize; the caller plans as before

    for name, fn, gate in _triton_pure_passes():
        if gate and os.environ.get(gate) == "1":
            continue                    # the run will not do it, so neither does the plan
        try:
            fn(None, tensors, ops, order)      # unbound: these take no `self`
        except Exception as exc:               # noqa: BLE001
            raise RuntimeError(
                f"normalize_for_branch({mode!r}): the {name!r} rewrite failed on this graph "
                f"({type(exc).__name__}: {str(exc)[:160]}).\n"
                f"  Planning on an un-normalized graph would hand the executor boundaries it "
                f"cannot find, which is the defect this pass exists to remove — so this is a "
                f"refusal rather than a silent fallback to the raw graph."
            ) from exc
    out["execution_order"] = order
    return out


def boundaries_present(graph: Dict[str, Any], bounds) -> list:
    """The boundary op ids in `bounds` that the graph's execution order does NOT contain.

    The invariant a gate should assert. Deliberately NOT a check for one op id: the boundary
    MOVES with the memory rung — the Mac measured `aten.silu::890` at its profile budget and
    `aten.silu::843` at 12 288 MB for the same model — so a cell pinned to an id would pass at
    another rung and prove nothing.
    """
    order = set(graph.get("execution_order") or [])
    return [b for pair in (bounds or []) for b in pair if b not in order]


def graph_fingerprint(graph: Dict[str, Any]) -> str:
    """The identity of the graph a plan was cut on: its op count and a sha256 of its execution
    order. Prism records it for every streamed component and `layer_streaming` refuses to cut a
    graph whose fingerprint differs — the door behind which a boundary check cannot pass by the
    luck of where the cut fell (register 106: a MoE fusion the plan did not make left the few
    boundaries of a 2-piece plan on untouched ops, and the presence check passed)."""
    order = list(graph.get("execution_order") or [])
    return f"{len(order)}:" + hashlib.sha256("\n".join(order).encode()).hexdigest()[:16]
