"""Which components' weights are on the device while a component runs — the flow's release schedule, read by the flow
that releases (`core/flow/iterative_process`) and by Prism that prices (`PrismSolver._try_op_level_tiling`): one rule.

An iterative flow outside serve mode runs its stages in order and releases as it goes: each pre-loop component is
released once it has run (forced, even eager: it runs once per request), every loop and pre-loop component is released
before the post-loop stage, and the post-loop components stay until the request ends. So while a component runs, the
resident others are: none for a pre-loop one; the other loop components for a loop one; the post-loop components before
it for a post-loop one. Measured (Allegro, compiled, 2026-10-21): text encoder loaded -> released -> transformer loaded
-> released -> VAE loaded; Prism priced the three as one resident set, and refused Sana 4K's VAE an op-level tiling
against the room the text encoder's and transformer's weights would have left, which they never take.

Serve mode (`persistent_mode`) releases nothing, and a flow with no stated schedule is not known to release: None, the
caller counts every component — the rule before this one. Standard library only (the Triton branch imports it, R33)."""
from __future__ import annotations

from typing import Any, FrozenSet, List, Mapping, Optional


def _stages(topology: Mapping[str, Any]):
    flow = (topology or {}).get("flow") or {}
    if flow.get("type") != "iterative_process":
        return None
    pre = list(flow.get("pre_loop") or [])
    loop = list((flow.get("loop") or {}).get("components") or [])
    post = list(flow.get("post_loop") or [])
    return pre, loop, post


def resident_beside(topology: Mapping[str, Any], component: str, persistent: bool) -> Optional[FrozenSet[str]]:
    """The OTHER components whose weights are on the device while `component` runs, or None when the flow states no
    release schedule (serve mode, a flow this module does not read): the caller then counts every component."""
    stages = None if persistent else _stages(topology)
    if stages is None:
        return None
    pre, loop, post = stages
    if component in pre:
        return frozenset()
    if component in loop:
        return frozenset(c for c in loop if c != component)
    if component in post:
        return frozenset(post[:post.index(component)])
    return None


def released_after_run(topology: Mapping[str, Any], component: str) -> bool:
    """Whether the flow releases `component` as soon as it has run: a pre-loop component (once per request)."""
    stages = _stages(topology)
    return stages is not None and component in stages[0]


def released_before_post_loop(topology: Mapping[str, Any], loop_components: List[str]) -> List[str]:
    """What the flow releases before its post-loop stage: the loop components, then the pre-loop ones."""
    stages = _stages(topology)
    pre = stages[0] if stages is not None else []
    return list(loop_components) + list(pre)
