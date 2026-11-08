"""A run that meets a device out-of-memory at a component re-plans once and re-enters its flow at that component (the
owner's rule 7; the Dell + the Mac, 2026-11-08 18:51): mochi-1-preview's first compiled run on an M4 Pro met an MPS
out-of-memory in its VAE after the streamed transformer, recorded its growth (10 033 MB) and exited 1; only the second
run, planned with the record, was green. The first run now records, releases the cache, asks Prism again and calls the
VAE again, the transformer's latents kept (`RuntimeExecutor._execute_component`). The entry point arms it with its
engine's facts (`cli/commands/run.py` `_arm_replan`); same plan or Prism's refusal: the run's named refusal. No torch (R33).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

#: the vendor file's `memory.` key naming the message of an allocator that raises its out-of-memory untyped
UNTYPED_OOM_KEY = "compiled_untyped_oom_message"

# what of a plan decides how a component executes: its allocation, and the plan's per-component entries
_ALLOCATION = ("devices", "dtype", "sharded", "shard_map", "strategy", "resident_weight_mb", "deferred_drain_cliff_bytes",
               "attention_scores_budget_bytes", "expert_cache_mb", "expert_files", "expert_cards")
_PER_COMPONENT = ("component_tiling", "component_stream", "runtime_op_tiling", "batch_split_components",
                  "layer_stream_plan", "layer_stream_graph", "layer_stream_moe", "layer_stream_chunks",
                  "layer_stream_resident")
_MEMBERSHIP = ("transient_components", "layer_stream_swapped", "cfg_split_components")
# where a flow attaches state to a component's executor between its calls (a KV cache, a decoder's interceptors)
_HELD = ("_op_interceptors", "_op_uid_interceptors", "_pending_triton_interceptors")


@dataclass
class Replan:
    """What the run's engine gives its executor to recover from a device out-of-memory at a component."""
    is_device_oom: Callable[[BaseException], bool]      # this engine's device out-of-memory, anywhere in the chain
    record_growth: Callable[[], Optional[float]]        # records the run's measured growth, its MB (None: not measured)
    solve: Callable[[], Any]                            # Prism's plan with the facts in force now; raises its refusal


def in_chain(exc: Optional[BaseException], test: Callable[[BaseException], bool]) -> bool:
    """`exc` or one it was raised from passes `test` (both sequences raise "Failed at op" `from` the allocator's error)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if test(exc):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def _fact(value):
    if isinstance(value, dict):
        return sorted((str(k), _fact(v)) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_fact(v) for v in value]
    if hasattr(value, "__dict__"):
        return _fact(vars(value))
    return value


def signature(plan: Any, component: str) -> str:
    """How `plan` executes `component`: its strategy, the component's allocation and the plan's entries for it."""
    alloc = (getattr(plan, "components", None) or {}).get(component)
    parts = [getattr(plan, "strategy", None), getattr(plan, "loading_mode", None), getattr(plan, "kv_cache_plan", None)]
    parts += [getattr(alloc, k, None) for k in _ALLOCATION]
    parts += [(getattr(plan, k, None) or {}).get(component) for k in _PER_COMPONENT]
    parts += [component in (getattr(plan, k, None) or ()) for k in _MEMBERSHIP]
    return repr(_fact(parts))


def held_state(executor, plan, component: str) -> list:
    """The interceptor registries holding state on `component`'s executor between its calls — a call that failed part
    way through them cannot be run again, nor the executor rebuilt; the plan's own op tiling holds none."""
    g = getattr(executor, "graph_executor", None) or executor
    own = "_op_uid_interceptors" if (getattr(plan, "runtime_op_tiling", None) or {}).get(component) else None
    return [k for k in _HELD if k != own and getattr(g, k, None)]
