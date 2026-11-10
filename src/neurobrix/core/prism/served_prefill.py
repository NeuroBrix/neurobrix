"""Chooses the rows a serving session's prefill chunk is priced at, on an engine that keeps no certified kernel times.

Why it exists. A served turn is prefilled in chunks of the rows the plan prices, and every chunk is one pass of the
whole graph: at 128 rows a 19 931-token turn of Qwen3-Coder-30B-A3B took 322 s on two V100-32GB (156 passes), at
2 048 rows 44 s (10 passes), the card's peak 160 MiB higher (measured 2026-11-10). The fewest passes the plan can hold
are the fastest turn.

Contract.
- In: `solve(input_config) -> ExecutionPlan`, the caller's own served solve (Prism decides every plan; this module
  only asks), and the session's `input_config` (a dataclass with `seq_len`, the rows a turn's prefill is priced at).
- Out: `(plan, input_config)` — the plan of the largest priced rows, doubling from the first plan's chunk up to the
  cache's reach, at which the plan HOLDS (`resident_signature` unchanged), and the config that plan was solved with:
  the caller keeps it, so a later re-plan prices the same rows.
- Unchanged, the first plan and the caller's config, when the plan has no KV cache, no chunk, or a chunk chosen from
  certified times (core/prism/prefill_chunk.py): that choice is not this module's.
- Refuses nothing: a size no plan fits (the solve's RuntimeError) or that moves a resident ends the raise at the size
  before it.
"""
import dataclasses
import logging
from typing import Any, Callable, Tuple

log = logging.getLogger(__name__)

#: the phrase by which a plan says its chunk was not chosen from certified times: `PrismSolver._prefill_chunk` builds
#: its reasons FROM this constant, so a rewording cannot turn the raise off
UNTIMED = "no certified kernel times"


def resident_signature(plan) -> tuple:
    """What of a served plan must not move when its prefill is priced at more rows: the strategy, each component's
    placement and what it keeps RESIDENT (its weights, the blocks whose experts each card holds), the cache's reach and
    the execution door. The experts' cache is not in it: it is the remainder of the card after the residents and the
    activations, so it is what a larger chunk's activations are taken from — until the solver would have to give up a
    resident block to keep it, which moves the signature."""
    kv = plan.kv_cache_plan
    return (plan.strategy, plan.loading_mode, None if kv is None else int(kv.max_cache_len),
            getattr(plan, "execution_door", None),
            tuple(sorted((n, a.strategy, tuple(a.devices), float(a.resident_weight_mb),
                          tuple(sorted((c, tuple(v["blocks"])) for c, v in (a.expert_cards or {}).items())))
                         for n, a in plan.components.items())))


def raise_while_the_plan_holds(solve: Callable[[Any], Any], input_config) -> Tuple[Any, Any]:
    """(plan, input config) of the largest priced rows at which the served plan holds — the module's contract."""
    plan = solve(input_config)
    kv = plan.kv_cache_plan
    if kv is None or not kv.prefill_chunk or UNTIMED not in str(kv.prefill_chunk_reason):
        return plan, input_config
    held, cfg, rows = resident_signature(plan), input_config, int(kv.prefill_chunk) * 2
    while rows <= int(kv.max_cache_len):
        trial_cfg = dataclasses.replace(input_config, seq_len=rows)
        try:
            trial = solve(trial_cfg)
        except RuntimeError as exc:
            log.info("Prism: the served prefill chunk stays at %d rows — no plan at %d rows (%s)",
                     rows // 2, rows, (str(exc).splitlines() or [""])[0])
            break
        if resident_signature(trial) != held:
            log.info("Prism: the served prefill chunk stays at %d rows — at %d the plan gives up a resident",
                     rows // 2, rows)
            break
        plan, cfg, rows = trial, trial_cfg, rows * 2
    return plan, cfg
