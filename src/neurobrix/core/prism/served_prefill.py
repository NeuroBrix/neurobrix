"""Chooses the rows a serving session's prefill chunk is priced at, on an engine that keeps no certified kernel times.

Why it exists. A served turn is prefilled in chunks of the rows the plan prices, and every chunk is one pass of the
whole graph: at 128 rows a 19 931-token turn of Qwen3-Coder-30B-A3B took 322 s on two V100-32GB (156 passes), at
2 048 rows 44 s (10 passes), the card's peak 160 MiB higher (measured 2026-11-10). The fewest passes the plan can hold
are the fastest turn.

Contract.
- In: `solve(input_config) -> ExecutionPlan`, the caller's own served solve (Prism decides every plan; this module
  only asks; `solve_served` builds it on a PrismSolver), and the session's `input_config` (a dataclass with `seq_len`, the rows a turn's prefill is priced at).
- Out: `(plan, input_config)` — the plan of the largest priced rows, doubling from the first plan's chunk up to the
  cache's reach, at which the plan HOLDS (`resident_signature` unchanged), and the config that plan was solved with:
  the caller keeps it, so a later re-plan prices the same rows. Every plan it returns for a kept or raised chunk is
  solved with the chunk's rows AND the cached keys it attends (`InputConfig.seq_len`, `prefill_keys` = the first
  plan's reach, `prefill_key_dtype`), so the reserve its attention over the whole cache spends — one figure,
  `price.session_chunk_attention_bytes`, which both engines' wrappers read from the plan's own chunk, reach and dtype
  — is in the price (`chunk_mask_bytes`); the plan the raise holds is the first one priced so.
- Unchanged, the first plan and the caller's config, when the plan has no KV cache or no chunk. A chunk this module
  does not raise (the Triton engines' ladder, core/prism/prefill_chunk.served_ladder): the plan solved once more at
  its top rung over the cached keys, then at each rung below while no plan holds, and that config — the ladder is
  not this module's, its price over the keys is.
- Refuses nothing of its own: a raise ends at the size before the one no plan fits (the solve's RuntimeError) or
  that moves a resident; a ladder whose bottom rung no plan holds lets that solve's refusal through.
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


def chunk_mask_bytes(input_config, mode: str) -> int:
    """The reserve a serving session's prefill chunk has for its attention over the cached keys
    (`price.session_chunk_attention_bytes`, the one figure both engines' wrappers spend), at the request's rows, cached
    keys and their dtype (`InputConfig.seq_len`, `prefill_keys`, `prefill_key_dtype`). 0 for a request that states no
    cached keys: a run's one pass attends its own rows, which the walk prices."""
    rows, keys = (int(getattr(input_config, f, 0) or 0) for f in ("seq_len", "prefill_keys"))
    if not rows or not keys:
        return 0
    from neurobrix.core.prism.host_footprint import engine_of
    from neurobrix.core.state.price import session_chunk_attention_bytes
    return session_chunk_attention_bytes(rows, keys, input_config.prefill_key_dtype, engine_of(mode))


def price_chunk_mask(component_memory, declared, input_config, mode: str, overhead: Callable) -> None:
    """Add `chunk_mask_bytes` to the priced activations of the flow's decoder — `declared[0]`, the solver's
    `_kv_declared` (None: the flow decodes nothing) — in `component_memory` {name: ComponentMemory}, and restate its
    overhead by the solver's own rule (`overhead(weight_bytes, activation_bytes, profiled)`)."""
    mem, more = (component_memory.get(declared[0]) if declared else None), chunk_mask_bytes(input_config, mode)
    if mem is None or not more:
        return
    mem.activation_bytes = int(mem.activation_bytes) + more
    mem.overhead_bytes = overhead(mem.weight_bytes, mem.activation_bytes, mem.activation_profiled)


def solve_served(container, profile, input_config, mode: str = "compiled") -> Tuple[Any, Any]:
    """(plan, input config) of a serving session — what the daemon calls: Prism's served plan, the prefill chunk
    raised while the plan holds."""
    from neurobrix.core.prism.solver import PrismSolver
    solver = PrismSolver()
    return raise_while_the_plan_holds(
        lambda cfg: solver.solve_smart(container, profile, cfg, serve_mode=True, mode=mode), input_config)


def raise_while_the_plan_holds(solve: Callable[[Any], Any], input_config) -> Tuple[Any, Any]:
    """(plan, input config) of the largest priced rows at which the served plan holds — the module's contract."""
    plan = solve(input_config)
    kv = plan.kv_cache_plan
    if kv is None or not kv.prefill_chunk:
        return plan, input_config
    keys, rows, held, cfg = int(kv.max_cache_len), int(kv.prefill_chunk), None, input_config
    if UNTIMED not in str(kv.prefill_chunk_reason):
        # a chunk not raised (the Triton engines' ladder, core/prism/prefill_chunk.served_ladder): the plan priced at
        # it over the cached keys it attends — unpriced, a served chunk's attention over a long session was never in
        # the plan. The ladder is descended while no plan holds its top rung; the refusal is the bottom rung's own.
        ladder = sorted({int(r) for r in (kv.prefill_chunk_rungs or [rows])}, reverse=True)
        for r in ladder:
            priced = dataclasses.replace(input_config, seq_len=r, prefill_keys=keys, prefill_key_dtype=str(kv.dtype))
            try:
                return solve(priced), priced
            except RuntimeError as exc:
                if r == ladder[-1]:
                    raise
                log.info("Prism: no served plan at chunks of %d rows (%s): the next rung down",
                         r, (str(exc).splitlines() or [""])[0])
    # from here every solve prices the chunk with the cached keys it attends (the session's reach): the first one at
    # the chunk the plan already has, and that priced plan is the one the raise must hold
    while rows <= keys:
        trial_cfg = dataclasses.replace(input_config, seq_len=rows, prefill_keys=keys, prefill_key_dtype=str(kv.dtype))
        try:
            trial = solve(trial_cfg)
        except RuntimeError as exc:
            log.info("Prism: the served prefill chunk stays at %d rows — no plan at %d rows (%s)",
                     rows // 2, rows, (str(exc).splitlines() or [""])[0])
            break
        if held is not None and resident_signature(trial) != held:
            log.info("Prism: the served prefill chunk stays at %d rows — at %d the plan gives up a resident",
                     rows // 2, rows)
            break
        held = resident_signature(trial) if held is None else held
        plan, cfg, rows = trial, trial_cfg, rows * 2
    return plan, cfg
