"""The prefill chunk C of a serving session, chosen by Prism from facts (docs/design/prism_kv.md §5bis).

A session's prefill runs in chunks of exactly C rows, the last completed with padding rows (to C, or — a served turn
on the Triton engines, `served_ladder` — to the smallest rung of the ladder that holds it), every chunk
attending over the cache's keys as whole pages: every chunk then meets census keys, and a turn served from the session
gives the bytes of the same turns prefilled together. C is chosen here, never written:

- the candidates are `core/state/price.prefill_chunk_candidates` (whole pages, at most the prefill rows the plan
  prices a session's activations at);
- each candidate's keys are the derived census's (`<cls>g.chunks.jsonl`, `kernels/census_table.read_chunks`): the
  graph at C rows and the attention over the cache at every whole-page key length up to the session's reach — derived
  from the graphs, never executed;
- a candidate is ELIGIBLE only when the certified directory serves every one of its keys on this memory class
  (`nbxtensor.ops.autotune_certified.served_certification`): an uncertified key would be tuned, or refused, at run time;
- its COST is the certified time of one turn of the priced prefill rows run in chunks of C from an empty cache,
  padding included: ceil(T / C) chunks of the graph's keys (each key's certified best time x its op count), plus the
  attention of chunk j at its key length (j + 1) x C;
- the cheapest eligible candidate is C, the smaller on a tie. None — one chunk of the turn's own rows — when no
  candidate is eligible, with the reason stated (the first candidate's first uncertified key).

Torch-free (both engines' planner)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

Served = Callable[[str, str, str], Optional[float]]     # (kernel, dtype, key text) -> certified best ms, or None


def chunk_cost(cand: Dict, turn_rows: int, served: Served) -> Tuple[Optional[float], Optional[str], int]:
    """(cost in ms, None) of one candidate record, or (None, its first uncertified key, how many are uncertified)."""
    C = int(cand["rows"])
    missing: List[str] = []
    graph_ms = 0.0
    for kernel, key, dtype, n_ops in cand["graph"]:
        ms = served(kernel, dtype, key)
        if ms is None:
            missing.append(f"{kernel.rsplit('.', 1)[-1]} {key}")
        else:
            graph_ms += float(ms) * int(n_ops)
    attn: Dict[int, float] = {}
    for tk, kernel, key, dtype, n_ops in cand["attention"]:
        ms = served(kernel, dtype, key)
        if ms is None:
            missing.append(f"{kernel.rsplit('.', 1)[-1]} {key} (key length {tk})")
        else:
            attn[int(tk)] = attn.get(int(tk), 0.0) + float(ms) * int(n_ops)
    if missing:
        return None, missing[0], len(missing)
    n = math.ceil(int(turn_rows) / C)
    return n * graph_ms + sum(attn.get((j + 1) * C, 0.0) for j in range(n)), None, 0


def choose(record: Optional[Dict], candidates: List[int], turn_rows: int, served: Served) -> Tuple[Optional[int], str]:
    """(C or None, the reason stated in the plan)."""
    if record is None:
        return None, "one chunk per turn: the derived census holds no prefill-chunk record for this model"
    by_rows = {int(c["rows"]): c for c in record.get("candidates") or []}
    costs, refused = [], []
    for C in candidates:
        cand = by_rows.get(int(C))
        if cand is None:
            refused.append(f"C={C}: not derived by the census")
            continue
        cost, first, n_missing = chunk_cost(cand, turn_rows, served)
        if cost is None:
            refused.append(f"C={C}: {n_missing} key(s) not certified, first {first}")
        else:
            costs.append((cost, C))
    if not costs:
        return None, ("one chunk per turn: no candidate's keys are all certified (" + "; ".join(refused) + ")")
    cost, C = min(costs)
    others = ", ".join(f"C={c} {t:.2f} ms" for t, c in sorted(costs) if c != C)
    return C, (f"C={C}: the cheapest certified turn of {turn_rows} rows, {cost:.2f} ms"
               + (f" (beside {others})" if others else "") + (f"; refused {'; '.join(refused)}" if refused else ""))


def certified_times(vendor: str, profile: str, memory_gb: Optional[int]) -> Served:
    """The certified best time (ms) of a key on this memory class, or None when the directory does not serve it."""
    from nbxtensor.ops import autotune_certified as AC
    generator = AC.running_generator()      # asked once: a plan prices thousands of keys

    def served(kernel: str, dtype: str, key: str) -> Optional[float]:
        cert = AC.served_certification(vendor, profile, kernel, dtype, key, memory_gb, generator=generator)
        if cert is None:
            return None
        ms = (cert.get("proof") or {}).get("best_ms")
        return float(ms) if ms is not None else None
    return served


def chunk_record(cache_dir: Path, vendor: str, profile: str, memory_gb: int, mode: str) -> Optional[Dict]:
    """This container's chunk record of the census table of (vendor, profile, memory class) for `mode` (the census's
    spelling, `triton` / `triton-sequential`), or None — none written, or written for another graph (a retrace)."""
    from neurobrix.kernels import census_table as T
    rec = T.read_chunks(T.table_path(vendor, profile, memory_gb)).get((Path(cache_dir).name, mode))
    if rec is None or rec.get("container") != T.container_sha(cache_dir):
        return None
    return rec


def served_rows(memory_class_gb: int) -> int:
    """The rows a served session prices one prefill pass at when no certified time chose its chunk, on a decoder card
    of `memory_class_gb`: `serving.session_prefill_rows_per_card_gb` of config/system.yml (its measurement is written
    beside it there) times the class. Refused by name when the setting is missing."""
    import yaml
    from neurobrix.core.workspace import SYSTEM_YML
    v = ((yaml.safe_load(SYSTEM_YML.read_text()) or {}).get("serving") or {}).get("session_prefill_rows_per_card_gb")
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise RuntimeError(f"ZERO FALLBACK: a served session's prefill needs `serving.session_prefill_rows_per_card_gb` "
                           f"in {SYSTEM_YML} (a positive number of rows), got {v!r}.")
    return v * int(memory_class_gb)


def served_ladder(page: int, reach: int, memory_class_gb: int, priced_rows: int = 0) -> Tuple[int, List[int], str]:
    """A SERVED turn's chunks on the Triton engines: (C, the rungs, the reason). The rungs are the ladder
    `price.prefill_chunk_candidates` up to `served_rows(memory_class_gb)`, never past the session's `reach`, nor past
    `priced_rows` when the plan is solved at stated rows (`served_prefill` descends the ladder until a plan holds).
    C is the top rung, the rows the plan prices one pass at over the session's keys — never one pass of rows the plan
    did not price: unchunked and priced at 128 rows, a served Qwen3-Coder-30B-A3B ran its 2 800- to 3 700-row agent
    turns in one pass and out of memory (2026-11-12). A turn runs whole chunks of C rows, then its last chunk
    completed to the SMALLEST rung that holds it: a 30-row turn is one 64-row pass, and every shape the session runs
    is a census row (a rung x a whole number of pages of keys), certifiable — a turn's own row count is not.
    The size is not chosen from certified kernel times: a pass has costs no kernel time holds (a mixture-of-experts
    model streams its experts at every pass — 128-row chunks took 51.6 s to the first token of a 1 124-token prompt
    against 10.2 s in one pass, same card, 2026-11-12), so the fewest priced passes are the fastest turn."""
    from neurobrix.core.state import price as P
    top = min(served_rows(memory_class_gb), int(reach), int(priced_rows) or int(reach))
    rungs = P.prefill_chunk_candidates(int(page), max(int(page), top))
    return rungs[-1], rungs, (
        f"chunks of {rungs[-1]} rows, the last on the smallest of the rungs {rungs} that holds it: the rows a served "
        f"turn's prefill is priced at on a {int(memory_class_gb)} GB card (serving.session_prefill_rows_per_card_gb, "
        f"whole pages of {page})" + (f", at most the {int(priced_rows)} rows this plan was asked to price"
                                     if priced_rows else ""))


def served_chunk(page: int, reach: int, memory_class_gb: int, input_config) -> Tuple[int, str, List[int]]:
    """(C, the reason, the rungs) of a served plan on the Triton engines — `served_ladder`, capped at the rows the
    plan is solved at when its request states them over cached keys (`InputConfig.seq_len` with `prefill_keys`:
    `served_prefill`'s priced solve)."""
    priced = int(getattr(input_config, "seq_len", 0) or 0) if getattr(input_config, "prefill_keys", 0) else 0
    C, rungs, why = served_ladder(page, reach, memory_class_gb, priced)
    return C, why, rungs
