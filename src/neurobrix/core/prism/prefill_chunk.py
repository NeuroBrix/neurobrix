"""The prefill chunk C of a serving session, chosen by Prism from facts (docs/design/prism_kv.md §5bis).

A session's prefill runs in chunks of exactly C rows, the last completed with padding rows, every chunk attending over
the cache's keys as whole pages: every chunk then meets the same kernel keys, and a turn served from the session gives
the bytes of the same turns prefilled together. C is chosen here, never written:

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


#: the phrase by which a plan says a turn's last chunk is NOT completed to the chunk's rows (`KVCachePlan.
#: prefill_chunk_pads` False): `served_untimed` builds its reason FROM this constant, and the solver reads it there
UNPADDED = "a turn's last chunk runs its own rows"


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


def served_untimed(why: str, page: int, reach: int, memory_class_gb: int) -> Tuple[int, str]:
    """A SERVED turn's chunk when no certified time chooses one (`choose` returned None for `why`): the whole pages of
    `served_rows(memory_class_gb)`, never more than the session's `reach` (at least one page), priced over the
    session's keys (`served_prefill`) — never one pass of rows the plan did not price. Unchunked and priced at 128
    rows, a served Qwen3-Coder-30B-A3B ran its 2 800- to 3 700-row agent turns in one pass and out of memory
    (2026-11-12). Its last chunk is NOT padded (`UNPADDED`): a turn that fits the chunk is one pass of its own rows,
    as before, now priced."""
    C = max(1, min(served_rows(memory_class_gb), int(reach)) // int(page)) * int(page)
    return C, (f"chunks of {C} rows: the rows a served turn's prefill is priced at on a {int(memory_class_gb)} GB card "
               f"(serving.session_prefill_rows_per_card_gb, whole pages of {page}); {why.split(': ', 1)[-1]}, so no "
               f"certified time chooses among them; {UNPADDED}")
