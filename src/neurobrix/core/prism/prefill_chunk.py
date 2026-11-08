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

    def served(kernel: str, dtype: str, key: str) -> Optional[float]:
        cert = AC.served_certification(vendor, profile, kernel, dtype, key, memory_gb)
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
