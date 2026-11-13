"""The KV cache's host tier, the engine's policy (reste_apres_064 point 10; the mechanism and the bookkeeping are
nbxtensor's: memory/kv_host_tier.py, kv_host_pages.py, the caches' `tier_*` protocol).

A serving daemon's cache holds one conversation at a time. When its positions are about to be dropped — another session
takes the cache (`park` before the clear), or a request's ids leave the held ones (`prepare` before the truncation) —
the whole pages it holds are parked in host memory, indexed by their ids. When a request's ids continue a parked
prefix, those pages are copied back into the sequence's own pages and the session holds them again: the existing match
then prefills only what follows. A page enters whole and its bytes are copied: a restored prefix is exactly a held one.

The budget is the machine's (`budget_mb`, the serving layer's from the profile and the user's optional cap); below one
page the tier is off and said. Host pages are page-locked on a discrete card and pageable on unified memory (`pinned`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from nbxtensor.memory.kv_host_tier import HostPagePool, PrefixIndex, host_tier_pages, sequence_pages


class KVTier:
    def __init__(self, budget_mb: float, pinned: bool, source: str, slice_positions: int):
        self.budget_bytes = int(budget_mb * 2 ** 20)
        self.slice_positions = int(slice_positions)     # host memory grows by one conversation of the plan at a time
        self.pinned = bool(pinned)
        self.source = source
        self.index: Optional[PrefixIndex] = None
        self.pool: Optional[HostPagePool] = None
        self.host = None
        self.parked_pages = 0
        self.restored_pages = 0

    def _ready(self, cache) -> bool:
        """The host pages' shape from the cache's own layout at first use, NOTHING allocated: the budget is a ceiling and
        the memory grows by slices as parks need it (`_host_page`). False when the budget holds no page."""
        if self.host is not None:
            return True
        from nbxtensor.memory.kv_host_pages import HostPages
        layers, kb, vb = cache.tier_layout()
        pages = host_tier_pages(self.budget_bytes, layers, kb, vb)
        if pages == 0:
            return False
        slice_pages = max(1, -(-self.slice_positions // cache.page_tokens))
        self.index, self.pool = PrefixIndex(cache.page_tokens), HostPagePool(0)
        self.host = HostPages(slice_pages, pages, layers, kb, vb, self.pinned)
        return True

    def _host_page(self, keep: Sequence) -> Optional[int]:
        """A free host page: a new slice while under the ceiling, else the least recently used parked leaf not in `keep`
        dropped (a trailing slice left empty by drops is freed)."""
        if self.pool.free_pages == 0:
            added = self.host.grow()
            if added:
                self.pool.grow(added)
            else:
                for leaf in self.index.host_victims(self.pool.capacity):
                    if leaf not in keep:
                        self.pool.give(self.index.drop(leaf))
                        break
        return self.pool.take() if self.pool.free_pages else None

    def host_bytes(self) -> int:
        """Host memory the parked pages' slices hold now."""
        return 0 if self.host is None else self.host.nbytes()

    def spill_budget(self) -> int:
        """The bytes a decoding sequence's mid-decode spill (nbxtensor memory/host_spill.py) may hold now: the same host
        budget, less what the parked pages hold — one budget for both, never twice it. A spilled sequence is not parked
        (below), so the tier grows only once its spill is cleared."""
        return max(0, self.budget_bytes - self.host_bytes())

    def park(self, ids: Sequence[int], cache) -> int:
        """The whole pages of `ids` the cache holds, parked (those not parked yet). Returns how many were copied. A
        sequence that spilled mid-decode is not parked (its oldest positions are not in the card's pages): 0."""
        if cache.spilled:
            return 0
        tpp = cache.page_tokens
        whole = min(len(ids), cache.tier_length()) // tpp
        if whole == 0 or not self._ready(cache):
            return 0
        known = self.index.match(list(ids[:whole * tpp]))
        pages = sequence_pages(cache.tables, cache.batch)
        todo: List[tuple] = []
        for i in range(len(known), whole):
            hp = self._host_page(known)
            if hp is None:
                break
            todo.append((i, hp))
        if not todo:
            return 0
        cache.tier_offload([pages[i] for i, _ in todo], self.host, [hp for _, hp in todo])
        chain = self.index.insert(list(ids[:(todo[-1][0] + 1) * tpp]), pages[:todo[-1][0] + 1])
        for i, hp in todo:
            self.index.offloaded(chain[i], hp)
        self.parked_pages += len(todo)
        return len(todo)

    def prepare(self, kv_session, cache, prompt_ids: Sequence[int]) -> int:
        """Before a session's prefill: the cache's held conversation parked where the request leaves it, then the
        parked continuation of `prompt_ids` restored. Returns the positions restored (0: none). A spilled held sequence
        is neither parked nor replaced by a parked one: the session's own prefix match keeps what it shares."""
        if cache.spilled:
            return 0
        held = list(kv_session.token_ids)
        common = 0
        while common < min(len(held), len(prompt_ids)) and held[common] == int(prompt_ids[common]):
            common += 1
        if len(held) > common:
            self.park(held, cache)
        if self.index is None:
            return 0
        tpp = cache.page_tokens
        chain = self.index.match(list(prompt_ids[:(len(prompt_ids) - 1) // tpp * tpp]))   # its last position prefills
        end = len(chain) * tpp
        if end <= common:
            return 0
        first = common // tpp
        cache.truncate(min(common, cache.tier_length()))
        cache.tier_resume(end)
        pages = sequence_pages(cache.tables, cache.batch)
        cache.tier_restore(self.host, [n.page for n in chain[first:]], pages[first:len(chain)])
        kv_session.token_ids = [int(t) for t in prompt_ids[:end]]
        self.restored_pages += len(chain) - first
        return end - common

    def status(self) -> dict:
        if self.host is None:
            return {"pages": 0, "mb": 0, "source": self.source}
        return {"pages": self.pool.capacity, "free": self.pool.free_pages, "mb": round(self.budget_bytes / 2 ** 20),
                "parked": self.parked_pages, "restored": self.restored_pages, "source": self.source}


def served_reach(kv_plan, tier: Optional[KVTier], window: int) -> "tuple[int, str]":
    """(positions, what they come from) a served session may hold — the ONE ceiling both doors read and the ready line
    states: the card's reach (serving.session.session_ceiling: the plan's session reach, else its cache), then, when
    the KV host tier serves and the model's window is longer, the host spill's reach on the tier's share now, never past
    the window. A request past it is refused before it runs."""
    from neurobrix.serving.session import session_ceiling
    card, source = session_ceiling(kv_plan)
    if tier is None or not window or window <= card:
        return card, source
    from nbxtensor.memory.host_spill import spill_bytes_per_position
    host = tier.spill_budget() // spill_bytes_per_position(kv_plan.num_layers, kv_plan.num_kv_heads,
                                                           kv_plan.k_head_dim, kv_plan.v_head_dim)
    total = min(int(window), card + host)
    return total, (f"{card} positions on the card ({source}) + {total - card} in host memory (the mid-decode spill on "
                   f"the KV host tier's share of {tier.budget_bytes / 2 ** 20:.0f} MB), the model's window {window}")


def session_chunk_reserve(kv_plan) -> Optional[int]:
    """The bytes the plan reserved for a session chunk's attention over the cache, as the compiled wrapper spends them
    on a spilled chunk (`price.session_chunk_attention_bytes`, the figure Prism priced from the same three fields);
    None for a plan with no session chunk (a run's one pass): the wrapper keeps its one page of rows."""
    if kv_plan is None or not getattr(kv_plan, "prefill_chunk", None):
        return None
    from neurobrix.core.state.price import session_chunk_attention_bytes
    return session_chunk_attention_bytes(kv_plan.prefill_chunk, kv_plan.max_cache_len, kv_plan.dtype, "compiled")


def attach_spill(cache, tier: Optional[KVTier], kv_plan, window: int) -> str:
    """A served session's cache past the plan's: when the model's window exceeds what the plan's cache holds on the
    card (residency first, c957cee3), the cache keeps at most the plan's pages per sequence on the card and spills its
    oldest pages to host memory past them (nbxtensor memory/host_spill.py) on the tier's share of the one host budget.
    Off (said) without a tier, or when the plan's cache already reaches the window. Returns the sentence that says it."""
    if tier is None or kv_plan is None:
        return "KV mid-decode spill: off (no KV host tier)"
    from neurobrix.serving.session import session_ceiling
    card = session_ceiling(kv_plan)[0]                 # the card's figure the served reach adds the host's to
    if not window or window <= card:
        return f"KV mid-decode spill: off (the plan's cache of {card} positions reaches the model's window {window})"
    from nbxtensor.memory.host_spill import HostSpill
    cache.card_pages = -(-card // cache.page_tokens)
    cache.host_spill = HostSpill(kv_plan.num_layers, kv_plan.num_kv_heads, kv_plan.k_head_dim, kv_plan.v_head_dim,
                                 tier.spill_budget)
    on_card = cache.card_pages * cache.page_tokens
    return (f"KV mid-decode spill: on — {on_card} positions on the card, then up to "
            f"{min(cache.host_spill.reach(), window - on_card)} in host memory ({cache.host_spill.position_bytes()} "
            f"bytes each, the tier's share of {tier.budget_bytes / 2 ** 20:.0f} MB), the model's window {window}")


def kv_tier_for(plan, profile) -> "tuple[Optional[KVTier], str]":
    """The serving daemon's tier for `plan` on `profile`, and the sentence that says it. The budget is the host law
    Prism plans by: the installed RAM less the system's measured share (PrismSolver._less_the_system_share), less the
    plan's whole host side (its host footprint, which on unified memory holds the device plan too); the user's
    `serving.kv_host_tier_mb` (system.yml) caps it when set (absent or null: the computed budget; 0: off). Page-locked
    host pages on a discrete card, pageable on unified memory. None (said) for a plan without a KV cache."""
    import yaml
    from neurobrix.core.prism.solver import PrismSolver
    from neurobrix.core.workspace import SYSTEM_YML
    if getattr(plan, "kv_cache_plan", None) is None:
        return None, "KV host tier: none (the plan holds no KV cache)"
    ram = float(getattr(getattr(profile, "cpu", None), "ram_mb", 0) or 0)
    if ram <= 0:
        raise RuntimeError("ZERO FALLBACK: the KV host tier sizes by the profile's installed RAM (cpu.ram_mb); "
                           "this profile states none")
    side = int((getattr(plan, "host_footprint", None) or {}).get("total_bytes", 0)) / 2 ** 20
    budget = max(0.0, PrismSolver._less_the_system_share(profile, ram) - side)
    cap = ((yaml.safe_load(SYSTEM_YML.read_text()) or {}).get("serving") or {}).get("kv_host_tier_mb")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap < 0):
        raise RuntimeError(f"ZERO FALLBACK: serving.kv_host_tier_mb in {SYSTEM_YML} is {cap!r}: a number of MB >= 0, "
                           f"or null for the computed budget")
    source = (f"{ram:.0f} MB installed less the system's share less the plan's host side {side:.0f} MB"
              + (f", capped by serving.kv_host_tier_mb = {cap}" if cap is not None else ""))
    if cap is not None:
        budget = min(budget, float(cap))
    pinned = not any(getattr(d, "has_unified_memory", False) for d in getattr(profile, "devices", []))
    if budget <= 0:
        return None, f"KV host tier: off ({source})"
    return KVTier(budget, pinned, source, plan.kv_cache_plan.max_cache_len), (f"KV host tier: {budget:.0f} MB of {'page-locked' if pinned else 'pageable'}"
                                            f" host memory ({source})")
