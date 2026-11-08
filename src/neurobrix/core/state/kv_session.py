"""A serving session's KV cache between its turns (docs/design/prism_kv.md §5bis, scope A).

The cache belongs to the SESSION, not to the request: turn n prefills only the positions the cache does not already
hold, on both engines. This module is the shared, device-free half (torch-free, R33: the Triton flow imports it): which
token ids the cache holds position by position, which engine cache holds them, and the per-turn ledger of what was
matched and what was prefilled. The engines' caches and the prefill from an offset are per engine
(core/flow/autoregressive.py, triton/flow/autoregressive.py).

What the cache holds is decided by ids, never by text: the chat template renders the whole conversation at every turn,
and the held ids are matched against the rendered ones. Whatever the template changes in an earlier turn simply
shortens the match. The cached positions past the match are dropped (the cache is truncated there); a position before
it is the same bytes whatever came after it (causal attention).

One engine cache is shared by every request of an executor on the Triton engine (its replay plans bake the buffers'
addresses). A session therefore marks the cache it fills (`holder`), and any other request that clears it unmarks it:
the session then finds nothing held and prefills its whole conversation. Exactness never rests on a cache another
request may have rewritten.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


class KVSession:
    """The positions one session's KV cache holds between turns.

    `ceiling`: the positions the session may hold — the plan's (serving/session.session_ceiling), with `source` naming
    the plan fact it came from. The flows never enlarge it; the serving layer refuses a turn past it by name.
    """

    def __init__(self, ceiling: int, source: str):
        if int(ceiling) <= 0:
            raise ValueError(f"KVSession: ceiling {ceiling} ({source})")
        self.ceiling = int(ceiling)
        self.source = str(source)
        self.token_ids: List[int] = []      # the ids whose K/V the cache holds, position by position
        self.turns: List[Dict[str, int]] = []
        self._owner: Any = None             # the executor whose cache holds the positions
        self._handle: Any = None            # the engine's object for that cache (Triton: the cache; compiled: the wrapper)
        self._cache: Any = None             # the cache object carrying the `holder` mark

    # -- which engine cache holds the session -------------------------------------------------------------------------
    def held(self, owner) -> Any:
        """The engine handle this session's positions live in on `owner`, or None when nothing is held there (another
        executor, another request cleared the cache, or the session never ran)."""
        if self._owner is not owner or self._cache is None or getattr(self._cache, "holder", None) is not self:
            return None
        return self._handle

    def bind(self, owner, handle, cache) -> None:
        """`handle` (its cache `cache`) now holds this session's positions. What the session held is kept only when it
        was this very cache, still marked by this session; otherwise the session holds nothing yet."""
        if self.held(owner) is not handle:
            self.token_ids = []
        self._owner, self._handle, self._cache = owner, handle, cache
        cache.holder = self

    # -- the turn -------------------------------------------------------------------------------------------------------
    def match(self, ids: Sequence[int]) -> int:
        """Positions of `ids` the cache already holds: the longest common prefix with the held ids. Never the whole
        request: its last position is always prefilled, so the turn has the logits it samples from."""
        n, lim = 0, min(len(self.token_ids), len(ids) - 1)
        held = self.token_ids
        while n < lim and held[n] == int(ids[n]):
            n += 1
        return n

    def record(self, ids_held: Sequence[int], matched: int, prompt: int, prefilled: int, padded: int) -> None:
        """The request ended with the cache holding `ids_held`; the turn matched `matched` positions of its `prompt`
        and prefilled `prefilled` rows (`padded` of them padding a planned chunk)."""
        self.token_ids = [int(t) for t in ids_held]
        self.turns.append({"prompt": int(prompt), "matched": int(matched), "prefilled": int(prefilled),
                           "padded": int(padded), "held": len(self.token_ids)})

    def forget(self) -> None:
        """The cache's contents are no longer the session's (a failed request, a new conversation): nothing is held,
        the next turn prefills its whole conversation."""
        self.token_ids = []
        if self._cache is not None and getattr(self._cache, "holder", None) is self:
            self._cache.holder = None

    @property
    def last_turn(self) -> Optional[Dict[str, int]]:
        return self.turns[-1] if self.turns else None


def forgetting_on_failure(kv_session: Optional[KVSession], run):
    """`run()`, a session's request; when it fails the session holds nothing (the cache's contents are no longer known
    to be its), and the failure propagates. Without a session, `run()` as it is."""
    try:
        return run()
    except BaseException:
        if kv_session is not None:
            kv_session.forget()
        raise


def session_turn(execute):
    """Decorates a flow handler's `execute(self, kv_session)`: the request's session (`self.ctx.kv_session`, None for
    every other request) is handed in, and a failed turn leaves the session holding nothing."""
    import functools

    @functools.wraps(execute)
    def run(self):
        kv_session = getattr(self.ctx, "kv_session", None)
        return forgetting_on_failure(kv_session, lambda: execute(self, kv_session))
    return run


def held_ids(prompt_ids: Sequence[int], generated: Sequence[int], cache_len: int) -> List[int]:
    """The ids a cache of `cache_len` positions holds after a request: its prompt then the generated tokens fed back by
    the decode, in order (the last sampled token is never fed, so the cache usually holds one fewer). A cache longer
    than what was fed is refused by name: its positions would be attributed to ids it never saw."""
    fed = [int(t) for t in prompt_ids] + [int(t) for t in generated]
    if cache_len > len(fed):
        raise RuntimeError(f"ZERO FALLBACK: the KV cache holds {cache_len} positions after the request, and only "
                           f"{len(fed)} ids were fed ({len(prompt_ids)} prompt, {len(generated)} generated)")
    return fed[:cache_len]
