"""The KV host tier's policy (core/state/kv_tier.py) on a cache of the `tier_*` protocol: a conversation whose positions
another one takes is parked in host memory and, when it comes back, its whole pages are copied back — the same bytes —
and the session holds them, so only what follows is prefilled. Card-less: a numpy cache with the protocol."""
import numpy as np
import pytest

from neurobrix.core.state.kv_tier import KVTier

TPP, LAYERS, H, D = 8, 2, 2, 4


class FakeCache:
    """One sequence's paged K/V in numpy pools, the `tier_*` protocol, and what a prefill writes (`fill`)."""

    def __init__(self, pages=16):
        self.page_tokens = TPP
        self.k = [np.zeros((pages, H, TPP, D), np.float16) for _ in range(LAYERS)]
        self.v = [np.zeros((pages, H, TPP, D), np.float16) for _ in range(LAYERS)]
        self.pages, self.length, self._next = [], 0, 0
        self.spilled = 0                        # positions a mid-decode spill moved to the host (none here)

    def _reserve(self, n):
        while len(self.pages) * TPP < n:
            self.pages.append(self._next); self._next += 1

    def fill(self, ids):                         # a prefill: position i's K/V a function of the ids up to i (causal)
        self._reserve(len(ids))
        for i in range(self.length, len(ids)):
            val = float(sum(ids[:i + 1]) % 997)
            for L in range(LAYERS):
                self.k[L][self.pages[i // TPP], :, i % TPP] = val + L
                self.v[L][self.pages[i // TPP], :, i % TPP] = -val - L
        self.length = len(ids)

    def clear(self):
        self.pages, self.length = [], 0

    def truncate(self, n):
        assert n <= self.length
        self.length = n

    def tier_layout(self):
        return LAYERS, H * TPP * D * 2, H * TPP * D * 2

    def tier_length(self):
        return self.length

    batch = 1

    @property
    def tables(self):                            # one sequence's page table (nbxtensor sequence_pages reads it)
        from types import SimpleNamespace
        return [SimpleNamespace(pages=self.pages)]

    def _view(self, host, hp, L, which):
        page = host.page_view(hp)
        off = L * (host.k_page_bytes + host.v_page_bytes) + (host.k_page_bytes if which == "v" else 0)
        return page[off:off + host.k_page_bytes].view(np.float16).reshape(H, TPP, D)

    def tier_offload(self, pages, host, hps):
        for p, hp in zip(pages, hps):
            for L in range(LAYERS):
                self._view(host, hp, L, "k")[:] = self.k[L][p]; self._view(host, hp, L, "v")[:] = self.v[L][p]

    def tier_restore(self, host, hps, pages):
        for hp, p in zip(hps, pages):
            for L in range(LAYERS):
                self.k[L][p] = self._view(host, hp, L, "k"); self.v[L][p] = self._view(host, hp, L, "v")

    def tier_resume(self, n):
        self._reserve(n)
        self.length = n

    def snapshot(self, n):
        return [(self.k[L][[self.pages[i // TPP] for i in range(n)], :, [i % TPP for i in range(n)]].copy(),
                 self.v[L][[self.pages[i // TPP] for i in range(n)], :, [i % TPP for i in range(n)]].copy())
                for L in range(LAYERS)]


class Session:
    def __init__(self):
        self.token_ids = []


def _turn(tier, session, cache, ids):
    """One request as the flows play it: the tier first, then the match, the prefill of the rest, the record."""
    restored = tier.prepare(session, cache, ids) if tier else 0
    held = session.token_ids
    start = 0
    while start < min(len(held), len(ids) - 1) and held[start] == ids[start]:
        start += 1
    cache.truncate(min(start, cache.length))
    cache.fill(ids)
    session.token_ids = list(ids)
    return restored, start


def test_a_conversation_comes_back_from_the_host_with_the_same_bytes():
    cache, s = FakeCache(), Session()
    tier = KVTier(budget_mb=1, pinned=False, source="test", slice_positions=2 * TPP)
    a = list(range(1, 42))                      # 41 ids: 5 whole pages + 1
    b = list(range(500, 530))
    _turn(tier, s, cache, a)
    expect = cache.snapshot(40)
    assert _turn(tier, s, cache, b) == (0, 0)   # B takes the cache: A parked, nothing of B held
    assert tier.parked_pages == 5
    a2 = a + [7, 7, 7]                           # A's next turn
    restored, start = _turn(tier, s, cache, a2)
    assert restored == 40 and start == 40, (restored, start)    # its 5 whole pages restored, the rest prefilled
    got = cache.snapshot(40)
    for (ek, ev), (gk, gv) in zip(expect, got):
        assert np.array_equal(ek, gk) and np.array_equal(ev, gv)


def test_the_last_position_is_always_prefilled():
    cache, s = FakeCache(), Session()
    tier = KVTier(budget_mb=1, pinned=False, source="test", slice_positions=2 * TPP)
    a = list(range(1, 17))                       # exactly 2 whole pages
    _turn(tier, s, cache, a)
    _turn(tier, s, cache, list(range(900, 910)))
    restored, start = _turn(tier, s, cache, a)
    assert (restored, start) == (8, 8)           # page 2 holds the last position: prefilled, never restored


def test_a_full_tier_drops_the_least_recently_used_conversation():
    cache, s = FakeCache(pages=64), Session()
    page_bytes = LAYERS * 2 * H * TPP * D * 2
    tier = KVTier(budget_mb=4 * page_bytes / 2 ** 20, pinned=False, source="test", slice_positions=2 * TPP)   # 4 host pages
    a, b, c = [list(range(k * 100, k * 100 + 17)) for k in (1, 2, 3)]
    for ids in (a, b, c):                         # A parked (2 pages), B parked (2): full; C parks by dropping A
        _turn(tier, s, cache, ids)
    _turn(tier, s, cache, list(range(7000, 7003)))
    assert _turn(tier, s, cache, a + [1])[0] == 0          # A was dropped: prefilled whole
    assert tier.parked_pages >= 6


def test_a_budget_below_one_page_leaves_the_tier_off():
    cache, s = FakeCache(), Session()
    tier = KVTier(budget_mb=1 / 2 ** 20, pinned=False, source="test", slice_positions=2 * TPP)
    _turn(tier, s, cache, list(range(1, 42)))
    assert _turn(tier, s, cache, list(range(500, 530))) == (0, 0)
    assert tier.status()["pages"] == 0 and tier.index is None


def test_parking_a_longer_turn_never_drops_its_own_parked_prefix():
    """The host full of A's own first pages: parking A's longer turn finds no page it may take — it never drops a page
    of the chain it extends (the extension waits), and A's prefix still comes back exact."""
    cache, s = FakeCache(pages=64), Session()
    page_bytes = LAYERS * 2 * H * TPP * D * 2
    tier = KVTier(budget_mb=2 * page_bytes / 2 ** 20, pinned=False, source="test", slice_positions=2 * TPP)   # 2 host pages
    a = list(range(1, 18))                        # 2 whole pages + 1
    _turn(tier, s, cache, a)
    _turn(tier, s, cache, [900, 901, 902])        # A parked: the 2 host pages
    a2 = a + list(range(30, 47))                  # A's longer turn: 4 whole pages + 2
    assert _turn(tier, s, cache, a2)[0] == 16
    expect = cache.snapshot(16)
    _turn(tier, s, cache, [903, 904, 905])        # parking A's 4 pages: only its own 2 parked ones exist
    assert _turn(tier, s, cache, a2)[0] == 16     # its prefix intact
    for (ek, ev), (gk, gv) in zip(expect, cache.snapshot(16)):
        assert np.array_equal(ek, gk) and np.array_equal(ev, gv)


from types import SimpleNamespace


def _profile(ram, share, unified):
    return SimpleNamespace(cpu=SimpleNamespace(ram_mb=ram), host_measured={"system_share_mb": share},
                           devices=[SimpleNamespace(has_unified_memory=unified)])


def _plan(host_side_mb, kv=True):
    return SimpleNamespace(kv_cache_plan=SimpleNamespace(max_cache_len=2048) if kv else None,
                           host_footprint={"total_bytes": host_side_mb * 2 ** 20})


@pytest.fixture
def system_yml(tmp_path, monkeypatch):
    import neurobrix.core.workspace as ws
    def write(cap):
        p = tmp_path / "system.yml"
        p.write_text("serving:\n" + ("" if cap == "absent" else f"  kv_host_tier_mb: {cap}\n"))
        monkeypatch.setattr(ws, "SYSTEM_YML", p)
    return write


def test_the_budget_is_the_host_law_and_the_users_cap(system_yml):
    from neurobrix.core.state.kv_tier import kv_tier_for
    system_yml("null")
    tier, note = kv_tier_for(_plan(3000), _profile(24576, 6994, unified=True))   # the M4 Pro: 17 582 MB pool
    assert tier.budget_bytes == (24576 - 6994 - 3000) * 2 ** 20 and not tier.pinned and "pageable" in note
    tier, _ = kv_tier_for(_plan(3000), _profile(65536, 0, unified=False))
    assert tier.pinned and tier.budget_bytes == (65536 - 3000) * 2 ** 20
    system_yml(2048)
    assert kv_tier_for(_plan(3000), _profile(24576, 6994, True))[0].budget_bytes == 2048 * 2 ** 20
    system_yml(0)
    assert kv_tier_for(_plan(3000), _profile(24576, 6994, True))[0] is None
    system_yml("absent")
    assert kv_tier_for(_plan(3000), _profile(24576, 6994, True))[0] is not None
    system_yml("-5")
    with pytest.raises(RuntimeError, match="kv_host_tier_mb"):
        kv_tier_for(_plan(3000), _profile(24576, 6994, True))
    assert kv_tier_for(_plan(3000, kv=False), _profile(24576, 6994, True))[0] is None


def test_the_budget_is_a_ceiling_a_park_allocates_one_slice_not_the_budget():
    """The Dell's review (2026-11-09 21:19): the whole budget allocated at the first park was ~240 GB page-locked on the
    rack. A park of N pages takes one slice (a conversation of the plan), the rest of the budget stays unallocated."""
    cache, s = FakeCache(pages=64), Session()
    tier = KVTier(budget_mb=1, pinned=False, source="test", slice_positions=2 * TPP)     # 1 MB: ~170 pages possible
    _turn(tier, s, cache, list(range(1, 18)))                  # 2 whole pages
    _turn(tier, s, cache, [900, 901, 902])                     # parked
    assert tier.parked_pages == 2
    assert tier.host.capacity == 2 and tier.pool.capacity == 2, (tier.host.capacity, tier.host.max_pages)
    assert tier.host.max_pages > 100
    _turn(tier, s, cache, list(range(40, 75)))                 # 4 whole pages
    _turn(tier, s, cache, [903, 904, 905])                     # parked: two more slices, still far below the ceiling
    assert tier.host.capacity == 6 and tier.parked_pages == 6


def test_a_spilled_conversation_is_never_parked_and_the_spill_takes_what_the_parked_pages_leave():
    """One host budget for the tier's parked pages and a decoding sequence's mid-decode spill (nbxtensor
    memory/host_spill.py): the spill's share is the budget less what the parked pages hold; a spilled sequence (its
    oldest positions are not in the card's pages) is neither parked nor replaced by a parked continuation."""
    cache, s = FakeCache(), Session()
    tier = KVTier(budget_mb=1, pinned=False, source="test", slice_positions=2 * TPP)
    assert tier.spill_budget() == tier.budget_bytes                     # nothing parked: the whole budget
    _turn(tier, s, cache, list(range(1, 42)))
    _turn(tier, s, cache, list(range(500, 530)))                       # the first conversation parked
    assert tier.parked_pages == 5 and tier.spill_budget() == tier.budget_bytes - tier.host_bytes() < tier.budget_bytes
    cache.spilled = TPP
    assert tier.park(list(range(500, 530)), cache) == 0
    assert tier.prepare(s, cache, list(range(1, 42)) + [9]) == 0 and tier.parked_pages == 5


def test_a_served_cache_spills_past_the_plan_s_cache_only_when_the_window_is_longer():
    """attach_spill (core/state/kv_tier): a served cache keeps the plan's cache on the card and spills past it on the
    tier's share when the model's window is longer; off, said, without a tier or when the plan reaches the window."""
    from types import SimpleNamespace
    from neurobrix.core.state.kv_tier import attach_spill
    plan = SimpleNamespace(max_cache_len=100, num_layers=LAYERS, num_kv_heads=H, k_head_dim=D, v_head_dim=D)
    tier = KVTier(budget_mb=1, pinned=False, source="test", slice_positions=2 * TPP)
    off = FakeCache()
    assert "off" in attach_spill(off, None, plan, 1000) and not hasattr(off, "host_spill")
    assert "off" in attach_spill(off, tier, plan, 100) and not hasattr(off, "host_spill")
    on = FakeCache()
    note = attach_spill(on, tier, plan, 1000)
    assert on.card_pages == -(-100 // TPP) and on.host_spill.budget_bytes == tier.spill_budget()
    on_card = on.card_pages * TPP
    assert "on" in note and f"up to {min(on.host_spill.reach(), 1000 - on_card)} in host memory" in note   # never past the window


def test_the_served_reach_is_the_card_then_the_host_spill_within_the_window():
    """served_reach (core/state/kv_tier): the one ceiling both serving doors read — the card's reach (the plan's
    session reach), plus the host spill's on the tier's share when a tier serves and the window is longer, never past
    the window (the rack's Qwen3-Coder: 65 664 on the card, a 79 963-position request refused before the spill)."""
    from types import SimpleNamespace
    from neurobrix.core.state.kv_tier import served_reach, spill_position_bytes
    plan = SimpleNamespace(max_cache_len=100, session_reach=90, num_layers=LAYERS, num_kv_heads=H, k_head_dim=D,
                           v_head_dim=D)
    tier = KVTier(budget_mb=1, pinned=False, source="test", slice_positions=2 * TPP)
    host = tier.spill_budget() // spill_position_bytes(plan)
    assert served_reach(plan, None, 10 ** 6)[0] == 90                    # no tier: the card's reach
    assert served_reach(plan, tier, 80)[0] == 90                          # the window within the card: no spill
    total, source = served_reach(plan, tier, 10 ** 6)
    assert total == 90 + host and "host memory" in source                 # the card, then the host's share
    assert served_reach(plan, tier, 120)[0] == 120                        # never past the window
