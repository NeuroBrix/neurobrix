"""A replay's slab is sized by what its first fit reaches over the measured run (`slab.first_fit_extent`), not by the run's
live peak: holes a first fit leaves cannot take a later, larger block. Sized by the live peak, Allegro-TI2V's encoder tile
(9.0 GB live) was 22.6 GB short at its recording; the shortfall retry then took 31.7 GB of a 32 GB card for a run that
never holds more than 8.6 GB in its slab, and the card ran out at the next allocation (2026-11-04).

What this test would do if the code were wrong: the extent computed as the live peak — red at the fragmenting sequence;
a simulation whose policy differs from the slab's own — red at the comparison with a SlabAllocator served the same events
(its device block stubbed)."""
from __future__ import annotations

import random

from nbxtensor.memory import slab as S


def test_a_hole_too_small_for_a_later_block_widens_the_slab():
    # A(4) B(4); A freed leaves a hole of 4 at the bottom; C(6) cannot use it and lands above B
    ev = [(1, 1, 4), (1, 2, 4), (-1, 1, 0), (1, 3, 6)]
    assert S.first_fit_extent(ev) == 14          # the live peak is 10, the sum 14


def test_the_extent_is_the_slabs_own_first_fit(monkeypatch):
    rng = random.Random(7)
    ev, live = [], []
    for k in range(1, 400):
        if live and rng.random() < 0.45:
            ev.append((-1, live.pop(rng.randrange(len(live))), 0))
        need = rng.choice([1, 2, 3, 8, 16, 64]) * S.ALIGN
        ev.append((1, k, need))
        live.append(k)
    top = S.first_fit_extent(ev)
    monkeypatch.setattr(S.DeviceAllocator, "malloc_direct", staticmethod(lambda n, d=None, pooled=True: 1 << 40))
    slab = S.SlabAllocator(top, 0)
    try:
        ptr_of, reached = {}, 0
        for sign, key, need in ev:
            if sign > 0:
                p = slab.malloc(need)
                assert p, f"the slab sized {top} could not serve block {key} the simulation placed"
                ptr_of[key] = p
                reached = max(reached, p - slab.base + need)
            else:
                slab.free(ptr_of.pop(key))
        assert reached == top
    finally:
        slab.allocs.clear()
        slab.base = 0                                  # never handed to the driver: the stub's block is no block
        if slab in S.ACTIVE_SLABS:
            S.ACTIVE_SLABS.remove(slab)
