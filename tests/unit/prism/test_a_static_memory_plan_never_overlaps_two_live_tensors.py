"""The static memory plan's offset assignment (core/prism/memory_plan.py, docs/design/triton-static-memory-plan.md):
greedy by size over known lifetimes. What each test would do if the code were wrong: two tensors alive at the same op
sharing a byte — the overlap check fails (red, seen with the lifetime test made exclusive); a workspace above the
lower bound on interval sets where greedy-by-size is optimal — red; an interval that ends before it starts accepted —
red."""
import random

import pytest

from neurobrix.core.prism.memory_plan import ALIGNMENT, assign_offsets, lower_bound


def _overlaps(t, off):
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            (a1, b1, n1), (a2, b2, n2) = t[i], t[j]
            if a1 <= b2 and a2 <= b1 and n1 and n2:
                lo1, hi1 = off[i], off[i] + -(-n1 // ALIGNMENT) * ALIGNMENT
                lo2, hi2 = off[j], off[j] + -(-n2 // ALIGNMENT) * ALIGNMENT
                if lo1 < hi2 and lo2 < hi1:
                    return (i, j)
    return None


def test_two_tensors_alive_at_the_same_op_never_share_a_byte():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, 30)
        t = []
        for _ in range(n):
            a = rng.randint(0, 40)
            t.append((a, a + rng.randint(0, 10), rng.choice([0, 1, 255, 256, 1000, 4096, 1 << 20])))
        off, peak = assign_offsets(t)
        assert _overlaps(t, off) is None, (t, off)
        assert peak >= lower_bound(t)
        assert all(o % ALIGNMENT == 0 for o in off)


def test_a_chain_reuses_the_bytes_of_what_died():
    """x0 -> x1 -> x2 -> x3, each read only by the next: two buffers suffice, the lower bound."""
    t = [(0, 1, 1000), (1, 2, 1000), (2, 3, 1000), (3, 4, 1000)]
    off, peak = assign_offsets(t)
    assert peak == lower_bound(t) == 2 * 1024


def test_a_long_lived_tensor_keeps_its_bytes_while_short_ones_reuse_theirs():
    t = [(0, 10, 4096)] + [(i, i, 1024) for i in range(10)]
    off, peak = assign_offsets(t)
    assert peak == lower_bound(t) == 4096 + 1024


def test_an_interval_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError, match="not an interval"):
        assign_offsets([(3, 2, 10)])
