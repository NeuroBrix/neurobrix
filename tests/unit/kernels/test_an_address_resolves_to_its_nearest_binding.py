"""An address inside overlapping in-place bindings resolves to the NEAREST one (greatest base at or below it), which
runs to the end of its own projection — never to a neighbour's shorter range that a launch would read past (DSV2
triton garbage, 2026-10-20)."""
from nbxtensor.tensor import DeviceAllocator as DA


def _with_ranges(ranges, fn):
    saved = (list(DA._range_bases), dict(DA._range_size))
    try:
        for b, n in ranges:
            DA._range_add(b, n)
        return fn()
    finally:
        DA._range_bases[:], DA._range_size = saved[0], saved[1]


def test_a_projection_sharing_its_neighbour_s_last_page_resolves_to_its_own_binding():
    page = 16384
    # expert A's projection binds [0x100000, +3 pages); expert B's starts in A's last page and binds 5 pages from it
    a, b = 0x100000, 0x100000 + 2 * page
    got = _with_ranges([(a, 3 * page), (b, 5 * page)], lambda: DA.containing(b + 100))
    assert got == (b, 5 * page)


def test_an_address_past_a_nested_range_resolves_to_the_enclosing_one():
    big, small = 0x200000, 0x200000 + 0x1000
    got = _with_ranges([(big, 0x10000), (small, 0x100)], lambda: DA.containing(small + 0x500))
    assert got == (big, 0x10000)


def test_an_address_no_range_holds_resolves_to_none():
    assert _with_ranges([(0x300000, 0x100)], lambda: DA.containing(0x300100)) is None
