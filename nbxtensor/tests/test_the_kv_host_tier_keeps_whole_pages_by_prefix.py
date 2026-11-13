"""The KV host tier's bookkeeping (nbxtensor.memory.kv_host_tier): whole pages indexed by their prefix, a shared prefix
held once, the least recently used leaves offloaded first, a page a request holds never chosen, host leaves dropped, the
budget in whole pages. Card-less, torch-free."""
import subprocess
import sys

import pytest

from nbxtensor.memory.kv_host_tier import DEVICE, HOST, HostPagePool, PrefixIndex, host_tier_pages

TPP = 4


def test_only_whole_pages_are_indexed_and_the_longest_held_prefix_matches():
    ix = PrefixIndex(TPP)
    chain = ix.insert(list(range(10)), [7, 8])                     # 10 ids = 2 whole pages + 2 positions
    assert [n.page for n in chain] == [7, 8] and len(chain) == 2
    assert [n.page for n in ix.match(list(range(10)) + [99, 98])] == [7, 8]
    assert [n.page for n in ix.match([0, 1, 2, 3, 9, 9, 9, 9])] == [7]       # diverges in the second page
    assert ix.match([1, 2, 3, 4]) == []                                       # same ids, other position: no match
    assert ix.match([0, 1, 2]) == []                                          # less than a page


def test_a_shared_prefix_is_held_once():
    ix = PrefixIndex(TPP)
    system = [5, 5, 5, 5]
    a = ix.insert(system + [1, 1, 1, 1], [0, 1])
    b = ix.insert(system + [2, 2, 2, 2], [0, 2])                  # the system page is the same node
    assert a[0] is b[0] and a[1] is not b[1]
    assert ix.counts() == {DEVICE: 3, HOST: 0}


def test_offload_takes_the_least_recently_used_leaves_and_parents_follow_their_children():
    ix = PrefixIndex(TPP)
    a = ix.insert([1] * 8, [0, 1])
    b = ix.insert([2] * 8, [2, 3])
    ix.match([1] * 8)                                              # A used last: B is the older conversation
    victims = ix.device_victims(2)
    assert victims == [b[1], b[0]], "B's leaf first, then its parent once the leaf is gone"
    for host, v in enumerate(victims):
        ix.offloaded(v, host)
    assert ix.counts() == {DEVICE: 2, HOST: 2}
    assert [n.where for n in ix.match([2] * 8)] == [HOST, HOST]    # B still matches, from the host


def test_a_page_a_request_holds_is_never_chosen():
    ix = PrefixIndex(TPP)
    a = ix.insert([1] * 4, [0])
    ix.hold(a)
    assert ix.device_victims(1) == []
    ix.let_go(a)
    assert ix.device_victims(1) == a
    with pytest.raises(RuntimeError):
        ix.let_go(a)


def test_a_restored_page_comes_back_and_a_host_leaf_is_dropped():
    ix = PrefixIndex(TPP)
    (n,) = ix.insert([3] * 4, [9])
    assert ix.offloaded(n, 0) == 9 and n.where == HOST
    assert ix.restored(n, 4) == 0 and (n.where, n.page) == (DEVICE, 4)
    ix.offloaded(n, 1)
    assert ix.host_victims(5) == [n]
    assert ix.drop(n) == 1 and ix.match([3] * 4) == []


def test_the_budget_holds_whole_pages_and_the_pool_refuses_by_name():
    assert host_tier_pages(10 * 1024, 2, 1024, 1024) == 2           # a page: 2 layers x (1 KB K + 1 KB V)
    assert host_tier_pages(4095, 2, 1024, 1024) == 0                # below one page: the tier is off
    with pytest.raises(ValueError, match="a page of 0 layers"):
        host_tier_pages(4096, 0, 1024, 1024)
    pool = HostPagePool(1)
    p = pool.take()
    with pytest.raises(RuntimeError, match="all held"):
        pool.take()
    pool.give(p)
    with pytest.raises(RuntimeError):
        pool.give(p)


def test_importing_it_loads_neither_torch_nor_triton():
    code = ("import sys, nbxtensor.memory.kv_host_tier; "
            "print([m for m in ('torch', 'triton') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", out
