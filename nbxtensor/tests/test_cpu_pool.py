"""The CPU pool (nbxtensor.backend.cpu.pool): the cut, the inline path, the measured size and threshold — card-less."""
import numpy as np
import pytest

from nbxtensor.backend.cpu import pool as P


@pytest.fixture(autouse=True)
def _clean():
    yield
    P.set_threads(None)
    for k in ("t", "exp"):
        P.set_split_threshold(k, None)


@pytest.mark.parametrize("n,parts", [(0, 4), (1, 4), (7, 3), (10, 10), (1000, 12), (13, 1)])
def test_the_pieces_cover_once_in_order(n, parts):
    got = P.pieces(n, parts)
    covered = [i for a, b in got for i in range(a, b)]
    assert covered == list(range(n)) and all(b > a for a, b in got) and len(got) <= max(1, parts)
    sizes = [b - a for a, b in got]
    assert not sizes or max(sizes) - min(sizes) <= 1


def test_the_machine_gives_a_core_count_and_an_application_may_set_its_own():
    assert P.measured_threads() >= 1
    P.set_threads(3)
    assert P.threads() == 3
    with pytest.raises(ValueError):
        P.set_threads(0)
    P.set_threads(None)
    assert P.threads() == P.measured_threads()


def test_under_the_threshold_the_work_runs_inline():
    calls = []
    P.set_threads(4)
    P.set_split_threshold("t", 100)
    P.parallel_for(99, lambda a, b: calls.append((a, b)), "t")
    assert calls == [(0, 99)]
    calls.clear()
    P.parallel_for(100, lambda a, b: calls.append((a, b)), "t")
    assert sorted(calls) == P.pieces(100, 4) and len(calls) == 4


def test_a_split_numpy_op_gives_the_same_bytes():
    x = np.random.default_rng(0).standard_normal(1 << 16).astype(np.float32)
    whole, out = np.exp(x), np.empty_like(x)
    P.set_threads(4)
    P.set_split_threshold("exp", 1)
    P.parallel_for(x.size, lambda a, b: np.exp(x[a:b], out=out[a:b]), "exp")
    assert out.tobytes() == whole.tobytes()


def test_a_kind_without_threshold_or_probe_is_refused():
    with pytest.raises(KeyError, match="no threshold set and no probe"):
        P.parallel_for(10, lambda a, b: None, "never-measured")


def test_the_threshold_is_measured_when_a_probe_is_given():
    x = np.random.default_rng(1).standard_normal(1 << 22).astype(np.float32)
    out = np.empty_like(x)
    probe = lambda n: (lambda a, b: np.exp(x[a:b], out=out[a:b]))
    t = P.split_threshold("exp", probe)
    assert 1 << 12 <= t <= 1 << 26 and P.split_threshold("exp") == t
