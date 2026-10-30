"""When the backend refuses EVERY candidate of a sweep, the error says so — not the error of the one config Triton
then launched.

Triton takes `min` over all-infinite timings and launches an arbitrary config; its OutOfResources named that config
alone (the matrix-unit split flash at head dim 192 on a V100, 2026-10-30), and `autotune_refusals.all_refused`, written
for exactly this, was called by nothing. What would this file do on that code? There is no named refusal to return:
the first cell RED; a partial refusal re-raised unchanged holds either way."""
from types import SimpleNamespace

from nbxtensor.ops._configs import every_candidate_refused

INF = [float("inf")] * 3


def test_every_candidate_infinite_is_named():
    tuned = SimpleNamespace(configs_timings={"a": INF, "b": INF}, base_fn=SimpleNamespace(__name__="flash_kernel"))
    got = every_candidate_refused(tuned, (1, 2), RuntimeError("out of resource: shared memory"))
    assert isinstance(got, RuntimeError)
    msg = str(got)
    assert "every candidate of flash_kernel at key (1, 2)" in msg and "2 of 2" in msg and "shared memory" in msg


def test_a_failure_with_a_runnable_candidate_is_not_renamed():
    tuned = SimpleNamespace(configs_timings={"a": INF, "b": [0.1, 0.1, 0.1]}, base_fn=SimpleNamespace(__name__="k"))
    assert every_candidate_refused(tuned, (1,), RuntimeError("x")) is None
    assert every_candidate_refused(SimpleNamespace(), (1,), RuntimeError("x")) is None   # no sweep ran


def test_the_one_config_the_prune_kept_refused_is_named():
    """Triton benches nothing when one config is left: it launches it. Its backend refusal is the sweep's."""
    from triton.runtime.errors import OutOfResources
    tuned = SimpleNamespace(configs=["the one"], base_fn=SimpleNamespace(__name__="flash_kernel"))
    got = every_candidate_refused(tuned, (3,), OutOfResources(102400, 98304, "shared memory"))
    assert isinstance(got, RuntimeError) and "the only one the prune kept" in str(got), got
    assert every_candidate_refused(tuned, (3,), ValueError("a bug, not a refusal")) is None
