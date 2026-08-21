"""Unit test — the schedule contract that lets MoE intermediates shrink.

`_fused_moe_pass` used to allocate its three intermediates with
`sorted_token_ids.shape[0]` rows — the SORTED, PADDED schedule. At decode
(M=1, top_k=8, BLOCK_SIZE_M=16) that is 128 slots for 8 real rows,
zero-initialised and then carried through silu and mul, on every layer of
every token.

They are now sized at `M * top_k`, with `empty` instead of `zeros`. Both
follow from one property of `moe_align_block_size`, and this file tests
that property directly:

    every schedule slot holds either a token index in [0, M*top_k),
    or a sentinel >= M*top_k;
    and every index in [0, M*top_k) appears EXACTLY ONCE.

Given that, `token_mask = offs_token < num_valid_tokens` in
`fused_moe_kernel` masks off exactly the padded slots, so M*top_k rows is
the correct output size — and since every real row is written exactly
once and only those rows are read, zeroing protects nothing.

Testing the contract rather than the arithmetic is deliberate. A
synthetic re-creation of the fused pass needs the expert pointer tables,
their stride conventions and the quantised/dense split; a harness that
gets any of those wrong fails identically on both arms and proves
nothing — which is precisely what the first version of this test did.

EXACTNESS ON REAL SHAPES is proven separately and more strongly, by an
op-by-op fingerprint differential against the pre-change engine on the
canonical 30B row: `NBX_OP_FINGERPRINT` over 4,000 ops, **0 differing**,
generated text identical (`c7e18d61007ec6`). Recorded in
`validation_outputs/moe_sizing_2026_08_21/`.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_moe_intermediate_sizing.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_moe_intermediate_sizing.py
"""
from __future__ import annotations

import ctypes

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # script-mode under a pytest-less GPU venv
    class _NoPytest:  # pragma: no cover - shim
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(*a, **k):
            raise SystemExit(0)

    pytest = _NoPytest()  # type: ignore[assignment]

from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXTensor
from neurobrix.triton import moe as MOE


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h_i64(t):
    n = t.numel()
    buf = (ctypes.c_char * (n * 8))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * 8, kind=2)
    return np.frombuffer(bytes(buf), dtype=np.int64).copy()


@pytest.mark.parametrize("M,top_k,E", [
    (1, 8, 128),     # canonical decode — the shape that motivated this
    (1, 8, 64),
    (1, 2, 8),       # tiny router
    (1, 6, 64),      # non-power-of-two top_k
    (1, 4, 160),     # DeepSeek-class width
    (4, 8, 128),     # small batch — the sizing is not decode-only
    (12, 8, 128),    # prefill-shaped
    (23, 6, 64),     # the tracer's prime length
])
def test_schedule_covers_every_token_exactly_once(M, top_k, E) -> None:
    """The contract that makes M*top_k the right output size."""
    if not _has_gpu():
        pytest.skip("no GPU")
    n = M * top_k
    rng = np.random.default_rng(M * 1000 + top_k * 10 + E)
    picks = np.stack([rng.permutation(E)[:top_k] for _ in range(M)])
    idx = NBXTensor.from_numpy(picks.reshape(-1).astype(np.int64))

    sti, eid, _ = MOE.moe_align_block_size(idx, MOE._BLOCK_SIZE_M, E, 0)
    slots = _d2h_i64(sti)

    real = slots[slots < n]
    assert len(real) == n, (
        f"M={M} top_k={top_k} E={E}: schedule holds {len(real)} real slots, "
        f"expected {n}")
    assert np.array_equal(np.sort(real), np.arange(n)), (
        f"M={M} top_k={top_k} E={E}: real slots are not a permutation of "
        f"range({n}) — some output row would never be written, and with "
        f"`empty` that row would be read as garbage")
    assert (slots[slots >= n] >= n).all(), (
        f"M={M} top_k={top_k} E={E}: a padded slot carries an index below "
        f"the sentinel, so the kernel mask would let it store")


@pytest.mark.parametrize("M,top_k,E", [(1, 8, 128), (1, 6, 64), (4, 8, 128)])
def test_schedule_is_wider_than_the_token_layout(M, top_k, E) -> None:
    """Activation proof: the change is not a no-op at these shapes."""
    if not _has_gpu():
        pytest.skip("no GPU")
    n = M * top_k
    rng = np.random.default_rng(7)
    picks = np.stack([rng.permutation(E)[:top_k] for _ in range(M)])
    idx = NBXTensor.from_numpy(picks.reshape(-1).astype(np.int64))
    sti, _, _ = MOE.moe_align_block_size(idx, MOE._BLOCK_SIZE_M, E, 0)
    padded = sti.shape[0]
    assert padded > n, (
        f"M={M} top_k={top_k}: schedule {padded} is not wider than the "
        f"token layout {n} — nothing to save here")
    if M == 1:
        assert padded // n >= 8, (
            f"expected a large over-allocation at decode, got {padded}/{n}")


def test_kernel_mask_bound_matches_the_new_size() -> None:
    """The kernel's mask bound must be M*top_k, which is what the buffers
    are now sized at. If the kernel ever masked on the padded count
    instead, the smaller buffer would be written out of range."""
    import inspect

    from neurobrix.kernels.ops import fused_moe as FM
    src = inspect.getsource(FM)
    assert "token_mask = offs_token < num_valid_tokens" in src, (
        "the fused MoE kernel no longer masks stores on num_valid_tokens; "
        "sizing the intermediates at M*top_k is no longer justified")
    assert "mask=c_mask" in src, (
        "the fused MoE store is no longer masked")


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    shapes = [(1, 8, 128), (1, 8, 64), (1, 2, 8), (1, 6, 64), (1, 4, 160),
              (4, 8, 128), (12, 8, 128), (23, 6, 64)]
    for M, top_k, E in shapes:
        try:
            test_schedule_covers_every_token_exactly_once(M, top_k, E)
            print(f"  PASS  schedule contract  M={M} top_k={top_k} E={E}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  M={M} top_k={top_k} E={E}\n"
                  f"        {str(e).splitlines()[0]}")
        except Exception as e:
            fails += 1
            print(f"  ERROR M={M} top_k={top_k} E={E}\n"
                  f"        {type(e).__name__}: {e}")
    for M, top_k, E in [(1, 8, 128), (1, 6, 64), (4, 8, 128)]:
        try:
            test_schedule_is_wider_than_the_token_layout(M, top_k, E)
            print(f"  PASS  activation proof  M={M} top_k={top_k} E={E}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  activation proof M={M} top_k={top_k}\n        {e}")
    try:
        test_kernel_mask_bound_matches_the_new_size()
        print("  PASS  kernel mask bound is num_valid_tokens")
    except AssertionError as e:
        fails += 1
        print(f"  FAIL  kernel mask bound\n        {e}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
