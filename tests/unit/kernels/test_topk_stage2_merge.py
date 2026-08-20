"""Unit test — top-k stage-2 bitonic merge (P-TOPK-STAGE2-MERGE).

Permanent differential microtest for `wrappers.topk_wrapper` against an
EXTERNAL reference (numpy), never against a recorded NeuroBrix baseline.

=== WHY AN EXTERNAL ORACLE IS THE POINT OF THIS FILE ===

The defect this test locks down shipped in 0.5.1 and survived every gate
we had, because every gate compared NeuroBrix to NeuroBrix. A byte gate
cannot see an error that is present on both sides of the diff, and a
kernel that returns both VALUES and INDICES can be wrong in the indices
while the values still look plausible. Any kernel of that shape needs a
differential against something outside the project.

=== THE DEFECT ===

`topk_wrapper` runs two stages: per-chunk top-k, then a bitonic merge of
the chunk winners. The merge loads `chunk_num * k` real entries into a
BLOCK = next_power_of_2(...) tile and pads the rest. Two independent
bugs made those pads lethal:

  1. `_get_finfo_val` took a dtype and ignored it, always returning the
     fp32 limit. Stage 2's buffer is fp16, and `tl.load(..., other=v)`
     materialises `v` in the pointer's element type, so -3.4028235e+38
     overflowed to -inf (measured directly).
  2. `_compare_and_swap` permuted with a masked SUM (`y * (1 - mask)`).
     `0 * -inf` is NaN, NaN loses every comparison, and the sort network
     collapsed — real values dropped out and pad slots surfaced carrying
     their INT32_MIN sentinel index.

Trigger: pads exist iff `chunk_num * k` is not already a power of two.
That is why the boundary is neither the vocabulary size nor k alone, and
why an early bisection that varied only V found a contradictory answer.

MoE expert routing was mostly, but not universally, immune. A router with
`num_experts <= 256` (or exactly 1024) lands on `chunk_num == 1`, and the
wrapper returns stage 1 directly without ever entering the merge — which
is why 64/128/160 experts measured exact while sampling was corrupt. That
immunity belongs to the chunk heuristic, not to routing: widths in
257..1023 DO enter the merge and were exposed. Both sides of that window
are covered below.

=== WHAT IS CHECKED, SEPARATELY ===

A kernel returning a (values, indices) pair has three failure modes that
must not be collapsed into one assertion:

  values      the returned values are the true top-k values
  indices     every index is in range and distinct
  coherence   x[index] == value, elementwise

Index EQUALITY against the reference's argsort is deliberately NOT
asserted: the wrapper computes in fp16, ties are common there, and two
correct sorts may break a tie differently. Coherence catches every real
corruption without manufacturing failures out of legitimate ties.

Runnable two ways:
  - pytest:  PYTHONPATH=src python3 -m pytest tests/unit/kernels/test_topk_stage2_merge.py -v
  - script:  PYTHONPATH=src python3 tests/unit/kernels/test_topk_stage2_merge.py
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

import triton

from neurobrix.kernels import wrappers as W
from neurobrix.kernels.nbx_tensor import DeviceAllocator, NBXDtype, NBXTensor

_NP = {
    NBXDtype.float32: (np.float32, 4),
    NBXDtype.float16: (np.float16, 2),
    NBXDtype.int64: (np.int64, 8),
    NBXDtype.int32: (np.int32, 4),
}

# Real vocabularies from the zoo, plus the shapes that bracket the
# padding boundary. `pad` is the property that actually decides the
# outcome; it is recomputed in the test so a future chunking heuristic
# change shows up as a changed label rather than a silent hole.
_VOCABS = [
    (4096, 5),        # 4 chunks x 5   = 20   -> BLOCK 32   padded
    (4096, 16),       # 4 chunks x 16  = 64   -> BLOCK 64   exact
    (16384, 5),       # 16 chunks x 5  = 80   -> BLOCK 128  padded
    (16384, 8),       # 16 chunks x 8  = 128  -> BLOCK 128  exact
    (16384, 32),      # 16 chunks x 32 = 512  -> BLOCK 512  exact
    (32000, 40),      # Llama-2 / TinyLlama
    (32768, 50),
    (128256, 20),     # Llama-3
    (151936, 1),      # Qwen3 - greedy width
    (151936, 5),
    (151936, 50),     # Qwen3 - the canonical decode row's sampler width
    (152064, 64),     # Qwen2
]

# Router widths. The first five bypass the merge (chunk_num == 1); 512
# and 768 fall inside the 257..1023 window that DOES enter it, so they
# were exposed to the shipped defect and must be covered explicitly.
_MOE = [(64, 6), (64, 8), (128, 8), (160, 8), (256, 4),
        (512, 2), (512, 6), (768, 6)]


def _has_gpu() -> bool:
    try:
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


def _d2h(t):
    dt, sz = _NP[t._dtype]
    n = t.numel()
    buf = (ctypes.c_char * (n * sz))()
    DeviceAllocator.memcpy(ctypes.addressof(buf), t.data_ptr(), n * sz, kind=2)
    return np.frombuffer(bytes(buf), dtype=dt).copy()


def _chunking(V: int, k: int):
    """Mirror of the wrapper's chunk heuristic, for labelling only."""
    chunk = 256 if V < 1024 else 1024
    if chunk < k:
        chunk = triton.next_power_of_2(k)
    n_chunks = triton.cdiv(V, chunk)
    stage2_n = n_chunks * k
    return n_chunks, stage2_n, triton.next_power_of_2(stage2_n)


def _run(x16: np.ndarray, k: int):
    """Return (values, indices, fp32 view of the fp16 input row)."""
    v, i = W.topk_wrapper(NBXTensor.from_numpy(x16), k, dim=-1)
    return _d2h(v).astype(np.float32), _d2h(i), x16[0].astype(np.float32)


def _assert_topk(x16: np.ndarray, k: int, label: str) -> None:
    got_v, got_i, ref = _run(x16, k)
    want_v = np.sort(ref)[::-1][:k]

    # 1. VALUES — the multiset must be the true top-k. Compared after a
    #    sort so this assertion is independent of ordering, which #4
    #    checks on its own.
    assert np.array_equal(np.sort(got_v)[::-1], want_v), (
        f"{label}: wrong VALUES\n  got  {got_v[:8]}\n  want {want_v[:8]}")

    # 2. INDICES — in range and distinct. A pad slot leaking through the
    #    merge shows up here as INT32_MIN.
    assert got_i.min() >= 0 and got_i.max() < x16.shape[-1], (
        f"{label}: index out of range (pad sentinel leaked?) {got_i[:8]}")
    assert len(np.unique(got_i)) == k, (
        f"{label}: duplicate indices {got_i[:8]}")

    # 3. COHERENCE — each index must actually carry its value. This is
    #    the check that a values-only comparison cannot make, and the one
    #    the shipped defect would have failed.
    assert np.array_equal(ref[got_i], got_v), (
        f"{label}: index/value incoherent\n"
        f"  x[idx] {ref[got_i][:8]}\n  vals   {got_v[:8]}")

    # 4. ORDER — sorted=True is part of the contract.
    assert np.all(np.diff(got_v) <= 0), f"{label}: not descending {got_v[:8]}"

    # 5. No non-finite output for finite input.
    assert np.isfinite(got_v).all(), f"{label}: non-finite output {got_v[:8]}"


@pytest.mark.parametrize("V,k", _VOCABS)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_topk_matches_numpy(V: int, k: int, seed: int) -> None:
    """Multi-chunk top-k against numpy, padded and unpadded alike."""
    if not _has_gpu():
        pytest.skip("no GPU")
    n_chunks, stage2_n, block = _chunking(V, k)
    label = (f"V={V} k={k} seed={seed} chunks={n_chunks} "
             f"stage2_n={stage2_n} BLOCK={block} "
             f"{'PADDED' if stage2_n != block else 'exact'}")
    x = np.random.default_rng(seed).standard_normal((1, V)).astype(np.float16)
    _assert_topk(x, k, label)


@pytest.mark.parametrize("V,k", _VOCABS[:6])
def test_topk_fp32_input(V: int, k: int) -> None:
    """fp32 in / fp32 out. The wrapper downcasts internally for the
    kernel; the oracle is built on the same fp16 rounding so this tests
    the kernel, not the cast."""
    if not _has_gpu():
        pytest.skip("no GPU")
    a32 = np.random.default_rng(3).standard_normal((1, V)).astype(np.float32)
    a16 = a32.astype(np.float16)
    v, i = W.topk_wrapper(NBXTensor.from_numpy(a32), k, dim=-1)
    got_v, got_i = _d2h(v).astype(np.float32), _d2h(i)
    ref = a16[0].astype(np.float32)
    want = np.sort(ref)[::-1][:k]
    assert np.array_equal(np.sort(got_v)[::-1], want), f"V={V} k={k} values"
    assert got_i.min() >= 0 and got_i.max() < V, f"V={V} k={k} index range"
    assert np.array_equal(ref[got_i], got_v), f"V={V} k={k} coherence"


@pytest.mark.parametrize("n_experts,k", _MOE)
def test_topk_moe_routing(n_experts: int, k: int) -> None:
    """MoE router widths. Expert routing has no tolerance — a wrong index
    sends the token to the wrong expert.

    Most router widths bypass the merge entirely (chunk_num == 1) and so
    were never exposed to the stage-2 defect, which is why 64/128/160
    experts measured clean while sampling was corrupt. That immunity is
    NOT a property of MoE routing though, it is a property of the chunk
    heuristic: `chunk = 256 if V < 1024 else 1024`. A router with 257 to
    1023 experts lands on chunk_num == 2+ and WAS exposed. This test
    covers both sides of that window rather than assuming the window
    away.
    """
    if not _has_gpu():
        pytest.skip("no GPU")
    n_chunks, _, _ = _chunking(n_experts, k)
    for seed in (0, 5):
        x = np.random.default_rng(seed).standard_normal(
            (1, n_experts)).astype(np.float16)
        _assert_topk(
            x, k,
            f"MoE experts={n_experts} k={k} seed={seed} chunks={n_chunks}")


def test_moe_single_chunk_window_is_documented() -> None:
    """Lock the boundary the routing immunity actually rests on.

    If the chunk heuristic changes, this fails and forces the exposure
    window to be re-derived instead of silently moving.
    """
    assert _chunking(256, 8)[0] == 1, "<=256 experts must bypass the merge"
    assert _chunking(257, 8)[0] == 2, "257 experts must enter the merge"
    assert _chunking(1023, 8)[0] == 4, "1023 experts must enter the merge"
    assert _chunking(1024, 8)[0] == 1, "1024 experts bypasses again"


@pytest.mark.parametrize("V,k", [(16384, 8), (16384, 5), (151936, 50)])
def test_topk_after_neg_inf_mask(V: int, k: int) -> None:
    """The live sampler shape: top-k over a vector already masked to -inf.

    `CombinedSampler` writes -inf into rejected entries and keeps the
    vector at full width, so every subsequent top-k sees a mostly
    non-finite input. The masked-sum permutation turned those -inf into
    NaN regardless of padding, which made this case fail even at widths
    where the padded/unpadded distinction says it should pass.
    """
    if not _has_gpu():
        pytest.skip("no GPU")
    base = np.random.default_rng(0).standard_normal((1, V)).astype(np.float16)
    keep = np.argsort(-base[0].astype(np.float32))[:max(k * 2, 50)]
    masked = np.full(V, -np.inf, dtype=np.float16)
    masked[keep] = base[0][keep]
    got_v, got_i, ref = _run(masked[None, :], k)
    want = np.sort(ref)[::-1][:k]
    assert np.array_equal(got_v, want), (
        f"V={V} k={k} masked: values\n  got  {got_v[:8]}\n  want {want[:8]}")
    assert got_i.min() >= 0 and got_i.max() < V, f"V={V} k={k} masked: range"
    assert np.array_equal(ref[got_i], got_v), f"V={V} k={k} masked: coherence"


def test_topk_all_equal() -> None:
    """Every entry identical: the sort is free to return any k indices,
    but they must be distinct, in range, and carry the right value."""
    if not _has_gpu():
        pytest.skip("no GPU")
    V, k = 16384, 5
    x = np.full((1, V), 1.5, dtype=np.float16)
    got_v, got_i, ref = _run(x, k)
    assert np.array_equal(got_v, np.full(k, 1.5, dtype=np.float32))
    assert len(np.unique(got_i)) == k and got_i.min() >= 0 and got_i.max() < V


if __name__ == "__main__":  # script mode
    if not _has_gpu():
        raise SystemExit("no GPU")
    fails = 0
    for V, k in _VOCABS:
        for seed in (0, 1, 7):
            n_chunks, s2n, blk = _chunking(V, k)
            tag = "PADDED" if s2n != blk else "exact "
            try:
                test_topk_matches_numpy(V, k, seed)
                print(f"  PASS  V={V:7d} k={k:3d} seed={seed} "
                      f"chunks={n_chunks:4d} {tag}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL  V={V:7d} k={k:3d} seed={seed} "
                      f"chunks={n_chunks:4d} {tag}\n        {exc}")
    try:
        test_moe_single_chunk_window_is_documented()
        print("  PASS  MoE single-chunk window (<=256 and 1024 bypass)")
    except AssertionError as exc:
        fails += 1
        print(f"  FAIL  MoE single-chunk window\n        {exc}")
    for n_experts, k in _MOE:
        try:
            test_topk_moe_routing(n_experts, k)
            print(f"  PASS  MoE experts={n_experts:4d} k={k}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL  MoE experts={n_experts:4d} k={k}\n        {exc}")
    for V, k in [(16384, 8), (16384, 5), (151936, 50)]:
        try:
            test_topk_after_neg_inf_mask(V, k)
            print(f"  PASS  -inf-masked V={V} k={k}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL  -inf-masked V={V} k={k}\n        {exc}")
    for name, fn in (("fp32 input", lambda: [test_topk_fp32_input(v, k)
                                             for v, k in _VOCABS[:6]]),
                     ("all-equal", test_topk_all_equal)):
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{'ALL GREEN' if not fails else f'{fails} FAILURE(S)'}")
    raise SystemExit(1 if fails else 0)
