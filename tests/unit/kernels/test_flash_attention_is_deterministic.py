"""Flash attention must return the same bytes for the same inputs.

Measured on M4 Pro / triton-ext, whisper-large-v3-turbo's encoder shape
(H=20, S=1500, D=64), 25 identical calls with Q/K/V hashes unchanged between
them:

    before : 25 distinct results, 1 426 428 of 1 920 000 elements differing,
             max |diff| 7.0e-03
    after  : 1 distinct result

The cause was in this kernel, twice:

    acc_o_scale = tl.exp(m_i - m_ij)
    tl.store(t_ptrs, acc_o_scale)      # store to global TMP
    acc_o_scale = tl.load(t_ptrs)      # read the same addresses straight back

a store to global memory followed immediately by a load of the same addresses,
inside one program, with NO barrier between them. Lanes reach the store and the
load at different times, so a lane can read a slot another lane has not written
yet. It was the reference implementation's workaround for an old NVIDIA
compiler bug and is not needed to compute the value.

Why it took controls to attribute: the output was FULLY written either way (a
sentinel poked into Out immediately before the driver launch left 0 survivors),
the kernel has no atomics and no split-K, and a DIFFERENT kernel through the
same tensors, launcher, driver and runtime was deterministic over 25 launches —
which is what isolated the kernel as the single differing variable.

Whisper is the symptom this protects: six runs against one pinned autotune cache
produced four different transcripts, one of them substituting a word.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("triton")


def _device_ready():
    try:
        from neurobrix.kernels.nbx_tensor import DeviceAllocator
        DeviceAllocator.set_device(0)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _device_ready(), reason="a GPU device is required")


def test_repeated_identical_calls_return_identical_bytes():
    from neurobrix.kernels.nbx_tensor import NBXTensor
    from neurobrix.kernels import wrappers as w

    # S=512 is the smallest shape MEASURED to detect the race: with the
    # round-trip restored it gives 7 distinct results in 8 calls, while S=256
    # gives 1 and would have passed a broken kernel. An instrument is only
    # trusted once it has been seen to detect what it claims.
    H, S, D, REPS = 4, 512, 64, 8
    rng = np.random.default_rng(9)
    q = (rng.standard_normal((1, H, S, D)) * 0.3).astype(np.float32)
    k = (rng.standard_normal((1, H, S, D)) * 0.3).astype(np.float32)
    v = (rng.standard_normal((1, H, S, D)) * 0.3).astype(np.float32)
    tq, tk, tv = (NBXTensor.from_numpy(a) for a in (q, k, v))

    seen = {}
    for i in range(REPS):
        try:
            out = w.scaled_dot_product_attention_wrapper(tq, tk, tv).numpy()
        except Exception as exc:                       # noqa: BLE001
            # The bledden fork refuses this kernel outright — "a 2-D axis reduce
            # whose loop-carried result is BOTH stored 1-D and broadcast back to
            # 2-D" — and falls back to CPU. A backend that cannot run the kernel
            # cannot answer a question about the kernel's determinism, so skip
            # BY NAME rather than fail or silently pass.
            from neurobrix.triton.metal_backend import is_backend_refusal
            if is_backend_refusal(exc) or "Refusing" in str(exc):
                pytest.skip(f"the selected Metal backend refuses this kernel: "
                            f"{str(exc)[:160]}")
            raise
        seen.setdefault(out.tobytes(), []).append(i)

    assert len(seen) == 1, (
        f"{len(seen)} distinct results in {REPS} identical calls — flash "
        f"attention is nondeterministic. Check for a store-then-load round trip "
        f"through global memory with no barrier between them.")


def test_the_unbarriered_tmp_round_trip_is_gone():
    """Guard the source: the race is invisible in any single run's output."""
    import inspect
    from neurobrix.kernels.ops import flash_attention as FA

    src = inspect.getsource(FA)
    assert "tl.store(t_ptrs" not in src, (
        "a store into the TMP scratch is back; if it is followed by a load of "
        "the same addresses with no barrier, lanes race and the kernel becomes "
        "nondeterministic")
