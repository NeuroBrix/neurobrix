"""A row program keeps at most the profile's `memory.max_registers_per_thread` elements a thread (NVIDIA: 255, the CUDA
programming guide). Measured 2026-10-26: one warp over a 16 384-wide fp16 softmax row (512 elements a thread)
compiled for over 30 minutes and stalled the certification. What this test would do if the code were wrong: the
1-warp candidate survives the prune for the 16 384 row — RED."""
import pytest
from neurobrix.core.prism.loader import load_profile
from neurobrix.kernels import wrappers as W
from neurobrix.kernels.ops import softmax as S
from neurobrix.kernels.ops import _configs as C


@pytest.fixture
def volta():
    prev = W.get_hardware_profile()
    W.set_hardware_profile(load_profile("v100-16g"))
    yield
    W.set_hardware_profile(prev)


def test_a_wide_row_keeps_only_the_warps_its_registers_hold(volta):
    kept = S._prune_softmax(S._softmax_configs(), {"feat_dim": 16384, "batch_dim": 15360})
    assert {(c.kwargs["BLOCK_SIZE_BATCH"], c.num_warps) for c in kept} == {(1, 4), (1, 8)}
    assert C.fits_thread_registers(16384, 4) and not C.fits_thread_registers(16384, 2)   # 128 vs 256 elements a thread


def test_a_short_row_keeps_its_candidates(volta):
    assert len(S._prune_softmax(S._softmax_configs(), {"feat_dim": 64, "batch_dim": 8})) == 16
