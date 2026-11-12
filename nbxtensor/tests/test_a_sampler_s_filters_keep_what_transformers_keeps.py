"""A decode step's sampling filters and host draws: `ops.token_filters.penalize_repeats` / `outside_nucleus` (on a
device row) and `random.seeded_draw.draw_from_probabilities` / `draw_from_masked_logits` (on a host row). They were four
copies in the engine (NeuroBrix's Triton sampler, the encoder-decoder, dual-AR and TTS flows, and the compiled twins of
the last two) until 2026-11-12.

What this test would do if the code were wrong: a seen id penalised twice or the wrong way round, a half row read at
the wrong width or left in float32, the most probable token masked by a peaked row, a token outside the nucleus or
under min-p drawn, a draw not greedy at temperature 0, two seeded draws that differ, a float32 row computed in another
dtype — RED."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.random.seeded_draw import draw_from_masked_logits, draw_from_probabilities
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


@pytest.mark.parametrize("dt", [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16])
def test_a_seen_id_is_divided_when_positive_multiplied_otherwise_once_and_in_its_dtype(dt):
    _accelerator()
    from nbxtensor.ops.token_filters import penalize_repeats
    logits = NBXTensor.from_numpy(np.array([[2.0, -2.0, 4.0, 1.0]], np.float32)).to_cuda(0).to(dt)
    ids = NBXTensor.from_numpy(np.array([[0, 1, 0, 9, -1]], np.int64)).to_cuda(0)     # a repeat, two out of range
    out = penalize_repeats(logits, ids, 2.0)
    assert out is logits and out.nbx_dtype == dt
    assert out.to(NBXDtype.float32).numpy().tolist() == [[1.0, -4.0, 4.0, 1.0]]
    assert penalize_repeats(logits, [2], 1.0).to(NBXDtype.float32).numpy().tolist() == [[1.0, -4.0, 4.0, 1.0]]


def test_the_most_probable_token_is_never_outside_the_nucleus():
    _accelerator()
    from nbxtensor.ops.token_filters import outside_nucleus
    cum = NBXTensor.from_numpy(np.array([[0.95, 0.97, 1.0], [0.3, 0.6, 1.0]], np.float32)).to_cuda(0)
    assert outside_nucleus(cum, 0.9).numpy().tolist() == [[False, True, True], [False, False, False]]
    assert outside_nucleus(cum, 0.5).numpy().tolist() == [[False, True, True], [False, False, True]]


def test_a_host_draw_is_greedy_at_zero_and_never_leaves_the_nucleus():
    z = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    for draw in (draw_from_probabilities, draw_from_masked_logits):
        assert draw(z, 0.0) == 0
        assert draw(z, 0.0, seen_ids=[0], repetition_penalty=4.0) == 1     # log 0.5 * 4 falls below log 0.3
        got = {draw(z, 1.0, top_p=0.7, rng=np.random.RandomState(s)) for s in range(200)}
        assert got == {0, 1}, (draw.__name__, got)
    got = {draw_from_masked_logits(z, 1.0, min_p=0.5, rng=np.random.RandomState(s)) for s in range(200)}
    assert got == {0, 1}                                                    # 0.15 < 0.5 x 0.5


def test_a_seeded_draw_repeats_and_a_float32_row_is_computed_in_float32():
    z = np.random.default_rng(1).standard_normal(1000)
    for draw in (draw_from_probabilities, draw_from_masked_logits):
        a = [draw(z, 0.8, top_p=0.9, rng=np.random.RandomState(7)) for _ in range(3)]
        assert len(set(a)) == 1
    # a two-token row whose probability straddles the seeded uniform, one float32 ulp apart: float32 and float64
    # arithmetic draw different tokens, so the row's own dtype decides
    def reference(row):
        e = np.exp(row / 0.8 - (row / 0.8).max())
        return int(np.random.RandomState(3).choice(2, p=e / e.sum()))
    u = np.random.RandomState(3).random_sample()
    base = np.float32(0.8 * np.log((1 - u) / u))
    rows = [np.array([0.0, base + np.float32(k) * np.spacing(base)], np.float32) for k in range(-300, 300)]
    row = next(r for r in rows if reference(r) != reference(r.astype(np.float64)))
    assert draw_from_probabilities(row, 0.8, rng=np.random.RandomState(3)) == reference(row)
    assert draw_from_probabilities(row.astype(np.float64), 0.8, rng=np.random.RandomState(3)) == \
        reference(row.astype(np.float64))
