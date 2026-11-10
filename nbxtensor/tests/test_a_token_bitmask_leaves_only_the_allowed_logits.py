"""nbxtensor.ops.token_bitmask: every logit whose bit is clear becomes -inf, the allowed ones keep their bytes, on the
machine's accelerator; the layout is llguidance's / XGrammar's (bit v % 32 of int32 word v // 32)."""
import numpy as np
import pytest

from nbxtensor.dtype.nbx_dtype import NBXDtype
from nbxtensor.ops.token_bitmask import apply_token_bitmask, bitmask_words, token_bitmask
from nbxtensor.tensor import NBXTensor


def _accelerator():
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() == 0:          # only a machine without an accelerator skips; any other error fails
        pytest.skip("no accelerator on this machine")


def test_the_bitmask_layout_is_bit_v_mod_32_of_word_v_div_32():
    m = token_bitmask([0, 31, 32, 70], 71)
    assert m.shape == (1, 3)
    w = m.view(np.uint32)[0]
    assert w[0] == (1 | (1 << 31)) and w[1] == 1 and w[2] == (1 << 6)


@pytest.mark.parametrize("vocab", [32000, 151936, 70])
def test_only_the_allowed_logits_survive(vocab):
    _accelerator()
    rng = np.random.default_rng(vocab)
    x = rng.standard_normal((2, vocab)).astype(np.float32)
    allowed = [sorted(rng.choice(vocab, 17, replace=False)), [0, vocab - 1]]
    mask = np.concatenate([token_bitmask(a, vocab) for a in allowed])
    t = NBXTensor.from_numpy(x)
    apply_token_bitmask(t, mask)
    got = t.numpy()
    for r, a in enumerate(allowed):
        keep = np.zeros(vocab, bool)
        keep[a] = True
        assert np.array_equal(got[r][keep], x[r][keep])
        assert np.all(np.isneginf(got[r][~keep]))


def test_a_mask_of_the_wrong_shape_or_allowing_nothing_is_refused():
    _accelerator()
    t = NBXTensor.from_numpy(np.zeros((1, 64), np.float32))
    with pytest.raises(ValueError, match="bitmask of"):
        apply_token_bitmask(t, np.zeros((1, 3), np.int32))
    with pytest.raises(RuntimeError, match="allows no token"):
        apply_token_bitmask(t, np.zeros((1, bitmask_words(64)), np.int32))


def test_the_mask_kernel_launches_first_in_a_fresh_process():
    """The kernel is the process's FIRST launch (no earlier op installed the library's launcher: the trap the gathered
    KV copies fell into on Metal, 2026-11-09), and no torch is imported."""
    import subprocess, sys, textwrap
    code = textwrap.dedent('''
        import sys, numpy as np
        from nbxtensor.ops.token_bitmask import apply_token_bitmask, token_bitmask
        from nbxtensor.tensor import NBXTensor
        t = NBXTensor.from_numpy(np.ones((1, 70), np.float32))
        apply_token_bitmask(t, token_bitmask([3, 69], 70))
        got = t.numpy()[0]
        assert got[3] == 1 and got[69] == 1 and np.isneginf(got[0]) and np.isneginf(got[68]), got
        print("torch" in sys.modules)
    ''')
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip() == "False"


def test_logits_on_another_card_than_the_current_one_are_masked_on_their_card():
    """The head on card 1 while the process's current card is 0 (a split plan): the mask is uploaded on the logits'
    card, the kernel launched there."""
    from nbxtensor.memory.allocator import DeviceAllocator
    if DeviceAllocator.device_count() < 2:
        pytest.skip("one accelerator: no second card to hold the logits")
    vocab = 151936
    x = np.random.default_rng(1).standard_normal((1, vocab)).astype(np.float32)
    DeviceAllocator.set_device(1)
    t = NBXTensor.from_numpy(x)
    assert t._device_idx == 1
    DeviceAllocator.set_device(0)
    apply_token_bitmask(t, token_bitmask([5, vocab - 1], vocab))
    got = t.numpy()[0]
    keep = np.zeros(vocab, bool)
    keep[[5, vocab - 1]] = True
    assert np.array_equal(got[keep], x[0][keep]) and np.all(np.isneginf(got[~keep]))
