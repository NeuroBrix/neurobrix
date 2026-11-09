"""An embedding model's pooling (ops/embedding_pool.py, point 9b): mean over the kept positions, cls, the last kept
token whatever the padding side; L2 normalisation; refusals by name. What this test would do if the code were wrong:
padding counted in the mean, the last PADDED position taken for the last token, a left-padded batch read as right-padded,
an unknown kind or an empty sequence pooled silently (red)."""
import numpy as np
import pytest

from nbxtensor.ops.embedding_pool import l2_normalize, pool_hidden

H = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)


def test_mean_excludes_padding():
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
    out = pool_hidden(H, mask, "mean")
    assert np.allclose(out[0], H[0, :2].mean(axis=0)) and np.allclose(out[1], H[1].mean(axis=0))


def test_cls_is_position_zero():
    assert np.array_equal(pool_hidden(H, np.ones((2, 4)), "cls"), H[:, 0])


@pytest.mark.parametrize("mask, want", [([[1, 1, 1, 0], [1, 1, 0, 0]], [2, 1]),     # right padding
                                        ([[0, 1, 1, 1], [0, 0, 1, 1]], [3, 3])])    # left padding
def test_last_is_the_last_kept_token(mask, want):
    out = pool_hidden(H, np.array(mask), "last")
    assert np.array_equal(out, H[[0, 1], want])


def test_l2_normalize():
    x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(l2_normalize(x), [[0.6, 0.8], [1.0, 0.0]])


@pytest.mark.parametrize("call", [lambda: pool_hidden(H, np.ones((2, 4)), "max"),
                                  lambda: pool_hidden(H, np.array([[1, 1, 1, 1], [0, 0, 0, 0]]), "mean"),
                                  lambda: pool_hidden(H, np.ones((2, 3)), "mean"),
                                  lambda: l2_normalize(np.zeros((1, 3), dtype=np.float32))])
def test_refused_by_name(call):
    with pytest.raises(ValueError):
        call()
