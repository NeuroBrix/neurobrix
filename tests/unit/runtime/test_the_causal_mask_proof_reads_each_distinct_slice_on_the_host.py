"""The compiled engine's causal-mask proof reads the mask on the host, one distinct [q, k] slice at a time.

On the card it held the mask's float copy, its boolean masks and the triangle's indices: 8 GiB for a served
32 768-row prefill chunk of Qwen3-30B-A3B-Thinking on a V100-32GB, and every request ran out of memory
(2026-11-13 00:35). The verdicts must stay those of the old proof."""
import pytest

torch = pytest.importorskip("torch")

from neurobrix.core.runtime.graph import attention_mask as am


def _additive(n, dtype=torch.float16):
    m = torch.zeros(n, n, dtype=dtype)
    return m.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), torch.finfo(dtype).min)


def test_the_causal_patterns_pass():
    n = 8
    assert am._is_causal_content(torch.ones(n, n, dtype=torch.bool).tril()[None, None])
    assert am._is_causal_content(_additive(n)[None, None])
    inf = torch.zeros(n, n).masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
    assert am._is_causal_content(inf[None, None])


def test_a_broadcast_axis_is_read_once_and_a_real_axis_slice_by_slice(monkeypatch):
    n = 8
    seen = []
    real_to = torch.Tensor.to

    def spy(self, *a, **k):
        seen.append(tuple(self.shape))
        return real_to(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "to", spy)
    assert am._is_causal_content(_additive(n)[None, None].expand(1, 4, n, n))
    assert len(seen) == 1
    seen.clear()
    heads = _additive(n)[None].repeat(3, 1, 1)[None]          # a real head axis: every slice is read
    assert am._is_causal_content(heads)
    assert len(seen) == 3


def test_a_mask_that_is_not_causal_is_refused():
    n = 8
    padded = _additive(n).clone()
    padded[:, 0] = torch.finfo(torch.float16).min              # a padding column
    assert not am._is_causal_content(padded[None, None])
    window = _additive(n).clone()
    window[n - 1, 0] = torch.finfo(torch.float16).min          # a sliding window drops the oldest key
    assert not am._is_causal_content(window[None, None])
    heads = _additive(n)[None].repeat(2, 1, 1)
    heads[1, 3, 1] = 0.5                                        # one head carries a bias
    assert not am._is_causal_content(heads[None])
    with pytest.raises(RuntimeError, match="not the causal"):
        am.require_causal_frozen_mask(padded[None, None], n, n, "test")
