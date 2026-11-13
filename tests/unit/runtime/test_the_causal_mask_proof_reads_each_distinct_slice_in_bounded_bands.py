"""The compiled engine's causal-mask proof reads each distinct [q, k] slice where the mask lives, in bands of rows whose
temporaries never exceed the mask's own bytes.

The whole-slice proof held the mask's float copy, its boolean masks and the triangle's indices on the card: 8 GiB for a
served 32 768-row prefill chunk of Qwen3-30B-A3B-Thinking on a V100-32GB, every request out of memory (2026-11-13 00:35).
Read on the host instead, it copied and scanned the mask at every layer: Qwen3-Coder-30B-A3B served --compiled ran
1.5-2.3x slower (2026-11-13 04:16). The verdicts must stay those of the old proof."""
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


def test_a_broadcast_axis_is_read_once_and_the_bands_hold_no_more_than_the_mask(monkeypatch):
    n = 8
    seen = []
    real = am._slice_is_causal

    def spy(m, r0, band):
        seen.append((r0, band, band * m.shape[-1] * am._PROOF_BYTES_PER_ELEMENT))
        return real(m, r0, band)

    monkeypatch.setattr(am, "_slice_is_causal", spy)
    mask = _additive(n)[None, None].expand(1, 4, n, n)
    assert am._is_causal_content(mask)
    assert [r0 for r0, _, _ in seen] == list(range(0, n, seen[0][1]))      # one slice, every band once
    assert all(scratch <= n * n * mask.element_size() for _, _, scratch in seen)
    seen.clear()
    assert am._is_causal_content(_additive(n)[None].repeat(3, 1, 1)[None])  # a real head axis: every slice is read
    assert len(seen) == 3 * len(range(0, n, seen[0][1]))


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
    late = _additive(n).clone()
    late[n - 1, n - 1] = torch.finfo(torch.float16).min        # a defect in the last band only
    assert not am._is_causal_content(late[None, None])
    with pytest.raises(RuntimeError, match="not the causal"):
        am.require_causal_frozen_mask(padded[None, None], n, n, "test")
