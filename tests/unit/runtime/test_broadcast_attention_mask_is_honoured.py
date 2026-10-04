"""A key-padding mask that BROADCASTS over the query axis ([B, 1, 1, Sk]) is applied
as a mask by the compiled engine's attention — never mistaken for a trace-time
constant shorter than the sequence and replaced by `is_causal=True`.

That mistake ran mochi-1-preview's joint video/text attention causally over the
raster-ordered video tokens in every block (mask [2, 1, 1, 4306]): query 0 returned
V row 0 exactly and the compiled engine rendered a mosaic (2026-10-04). Reference:
torch's own SDPA with the same mask. What would these tests do if the code were wrong?
Under the old test (any extent below the sequence is short) the [B, 1, 1, S] and
[B, 1, S, 1] cases take the causal branch and their equality assertions fail, as does
the extent-1 row of the parametrized helper test (seen failing on injection,
2026-10-04: 4 failed / 5 passed). The expanded zero-stride case is NOT a guard of this
fix — it passes under both conditions; it pins `_cast_attn_mask`'s broadcast path.
"""
import pytest
import torch
import torch.nn.functional as F

from neurobrix.core.runtime.graph import compiled_ops as CO

B, H, S, D = 2, 3, 9, 4          # S != D: the K layout is unambiguous from the shape


def _attention():
    resolver = CO.CompiledOpResolver(torch.device("cpu"), torch.float32)
    return resolver.get_op_func("scaled_dot_product_attention",
                                {"args": [], "kwargs": {}, "nbx_k_pre_transposed": False})


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(B, H, S, D, generator=g) for _ in range(3))


def _padding_mask():
    keep = torch.ones(B, S, dtype=torch.bool)
    keep[0, -3:] = False                                  # batch 0: three padded keys
    keep[1, -7:] = False                                  # batch 1: seven padded keys
    return keep


def test_a_broadcast_additive_padding_mask_is_applied_not_made_causal():
    q, k, v = _qkv()
    mask = torch.where(_padding_mask(), 0.0, float("-inf"))[:, None, None, :]   # [B, 1, 1, S]
    ours = _attention()(q, k, v, mask)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-6)
    assert not torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-3)


def test_a_broadcast_boolean_padding_mask_is_applied_not_made_causal():
    q, k, v = _qkv(1)
    mask = _padding_mask()[:, None, None, :]
    ours = _attention()(q, k, v, mask)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-6)


def test_an_expanded_zero_stride_padding_mask_is_applied():
    """Full-extent axes: passes under the old and new test alike; it pins the
    zero-stride broadcast handling of `_cast_attn_mask`, not the short-axis rule."""
    q, k, v = _qkv(2)
    mask = torch.where(_padding_mask(), 0.0, float("-inf"))[:, None, None, :].expand(B, H, S, S)
    ours = _attention()(q, k, v, mask)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-6)


def test_a_key_axis_broadcast_mask_is_applied():
    """[B, 1, Sq, 1] — a per-query bias broadcast over keys — is not short either."""
    q, k, v = _qkv(3)
    mask = torch.randn(B, 1, S, 1)
    ours = _attention()(q, k, v, mask)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-6)


@pytest.mark.parametrize("extent,seq,short", [(1, 4306, False), (1, 1, False), (4306, 4306, False),
                                               (23, 40, True), (2, 40, True)])
def test_only_an_axis_between_one_and_the_sequence_is_short(extent, seq, short):
    assert CO._mask_axis_short(extent, seq) is short
