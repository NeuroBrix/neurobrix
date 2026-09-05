"""The efficient-attention variant's side outputs (lse, philox seed/offset) are
placeholders with no consumer; they must not be rebuilt per call — a fresh
torch.tensor(0, device=cuda) is a pageable H2D copy + a stream sync, 2,240 of
them on a 20-step PixArt request (2026-09-05)."""
import pytest
import torch

from neurobrix.core.runtime.graph import compiled_ops as CO


def test_philox_and_lse_placeholders_are_shared_across_calls():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = CO._placeholder_philox(dev); b = CO._placeholder_philox(dev)
    assert a[0] is b[0] and a[1] is b[1]
    assert a[0].dtype == torch.int64 and int(a[0]) == 0
    z1 = CO._placeholder_zeros((2, 16, 64), dev, torch.float16)
    z2 = CO._placeholder_zeros((2, 16, 64), dev, torch.float16)
    assert z1 is z2 and z1.shape == (2, 16, 64) and not z1.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU: counts host-to-device copies")
def test_efficient_attention_variant_issues_no_host_copy_per_call():
    from torch.profiler import profile, ProfilerActivity
    resolver = CO.CompiledOpResolver(torch.device("cuda"), torch.float16)
    fn = resolver.get_op_func("_scaled_dot_product_efficient_attention",
                              {"args": [], "kwargs": {}})
    q = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.float16)
    out = fn(q, q, q, None, False)
    assert isinstance(out, tuple) and out[0].shape == q.shape
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(5):
            fn(q, q, q, None, False)
        torch.cuda.synchronize()
    h2d = sum(e.count for e in p.key_averages() if "Memcpy HtoD" in e.key)
    assert h2d == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel equality")
def test_efficient_attention_accepts_transposed_views_without_copy():
    from torch.profiler import profile, ProfilerActivity
    resolver = CO.CompiledOpResolver(torch.device("cuda"), torch.float16)
    fn = resolver.get_op_func("_scaled_dot_product_efficient_attention", {"args": [], "kwargs": {}})
    x = torch.randn(2, 256, 8, 40, device="cuda", dtype=torch.float16)
    q, k, v = (x.transpose(1, 2) for _ in range(3))          # [B, H, S, D] strided views
    assert not q.is_contiguous() and q.stride(-1) == 1
    out_view = fn(q, k, v, None, False)[0]
    out_contig = fn(q.contiguous(), k.contiguous(), v.contiguous(), None, False)[0]
    assert torch.equal(out_view, out_contig)                # same kernel, same result
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(q, k, v, None, False); torch.cuda.synchronize()
    copies = sum(e.count for e in p.key_averages() if "direct_copy" in e.key or "copy_" in e.key)
    assert copies == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel equality")
def test_expanded_attention_bias_is_not_materialised_and_results_are_equal():
    from torch.profiler import profile, ProfilerActivity
    B, H, Sq, Sk, D = 2, 4, 512, 300, 40
    q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, Sk, D, device="cuda", dtype=torch.float16)
    small = (torch.rand(B, 1, 1, Sk, device="cuda") > 0.2).float() * -1e4    # fp32 traced mask
    expanded = small.expand(B, H, Sq, Sk)                                      # zero-stride view
    prepared = CO._cast_attn_mask(expanded, q)
    assert tuple(prepared.shape) == (B, H, Sq, Sk) and prepared.dtype == torch.float16
    assert prepared.stride(2) == 0                                              # still a broadcast view
    assert prepared.stride(-2) % CO._ATTN_BIAS_ALIGN == 0                       # aligned row stride
    assert torch.equal(prepared.float(), expanded)                              # same values
    naive = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=expanded.half())
    ours = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=prepared)
    assert torch.equal(naive, ours)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        CO._cast_attn_mask(expanded, q); torch.cuda.synchronize()
    big = [e for e in p.key_averages() if ("copy" in e.key or "elementwise" in e.key)]
    # the only kernels are on the [B,1,1,aligned] buffer, never on B*H*Sq*Sk elements
    assert all(e.count <= 3 for e in big)
