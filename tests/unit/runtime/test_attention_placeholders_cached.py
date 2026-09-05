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
