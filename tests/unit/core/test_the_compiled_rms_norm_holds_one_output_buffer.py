"""The compiled `custom::rms_norm` (`DtypeEngine.rms_norm_fp32`) holds its input and ONE output buffer at its peak, as
Prism prices it: applying the weight out of place allocated a second full buffer while the normalised one was alive —
Sana 4K's VAE, sliced on a 16 GB card, died asking 2.69 GiB at custom.rms_norm::24 (2026-10-20).

What this test would do if the code were wrong: `* weight` out of place -> the peak rises by one more output -> red."""
import pytest
import torch

from neurobrix.core.dtype.engine import rms_norm_fp32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the allocator's peak is read on a device")
def test_the_op_allocates_one_buffer_beyond_its_input_at_its_peak():
    x = torch.randn(64, 256, 1024, device="cuda", dtype=torch.float32)
    w = torch.randn(1024, device="cuda", dtype=torch.float32)
    one = x.numel() * x.element_size()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = rms_norm_fp32(x, w)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    # x.pow(2) is freed by its mean before the normalised buffer is made, and that buffer IS the output: one buffer
    assert extra <= one + (1 << 20), extra / one
    assert out.data_ptr() != x.data_ptr()
