"""The VLM flows project their logits at the head weight's dtype (the DtypeEngine placed the weight there): the hidden
row is cast to it, the weight never to the hidden's. MiniCPM-o-4_5 triton on a V100-16GB (gate of 2026-11-01) was
refused the 2.4 GB fp32 copy of its 151 748 x 4 096 head the old projection made at every token; the compiled flow made
the same copy.

What each test would do if the code were wrong: a cast of the head weight is recorded (red); the logits' dtype is the
weight's, not the hidden's (red). The compiled one card-less; the Triton one on a card."""
import numpy as np
import pytest
import torch

from neurobrix.core.flow.vlm import VLMEngine


def test_the_compiled_head_casts_the_hidden_row_never_the_weight(monkeypatch):
    weight = torch.randn(64, 16).to(torch.float16)
    hidden = torch.randn(1, 1, 16)
    cast = []
    real_to = torch.Tensor.to

    def spy(self, *a, **k):
        dt = k.get("dtype") or next((x for x in a if isinstance(x, torch.dtype)), None)
        if dt is not None and self.shape == weight.shape and dt != self.dtype:
            cast.append(dt)
        return real_to(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "to", spy)
    logits = VLMEngine._head_logits(hidden, weight)
    assert cast == [], f"the head weight was cast to {cast}"
    assert logits.dtype == hidden.dtype and logits.shape == (1, 1, 64)
    ref = (hidden.to(torch.float16).float() @ weight.float().T)
    assert torch.allclose(logits, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.usefixtures("needs_device")
def test_the_triton_head_casts_the_hidden_row_never_the_weight(monkeypatch):
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    from neurobrix.triton.flow.vlm import TritonVLMEngine

    w_np = np.random.RandomState(0).randn(64, 16).astype(np.float16)
    h_np = np.random.RandomState(1).randn(1, 1, 16).astype(np.float32)
    weight = NBXTensor.from_numpy(w_np).to("cuda:0")
    hidden = NBXTensor.from_numpy(h_np).to("cuda:0")
    cast = []
    real_to = NBXTensor.to

    def spy(self, *a, **k):
        if tuple(self.shape) == tuple(weight.shape) and any(x == NBXDtype.float32 for x in a):
            cast.append(a)
        return real_to(self, *a, **k)

    monkeypatch.setattr(NBXTensor, "to", spy)
    eng = TritonVLMEngine.__new__(TritonVLMEngine)
    logits = eng._compute_logits(hidden, weight, "embed", None)
    assert cast == [], "the head weight was cast to float32"
    assert logits._dtype == NBXDtype.float32 and tuple(logits.shape) == (1, 1, 64)
    ref = h_np.reshape(1, 16).astype(np.float16).astype(np.float32) @ w_np.astype(np.float32).T
    assert np.allclose(logits.to_cpu().numpy().reshape(1, 64), ref, atol=2e-2, rtol=2e-2)
