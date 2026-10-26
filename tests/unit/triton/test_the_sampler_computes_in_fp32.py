"""The Triton sampler computes in fp32 from end to end (the owner, 2026-10-26 19:21, as vLLM): the generator widens the
logits once, whatever the head stored. What this test would do if the code were wrong: a half logits row reaches the
sampler in half — RED."""
import os
import numpy as np
import pytest
from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype
from neurobrix.triton import generator as G

pytestmark = pytest.mark.skipif(not os.environ.get("CUDA_VISIBLE_DEVICES"), reason="needs a card")


@pytest.mark.parametrize("dt", [NBXDtype.float16, NBXDtype.bfloat16])
def test_the_sampler_receives_fp32_logits(dt):
    seen = []
    gen = G.TritonGenerator.__new__(G.TritonGenerator) if hasattr(G, "TritonGenerator") else None
    if gen is None:
        pytest.skip("the generator class is not named TritonGenerator")
    gen._sampler = lambda logits, **kw: (seen.append(logits._dtype), logits.select(1, 0))[1]
    gen._prompt_ids, gen._state = [1], type("S", (), {"generated_tokens": [], "device_idx": 0})()
    logits = NBXTensor.from_numpy(np.arange(8, dtype=np.float32).reshape(1, 8)).to("cuda").to(dt)
    try:
        gen.step(logits, 0)
    except Exception:
        pass                                  # what follows the draw is not this test's business
    assert seen and seen[0] == NBXDtype.float32, seen
