"""The Triton sampler computes in fp32 from end to end (the owner, 2026-10-26 19:21, as vLLM): the generator widens the
logits once, whatever the head stored. What this test would do if the code were wrong: a half logits row reaches the
sampler in half — RED."""
import os
import numpy as np
import pytest
from nbxtensor.tensor import NBXTensor, NBXDtype
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


@pytest.mark.parametrize("dt", [NBXDtype.float16, NBXDtype.bfloat16])
@pytest.mark.parametrize("helper", ["audio_llm", "encoder_decoder"])
def test_the_flow_samplers_draw_in_fp32(dt, helper):
    """The flows' own samplers (audio_llm's, which vlm also draws with, and the encoder-decoder's): a half row whose
    largest logit divided by the temperature is past the half range (70 / 1e-3 = 70 000 > 65 504) — in half it becomes
    inf, the softmax NaN and the draw garbage; in fp32 the draw is the argmax, token 7."""
    import importlib
    mod = importlib.import_module(f"neurobrix.triton.flow.{helper}")
    logits = NBXTensor.from_numpy((np.arange(8, dtype=np.float32) * 10).reshape(1, 8)).to("cuda").to(dt)
    assert [mod._sample_token_nbx(logits, 1e-3) for _ in range(4)] == [7, 7, 7, 7]
