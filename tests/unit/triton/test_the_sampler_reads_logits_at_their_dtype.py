"""The repetition penalty reads and writes the logits at THEIR dtype (2026-10-26, gate tha2: orpheus's lm_head stored
fp16 logits under the fp16 contract and the fp32-sized copy read past them — CUDA error 1, the run failed). What this
test would do if the code were wrong: the fp16 / bf16 cases read twice the bytes (a CUDA error) or decode garbage."""
import numpy as np
import pytest
from nbxtensor.tensor import NBXTensor, NBXDtype
from neurobrix.triton import samplers as S

pytestmark = pytest.mark.skipif(not __import__("os").environ.get("CUDA_VISIBLE_DEVICES"), reason="needs a card")


@pytest.mark.parametrize("dt", [NBXDtype.float32, NBXDtype.float16, NBXDtype.bfloat16])
def test_the_penalty_divides_seen_positive_logits_and_keeps_the_dtype(dt):
    vals = np.array([2.0, -2.0, 4.0, 1.0], dtype=np.float32)
    logits = NBXTensor.from_numpy(vals).to("cuda").to(dt)
    ids = NBXTensor.from_numpy(np.array([0, 1], dtype=np.int64)).to("cuda")
    out = S._decode_float(S._host_bytes(logits, 4), dt)
    assert np.allclose(out, vals)                                   # the decode is exact for these values
    sampler = S.__dict__[[n for n in dir(S) if hasattr(getattr(S, n), "_apply_repetition_penalty")][0]].__new__(
        S.__dict__[[n for n in dir(S) if hasattr(getattr(S, n), "_apply_repetition_penalty")][0]])
    sampler.repetition_penalty = 2.0
    res = sampler._apply_repetition_penalty(logits, ids)
    assert res._dtype == dt
    got = S._decode_float(S._host_bytes(res, 4), dt)
    assert np.allclose(got, [1.0, -4.0, 4.0, 1.0]), got
