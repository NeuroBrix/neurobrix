"""Both generators follow the request's decode constraint (core/state/constraint.py): before each step the tokens it
disallows are set to -inf (Triton: nbxtensor.ops.token_bitmask; compiled: its torch twin from the same words), so every
sampled token is one it allows — even when the model's best logit is a disallowed one — and each is reported back
(`advance`). Without a constraint nothing changes."""
import numpy as np
import pytest

from nbxtensor.ops.token_bitmask import token_bitmask

V = 151936
ALLOWED = [5, 9, 77, 150000]


class Fake:
    def __init__(self):
        self.seen = []

    def mask(self, vocab):
        assert vocab == V
        return token_bitmask(ALLOWED, vocab)[0]

    def advance(self, token_id):
        self.seen.append(token_id)


def _config():
    return {"max_tokens": 64, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
            "eos_token_id": None, "pad_token_id": None}


def _logits(rng):
    x = rng.standard_normal((1, V)).astype(np.float32)
    x[0, 1234] = 50.0                                  # the model's choice, which the constraint forbids
    return x


def test_the_compiled_generator_samples_only_allowed_tokens():
    torch = pytest.importorskip("torch")
    from neurobrix.core.module.autoregressive.generator import AutoregressiveGenerator
    g = AutoregressiveGenerator(_config())
    g.constraint = Fake()
    rng = np.random.default_rng(0)
    toks = [int(g.step(torch.from_numpy(_logits(rng)), i)[0].reshape(-1)[0]) for i in range(20)]
    assert set(toks) <= set(ALLOWED) and g.constraint.seen == toks
    plain = AutoregressiveGenerator({**_config(), "temperature": 0.0})
    assert int(plain.step(torch.from_numpy(_logits(rng)), 0)[0].reshape(-1)[0]) == 1234


def test_the_triton_generator_samples_only_allowed_tokens():
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.tensor import NBXTensor
    try:
        NBXTensor.zeros((1,), dtype=NBXDtype.float32, device="cuda:0")
    except Exception as e:
        pytest.skip(f"no accelerator: {e}")
    from neurobrix.triton.generator import TritonGenerator
    g = TritonGenerator(_config())
    g.set_generation_params(device_idx=0)
    g.constraint = Fake()
    rng = np.random.default_rng(1)
    toks = [int(g.step(NBXTensor.from_numpy(_logits(rng)), i)[0].item()) for i in range(20)]
    assert set(toks) <= set(ALLOWED) and g.constraint.seen == toks
