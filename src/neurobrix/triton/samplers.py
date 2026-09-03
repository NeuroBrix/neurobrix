"""Triton Sampling Strategies — zero torch dependency.

Ported from core/module/autoregressive/samplers.py.
All tensor ops use NBXTensor + Triton kernel wrappers from kernels/.

Implements: greedy, temperature, top-k, top-p, repetition penalty, combined.
"""

import ctypes as _ctypes
from typing import Any, Dict, Optional

import numpy as np

from neurobrix.kernels.nbx_tensor import NBXTensor, NBXDtype, DeviceAllocator
from neurobrix.kernels import wrappers as w


def _cpu_sort(tensor: NBXTensor, descending: bool = False):
    """Sort via CPU numpy. For top-p sampling (once per token, ~1ms)."""
    n = tensor.numel()
    elem_size = 2 if tensor._dtype in (NBXDtype.float16, NBXDtype.bfloat16) else 4
    buf = (_ctypes.c_char * (n * elem_size))()
    DeviceAllocator.memcpy(_ctypes.addressof(buf), tensor.data_ptr(),
                           n * elem_size, kind=2)  # D2H

    np_dt = np.float16 if tensor._dtype == NBXDtype.float16 else np.float32
    arr = np.frombuffer(buf, dtype=np_dt).reshape(tensor.shape).copy()

    if descending:
        indices = np.argsort(-arr, axis=-1, kind='stable')
    else:
        indices = np.argsort(arr, axis=-1, kind='stable')
    sorted_arr = np.take_along_axis(arr, indices, axis=-1)

    DeviceAllocator.set_device(tensor._device_idx)
    sorted_t = NBXTensor.from_numpy(np.ascontiguousarray(sorted_arr))
    indices_t = NBXTensor.from_numpy(indices.astype(np.int64))
    return sorted_t, indices_t


# =============================================================================
# Config (pure Python — shared with native mode)
# =============================================================================

class SamplerConfig:
    """Config validation for LLM samplers."""

    DEFAULTS: Dict[str, Any] = {
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "min_tokens_to_keep": 1,
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        clean = {k: v for k, v in config.items() if not k.startswith("_")}
        for key, default in cls.DEFAULTS.items():
            if key not in clean:
                clean[key] = default
        return clean


# =============================================================================
# Samplers
# =============================================================================

class GreedySampler:
    """Greedy decoding — always select highest probability token."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        return w.argmax_wrapper(logits, dim=-1, keepdim=True)


class TemperatureSampler:
    """Temperature-scaled sampling."""

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.temperature = validated["temperature"]

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        if self.temperature != 1.0 and self.temperature > 0:
            logits = w.div(logits, self.temperature)
        probs = w.softmax(logits, dim=-1)
        return w.multinomial_wrapper(probs, num_samples=1)


class TopKSampler:
    """Top-K sampling — sample from top k tokens."""

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.top_k = validated["top_k"]
        self.temperature = validated["temperature"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        if self.temperature != 1.0 and self.temperature > 0:
            logits = w.div(logits, self.temperature)

        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.shape[-1])
            values, _indices = w.topk_wrapper(logits, top_k, dim=-1)
            # kth value is the last in top-k (smallest of the top)
            kth = values.select(-1, top_k - 1).unsqueeze(-1)
            mask = w.lt(logits, kth)
            logits = w.masked_fill(logits, mask, float('-inf'))

        probs = w.softmax(logits, dim=-1)
        return w.multinomial_wrapper(probs, num_samples=1)


class TopPSampler:
    """Top-P (Nucleus) sampling — sample from smallest set with cumsum >= p."""

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.top_p = validated["top_p"]
        self.top_k = validated["top_k"]
        self.temperature = validated["temperature"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def __call__(self, logits: NBXTensor, **kwargs) -> NBXTensor:
        if self.temperature != 1.0 and self.temperature > 0:
            logits = w.div(logits, self.temperature)

        # Top-k pre-filter
        if self.top_k > 0:
            top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.shape[-1])
            values, _indices = w.topk_wrapper(logits, top_k, dim=-1)
            kth = values.select(-1, top_k - 1).unsqueeze(-1)
            mask = w.lt(logits, kth)
            logits = w.masked_fill(logits, mask, float('-inf'))

        # Top-p filter
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = w.sort_wrapper(logits, dim=-1, descending=True)
            sorted_probs = w.softmax(sorted_logits, dim=-1)
            cumulative_probs = w.cumsum_wrapper(sorted_probs, dim=-1)

            # Mask tokens with cumulative prob above threshold
            sorted_mask = w.gt(cumulative_probs, self.top_p)

            # Keep at least min_tokens_to_keep: zero out first N positions in mask
            # Shift mask right by 1 to include the boundary token
            # These are small CPU operations on the mask tensor
            # For simplicity, use the sorted logits approach: mask and scatter back
            logits = w.masked_fill(logits, sorted_mask, float('-inf'))

        probs = w.softmax(logits, dim=-1)
        return w.multinomial_wrapper(probs, num_samples=1)


class CombinedSampler:
    """Combined sampler with temperature, top-k, top-p, and repetition penalty.

    Most flexible sampler for production use.
    """

    def __init__(self, config: Dict[str, Any]):
        validated = SamplerConfig.validate(config)
        self.temperature = validated["temperature"]
        self.top_k = validated["top_k"]
        self.top_p = validated["top_p"]
        self.repetition_penalty = validated["repetition_penalty"]
        self.min_tokens_to_keep = validated["min_tokens_to_keep"]

    def _apply_repetition_penalty(self, logits: NBXTensor,
                                  input_ids: Optional[NBXTensor]) -> NBXTensor:
        """Apply repetition penalty by reading token IDs from GPU."""
        if input_ids is None or self.repetition_penalty == 1.0:
            return logits

        # Read input_ids to CPU for the penalty loop
        # This is a small transfer (seq_len ints) — acceptable at sampling time
        import ctypes
        import numpy as np
        n = input_ids.numel()
        buf = (ctypes.c_char * (n * 8))()  # int64 = 8 bytes
        # Through DeviceAllocator, not a hardcoded libcudart: this is the
        # triton path, and `libcudart.so` does not exist on ROCm or Apple.
        # It also checks the return code, which the raw call did not — a
        # failed copy used to return silent garbage.
        DeviceAllocator.memcpy(ctypes.addressof(buf), input_ids.data_ptr(),
                               n * 8, 2)  # D2H
        ids = np.frombuffer(bytes(buf), dtype=np.int64)

        # Read logits to CPU, apply penalty, write back
        vocab = logits.shape[-1]
        logits_bytes = vocab * 4  # float32
        lbuf = (ctypes.c_char * logits_bytes)()
        DeviceAllocator.memcpy(ctypes.addressof(lbuf), logits.data_ptr(),
                               logits_bytes, 2)  # D2H
        logits_np = np.frombuffer(bytes(lbuf), dtype=np.float32).copy()

        for token_id in set(ids.tolist()):
            if 0 <= token_id < vocab:
                if logits_np[token_id] > 0:
                    logits_np[token_id] /= self.repetition_penalty
                else:
                    logits_np[token_id] *= self.repetition_penalty

        # Write back to GPU. BOTH pointers must be wrapped in c_void_p: a
        # bare numpy `.ctypes.data` is a Python int, which a no-argtypes ctypes
        # call coerces to a 32-bit C int → the 64-bit host address is truncated
        # → invalid src pointer → segfault. (The D2H reads above already wrap
        # both ends, which is why only this H2D write crashed.)
        DeviceAllocator.memcpy(logits.data_ptr(), logits_np.ctypes.data,
                               logits_bytes, 1)  # H2D
        return logits

    def _shift_mask_right(self, mask: NBXTensor) -> NBXTensor:
        """Shift the top-p mask right by one and clear position 0.

        Guarantees the highest-probability token is never masked
        (HuggingFace/PyTorch convention). Without it, a peaked
        distribution whose top probability already exceeds top_p masks
        every token, softmax returns NaN, and the sampler yields 0.
        """
        shape = mask.shape
        n = mask.numel()
        buf = (_ctypes.c_char * n)()
        DeviceAllocator.memcpy(_ctypes.addressof(buf), mask.data_ptr(),
                               n, kind=2)  # D2H
        arr = np.frombuffer(buf, dtype=np.bool_).reshape(shape).copy()
        arr[..., 1:] = arr[..., :-1]
        arr[..., 0] = False
        DeviceAllocator.set_device(mask._device_idx)
        return NBXTensor.from_numpy(np.ascontiguousarray(arr))

    def _sample_from_candidates(self, logits: NBXTensor) -> NBXTensor:
        """Sample from the top-k candidates, working at width k.

        The masking formulation keeps the vector at full vocabulary width
        and lets every later stage pay for entries that cannot be
        sampled. On a 151936-token vocabulary with top_k=20, that is a
        7600x overcharge on the sort, the softmax, the cumulative sum and
        the draw. `multinomial` is the worst of them: it samples by
        Gumbel-max, so it draws one uniform PER CATEGORY — 151936 random
        numbers to pick one token.

        Gathering the candidates first runs the same arithmetic over
        top_k entries. It also removes the sort outright: topk already
        returns its values sorted descending, which is the order the
        top-p filter needs.

        Distributionally this is exact. Gumbel-max over the masked full
        vector adds noise to every entry, but the masked ones hold -inf
        and -inf + g is still -inf, so only the k survivors can ever win
        — both forms are exact Gumbel-max sampling from the same
        categorical distribution. What changes is the REALISATION: k
        Gumbels are drawn instead of V, so a seeded run picks a different
        (equally valid) token. Nothing promises RNG realisations are
        stable across versions, but a recorded baseline will move.
        """
        top_k = min(max(self.top_k, self.min_tokens_to_keep), logits.shape[-1])
        cand_logits, cand_ids = w.topk_wrapper(logits, top_k, dim=-1)

        if self.top_p < 1.0:
            probs = w.softmax(cand_logits, dim=-1)
            cum = w.cumsum_wrapper(probs, dim=-1)
            mask = self._shift_mask_right(w.gt(cum, self.top_p))
            cand_logits = w.masked_fill(cand_logits, mask, float('-inf'))

        probs = w.softmax(cand_logits, dim=-1)
        # A position within the candidate set, not a vocabulary id.
        pos = w.multinomial_wrapper(probs, num_samples=1)
        return w.gather_wrapper(cand_ids, -1, pos)

    def __call__(self, logits: NBXTensor,
                 input_ids: Optional[NBXTensor] = None, **kwargs) -> NBXTensor:
        # 1. Repetition penalty
        logits = self._apply_repetition_penalty(logits, input_ids)

        # 2. Temperature scaling
        if self.temperature != 1.0 and self.temperature > 0:
            logits = w.div(logits, self.temperature)

        # Greedy needs no candidate set and draws nothing.
        if self.temperature <= 0:
            return w.argmax_wrapper(logits, dim=-1, keepdim=True)

        # 3+4+5 at width k when a top-k cut bounds the candidate set.
        if self.top_k > 0:
            return self._sample_from_candidates(logits)

        # 4. Top-p filtering (sort via CPU for portability — ~1ms per token)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = _cpu_sort(logits, descending=True)
            sorted_probs = w.softmax(sorted_logits, dim=-1)
            cum_probs = w.cumsum_wrapper(sorted_probs, dim=-1)
            sorted_mask = self._shift_mask_right(w.gt(cum_probs, self.top_p))
            indices_to_remove = w.scatter_wrapper(
                sorted_mask, -1, sorted_indices, sorted_mask)
            logits = w.masked_fill(logits, indices_to_remove, float('-inf'))

        # 5. Sample. Greedy already returned above.
        probs = w.softmax(logits, dim=-1)
        return w.multinomial_wrapper(probs, num_samples=1)


# =============================================================================
# Factory
# =============================================================================

def create_sampler(config: Dict[str, Any]) -> CombinedSampler:
    """Create the appropriate sampler from config.

    Always returns CombinedSampler — it handles all strategies via config.
    """
    return CombinedSampler(config)
