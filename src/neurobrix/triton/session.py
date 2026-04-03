"""Triton LM Session — zero torch dependency.

Ported from GraphLMSession in core/flow/autoregressive.py.
Manages prefill/decode lifecycle with TritonSequence + KV cache.
All tensors are NBXTensor. No torch imports.
"""

import numpy as np
from typing import Dict, List, Optional

from neurobrix.kernels.nbx_tensor import NBXTensor, DeviceAllocator
from neurobrix.kernels import wrappers as w


class TritonLMSession:
    """Encapsulates TritonSequence + KV cache lifecycle.

    Ported from GraphLMSession. Always uses KV cache (no O(n) fallback).
    All tensors are NBXTensor throughout.
    """

    def __init__(self, executor, kv_wrapper, hidden_dim: int,
                 graph_inputs: List[str], uses_embeds: bool,
                 uses_absolute_position: bool = False,
                 device_idx: int = 0):
        self.executor = executor
        self.kv_wrapper = kv_wrapper
        self.hidden_dim = hidden_dim
        self.graph_inputs = graph_inputs
        self.uses_embeds = uses_embeds
        self.uses_absolute_position = uses_absolute_position
        self.device_idx = device_idx
        self._accumulated_ids: Optional[NBXTensor] = None

    def prefill(self, input_ids: NBXTensor, batch_size: int) -> NBXTensor:
        """Run prefill pass, switch to decode mode, return hidden states."""
        if self.kv_wrapper is not None:
            self.kv_wrapper.reset()

        seq_len = input_ids.shape[1]

        # Position IDs: [0, 1, 2, ..., seq_len-1]
        pos_np = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        if batch_size > 1:
            pos_np = np.tile(pos_np, (batch_size, 1))
        DeviceAllocator.set_device(self.device_idx)
        position_ids = NBXTensor.from_numpy(pos_np)

        # Build inputs
        if self.uses_embeds:
            embed_weight = self.executor.get_embed_tokens()
            if embed_weight is None:
                raise RuntimeError("Graph expects inputs_embeds but no embedding weight found.")
            inputs_embeds = w.embedding(embed_weight, input_ids)
            run_inputs = {"inputs_embeds": inputs_embeds}
            if "position_ids" in self.graph_inputs:
                run_inputs["position_ids"] = position_ids
        elif "position_ids" in self.graph_inputs:
            run_inputs = {"input_ids": input_ids, "position_ids": position_ids}
        else:
            run_inputs = {"input_ids": input_ids}

        outputs = self.executor.run(run_inputs)

        # Switch to decode mode / init O(n) accumulator
        if self.kv_wrapper is not None:
            self.kv_wrapper.set_decode_mode()
        else:
            self._accumulated_ids = input_ids

        # Extract hidden states from outputs (NBXTensor)
        hidden = self._extract_hidden(outputs)
        if hidden is None:
            raise RuntimeError("Prefill could not extract hidden_states.")

        while hidden.ndim > 3:
            hidden = hidden.squeeze(0)

        return hidden

    def decode_step(self, input_ids: NBXTensor,
                    inputs_embeds: Optional[NBXTensor] = None) -> NBXTensor:
        """Single decode step. O(1) with KV cache, O(n) without."""
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        if self.kv_wrapper is None:
            # O(n) fallback: accumulate tokens and re-run full context
            return self._decode_step_full_context(input_ids)

        self.kv_wrapper.update_position_offset()
        cache_len = self.kv_wrapper.get_cache_len()

        # Position IDs
        if self.uses_absolute_position:
            pos_np = np.full((batch_size, seq_len), cache_len, dtype=np.int64)
        else:
            pos_np = np.zeros((batch_size, seq_len), dtype=np.int64)
        DeviceAllocator.set_device(self.device_idx)
        position_ids = NBXTensor.from_numpy(pos_np)

        # Build inputs
        if self.uses_embeds:
            embeds = inputs_embeds if inputs_embeds is not None else self._embed_from_ids(input_ids)
            run_inputs = {"inputs_embeds": embeds}
            if "position_ids" in self.graph_inputs:
                run_inputs["position_ids"] = position_ids
        elif "position_ids" in self.graph_inputs:
            run_inputs = {"input_ids": input_ids, "position_ids": position_ids}
        else:
            run_inputs = {"input_ids": input_ids}

        outputs = self.executor.run(run_inputs)

        hidden = self._extract_hidden(outputs)
        if hidden is None:
            raise RuntimeError("Decode step could not extract hidden_states.")
        while hidden.ndim > 3:
            hidden = hidden.squeeze(0)
        return hidden

    def _decode_step_full_context(self, new_token_ids: NBXTensor) -> NBXTensor:
        """O(n) decode: concatenate all tokens and re-run full context."""
        if new_token_ids.ndim == 1:
            new_token_ids = new_token_ids.unsqueeze(0)
        self._accumulated_ids = NBXTensor.cat(
            [self._accumulated_ids, new_token_ids], dim=1)

        seq_len = self._accumulated_ids.shape[1]
        batch_size = self._accumulated_ids.shape[0]

        DeviceAllocator.set_device(self.device_idx)
        pos_np = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        if batch_size > 1:
            pos_np = np.tile(pos_np, (batch_size, 1))
        position_ids = NBXTensor.from_numpy(pos_np)

        if "position_ids" in self.graph_inputs:
            run_inputs = {"input_ids": self._accumulated_ids,
                          "position_ids": position_ids}
        else:
            run_inputs = {"input_ids": self._accumulated_ids}

        outputs = self.executor.run(run_inputs)
        hidden = self._extract_hidden(outputs)
        if hidden is None:
            raise RuntimeError("O(n) decode could not extract hidden_states.")
        while hidden.ndim > 3:
            hidden = hidden.squeeze(0)
        return hidden

    def _extract_hidden(self, outputs: dict) -> Optional[NBXTensor]:
        """Extract hidden states from run outputs.

        The graph output with last dim == hidden_dim is the hidden states.
        """
        for name, tensor in outputs.items():
            if hasattr(tensor, 'shape') and tensor.shape[-1] == self.hidden_dim:
                return tensor
        # Fallback: return the first output
        for tensor in outputs.values():
            if hasattr(tensor, 'shape'):
                return tensor
        return None

    def _embed_from_ids(self, input_ids: NBXTensor) -> NBXTensor:
        """Lookup embedding for token IDs."""
        embed_weight = self.executor.get_embed_tokens()
        if embed_weight is None:
            raise RuntimeError("No embedding weight found for embed lookup.")
        return w.embedding(embed_weight, input_ids)

    def set_decode_mode(self):
        """Switch from prefill to decode mode."""
        if self.kv_wrapper is not None:
            self.kv_wrapper.set_decode_mode()

    def reset_for_new_sequence(self):
        """Reset KV cache for new generation."""
        if self.kv_wrapper is not None:
            self.kv_wrapper.reset()

    def get_cache_len(self) -> int:
        """Get current KV cache length for position tracking."""
        if self.kv_wrapper is not None:
            return self.kv_wrapper.get_cache_len()
        return 0

    def cleanup(self):
        """Release per-request resources."""
        if self.kv_wrapper is not None:
            self.kv_wrapper.reset()
        if self.executor is not None:
            self.executor.cleanup()
