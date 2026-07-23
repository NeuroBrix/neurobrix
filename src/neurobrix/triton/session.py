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
                 device_idx: int = 0,
                 position_ids_rank: int = 2,
                 visual_stub_specs: Optional[Dict[str, list]] = None):
        self.executor = executor
        self.kv_wrapper = kv_wrapper
        self.hidden_dim = hidden_dim
        self.graph_inputs = graph_inputs
        self.uses_embeds = uses_embeds
        self.uses_absolute_position = uses_absolute_position
        self.device_idx = device_idx
        self.position_ids_rank = position_ids_rank
        # R30 mirror of GraphLMSession.visual_stub_specs: DeepStack-lineage
        # graphs declare visual inputs a pure-text request never sets —
        # filled with empty stubs (all-False mask + zero-length embeds) at
        # every run site so the injection ops are exact no-ops.
        self.visual_stub_specs = visual_stub_specs or {}
        self._accumulated_ids: Optional[NBXTensor] = None
        # O(n)-fallback embeds accumulator (uses_embeds graphs — image-AR:
        # decode inputs are aligned VQ embeddings, NOT text tokens, so the
        # full-context re-run must replay the exact embedding stream).
        self._accumulated_embeds: Optional[NBXTensor] = None

    def _add_visual_stubs(self, run_inputs: Dict[str, NBXTensor]) -> None:
        """R30 mirror of GraphLMSession._add_visual_stubs — numpy build is
        allowed CPU glue (the stubs are decode-control data, not compute)."""
        if not self.visual_stub_specs:
            return
        stream = run_inputs.get("inputs_embeds")
        if stream is None:
            stream = run_inputs.get("input_ids")
        if stream is None:
            return
        b, s = stream.shape[0], stream.shape[1]
        DeviceAllocator.set_device(self.device_idx)
        for name, spec_shape in self.visual_stub_specs.items():
            if name in run_inputs:
                continue
            if name == "visual_pos_masks":
                h = spec_shape[2] if len(spec_shape) == 3 else self.hidden_dim
                run_inputs[name] = NBXTensor.from_numpy(
                    np.zeros((b, s, h), dtype=bool))
            else:  # deepstack_visual_embeds.N — [0, H] zero-length embeds
                h = spec_shape[1] if len(spec_shape) == 2 else self.hidden_dim
                run_inputs[name] = NBXTensor.from_numpy(
                    np.zeros((0, h), dtype=np.float16))

    def _lift_positions(self, pos_np: 'np.ndarray') -> 'np.ndarray':
        """Lift [B, S] positions to the graph's declared rank.

        R30 mirror of the compiled GraphLMSession._shape_positions: M-RoPE
        graphs declare position_ids [3, B, S] (temporal/height/width
        planes); a pure-text stream feeds three IDENTICAL planes (vendor
        get_rope_index semantics). Rank comes from the graph input spec —
        data-driven, no family/model branch (R15).
        """
        if self.position_ids_rank == 3 and pos_np.ndim == 2:
            return np.ascontiguousarray(
                np.broadcast_to(pos_np, (3,) + pos_np.shape))
        return pos_np

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
        position_ids = NBXTensor.from_numpy(self._lift_positions(pos_np))

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

        # Mark the pre-lm_head tensor as persistent BEFORE first run so the
        # triton sequence's liveness analysis keeps its slot alive. Janus-
        # style image-AR graphs declare text-vocab logits (shape[-1]=vocab,
        # not hidden_dim) as the graph output; without this the session
        # would fall back to returning the logits as "hidden states" and
        # feed the wrong shape into gen_head.
        if hasattr(self.executor, "enable_hidden_states_capture"):
            self.executor.enable_hidden_states_capture()

        # Prefill: distinct per-op output slots for the full prompt,
        # kill_slots MUST fire during the forward pass. Pass False
        # explicitly to document the intent (default is also False now
        # that the shape-based is_decode heuristic is gone — see
        # graph_executor._run_triton_compiled).
        self._add_visual_stubs(run_inputs)
        outputs = self.executor.run(run_inputs, skip_kills=False)

        # Switch to decode mode / init O(n) accumulator
        if self.kv_wrapper is not None:
            self.kv_wrapper.set_decode_mode()
        else:
            self._accumulated_ids = input_ids
            # uses_embeds graphs replay the embedding stream at O(n) decode
            # (image-AR feeds aligned VQ embeddings, not re-embedded ids).
            self._accumulated_embeds = (run_inputs.get("inputs_embeds")
                                        if self.uses_embeds else None)

        # Prefer the explicit pre-lm_head tensor via get_hidden_states;
        # fall back to the output-scan (graph output IS hidden states,
        # e.g. DeepSeek-MoE).
        hidden = None
        if hasattr(self.executor, "get_hidden_states"):
            hidden = self.executor.get_hidden_states(
                expected_hidden_dim=self.hidden_dim,
                expected_batch_size=batch_size,
            )
        if hidden is None:
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
            # O(n) fallback: accumulate tokens and re-run full context.
            # inputs_embeds MUST thread through (image-AR decode input is
            # the aligned VQ embedding — dropping it here re-embedded the
            # SHIFTED token id through the text table and fed the LM a
            # different stream than the KV-cache path, R30 asymmetry).
            return self._decode_step_full_context(input_ids, inputs_embeds)

        self.kv_wrapper.update_position_offset()
        cache_len = self.kv_wrapper.get_cache_len()

        # Position IDs
        if self.uses_absolute_position:
            pos_np = np.full((batch_size, seq_len), cache_len, dtype=np.int64)
        else:
            pos_np = np.zeros((batch_size, seq_len), dtype=np.int64)
        DeviceAllocator.set_device(self.device_idx)
        position_ids = NBXTensor.from_numpy(self._lift_positions(pos_np))

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

        # Decode step with KV cache: same-size intermediates every
        # step, arena slots are always overwritten before being read,
        # kill_slots are redundant work — skip them. This is the
        # ONE legitimate skip_kills=True site in the triton runtime
        # (mirrored below for the O(n) fallback).
        self._add_visual_stubs(run_inputs)
        outputs = self.executor.run(run_inputs, skip_kills=True)

        hidden = None
        if hasattr(self.executor, "get_hidden_states"):
            hidden = self.executor.get_hidden_states(
                expected_hidden_dim=self.hidden_dim,
                expected_batch_size=batch_size,
            )
        if hidden is None:
            hidden = self._extract_hidden(outputs)
        if hidden is None:
            raise RuntimeError("Decode step could not extract hidden_states.")
        while hidden.ndim > 3:
            hidden = hidden.squeeze(0)
        return hidden

    def _to_session_device(self, t: NBXTensor) -> NBXTensor:
        """Move a tensor to the session device if it lives elsewhere
        (e.g. produced by a head/aligner executor pinned to another GPU,
        or a zero3 CPU-offloaded producer). Delegates to the consolidated
        device_transfer brick — it materialises expanded views before the
        copy and picks H2D vs D2D by source, which a flat memcpy of
        `_nbytes` would get wrong."""
        from neurobrix.triton.device_transfer import needs_move, transfer_tensor
        if needs_move(t, self.device_idx):
            return transfer_tensor(t, self.device_idx)
        return t

    def _decode_step_full_context(self, new_token_ids: NBXTensor,
                                  new_embeds: Optional[NBXTensor] = None
                                  ) -> NBXTensor:
        """O(n) decode: concatenate all tokens and re-run full context.

        For uses_embeds graphs the full embedding sequence is accumulated
        and re-fed (image-AR: the decode input is the gen_embed→gen_aligner
        output for the sampled VQ code — re-embedding the shifted token id
        through the text table would diverge from the KV-cache path).
        """
        DeviceAllocator.ensure_triton_device(self.device_idx)
        if new_token_ids.ndim == 1:
            new_token_ids = new_token_ids.unsqueeze(0)
        # Ensure token is on the correct device (may be on wrong GPU after lm_head run)
        new_token_ids = self._to_session_device(new_token_ids)
        acc_ids = NBXTensor.cat([self._accumulated_ids, new_token_ids], dim=1)

        acc_embeds = None
        if self.uses_embeds:
            embeds = (new_embeds if new_embeds is not None
                      else self._embed_from_ids(new_token_ids))
            embeds = self._to_session_device(embeds)
            acc_embeds = NBXTensor.cat([self._accumulated_embeds, embeds],
                                       dim=1)

        # cat launches are ASYNC (raw kernel launch outside the dispatcher's
        # synchronous-dispatch contract) and rebinding the accumulator drops
        # the old buffer's last ref → free_cuda. On the default path cudaFree
        # blocks until device idle, so the in-flight cat kernel finishes
        # first — but under NBX_ALLOC_POOL=1 the pointer goes back to the
        # free-list WITHOUT a sync and the next malloc may hand the block to
        # a concurrent writer while the cat kernel still reads it. Sync
        # BEFORE the rebind so the O(n) accumulators stay pool-safe (this
        # path is the op-by-op determinism reference; one sync per step is
        # noise here).
        DeviceAllocator.sync_device()
        self._accumulated_ids = acc_ids
        if self.uses_embeds:
            self._accumulated_embeds = acc_embeds

        seq_len = self._accumulated_ids.shape[1]
        batch_size = self._accumulated_ids.shape[0]
        pos_np = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
        if batch_size > 1:
            pos_np = np.tile(pos_np, (batch_size, 1))
        position_ids = NBXTensor.from_numpy(self._lift_positions(pos_np))

        if self.uses_embeds:
            run_inputs = {"inputs_embeds": self._accumulated_embeds}
            if "position_ids" in self.graph_inputs:
                run_inputs["position_ids"] = position_ids
        elif "position_ids" in self.graph_inputs:
            run_inputs = {"input_ids": self._accumulated_ids,
                          "position_ids": position_ids}
        else:
            run_inputs = {"input_ids": self._accumulated_ids}

        # O(n) fallback decode (no KV cache). Logically a growing
        # prefill: the full context is re-run every step and the
        # per-op intermediates GROW with seq_len. skip_kills controls
        # ONLY intra-run liveness (whether kill_slots free dead
        # intermediates DURING the forward) — it does NOT affect the
        # cross-call slot-overwrite safety (output slots are rebound
        # at the start of each run regardless). The KV fast path above
        # can skip kills because its intermediates are fixed-size
        # (seq_len==1); here they are full-context and large, so
        # skipping kills retains every intermediate of the whole
        # forward and blows up VRAM as the context lengthens
        # (audit #2 F2). Run kills like prefill (skip_kills=False):
        # the deferred-drain sync still guarantees no UAF.
        self._add_visual_stubs(run_inputs)
        outputs = self.executor.run(run_inputs, skip_kills=False)
        # Prefer the explicit pre-lm_head capture (same cascade as prefill
        # and the KV-cache decode path — R30 symmetry; the former direct
        # _extract_hidden call skipped the capture and its first-output
        # fallback handed image-AR graphs their text-vocab logits).
        hidden = None
        if hasattr(self.executor, "get_hidden_states"):
            hidden = self.executor.get_hidden_states(
                expected_hidden_dim=self.hidden_dim,
                expected_batch_size=batch_size,
            )
        if hidden is None:
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
