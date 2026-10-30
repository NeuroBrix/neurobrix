"""Flow steps the compiled and the Triton handlers ran alike, body for body — written once (Hocine 2026-10-29 22:39).

One mixin per flow (VLM, audio, RNN-T, static graph), inherited by both of its handlers. Torch-free at import (the
Triton branch inherits them: R33); the tensors they touch are each branch's own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nbxtensor.tensor import NBXTensor


class VLMSteps:
    """The steps the compiled `VLMEngine` and the Triton `TritonVLMEngine` ran alike (core/flow/vlm.py, triton/flow/vlm.py)."""

    @staticmethod
    def _mrope_segments(seq_len: int, img_span: Tuple[int, int],
                        aud_span: Tuple[int, int],
                        vis_grid_llm: Optional[Tuple[int, int, int]]):
        """Segment layout for the M-RoPE builder (R30 mirror): text runs
        between the modality spans, the visual span as a t/h/w grid, the
        audio span text-like on the three planes."""
        spans = []
        if img_span[1] > img_span[0]:
            if vis_grid_llm is None:
                raise RuntimeError(
                    "ZERO FALLBACK: a visual span needs its llm grid "
                    "(t, h/merge, w/merge) for the M-RoPE builder.")
            spans.append((img_span[0], img_span[1], "visual", vis_grid_llm))
        if aud_span[1] > aud_span[0]:
            spans.append((aud_span[0], aud_span[1], "text", None))
        spans.sort(key=lambda s: s[0])
        segments: List[Tuple[str, int, Optional[Tuple[int, int, int, float]]]] = []
        cursor = 0
        for lo, hi, kind, grid in spans:
            if lo > cursor:
                segments.append(("text", lo - cursor, None))
            if kind == "visual":
                gt, gh, gw = grid
                segments.append(("visual", hi - lo, (gt, gh, gw, 1.0)))
            else:
                segments.append(("text", hi - lo, None))
            cursor = hi
        if cursor < seq_len:
            segments.append(("text", seq_len - cursor, None))
        return segments

    def _get_embed_weight(self, comp_name: str) -> Optional[NBXTensor]:
        executor = self.ctx.executors.get(comp_name)
        if executor is None:
            return None
        for key in executor._weights:
            if ("embed_tokens" in key or "token_embed" in key
                    or key == "embed.weight"):
                return executor._weights[key]
        # Second pass (R30 mirror) — the NeuroTax substring rule
        # ("embed" in key) with the declared vocab_size as the
        # discriminator: a rank-2 weight whose rows equal vocab_size IS
        # the token embedding (e.g. `word_embeddings.weight`). Additive:
        # builds resolved by the exact names above never reach it.
        vocab = (self.ctx.pkg.defaults.get("lm_config", {}) or {}).get(
            "vocab_size")
        if vocab is None:
            return None
        for key, tensor in executor._weights.items():
            if ("embed" in key and tensor is not None
                    and getattr(tensor, "ndim", 0) == 2
                    and int(tensor.shape[0]) == int(vocab)):
                return tensor
        return None


class AudioSteps:
    """The steps the compiled `AudioEngine` and the Triton `TritonAudioEngine` ran alike (core/flow/audio.py, triton/flow/audio.py)."""

    def _preprocess_input(self, input_config: Dict, direction: str, stages: List) -> None:
        """Preprocess input based on modality and preprocessing type."""
        modality = input_config.get("modality", "audio" if direction == "stt" else "text")

        if modality == "audio":
            self._preprocess_audio_input(input_config, stages)
        elif modality == "text":
            self._preprocess_text_input(input_config)

    def _postprocess_output(self, output_config: Dict, direction: str) -> None:
        """Postprocess output based on modality."""
        modality = output_config.get("modality", "text" if direction == "stt" else "audio")

        if modality == "text":
            self._postprocess_text_output(output_config)
        elif modality == "audio":
            self._postprocess_audio_output(output_config)

    def _postprocess_text_output(self, output_config: Dict) -> None:
        """Decode generated token IDs to text (STT)."""
        generated_ids = self.ctx.variable_resolver.resolved.get("global.generated_token_ids")
        if generated_ids is None:
            return

        tokenizer = self.ctx.modules.get("tokenizer")
        if tokenizer is not None:
            from neurobrix.core.module.audio.output_processor import AudioOutputProcessor
            text = AudioOutputProcessor.decode_tokens(generated_ids, tokenizer)
        else:
            text = str(generated_ids)

        variable = output_config.get("variable", "global.transcription")
        self.ctx.variable_resolver.resolved[variable] = text
        self.ctx.variable_resolver.resolved["global.transcription"] = text
        print(f"   [Output] Transcription: {text[:100]}{'...' if len(text) > 100 else ''}")


class RNNTSteps:
    """The steps the compiled `RNNTEngine` and the Triton `TritonRNNTEngine` ran alike (core/flow/rnnt.py, triton/flow/rnnt.py)."""

    def _get_component_output(self, comp_name: str):
        """Get the primary output tensor from a component."""
        resolved = self.ctx.variable_resolver.resolved
        for key in [f"{comp_name}.output_0", f"{comp_name}.output"]:
            if key in resolved:
                return resolved[key]
        executor = self.ctx.executors.get(comp_name)
        if executor and hasattr(executor, '_last_outputs') and executor._last_outputs:
            return executor._last_outputs[0]
        raise RuntimeError(f"ZERO FALLBACK: No output found for {comp_name}")

    def _find_tokenizer_model(self) -> Optional[Path]:
        """Find sentencepiece tokenizer.model in NBX cache."""
        cache_path = self.ctx.pkg.cache_path
        if cache_path is None:
            return None
        sp_paths = list(cache_path.rglob("*.model"))
        return sp_paths[0] if sp_paths else None


class StaticGraphSteps:
    """The steps the compiled `StaticGraphHandler` and the Triton `TritonStaticGraphHandler` ran alike (core/flow/static_graph.py, triton/flow/static_graph.py)."""

    def execute(self) -> Dict[str, Any]:
        """
        Execute static graph (no loop).

        Executes all components in order: pre_loop + loop.components + post_loop

        Returns:
            Dict of resolved variables/outputs
        """
        flow = self.ctx.pkg.topology.get("flow", {})

        # Execute all components in topological order
        all_components = (
            flow.get("pre_loop", []) +
            flow.get("loop", {}).get("components", []) +
            flow.get("post_loop", [])
        )

        for comp_name in all_components:
            self._execute_component(comp_name, "static", None)

            # Unload after each component
            self._unload_component(comp_name)

        return self.ctx.variable_resolver.resolve_all()
