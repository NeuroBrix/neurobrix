"""
Text Encoder Component Handler

Handles T5, CLIP, Gemma, and other text encoder variants.

Responsibilities:
- Provide max sequence length from config
- Handle output routing for CFG negative encoding
- Support pooled output extraction (for CLIP)

ZERO HARDCODE: All values from config, none hardcoded.
"""

from typing import Dict, Any, Optional, List

import torch

from ..base import ComponentHandler, ComponentConfig
from ..registry import register_handler


def _cat(tensors, dim):
    """Universal concatenation across torch.Tensor and NBXTensor.

    NBXTensor exposes a class-level cat() classmethod (see
    src/neurobrix/kernels/nbx_tensor.py); torch.Tensor does not, so we
    dispatch on the runtime type of the first tensor.
    """
    if hasattr(type(tensors[0]), 'cat'):
        return type(tensors[0]).cat(tensors, dim=dim)
    return torch.cat(tensors, dim=dim)


@register_handler("text_encoder")
class TextEncoderComponentHandler(ComponentHandler):
    """
    Handles all text encoder variants: T5, CLIP, Gemma, etc.

    DATA-DRIVEN:
    - max_position_embeddings from profile.json
    - hidden_size from profile.json
    - vocab_size from profile.json
    - pooler_output extraction for CLIP-style encoders
    """

    # Supported text encoder class names
    SUPPORTED_CLASSES = {
        "T5EncoderModel",
        "T5Model",
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
        "GemmaModel",
        "Gemma2Model",
        "LlamaModel",
        "Qwen2Model",
    }

    # Classes that produce pooled output (for conditioning)
    POOLED_OUTPUT_CLASSES = {
        "CLIPTextModel",
        "CLIPTextModelWithProjection",
    }

    @classmethod
    def can_handle(cls, class_name: str, component_type: str) -> bool:
        """Check if this handler supports the component."""
        if component_type == "text_encoder":
            return True
        if class_name in cls.SUPPORTED_CLASSES:
            return True
        # Handle class name variations
        class_lower = class_name.lower()
        return (
            "t5" in class_lower or
            "clip" in class_lower or
            "gemma" in class_lower or
            "llama" in class_lower or
            "encoder" in class_lower
        )

    def get_max_sequence_length(self) -> int:
        """
        Get maximum sequence length from config.

        Uses max_position_embeddings from profile.json.
        Falls back to common values based on class if not set.

        Returns:
            Maximum sequence length
        """
        # Try config first (DATA-DRIVEN)
        if self.config.max_position_embeddings is not None:
            return self.config.max_position_embeddings

        # Try raw profile config
        max_len = self.config.get("max_position_embeddings")
        if max_len is not None:
            return max_len

        # Try n_positions for older model configs
        max_len = self.config.get("n_positions")
        if max_len is not None:
            return max_len

        # ZERO FALLBACK: Crash explicitly if we cannot determine max sequence length
        # Do NOT guess based on class name - that breaks universality
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine max_sequence_length for {self.config.class_name}. "
            "Missing 'max_position_embeddings' and 'n_positions' in profile.json. "
            "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
        )

    def get_hidden_size(self) -> int:
        """
        Get hidden size from config.

        Returns:
            Hidden dimension size
        """
        if self.config.hidden_size is not None:
            return self.config.hidden_size

        # Try alternative key names
        hidden = self.config.get("d_model")  # T5 uses d_model
        if hidden is not None:
            return hidden

        hidden = self.config.get("hidden_size")
        if hidden is not None:
            return hidden

        # ZERO FALLBACK: Should not reach here if config is complete
        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine hidden_size for {self.config.class_name}. "
            f"Missing hidden_size or d_model in profile.json"
        )

    def produces_pooled_output(self) -> bool:
        """
        Check if this encoder produces pooled output.

        CLIP-style encoders produce pooled_output in addition to
        last_hidden_state, which is used for conditioning.

        Returns:
            True if encoder produces pooled_output
        """
        return self.config.class_name in self.POOLED_OUTPUT_CLASSES

    @staticmethod
    def preprocess_prompt(
        prompt: str,
        tokenizer_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Preprocess prompt with model-specific modifications.

        UNIVERSAL: Handles model-specific prompt preprocessing centrally.
        - Sana with CHI: Prepends complex_human_instruction to prompt
        - PixArt/other: Returns prompt unchanged

        This method is the SINGLE SOURCE OF TRUTH for prompt preprocessing.
        IterativeProcessHandler calls this before tokenization.

        Args:
            prompt: User's input prompt
            tokenizer_config: Tokenizer configuration (extracted_values)

        Returns:
            Preprocessed prompt (possibly with CHI prefix)
        """
        if tokenizer_config is None:
            return prompt

        complex_human_instruction = tokenizer_config.get("complex_human_instruction")

        if complex_human_instruction and isinstance(complex_human_instruction, list):
            chi_prompt = "\n".join(complex_human_instruction)
            processed = chi_prompt + prompt
            return processed

        return prompt

    def finalize_embeddings(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize embeddings after encoding.

        UNIVERSAL: Handles model-specific slicing logic centrally.
        - Sana with CHI: 506 tokens -> [0] + last 299 = 300 tokens (BOS + user content)
        - PixArt/other: No modification needed

        This method is the SINGLE SOURCE OF TRUTH for embedding finalization.
        Called for BOTH positive and negative embeddings during CFG.

        Args:
            hidden_state: Output hidden states [batch, seq_len, dim]
            attention_mask: Optional attention mask [batch, seq_len]
            tokenizer_config: Tokenizer configuration (extracted_values)

        Returns:
            Dict containing 'hidden_state' and 'attention_mask' (potentially modified)
        """
        result: Dict[str, torch.Tensor] = {}
        if tokenizer_config is None:
            result["hidden_state"] = hidden_state
            if attention_mask is not None:
                result["attention_mask"] = attention_mask
            return result

        complex_human_instruction = tokenizer_config.get("complex_human_instruction")
        max_sequence_length = tokenizer_config.get("max_sequence_length", 300)

        # Check if slicing is needed (Sana-style CHI handling)
        if complex_human_instruction and hidden_state.shape[1] > max_sequence_length:
            # Sana CHI slicing: keep [BOS token at index 0] + [last (N-1)
            # tokens] which contain the user prompt. The CHI prefix in the
            # middle is discarded. Matches the HuggingFace
            # Sana_1600M_1024px_MultiLing pipeline behavior.
            #
            # Implemented via narrow + cat instead of fancy indexing
            # (hidden_state[:, list_of_indices]) because NBXTensor in
            # --triton mode does not support fancy indexing by Python list.
            # narrow + cat is universally compatible with both torch.Tensor
            # and NBXTensor and produces a single contiguous output ready
            # for downstream ops.
            orig_seq_len = hidden_state.shape[1]
            tail_len = max_sequence_length - 1
            tail_start = orig_seq_len - tail_len
            first_token = hidden_state.narrow(1, 0, 1)
            last_chunk = hidden_state.narrow(1, tail_start, tail_len)
            hidden_state = _cat([first_token, last_chunk], dim=1)

            # Apply same slicing to attention_mask if present
            if attention_mask is not None and attention_mask.shape[1] > max_sequence_length:
                mask_first = attention_mask.narrow(1, 0, 1)
                mask_last = attention_mask.narrow(1, tail_start, tail_len)
                attention_mask = _cat([mask_first, mask_last], dim=1)

        # Zero the padded text embeddings (data-driven via tokenizer_config flag).
        # T5/UMT5 encoders emit NON-ZERO embeddings for pad tokens; a DiT that
        # cross-attends to the full sequence without masking would attend to that
        # padding and dilute the conditioning. The vendor pipelines trim each
        # prompt to its real length and re-pad with ZEROS (diffusers Wan
        # `_get_t5_prompt_embeds`). Reproduce that by masking the embedding with
        # the attention_mask. Gated by `zero_pad_embeddings` so models whose DiT
        # already masks (or that slice via CHI) are untouched (R23).
        if (tokenizer_config.get("zero_pad_embeddings")
                and attention_mask is not None
                and attention_mask.shape[1] == hidden_state.shape[1]):
            _m = attention_mask.to(hidden_state.dtype).unsqueeze(-1)
            hidden_state = hidden_state * _m

        # Extend the (zeroed) sequence up to max_sequence_length with zeros, to
        # match the vendor's trim+re-pad-to-max contract. The Wan I2V/T2V DiTs
        # cross-attend to the FULL max_sequence_length UNMASKED (no attention_mask
        # reaches the transformer blocks), so the COUNT of trailing zero-pad
        # tokens is part of the trained conditioning — diffusers Wan
        # `_get_t5_prompt_embeds` pads every prompt to max_sequence_length. The
        # text-encoder graph was traced at a shorter stimulus length (e.g. 226),
        # so the runtime output is shorter than the model's design length; pad it
        # up. Safe because the DiT's encoder_hidden_states seq dim is symbolic
        # (verified in the Wan transformer graph). Gated by `zero_pad_embeddings`
        # (R23 inert for every other model); `.new_zeros` works for both
        # torch.Tensor and NBXTensor, so this holds in compiled AND triton (R30).
        if (tokenizer_config.get("zero_pad_embeddings")
                and hidden_state.shape[1] < max_sequence_length):
            pad_len = max_sequence_length - hidden_state.shape[1]
            b, _seq, d = hidden_state.shape
            hidden_state = _cat(
                [hidden_state, hidden_state.new_zeros((b, pad_len, d))], dim=1)
            if (attention_mask is not None
                    and attention_mask.shape[1] < max_sequence_length):
                attention_mask = _cat(
                    [attention_mask, attention_mask.new_zeros((b, pad_len))], dim=1)

        result["hidden_state"] = hidden_state
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
        return result
