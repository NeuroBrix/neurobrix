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
            # Select_index pattern: [BOS token] + [last N-1 tokens]
            # The CHI prefix occupies the first ~208 tokens. With padding="max_length",
            # the user's actual prompt content is at the END of the sequence.
            # This pattern keeps the BOS token (position 0) and the last (N-1) tokens
            # which contain the user prompt, discarding the CHI prefix in the middle.
            # This matches the original Sana pipeline behavior exactly.
            orig_seq_len = hidden_state.shape[1]
            select_index = [0] + list(range(-max_sequence_length + 1, 0))
            hidden_state = hidden_state[:, select_index]

            # Apply same slicing to attention_mask if present
            if attention_mask is not None and attention_mask.shape[1] > max_sequence_length:
                attention_mask = attention_mask[:, select_index]

        result["hidden_state"] = hidden_state
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
        return result
