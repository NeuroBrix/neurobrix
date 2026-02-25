"""
TextProcessor — Unified Tokenization Brick

ZERO SEMANTIC: No domain knowledge. Tokenizes prompts using data-driven config.
ZERO HARDCODE: max_length, chat_mode, SFT format all from NBX container.

Consolidates tokenization from:
- autoregressive.py:_tokenize_prompt() (chat template, SFT, basic)
- iterative_process.py:_preprocess_inputs() (diffusion tokenization)

Usage:
    tp = TextProcessor(tokenizer, defaults, topology, variable_resolver)
    input_ids = tp.tokenize(prompt, device)
    neg_ids = tp.tokenize_negative(device, encoder_name="text_encoder")
"""

import torch
from typing import Any, Dict, Optional, Tuple


class TextProcessor:
    """
    Unified tokenization brick for all NeuroBrix flow types.

    Handles:
    - HuggingFace chat_template (LLMs with chat)
    - SFT format (Janus-style image generation)
    - Basic tokenization (base models, diffusion)
    - Negative/unconditional tokenization (CFG)
    - Max length cascade (tokenizer config > graph shape > family default)
    """

    def __init__(
        self,
        tokenizer: Any,
        defaults: Dict[str, Any],
        topology: Dict[str, Any],
        variable_resolver: Any,
        tokenizer_name: str = "tokenizer",
    ):
        self._tokenizer = tokenizer
        self._defaults = defaults
        self._topology = topology
        self._resolver = variable_resolver
        self._tokenizer_name = tokenizer_name

        # Pre-compute extracted values
        self._tokenizer_vals = topology.get("extracted_values", {}).get(tokenizer_name, {})

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_max_sequence_length(self, encoder_name: str = "text_encoder") -> int:
        """
        Get max tokenization length from config cascade.

        Priority:
        1. Graph shape from topology (trace-time stimulus length — the size the
           encoder graph actually expects). For models with CHI (Sana), the graph
           was traced with input_ids shape [1, 506] (CHI prefix + user prompt).
           Tokenization MUST match the graph's expected input shape.
        2. max_sequence_length from tokenizer config (model design intent)
        3. ZERO FALLBACK crash

        Args:
            encoder_name: Text encoder component name (for graph shape lookup)

        Returns:
            Max sequence length
        """
        # Priority 1: Graph shape (trace-time stimulus length)
        text_encoder_shapes = (
            self._topology.get("components", {})
            .get(encoder_name, {})
            .get("shapes", {})
        )
        if "input_ids" in text_encoder_shapes:
            return int(text_encoder_shapes["input_ids"][1])

        # Priority 2: max_sequence_length from tokenizer config
        max_length = self._tokenizer_vals.get("max_sequence_length")
        if max_length is not None:
            return int(max_length)

        raise RuntimeError(
            f"ZERO FALLBACK: Cannot determine max_sequence_length for tokenization. "
            f"Missing 'input_ids' in topology.components.{encoder_name}.shapes "
            f"and 'max_sequence_length' in topology.extracted_values.{self._tokenizer_name}. "
            "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
        )

    def tokenize(
        self,
        prompt: str,
        device: str,
        is_unconditional: bool = False,
        chat_mode: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Tokenize prompt using the data-driven priority cascade.

        Priority:
        1. HuggingFace chat_template (models with chat_template)
        2. SFT format (Janus-style image generation)
        3. Basic tokenization (fallback)

        Args:
            prompt: Text prompt to tokenize
            device: Target device string
            is_unconditional: If True, create unconditional tokens for CFG
            chat_mode: Override chat mode (None = use defaults)

        Returns:
            input_ids tensor [1, seq_len]
        """
        # Resolve chat_mode: CLI override > defaults > False
        if chat_mode is None:
            _SENTINEL = object()
            cli_chat = self._resolver.get("global.chat_mode", default=_SENTINEL)
            if cli_chat is not _SENTINEL:
                chat_mode = bool(cli_chat)
            else:
                chat_mode = self._defaults["chat_mode"]

        has_template = hasattr(self._tokenizer, 'has_chat_template') and self._tokenizer.has_chat_template()
        # Priority 1: HuggingFace chat_template
        if (chat_mode
            and has_template
            and not is_unconditional):
            messages = [{"role": "user", "content": prompt}]
            token_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            return input_ids

        # Priority 2: SFT format (Janus-style)
        sft_format = self._defaults.get("sft_format")
        special_token_ids = self._defaults.get("special_token_ids")

        if sft_format and special_token_ids and hasattr(self._tokenizer, 'format_generation_prompt'):
            token_ids = self._tokenizer.format_generation_prompt(
                prompt=prompt,
                sft_format=sft_format,
                special_token_ids=special_token_ids,
                is_unconditional=is_unconditional,
            )
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            return input_ids

        # Priority 3: Basic tokenization
        if hasattr(self._tokenizer, 'encode_with_mask'):
            # Use config cascade for max_length (ZERO HARDCODE)
            try:
                basic_max_length = self.get_max_sequence_length()
            except RuntimeError:
                basic_max_length = None  # No encoder shape available — let tokenizer decide
            token_result = self._tokenizer.encode_with_mask(
                prompt,
                max_length=basic_max_length,
                padding=False,
                add_special_tokens=True
            )
            input_ids = torch.tensor([token_result["input_ids"]], dtype=torch.long, device=device)
        else:
            token_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        return input_ids

    def tokenize_for_diffusion(
        self,
        prompt: str,
        device: str,
        encoder_name: str = "text_encoder",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Tokenize prompt for diffusion models (iterative_process flow).

        Applies model-specific preprocessing:
        - Chat template formatting for chat-based text encoders (FLUX.2/Mistral)
        - CHI prefix for Sana
        - Basic tokenization for other models

        Args:
            prompt: Text prompt
            device: Target device
            encoder_name: Text encoder component name

        Returns:
            (input_ids, attention_mask) tuple
        """
        from neurobrix.core.components.handlers.text_encoder_handler import TextEncoderComponentHandler as TEHandler

        # Apply model-specific preprocessing (CHI for Sana, etc.)
        if hasattr(TEHandler, 'preprocess_prompt'):
            prompt = TEHandler.preprocess_prompt(prompt, self._tokenizer_vals)

        max_length = self.get_max_sequence_length(encoder_name)

        # Check if text encoder is a chat model that needs formatted input.
        # DATA-DRIVEN: model_type from topology.extracted_values.text_encoder.
        te_config = self._topology.get("extracted_values", {}).get(encoder_name, {})
        model_type = te_config.get("model_type", "")
        system_message = (
            self._topology.get("extracted_values", {})
            .get("tokenizer", {})
            .get("system_message", "")
        )

        if (system_message
            and hasattr(self._tokenizer, 'encode_chat_for_diffusion')):
            result = self._tokenizer.encode_chat_for_diffusion(
                prompt, system_message, max_length,
            )
            input_ids = torch.tensor([result["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.tensor([result["attention_mask"]], dtype=torch.long, device=device)
            return input_ids, attention_mask

        # Fallback: basic tokenization (non-chat models)

        tokens = self._tokenizer(
            prompt,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        if isinstance(tokens, dict):
            input_ids = tokens.get("input_ids")
            attention_mask = tokens.get("attention_mask")
        else:
            input_ids = getattr(tokens, "input_ids", None)
            attention_mask = getattr(tokens, "attention_mask", None)

        if input_ids is None:
            raise RuntimeError(f"ZERO FALLBACK: Tokenizer returned no input_ids: {type(tokens)}")

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        return input_ids, attention_mask

    def tokenize_negative(
        self,
        device: str,
        encoder_name: str = "text_encoder",
        negative_prompt: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize negative/unconditional prompt for CFG.

        Args:
            device: Target device
            encoder_name: Text encoder for shape lookup
            negative_prompt: Negative prompt (default: empty string)

        Returns:
            (neg_input_ids, neg_attention_mask) tuple
        """
        # Determine tokenizer for this encoder
        suffix = ""
        if encoder_name != "text_encoder" and encoder_name.startswith("text_encoder_"):
            suffix = "_" + encoder_name.split("text_encoder_")[1]
        tokenizer_name = f"tokenizer{suffix}"

        # Get max_length (DATA-DRIVEN)
        tokenizer_vals = self._topology.get("extracted_values", {}).get(tokenizer_name, {})
        neg_max_length = tokenizer_vals.get("max_sequence_length")

        if neg_max_length is None:
            text_encoder_shapes = (
                self._topology.get("components", {})
                .get(encoder_name, {})
                .get("shapes", {})
            )
            if "input_ids" in text_encoder_shapes:
                neg_max_length = text_encoder_shapes["input_ids"][1]

        if neg_max_length is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Cannot determine max_sequence_length for CFG negative encoding "
                f"of {encoder_name}. "
                "Model data incomplete. Re-import: neurobrix remove <model> && neurobrix import <org>/<model>"
            )

        neg_tokens = self._tokenizer(text=negative_prompt, max_length=neg_max_length)
        neg_input_ids = neg_tokens["input_ids"].to(device)
        neg_attention_mask = neg_tokens["attention_mask"].to(device)

        return neg_input_ids, neg_attention_mask
