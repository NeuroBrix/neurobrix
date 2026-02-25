# core/module/tokenizer/factory.py
"""
NeuroBrix Tokenizer Factory.

Creates tokenizer instances from NBX container.
ZERO HARDCODE: Tokenizer type is determined from config.
ZERO FALLBACK: Crash if tokenizer cannot be loaded.

Classes:
- TokenizerWrapper: Wrapper around HuggingFace tokenizers
- TokenizerFactory: Factory to create tokenizer from NBX
"""

import os
import zipfile
import tempfile
import shutil
from typing import Dict, Any, Optional
from pathlib import Path


class TokenizerWrapper:
    """
    Wrapper around HuggingFace tokenizers that provides a callable interface.

    When called, returns dict with input_ids and attention_mask.
    """

    def __init__(self, tokenizer, max_length: int = 512, padding: str = "max_length"):
        self._tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, text: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Tokenize text and return dict with input_ids and attention_mask.

        Args:
            text: Text to tokenize (can also be passed as 'prompt')
            **kwargs: Additional arguments (prompt, max_length, etc.)

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        import torch

        # Get text from various sources
        if text is None:
            text = kwargs.get("prompt", kwargs.get("text", ""))

        max_length = kwargs.get("max_length", self.max_length)
        padding = kwargs.get("padding", self.padding)

        # Tokenize
        encoded = self._tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

    def encode(self, text: str, **kwargs) -> Any:
        """Direct access to encode method."""
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, **kwargs) -> str:
        """Direct access to decode method."""
        return self._tokenizer.decode(token_ids, **kwargs)

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs
    ):
        """
        Apply chat template to messages.

        Universal method that works with any HuggingFace tokenizer that has a chat_template.
        Used for chat models like DeepSeek, Llama, etc.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            tokenize: If True, return token IDs; if False, return formatted string
            add_generation_prompt: If True, add assistant prompt suffix
            **kwargs: Additional arguments passed to HuggingFace apply_chat_template

        Returns:
            Token IDs (if tokenize=True) or formatted string (if tokenize=False)
        """
        if not hasattr(self._tokenizer, 'apply_chat_template'):
            # Fallback: simple concatenation
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            if add_generation_prompt:
                text += "assistant:"
            if tokenize:
                return self._tokenizer.encode(text, add_special_tokens=True)
            return text

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )

    def has_chat_template(self) -> bool:
        """Check if tokenizer has a chat template configured."""
        return hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template is not None

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        result = getattr(self._tokenizer, 'bos_token_id', None)
        if result is None:
            raise RuntimeError("ZERO FALLBACK: bos_token_id not available")
        return result


class TokenizerFactory:
    """
    Factory to create tokenizer instances from NBX container.

    ZERO HARDCODE: Tokenizer class determined from config.
    ZERO FALLBACK: Crash if tokenizer cannot be loaded.
    """

    @classmethod
    def create(
        cls,
        nbx_path: str,
        module_path: str = "modules/tokenizer",
        max_length_override: Optional[int] = None
    ) -> TokenizerWrapper:
        """
        Create a tokenizer instance from NBX container.

        Args:
            nbx_path: Path to NBX container file
            module_path: Path within NBX to tokenizer files
            max_length_override: Override for max_length (from topology shapes)

        Returns:
            TokenizerWrapper instance
        """
        # Normalize module_path (remove trailing slash)
        module_path = module_path.rstrip("/")

        # Extract tokenizer files to temp directory
        temp_dir = tempfile.mkdtemp(prefix="nbx_tokenizer_")

        try:
            with zipfile.ZipFile(nbx_path, 'r') as z:
                # Find all tokenizer files
                tokenizer_files = [
                    name for name in z.namelist()
                    if name.startswith(module_path + "/")
                ]

                if not tokenizer_files:
                    raise RuntimeError(
                        f"ZERO FALLBACK: No tokenizer files found in {module_path}.\n"
                        f"NBX must contain tokenizer module."
                    )

                # Extract to temp directory
                for file_path in tokenizer_files:
                    # Get relative path within module
                    rel_path = file_path[len(module_path) + 1:]
                    if not rel_path:
                        continue

                    target_path = Path(temp_dir) / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    with z.open(file_path) as src:
                        with open(target_path, 'wb') as dst:
                            dst.write(src.read())

            # Determine tokenizer class from config
            config_path = Path(temp_dir) / "tokenizer_config.json"
            if not config_path.exists():
                raise RuntimeError(
                    f"ZERO FALLBACK: tokenizer_config.json not found.\n"
                    f"Files extracted: {list(Path(temp_dir).rglob('*'))}"
                )

            import json
            with open(config_path) as f:
                config = json.load(f)

            # ZERO FALLBACK: Required config values must exist
            tokenizer_class_name = config.get("tokenizer_class")
            if tokenizer_class_name is None:
                raise RuntimeError(
                    "ZERO FALLBACK: 'tokenizer_class' not found in tokenizer_config.json.\n"
                    "This value is required to load the correct tokenizer."
                )

            # VARIABLE EMBARQUÉE: max_length_override has priority (from topology.shapes)
            if max_length_override is not None:
                max_length = max_length_override
            else:
                max_length = config.get("model_max_length")
                if max_length is None:
                    raise RuntimeError(
                        "ZERO FALLBACK: 'model_max_length' not found in tokenizer_config.json.\n"
                        "Provide max_length_override or ensure config has model_max_length."
                    )

            # Load tokenizer using transformers
            tokenizer = cls._load_tokenizer(temp_dir, tokenizer_class_name)

            return TokenizerWrapper(tokenizer, max_length=max_length)

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def _load_tokenizer(cls, tokenizer_path: str, class_name: str):
        """
        Load tokenizer from extracted files.

        ZERO HARDCODE: Uses dynamic class loading via getattr(transformers, class_name).
        ZERO FALLBACK: No silent fallback - crash explicitly if class not found.

        Args:
            tokenizer_path: Path to extracted tokenizer files
            class_name: Tokenizer class name from config

        Returns:
            HuggingFace tokenizer instance

        Raises:
            RuntimeError: If tokenizer class not found or loading fails
        """
        import transformers

        # VARIABLE EMBARQUÉE: Dynamic class loading from config
        tokenizer_cls = getattr(transformers, class_name, None)

        if tokenizer_cls is None:
            raise RuntimeError(
                f"ZERO FALLBACK: Tokenizer class '{class_name}' not found in transformers.\n"
                f"Ensure the class name in tokenizer_config.json is correct."
            )

        try:
            return tokenizer_cls.from_pretrained(tokenizer_path)
        except Exception as e:
            raise RuntimeError(
                f"ZERO FALLBACK: Failed to load {class_name} from {tokenizer_path}.\n"
                f"Original error: {e}"
            ) from e
