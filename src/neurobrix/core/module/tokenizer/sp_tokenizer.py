# core/module/tokenizer/sp_tokenizer.py
"""
Native Tokenizers - ZERO HUGGINGFACE TRANSFORMERS.

Direct SentencePiece and BPE integration for T5/CLIP tokenization.
Loads spiece.model or merges.txt+vocab.json from .nbx container.

ZERO HARDCODE: Tokenizer type is detected from available files.

Classes:
- SPTokenizer: SentencePiece tokenizer (T5/FLAN)
- BPETokenizer: Byte-Pair Encoding tokenizer (CLIP)
- HFTokenizer: HuggingFace fast tokenizer wrapper

Functions:
- load_tokenizer_from_path: Load tokenizer with automatic type detection
- load_tokenizer_from_nbx: Load tokenizer from NBX container
"""

import sentencepiece as spm
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from io import BytesIO
from pathlib import Path
import tempfile
import json
import os
import re

if TYPE_CHECKING:
    from nbx import NBXContainer


class SPTokenizer:
    """
    SentencePiece tokenizer - ZERO HUGGINGFACE.

    Loads spiece.model directly from bytes.
    Compatible with T5/FLAN tokenizer format.

    ZERO HARDCODE: Special token IDs come from config, not hardcoded.
    """

    def __init__(self, spiece_model_bytes: bytes, config: Optional[dict] = None):
        """
        Initialize tokenizer from spiece.model bytes.

        Args:
            spiece_model_bytes: Raw bytes of spiece.model file
            config: Config dict with pad_token_id, eos_token_id, unk_token_id, max_length

        Raises:
            RuntimeError: If required config values are missing
        """
        self._sp = spm.SentencePieceProcessor()

        # SentencePiece requires file path, use temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".model") as f:
            f.write(spiece_model_bytes)
            tmp_path = f.name

        try:
            self._sp.Load(tmp_path)
        finally:
            os.unlink(tmp_path)

        self._config = config or {}
        self._vocab_size = self._sp.GetPieceSize()

        # VARIABLE EMBARQUÉE: Special token IDs from config
        # Use SentencePiece defaults if not in config (SP standard: pad=0, eos=1, unk=2)
        self._pad_id = self._config.get("pad_token_id", self._sp.pad_id() if hasattr(self._sp, 'pad_id') else 0)
        self._eos_id = self._config.get("eos_token_id", self._sp.eos_id() if hasattr(self._sp, 'eos_id') else 1)
        self._unk_id = self._config.get("unk_token_id", self._sp.unk_id() if hasattr(self._sp, 'unk_id') else 2)
        self._bos_id = self._config.get("bos_token_id", 2)  # Default BOS=2 for Gemma-style
        self._max_length = self._config.get("model_max_length", 512)

        # DATA-DRIVEN: add_bos_token and add_eos_token from config
        # These control whether BOS/EOS are added during encoding
        self._add_bos_token = self._config.get("add_bos_token", False)
        self._add_eos_token = self._config.get("add_eos_token", True)  # Legacy default

        # DATA-DRIVEN: padding_side from config
        # "left" = padding tokens at start (Gemma/Sana), "right" = padding at end (legacy)
        self._padding_side = self._config.get("padding_side", "right")

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        """Return pad token ID."""
        return self._pad_id

    @property
    def eos_token_id(self) -> int:
        """Return EOS token ID."""
        return self._eos_id

    @property
    def unk_token_id(self) -> int:
        """Return UNK token ID."""
        return self._unk_id

    @property
    def bos_token_id(self) -> int:
        """Return BOS token ID."""
        return self._bos_id

    @property
    def max_length(self) -> int:
        """Return max sequence length from config."""
        return self._max_length

    def __call__(
        self,
        text: Optional[str] = None,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs
    ) -> dict:
        """
        Callable interface for variable resolver compatibility.

        Accepts either 'text' or 'prompt' as input.
        Returns dict with 'input_ids' and 'attention_mask' as torch tensors.

        If complex_human_instruction is configured (Sana-style models),
        the instruction is prepended to the user prompt.

        Args:
            text: Text to tokenize
            prompt: Alias for text (for variables.json compatibility)
            max_length: Max sequence length (uses config default if None)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        import torch

        # Accept either 'text' or 'prompt'
        input_text = text if text is not None else prompt
        if input_text is None:
            raise RuntimeError(
                "ZERO FALLBACK: SPTokenizer requires 'text' or 'prompt' argument."
            )

        # NOTE: complex_human_instruction is for external prompt enhancement (LLM rewriting)
        # NOT for direct tokenization. Diffusers uses it with a separate model.
        # We tokenize the raw prompt directly.

        # Use encode_with_mask for the core logic
        result = self.encode_with_mask(input_text, max_length=max_length)

        # Convert to tensors (add batch dimension)
        return {
            "input_ids": torch.tensor([result["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([result["attention_mask"]], dtype=torch.long),
        }

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            max_length: Maximum sequence length (uses config default if None)
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add BOS/EOS tokens (respects config flags)

        Returns:
            List of token IDs
        """
        # Use config max_length if not specified
        if max_length is None:
            max_length = self._max_length

        assert isinstance(max_length, int), "max_length must be int"

        # Encode with SentencePiece
        ids = self._sp.EncodeAsIds(text)

        # DATA-DRIVEN: Add BOS if config says add_bos_token=true
        if add_special_tokens and self._add_bos_token:
            ids = [self._bos_id] + ids

        # DATA-DRIVEN: Add EOS if config says add_eos_token=true
        if add_special_tokens and self._add_eos_token:
            ids = ids + [self._eos_id]

        # Truncate if too long
        if len(ids) > max_length:
            ids = ids[:max_length]
            # Ensure last special token preserved if truncated
            if add_special_tokens and self._add_eos_token:
                ids[-1] = self._eos_id

        # Pad if requested - respect padding_side from config
        if padding and len(ids) < max_length:
            pad_len = max_length - len(ids)
            if self._padding_side == "left":
                # LEFT padding: [PAD, PAD, ..., BOS, tokens]
                ids = [self._pad_id] * pad_len + ids
            else:
                # RIGHT padding: [BOS, tokens, ..., PAD, PAD]
                ids = ids + [self._pad_id] * pad_len

        return ids

    def encode_with_mask(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Encode text to token IDs with attention mask.

        Same as encode() but returns dict with both input_ids and attention_mask.
        Matches HuggingFace tokenizer output format.

        Args:
            text: Input text string
            max_length: Maximum sequence length (uses config default if None)
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add BOS/EOS tokens (respects config flags)

        Returns:
            Dict with 'input_ids' and 'attention_mask' (both List[int])
        """
        # Use config max_length if not specified
        if max_length is None:
            max_length = self._max_length

        assert isinstance(max_length, int), "max_length must be int"

        # Encode with SentencePiece
        ids = self._sp.EncodeAsIds(text)

        # DATA-DRIVEN: Add BOS if config says add_bos_token=true
        if add_special_tokens and self._add_bos_token:
            ids = [self._bos_id] + ids

        # DATA-DRIVEN: Add EOS if config says add_eos_token=true
        if add_special_tokens and self._add_eos_token:
            ids = ids + [self._eos_id]

        # Truncate if too long
        if len(ids) > max_length:
            ids = ids[:max_length]
            if add_special_tokens and self._add_eos_token:
                ids[-1] = self._eos_id

        # Create attention mask BEFORE padding (1 for real tokens)
        actual_len = len(ids)
        attention_mask = [1] * actual_len

        # Pad if requested - respect padding_side from config
        if padding and len(ids) < max_length:
            pad_len = max_length - len(ids)
            if self._padding_side == "left":
                # LEFT padding: [0, 0, ..., 1, 1, 1] for mask
                ids = [self._pad_id] * pad_len + ids
                attention_mask = [0] * pad_len + attention_mask
            else:
                # RIGHT padding: [1, 1, 1, ..., 0, 0] for mask
                ids = ids + [self._pad_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip PAD/EOS tokens

        Returns:
            Decoded text string
        """
        if skip_special_tokens:
            # Remove PAD and EOS tokens
            ids = [i for i in ids if i not in (self._pad_id, self._eos_id)]

        return self._sp.DecodeIds(ids)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 120,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add EOS token

        Returns:
            List of token ID lists
        """
        return [
            self.encode(text, max_length, padding, add_special_tokens)
            for text in texts
        ]


class BPETokenizer:
    """
    BPE Tokenizer - ZERO HUGGINGFACE TRANSFORMERS.

    Native implementation for CLIP-style tokenizers.
    Loads merges.txt + vocab.json directly from disk.

    ZERO HARDCODE: Special token IDs come from config, not hardcoded.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[str],
        config: Optional[dict] = None,
    ):
        """
        Initialize BPE tokenizer.

        Args:
            vocab: Token to ID mapping (from vocab.json)
            merges: List of BPE merge rules (from merges.txt)
            config: Config dict with special token IDs and settings

        Raises:
            RuntimeError: If required config values are missing
        """
        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        self._config = config or {}
        self._vocab_size = len(vocab)

        # Build BPE merge ranking
        self._bpe_ranks = {}
        for i, merge in enumerate(merges):
            parts = merge.split()
            if len(parts) == 2:
                self._bpe_ranks[tuple(parts)] = i

        # Get special token strings from config
        self._bos_token = self._config.get("bos_token", "<|startoftext|>")
        self._eos_token = self._config.get("eos_token", "<|endoftext|>")
        self._pad_token = self._config.get("pad_token", self._eos_token)
        self._unk_token = self._config.get("unk_token", self._eos_token)

        # ZERO HARDCODE: Get token IDs from VOCAB (not config's *_id fields which may be wrong)
        # Config *_id fields are sometimes internal references, not actual vocab IDs
        self._bos_id = self._vocab.get(self._bos_token)
        self._eos_id = self._vocab.get(self._eos_token, 49407)
        self._pad_id = self._vocab.get(self._pad_token, self._eos_id)
        self._unk_id = self._vocab.get(self._unk_token, self._eos_id)

        self._max_length = self._config.get("model_max_length", 77)

        # Text processing settings
        self._do_lower_case = self._config.get("do_lower_case", True)
        self._add_prefix_space = self._config.get("add_prefix_space", False)

        # Build byte encoder for handling bytes
        self._byte_encoder = self._bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Build byte to unicode mapping (CLIP/GPT-2 style).

        Returns a mapping of bytes to unicode strings to avoid BPE
        merges breaking on special bytes.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def _get_pairs(self, word: tuple) -> set:
        """Get all adjacent pairs in a word."""
        pairs = set()
        prev = word[0]
        for char in word[1:]:
            pairs.add((prev, char))
            prev = char
        return pairs

    def _bpe(self, token: str) -> str:
        """Apply BPE to a single token."""
        word = tuple(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self._bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self._bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        return " ".join(word)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        """Return pad token ID."""
        return self._pad_id

    @property
    def eos_token_id(self) -> int:
        """Return EOS token ID."""
        return self._eos_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Return BOS token ID."""
        return self._bos_id

    @property
    def unk_token_id(self) -> int:
        """Return UNK token ID."""
        return self._unk_id

    @property
    def max_length(self) -> int:
        """Return max sequence length from config."""
        return self._max_length

    def __call__(
        self,
        text: Optional[str] = None,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Callable interface for variable resolver compatibility.

        Accepts either 'text' or 'prompt' as input.
        Returns dict with 'input_ids' and 'attention_mask' as torch tensors.

        Args:
            text: Text to tokenize
            prompt: Alias for text (for variables.json compatibility)
            max_length: Max sequence length (uses config default if None)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        import torch

        # Accept either 'text' or 'prompt'
        input_text = text if text is not None else prompt
        if input_text is None:
            raise RuntimeError(
                "ZERO FALLBACK: BPETokenizer requires 'text' or 'prompt' argument."
            )

        # Use encode_with_mask for the core logic
        result = self.encode_with_mask(input_text, max_length=max_length)

        # Convert to tensors (add batch dimension)
        return {
            "input_ids": torch.tensor([result["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([result["attention_mask"]], dtype=torch.long),
        }

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Split a word into subwords using CLIP-style tokenization.

        CLIP splits on punctuation as word boundaries. E.g.:
        "hyper-realistic" -> ["hyper", "-", "realistic"]
        """
        # CLIP regex pattern: split on punctuation and contractions
        # Matches: contractions ('s, 't, etc), letters, numbers, or non-space non-alphanumeric
        import re
        # Simple pattern: split around punctuation, keeping punctuation as separate tokens
        pattern = r"(['\-\.,!?:;\(\)\[\]\{\}\"'`])"
        parts = re.split(pattern, word)
        # Filter out empty strings
        return [p for p in parts if p]

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs (CLIP-style BPE).

        Args:
            text: Input text string
            max_length: Maximum sequence length (uses config default if None)
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self._max_length

        assert isinstance(max_length, int), "max_length must be int"

        # Preprocess text
        if self._do_lower_case:
            text = text.lower()

        tokens = []

        # Handle BOS token
        if add_special_tokens and self._bos_id is not None:
            tokens.append(self._bos_id)

        # CLIP tokenization: split on whitespace, then on punctuation
        for word in text.strip().split():
            # Split word on punctuation boundaries (CLIP-style)
            subwords = self._tokenize_word(word)

            for i, subword in enumerate(subwords):
                # Last subword in the word gets </w> suffix
                is_last = (i == len(subwords) - 1)
                subword_with_suffix = subword + "</w>" if is_last else subword + "</w>"

                # Check if whole subword is in vocabulary
                if subword_with_suffix in self._vocab:
                    tokens.append(self._vocab[subword_with_suffix])
                else:
                    # Apply BPE to break down unknown subword
                    bpe_result = self._bpe(subword_with_suffix)
                    for bpe_token in bpe_result.split(" "):
                        if bpe_token in self._vocab:
                            tokens.append(self._vocab[bpe_token])
                        else:
                            # Unknown subword - use unk token
                            tokens.append(self._unk_id)

        # Add EOS token
        if add_special_tokens:
            tokens.append(self._eos_id)

        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            # Ensure EOS at end if truncated
            if add_special_tokens:
                tokens[-1] = self._eos_id

        # Pad if requested
        if padding and len(tokens) < max_length:
            pad_len = max_length - len(tokens)
            tokens = tokens + [self._pad_id] * pad_len

        return tokens

    def encode_with_mask(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Encode text to token IDs with attention mask.

        Same as encode() but returns dict with both input_ids and attention_mask.
        Matches HuggingFace tokenizer output format.

        Args:
            text: Input text string
            max_length: Maximum sequence length (uses config default if None)
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            Dict with 'input_ids' and 'attention_mask' (both List[int])
        """
        if max_length is None:
            max_length = self._max_length

        assert isinstance(max_length, int), "max_length must be int"

        # Encode without padding first
        ids = self.encode(text, max_length, padding=False, add_special_tokens=add_special_tokens)

        # Create attention mask BEFORE padding (1 for real tokens)
        actual_len = len(ids)
        attention_mask = [1] * actual_len

        # Truncate if needed
        if len(ids) > max_length:
            ids = ids[:max_length]
            attention_mask = attention_mask[:max_length]
            actual_len = max_length
            if add_special_tokens:
                ids[-1] = self._eos_id

        # Pad if requested
        if padding and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self._pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        tokens = []
        for id_ in ids:
            if skip_special_tokens:
                if id_ in (self._pad_id, self._eos_id, self._bos_id):
                    continue
            if id_ in self._id_to_token:
                tokens.append(self._id_to_token[id_])

        # Join and decode from byte representation
        text = "".join(tokens)
        text = text.replace("</w>", " ")

        # Decode bytes back to text
        try:
            decoded_bytes = bytearray(
                [self._byte_decoder.get(c, ord(c)) for c in text]
            )
            text = decoded_bytes.decode("utf-8", errors="replace")
        except Exception:
            pass

        return text.strip()

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token ID lists
        """
        return [
            self.encode(text, max_length, padding, add_special_tokens)
            for text in texts
        ]


class HFTokenizer:
    """
    HuggingFace Fast Tokenizer wrapper - uses tokenizers library.

    Loads tokenizer.json directly without HuggingFace transformers dependency.
    Provides same interface as SPTokenizer/BPETokenizer for NeuroBrix compatibility.

    Supports:
    - format_generation_prompt(): Janus-style SFT formatting
    - apply_chat_template(): Universal HuggingFace chat template (Jinja2)
    """

    def __init__(self, tokenizer_path: str, config: Optional[dict] = None):
        """
        Initialize from tokenizer.json path.

        Args:
            tokenizer_path: Path to tokenizer.json file
            config: Optional config from tokenizer_config.json
        """
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._config = config if config is not None else {}

        # Max length from config
        self._max_length = self._config.get("model_max_length", 2048)

        # Special token IDs (get from tokenizer or config)
        self._pad_id = self._get_token_id("pad_token", 0)
        self._eos_id = self._get_token_id("eos_token", 2)
        self._bos_id = self._get_token_id("bos_token", 1)

        # Chat template (Jinja2 format, from tokenizer_config.json)
        self._chat_template = self._config.get("chat_template")

        # BOS/EOS behavior from tokenizer_config.json (DATA-DRIVEN)
        # The raw tokenizers library doesn't add BOS/EOS — that's handled by
        # the transformers PreTrainedTokenizerFast wrapper. We replicate it.
        self._add_bos = self._config.get("add_bos_token", False)
        self._add_eos = self._config.get("add_eos_token", False)

        # Special token strings for chat template rendering
        self._bos_token = self._get_token_string("bos_token", "<s>")
        self._eos_token = self._get_token_string("eos_token", "</s>")

    def _get_token_id(self, token_key: str, default: int) -> int:
        """Get token ID from config or tokenizer."""
        # Try config first (may have id directly or token string)
        token_value = self._config.get(token_key)
        if token_value is None:
            return default

        # If it's a dict (HuggingFace AddedToken format):
        # {"__type": "AddedToken", "content": "<｜begin▁of▁sentence｜>", ...}
        # Extract content string and look up in vocab
        if isinstance(token_value, dict):
            if "id" in token_value:
                return token_value["id"]
            content = token_value.get("content")
            if content is not None:
                token_id = self._tokenizer.token_to_id(content)
                if token_id is not None:
                    return token_id
            return default

        # If it's a string, look up in vocab
        if isinstance(token_value, str):
            token_id = self._tokenizer.token_to_id(token_value)
            if token_id is not None:
                return token_id

        # If it's an int
        if isinstance(token_value, int):
            return token_value

        return default

    def _get_token_string(self, token_key: str, default: str) -> str:
        """Get token string from config."""
        token_value = self._config.get(token_key)
        if token_value is None:
            return default

        # If it's a dict with content field (HuggingFace AddedToken format)
        if isinstance(token_value, dict):
            return token_value.get("content", default)

        # If it's a string
        if isinstance(token_value, str):
            return token_value

        return default

    def has_chat_template(self) -> bool:
        """Check if tokenizer has a chat template configured."""
        return self._chat_template is not None

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> Any:
        """
        Apply chat template to messages using Jinja2.

        Universal method that renders the chat_template from tokenizer_config.json.
        Compatible with HuggingFace tokenizer format.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            tokenize: If True, return token IDs; if False, return formatted string
            add_generation_prompt: If True, add assistant prompt suffix
            **kwargs: Additional arguments (ignored)

        Returns:
            Token IDs (if tokenize=True) or formatted string (if tokenize=False)
        """
        if self._chat_template is None:
            raise RuntimeError(
                "ZERO FALLBACK: No chat_template found in tokenizer_config.json.\n"
                "Cannot apply chat template to messages."
            )

        try:
            from jinja2 import Environment
        except ImportError:
            raise RuntimeError(
                "ZERO FALLBACK: jinja2 required for chat template.\n"
                "Install with: pip install jinja2"
            )

        # Render the Jinja2 template using Environment with HuggingFace-compatible settings.
        # lstrip_blocks=True: strip leading whitespace before {% %} tags
        # trim_blocks=True:   strip newline after {% %} tags
        # Without these, every {% if %}/{% for %} emits spurious newlines that corrupt the prompt.
        env = Environment(lstrip_blocks=True, trim_blocks=True)
        template = env.from_string(self._chat_template)
        formatted = template.render(
            messages=messages,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            add_generation_prompt=add_generation_prompt,
        )

        if not tokenize:
            return formatted

        # Tokenize the formatted string
        encoding = self._tokenizer.encode(formatted)
        return encoding.ids

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            max_length: Maximum sequence length (uses config default if None)
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        if max_length is None:
            max_length = self._max_length

        assert isinstance(max_length, int), "max_length must be int"

        # Encode (raw tokenizers library doesn't add BOS/EOS — we handle it)
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids

        # Add BOS/EOS per tokenizer_config.json (matches transformers behavior)
        if add_special_tokens:
            if self._add_bos and (not ids or ids[0] != self._bos_id):
                ids = [self._bos_id] + ids
            if self._add_eos and (not ids or ids[-1] != self._eos_id):
                ids = ids + [self._eos_id]

        # Truncate if needed
        if len(ids) > max_length:
            ids = ids[:max_length]

        # Pad if needed
        if padding and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self._pad_id] * pad_len

        return ids

    def encode_with_mask(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Encode text to token IDs with attention mask.

        Args:
            text: Input text string
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add special tokens

        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        if max_length is None:
            max_length = self._max_length

        assert isinstance(max_length, int), "max_length must be int"

        # Encode without padding first to get actual length
        ids = self.encode(text, max_length, padding=False, add_special_tokens=add_special_tokens)
        actual_len = len(ids)

        # Create attention mask (1 for real tokens)
        attention_mask = [1] * actual_len

        # Pad if needed
        if padding and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self._pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }

    def __call__(
        self,
        text: Optional[str] = None,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        **kwargs
    ) -> dict:
        """
        Callable interface for variable resolver compatibility.

        Accepts either 'text' or 'prompt' as input.
        Returns dict with 'input_ids' and 'attention_mask' as torch tensors.

        Args:
            text: Text to tokenize
            prompt: Alias for text (for variables.json compatibility)
            max_length: Max sequence length (uses config default if None)
            **kwargs: Additional arguments (ignored)

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        import torch

        # Accept either 'text' or 'prompt'
        input_text = text if text is not None else prompt
        if input_text is None:
            raise RuntimeError(
                "ZERO FALLBACK: HFTokenizer requires 'text' or 'prompt' argument."
            )

        # Use encode_with_mask for the core logic
        result = self.encode_with_mask(input_text, max_length=max_length)

        # Convert to tensors (add batch dimension)
        return {
            "input_ids": torch.tensor([result["input_ids"]], dtype=torch.long),
            "attention_mask": torch.tensor([result["attention_mask"]], dtype=torch.long),
        }

    def encode_chat_for_diffusion(
        self,
        prompt: str,
        system_message: str,
        max_length: int,
    ) -> dict:
        """
        Encode prompt with chat template formatting for diffusion text encoders.

        Constructs: [BOS] [SYSTEM_PROMPT] system_msg [/SYSTEM_PROMPT] [INST] prompt [/INST]
        Then pads/truncates to max_length with attention mask.

        DATA-DRIVEN: System message and max_length from topology, not hardcoded.

        Args:
            prompt: User prompt text
            system_message: System message for the chat template
            max_length: Max sequence length for padding/truncation

        Returns:
            Dict with 'input_ids' and 'attention_mask' (both List[int])
        """
        # Build chat-formatted string with special token markers
        formatted = f"[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT][INST]{prompt}[/INST]"

        # Encode — the tokenizer handles special tokens like [INST], [SYSTEM_PROMPT]
        # as single tokens (they're in the vocabulary as added_tokens).
        # BOS is prepended automatically if add_bos_token=True in config.
        encoding = self._tokenizer.encode(formatted)
        ids = list(encoding.ids)

        # Add BOS if config says add_bos_token=True and not already present
        if self._add_bos and (not ids or ids[0] != self._bos_id):
            ids = [self._bos_id] + ids

        # Truncate if needed
        if len(ids) > max_length:
            ids = ids[:max_length]

        # Create attention mask before padding
        actual_len = len(ids)
        attention_mask = [1] * actual_len

        # Pad to max_length
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self._pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token ID lists
        """
        return [
            self.encode(text, max_length, padding, add_special_tokens)
            for text in texts
        ]

    def format_generation_prompt(
        self,
        prompt: str,
        sft_format: str,
        special_token_ids: Dict[str, int],
        is_unconditional: bool = False,
    ) -> List[int]:
        """
        Format prompt with SFT template for autoregressive generation.

        DATA-DRIVEN: Template is constructed from special_token_ids, not hardcoded strings.
        UNIVERSAL: Supports different SFT formats (deepseek, llama, etc.)

        For CFG (Classifier-Free Guidance) - matches native Janus:
        - conditional: Full prompt with user content
        - unconditional: Keep first (BOS) and last (begin_of_image), pad ALL middle tokens

        Native Janus CFG pattern:
            tokens[i, 1:-1] = pad_id  # Keep first and last, pad everything between

        Args:
            prompt: User prompt text
            sft_format: Template format name (e.g., "deepseek")
            special_token_ids: Dict mapping token names to IDs (from defaults.json)
            is_unconditional: If True, replace ALL middle tokens with pad (native Janus style)

        Returns:
            List of token IDs ready for model input
        """
        # Get token IDs (with fallbacks for missing tokens)
        bos_id = special_token_ids.get("bos", 1)
        pad_id = special_token_ids.get("pad", 0)
        user_id = special_token_ids.get("user")
        assistant_id = special_token_ids.get("assistant")
        begin_of_image_id = special_token_ids.get("begin_of_image")

        if sft_format == "deepseek":
            # DeepSeek SFT template (Janus image generation):
            # <bos>User: {prompt}\n\nAssistant:<begin_of_image>
            #
            # CRITICAL: Native Janus uses plain text "User" and "Assistant",
            # NOT special tokens like <|User|> and <|Assistant|>.
            # Only <begin_of_image> is a special token.
            #
            # Native format produces:
            # [100000, 5726, 25, <prompt_tokens>, 185, 185, 77398, 25, 100016]
            # Where: 100000=BOS, 5726="User", 25=":", 185="\n", 77398="Assistant", 100016=<begin_of_image>

            # Build full conditional template as plain text
            # Template: "User: {prompt}\n\nAssistant:"
            full_text = f"User: {prompt}\n\nAssistant:"

            # Encode the full text
            # NOTE: HFTokenizer adds BOS automatically (configured in tokenizer.json)
            # so we don't add it manually
            result = self.encode(full_text, padding=False, add_special_tokens=False)

            # Add begin_of_image token (this IS a special token)
            if begin_of_image_id is not None:
                result.append(begin_of_image_id)

            # For CFG unconditional: replace ALL middle tokens with pad
            # Native Janus pattern: tokens[1:-1] = pad_id
            # Keep: result[0] (BOS) and result[-1] (begin_of_image)
            # Pad: everything in between
            if is_unconditional and len(result) > 2:
                middle_length = len(result) - 2  # Exclude first and last
                result = [result[0]] + [pad_id] * middle_length + [result[-1]]

            return result

        else:
            # Unknown format: fall back to simple encoding with BOS
            ids = self.encode(prompt, padding=False, add_special_tokens=False)
            if begin_of_image_id is not None:
                ids.append(begin_of_image_id)
            result = [bos_id] + ids

            # Apply same CFG pattern for unknown formats
            if is_unconditional and len(result) > 2:
                middle_length = len(result) - 2
                result = [result[0]] + [pad_id] * middle_length + [result[-1]]

            return result

    @property
    def pad_token_id(self) -> int:
        """Return pad token ID."""
        return self._pad_id

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._tokenizer.get_vocab_size()


class TiktokenTokenizer:
    """
    Tiktoken tokenizer for fish-speech / OpenAudio models.

    Loads *.tiktoken file with optional special_tokens.json.
    Provides same interface as SPTokenizer/HFTokenizer.
    """

    def __init__(self, tiktoken_path: Path, tokenizer_dir: Path,
                 config: Optional[dict] = None):
        config = config or {}
        self.max_length = config.get("model_max_length", 2048)

        # Load special tokens
        special_tokens = {}
        st_path = tokenizer_dir / "special_tokens.json"
        if st_path.exists():
            with open(st_path) as f:
                special_tokens = json.load(f)

        # Try tiktoken library first
        try:
            import tiktoken
            # Load BPE ranks from .tiktoken file
            with open(tiktoken_path, "rb") as f:
                contents = f.read()
            import base64
            bpe_ranks = {}
            for line in contents.splitlines():
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        token = base64.b64decode(parts[0])
                        rank = int(parts[1])
                        bpe_ranks[token] = rank
            # Build special token mapping
            special_token_map = {}
            for token_name, token_info in special_tokens.items():
                if isinstance(token_info, dict):
                    content = token_info.get("content", token_name)
                    token_id = token_info.get("id")
                    if token_id is not None:
                        special_token_map[content] = token_id
                elif isinstance(token_info, int):
                    special_token_map[token_name] = token_info

            self._enc = tiktoken.Encoding(
                name="fish_speech",
                pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
                mergeable_ranks=bpe_ranks,
                special_tokens=special_token_map,
            )
            self._use_tiktoken = True
        except (ImportError, Exception):
            # Fallback: basic tokenization
            self._enc = None
            self._use_tiktoken = False
            self._vocab = {}
            # Build vocab from tiktoken file
            import base64
            with open(tiktoken_path, "rb") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            token = base64.b64decode(parts[0]).decode("utf-8", errors="replace")
                            rank = int(parts[1])
                            self._vocab[token] = rank
                        except Exception:
                            pass

        # Store special tokens for encode/decode
        self._special_tokens = special_tokens
        self._bos_id = None
        self._eos_id = None
        self._pad_id = 0
        for name, info in special_tokens.items():
            tid = info.get("id") if isinstance(info, dict) else info
            if "bos" in name.lower() or "begin" in name.lower():
                self._bos_id = tid
            elif "eos" in name.lower() or "end" in name.lower():
                self._eos_id = tid
            elif "pad" in name.lower():
                self._pad_id = tid

    def encode(self, text: str, add_special_tokens: bool = True,
               return_tensors: Optional[str] = None, **kwargs) -> Any:
        if self._use_tiktoken and self._enc is not None:
            ids = self._enc.encode(text, allowed_special="all")
        else:
            # Byte-level fallback
            ids = list(text.encode("utf-8"))

        if add_special_tokens and self._bos_id is not None:
            ids = [self._bos_id] + ids

        if return_tensors == "pt":
            import torch
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if self._use_tiktoken and self._enc is not None:
            if skip_special_tokens:
                special_ids = set()
                for _name, info in self._special_tokens.items():
                    tid = info.get("id") if isinstance(info, dict) else info
                    if tid is not None:
                        special_ids.add(tid)
                ids = [i for i in ids if i not in special_ids]
            return self._enc.decode(ids)
        return "".join(chr(i) if 32 <= i < 127 else "?" for i in ids)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_id


def load_tokenizer_from_path(
    tokenizer_dir: Path,
    max_length: Optional[int] = None,
) -> "SPTokenizer | BPETokenizer | HFTokenizer | TiktokenTokenizer":
    """
    Load tokenizer from directory - DATA-DRIVEN TYPE DETECTION.

    ZERO HARDCODE: Tokenizer type is detected from available files:
    - If *.model exists → SentencePiece
    - If tokenizer.json exists → HuggingFace Fast Tokenizer
    - If merges.txt + vocab.json exist → BPE (only when tokenizer.json absent)

    Args:
        tokenizer_dir: Path to tokenizer directory
        max_length: Override max sequence length

    Returns:
        SPTokenizer, BPETokenizer, or HFTokenizer instance

    Raises:
        RuntimeError: If no valid tokenizer format found
    """
    # Load config if exists
    config = {}
    config_path = tokenizer_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Override max_length if provided
    if max_length is not None:
        config["model_max_length"] = max_length

    # DATA-DRIVEN TYPE DETECTION
    # Priority 0: If chat_template exists AND tokenizer.json exists, prefer HFTokenizer.
    # HFTokenizer supports apply_chat_template() — SentencePiece does not.
    # This is critical for LLM chat mode. Image/audio models are unaffected
    # (they have no chat_template in tokenizer_config.json).
    has_chat_template = config.get("chat_template") is not None
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if has_chat_template and tokenizer_json_path.exists():
        return HFTokenizer(str(tokenizer_json_path), config)

    # Priority 1: SentencePiece (*.model)
    model_files = list(tokenizer_dir.glob("*.model"))
    if model_files:
        model_file = model_files[0]

        with open(model_file, "rb") as f:
            spiece_bytes = f.read()

        return SPTokenizer(spiece_bytes, config)

    # Priority 2: HuggingFace Fast Tokenizer (tokenizer.json)
    # Checked BEFORE BPE because many modern tokenizers (Qwen, Llama3, etc.)
    # ship both tokenizer.json AND merges.txt+vocab.json. The HF format is
    # more complete (supports chat templates, special tokens, pre-tokenizers).
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        return HFTokenizer(str(tokenizer_json_path), config)

    # Priority 3: BPE (merges.txt + vocab.json)
    # Only used when tokenizer.json is absent (e.g. older CLIP/GPT-2 models).
    merges_path = tokenizer_dir / "merges.txt"
    vocab_path = tokenizer_dir / "vocab.json"

    if merges_path.exists() and vocab_path.exists():

        # Load vocab
        with open(vocab_path) as f:
            vocab = json.load(f)

        # Load merges (skip version header)
        with open(merges_path) as f:
            lines = f.read().split("\n")
            merges = [line for line in lines if line and not line.startswith("#")]

        return BPETokenizer(vocab, merges, config)

    # Priority 4: Tiktoken (*.tiktoken + special_tokens.json)
    # Used by fish-speech / OpenAudio models
    tiktoken_files = list(tokenizer_dir.glob("*.tiktoken"))
    if tiktoken_files:
        return TiktokenTokenizer(tiktoken_files[0], tokenizer_dir, config)

    # No valid format found
    available = [f.name for f in tokenizer_dir.iterdir()] if tokenizer_dir.exists() else []
    raise RuntimeError(
        f"ZERO FALLBACK: No tokenizer format detected.\n"
        f"Expected: *.model (SentencePiece) OR merges.txt+vocab.json (BPE) "
        f"OR tokenizer.json (HF Fast) OR *.tiktoken\n"
        f"Directory: {tokenizer_dir}\n"
        f"Available files: {available}"
    )


def load_tokenizer_from_nbx(
    container: "NBXContainer",
    tokenizer_name: str = "tokenizer",
) -> SPTokenizer:
    """
    Load SPTokenizer from NBXContainer.

    Args:
        container: NBXContainer instance
        tokenizer_name: Name of tokenizer component (default: "tokenizer")

    Returns:
        SPTokenizer instance

    Raises:
        RuntimeError: If spiece.model not found
    """
    # Path to spiece.model in NBX container
    spiece_path = f"components/{tokenizer_name}/spiece.model"

    # Access files via _files dict (NBXContainer API)
    spiece_bytes = container._files.get(spiece_path)

    if spiece_bytes is None:
        # List available tokenizer files for debug
        tokenizer_files = [k for k in container._files.keys() if "tokenizer" in k.lower()]
        raise RuntimeError(
            f"ZERO FALLBACK: spiece.model not found at '{spiece_path}'.\n"
            f"Available tokenizer files: {tokenizer_files}"
        )

    # Load optional config
    config = None
    config_path = f"components/{tokenizer_name}/tokenizer_config.json"
    config_bytes = container._files.get(config_path)
    if config_bytes:
        import json
        config = json.loads(config_bytes.decode("utf-8"))

    return SPTokenizer(spiece_bytes, config)
