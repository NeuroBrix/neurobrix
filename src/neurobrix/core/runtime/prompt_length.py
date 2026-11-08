"""The prompt's length as the autoregressive flow will prefill it — one rule for the plan and the derived census.

A language model's prefill runs at the token count of the prompt as the flow renders it (`prompt_token_ids` below: the SFT
format of an image-AR model, the chat template when the container sets `chat_mode`, else a basic encode with the
tokenizer's specials, unpadded), through the container's own tokenizer module. Prism prices the decoder's sequence
symbol at that count (`cli/commands/run.request_input_config`); without it the symbol bound to the trace's extent and
TinyLlama's prefill was priced at ~0.01 GiB for a 20-, 200- or 400-token prompt alike while the Triton arena held 102,
314 and 652 MB (2026-11-08). The derived census reads the same function, so the plan and the census agree."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _flatten_tokenizer_output(token_ids: Any) -> List[int]:
    """Normalize tokenizer return into a flat List[int]."""
    if hasattr(token_ids, 'input_ids'):
        token_ids = token_ids['input_ids']
    elif (isinstance(token_ids, dict) or
          (hasattr(token_ids, '__getitem__') and not isinstance(token_ids, list)
           and 'input_ids' in token_ids)):
        token_ids = token_ids['input_ids']
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    return token_ids


def prompt_token_ids(tokenizer, prompt: str, defaults: dict, is_image_ar: bool, chat_mode: bool):
    """The prompt's token ids by the autoregressive flow's rule (`_tokenize` priorities 1-3): the
    SFT format for an image-AR model, the HF chat template when chat_mode is on, else a basic encode
    WITH the tokenizer's specials and no padding (orpheus, openaudio). `_tokenize` calls it; so does
    the derived census, whose prefill length is its length."""
    sft_format = defaults.get("sft_format")
    special_token_ids = defaults.get("special_token_ids")
    if (is_image_ar and sft_format and special_token_ids
            and hasattr(tokenizer, "format_generation_prompt")):
        # Priority 1: SFT format (Janus-style image AR). Only taken for
        # image-AR models to avoid disturbing the LLM path.
        token_ids = tokenizer.format_generation_prompt(
            prompt=prompt,
            sft_format=sft_format,
            special_token_ids=special_token_ids,
            is_unconditional=False,
        )
    elif (chat_mode and not is_image_ar
          and hasattr(tokenizer, "apply_chat_template")
          and (not hasattr(tokenizer, "has_chat_template")
               or tokenizer.has_chat_template())):
        # Priority 2: HF chat_template — ONLY when chat_mode is enabled
        # (TextProcessor parity). Preserves the Triton chat-LLM path
        # (TinyLlama, Qwen3, DeepSeek-MoE all set chat_mode=True).
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
    elif hasattr(tokenizer, "encode_with_mask"):
        # Priority 3a: basic tokenization WITH special tokens (TextProcessor
        # parity for chat_mode=False models — orpheus, openaudio).
        _r = tokenizer.encode_with_mask(
            prompt, padding=False, add_special_tokens=True)
        token_ids = _r["input_ids"] if isinstance(_r, dict) else _r
    elif hasattr(tokenizer, "encode"):
        # Priority 3b: basic encode (with special tokens when supported).
        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        except TypeError:
            token_ids = tokenizer.encode(prompt)
    else:
        raise RuntimeError("Tokenizer has no encode method.")

    return _flatten_tokenizer_output(token_ids)


def generation_type(root: Path) -> Optional[str]:
    """The flow's generation type the container declares (`topology.json` flow.generation.type), or None."""
    topo = json.loads((Path(root) / "topology.json").read_text())
    return ((topo.get("flow") or {}).get("generation") or {}).get("type")


def container_tokenizer(root: Path):
    """The container's tokenizer module, where the executor loads it from; a named refusal when it has none."""
    from neurobrix.core.module.tokenizer.sp_tokenizer import load_tokenizer_from_path
    root = Path(root)
    topo = json.loads((root / "topology.json").read_text())
    tok = (topo.get("modules") or {}).get("tokenizer")
    path = ((tok.get("path") if isinstance(tok, dict) else tok) or "modules/tokenizer")   # the executor's default
    if not (root / str(path).rstrip("/")).exists():
        raise FileNotFoundError(f"ZERO FALLBACK: {root.name}: no tokenizer module at {path} — the prompt's length "
                                f"cannot be counted the way the flow will count it")
    return load_tokenizer_from_path(root / str(path).rstrip("/"), None)


def prompt_tokens(root: Path, prompt: str, defaults: Optional[Dict[str, Any]] = None) -> int:
    """The prefill length of `prompt` for the container at `root`, by the autoregressive flow's own rule."""
    root = Path(root)
    if defaults is None:
        p = root / "runtime" / "defaults.json"
        defaults = json.loads(p.read_text()) if p.exists() else {}
    return len(prompt_token_ids(container_tokenizer(root), prompt, defaults,
                                generation_type(root) == "autoregressive_image",
                                bool(defaults.get("chat_mode", False))))
