"""The ATen branch's stored-ids method for every flow that tokenizes a prompt: vlm, dual_ar, next_token_diffusion.

A tokenizer without its own text encoder carries its vendor's model_max_length (98b6a0f7f1b3): granite's 2**63-1,
Ming-Lite-Omni's 1e30. A contract resolving `global.input_ids` / `global.attention_mask` through the tokenizer module
then had `resolve_all` tokenize the container's default prompt padded to it: a MemoryError after the answer. Each
flow states first the ids it prefilled. The Triton branch's twin is `triton/flow/prefilled_ids.py` (NBXTensor).
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch


class StatesThePrefilledIds:
    """A flow handler's mixin: its outputs are resolved with the ids it prefilled stated first (torch, ATen branch)."""

    def _resolved_with_the_prefilled_ids(self, prompt_ids: List[int]) -> Dict[str, Any]:
        """The flow's outputs (`resolve_all`), `global.input_ids` / `global.attention_mask` stated first as the ids
        this flow prefilled ([1, N] int64, on the host: a record, read by no later compute) and their all-ones mask:
        the tokenizer module is not called again on the default prompt padded to its max length."""
        ids = torch.tensor([[int(i) for i in prompt_ids]], dtype=torch.long)
        resolved = self.ctx.variable_resolver.resolved
        resolved["global.input_ids"] = ids
        resolved["global.attention_mask"] = torch.ones_like(ids)
        return self.ctx.variable_resolver.resolve_all()
