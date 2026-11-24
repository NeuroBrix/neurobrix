"""Every flow that tokenizes a prompt states, before `resolve_all`, the ids it prefilled: the tokenizer is not called again.

qa of 98b6a0f7f1b3 (task 1340): once the executor no longer gave a decoder's build extent ([1, 23]) to a tokenizer
without its own text encoder, a contract resolving `global.input_ids` / `global.attention_mask` through the tokenizer
module had `resolve_all` tokenize the container's default prompt padded to the vendor's model_max_length (granite's
2**63 - 1): a MemoryError after the answer. The autoregressive flow was given `_resolved_with_the_prefilled_ids`; the
vlm, dual_ar and next_token_diffusion flows tokenize a prompt too and get the same twin method, one copy per branch.
The audio_llm flow tokenizes no prompt (its prefix and suffix ids are the container's `stt_prefix_ids` /
`stt_suffix_ids`, core/flow/audio_llm.py:204, triton/flow/audio_llm.py:202) and is given none.

What this test does if the code is wrong: the handler has no such method (red on 98b6a0f7f1b3), or the stored ids are
not the prefilled ones, or the tokenizer module is called.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

PREFILLED = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 77091, 198]

_CONTRACT = {name: {"type": "tensor", "resolver": {"method": "module_output", "module": "tokenizer", "output_key": key,
                                                   "inputs": {"prompt": "$defaults.prompt"}}}
             for name, key in (("global.input_ids", "input_ids"), ("global.attention_mask", "attention_mask"))}

_HANDLERS = {
    ("vlm", "compiled"): ("neurobrix.core.flow.vlm", "VLMEngine"),
    ("vlm", "triton"): ("neurobrix.triton.flow.vlm", "TritonVLMEngine"),
    ("dual_ar", "compiled"): ("neurobrix.core.flow.dual_ar", "DualAREngine"),
    ("dual_ar", "triton"): ("neurobrix.triton.flow.dual_ar", "TritonDualAREngine"),
    ("next_token_diffusion", "compiled"): ("neurobrix.core.flow.next_token_diffusion", "NextTokenDiffusionEngine"),
    ("next_token_diffusion", "triton"): ("neurobrix.triton.flow.next_token_diffusion",
                                         "TritonNextTokenDiffusionEngine"),
}


class _NoPaddedTokenizer:
    """A tokenizer module the contract's variables must not call again after the prefill."""

    calls = 0

    def __call__(self, **_):
        """Refuses: a call here means `resolve_all` tokenized the default prompt once more, padded to max length."""
        type(self).calls += 1
        raise MemoryError("the contract's input_ids were tokenized again, padded to the tokenizer's max length")


def _host(t, branch):
    """The values of a branch's tensor on the host, as a nested list (torch on ATen, NBXTensor on Triton)."""
    if branch == "compiled":
        return t.detach().cpu().numpy().tolist()
    return t.numpy().tolist()


@pytest.mark.parametrize("flow,branch", sorted(_HANDLERS))
def test_the_contract_s_ids_are_the_ones_the_flow_prefilled(flow, branch):
    """After the answer, `resolve_all` reads the ids each flow's branch prefilled, all-ones mask; no tokenizer call."""
    import importlib
    if branch == "compiled":
        pytest.importorskip("torch")
    else:
        pytest.importorskip("numpy")
        pytest.importorskip("nbxtensor")
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
    module, name = _HANDLERS[(flow, branch)]
    handler_cls = getattr(importlib.import_module(module), name)
    tokenizer = _NoPaddedTokenizer()
    _NoPaddedTokenizer.calls = 0
    resolver = VariableResolver(_CONTRACT, {"prompt": ""}, {}, {"tokenizer": tokenizer}, {}, device="cpu",
                                mode=branch)
    handler = object.__new__(handler_cls)
    handler.ctx = SimpleNamespace(variable_resolver=resolver)
    out = handler._resolved_with_the_prefilled_ids(list(PREFILLED))
    assert _NoPaddedTokenizer.calls == 0
    assert _host(out["global.input_ids"], branch) == [PREFILLED]
    assert _host(out["global.attention_mask"], branch) == [[1] * len(PREFILLED)]
