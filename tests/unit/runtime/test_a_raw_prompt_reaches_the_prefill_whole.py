"""A raw prompt (`neurobrix run --no-chat`) reaches a language model's prefill whole: N tokens in, N ids out.

The Mac, 24/11: `neurobrix run --no-chat` on granite-3.1-1b-a400m-instruct prefilled the first 23 of a 67-token prompt
(-vv "Bound s1=23", NBX_DEBUG_DECODE cache_seq=23: the prompt cut after "Today's Date: November 2") and the model
continued from there. The executor gave the container's tokenizer a max_length found by searching EVERY component for
an `input_ids` shape: a language model has no text encoder, so the search took the decoder's own `input_ids` [1, 23],
the build's stimulus extent of a symbolic sequence axis, and the raw path's basic encode truncated the prompt to it.
TinyLlama and Qwen2.5-Coder carry the same [1, 23] and were cut the same way. Both engines load their modules through
the one executor, so both branches cut the prompt.

The container here has no weights: a manifest, a topology whose decoder declares `input_ids` [1, 23] (as granite's
does), and a tokenizer.json of one id per character. What this test does if the code is wrong: the prompt of 67
characters comes back as 23 ids (red on 1b9793adc3ce).
"""
from __future__ import annotations

import json
import string
from types import SimpleNamespace

import pytest

from neurobrix.core.module.text.processor import TextProcessor
from neurobrix.core.runtime.executor import RuntimeExecutor

BUILD_EXTENT, N = 23, 67


def _container(root):
    """A weightless language-model container: its decoder's build extent is 23 tokens, its tokenizer one id per character."""
    (root / "manifest.json").write_text(json.dumps({"model_name": "raw-prompt-lm"}))
    topology = {"flow": {"type": "autoregressive_generation"},
                "components": {"model": {"shapes": {"input_ids": [1, BUILD_EXTENT], "position_ids": [1, BUILD_EXTENT]}},
                               "lm_head": {"shapes": {"input": [1, BUILD_EXTENT, 8]}}}}
    tok = root / "modules" / "tokenizer"
    tok.mkdir(parents=True)
    vocab = {"<unk>": 0, **{c: i + 1 for i, c in enumerate(string.ascii_lowercase)}}
    (tok / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [], "normalizer": None, "pre_tokenizer": None, "post_processor": None, "decoder": None,
        "model": {"type": "BPE", "unk_token": "<unk>", "vocab": vocab, "merges": []}}))
    (tok / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096, "chat_template": "{{ messages }}",
        "add_bos_token": False, "add_eos_token": False}))
    return topology


class _Resolver:
    """The request's runtime variables: `--no-chat`, on the ATen branch's container."""
    mode = "compiled"

    def get(self, key, default=None):
        """The one variable the raw path reads: chat_mode False, as `--no-chat` sets it."""
        return {"global.chat_mode": False}.get(key, default)


def test_a_raw_prompt_longer_than_the_build_extent_reaches_the_prefill_whole(tmp_path):
    """The executor's tokenizer encodes a 67-token raw prompt as 67 ids, past the decoder's 23-token build extent."""
    pytest.importorskip("torch")
    root = tmp_path / "raw-prompt-lm"          # a container's directory carries its model's name
    root.mkdir()
    topology = _container(root)
    defaults = {"type": "autoregressive_text", "chat_mode": True}
    executor = object.__new__(RuntimeExecutor)
    executor.pkg = SimpleNamespace(topology=topology, defaults=defaults,
                                   modules={"tokenizer": {"type": "tokenizer", "path": "modules/tokenizer/"}})
    executor._nbx_path_str = str(root)
    executor.modules = {}
    executor._setup_tokenizer("tokenizer", {"path": "modules/tokenizer/"})

    prompt = (string.ascii_lowercase * 3)[:N]
    ids = TextProcessor(executor.modules["tokenizer"], defaults, topology, _Resolver()).tokenize(prompt, "cpu")
    assert ids.shape[1] == N, (f"a {N}-token raw prompt reached the prefill as {ids.shape[1]} ids: the tokenizer was "
                               f"given the decoder's build extent ({BUILD_EXTENT}) as its max length")


class _NoPaddedTokenizer:
    """A tokenizer module the contract's variables must not call again after the prefill."""

    def __call__(self, **_):
        """Refuses: a call here means `resolve_all` tokenized the default prompt once more, padded to max length."""
        raise MemoryError("the contract's input_ids were tokenized again, padded to the tokenizer's max length")


_CONTRACT = {name: {"type": "tensor", "resolver": {"method": "module_output", "module": "tokenizer", "output_key": key,
                                                   "inputs": {"prompt": "$defaults.prompt"}}}
             for name, key in (("global.input_ids", "input_ids"), ("global.attention_mask", "attention_mask"))}


@pytest.mark.parametrize("branch", ["compiled", "triton"])
def test_the_contract_s_ids_are_the_prefilled_ones_on_both_branches(branch):
    """After the answer, `resolve_all` reads the ids each branch's flow prefilled: the tokenizer is not called again.

    With granite's tokenizer max length (2**63 - 1, the vendor's) the second call padded the default prompt to it and
    raised a MemoryError once the prompt was no longer clipped to the decoder's build extent (seen failing)."""
    from neurobrix.core.runtime.resolution.variable_resolver import VariableResolver
    if branch == "compiled":
        torch = pytest.importorskip("torch")
        from neurobrix.core.flow.autoregressive import AutoregressiveHandler as Handler
        ids = torch.arange(N, dtype=torch.long).unsqueeze(0)
    else:
        np = pytest.importorskip("numpy")
        from nbxtensor.tensor import NBXTensor
        from neurobrix.triton.flow.autoregressive import TritonAutoregressiveHandler as Handler
        ids = NBXTensor.from_numpy(np.arange(N, dtype=np.int64)[None, :])
    resolver = VariableResolver(_CONTRACT, {"prompt": ""}, {}, {"tokenizer": _NoPaddedTokenizer()}, {}, device="cpu",
                                mode=branch)
    handler = object.__new__(Handler)
    handler.ctx = SimpleNamespace(variable_resolver=resolver)
    out = handler._resolved_with_the_prefilled_ids(ids)
    assert out["global.input_ids"] is ids
    assert tuple(out["global.attention_mask"].shape) == (1, N)
