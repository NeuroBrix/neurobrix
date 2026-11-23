"""A request's max_tokens is read before the container is asked, at every site of both branches.

`_ov.get("global.max_tokens", require_max_tokens(defaults))` evaluates its default first: a container that declares
no max_tokens was refused even when the request carried the bound (serving/engine.py passes one in every request,
the warm-up passes 1). The Triton serve of such a container was refused at its start by the Triton text flow's
`_build_generator_config`, which asked the container before it read the request.

The order, the same everywhere: the request's `global.max_tokens`, then the container's `max_tokens`, then the
refusal by name (`MissingRuntimeValue`, its wording unchanged).
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurobrix.core.runtime_values import MissingRuntimeValue

SRC = Path(__file__).resolve().parents[3] / "src" / "neurobrix"

# The flows that bound a generation with the request's max_tokens over the container's, one file per branch.
SITES = [
    "core/flow/dual_ar.py",
    "core/flow/next_token_diffusion.py",
    "core/flow/tts_llm.py",
    "core/runtime/prompt_length.py",
    "triton/flow/dual_ar.py",
    "triton/flow/next_token_diffusion.py",
    "triton/flow/tts_llm.py",
    "triton/flow/autoregressive.py",
]

SAMPLING = {"temperature": 0.7, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}


def _helper():
    from neurobrix.core.runtime_values import max_tokens_of
    return max_tokens_of


def _triton_config(resolved, defaults):
    from neurobrix.triton.flow.autoregressive import _build_generator_config
    return _build_generator_config({**SAMPLING, **defaults}, SimpleNamespace(resolved=resolved, family_defaults={}))


READERS = {
    "max_tokens_of": lambda resolved, defaults: _helper()(resolved, defaults),
    "triton _build_generator_config": lambda resolved, defaults: _triton_config(resolved, defaults)["max_tokens"],
}


@pytest.mark.parametrize("reader", READERS)
def test_the_requests_bound_serves_a_container_without_max_tokens(reader):
    assert READERS[reader]({"global.max_tokens": 17}, {}) == 17


@pytest.mark.parametrize("reader", READERS)
def test_neither_is_refused_by_name(reader):
    with pytest.raises(MissingRuntimeValue, match="'max_tokens' is required to run this model and nothing declares it"):
        READERS[reader]({}, {})


@pytest.mark.parametrize("reader", READERS)
def test_the_request_wins_over_the_container(reader):
    assert READERS[reader]({"global.max_tokens": 5}, {"max_tokens": 512}) == 5
    assert READERS[reader]({"global.max_tokens": None}, {"max_tokens": 512}) == 512


def _calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]


@pytest.mark.parametrize("site", SITES)
def test_each_site_reads_the_bound_through_the_one_order(site):
    tree = ast.parse((SRC / site).read_text())
    assert _calls(tree, "max_tokens_of"), f"{site} does not read its bound through max_tokens_of"
    assert not _calls(tree, "require_max_tokens"), f"{site} asks the container itself"


def test_no_get_asks_the_container_as_its_default():
    eager = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get":
                for arg in list(node.args[1:]) + [k.value for k in node.keywords]:
                    if any(_calls(arg, "require_max_tokens")):
                        eager.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not eager, f"an argument default asks the container before the request is read: {eager}"
