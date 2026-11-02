"""ONE sampling path for every text, vlm and audio_llm decode, both branches (the supervisor 2026-11-02 06:40): the four
parameters (temperature, top-k, top-p, repetition penalty) are resolved one way — the request, then the container, then
the family file — and drawn by the branch's one CombinedSampler.

What this test would do if the code were wrong:
  - a flow that reads its sampling by hand (the vlm and audio_llm decodes read `global.temperature` themselves, audio_llm
    took the container's temperature whatever the request said, the Triton text table dropped --repetition-penalty) or
    draws with its own helper (`audio_utils.sample_token`, `_sample_token_nbx`, neither knowing top-k): RED by file;
  - a request's value that loses to the container, a container null that hides the family's value, or a missing value
    given a literal instead of a refusal: RED;
  - the warm daemon path dropping --top-k, --top-p or --max-tokens, or the daemon refusing them: RED.
"""
from __future__ import annotations

import ast
import pathlib
import types

import pytest


SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "neurobrix"
DECODE_FLOWS = [f"{b}/flow/{f}.py" for b in ("core", "triton") for f in ("autoregressive", "vlm", "audio_llm")]
OWN_SAMPLERS = {"sample_token", "_sample_token_nbx"}
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")


def _resolver(request=None, family=None):
    return types.SimpleNamespace(resolved={f"global.{k}": v for k, v in (request or {}).items()},
                                 family_defaults=family or {})


def test_the_request_wins_then_the_container_then_the_family():
    from neurobrix.core.module.sampling_contract import resolve_sampling
    family = {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.0}
    container = {"temperature": 0.6, "top_k": 20, "top_p": None, "repetition_penalty": 1.2}
    got = resolve_sampling(_resolver({"top_k": 3}, family), container, "test")
    assert got == {"temperature": 0.6, "top_k": 3, "top_p": 0.9, "repetition_penalty": 1.2}
    assert isinstance(got["top_k"], int) and isinstance(got["temperature"], float)


def test_a_value_no_source_carries_is_refused_by_name():
    from neurobrix.core.module.sampling_contract import resolve_sampling
    with pytest.raises(RuntimeError, match="no value for top_k"):
        resolve_sampling(_resolver(), {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}, "test")


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            yield f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


@pytest.mark.parametrize("rel", DECODE_FLOWS)
def test_every_decode_flow_samples_through_the_one_path(rel):
    text = (SRC / rel).read_text()
    tree = ast.parse(text)
    called = set(_calls(tree))
    own = called & OWN_SAMPLERS
    assert not own, f"{rel} draws with its own sampler {sorted(own)}, not the branch's CombinedSampler"
    assert called & {"resolve_sampling", "decode_sampler"}, f"{rel} resolves no sampling through resolve_sampling"
    by_hand = [k for k in SAMPLING_KEYS
               if f'"global.{k}"' in text or f"defaults.get(\"{k}\"" in text or f'defaults["{k}"]' in text]
    assert not by_hand, f"{rel} reads {by_hand} by hand instead of resolve_sampling"


def test_the_triton_text_table_reads_the_requested_penalty():
    from neurobrix.triton.flow.autoregressive import _build_generator_config
    container = {"max_tokens": 8, "temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}
    config = _build_generator_config(container, _resolver({"repetition_penalty": 1.3}))
    assert config["repetition_penalty"] == 1.3


def test_the_decode_samplers_honour_top_k():
    """vlm and audio_llm decodes drew with helpers that had no top-k (MiniCPM-o's container declares top_k=20, Qwen3-VL's
    20): through the branch's sampler a top-1 cut leaves the largest logit only, whatever the draw."""
    torch = pytest.importorskip("torch")
    from neurobrix.core.module.autoregressive.samplers import decode_sampler, sample_next_id
    container = {"temperature": 1.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0}
    sampler = decode_sampler(_resolver(), container, "test")
    logits = torch.tensor([[[0.0, 2.0, 2.5, 1.0]]])
    assert {sample_next_id(sampler, logits, []) for _ in range(16)} == {2}


def test_the_warm_path_carries_every_sampling_flag(monkeypatch):
    from neurobrix.cli.commands import run as R
    from neurobrix.serving import client as C
    seen = {}

    class Fake:
        is_running = staticmethod(lambda: True)
        get_pid = staticmethod(lambda: 0)
        def connect(self): pass
        def close(self): pass
        def status(self): return {"model": "m", "family": None}
        def generate(self, **kw):
            seen.update(kw)
            raise RuntimeError("stop after the request")
    monkeypatch.setattr(C, "DaemonClient", Fake)
    from neurobrix.cli import create_parser
    args = create_parser().parse_args(["run", "--model", "m", "--prompt", "p", "--top-k", "5", "--top-p", "0.5",
                                      "--max-tokens", "7", "--temperature", "0.2"])
    try:
        R._try_warm_path(args)
    except Exception:
        pass
    assert {k: seen.get(k) for k in ("top_k", "top_p", "max_tokens", "temperature")} == {
        "top_k": 5, "top_p": 0.5, "max_tokens": 7, "temperature": 0.2}, seen


def test_the_daemon_reads_top_k_and_top_p():
    from neurobrix.serving.engine import InferenceEngine
    inputs = InferenceEngine._decode_inputs({"top_k": 5, "top_p": 0.5, "temperature": 0.2})
    assert inputs["global.top_k"] == 5 and inputs["global.top_p"] == 0.5
