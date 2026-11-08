"""A language model's prefill is priced at the request's own prompt length, never at the cache's prompt margin.

2026-11-08 (the arena measurement): TinyLlama-1.1B-Chat-v1.0 on the Triton engine planned ~0.01 GiB of activations for a
20-, a 200- and a 400-token prompt alike while the Triton arena held 102, 314 and 652 MB: `request_input_config` left
`seq_len` unset for a tokenizer model and `FlowBindings.prompt_prefill_length` bound the generating LM's sequence to
`defaults.prompt_margin` (128). The prompt's length is now counted by the flow's own rule (core/runtime/prompt_length)
and bound to the LM's sequence symbol; the margin stays the bound only when the request states no prompt.

What this test does if the code is wrong: the first cell fails when the request's InputConfig carries no seq_len (red
on main: None); the second when the LM's sequence symbol is bound to the margin instead of the request (red on main:
128); the third when the planned walk (KV apart) does not grow with the prompt (red on main: equal).

Run: python -m pytest tests/unit/prism/test_a_prompt_is_priced_at_its_own_length.py
"""
import json

import pytest

from tests.unit.prism._pinned_machine import container_root, profile, V100_16GB, pin_no_calibration

MODEL = "TinyLlama-1.1B-Chat-v1.0"
SHORT = "Explain why the sky is blue."
LONG = " ".join(["The quick brown fox jumps over the lazy dog near the river bank."] * 30)


def _request(prompt):
    from neurobrix.cli import create_parser
    from neurobrix.cli.commands.run import request_input_config
    from neurobrix.nbx import NBXContainer
    c = NBXContainer.load(str(container_root(MODEL)))
    man = c.get_manifest() or {}
    args = create_parser().parse_args(["run", "--model", MODEL, "--prompt", prompt, "--max-tokens", "4", "--triton"])
    return c, request_input_config(args, man, man.get("family"), c.cache_path)


def test_the_request_carries_the_prompts_token_count():
    from neurobrix.core.runtime.prompt_length import prompt_tokens
    c, ic = _request(LONG)
    assert ic.seq_len == prompt_tokens(c.cache_path, LONG)
    assert ic.seq_len > 128                      # longer than the margin: the case the margin hid


def test_the_language_models_sequence_is_bound_to_the_request_not_the_margin():
    c, ic = _request(LONG)
    dag = json.loads((c.cache_path / "components" / "model" / "graph.json").read_text())
    seq = [sid for sid, s in dag["symbolic_context"]["symbols"].items() if s["name"] == "seq_len"]
    assert seq, "the decoder's graph names its sequence symbol seq_len"
    bound = ic.flow.overrides(dag, ic)
    assert all(bound[sid] == ic.seq_len for sid in seq)


def test_the_planned_activation_grows_with_the_prompt(monkeypatch):
    from neurobrix.core.prism import PrismSolver
    pin_no_calibration(monkeypatch)
    act = {}
    for name, prompt in (("short", SHORT), ("long", LONG)):
        c, ic = _request(prompt)
        plan = PrismSolver().solve(c, profile(V100_16GB), ic, mode="triton")
        m = plan.component_memory["model"]
        # the walk without the KV estimate the LM's figure carries (it grows with the decode bound on main too)
        act[name] = int(m.activation_bytes) - int(getattr(m, "kv_bytes", 0) or 0)
    assert act["long"] > 4 * act["short"], act
