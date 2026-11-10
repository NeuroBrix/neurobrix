"""What a chat request's decode is constrained by — nothing, a JSON schema, a forced tool call's envelope, with or
without the reasoning's end in front — or the refusal it gets, for every combination of the request's constraining
fields, compared with a recording.

Why it exists. The decision is being regrouped into one module (the bricks rule: the decision gets its brick; no
behaviour change). The recording (`data/constraint_decisions.json`) was taken on the tree BEFORE the move — commit
of this file alone — through the OpenAI door with the test engine, the grammar compilers replaced by recorders: for
each case the HTTP status, the error's code, param and message, and the exact text each compiler was handed. The move must
reproduce it with this file untouched.

What this test does if the code is wrong: a combination answers another status, code or param, hands a compiler
another description (a key, the reasoning marker, the parallel flag, the named function), compiles when it should
not or does not when it should — red, naming the case. Seen red by changing one recorded entry.

Record again (only on a tree whose behaviour is the reference): NBX_RECORD_CONSTRAINT_TABLE=1 python -m pytest <this file>
"""
import itertools
import json
import os
from pathlib import Path

import pytest

from tests.unit.serving.test_the_openai_door_answers_in_openai_shapes import FakeEngine, _post, make  # noqa: F401

RECORD = Path(__file__).with_name("data") / "constraint_decisions.json"

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
TOOLS = [{"type": "function", "function": {"name": "get_time", "parameters": {
    "type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]

#: the engines: what the template declares (a tool-call dialect, a reasoning the prompt opens in) and the family
ENGINES = {
    "plain": dict(),
    "dialect": dict(template="{% for t in tools %}{{ t }}{% endfor %} <tool_call> </tool_call>"),
    "reasoning": dict(pieces=["The user", "</think>", "\n\n", "{}"], template="<think> ... </think>", opening="<think>\n"),
    "dialect+reasoning": dict(pieces=["The user", "</think>", "\n\n", "{}"], opening="<think>\n",
                              template="{% for t in tools %}{{ t }}{% endfor %} <tool_call> </tool_call> <think> ... </think>"),
    "chat-family": dict(family="vlm"),
}
FORMATS = {"none": None, "text": {"type": "text"}, "json_object": {"type": "json_object"},
           "json_schema": {"type": "json_schema", "json_schema": {"name": "c", "schema": SCHEMA}},
           "json_schema-unenforceable": {"type": "json_schema", "json_schema": {"schema": {"type": "string", "pattern": "a+"}}}}
TOOLING = {"none": {}, "tools": {"tools": TOOLS}, "tools+auto": {"tools": TOOLS, "tool_choice": "auto"},
           "tools+none": {"tools": TOOLS, "tool_choice": "none"}, "tools+required": {"tools": TOOLS, "tool_choice": "required"},
           "tools+named": {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "get_time"}}},
           "tools+named-unknown": {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "nope"}}},
           "required-without-tools": {"tool_choice": "required"}}
PARALLEL = {"unset": {}, "false": {"parallel_tool_calls": False}}


def _engine(name):
    eng = FakeEngine(**ENGINES[name])
    if "dialect" in name:
        eng.tokenizer.encode = lambda text, add_special_tokens=True: (
            [900] if text in ("<tool_call>", "</tool_call>") else list(text.encode()))
    return eng


def _decide(make, monkeypatch, engine, fmt, tooling, parallel):
    from neurobrix.core.state import constraint
    seen = []
    monkeypatch.setattr(constraint, "compile_description", lambda text: seen.append(["description", json.loads(text)]) or object())
    monkeypatch.setattr(constraint, "compile_json_schema", lambda text: seen.append(["schema", json.loads(text)]) or object())
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], **TOOLING[tooling], **PARALLEL[parallel]}
    if FORMATS[fmt] is not None:
        body["response_format"] = FORMATS[fmt]
    _, base = make(_engine(engine))
    status, answer, _ = _post(base + "/v1/chat/completions", body)
    err = (answer or {}).get("error") or {}
    return {"status": status, "code": err.get("code"), "param": err.get("param"), "message": err.get("message"),
            "compiled": seen}


def _cases():
    return ["|".join(c) for c in itertools.product(ENGINES, FORMATS, TOOLING, PARALLEL)]


def test_every_combination_is_decided_as_recorded(make, monkeypatch):  # noqa: F811
    from tests.unit.grammar_engine import load
    load()
    got = {case: _decide(make, monkeypatch, *case.split("|")) for case in _cases()}
    if os.environ.get("NBX_RECORD_CONSTRAINT_TABLE") == "1":
        RECORD.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    want = json.loads(RECORD.read_text())
    assert sorted(got) == sorted(want), "the recorded cases and the table's differ: record again on the reference tree"
    wrong = [f"{case}: recorded {want[case]} — now {got[case]}" for case in sorted(got) if got[case] != want[case]]
    assert not wrong, f"{len(wrong)} of {len(got)} decisions moved:\n" + "\n".join(wrong[:8])


def test_the_recording_covers_every_branch():
    want = json.loads(RECORD.read_text())
    kinds = {(v["status"], v["code"], v["param"], tuple(k for k, _ in v["compiled"])) for v in want.values()}
    assert any(k[3] == ("schema",) for k in kinds) and any(k[3] == ("description",) for k in kinds)
    assert any(k[0] == 200 and not k[3] for k in kinds)
    assert {k[0] for k in kinds} >= {200, 400}
    descriptions = [d for v in want.values() for k, d in v["compiled"] if k == "description"]
    assert any("until" in json.dumps(d) for d in descriptions), "no recorded case opens inside a reasoning"
