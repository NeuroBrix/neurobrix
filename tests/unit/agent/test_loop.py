"""Loop tests — scripted complete_fn, no engine: turn flow, tool execution,
malformed retry, max-turns bound, transcript artifact."""

import json

import pytest

from neurobrix.agent import AgentConfig, SandboxPolicy, run_agent_task

_TEMPLATE = "{% if tools %}<tool_call><function=...{% endif %}"  # xml markers


def _scripted(turns):
    state = {"i": 0}

    def complete_fn(messages, tools_payload, **gen):
        assert tools_payload, "tools schema must reach the completion call"
        text = turns[state["i"]]
        state["i"] += 1
        return text

    return complete_fn


def _run(tmp_path, turns, **config_kwargs):
    return run_agent_task(
        complete_fn=_scripted(turns),
        task="do the thing",
        workdir=str(tmp_path),
        template_text=_TEMPLATE,
        transcript_dir=str(tmp_path / "_t"),
        policy=SandboxPolicy(mode="yolo"),
        config=AgentConfig(**config_kwargs),
    )


def test_write_run_fix_session(tmp_path):
    turns = [
        "<tool_call>\n<function=write_file>\n<parameter=path>\nm.py\n"
        "</parameter>\n<parameter=content>\nX = 1\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        "python3 -c 'import m; print(m.X)'\n</parameter>\n</function>\n</tool_call>",
        "The module works. Done.",
    ]
    result = _run(tmp_path, turns)
    assert result.stop_reason == "final_answer"
    assert result.turns == 3
    assert (tmp_path / "m.py").read_text() == "X = 1"


def test_tool_results_are_reinjected(tmp_path):
    captured = {}

    def complete_fn(messages, tools_payload, **gen):
        if len(messages) == 2:  # system + task
            return (
                "<tool_call>\n<function=bash>\n<parameter=command>\necho hello-42\n"
                "</parameter>\n</function>\n</tool_call>"
            )
        captured["messages"] = [dict(m) for m in messages]
        return "done"

    run_agent_task(
        complete_fn=complete_fn,
        task="t",
        workdir=str(tmp_path),
        template_text=_TEMPLATE,
        transcript_dir=str(tmp_path / "_t"),
        policy=SandboxPolicy(mode="yolo"),
    )
    tool_messages = [m for m in captured["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "hello-42" in tool_messages[0]["content"]


def test_malformed_retries_once_then_errors(tmp_path):
    turns = ["<tool_call>\n<function=bash>", "<tool_call>\n<function=bash>"]
    result = _run(tmp_path, turns, malformed_retries=1)
    assert result.stop_reason == "error"
    assert result.turns == 2


def test_max_turns_bound(tmp_path):
    call = (
        "<tool_call>\n<function=bash>\n<parameter=command>\ntrue\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    result = _run(tmp_path, [call] * 5, max_turns=3)
    assert result.stop_reason == "max_turns"
    assert result.turns == 3


def test_jail_escape_is_surfaced_not_fatal(tmp_path):
    turns = [
        "<tool_call>\n<function=read_file>\n<parameter=path>\n../../etc/passwd\n"
        "</parameter>\n</function>\n</tool_call>",
        "understood, staying inside",
    ]
    result = _run(tmp_path, turns)
    assert result.stop_reason == "final_answer"
    events = [
        json.loads(line)
        for line in (tmp_path / "_t" / "transcript.jsonl").read_text().splitlines()
    ]
    refusals = [
        e for e in events
        if e["kind"] == "tool_result" and "REFUSED by sandbox" in e["output"]
    ]
    assert len(refusals) == 1


def test_transcript_files_exist_and_finish(tmp_path):
    _run(tmp_path, ["done immediately"])
    md = (tmp_path / "_t" / "transcript.md").read_text()
    assert "Finished — final_answer" in md
    events = [
        json.loads(line)
        for line in (tmp_path / "_t" / "transcript.jsonl").read_text().splitlines()
    ]
    assert events[-1]["kind"] == "finish"


def test_no_tools_template_is_refused(tmp_path):
    with pytest.raises(ValueError):
        run_agent_task(
            complete_fn=lambda *a, **k: "x",
            task="t",
            workdir=str(tmp_path),
            template_text="a template with no tool contract",
            transcript_dir=str(tmp_path / "_t"),
        )
