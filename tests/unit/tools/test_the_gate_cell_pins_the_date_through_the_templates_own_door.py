"""The same-request gate cell pins the date a template renders through the template's OWN system-message door.
Without a card or a container: small template strings stand in for the vendor templates.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import gate_cell_same_request as G  # noqa: E402

GRANITE = (
    "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n"
    "    {%- set loop_messages = messages[1:] %}\n{%- else %}\n"
    "    {%- set system_message = \"Knowledge Cutoff Date: April 2024.\nToday's Date: \" + strftime_now('%B %d, %Y') + "
    "\".\nYou are Granite, developed by IBM.\" %}\n"
    "    {%- set system_message = system_message + \" You are a helpful AI assistant.\" %}\n"
    "    {%- set loop_messages = messages %}\n{%- endif %}\n"
    "{{- '<|start_of_role|>system<|end_of_role|>' + system_message + '<|end_of_text|>\n' }}\n"
    "{%- for message in loop_messages %}\n"
    "    {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}\n"
    "    {%- if loop.last and add_generation_prompt %}\n        {{- '<|start_of_role|>assistant<|end_of_role|>' }}\n"
    "    {%- endif %}\n{%- endfor %}")

ORPHEUS = (
    "{%- if not date_string is defined %}\n    {%- set date_string = strftime_now(\"%d %b %Y\") %}\n{%- endif %}\n"
    "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n"
    "    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n"
    "{{- \"<|start_header_id|>system<|end_header_id|>\n\nToday Date: \" + date_string + \"\n\n\" }}\n"
    "{{- system_message }}\n{%- for message in messages %}\n    {{- message['content'] }}\n{%- endfor %}")

NO_DATE = "{%- for message in messages %}{{- message['role'] + ': ' + message['content'] + '\n' }}{%- endfor %}"
PINNED = datetime(2026, 11, 23)
USER = [{"role": "user", "content": "Name a planet of the solar system."}]


def test_the_system_message_is_read_from_the_template_with_only_the_date_fixed():
    """The message is the template's default text at the pinned date, taken from the template and not from this tool."""
    said = G.default_system_message(GRANITE, PINNED)
    assert said == ("Knowledge Cutoff Date: April 2024.\nToday's Date: November 23, 2026.\n"
                    "You are Granite, developed by IBM. You are a helpful AI assistant.")
    assert G.default_system_message(GRANITE.replace("April 2024", "May 2025"), PINNED).startswith("Knowledge Cutoff Date: May 2025.")


def test_the_prompt_with_the_tools_system_message_equals_the_default_render_at_the_pinned_date():
    """The explicit system message and the default one give the same prompt, byte for byte, at the pinned date."""
    said = G.default_system_message(GRANITE, PINNED)
    explicit = G.render(GRANITE, [{"role": "system", "content": said}] + USER)
    assert explicit == G.render(GRANITE, USER, PINNED)
    assert "Today's Date: November 23, 2026." in explicit
    assert explicit.endswith("<|start_of_role|>assistant<|end_of_role|>")
    # and it no longer moves with the day the request is sent
    assert explicit == G.render(GRANITE, [{"role": "system", "content": said}] + USER, datetime(2030, 1, 1))


def test_a_template_that_writes_the_date_elsewhere_is_refused_by_name():
    """A Llama-3 `date_string` template is refused with the tool's own words: a system message cannot pin it."""
    with pytest.raises(G.DateCannotBePinned, match="this template's date cannot be pinned through a system message"):
        G.default_system_message(ORPHEUS, PINNED)


def test_a_template_without_a_date_needs_no_system_message():
    """A template that renders no date sends no system message: nothing to pin."""
    assert G.default_system_message(NO_DATE, PINNED) is None


def test_the_gating_bodies_carry_only_the_model_and_the_messages():
    """The body is empty of sampling keys, so the container's greedy defaults apply; the system message leads each."""
    bodies = G.bodies("m", "SYS", ["q1", "q2", "q3"])
    assert len(bodies) == 3 and bodies[0]["messages"][1]["content"] == "q1"
    assert all(set(b) == {"model", "messages"} and b["messages"][0] == {"role": "system", "content": "SYS"} for b in bodies)
    assert all(len(b["messages"]) == 1 for b in G.bodies("m", None, ["q1", "q2"]))
