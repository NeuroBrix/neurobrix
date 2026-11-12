"""Decides what a chat request's decode is constrained by.

Contract.
- In: the request's own constraining fields — `response_format`; its tools and whether `tool_choice` forces a call
  (`required`, or a named function); `parallel_tool_calls`; the dialect parser its template declares; the reasoning's
  close marker when the prompt opens inside a reasoning (else None), and what the template writes between that marker
  and the answer (None when the template does not show it).
- Out: a `Constrain` (the execution seam's own input type, core/state/constraint.py) — WHAT the decode must follow: a JSON schema, or a forced tool call (the dialect, the tools, the
  named function, one call or several), each optionally after the reasoning's end — or None: a request that
  constrains nothing pays nothing. Data in, data out: no tokenizer, no grammar, no token. The execution seam
  (`core/state/constraint.compile_constraint`) turns the value into a compiled grammar.
- Refuses with `ConstraintRefused(kind, param, why)`; the route maps the kind to its HTTP answer and nothing here
  knows a status: `unsupported` (a `response_format` other than text or json_schema; a json_schema answer together
  with a forced call, or with tools; one call at most asked without a forced call), `invalid` (a json_schema with no
  schema object).
- Two decisions, asked in this order by the route, because the answer's schema is compiled (and may be refused by
  the grammar engine) before the tools are weighed against it: `answer_constraint`, then `with_tools`.

The OpenAI route is today the only door that can carry these fields (the daemon's socket protocol carries neither a
response format nor tools): the module lives under it, and moves up to serving/ the day a second door constrains.
"""
from __future__ import annotations

from typing import Any, Optional

from neurobrix.core.state.constraint import Constrain, ForcedCall      # the execution seam's input: its own types


class ConstraintRefused(Exception):
    """A combination of constraining fields the server does not honour: `kind` (unsupported / invalid), the request
    parameter at fault, why."""

    def __init__(self, kind: str, param: str, why: str) -> None:
        super().__init__(why)
        self.kind, self.param, self.why = kind, param, why


def single_call_needs_a_forced_call(parallel_tool_calls: Any, forced: bool) -> None:
    """`parallel_tool_calls: false` is enforced by the forced call's grammar: without one it is refused."""
    if parallel_tool_calls is False and not forced:
        raise ConstraintRefused("unsupported", "parallel_tool_calls",
                                "one call at most is enforced only with tool_choice required "
                                "or a named function (with auto the turn may also answer in prose)")


def answer_constraint(response_format: Any, reasoning_end: Optional[str] = None,
                      reasoning_gap: Optional[str] = None) -> Optional[Constrain]:
    """What `response_format` asks of the answer: None for no format or text; the schema of a json_schema (after the
    reasoning's end when the prompt opens inside one); json_object has no schema to enforce and is refused."""
    if response_format is None or response_format == {"type": "text"}:
        return None
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        raise ConstraintRefused("unsupported", "response_format",
                                "only text or json_schema (json_object has no schema to enforce)")
    schema = (response_format.get("json_schema") or {}).get("schema")
    if not isinstance(schema, dict):
        raise ConstraintRefused("invalid", "response_format",
                                "response_format.json_schema.schema must be a JSON Schema object")
    return Constrain(schema=schema, after_reasoning=reasoning_end or None, reasoning_gap=reasoning_gap)


def with_tools(answer: Optional[Constrain], forced: bool, tools: Optional[list], parser: Any, named: Optional[str],
               parallel_tool_calls: Any, reasoning_end: Optional[str] = None,
               reasoning_gap: Optional[str] = None) -> Optional[Constrain]:
    """The answer's constraint weighed against the tools: a forced call replaces nothing — a schema with it, or with
    tools the turn may call, is refused (the grammar would bind the calls too); a forced call alone is the constraint."""
    if forced:
        if answer is not None:
            raise ConstraintRefused("unsupported", "response_format",
                                    "a json_schema answer and a forced tool call in one request")
        return Constrain(call=ForcedCall(parser, tools, named, parallel_tool_calls is not False),
                         after_reasoning=reasoning_end or None, reasoning_gap=reasoning_gap)
    if answer is not None and tools:
        raise ConstraintRefused("unsupported", "response_format",
                                "a json_schema answer and tools in one request: the grammar would "
                                "also bind the tool calls")
    return answer
