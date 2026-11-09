"""The tool-call envelope of a request, as a grammar description our grammar engine enforces (constrained decoding,
reste_apres_064 point 9a; the Dell's review 2026-11-09 22:21).

The dialect is the template's (agent/parsers: the parser select_parser picked), never a model's name. Each dialect's
format is the one its parser reads back: json (`<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>`), xml
(`<tool_call>\\n<function=NAME>\\n<parameter=K>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>`), json_list
(`<|tool_call|>[{"name": ..., "arguments": {...}}, ...]`). A marker the tokenizer writes as ONE id is a token step (the
model emits it only as that id), any other marker its text. The function is chosen by its name, its arguments by the
tool's own JSON schema. `required`: at least one call; a named function: only that one; `parallel`: False -> one call.
After the calls, only the end of the turn. The xml dialect writes the required parameters in their declared order
(stricter than the schema, never looser); a string parameter is free text up to its closing tag.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional


def encoder(tokenizer) -> Callable[[str], List[int]]:
    """The tokenizer's own ids for a marker's text: no special tokens added, no padding (where the class pads)."""
    import inspect
    params = inspect.signature(tokenizer.encode).parameters
    kw = {"add_special_tokens": False, **({"padding": False} if "padding" in params else {})}
    return lambda text: [int(i) for i in tokenizer.encode(text, **kw)]


def marker(text: str, encode: Callable[[str], List[int]]) -> Dict[str, Any]:
    ids = encode(text)
    return {"token": int(ids[0])} if len(ids) == 1 else {"lit": text}


def _params(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool.get("function") or {}
    p = fn.get("parameters") or {"type": "object", "properties": {}}
    if p.get("type") != "object":
        raise ValueError(f"tool {fn.get('name')!r}: parameters must be a JSON Schema object")
    return p


def _xml_call(tool: Dict[str, Any]) -> Dict[str, Any]:
    p = _params(tool)
    props, required = p.get("properties") or {}, list(p.get("required") or [])
    parts: List[Dict[str, Any]] = [{"lit": "\n"}]
    for name in [k for k in props if k in required]:
        sch = props[name]
        parts.append({"lit": f"<parameter={name}>\n"})
        if sch.get("type") == "string" and not ({"enum", "const"} & set(sch)):
            parts.append({"until": "\n</parameter>\n"})
        else:
            parts += [{"schema": sch}, {"lit": "\n</parameter>\n"}]
    parts.append({"lit": "</function>\n"})
    return {"seq": parts}


def envelope(dialect: str, tools: List[Dict[str, Any]], encode: Callable[[str], List[int]],
             only: Optional[str] = None, parallel: bool = True) -> Dict[str, Any]:
    """The grammar description of the tool calls a turn must make (`only`: the one function named by tool_choice)."""
    chosen = [t for t in tools if only is None or (t.get("function") or {}).get("name") == only]
    if not chosen:
        raise ValueError(f"tool_choice names {only!r}, which is not among the request's tools")
    names = [(t.get("function") or {}).get("name") for t in chosen]
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("every tool needs a function name")
    rep = {"min": 1} if parallel else {"min": 1, "max": 1}
    if dialect == "json":
        open_, close = marker("<tool_call>", encode), marker("</tool_call>", encode)
        choice = [{"lit": json.dumps(n) + ", ", "then": {"seq": [
            {"lit": "\"arguments\": "}, {"schema": _params(t)}, {"lit": "}\n"}, close]}} for n, t in zip(names, chosen)]
        return {"repeat": {"seq": [open_, {"lit": "\n{\"name\": "}, {"choice": choice}]}, **rep, "sep": "\n"}
    if dialect == "xml":
        open_, close = marker("<tool_call>", encode), marker("</tool_call>", encode)
        choice = [{"lit": f"{n}>", "then": {"seq": [_xml_call(t), close]}} for n, t in zip(names, chosen)]
        return {"repeat": {"seq": [open_, {"lit": "\n<function="}, {"choice": choice}]}, **rep, "sep": "\n"}
    if dialect == "json_list":
        from neurobrix.agent.parsers.jsonlist import JsonListToolCallParser
        choice = [{"lit": "{\"name\": " + json.dumps(n) + ", ", "then": {"seq": [
            {"lit": "\"arguments\": "}, {"schema": _params(t)}, {"lit": "}"}]}} for n, t in zip(names, chosen)]
        return {"seq": [marker(JsonListToolCallParser.start_marker, encode), {"lit": "["},
                        {"repeat": {"choice": choice}, **rep, "sep": ", "}, {"lit": "]"}]}
    raise ValueError(f"no tool-call envelope for the dialect {dialect!r}")
