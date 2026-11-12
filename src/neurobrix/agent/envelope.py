"""The tool-call envelope of a request, as a grammar description our grammar engine enforces (constrained decoding,
reste_apres_064 point 9a; the Dell's review 2026-11-09 22:21).

The dialect is the template's (agent/parsers: the parser select_parser picked), never a model's name. Each dialect's
format is the one its parser reads back: json (`<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>`), xml
(`<tool_call>\\n<function=NAME>\\n<parameter=K>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>`), json_list
(`<|tool_call|>[{"name": ..., "arguments": {...}}, ...]`). A marker the tokenizer writes as ONE id is a token step (the
model emits it only as that id), any other marker its text. The function is chosen by its name, its arguments by the
tool's own JSON schema. `required`: at least one call; a named function: only that one; `parallel`: False -> one call.
After the calls, only the end of the turn. The xml dialect writes the required parameters in their declared order
(stricter than the schema, never looser); a string parameter is free text up to its closing tag, an enum or a const
one of its values written bare (a string as its text, as the dialect's parser reads it back).
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


def reasoning_close(end: str, encode: Callable[[str], List[int]]) -> Dict[str, Any]:
    """The reasoning a prompt opens in, free text up to its end marker. A marker the tokenizer writes as ONE id (Qwen3's
    `</think>`, a special token with no bytes) ends it as that id too: the model closes its reasoning with the token it
    was trained on, never forced to spell the marker in plain bytes."""
    ids = encode(end)
    return {"until": end, "end_token": int(ids[0])} if len(ids) == 1 else {"until": end}


def _params(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool.get("function") or {}
    p = fn.get("parameters") or {"type": "object", "properties": {}}
    if p.get("type") != "object":
        raise ValueError(f"tool {fn.get('name')!r}: parameters must be a JSON Schema object")
    return p


def _raw(value: Any) -> str:
    """A value as the xml dialect writes it: a string bare (the parser keeps it as text), anything else as its JSON."""
    return value if isinstance(value, str) else json.dumps(value)


def _free_text(name: str, sch: Dict[str, Any], dialect: str) -> None:
    """A string parameter a dialect writes raw is free text, which cannot enforce a constraint on it: refused by name,
    never dropped."""
    extra = set(sch) - {"type", "description", "title", "examples", "default"}
    if extra:
        raise ValueError(f"parameter {name!r}: {sorted(extra)} cannot be enforced on a raw {dialect} string value")


def _xml_call(tool: Dict[str, Any]) -> Dict[str, Any]:
    p = _params(tool)
    props, required = p.get("properties") or {}, list(p.get("required") or [])
    parts: List[Dict[str, Any]] = [{"lit": "\n"}]
    for name in [k for k in props if k in required]:
        sch = props[name]
        parts.append({"lit": f"<parameter={name}>\n"})
        if {"enum", "const"} & set(sch):        # the bare alternatives, never their JSON literals (the Dell's D3)
            values = [sch["const"]] if "const" in sch else list(sch["enum"])
            close = "\n</parameter>"            # in the literal: "a" and "ab" stay distinct alternatives
            parts.append({"choice": [{"lit": _raw(v) + close, "then": {"lit": "\n"}} for v in values]})
        elif sch.get("type") == "string":
            _free_text(name, sch, "xml")
            parts.append({"until": "\n</parameter>\n"})
        else:
            parts += [{"schema": sch}, {"lit": "\n</parameter>\n"}]
    parts.append({"lit": "</function>\n"})
    return {"seq": parts}


def _bare_choice(options, what: str) -> Dict[str, Any]:
    """A choice among literals written bare, each followed by its marker and the rest of its parts:
    `options` = [(literal, marker, rest)], with `rest` a list of parts. A marker that is text joins its literal (the
    literals then differ by bytes). A marker that is one token cannot, and the grammar decides a choice by bytes, so
    no literal may then be a prefix of another: refused by name (GLM and Mistral write a name, and GLM an enum value,
    bare before a marker token)."""
    if any("token" in after for _, after, _ in options):
        lits = [lit for lit, _, _ in options]
        clash = next(((a, b) for a in lits for b in lits if a != b and b.startswith(a)), None)
        if clash:
            raise ValueError(f"{what}: {clash[0]!r} is a prefix of {clash[1]!r}; this dialect writes it bare before a "
                             "marker token, and the grammar engine decides a choice by its bytes")
    out = []
    for lit, after, rest in options:
        if "token" in after:
            out.append({"lit": lit, "then": {"seq": [after, *rest]}})
        elif rest:
            out.append({"lit": lit + after["lit"], "then": {"seq": rest}})
        else:                          # nothing after the marker: its last byte is the choice's continuation
            text = lit + after["lit"]
            out.append({"lit": text[:-1], "then": {"lit": text[-1]}})
    return {"choice": out}


def _glm_arguments(params, inner, encode, close_call) -> List[Dict[str, Any]]:
    """For each required parameter in its declared order, its key then its value, each between its pair of the
    parser's `inner` markers (key open, key close, value open, value close), then the call's close: a string free
    text up to the value's close, an enum or a const one of its values bare, anything else the schema's JSON (GLM's
    template writes `v | tojson if v is not string else v`)."""
    if not params:
        return [close_call]
    (name, sch), rest = params[0], _glm_arguments(params[1:], inner, encode, close_call)
    key_open, key_close, value_open, end = (marker(m, encode) for m in inner)
    head = [key_open, {"lit": name}, key_close, value_open]
    if {"enum", "const"} & set(sch):
        values = [sch["const"]] if "const" in sch else list(sch["enum"])
        return head + [_bare_choice([(_raw(v), end, rest) for v in values], f"parameter {name!r}")]
    if sch.get("type") == "string":
        _free_text(name, sch, "glm")
        return head + [reasoning_close(inner[-1], encode)] + rest
    return head + [{"schema": sch}, end] + rest


def _glm(parser, names, chosen, encode, rep) -> Dict[str, Any]:
    close = marker(parser.end_marker, encode)
    options = []
    for n, t in zip(names, chosen):
        p = _params(t)
        props, required = p.get("properties") or {}, set(p.get("required") or [])
        parts = _glm_arguments([(k, props[k]) for k in props if k in required], parser.inner_markers, encode, close)
        options.append((n, parts[0], parts[1:]))
    return {"repeat": {"seq": [marker(parser.start_marker, encode), _bare_choice(options, "tool names")]}, **rep}


def _mistral_args(parser, names, chosen, encode, rep) -> Dict[str, Any]:
    args = marker(parser.inner_markers[0], encode)
    options = [(n, args, [{"schema": _params(t)}]) for n, t in zip(names, chosen)]
    return {"repeat": {"seq": [marker(parser.start_marker, encode), _bare_choice(options, "tool names")]}, **rep}


# the dialects whose name is written bare before a marker (the others are written below, in envelope)
_BARE_NAME_DIALECTS = {"glm": _glm, "mistral_args": _mistral_args}


def envelope(parser, tools: List[Dict[str, Any]], encode: Callable[[str], List[int]],
             only: Optional[str] = None, parallel: bool = True) -> Dict[str, Any]:
    """The grammar description of the tool calls a turn must make (`only`: the one function named by tool_choice)."""
    chosen = [t for t in tools if only is None or (t.get("function") or {}).get("name") == only]
    if not chosen:
        raise ValueError(f"tool_choice names {only!r}, which is not among the request's tools")
    names = [(t.get("function") or {}).get("name") for t in chosen]
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("every tool needs a function name")
    # a named function is exactly ONE call of it (OpenAI, vLLM); parallel false: one call too
    rep = {"min": 1} if parallel and only is None else {"min": 1, "max": 1}
    dialect = parser.format_name          # the markers are the parser's own, never retyped here
    if dialect in _BARE_NAME_DIALECTS:
        return _BARE_NAME_DIALECTS[dialect](parser, names, chosen, encode, rep)
    if dialect == "json":
        open_, close = marker(parser.start_marker, encode), marker(parser.end_marker, encode)
        choice = [{"lit": json.dumps(n) + ", ", "then": {"seq": [
            {"lit": "\"arguments\": "}, {"schema": _params(t)}, {"lit": "}\n"}, close]}} for n, t in zip(names, chosen)]
        return {"repeat": {"seq": [open_, {"lit": "\n{\"name\": "}, {"choice": choice}]}, **rep, "sep": "\n"}
    if dialect == "xml":
        open_, close = marker(parser.start_marker, encode), marker(parser.end_marker, encode)
        choice = [{"lit": f"{n}>", "then": {"seq": [_xml_call(t), close]}} for n, t in zip(names, chosen)]
        return {"repeat": {"seq": [open_, {"lit": "\n<function="}, {"choice": choice}]}, **rep, "sep": "\n"}
    if dialect == "json_list":
        choice = [{"lit": "{\"name\": " + json.dumps(n) + ", ", "then": {"seq": [
            {"lit": "\"arguments\": "}, {"schema": _params(t)}, {"lit": "}"}]}} for n, t in zip(names, chosen)]
        return {"seq": [marker(parser.start_marker, encode), {"lit": "["},
                        {"repeat": {"choice": choice}, **rep, "sep": ", "}, {"lit": "]"}]}
    raise ValueError(f"no tool-call envelope for the dialect {dialect!r}")
