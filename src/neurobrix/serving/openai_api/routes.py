"""The OpenAI endpoints, each mapped to the loaded family's own flow.

Which endpoint a model serves is read from its FAMILY's declaration (config/families/<family>.yml: `modes.supported`,
`modes.multimodal_strict`, `inputs.required`, `output.default_extension_per_mode`), never from a model name:

  * /v1/chat/completions — a family declaring the `agent` mode: the caller's messages through the chat template
    (`complete_turn`), tools included; a family declaring `chat` (or a multimodal family's `text` mode) without
    `agent`: one user message, its text and the ONE image or audio part the family requires, through `generate`;
  * /v1/completions — a family declaring `agent`: the raw prompt, no template (`generate`, chat_mode off);
  * /v1/images/generations — a family with a prompt-only mode whose output is a `.png`;
  * /v1/audio/speech — a family with a prompt-only mode whose output is a `.wav`;
  * /v1/audio/transcriptions — a family with a `.txt` mode that requires `--audio`;
  * /v1/embeddings — a family declaring an `embedding` mode (config/families/embedding.yml; none in the catalogue today:
    refused by name for every other family), pooled as the container declares.

Every work request is one operation of the daemon (`ServingDaemon.run_operation`), on the one inference worker,
behind every request of both doors.
"""

from __future__ import annotations

import base64
import binascii
import json
import shutil
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neurobrix.serving.openai_api import constraint_policy as policy, errors
from neurobrix.serving.openai_api.turn import TurnReader, reasoning_gap, select_reasoning, typed_arguments

# -- what each endpoint reads -------------------------------------------------------------------------------
#: Parameters read by each endpoint. A key outside its set is refused by name; a key of `_NEUTRAL` is accepted
#: only at the value that asks for nothing (what the engine does anyway), refused by name at any other.
_CHAT_KEYS = {"model", "messages", "response_format", "stream", "stream_options", "max_tokens", "max_completion_tokens",
              "temperature", "top_p", "top_k", "repetition_penalty", "seed", "tools", "tool_choice",
              "parallel_tool_calls", "user"}
_COMPLETION_KEYS = {"model", "prompt", "stream", "stream_options", "max_tokens", "temperature", "top_p", "top_k",
                    "repetition_penalty", "seed", "user"}
_IMAGE_KEYS = {"model", "prompt", "n", "size", "response_format", "user", "seed", "steps", "guidance_scale"}
_SPEECH_KEYS = {"model", "input", "voice", "response_format", "seed", "temperature"}
_EMBEDDING_KEYS = {"model", "input", "encoding_format", "dimensions", "user"}
_TRANSCRIPTION_KEYS = {"model", "file", "prompt", "response_format", "temperature"}

_NEUTRAL = {
    "n": (1,), "logprobs": (False, None), "top_logprobs": (None, 0), "stop": (None,), "echo": (False,),
    "suffix": (None,), "best_of": (1, None), "frequency_penalty": (0, 0.0, None), "presence_penalty": (0, 0.0, None),
    "logit_bias": (None,), "speed": (1, 1.0, None), "response_format": (None,), "store": (False, None),
    "metadata": (None,), "service_tier": (None, "auto"), "reasoning_effort": (None,),
}
_WHY = {
    "n": "one choice per request; send n requests",
    "logprobs": "the decode does not return log-probabilities",
    "top_logprobs": "the decode does not return log-probabilities",
    "stop": "the decode does not stop on caller strings; it stops on the model's own end-of-turn",
    "echo": "the prompt is not echoed",
    "suffix": "no insertion mode",
    "best_of": "one sample per request",
    "frequency_penalty": "not a sampling control of the engine (it reads repetition_penalty)",
    "presence_penalty": "not a sampling control of the engine (it reads repetition_penalty)",
    "logit_bias": "the decode takes no logit bias",
    "speed": "the speech flow has no speed control",
    "store": "nothing is stored",
    "metadata": "nothing is stored",
    "service_tier": "one tier",
    "reasoning_effort": "the reasoning length is the model's own",
}


def _refusal(e) -> errors.ApiError:
    """The HTTP answer of a constraint the policy refuses (constraint_policy.ConstraintRefused): the policy names the
    kind and the parameter, the status is this door's."""
    if e.kind == "unsupported":
        return errors.unsupported(e.param, e.why)
    return errors.ApiError(400, e.why, param=e.param)


#: what a compile failure of each kind of constraint is answered with: (the request field said in the message, the
#: param of a 400 — the grammar engine cannot enforce it —, the param of a 501 — the engine is not installed). A forced
#: call's envelope is built from `tools`, and it is `tool_choice` that asked for a grammar at all.
_COMPILE_ANSWERS = {"answer": ("response_format", "response_format", "response_format"),
                    "call": ("tools", "tools", "tool_choice")}


def _compiled(want, tokenizer, kind: str):
    """The grammar of what the policy decided (core/state/constraint.compile_constraint): what the grammar engine
    cannot enforce answers 400 naming it; the engine absent on this machine is refused by name, never a 500."""
    from neurobrix.core.state.constraint import compile_constraint
    said, invalid_param, missing_param = _COMPILE_ANSWERS[kind]
    try:
        return compile_constraint(want, tokenizer)
    except ValueError as e:
        raise errors.ApiError(400, f"{said}: {e}", param=invalid_param) from e
    except RuntimeError as e:
        raise errors.ApiError(501, f"constrained decoding is not installed on this machine ({e})", type_="server_error",
                              param=missing_param, code="constrained_decoding_not_installed") from e


def _decided(decide, *facts):
    """A decision of the constraint policy, its refusal answered as this door answers."""
    try:
        return decide(*facts)
    except policy.ConstraintRefused as e:
        raise _refusal(e) from e


def _constrained(body, tokenizer, forced, tools, parser, named, reasoning):
    """The grammar a chat request's decode follows, or None: decided by the policy (constraint_policy), compiled by the
    execution seam. The answer's schema is compiled before the tools are weighed against it; a forced call's
    envelope (the template's own, point 9a) after. `reasoning`: the pair the prompt opens inside (None: none); the
    answer then starts after its end and the text the template writes there (turn.reasoning_gap)."""
    reasoning_end, gap = (reasoning[1], reasoning_gap(tokenizer, reasoning)) if reasoning else (None, None)
    want = _decided(policy.answer_constraint, body.get("response_format"), reasoning_end, gap)
    grammar = _compiled(want, tokenizer, "answer")
    call = _decided(policy.with_tools, want, forced, tools, parser, named, body.get("parallel_tool_calls"),
                    reasoning_end, gap)
    return grammar if call is want else _compiled(call, tokenizer, "call")


def _is_neutral(key: str, value: Any) -> bool:
    if key == "response_format" and isinstance(value, dict):
        return value == {"type": "text"}
    return any(value is a or (type(value) is type(a) and value == a) for a in _NEUTRAL[key])


def _check_keys(body: Dict[str, Any], allowed: set, endpoint: str) -> None:
    for key, value in body.items():
        if key in allowed:
            continue
        if key in _NEUTRAL and _is_neutral(key, value):
            continue
        raise errors.unsupported(key, _WHY.get(key, f"{endpoint} does not read it"))


def _number(body, key, kind=float, low=None, high=None):
    if key not in body or body[key] is None:
        return None
    v = body[key]
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) if kind is float else \
        isinstance(v, int) and not isinstance(v, bool)
    if not ok or (low is not None and v < low) or (high is not None and v > high):
        raise errors.ApiError(400, f"`{key}` must be a {'number' if kind is float else 'integer'}"
                             f"{f' >= {low}' if low is not None else ''}"
                             f"{f' <= {high}' if high is not None else ''}, got {v!r}", param=key)
    return v


def _sampling(body) -> Dict[str, Any]:
    out = {}
    for key, kind, low, high in (("temperature", float, 0, None), ("top_p", float, 0, 1),
                                 ("top_k", int, 0, None), ("repetition_penalty", float, 0, None),
                                 ("seed", int, None, None)):
        v = _number(body, key, kind, low, high)
        if v is not None:
            out[key] = v
    return out


# -- the family's declaration ---------------------------------------------------------------------------------
class Served:
    """What the loaded model's family declares, read once per request (the YAML is cached by the config layer)."""

    def __init__(self, engine) -> None:
        from neurobrix.core.config import get_family_config
        self.family = engine.family or ""
        cfg = get_family_config(self.family)
        modes = cfg.get("modes") or {}
        self.supported: List[str] = list(modes.get("supported") or [])
        self.default_mode = modes.get("default")
        self.strict = bool(modes.get("multimodal_strict"))
        self.required: List[str] = list((cfg.get("inputs") or {}).get("required") or [])
        self.ext: Dict[str, str] = dict((cfg.get("output") or {}).get("default_extension_per_mode") or {})

    def mode_with(self, ext: str, prompt_only: bool) -> Optional[str]:
        """The mode whose output has `ext` (the default mode first), when the family's required inputs allow a
        request carrying only a prompt (`prompt_only`); None when the family serves no such mode."""
        if prompt_only and any(r != "--prompt" for r in self.required):
            return None
        modes = [m for m in self.supported if self.ext.get(m) == ext]
        if not modes:
            return None
        return self.default_mode if self.default_mode in modes else modes[0]

    def mode_arg(self, mode: Optional[str]) -> Dict[str, Any]:
        """The `mode` input of a multimodal-strict family (the others resolve their own)."""
        return {"mode": mode} if (self.strict and mode is not None) else {}


# -- helpers shared by the endpoints ---------------------------------------------------------------------------
def _model(door, body) -> str:
    served = door.daemon.served_name()
    if "model" not in body:
        raise errors.ApiError(400, "`model` is required", param="model")
    if body["model"] != served:
        raise errors.model_not_found(body["model"], served)
    return served


def _run(door, handler, method: str, work, deliver=None) -> Tuple[Any, Dict[str, Any]]:
    """One operation of the daemon for this HTTP request: admitted under the queue bound, run on the worker,
    watched for a vanished client; returns (op, the result) or raises the OpenAI error."""
    daemon = door.daemon
    from neurobrix.serving.operations import ServerBusy
    try:
        op = daemon._ops.open(method, handler.requested_op_id(), max_queued=door.settings["http_max_queued"])
    except ServerBusy as e:
        raise errors.ApiError(429, str(e), "server_busy", code="server_busy")
    except ValueError as e:
        raise errors.ApiError(400, str(e), code="op_id")
    handler.op_id = op.op_id
    door.watch(handler, op)
    state, result = daemon.run_operation(op, work, deliver)
    snap = daemon._ops.snapshot(op)
    if snap["state"] == "failed":
        raise errors.ApiError(500, snap.get("detail") or "the request failed", "server_error", code="engine_error")
    if snap["state"] in ("stopped",):
        raise errors.ApiError(503, "the daemon is shutting down; the request was stopped", "server_error",
                              None, "shutting_down")
    return op, dict(result or {})


def _spool(door, op_hint: str) -> Path:
    path = door.spool_root / op_hint
    path.mkdir(parents=True, exist_ok=False)
    return path


def _media_part(part: Dict[str, Any], spool: Path) -> Tuple[str, Path]:
    """An `image_url` (a data URL) or `input_audio` (base64 + format) part, written to the request's spool."""
    kind = part.get("type")
    if kind == "image_url":
        url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None
        if not isinstance(url, str) or not url.startswith("data:"):
            raise errors.unsupported("image_url", "only a data URL (data:image/...;base64,...) is read; the "
                                     "server fetches nothing from the network")
        head, _, data = url.partition(",")
        if ";base64" not in head:
            raise errors.ApiError(400, "an image data URL must be base64", param="image_url")
        suffix = "." + head[5:].split(";")[0].split("/")[-1]
        payload, name = data, "image"
        flag = "--input-image"
    elif kind == "input_audio":
        audio = part.get("input_audio") if isinstance(part.get("input_audio"), dict) else {}
        payload, fmt = audio.get("data"), audio.get("format")
        if not isinstance(payload, str) or not isinstance(fmt, str) or not fmt.isalnum():
            raise errors.ApiError(400, "input_audio needs `data` (base64) and `format` (e.g. wav)", param="input_audio")
        suffix, name, flag = "." + fmt, "audio", "--audio"
    else:
        raise errors.unsupported(f"content part type {kind!r}", "a part is text, image_url or input_audio")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise errors.ApiError(400, f"the {kind} part is not valid base64", param=kind)
    path = spool / f"{name}{suffix if suffix[1:].isalnum() else ''}"
    path.write_bytes(raw)
    return flag, path


def _text_of(content) -> Tuple[str, List[Dict[str, Any]]]:
    """A message's text and its non-text parts."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise errors.ApiError(400, "a message's `content` is a string or a list of parts", param="messages")
    texts, media = [], []
    for part in content:
        if not isinstance(part, dict):
            raise errors.ApiError(400, "a content part is an object", param="messages")
        if part.get("type") == "text":
            texts.append(str(part.get("text", "")))
        else:
            media.append(part)
    return "".join(texts), media


def _template_messages(messages) -> List[Dict[str, Any]]:
    """The caller's OpenAI messages as the chat template reads them (the transformers convention): text content,
    an assistant's tool calls with their arguments as an object."""
    if not isinstance(messages, list) or not messages:
        raise errors.ApiError(400, "`messages` must be a non-empty list", param="messages")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise errors.ApiError(400, f"messages[{i}] needs a role among system, developer, user, assistant, tool", param="messages")
        text, media = _text_of(m.get("content"))
        if media:
            raise errors.unsupported(f"messages[{i}].content", "this model's family reads text only")
        role = "system" if m["role"] == "developer" else m["role"]
        msg: Dict[str, Any] = {"role": role, "content": text}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = (c or {}).get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        raise errors.ApiError(400, f"messages[{i}].tool_calls: arguments is not JSON", param="messages")
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": fn.get("name"), "arguments": args}})
            msg["tool_calls"] = calls
        for key in ("tool_call_id", "name"):
            if key in m:
                msg[key] = m[key]
        out.append(msg)
    return out


def _call_record(op_id: str, index: int, call, tools) -> Dict[str, Any]:
    arguments = typed_arguments(call.arguments, call.name, tools)
    return {"id": f"call_{op_id}_{index}", "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


# -- endpoints ------------------------------------------------------------------------------------------------
def models(door) -> Dict[str, Any]:
    return {"object": "list", "data": [model_record(door)]}


def model_record(door) -> Dict[str, Any]:
    engine = door.daemon._engine
    return {"id": door.daemon.served_name(), "object": "model", "created": door.started, "owned_by": "neurobrix",
            "neurobrix": {"family": engine.family, "warm_serving": bool(getattr(engine, "_warm_serving", False))}}


def chat_completions(door, handler, body) -> Optional[Dict[str, Any]]:
    _check_keys(body, _CHAT_KEYS, "/v1/chat/completions")
    model = _model(door, body)
    engine = door.daemon._engine
    served = Served(engine)
    stream = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage")) if stream else False
    max_tokens = _number(body, "max_completion_tokens", int, 1)
    if max_tokens is None:
        max_tokens = _number(body, "max_tokens", int, 1)
    decode = _sampling(body)
    if max_tokens is not None:
        decode["max_tokens"] = max_tokens
    tools = body.get("tools") or None
    choice = body.get("tool_choice")
    if tools is not None:
        if not isinstance(tools, list) or any((t or {}).get("type") != "function" for t in tools):
            raise errors.ApiError(400, "`tools` is a list of {type: function, function: {...}}", param="tools")
    named = (choice.get("function") or {}).get("name") if isinstance(choice, dict) else None
    if isinstance(choice, dict) and (choice.get("type") != "function" or not isinstance(named, str)):
        raise errors.ApiError(400, "tool_choice names a function as {type: function, function: {name}}", param="tool_choice")
    if not isinstance(choice, dict) and choice not in (None, "auto", "none", "required"):
        raise errors.ApiError(400, f"tool_choice {choice!r}: auto, none, required or a named function", param="tool_choice")
    forced = choice == "required" or named is not None
    if forced and not tools:
        raise errors.ApiError(400, "tool_choice forces a call and the request declares no tools", param="tool_choice")
    _decided(policy.single_call_needs_a_forced_call, body.get("parallel_tool_calls"), forced)

    tokenizer = engine.tokenizer
    template = engine.chat_template_text()
    if "agent" in served.supported:
        messages = _template_messages(body.get("messages"))
        parser = None
        if tools:
            from neurobrix.agent.parsers import select_parser
            parser = select_parser(template)
            if parser is None:
                raise errors.not_served("/v1/chat/completions with tools", model, served.family,
                                        "its chat template declares no tool-call format")
            if choice == "none":
                parser = None
        reasoning = select_reasoning(template)
        opens = False
        if reasoning is not None:
            kwargs = {"tools": tools} if tools else {}
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            opens = prompt.rstrip().endswith(reasoning[0])
        reader = TurnReader(tokenizer, parser=parser, reasoning=reasoning, opens_in_reasoning=opens)
        grammar = _constrained(body, tokenizer, forced, tools, parser, named, reasoning if opens else None)
        extra = {"grammar": grammar} if grammar is not None else {}
        from neurobrix.core.runtime.prompt_length import ContextLengthExceeded
        try:                               # past what a served session holds: refused before it runs (OpenAI's code)
            engine.check_context(messages, tools=tools, **decode)
        except ContextLengthExceeded as e:
            raise errors.ApiError(400, str(e), param="messages", code="context_length_exceeded") from e
        work = lambda: engine.complete_turn(messages, tools=tools, **extra, **decode)   # noqa: E731
        spool = None
    else:
        mode = "chat" if "chat" in served.supported else ("text" if served.strict and "text" in served.supported
                                                          else None)
        if mode is None:
            raise errors.not_served("/v1/chat/completions", model, served.family,
                                    "its family declares no chat or text-answer mode")
        if _constrained(body, None, False, None, None, None, None) is not None:
            raise errors.not_served("/v1/chat/completions with response_format json_schema", model, served.family,
                                    "a grammar binds the agent path's decode; this family serves chat without it")
        if tools:
            raise errors.not_served("/v1/chat/completions with tools", model, served.family,
                                    "its family does not declare the agent mode")
        messages = body.get("messages")
        if not isinstance(messages, list) or len(messages) != 1 or (messages[0] or {}).get("role") != "user":
            raise errors.unsupported("messages", f"a {served.family} model reads ONE user message (its text and "
                                     "its media); it keeps no history")
        text, media = _text_of(messages[0].get("content"))
        spool = _spool(door, f"{int(time.time() * 1e6)}")
        try:
            params: Dict[str, Any] = {"prompt": text, **decode, **served.mode_arg(mode)}
            given = {}
            for part in media:
                flag, path = _media_part(part, spool)
                if flag in given:
                    raise errors.unsupported("messages", f"one {part.get('type')} part per request")
                given[flag] = path
            for flag in ("--input-image", "--audio"):
                if flag in served.required and flag not in given:
                    raise errors.ApiError(400, f"a {served.family} model needs a "
                                         f"{'image_url' if flag == '--input-image' else 'input_audio'} part", param="messages")
            for flag, path in given.items():
                params["image_path" if flag == "--input-image" else "audio_path"] = str(path)
        except Exception:
            shutil.rmtree(spool, ignore_errors=True)
            raise
        reader = TurnReader(tokenizer) if tokenizer is not None else None
        work = lambda: door.daemon._call_engine("generate", dict(params))   # noqa: E731
        tools = None

    created = int(time.time())
    method = "complete" if "agent" in served.supported else "generate"
    try:
        if stream:
            return _stream_chat(door, handler, method, model, created, reader, work, tools, include_usage)
        op, result = _run(door, handler, method, work)
    finally:
        if spool is not None:
            shutil.rmtree(spool, ignore_errors=True)
    ids = result.get("token_ids")
    if reader is not None and ids is not None:
        for t in ids:
            reader.push(t)
        reader.finish()
        turn = reader.turn
        content, reasoning_text, calls = turn.content, turn.reasoning, turn.calls
    else:
        content, reasoning_text, calls = result.get("text", ""), "", []
    message: Dict[str, Any] = {"role": "assistant", "content": content if (content or not calls) else None}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    if calls:
        message["tool_calls"] = [_call_record(op.op_id, i, c, tools) for i, c in enumerate(calls)]
    out = {"id": f"chatcmpl-{op.op_id}", "object": "chat.completion", "created": created, "model": model,
           "choices": [{"index": 0, "message": message,
                        "finish_reason": _finish(result, calls, ids)}]}
    usage = _usage(result, ids)
    if usage is not None:
        out["usage"] = usage
    return out


def _finish(result, calls, ids) -> str:
    if calls:
        return "tool_calls"
    bound = result.get("max_tokens")
    if bound is not None and ids is not None and len(ids) >= int(bound):
        return "length"
    return "stop"


def _usage(result, ids) -> Optional[Dict[str, int]]:
    if result.get("prompt_tokens") is None or ids is None:
        return None
    p, c = int(result["prompt_tokens"]), len(ids)
    out = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
    if result.get("cached_tokens") is not None:          # held on the card or restored from the host tier
        out["prompt_tokens_details"] = {"cached_tokens": int(result["cached_tokens"])}
    return out


def _stream_chat(door, handler, method, model, created, reader, work, tools, include_usage) -> None:
    sse = handler.start_sse()
    state = {"id": None, "calls": 0}

    def chunk(delta, finish=None):
        return {"id": state["id"], "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def send_deltas(deltas):
        for d in deltas:
            if d.channel == "content":
                sse.send(chunk({"content": d.text}))
            elif d.channel == "reasoning":
                sse.send(chunk({"reasoning_content": d.text}))
            else:
                rec = _call_record(handler.op_id, state["calls"], d.call, tools)
                sse.send(chunk({"tool_calls": [{"index": state["calls"], **rec}]}))
                state["calls"] += 1

    def deliver(step, n, token_id, done):
        if state["id"] is None:
            state["id"] = f"chatcmpl-{handler.op_id}"
            sse.send(chunk({"role": "assistant", "content": ""}))
        if reader is not None:
            send_deltas(reader.push(token_id))

    try:
        op, result = _run(door, handler, method, work, deliver=lambda *a: sse.guard(deliver, *a))
    except errors.ApiError as e:
        if not sse.started:
            raise                   # nothing streamed yet: the error answers with its own status
        sse.send(e.body())
        sse.done()
        return None
    if state["id"] is None:
        state["id"] = f"chatcmpl-{op.op_id}"
        sse.send(chunk({"role": "assistant", "content": ""}))
    ids = result.get("token_ids")
    if reader is not None:
        send_deltas(reader.finish())
        calls = reader.turn.calls
    else:
        text = result.get("text", "")
        if text:
            sse.send(chunk({"content": text}))
        calls = []
    sse.send(chunk({}, _finish(result, calls, ids)))
    if include_usage:
        usage = _usage(result, ids)
        sse.send({"id": state["id"], "object": "chat.completion.chunk", "created": created, "model": model,
                  "choices": [], "usage": usage})
    sse.done()
    return None


def completions(door, handler, body) -> Optional[Dict[str, Any]]:
    _check_keys(body, _COMPLETION_KEYS, "/v1/completions")
    model = _model(door, body)
    engine = door.daemon._engine
    served = Served(engine)
    if "agent" not in served.supported:
        raise errors.not_served("/v1/completions", model, served.family,
                                "a raw-prompt completion is a language model's (a family declaring `agent`)")
    prompt = body.get("prompt")
    if not isinstance(prompt, str):
        raise errors.unsupported("prompt", "one prompt string per request")
    decode = _sampling(body)
    mt = _number(body, "max_tokens", int, 1)
    if mt is not None:
        decode["max_tokens"] = mt
    params = {"prompt": prompt, "chat_mode": False, **decode}
    work = lambda: door.daemon._call_engine("generate", dict(params))   # noqa: E731
    created = int(time.time())
    if body.get("stream"):
        sse = handler.start_sse()
        reader = TurnReader(engine.tokenizer)

        def deliver(step, n, token_id, done):
            for d in reader.push(token_id):
                sse.send({"id": f"cmpl-{handler.op_id}", "object": "text_completion", "created": created,
                          "model": model, "choices": [{"index": 0, "text": d.text, "finish_reason": None}]})
        try:
            op, result = _run(door, handler, "generate", work, deliver=lambda *a: sse.guard(deliver, *a))
        except errors.ApiError as e:
            if not sse.started:
                raise               # nothing streamed yet: the error answers with its own status
            sse.send(e.body())
            sse.done()
            return None
        tail = "".join(d.text for d in reader.finish())
        sse.send({"id": f"cmpl-{op.op_id}", "object": "text_completion", "created": created, "model": model,
                  "choices": [{"index": 0, "text": tail, "finish_reason": "stop"}]})
        sse.done()
        return None
    op, result = _run(door, handler, "generate", work)
    text = result.get("text", "")
    return {"id": f"cmpl-{op.op_id}", "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}]}


def _binary(door, handler, endpoint, mode, params, ext) -> bytes:
    """Run a generate whose output is a file of the family's own format; return its bytes."""
    spool = _spool(door, f"{int(time.time() * 1e6)}")
    try:
        params = dict(params, output_path=str(spool / f"out{ext}"))
        op, result = _run(door, handler, "generate", lambda: door.daemon._call_engine("generate", params))
        saved = result.get("output_path")
        if not saved or not Path(saved).is_file():
            raise errors.ApiError(500, f"{endpoint}: the flow wrote no {ext} file", "server_error", code="engine_error")
        return Path(saved).read_bytes()
    finally:
        shutil.rmtree(spool, ignore_errors=True)


def images_generations(door, handler, body) -> Dict[str, Any]:
    _check_keys(body, _IMAGE_KEYS, "/v1/images/generations")
    model = _model(door, body)
    served = Served(door.daemon._engine)
    mode = served.mode_with(".png", prompt_only=True)
    if mode is None:
        raise errors.not_served("/v1/images/generations", model, served.family,
                                "its family has no prompt-only mode that writes an image")
    if not isinstance(body.get("prompt"), str) or not body["prompt"]:
        raise errors.ApiError(400, "`prompt` is required", param="prompt")
    fmt = body.get("response_format")
    if fmt not in (None, "b64_json"):
        raise errors.unsupported("response_format", "only b64_json: the server hosts no files to link to")
    n = _number(body, "n", int, 1) or 1
    params: Dict[str, Any] = {"prompt": body["prompt"], **served.mode_arg(mode)}
    size = body.get("size")
    if size not in (None, "auto"):
        try:
            w, h = (int(x) for x in str(size).lower().split("x"))
        except ValueError:
            raise errors.ApiError(400, f"`size` is WIDTHxHEIGHT, got {size!r}", param="size")
        params["width"], params["height"] = w, h
    steps = _number(body, "steps", int, 1)
    if steps is not None:
        params["steps"] = steps
    cfg = _number(body, "guidance_scale", float, 0)
    if cfg is not None:
        params["cfg"] = cfg
    seed = _number(body, "seed", int)
    data = []
    for i in range(n):
        one = dict(params)
        if seed is not None:
            one["seed"] = seed + i
        png = _binary(door, handler, "/v1/images/generations", mode, one, ".png")
        data.append({"b64_json": base64.b64encode(png).decode("ascii")})
    return {"created": int(time.time()), "data": data}


def audio_speech(door, handler, body) -> bytes:
    _check_keys(body, _SPEECH_KEYS, "/v1/audio/speech")
    model = _model(door, body)
    served = Served(door.daemon._engine)
    mode = served.mode_with(".wav", prompt_only=True)
    if mode is None:
        raise errors.not_served("/v1/audio/speech", model, served.family,
                                "its family has no prompt-only mode that writes audio")
    if not isinstance(body.get("input"), str) or not body["input"]:
        raise errors.ApiError(400, "`input` is required", param="input")
    if body.get("response_format") not in (None, "wav"):
        raise errors.unsupported("response_format", "only wav: the engine ships no audio encoder")
    params: Dict[str, Any] = {"prompt": body["input"], **served.mode_arg(mode)}
    if body.get("voice") is not None:
        params["speaker"] = body["voice"]
    for key, kind in (("seed", int), ("temperature", float)):
        v = _number(body, key, kind)
        if v is not None:
            params[key] = v
    return _binary(door, handler, "/v1/audio/speech", mode, params, ".wav")


def audio_transcriptions(door, handler, form: Dict[str, Any]) -> Any:
    _check_keys(form, _TRANSCRIPTION_KEYS, "/v1/audio/transcriptions")
    model = _model(door, form)
    served = Served(door.daemon._engine)
    if "--audio" not in served.required or not any(served.ext.get(m) == ".txt" for m in served.supported):
        raise errors.not_served("/v1/audio/transcriptions", model, served.family,
                                "its family does not answer an audio input with text")
    upload = form.get("file")
    if not isinstance(upload, tuple):
        raise errors.ApiError(400, "`file` (multipart/form-data) is required", param="file")
    filename, raw = upload
    fmt = form.get("response_format")
    if fmt not in (None, "json", "text"):
        raise errors.unsupported("response_format", "json or text: the flow returns no segment timings")
    params: Dict[str, Any] = {}
    prompt = form.get("prompt")
    if "--prompt" in served.required and not prompt:
        raise errors.ApiError(400, f"a {served.family} model needs a `prompt` (the instruction it applies to the audio)", param="prompt")
    if prompt:
        params["prompt"] = prompt
    if form.get("temperature") not in (None, ""):
        try:
            params["temperature"] = float(form["temperature"])
        except ValueError:
            raise errors.ApiError(400, "`temperature` is a number", param="temperature")
    suffix = Path(filename or "").suffix
    if not suffix[1:].isalnum():
        suffix = ""
    spool = _spool(door, f"{int(time.time() * 1e6)}")
    try:
        path = spool / f"audio{suffix}"
        path.write_bytes(raw)
        params["audio_path"] = str(path)
        if served.strict:
            params["mode"] = "text"
        op, result = _run(door, handler, "generate", lambda: door.daemon._call_engine("generate", params))
    finally:
        shutil.rmtree(spool, ignore_errors=True)
    text = result.get("transcription") if result.get("transcription") is not None else result.get("text", "")
    if fmt == "text":
        return str(text)
    return {"text": str(text)}


def embeddings(door, handler, body) -> Dict[str, Any]:
    """One vector per input, pooled as the loaded embedding model's container declares (output_dispatch.extract_
    embeddings); a family that declares no embedding mode is refused by name (none in the catalogue today)."""
    _check_keys(body, _EMBEDDING_KEYS, "/v1/embeddings")
    model = _model(door, body)
    served = Served(door.daemon._engine)
    if "embedding" not in served.supported:
        raise errors.not_served("/v1/embeddings", model, served.family, "its family declares no embedding output")
    inputs = [body["input"]] if isinstance(body.get("input"), str) else body.get("input")
    if not isinstance(inputs, list) or not inputs or not all(isinstance(x, str) and x for x in inputs):
        raise errors.ApiError(400, "`input` is a non-empty string or a list of non-empty strings (token-id arrays are "
                                   "not accepted: the model's own tokenizer reads the text)", param="input")
    fmt = body.get("encoding_format") or "float"
    if fmt not in ("float", "base64"):
        raise errors.ApiError(400, f"`encoding_format` is float or base64, got {fmt!r}", param="encoding_format")
    data, tokens = [], 0
    for i, text in enumerate(inputs):
        params = {"prompt": text, **served.mode_arg("embedding")}
        _op, result = _run(door, handler, "generate", lambda p=params: door.daemon._call_engine("generate", p))
        if "dimensions" in body and body["dimensions"] != result["dimension"]:
            raise errors.ApiError(400, f"`dimensions` must be this model's {result['dimension']}: its vectors are not "
                                       f"truncated", param="dimensions")
        vec = result["embeddings"][0]
        tokens += int(result["tokens"])
        data.append({"object": "embedding", "index": i, "embedding": vec if fmt == "float" else
                     base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")})
    return {"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}
