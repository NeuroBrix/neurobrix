"""The OpenAI-compatible HTTP door (serving/openai_api/): a real ServingDaemon, its real HTTP door on a loopback
port and its Unix socket, with fake engines per family (no model, no card). The family's REAL declaration
(config/families/*.yml) decides what each endpoint serves.

Injections (each RED, then GREEN; see the commit message):
  * the queue bound not checked (`max_queued` ignored in OperationTable.open) — the 429 test fails;
  * the API key not compared (`_authorized` always True) — the auth test fails;
  * an unread parameter accepted (`_check_keys` a no-op) — the refusal test fails;
  * the HTTP work run on the connection thread instead of the worker — the shared-worker test sees overlap.
"""
from __future__ import annotations

import base64
import json
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from neurobrix import decode_progress, decode_stop
from neurobrix.serving import client as client_mod
from neurobrix.serving.client import DaemonClient
from neurobrix.serving.openai_api.door import HttpDoor, port_owner
from neurobrix.serving.server import ServingDaemon

pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix-socket transport")

STEP_S = 0.02


class PieceTokenizer:
    def __init__(self, pieces, special=()):
        self.pieces = list(pieces)
        self.special = set(special)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids if not (skip_special_tokens and i in self.special))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kw):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>" + self.opening
        return list(range(len(text))) if tokenize else text

    opening = ""


class FakeEngine:
    model_name = "fake-model"
    _warm_serving = True
    _session = None
    hardware_id = "none"
    mode = "compiled"

    def __init__(self, family="llm", pieces=("Hello", " world", "."), special=(), template="", answer=None,
                 opening="", step_s=STEP_S):
        self.family = family
        self.tokenizer = PieceTokenizer(pieces, special)
        self.tokenizer.opening = opening
        self.template = template
        self.answer = list(answer if answer is not None else range(len(pieces)))
        self.step_s = step_s
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _decode(self, max_tokens=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        ids = []
        try:
            for i, t in enumerate(self.answer[: max_tokens or len(self.answer)]):
                time.sleep(self.step_s)
                ids.append(t)
                done = len(ids) == len(self.answer)
                stop = not done and decode_stop.requested(len(ids))
                decode_progress.emit(i, len(ids), t, done or stop)
                if stop:
                    break
        finally:
            with self._lock:
                self.active -= 1
        return ids

    def complete_turn(self, messages, tools=None, **kw):
        self.calls.append(("complete", messages, tools, kw))
        ids = self._decode(kw.get("max_tokens"))
        return {"text": self.tokenizer.decode(ids), "token_ids": ids, "prompt_tokens": 7, "cached_tokens": 4,
                "max_tokens": kw.get("max_tokens") or 64}

    def complete(self, messages, tools=None, **kw):
        return self.complete_turn(messages, tools, **kw)["text"]

    def generate(self, prompt="", **kw):
        self.calls.append(("generate", prompt, kw))
        for key in ("image_path", "audio_path"):
            if key in kw:
                assert Path(kw[key]).is_file(), f"{key} was not spooled"
                self.calls.append((key, Path(kw[key]).read_bytes()))
        if self.family in ("image", "tts"):
            return {"outputs": {"x": 1}}
        if self.family == "stt":
            return {"transcription": "the transcript"}
        ids = self._decode(kw.get("max_tokens"))
        return {"text": self.tokenizer.decode(ids), "tokens": len(ids)}

    def save_output(self, outputs, output_path, mode=None, orig_hw=None):
        data = b"\x89PNG fake" if output_path.endswith(".png") else b"RIFF fake wav"
        Path(output_path).write_bytes(data)
        return output_path

    def sync_devices(self):
        pass

    def get_status(self):
        return {"model": self.model_name, "loaded": True}

    def chat_template_text(self):
        return self.template

    def new_conversation(self):
        pass

    def unload(self):
        pass


SETTINGS = {"http_host": "127.0.0.1", "http_port": 1, "http_max_queued": 8, "http_max_body_mb": 4}


@pytest.fixture
def make(monkeypatch):
    made = []

    def _make(engine, api_key=None, settings=None):
        d = Path(tempfile.mkdtemp(prefix="nbxh"))
        path = d / "d.sock"
        srv = ServingDaemon("fake", "none", engine=engine, idle_timeout=60)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        sock.listen(srv._settings["listen_backlog"])
        door = HttpDoor(srv, host="127.0.0.1", port=0, api_key=api_key, settings=settings or SETTINGS)
        door.spool_root = d / "spool"
        srv._door = door
        door.open()
        t = threading.Thread(target=srv.serve, args=(sock,), daemon=True)
        t.start()
        monkeypatch.setattr(client_mod, "SOCKET_PATH", path)
        made.append((srv, t, d))
        return srv, f"http://127.0.0.1:{door.port}"

    yield _make
    for srv, t, d in made:
        srv._ops.stop_all("shutdown")
        srv._close_listener()
        t.join(10)
        srv._door.close()
        shutil.rmtree(d, ignore_errors=True)


def _post(url, body, headers=None, raw=False):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = r.read()
            return r.status, (payload if raw else json.loads(payload)), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _sse(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        lines = [ln.decode() for ln in r.read().split(b"\n\n") if ln.strip()]
    assert lines[-1] == "data: [DONE]", lines[-1]
    return [json.loads(ln[len("data: "):]) for ln in lines[:-1]]


def _assert_error(status, body, want_status, param=None, code=None):
    assert status == want_status, (status, body)
    err = body["error"]
    assert set(err) == {"message", "type", "param", "code"} and err["message"]
    if param is not None:
        assert err["param"] == param, err
    if code is not None:
        assert err["code"] == code, err


def test_models_lists_the_one_model_and_an_unknown_one_is_named(make):
    _, base = make(FakeEngine())
    status, body = _get(base + "/v1/models")
    assert status == 200 and [m["id"] for m in body["data"]] == ["fake-model"]
    assert body["data"][0]["object"] == "model"
    _assert_error(*_get(base + "/v1/models/other"), 404, code="model_not_found")
    status, body, _ = _post(base + "/v1/chat/completions", {"model": "other", "messages": [{"role": "user",
                                                                                          "content": "x"}]})
    _assert_error(status, body, 404, param="model", code="model_not_found")


def test_a_chat_completion_carries_the_answer_its_usage_and_the_op_id(make):
    srv, base = make(FakeEngine())
    status, body, headers = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
        "temperature": 0.5, "n": 1})
    assert status == 200, body
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world."}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
                             "prompt_tokens_details": {"cached_tokens": 4}}     # held on the card or the host tier
    op_id = headers["X-NeuroBrix-Op-Id"]
    assert body["id"] == f"chatcmpl-{op_id}"
    _, messages, _, decode = srv._engine.calls[-1]
    assert messages == [{"role": "user", "content": "hi"}] and decode == {"temperature": 0.5, "max_tokens": 16}


def test_tool_calls_are_parsed_server_side_from_the_template_s_dialect(make):
    pieces = ["<|end|>", "<|tool_call|>", '[{"name": "get_weather", "arguments": {"city": "Paris"}}]']
    engine = FakeEngine(pieces=pieces, special=[0, 1], answer=[1, 2, 0],
                        template="... respond with <|tool_call|> followed by a JSON list of tools used ...")
    _, base = make(engine)
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}}]
    status, body, _ = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "weather?"}], "tools": tools})
    assert status == 200, body
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    (call,) = choice["message"]["tool_calls"]
    assert call["type"] == "function" and call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    # the history a client sends back (arguments as a JSON string) reaches the template as an object
    status, body, _ = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "tools": tools, "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "sunny"}]})
    assert status == 200, body
    sent = engine.calls[-1][1]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert sent[2] == {"role": "tool", "content": "sunny", "tool_call_id": call["id"]}


def test_tools_on_a_template_without_a_tool_format_are_refused_by_name(make):
    _, base = make(FakeEngine(template="{{ messages }}"))
    status, body, _ = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f"}}]})
    _assert_error(status, body, 409, code="model_does_not_serve_endpoint")
    assert "tool-call format" in body["error"]["message"]


def test_a_streamed_chat_sends_role_then_stable_deltas_reasoning_apart_then_done(make):
    pieces = ["The user", " wants 4.", "</think>", "\n\n", "4"]
    engine = FakeEngine(pieces=pieces, template="<think> ... </think>", opening="<think>\n")
    _, base = make(engine)
    chunks = _sse(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "2+2"}], "stream": True,
        "stream_options": {"include_usage": True}})
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert reasoning == "The user wants 4." and content == "\n\n4"
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert [c["choices"][0]["finish_reason"] for c in finals] == ["stop"]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 5
    assert len({c["id"] for c in chunks}) == 1


def test_an_unread_parameter_is_refused_by_name_and_a_neutral_one_is_accepted(make):
    _, base = make(FakeEngine())
    msg = [{"role": "user", "content": "x"}]
    for key, value in (("stop", ["\n"]), ("logprobs", True), ("n", 2), ("frequency_penalty", 0.5),
                       ("tool_choice", "required"), ("parallel_tool_calls", False), ("made_up", 1)):
        status, body, _ = _post(base + "/v1/chat/completions", {"model": "fake-model", "messages": msg, key: value})
        _assert_error(status, body, 400, param=key)
    status, body, _ = _post(base + "/v1/chat/completions", {"model": "fake-model", "messages": msg, "n": 1,
                                                            "stop": None, "presence_penalty": 0,
                                                            "response_format": {"type": "text"}})
    assert status == 200, body


def test_the_api_key_is_required_when_the_door_has_one(make):
    _, base = make(FakeEngine(), api_key="s3cret")
    _assert_error(*_get(base + "/v1/models"), 401, code="invalid_api_key")
    _assert_error(*_get(base + "/v1/models", {"Authorization": "Bearer wrong"}), 401)
    assert _get(base + "/v1/models", {"Authorization": "Bearer s3cret"})[0] == 200


def test_a_non_loopback_door_without_a_key_is_refused_at_start():
    with pytest.raises(RuntimeError, match="needs an API key"):
        HttpDoor(object(), host="0.0.0.0", port=0, settings=SETTINGS)


def test_a_taken_port_is_refused_naming_the_port_and_its_holder():
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        door = HttpDoor(object(), host="127.0.0.1", port=port, settings=SETTINGS)
        with pytest.raises(RuntimeError) as e:
            door.open()
        assert str(port) in str(e.value) and "held by" in str(e.value), str(e.value)
        assert "PID" in port_owner(port)
    finally:
        holder.close()


def test_beyond_the_queue_bound_a_request_gets_429_server_busy(make):
    engine = FakeEngine(pieces=[str(i) for i in range(40)], step_s=0.05)
    _, base = make(engine, settings={**SETTINGS, "http_max_queued": 1})
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "x"}]}
    results = []
    threads = [threading.Thread(target=lambda: results.append(_post(base + "/v1/chat/completions", body)))
               for _ in range(2)]
    for t in threads:          # one runs, one waits
        t.start()
        time.sleep(0.3)
    status, err, _ = _post(base + "/v1/chat/completions", body)
    _assert_error(status, err, 429, code="server_busy")
    for t in threads:
        t.join(30)
    assert sorted(r[0] for r in results) == [200, 200]


def test_both_doors_share_the_one_worker(make):
    engine = FakeEngine(pieces=[str(i) for i in range(20)], step_s=0.03)
    srv, base = make(engine)
    c = DaemonClient()
    c.connect()
    t = threading.Thread(target=lambda: c.generate("socket", max_tokens=20))
    t.start()
    time.sleep(0.1)
    status, _, _ = _post(base + "/v1/chat/completions",
                         {"model": "fake-model", "messages": [{"role": "user", "content": "x"}]})
    t.join(30)
    c.close()
    assert status == 200 and engine.max_active == 1, f"two generations overlapped ({engine.max_active})"


def test_a_stream_client_that_vanishes_orphans_its_operation(make):
    engine = FakeEngine(pieces=[str(i) for i in range(200)], step_s=0.02)
    srv, base = make(engine)
    s = socket.create_connection(("127.0.0.1", int(base.rsplit(":", 1)[1])))
    payload = json.dumps({"model": "fake-model", "messages": [{"role": "user", "content": "x"}],
                          "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              + f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
    s.recv(4096)
    s.close()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        ops = list(srv._ops._ops.values())
        if ops and ops[-1].state == "orphaned":
            break
        time.sleep(0.05)
    assert ops[-1].state == "orphaned", ops[-1].state


def test_a_vlm_reads_one_user_message_and_its_image_part_spooled_then_removed(make):
    engine = FakeEngine(family="vlm")
    srv, base = make(engine)
    image = base64.b64encode(b"\x89PNG pixels").decode()
    status, body, _ = _post(base + "/v1/chat/completions", {"model": "fake-model", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "what is it?"},
                                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}]})
    assert status == 200, body
    assert body["choices"][0]["message"]["content"] == "Hello world."
    assert ("image_path", b"\x89PNG pixels") in engine.calls
    assert not any(srv._door.spool_root.iterdir()), "the request's spool was left behind"
    status, body, _ = _post(base + "/v1/chat/completions", {"model": "fake-model", "messages": [
        {"role": "user", "content": "no image"}]})
    _assert_error(status, body, 400, param="messages")


def test_an_image_generation_answers_base64_png_and_refuses_urls(make):
    engine = FakeEngine(family="image")
    _, base = make(engine)
    status, body, _ = _post(base + "/v1/images/generations", {"model": "fake-model", "prompt": "a lighthouse",
                                                             "size": "512x256", "n": 2, "seed": 5})
    assert status == 200, body
    assert [base64.b64decode(d["b64_json"]) for d in body["data"]] == [b"\x89PNG fake"] * 2
    gens = [c for c in engine.calls if c[0] == "generate"]
    assert [(g[2]["width"], g[2]["height"], g[2]["seed"]) for g in gens] == [(512, 256, 5), (512, 256, 6)]
    _assert_error(*_post(base + "/v1/images/generations",
                         {"model": "fake-model", "prompt": "x", "response_format": "url"})[:2], 400,
                  param="response_format")
    _assert_error(*_post(base + "/v1/audio/speech", {"model": "fake-model", "input": "x"})[:2], 409,
                  code="model_does_not_serve_endpoint")


def test_speech_answers_wav_bytes(make):
    engine = FakeEngine(family="tts")
    _, base = make(engine)
    status, data, headers = _post(base + "/v1/audio/speech",
                                  {"model": "fake-model", "input": "hello", "voice": "af_heart"}, raw=True)
    assert status == 200 and headers["Content-Type"] == "audio/wav" and data == b"RIFF fake wav"
    assert engine.calls[-1][2]["speaker"] == "af_heart"


def test_a_transcription_reads_a_multipart_upload(make):
    engine = FakeEngine(family="stt")
    _, base = make(engine)
    boundary = "nbxboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nfake-model\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + b"RIFFaudio" + f"\r\n--{boundary}--\r\n".encode()
    status, out, _ = _post(base + "/v1/audio/transcriptions", body,
                           {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 200 and out == {"text": "the transcript"}, out
    assert ("audio_path", b"RIFFaudio") in engine.calls


def test_embeddings_are_refused_by_name_where_no_family_declares_them(make):
    _, base = make(FakeEngine())
    status, body, _ = _post(base + "/v1/embeddings", {"model": "fake-model", "input": "x"})
    _assert_error(status, body, 409, code="model_does_not_serve_endpoint")


def test_an_engine_failure_is_a_500_with_its_named_refusal_and_the_daemon_serves_on(make):
    engine = FakeEngine()

    def boom(*a, **k):
        raise RuntimeError("Prism: CUDA out of memory on cuda:0 (needs 2.1 GB, 0.4 GB free)")
    engine.complete_turn = boom
    _, base = make(engine)
    status, body, _ = _post(base + "/v1/chat/completions",
                            {"model": "fake-model", "messages": [{"role": "user", "content": "x"}]})
    _assert_error(status, body, 500, code="engine_error")
    assert "out of memory" in body["error"]["message"]
    assert _get(base + "/v1/models")[0] == 200


def test_a_json_schema_answer_is_compiled_and_handed_to_the_decode(make):
    """response_format json_schema: the schema compiled by our grammar engine and handed to complete_turn; a construct it
    cannot enforce answers 400 naming it; json_object refused by name (point 9a)."""
    pytest.importorskip("nbx_grammar")
    eng = FakeEngine()
    srv, base = make(eng)
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    status, body, _ = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": schema}}})
    assert status == 200, body
    kw = [c for c in eng.calls if c[0] == "complete"][-1][3]
    assert kw.get("grammar") is not None
    status, body, _ = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "string", "pattern": "a+"}}}})
    _assert_error(status, body, 400, param="response_format")
    assert "pattern" in body["error"]["message"]
    status, body, _ = _post(base + "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_object"}})
    assert status == 400 and body["error"]["param"] == "response_format"
