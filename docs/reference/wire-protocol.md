# The daemon wire protocol

What a client needs to talk to a running `neurobrix serve` daemon, written for
a reader with no access to the engine's code (Studio, a script, another
language). Protocol version **1**. Source of truth in the code:
`src/neurobrix/serving/protocol.py` (framing, endpoints, version, limits) and
`src/neurobrix/serving/server.py` (methods).

## 1. Where the daemon listens

One daemon serves one loaded model. Several can run side by side; each is an
*instance*, named by the environment variable `NBX_SOCKET_PATH` (the same
variable on the daemon and on every client of it).

| OS | transport | default instance (`NBX_SOCKET_PATH` unset) | named instance (`NBX_SOCKET_PATH=<path>`) |
|---|---|---|---|
| macOS, Linux | Unix domain socket (`AF_UNIX`, stream) | `~/.neurobrix/daemon.sock` | the socket at `<path>` |
| Windows | TCP on loopback `127.0.0.1` only | port `19384` | a port the OS chooses at bind, written as decimal text to `<path minus extension>.port`; read it there before connecting. No port file = no daemon (refused, never guessed) |

Beside the endpoint, each instance owns a pid file and a log file:
`~/.neurobrix/daemon.pid` / `daemon.log` for the default instance,
`<stem>.pid` / `<stem>.log` for a named one. `neurobrix info --json` prints the
endpoint of the instance the calling environment names, as
`endpoint: {"kind": "unix", "path": ...}` or
`{"kind": "tcp", "address": "127.0.0.1", "port": ..., "port_file"?: ...}`.

Nothing listens on a network interface: there is no HTTP, REST or gRPC layer.

## 2. Framing

Every message, in both directions, is one frame:

```
+----------------------------+-----------------------------------+
| length: uint32, big-endian | payload: `length` bytes, UTF-8 JSON |
+----------------------------+-----------------------------------+
```

- `length` counts payload bytes only (not the 4-byte header).
- The payload is one JSON object.
- **Hard limit: `length` ≤ 67 108 864 bytes (64 MiB).** A larger declared
  length is refused before any payload byte is read.
- A connection carries any number of request/response exchanges, one after
  the other. A client that closes the connection ends it; a frame cut short by
  a disconnect is discarded.
- A malformed frame is answered with one error envelope whose `error` starts
  with `malformed frame:`. An empty payload, a payload that is not UTF-8 JSON,
  or JSON that is not an object leaves the stream framed: the connection stays
  open for the next frame. A declared length above the limit leaves no frame
  boundary to trust: the connection is closed after the answer.

## 3. Requests and responses

### Request

```json
{"protocol": 1, "method": "generate", "op_id": "abc", "params": {"prompt": "a red apple", "steps": 4}}
```

| field | type | meaning |
|---|---|---|
| `protocol` | integer | the protocol version the client speaks (section 5) |
| `method` | string | one of the methods in section 6 |
| `params` | object | the method's inputs; `{}` (or absent, or `null`) when it takes none |
| `op_id` | string, optional | work methods only (`generate`, `chat`, `complete`): the client's name for this operation, used to cancel it and to match stream events. Absent: the daemon assigns `"<pid>-<n>"`. An `op_id` equal to one still running is refused |

### Response

Every response carries the protocol version and the engine version that
produced it:

```json
{"protocol": 1, "engine": "0.x.y", "op_id": "abc", "result": { ... }}
{"protocol": 1, "engine": "0.x.y", "op_id": "abc", "error": "what went wrong"}
```

Exactly one of `result` or `error` is present. `op_id` is present on every
answer to a work method, error or not, once the operation was opened.

### Stream events (generate with `"stream": true`)

Before the final response, the daemon sends one frame per produced token on the
same connection:

```json
{"stream": {"step": 3, "n": 64, "token": 1234, "done": false, "op_id": "abc", "text": "fox"}}
```

`step` counts from 0, `n` is the token budget, `done` is true on the last one;
for a family whose answer is text, every event also carries `text` (the
decoded piece, possibly empty) and `rewind` (characters of the text already
sent to erase before appending `text`). A frame without a `stream` key is the
final response.

## 4. Errors

All errors use the response envelope with an `error` string. The string names
what was refused and why; it is meant to be shown to a person. Errors a client
can meet without any engine fault:

| situation | `error` begins with | connection |
|---|---|---|
| declared frame length above 64 MiB | `malformed frame: Message too large` | closed |
| empty payload / not UTF-8 JSON / not an object | `malformed frame:` | stays open |
| `protocol` outside the daemon's range | `protocol <n> is not served here` (section 5) | closed |
| `protocol` present but not an integer (`"1"`, `true`, `1.0`, `null`, ...) | ``malformed request: `protocol` must be an integer`` | closed |
| `method` not a string, or `params` not an object | `malformed request:` | stays open |
| unknown `method` | `Unknown method: <name>` | stays open |
| `cancel` without a non-empty string `params.op_id` | `cancel needs params.op_id` | stays open |
| work request whose `op_id` is not a non-empty string, or names a live operation | `refused: op_id ...` | stays open |
| engine failure during the work | the engine's message; `op_id` present | stays open |

## 5. Versions and the accepted range

`protocol` is one integer. It changes when the shape of a method's request or
response changes; adding a method does not change it.

A daemon accepts a **range** `[min, max]` of client versions, published by
`neurobrix info --json` (`protocol_range`) and by the `status` method. A request
whose `protocol` is outside the range is refused before any work, on whichever
frame of the connection carries it, with an error that names the version, the
range and the side to update:

```
protocol 2 is not served here: this daemon (engine 0.x.y) accepts protocol 1..1; update the daemon
protocol 0 is not served here: this daemon (engine 0.x.y) accepts protocol 1..1; update the client
```

and the connection is closed. A version must be a JSON integer; `true` and
`1.0` are refused, not read as 1. A request that carries no `protocol` field is a
version-1 request: clients written before the field existed never sent it. That
rule holds only while `min` is 1; once `min` rises above 1, a request without
the field is refused like any other out-of-range version.

## 6. Methods

| method | params | result |
|---|---|---|
| `generate` | `prompt`; per family: `steps`, `height`, `width`, `num_frames`, `seed`, `cfg`, `max_tokens`, `temperature`, `repetition_penalty`, `speaker`, `mode`, `image_path`, `audio_path`, `chat_mode`; `stream` (bool); `output_path` (image, video and audio families: where to save) | `family`, `timing.total_s`, the family's output (text families: the text), `op_id`, `status` ∈ `completed`/`cancelled`/`orphaned`/`stopped`; when not `completed`, `steps` (tokens produced) and `detail` if any; `output_path` (the saved file) when `output_path` was given for an image, video or audio family. A params key not in this list is ignored today, without an error |
| `chat` | `message`, sampling params | `text`, `context` (session window), `summarized` (bool) |
| `complete` | `messages` (list of role/content), `tools` (optional), sampling params | `text` |
| `template` | — | `text`: the model's chat template |
| `new_chat` | — | `{"status": "ok"}` |
| `cancel` | `op_id` | `outcome` (below) |
| `status` | — | `model`, `hardware`, `family`, `loaded`, `mode`, `warm_serving`; the identity `neurobrix info --json` prints (`engine`, `protocol`, `protocol_range`, `endpoint`, `operations` = method names) — then `operations` is replaced by the live operations (`op_id`, `method`, `state`, `steps`, `stop_reason`?, `detail`?); `strategy`, `vram_used_gb`, `context` when known |
| `shutdown` | — | `{"status": "ok"}`; the daemon unloads and exits after answering |

### Cancel

`{"protocol": 1, "method": "cancel", "params": {"op_id": "abc"}}` is answered
only after the cancelled decode loop has returned and the devices are
synchronised, so a `cancelled` answer means the work has stopped.

| `result.outcome` | other fields | meaning |
|---|---|---|
| `cancelled` | `op_id`, `state: "cancelled"`, `steps` | stopped by this request |
| `already_cancelled` | same | an earlier cancel stopped it (this answer also waits for the stop) |
| `already_finished` | `op_id`, `state` ∈ `completed`/`orphaned`/`stopped`/`failed`, `steps` | it ended before the cancel |
| `unknown_op` | `op_id` | no such operation |
| `cancel_pending` | `op_id`, `state: "running"`, `steps` | the daemon's `serving.cancel_ack_timeout_s` elapsed before the stop: NOT a confirmation; ask `status` |

A client that disconnects during a generation orphans it: the daemon stops it at
the next token and records it `orphaned`, never `cancelled`. A client-side
timeout changes nothing on the daemon.

## 7. Readiness

A daemon is ready when its model is loaded and its endpoint accepts
connections. It says so once, as one **ready record** (a JSON object), so a
launcher waits for it instead of retrying connections:

```json
{"state": "ready", "pid": 4242, "model": "Qwen/Qwen3-0.6B", "family": "llm", "serving": "warm",
 "engine": "0.x.y", "protocol": 1, "protocol_range": [1, 1],
 "endpoint": {"kind": "unix", "path": "/Users/me/.neurobrix/daemon.sock"}, "operations": [...]}
{"state": "failed", "pid": 4242, "model": "...", "engine": "0.x.y", "protocol": 1, "protocol_range": [1, 1],
 "endpoint": {...}, "operations": [...], "error": "FileNotFoundError: Model '...' not found. ..."}
```

`state` is `ready` or `failed`; a failed daemon exits after saying it. The
identity fields are those of `neurobrix info --json`. The record is delivered:

| to | how | for |
|---|---|---|
| stdout, every run | one line `NBX-READY <record>`, flushed | a launcher that runs `neurobrix serve --foreground` as its child and reads its stdout |
| `--ready-fd N` (macOS, Linux) | the record as one line on fd `N`, which the daemon inherited, then closed | a launcher that passes the write end of a pipe and blocks on the read end: no polling and no deadline, because a daemon that dies closes the pipe (EOF without a record = failed; the exit code says how) |
| `--ready-file PATH` | the record, written whole (written beside, then renamed) | a launcher that can pass no fd. `neurobrix serve` on Windows uses the instance's own file, `<stem>.ready` (named) or `~/.neurobrix/daemon.ready` (default), and looks for it every `serving.ready_poll_s` until it appears or the daemon exits |

`neurobrix serve` without `--foreground` is itself such a launcher: it prints
`[Serve] Model loaded — <model> (<family>, <warm|cold> serving)` or
`[Serve] Failed: <error>` and exits 0 or 1. A record that cannot be delivered
(the launcher is gone) is said in the daemon's log; the daemon serves on.
