# The daemon's wire protocol

What a client needs to talk to a NeuroBrix serving daemon without reading the engine's code:
where it listens, how a message is framed, what an envelope carries, the version and the range the
daemon serves, every method with its inputs and outputs, the shape of an error, how readiness is
signalled, and the hard limits. The engine's own client (`serving/client.py`) and the CLI (`chat`,
`stop`) speak exactly this; a second client (NeuroBrix Studio) needs nothing more.

Version 1 of the protocol, engine 0.x, written 2026-10-13. Every number here is read from the
engine's sources named beside it; a change to any of them is a change to this page.

## 1. Transport and endpoints

Same-machine IPC only: no HTTP, no REST, no gRPC. One daemon instance serves one model.

| platform | transport | default instance | named instance (`NBX_SOCKET_PATH=<stem>`) |
|---|---|---|---|
| Linux, macOS | Unix domain socket (`AF_UNIX`, stream) | `~/.neurobrix/daemon.sock` | the socket at `<stem>`; pid file `<stem>.pid`, log `<stem>.log` |
| Windows | TCP on loopback (`AF_INET`, stream) | `127.0.0.1:19384` (fixed) | a port the OS hands out at bind, written to `<stem>.port`; pid `<stem>.pid`, log `<stem>.log` |

The default instance keeps `~/.neurobrix/daemon.pid` and `~/.neurobrix/daemon.log`. A client of a
named Windows instance whose `.port` file is absent has no daemon: it is refused by name, never
given a guessed port. Two instances never share a socket or a port (`serving/protocol.py::Instance`).

The endpoint as a record, the same from `neurobrix info --json` and from the daemon's `status`:
`{"kind": "unix", "path": "..."}` or `{"kind": "tcp", "address": "127.0.0.1", "port": N}` (a named
Windows instance adds `"port_file"`).

## 2. Framing

One message = one frame = `[4 bytes: uint32, big-endian, the payload length][payload]`, the
payload being one UTF-8 JSON **object**. Both directions, every message, including stream events.

| limit | value | source |
|---|---|---|
| header | 4 bytes | `serving/protocol.py::HEADER_SIZE` |
| maximum payload | 64 MiB (67 108 864 bytes) | `serving/protocol.py::MAX_MESSAGE_SIZE` |

A frame that is not a protocol message is answered with an error naming what was wrong, and then:

| the frame | answer | the connection |
|---|---|---|
| length 0 | `malformed frame: Empty frame: a message is one JSON object` | kept: the next frame is read |
| payload not UTF-8 JSON, or JSON that is not an object | `malformed frame: ...` | kept |
| length above the maximum | `malformed frame: Message too large: N bytes (max ...)` | **closed** after the answer: no frame boundary is left to trust |
| the peer closes mid-frame | nothing | closed |

## 3. Envelopes

A **request**:

```json
{"protocol": 1, "method": "generate", "params": {"prompt": "..."}, "op_id": "mine-7"}
```

* `protocol` — the client's wire version (section 4). The engine's client stamps it on every
  request. A request without it is a client from before the handshake and is served as protocol 1.
* `method` — a string from section 5. `params` — an object (absent or `null` = empty). Any other
  type is answered `malformed request: ...`.
* `op_id` — optional, only meaningful on a work method (section 6): the client's own name for the
  operation, so it can be cancelled from another connection before the reply; absent, the daemon
  assigns one.

A **response**, always exactly one per request, on the same connection, in order:

```json
{"protocol": 1, "engine": "0.x", "result": {...}}
{"protocol": 1, "engine": "0.x", "error": "..."}
```

`protocol` is the daemon's wire version, `engine` its package version; both on every envelope,
errors included. A reply to a work request also carries `op_id`.

A **stream event** (only during a `generate` with `"stream": true`, before its response):

```json
{"stream": {"step": 3, "n": 4, "token": 1234, "done": false, "op_id": "mine-7", "text": " Par", "rewind": 0}}
```

Requests on one connection are served in order. A control request (section 5) sent on a second
connection is answered while a generation runs on the first.

## 4. Version and range

`PROTOCOL_VERSION = 1`, `PROTOCOL_MIN = 1` (`serving/protocol.py`). The version changes when a
method's request or response shape changes; adding a method does not change it.

A request whose `protocol` is outside `[PROTOCOL_MIN, PROTOCOL_VERSION]`, or is not an integer, is
refused **at its first frame**: the daemon answers an error naming the client's number, the range it
serves and the engine version, then closes the connection:

```
protocol 7 is outside the range; this daemon serves protocols 1..1 (engine 0.x)
protocol '1' is not an integer wire version; this daemon serves protocols 1..1 (engine 0.x)
```

A client learns the range before connecting from `neurobrix info --json` (`protocol`) or from the
daemon's `status`.

## 5. Methods

Two kinds. **Work** methods run under the daemon's single inference lock, one at a time, and are
operations (section 6). **Control** methods never wait on a generation.

| method | kind | params | result |
|---|---|---|---|
| `status` | control | — | the engine's status: `model`, `hardware`, `family`, `loaded`, `mode`, `warm_serving`, `strategy` (when planned), `vram_used_gb` (when loaded on a device), `context` (chat session, when any), plus the identity (`engine`, `protocol`, `endpoint`, `operations`: the method names served) and `operations` as the list of LIVE operation records (section 6) |
| `generate` | work | `prompt`; and, by family: `max_tokens`, `temperature`, `repetition_penalty`, `steps`, `height`, `width`, `cfg`, `num_frames`, `mode`, `speaker`, `chat_mode`, `image_path`, `audio_path`; `output_path` (where a binary output is saved); `stream` (`true` = per-token events) | the family's result record — a text family: `text`, `tokens`; a binary family: `output_path` (the saved file) — plus `op_id`, `status` (section 6) and, when not `completed`, `steps` and `detail` |
| `chat` | work | `message`, `max_tokens`, `temperature`, `repetition_penalty` | `text`, `context` (the session's context record), `summarized` (whether old turns were summarised this turn), `op_id`, `status` |
| `complete` | work | `messages` (caller-managed list), `tools` (optional), the sampling parameters of `chat` | `text`, `op_id`, `status` |
| `template` | control | — | `text`: the loaded model's chat template |
| `new_chat` | control | — | `{"status": "ok"}`; the chat session is reset (under the inference lock) |
| `cancel` | control | `op_id` (non-empty string) | `outcome`, `op_id`, and `state`, `steps` of the operation (section 6) |
| `shutdown` | control | — | `{"status": "ok"}` once every running operation has stopped and the model is unloaded; the listener closes after this reply |

Unknown method: `Unknown method: <name>`. A parameter the family does not read is ignored by the
engine's input resolution; one it refuses is an error with the engine's own message.

## 6. Operations and their states

Every work request is an operation with an `op_id` — the client's (a non-empty string that names no
live operation, else `refused: ...`) or the daemon's (`op-<n>`, counted from 1 per daemon).
The table of operations is the one shared state between connections; control reads it without
touching the inference lock (`serving/operations.py`).

States: `queued` (waiting for the inference lock) → `running` → one terminal state:

| state | meaning |
|---|---|
| `completed` | ended on its own (end of sequence, or its length bound) |
| `cancelled` | a client's `cancel` stopped it: the decode loop read the stop flag at its token seam, returned, and the devices were synced |
| `orphaned` | its client vanished (connection closed or reset); the daemon stopped it — never reported as `cancelled` |
| `stopped` | a daemon `shutdown` stopped it |
| `failed` | the engine raised (the work reply is then an error carrying `op_id`) |

A terminal state is set exactly once, after the native work has stopped and the device has synced.
A disconnect or a timeout is never an acknowledgement.

An operation record (in `status`, and the basis of every work reply): `op_id`, `method`, `state`,
`steps` (tokens produced so far), and when set `stop_reason` (`cancel`, `orphan`, `shutdown`) and
`detail`.

`cancel` outcomes — every one named, none a silent ok:

| `outcome` | meaning |
|---|---|
| `cancelled` | THIS request stopped it; its loop returned and the devices were synced before this reply |
| `already_cancelled` | an earlier cancel stopped (or is stopping) it |
| `already_finished` | it ended on its own, or was orphaned / stopped / failed — `state` says which |
| `unknown_op` | no operation has, or still remembers, this id (the last `serving.operations_retained` finished ones are remembered) |
| `cancel_pending` | the stop is set but the loop did not return within `serving.cancel_ack_timeout_s` — NOT a confirmation |

Only `cancelled` confirms that the request stopped the work. Send `cancel` on a connection other
than the one whose work reply is awaited.

## 7. Streaming

`generate` with `"stream": true` sends one event per decoded token before the final response:
`step` (the decode step), `n` (tokens so far), `token` (the id), `done` (the last event), `op_id`.
For a family whose answer is text, the engine's own tokenizer adds `text` (the delta to append) and
`rewind` (characters to take back first, when a later token rewrites text already emitted); a split
multi-byte character is held until complete. The final response follows the last event. The events
are sent from the decode loop itself, so the first event's wall-clock is the real time to first
token.

## 8. Errors

Every error is the envelope `{"protocol", "engine", "error": "<message>"}` (with `op_id` on a work
request). The message starts with what it is:

| prefix | when |
|---|---|
| `malformed frame: ...` | section 2 |
| `malformed request: ...` | `method` not a string or `params` not an object |
| `protocol N is outside the range; ...` / `protocol X is not an integer wire version; ...` | section 4; the connection is closed |
| `refused: ...` | an `op_id` that is empty or names a live operation |
| `Unknown method: ...` | section 5 |
| `cancel needs params.op_id ...` | a `cancel` without a usable id |
| `shutdown: the running generation did not stop within serving.cancel_ack_timeout_s; ...` | the daemon exits without unloading |
| anything else | the engine's own exception text for that request |

There is no error code: the message is the contract, the prefix is stable.

## 9. Lifecycle, readiness and liveness

`neurobrix serve --model <name> [--hardware <profile>] [--timeout <s>] [--foreground] [--json]`
loads the model, binds the endpoint, writes the pid file and serves until `shutdown`, a signal
(SIGINT, SIGTERM) or the idle timeout (default 1800 s with no request).

Readiness, a signal in every form — never a poll:

| form | signal |
|---|---|
| background, Linux | the launching process blocks on a pipe until the daemon writes `ready:...` or `error:...`, then prints it and exits 0 / 1 |
| background, Windows and macOS | the daemon writes `ready:...` / `error:...` to `~/.neurobrix/daemon.ready`; the launcher waits for it |
| foreground, any platform, `--json` | one NDJSON line on stdout: `{"schema": "neurobrix.serve/1", "engine": "...", "event": "ready", "endpoint": {...}, "protocol": 1, "model": "...", "family": "...", "strategy": "warm" or "cold", "pid": N}` — or `{"event": "error", "message": "..."}` as the terminal event; every human line goes to stderr |
| foreground, no `--json` | the human lines on stdout (`[Daemon] Listening on ...`) |

Liveness: a daemon is running when its pid file names a live process and (on Unix) its socket file
exists (`serving/client.py::DaemonClient.is_running`). `neurobrix stop` sends `shutdown`, then
escalates to SIGTERM and a kill if the daemon does not exit, and removes the instance's files.

## 10. Settings and limits

| setting (`config/system.yml`, `serving:`) | value | what it bounds |
|---|---|---|
| `cancel_ack_timeout_s` | 120 | how long `cancel` and `shutdown` wait for the decode loop to read the stop and return before answering `cancel_pending` / the shutdown error |
| `operations_retained` | 256 | finished operations remembered for a late `cancel` |
| `listen_backlog` | 16 | pending connections the listening socket queues while one is handed to its thread |
| `orphan_poll_s` | 0.5 | how often a non-streaming client is checked for having vanished |

Hard limits: one message ≤ 64 MiB; one generation at a time (the inference lock); no preemption
inside a decode step (a step is the unit); no change to framing between versions of this page
without a protocol version change.

## 11. Where the pieces live

`serving/protocol.py` (framing, envelopes, version, range, endpoints), `serving/server.py` (the
accept loop, one thread per connection, dispatch, work and control), `serving/operations.py` (the
operation table), `serving/text_stream.py` (text deltas), `serving/client.py` (the reference
client), `cli/json_out.py` (the NDJSON event form). The gates: `tests/unit/serving/`.
