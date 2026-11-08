# Serving and chat

`neurobrix run` loads a model, runs one request and exits. `neurobrix serve` starts a **daemon**: a background process
that loads one model once and answers requests over a local socket until it is stopped or stays idle too long.
`neurobrix chat` is an interactive client of that daemon for language models, and `neurobrix stop` ends it.

```bash
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0     # load once; returns when the model is loaded
neurobrix chat                                      # multi-turn chat with the loaded model
neurobrix run --prompt "Hello"                      # answered by the daemon (same model)
neurobrix stop                                      # unload and exit
```

The wire format (framing, every method's fields, error strings, versions, the ready record) is specified in
[the daemon wire protocol](reference/wire-protocol.md). This page describes what the commands do, what "warm" means,
and how the conversation and its KV cache behave today.

---

## Commands

### `neurobrix serve`

```bash
neurobrix serve --model <name> [--hardware <profile>] [--timeout <seconds>] [--foreground]
                [--compiled | --sequential | --triton | --triton-sequential]
                [--ready-fd <fd>] [--ready-file <path>]
```

| flag | meaning | default |
|---|---|---|
| `--model` | the model name (as `neurobrix list` shows it); required | — |
| `--hardware` | hardware profile ID | the default profile of this machine, created at first use |
| `--timeout` | idle timeout in seconds (see [Lifecycle](#lifecycle)) | 1800 |
| `--foreground` | serve in this process and block the terminal | off: the daemon goes to the background |
| `--compiled` | PyTorch engine, fused graph | the default where PyTorch is installed |
| `--sequential` | PyTorch engine, one operation at a time (debugging) | — |
| `--triton` | NeuroBrix Triton engine | the default where PyTorch is not installed |
| `--triton-sequential` | Triton engine, one operation at a time (debugging) | — |
| `--ready-fd` | POSIX: an inherited pipe fd; the ready record (one JSON line) is written there once the daemon accepts, then the fd is closed | — |
| `--ready-file` | a file the ready record is written to, whole, once the daemon accepts | — |

At most one engine flag may be given; two or more are refused. When none is given, `serve` prints which engine it chose.
Choosing the PyTorch engine where PyTorch is not installed is refused before the daemon starts.

`serve` refuses to start (exit code 1) when a daemon of the same instance is already running.

Without `--foreground`, `serve` starts the daemon in the background, prints its PID and the path of its log file, waits
until the model is loaded and the socket accepts, then prints one of:

```
[Serve] Model loaded — <model> (<family>, warm serving)
[Serve] Model loaded — <model> (<family>, cold serving)
[Serve] Failed: <error>
```

and exits 0 (loaded) or 1 (failed). `warm` and `cold` are explained in [Warm and cold serving](#warm-and-cold-serving).
On Linux the background daemon is a fork of the `serve` process; on macOS and Windows it is a new process running
`neurobrix serve ... --foreground` (with the same model, the resolved hardware profile, engine and timeout).

`--ready-fd` and `--ready-file` are for programs that launch the daemon themselves; `serve` also prints the ready record
on stdout as one line `NBX-READY <json>`. See [Readiness](reference/wire-protocol.md#7-readiness).

### `neurobrix chat`

```bash
neurobrix chat [--max-tokens N] [--temperature T] [--repetition-penalty P]
```

| flag | meaning | default |
|---|---|---|
| `--max-tokens` | maximum tokens per answer | the model's default |
| `--temperature` | sampling temperature | the model's default |
| `--repetition-penalty` | repetition penalty (1.0 = none) | the model's default |

`chat` needs a running daemon (otherwise it prints how to start one and exits 1) whose model is of the `llm` family
(otherwise it tells you to use `neurobrix run` and exits 1). It then reads lines from the terminal; each line is one
user message, and the full answer is printed when it is complete (chat answers are not streamed).

| command inside chat | effect |
|---|---|
| `/new` | start a new conversation: the daemon's message history is cleared |
| `/context` | number of messages, tokens, percentage of the ceiling and tokens held by the cache (see [The `/context` figure](#the-context-figure)) |
| `/status` | model, family, engine mode and the device memory in use |
| `/quit`, `/exit`, `/q` | leave chat (the daemon keeps running) |

End-of-file or Ctrl-C also leaves chat. A turn past the session's ceiling is refused by name (see
[When the conversation outgrows the cache](#when-the-conversation-outgrows-the-cache)).

`chat --agent` starts the agent loop over the same daemon instead (each input is a task the model completes with tools
inside a working-directory jail). Its flags are `--workdir`, `--max-turns`, `--bash-timeout`, `--allow-network`,
`--approve-all`, `--yolo` and `--transcript-dir`; see the [CLI reference](reference/cli.md#chat).

### `neurobrix stop`

```bash
neurobrix stop
```

No flags. When no daemon is running it says so, removes the instance's leftover files (socket, pid, port file) and
exits 0. Otherwise it escalates:

1. a `shutdown` request on the socket (3 s socket timeout), then up to 5 s for the process to exit;
2. a termination signal (`SIGTERM`), then up to 3 s;
3. a forced kill (`SIGKILL`; `taskkill /F` on Windows).

It then removes the instance's files. If the process is still alive it prints the command to kill it by hand and
exits 1.

### `neurobrix run` while a daemon is running

- When `--model` is omitted, `run` takes the model of the running daemon.
- When the daemon serves the same model, `run` sends its request to the daemon (`[Run] Using warm daemon (PID ...)`)
  with its sampling, size, step, seed, mode, speaker and audio flags. Image, video and audio results are saved by the
  daemon at the `--output` path; text is printed on stdout and not written to `--output`.
- When the daemon serves another model, or the request to it fails, `run` refuses to load a second copy beside it and
  tells you to use the daemon or stop it.
- `run --certified-only` is never answered by a daemon: it is refused while one is running.

### Several daemons

One daemon serves one model. The environment variable `NBX_SOCKET_PATH` names an **instance**: set it to the same path
for the `serve` and for every client command of that daemon, and two daemons with different paths run side by side.
Unset, the default instance is used. See [Where the daemon listens](reference/wire-protocol.md#1-where-the-daemon-listens).

---

## Lifecycle

1. **Load.** The daemon writes its pid file, then: finds the model, reads its container, loads the hardware profile,
   solves the Prism plan once **for serving** (see below), applies the profile's CPU settings, loads the runtime package
   and runs the executor's one-time setup. For a warm plan of the `llm` family it runs one one-token generation
   (`"warmup"`) so that the weights are on the device before the first request. It prints
   `<model> | <profile> | <strategy> | Ready in <s>s`.
2. **Listen.** It binds its socket, says it is ready, and accepts connections. Each connection gets its own thread.
3. **Serve.** A connection may send any number of requests, one after another.
4. **Exit.** On a `shutdown` request, on `SIGTERM`/`SIGINT`, or on the idle timeout. Before exiting it stops every
   running operation, waits up to `serving.drain_timeout_s` for their replies to be delivered, closes the connections and
   removes its socket, pid and port files. On a `shutdown` request it also unloads the weights.

**Idle timeout.** The accept loop waits for a new connection for `--timeout` seconds. When that wait elapses and no
connection is open, the daemon exits; while any connection is open (a `chat` session, a running request) it does not.

**Load failure.** The daemon reports `failed` with the error, and exits; the launching `serve` prints the error and
exits 1. The log file keeps the daemon's output.

### Files

All in the engine home (`$NEUROBRIX_HOME`, else the `home` of `~/.neurobrix/paths.json`, else `~/.neurobrix`), for the
default instance:

| file | content |
|---|---|
| `daemon.sock` | the Unix socket (macOS, Linux) |
| `daemon.pid` | the daemon's PID |
| `daemon.log` | the daemon's stdout and stderr |
| `daemon.ready` | Windows only: the ready record while the launcher waits for it |

A named instance (`NBX_SOCKET_PATH=<path>`) uses `<path>` as its socket and `<stem>.pid`, `<stem>.log`,
`<stem>.ready` (and on Windows `<stem>.port`) beside it.

---

## Warm and cold serving

When the daemon loads, Prism solves the plan with `serve_mode=True`. Each plan carries a **loading mode**:

- **eager**: the daemon marks every graph executor *persistent*: weights are not released at the end of a request and
  stay on the device between requests, with the executor's compiled state. This is **warm** serving.
- **lazy**: weights are loaded when the flow needs them and released as it goes, as in a single `run`; the executor's
  setup still survives between requests, the weights do not. This is **cold** serving: every request pays the loading
  again. The daemon prints `Cold serving — weights reload per request, setup/context persists`.

In serve mode Prism prices `single_gpu` with **every weight resident** at once (the sum of all weights plus the
largest activation), not one component at a time as for a single `run`. Prism ranks the
strategies by what they do with the weights and takes the fastest plan that fits within the best rank (lower rank
first):

| strategy | rank | loading mode when serving | daemon |
|---|---|---|---|
| `single_gpu` | 1 | eager | warm |
| `single_gpu_lifecycle` | 1 | eager | warm |
| `component_placement` | 2 | eager | warm |
| `block_scatter` | 2 | eager | warm |
| `pipeline_parallel` | 2 | eager | warm |
| `weight_sharding` | 2 | eager | warm |
| `op_level_tiling` | 3 | eager when the plan's memory is at most 90% of the summed capacity of the plan's CUDA devices, otherwise lazy (so lazy on a device that is not CUDA) | warm or cold |
| `zero3` | 4 | lazy | cold |
| `component_placement_lazy` | 4 | lazy | cold |
| `lazy_sequential` | 4 | lazy | cold |
| `layer_streaming` | 4 | lazy (always) | cold |
| `cpu_execution` | 6 | lazy | cold |
| `cpu_streaming` | 6 | lazy (always) | cold |

**The cold fallback.** When no strategy fits under the serve-mode budgets, Prism says so in a warning and solves
again with the budgets of a single run. The plan it finds then is **lazy whatever its strategy**, so the daemon serves cold.

The result is visible in three places: the `serve` line (`warm serving` / `cold serving`), the ready record's
`"serving"` field, and the `status` method's `warm_serving` and `strategy` fields.

The plan is solved once, at load. On a machine busy at that moment, Prism re-plans on the memory really free (as it does
for `run`); it is not re-planned between requests.

---

## The socket and its operations

| platform | transport |
|---|---|
| macOS, Linux | Unix domain socket, `daemon.sock` in the engine home (or `NBX_SOCKET_PATH`) |
| Windows | TCP on `127.0.0.1` only: a fixed port for the default instance, a port chosen by the OS (written to `<stem>.port`) for a named instance |

Nothing listens on a network interface; there is no HTTP layer. Each message is a 4-byte big-endian length followed by
one UTF-8 JSON object of at most 64 MiB. Every request carries `protocol` (today `1`), `method` and `params`.

| method | kind | what it does |
|---|---|---|
| `generate` | work | one request of any family, as `neurobrix run` would make it; `"stream": true` sends one event per produced token before the reply |
| `chat` | work | `llm` family only: adds `message` to the daemon's conversation and answers it (see [Chat and the KV cache](#chat-and-the-kv-cache-today)) |
| `complete` | work | families that declare the `agent` mode: one answer over a message list the caller owns (roles, tool results, optional `tools`) |
| `template` | control | the model's chat template text (`""` when it has none) |
| `new_chat` | control | clears the daemon's conversation |
| `status` | control | model, profile, family, engine mode, `warm_serving`, `strategy`, device memory in use, the conversation's `context`, the live operations, the engine and protocol versions |
| `cancel` | control | stops one operation by `op_id` (below) |
| `shutdown` | control | stops running work, unloads, exits |

`chat` and `complete` read `temperature`, `repetition_penalty`, `top_k`, `top_p`, `max_tokens` and `seed`; a parameter
the engine does not read is refused by name before anything runs. The full field lists are in
[Methods](reference/wire-protocol.md#6-methods).

**One inference at a time.** `generate`, `chat`, `complete` and `new_chat` take one lock: a work request that arrives
while another runs waits (its state is `queued`). `status`, `cancel`, `template` and `shutdown` are answered while a
generation runs, from any other connection.

**Operations.** Every work request is an operation with an id: the client's (`op_id` in the request) or one the daemon
assigns (`<pid>-<n>`). The reply, its errors and its stream events carry it. Its state is `queued`, `running`, then one
of `completed`, `cancelled`, `orphaned` (its client went away), `stopped` (daemon shutdown) or `failed`. The daemon
remembers the last `serving.operations_retained` finished operations.

A Python program can use the client the CLI uses, `neurobrix.serving.client.DaemonClient` (`connect`, `generate`,
`generate_stream`, `chat`, `complete`, `template`, `new_chat`, `status`, `cancel`, `shutdown`, `close`).

---

## Cancellation

`cancel` with `params.op_id`, sent on a connection other than the one running the work, sets that operation's stop
flag. The autoregressive decode loops of both engines (the text generators, and the vision-language, audio-language,
encoder-decoder and text-to-speech decode loops) read the flag once per token, after the token is recorded, and end as if
their length bound were reached: the reply carries what was produced so far, with `status: "cancelled"` and `steps`.

The `cancel` answer waits until the decode loop has returned **and** every device of the plan has been synchronised, so
`cancelled` means no kernel of that generation is still running. Its `outcome` is one of:

| outcome | meaning |
|---|---|
| `cancelled` | this request stopped it |
| `already_cancelled` | an earlier cancel stopped it |
| `already_finished` | it had ended (`state` says how: `completed`, `orphaned`, `stopped`, `failed`) |
| `unknown_op` | no remembered operation has this id |
| `cancel_pending` | the stop was not honoured within `serving.cancel_ack_timeout_s`; not a confirmation |

A flow without a per-token decode loop (image or video denoising, for example) does not read the flag: the work runs
to its end and the cancel then answers `already_finished`, or `cancel_pending` if the timeout comes first.

**A client that disconnects** during a work request (its socket closes, or a stream event cannot be sent) orphans the
operation: the daemon stops it at the next token through the same flag and records it `orphaned`, never `cancelled`.
Pressing Ctrl-C in `neurobrix chat` while an answer is generated closes the client, so the daemon stops that answer
this way.

**Shutdown** stops every live operation (`stopped`), waits up to `serving.cancel_ack_timeout_s` for it, then unloads.

### Settings

The dispatcher's limits are read from the `serving:` section of `config/system.yml` (each is required; a missing or
non-positive value is refused by name):

| key | shipped value | meaning |
|---|---|---|
| `cancel_ack_timeout_s` | 120 | how long a `cancel` (and a `shutdown`) waits for the stop |
| `operations_retained` | 256 | finished operations remembered for a late `cancel` |
| `listen_backlog` | 16 | pending connections the socket queues |
| `orphan_poll_s` | 0.5 | how often a non-streamed request's connection is checked for a vanished client |
| `drain_timeout_s` | 30 | how long an exiting daemon waits for replies to be delivered |
| `ready_poll_s` | 0.5 | Windows: how often the launcher looks for the ready file |

---

## Chat and the KV cache today

This is the state after 0.6.2 (unreleased): the KV cache belongs to the chat session (scope A of
`docs/design/prism_kv.md`). Host and disk tiers and sharing between sessions are not built.

### One conversation per daemon

The daemon holds **one** conversation (a list of user and assistant messages), created at the first `chat` request.
Every client's `chat` reads and extends that same history; `new_chat` (or `/new`) clears it. The `chat` method takes no
system message. `generate` does not touch it.

### A turn prefills only its new tokens

The session keeps its KV cache between turns. For each `chat` request the daemon:

1. appends the user message to the history;
2. renders the **whole** history with the model's chat template (with the generation prompt) and tokenizes it; a
   tokenizer without a chat template is refused;
3. checks that the conversation and the turn's `max_tokens` fit the session's ceiling (below);
4. runs one generation from those token IDs. The cache already holds the token IDs of the previous turns (the
   prompts and the answers the decode fed back); the new IDs are matched against them, and only the positions after
   the longest common prefix are prefilled. Whatever the template renders differently in an earlier turn only
   shortens the match;
5. appends the answer to the history.

Turn *n* therefore costs a prefill of its own new tokens, not of the conversation. A prefill that continues a cache
attends over the cached keys read back as whole cache pages, the keys past the true end masked, so the bytes do not
depend on where a turn ended. With a planned prefill chunk (`KVCachePlan.prefill_chunk`, not chosen by Prism yet) a
turn served from the session gives the same bytes as the same turns prefilled together in the same chunks, on both
engines. `complete` keeps one cache record for the agent path the same way: a fixed system prompt and tools contract
are prefilled once.

Any other request on the same engine (a `generate`, a warmup) that reuses the cache empties it, and the next turn
then prefills the whole conversation again. A failed turn leaves the session holding nothing.

On the Triton engine under `--certified-only`, a continued prefill needs the kernel settings of attention with fewer
query rows than keys; the census does not derive them yet, so such a run stops naming the missing setting.

### The cache Prism plans for serving

Prism sizes the decoder's KV cache in the plan (`KVCachePlan`), from the KV state the container declares (layers,
key-value heads, key and value head dimensions, window):

- **cost per position**: layers × key-value heads × (key dim + value dim) × the width of the compute dtype;
- **ceiling** (`max_cache_len`): when serving, as many positions as the memory left by the plan affords, capped by the
  decoder's declared window. A container whose KV state declares no window cannot be served (refused by name);
- **floor**: when serving, `2 × max_tokens + prompt_margin` positions (`prompt_margin` from the container's defaults,
  else 128; 64 positions when the container declares no `max_tokens`). A strategy whose remaining memory cannot hold
  the floor is rejected ("KV cache does not fit") and the next one is tried;
- **initial size** (`initial_cache_len`): when serving, `max_tokens + prompt_margin` positions (at most the ceiling).

For comparison, a single `run` caps the cache at that request's `max_tokens + prompt_margin` instead of the window.

### When the conversation outgrows the cache

The session's ceiling is the plan's: its `session_reach` (the positions a serving session holds in the room the cache
was sized from, within the declared window), or its `max_cache_len` when the decoder declares no window. A turn whose
rendered conversation plus `max_tokens` (the request's, else the container's) passes it is **refused by name**, with
the ceiling and the plan fact it comes from; the conversation is left as it was. Nothing is summarised or dropped:
start a new conversation (`/new`). The reply's `summarized` field stays in the protocol and is always `false`.

### The `/context` figure

`/context` (the reply's and `status`'s `context` record) shows:

| field | content |
|---|---|
| `turns` | the number of **messages** in the history |
| `tokens` | the conversation's token count, rendered by the chat template |
| `max_context` | the session's ceiling (above) |
| `usage_pct` | `tokens / max_context`, in percent |
| `cached_tokens` | the positions the session's KV cache holds |
| `last_turn` | the last turn's `prompt`, `matched`, `prefilled`, `padded` and `held` token counts |

---

## Examples

A warm language model, a few turns, then stop:

```bash
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --timeout 3600
neurobrix chat --max-tokens 256 --temperature 0.7
#   You: What is a KV cache?
#   You: /context
#   You: /new
#   You: /quit
neurobrix stop
```

The Triton engine and an explicit hardware profile:

```bash
neurobrix serve --model TinyLlama-1.1B-Chat-v1.0 --hardware v100-32g --triton
```

Two daemons side by side (one model each):

```bash
NBX_SOCKET_PATH=/path/to/llm.sock   neurobrix serve --model TinyLlama-1.1B-Chat-v1.0
NBX_SOCKET_PATH=/path/to/image.sock neurobrix serve --model PixArt-Sigma-XL-2-1024-MS
NBX_SOCKET_PATH=/path/to/llm.sock   neurobrix chat
NBX_SOCKET_PATH=/path/to/image.sock neurobrix run --prompt "a lighthouse at dusk" --output lighthouse.png
NBX_SOCKET_PATH=/path/to/llm.sock   neurobrix stop
NBX_SOCKET_PATH=/path/to/image.sock neurobrix stop
```

From Python, a streamed generation cancelled from a second connection:

```python
import threading
from neurobrix.serving.client import DaemonClient

work = DaemonClient(); work.connect()
control = DaemonClient(); control.connect()

def stop_soon():
    print(control.cancel("my-op"))           # {'outcome': 'cancelled', 'op_id': 'my-op', ...}

for kind, event in work.generate_stream("Tell me a long story.", op_id="my-op", max_tokens=512):
    if kind == "token":
        print(event.get("text", ""), end="", flush=True)
        if event["step"] == 20:
            threading.Thread(target=stop_soon).start()
    else:
        print("\n", event["status"], event.get("steps"))   # "cancelled" and the tokens produced
work.close(); control.close()
```
