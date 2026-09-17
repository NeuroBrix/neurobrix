# `--json` — one record per read command, the contract a client reads

Every read command of the CLI accepts `--json`. It then prints **exactly one
JSON object on stdout and nothing else there**; every human line the same
command would have printed goes to **stderr**. A client — the TUI, NeuroBrix
Studio — reads the record and never parses prose, never guesses a version,
never re-implements an engine decision (Studio requests 1, 5 and 7 in
`studio-engine-requests.md`).

Each record carries two keys before its own:

| key | meaning |
|---|---|
| `schema` | `neurobrix.<command>/<version>` — the version table is `src/neurobrix/cli/json_out.py::SCHEMAS`. A version changes when a key is removed or its meaning changes; adding a key does not change it. A client refuses a schema it does not know. |
| `engine` | the engine's version (`neurobrix.__version__`) |

The gate: `tests/unit/cli/test_read_commands_speak_json.py` runs every read
command with `--json`, parses stdout, and refuses an output that does not
parse or lacks its schema (seen red on an injected stray print).

## The commands and their records (schema version 1)

| command | record |
|---|---|
| `info --json` | `version`, `package`, `cache`, `store`, `python`, `torch`, `cuda_available`, `gpus[{index,name,memory_bytes}]`, `models[{name,size_bytes}]`, `hardware_profiles[]` — machine-readable discovery without loading a model |
| `list --json` | `models[{name,family,size_bytes,license,in_store}]`, `store_only[{name,size_bytes}]`, `store{path,files[{name,size_bytes}],bytes}` |
| `hub --json [--category C] [--search S]` | `registry`, `query{category,search}`, `total`, `models[{slug,name,category,size_bytes,license,downloads,visibility,installed}]` — the registry's own answer, `installed` read from this machine's cache |
| `inspect <path-or-model> --json` | `path`, `model`, `nbx_version`, `family`, `components[{name,category,neural,ops?,shards?}]` |
| `coverage [SYMBOL] [--rarest N] [--unreached] [--field KEY] --json` | `mode` ∈ `index`/`symbol`/`rarest`/`unreached`/`field`/`empty` and the mode's fields: `symbol`+`carried_by[]`, `ops[{op,containers,carried_by[]}]`, `unreached[]`+`claimed`, `declared_by[{container,values[]}]`, `distinct_ops`; always `containers` |
| `doctor --json` | `ok`, `problems[]` — the diagnosis itself stays on stderr |
| `autotune status --json` | `profile`, `directory`, `enabled`, `files`, `shapes`, `served_by_memory_class_gb{"16":n,"32":n}`, `proven_on_unknown_card`, `this_card_class_gb`, `would_be_served_here` |
| `autotune check --json` | `directory`, `files[{path,ok,shapes?,problems?}]`, `refused` |
| `run --model M … --explain-plan --json` | `model`, `strategy`, `loading_mode`, `dtype`, `why`, `candidates[{strategy,score}]`, `refused[{strategy,score,why}]`, `planned_memory_mb`, `cpu_ram_mb`, `components[{name,devices[],dtype,sharded,weight_bytes,activation_bytes,overhead_bytes,peak_op_uid,activation_profiled}]`, `op_level_tiling[]`, `component_tiling{}`, `kv_cache{max_cache_len,memory_bytes,dtype}?` — read from the plan, never recomputed: what is printed is what will run |
| `validate … --json` | the validation results (pre-existing) |

Exit codes are unchanged by `--json`: a `doctor` with problems still exits 1,
an `autotune check` with a refused file still exits 1, a `coverage` on an
empty cache still exits 1 (its record says `mode: empty`).

## Long-running commands: one NDJSON event per phase

`import --json` does not print one record; it prints one compact JSON object
per line as the import advances (NDJSON), each with `schema`
`neurobrix.import/1`, `engine` and `event`:

| event | fields | when |
|---|---|---|
| `info` | `model`, `category`, `bytes`, `license`, `license_name`, `gated` | the registry answered |
| `license` | `license`, `accepted_via` (`--accept-license`, `NBX_ACCEPT_LICENSE=1`, `prompt`) | a gated licence was accepted; under `--json` a licence is an explicit parameter — the command never prompts, it refuses with `error` naming `--accept-license` |
| `download` | `file`, `bytes`, `total` | the real byte count, at most once a second and always at the end of the stream (a resume starts at the bytes it already had) |
| `downloaded` | `file`, `store`, `bytes` | the archive is complete under its final name |
| `extracting` | `store`, `cache` | extraction begins — into `<cache>.installing`, never under the final name |
| `installed` | `model`, `cache`, `already` | the model is visible under its name (`already: true` when it was there and `--force` was not given) |
| `done` | `model`, `cache`, `store` (null under `--no-keep`) | terminal, success |
| `error` | `message` | terminal, exit 1 — every refusal of the command ends in exactly one |

A stream that ends without `done` or `error` was cut. A model is visible to
`list`, `run` and a client only after its extraction succeeded: the staging
directory is renamed in one motion, and a `*.installing` directory is never
listed.

`remove --json` prints one record: `model`, `found`, `removed[{kind ∈ cache/store, path, bytes}]`.

## What is not here yet

`clean` keeps its prompt; the daemon does not refuse a client outside a
declared protocol range (Studio request 7, with the dispatcher work of
requests 3 and 4). Each lands as a row of `studio-engine-requests.md` when it
does.
