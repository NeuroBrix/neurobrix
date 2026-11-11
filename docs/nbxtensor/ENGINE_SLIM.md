# The engine slims: what it carries that nbxtensor makes unnecessary

Hocine Benkelaya's decision (2026-10-29 22:39, precised 22:45): the engine imports nbxtensor and keeps only its own job — no
duplication, no dead function, no adaptation layer, no over-engineering — and its size can be measured. nbxtensor
replaces torch for the whole project: tensor work the engine still carries (an operation, a shape or stride
computation, a conversion, a copy, a reduction, a memory placement, a launch utility, a device probe) moves INTO the
library as a generic API, and the engine calls it. The mechanism goes to the library; the policy stays in the engine
(rule 3); a piece that mixes the two is cut in two.

## The figures

Two figures, kept apart (the supervisor, 2026-10-29 23:40):

- **Third-party code taken out of the engine**: the vendored reference tree `kernels/triton_kernels_ref/`, 262 132
  lines, never imported and never packaged — a census of Triton kernels written elsewhere, which inspired the engine's
  own; the engine now holds every kernel it needs, and the sources are on their own repositories (Hocine Benkelaya, 2026-10-30
  00:46: deleted outright).
- **The engine's own cleaning** — the figure that says whether the engine slims, tracked from here: the Python lines
  of `src/neurobrix` and the open findings, both recorded in the header of `engine_slim_list.tsv` at every commit.
  At the ratchet's first commit (2026-10-29, on nbxtensor 6c059035): 146 646 lines, 842 open findings.
  On 2026-11-02 (nbxtensor ff5c55a3): 137 016 lines, 48 open findings — every one in a zone held elsewhere: core/strategies
  (29), core/prism (13), triton/weight_loader and triton/sequence (the Dell's lots in flight), and the sampling contract
  (`audio_utils.sample_token(top_p)`, assigned to the Dell on 2026-11-02 06:40: one sampling path for both engines).

## Lot A, the session-owned KV cache (2026-11-08): what moved, what stays as policy

The owner's rule of 2026-11-08 16:43: every generic mechanism lives in nbxtensor; the engine keeps policy only (Prism
decides, the flow decides when, the CLI displays); the findings never rise (169 before the lot, 165 after it). Lot A
moved into nbxtensor 0.0.8: the paged KV cache (`memory.paged_kv`, from triton/paged_kv_cache.py, -377), the page
bookkeeping (`memory.pages`, from core/state/pages.py, -181), the session record and the chunk row plan
(`memory.kv_session`), the whole-page gather kernel and the chunk's masked attention (`memory.paged_kv.chunk_attention`,
`ops.attention_wrappers.paged_kv_gather`). The serving session lost its summarisation (-79, -16, -4, -1). What stays,
one row per file, is POLICY, and the header rose by exactly these lines, measured: +778 kept - 658 moved or deleted
+ 46 of the gate's three fixes (a reloaded component's fp32 constants, a lazy plan's release, a recompile keeping the
first compile's precision contract) = +166, 137 968 (main 6d72eef5) to 138 134.

| file | lines | the policy it holds |
|---|---|---|
| core/flow/autoregressive.py | +118 | the compiled flow decides when a request is a session's turn, from which cached position it prefills and in which chunks, and what the session keeps at the end |
| triton/session.py | +85 | the Triton session's same decisions: prefill from the matched position, run each planned chunk, keep the real rows |
| triton/flow/autoregressive.py | +43 | the Triton flow's decisions: the session's cache kept or cleared, the chunk only for a session's turn, the ledger; `retire_replay_plans`, the cache's `on_reallocate` hook into the engine's own replay plans |
| core/runtime/graph/kv_cache_wrapper.py | +83 | the compiled engine's torch attention over its cache (the whole-page path and its mask): torch, which nbxtensor never holds (R33) |
| core/runtime/graph/paged_kv_cache.py | +32 | the compiled engine's torch cache: truncate and the whole-page read, on torch tensors (R33) |
| triton/kv_cache.py | +60 | the interceptor's routing: which prefill takes the whole-page path (`begin_chunk`), the chunk's positions for the positional ops, the request's end |
| core/state/kv_session.py | +17 | `session_turn`: the flow decides that a request is a session's turn |
| core/prism/prefill_chunk.py | +99 | Prism decides the session's prefill chunk from the certified times of its candidates |
| core/prism/solver.py | +85 | Prism states the chunk and its reason in the plan, its record and `--explain-plan` |
| core/state/price.py | +17 | the candidates Prism and the census share (whole pages up to the prefill a session is priced at) |
| kernels/census_table.py | +42 | the chunk record Prism reads (`read_chunks`) and the container hash it matches it with; the table's lock taken on its directory, no sidecar file left in the package |
| core/runtime/graph_executor.py | +32 | the executor keeps what its first compile decided: the constants it bound in fp32, bound again on a reloaded component's first run, and the precision contract the first compile resolved, served to a compile after a release (the sequence rewrites its graph in place) |
| core/flow/autoregressive.py, triton/flow/autoregressive.py (release) | +6, +8 | a non-persistent run releases its language model at each request's end through the strategy's own unload, so the next request loads it again rather than replaying into freed memory |
| core/runtime/executor.py, core/flow/base.py | +6, +4 | the request carries its session to the flow |
| cli/commands/run.py, cli/__init__.py, cli/commands/registry.py, core/runtime/output_dispatch.py | +80, +4, +2, +1 | the CLI displays a session of turns (`run --turns`) and the family's "one of" requirement |

## The OpenAI lot (2026-11-09): what it adds, all policy

The lot of release 0.6.3 adds serving policy and naming, no tensor work: an OpenAI-compatible HTTP door on the same
daemon, the one inference thread, the reading of an assistant turn, the one name a model is asked by. Serving is
engine policy (the owner's rule of 2026-11-08 16:43); none of it computes on a tensor, so nothing of it belongs in
nbxtensor. The scan's findings stay at zero new (the 16 it first raised were removed in code: dead methods deleted or
named by assignment for http.server, one-line error constructors and waits inlined, the two exceptions given a body,
a never-passed parameter dropped). The header rose by exactly these lines, measured: 138 134 (main e7260834) to
140 058, +1 924.

| file | lines | the policy it holds |
|---|---|---|
| serving/openai_api/routes.py | +628 | which OpenAI endpoint the loaded family serves, read from its YAML; what each request maps to (`complete_turn`, `generate`); the parameters read and the ones refused by name; the OpenAI response shapes |
| serving/openai_api/door.py | +388 | the HTTP door: bind, the API key, body and multipart parsing, the SSE stream, a vanished client's operation stopped, the spool of uploads |
| serving/openai_api/turn.py | +237 | the turn read as it streams: what is answer, reasoning or a tool call, released only once stable (the template's markers) |
| serving/openai_api/errors.py, serving/openai_api/__init__.py | +44, +5 | the OpenAI error object and the named refusals |
| serving/worker.py | +114 | the daemon's one inference thread: every engine call of both doors, in order (Metal keeps per-thread state) |
| serving/server.py | +90 | the dispatcher on that thread for both doors; the HTTP door opened beside the socket; the profile resolved in the daemon's own process; a dying daemon's stack (faulthandler) |
| serving/engine.py | +49 | `complete_turn` (the turn's ids, prompt length, bound, the dialect's markers kept); a cold daemon releases each request's components |
| serving/text_stream.py | +46 | `MarkerDecoder`: the special tokens a tool-call dialect writes kept in the decoded text |
| serving/operations.py | +19 | the HTTP queue's bound (`ServerBusy`), counted under the operation table's lock |
| agent/parsers/jsonlist.py, base.py, xml.py, jsonfmt.py, __init__.py | +61, +9, +2, +2, +4 | the JSON-list tool-call dialect (granite), every dialect's opening and closing markers |
| nbx/names.py | +104 | a model's one name: the hub name `import` records beside its entry, resolved for every command; the record never counted as a container file |
| cli/commands/registry.py, cli/utils.py | +60, +17 | `import` records the name (also on an installed entry), `list` shows it with the migration note; `--model` resolved once |
| cli/commands/serve.py, cli/__init__.py, run.py, drift.py, calibrate.py, upscale.py | +14, +8, +6, +4, +2, +2 | `serve --http` and its flags, the default profile left to the daemon, a killed daemon named by `stop`; the one name taken by every command |
| core/runtime/graph_executor.py, nbx/container.py | +8, +1 | the per-request cleanup releases the request's context and pinned block with its weights; the name record skipped as a container file |

## Served requests re-plan after a device OOM (2026-11-09): policy

The next useful version's point 8 (the owner, 2026-11-09 18:58): a request served by `neurobrix serve` meets a device
out-of-memory the way `run` does. The recovery itself is unchanged (core/runtime/replan.py, the executor's component
boundary); what is added is WHEN it is armed and what the daemon does with the plan it leaves: serving policy, no tensor
work. Measured: 140 044 (main 317b9217) to 140079, +35.

| file | lines | the policy it holds |
|---|---|---|
| serving/engine.py | +29 | every request runs with `run`'s recovery armed (`_execute`); a re-planned request leaves the daemon on the new plan, its device and its warm or cold serving (`_adopt_plan`) |
| cli/commands/run.py | +6 | the recovery's arming and the runtime-growth baseline become one function each, called by both entry points (`arm_replan`, `growth_baseline`, `compiled_sampler`) |

## The hardware profile reads more of the machine (2026-11-09): policy

Two points of reste_apres_064 (the owner's rule 1: the profile is the machine's identity card): an integrated GPU beside
an NVIDIA card is named (point 2), and a hybrid CPU's core classes are read on Linux and Windows (point 3). Both are
Prism's detection of the machine's facts from what the operating system states (sysfs, the PCI bus,
GetLogicalProcessorInformationEx); no tensor, kernel or device memory is touched, so nothing of it belongs in nbxtensor.
No new scan finding. Measured: 140 079 (main d4de640f) to 140 172, +93.

| file | lines | the policy it holds |
|---|---|---|
| core/prism/autodetect.py | +93 | `_beside`: the other vendors' GPUs the OS lists beside the driven cards, each refused by name; `_hybrid_cores` (+ `_hybrid_cores_linux`, `_cpu_list`, `_core_efficiency_classes`, `_hybrid_cores_windows`): a hybrid CPU's performance and efficiency cores as the OS states them, filled into the profile on every OS |

## An installed entry found by its container's sha256 (2026-11-09): policy

Point 13 of reste_apres_064 (agreed with the hub, 19:41): the name record keeps the sha256 of the container an entry was
installed from, `import` finds an installed entry by it before any byte is fetched, and `--container FILE.nbx` verifies
a local copy against the hub's sha256 and installs from it or records the name, with no download. The CLI and the name
record are engine policy; no tensor work. Measured: 140 172 (main 7f4fed4b) to 140 222, +50.

| file | lines | the policy it holds |
|---|---|---|
| cli/commands/registry.py, cli/__init__.py | +25, +4 | `import`: the entry found by the hub's sha, `--container` verified and never deleted by `--no-keep`, the sha recorded at install |
| nbx/names.py | +21 | the record's `container_sha256`; `entry_of_container` |

## The daemon's answers named (2026-11-09): policy

The Mac's findings validating point 8 on MPS (20:34): the daemon's error ends a run (no cold attempt after it), the
model is shown by its one name where a user reads it (serve's line, chat's status), and a socket path longer than
AF_UNIX holds is refused by name. Serving policy, no tensor work. Measured: 140 222 (main 931ae432) to 140 246, +24.

| file | lines | the policy it holds |
|---|---|---|
| serving/server.py | +16 | `_bind`: the long socket path refused naming the path and NEUROBRIX_HOME; the ready record's `name` |
| serving/engine.py | +7 | `one_name`, the status's `name` |
| cli/commands/run.py, cli/commands/chat.py | +1, 0 | the daemon's error is the run's answer; chat's status shows the name |

## The embedding family (2026-11-09): policy

Point 9b, route B (the owner 21:25): a generic `embedding` family whose pooling the CONTAINER declares, and
`/v1/embeddings`. The pooling math is nbxtensor's (`ops/embedding_pool`); the engine holds which output and mask to pool,
the refusals, and the OpenAI shapes. Measured: 140 246 (main 9ea8d314) to 140 323, +77.

| file | lines | the policy it holds |
|---|---|---|
| core/runtime/output_dispatch.py | +48 | `embedding_declaration` (the container's pooling, refused when absent), `extract_embeddings` / `save_embeddings` (the declared output and mask through the existing `final_as_array`, now with the whole batch), the `json` format |
| serving/openai_api/routes.py | +24 | `/v1/embeddings`: input a string or strings, float or base64, `dimensions` only the model's, usage |
| serving/server.py | +5 | a warm `run --output` of an embedding model: the daemon writes the vectors where it was asked |

## The KV cache's host tier (2026-11-09): policy, and the compiled engine's own copies

The next useful version's point 10 (the owner, 2026-11-09 18:58): a serving daemon parks a conversation that leaves the
cache in host memory and copies it back when it returns. The mechanism and the bookkeeping are nbxtensor's
(memory/kv_host_tier.py: the prefix index, the host page pool, `sequence_pages`; memory/kv_host_pages.py: the host pages
and the Triton cache's copies; paged_kv.py: the Triton cache's `tier_*` protocol). What the engine adds is WHEN a
conversation is parked or restored and HOW MUCH host memory the tier may hold (policy), and the compiled engine's own
copies over its torch pools (its paged cache is the engine's, torch-side: nbxtensor imports no torch, R33). Measured:
140 323 (main 3370ec86, rebased) to 140 537, +214.

| file | lines | what it holds |
|---|---|---|
| core/state/kv_tier.py | +145 | the tier's policy: park the held conversation before the cache drops it, restore a parked continuation before the prefill, the budget from the profile's host law (installed RAM less the system's share less the plan's host side) and the user's optional cap, a CEILING: host memory grows by one conversation of the plan as parks need it |
| core/runtime/graph/paged_kv_cache.py | +50 | the compiled engine's `tier_*` protocol over its torch pools (layout, length, offload, restore, resume) in inference mode: one indexed gather per pool and K/V, never a copy per page |
| triton/flow/autoregressive.py, core/flow/autoregressive.py | +5, +2 | the two hooks: park before a cleared cache, prepare before the match |
| serving/engine.py, serving/openai_api/routes.py | +4, +3 | the daemon's tier from its plan and profile; `usage.prompt_tokens_details.cached_tokens` |
| core/flow/base.py, core/runtime/executor.py, config/system.yml | +3, +2, +3 | the tier handed to the flow; `serving.kv_host_tier_mb` (null = the computed budget) |
## Constrained decoding's seam (2026-11-09): policy, and the compiled engine's own mask

The next useful version's point 9a (the owner 2026-11-09 18:58; our own grammar engine, 21:39): a request's decode
follows a constraint. The mechanism that touches the logits is nbxtensor's (ops/token_bitmask.py, the Triton branch);
what the engine adds is the constraint's protocol, where it is handed to the decode (the flow context, the executor,
both flows) and, in the compiled engine, its own torch twin of the mask from the same words (nbxtensor imports no torch,
R33; no compute shared between the engines). Measured on main d2a345d8: 140 537 to 140 580, +43.

| file | lines | what it holds |
|---|---|---|
| core/state/constraint.py | +21 | the constraint's protocol: the mask words before a step, the chosen token after |
| core/module/autoregressive/generator.py, triton/generator.py | +9, +8 | the mask before the sampler (compiled: masked_fill from the unpacked words; Triton: the nbxtensor kernel), the token reported after |
| core/flow/base.py, core/runtime/executor.py, both autoregressive flows | +2, +1, +1, +1 | the request's constraint handed to the flow and to its generator |

## A routed model on two cards, planned and served at speed (2026-11-10): policy and the compiled engine's own loop

The owner's order of 2026-11-10 02:49 (a served Qwen3-30B-A3B at 6.7 s a token): Prism's placement and budget rules,
the daemon's warm/cold decision and the compiled engine's multi-device loop — planning, serving and the torch branch's
execution, none of them a tensor mechanism of nbxtensor (the compiled engine's loop is torch-side; nbxtensor imports
no torch, R33). Measured: 140 537 (main) to 140 626, +89.

| file | lines | what it holds |
|---|---|---|
| core/prism/solver.py | +39 | a card whose runtime facts the profile lacks refused by name (`ProfileFactMissing`, let through the reading's handler); weight_sharding sized against `_usable_mb` with its split's real bytes and the head placed against what each card holds; a served cache on a card holding a zero3 component sized against the usable part, a routed component's expert-store floor counted as resident there; `_cost_multiplier_on` |
| core/runtime/graph/compiled_sequence.py | +25 | a boundary op's outputs delivered on its planned card (an arena-reading op such as the fused MoE); a failed multi-device op's error names its placement and each input's producer |
| core/flow/base.py | +12 | `holds_every_component`: a flow that holds every component together during a request |
| serving/engine.py | +8 | the daemon serves such a plan warm |
| core/runtime/executor.py | +5 | each component's strategy as it will execute, at -v |
| core/prism/solver.py (then) | +41 | a served cache sized after the model's residency (`_served_residency_bytes`; decided between the agents 04:37): 140 626 to 140 667 |

## Constrained decoding's vocabulary (2026-11-09): token_bytes per tokenizer class

Point 9a's grammar engine walks the tokenizer's tokens as BYTES. Each tokenizer class of core/module/tokenizer answers
`token_bytes(id)` by its own rule (the Dell's review 2026-11-09 21:35: by class, never by model): json_bpe applies its
decoder chain's per-token steps up to Fuse; sp_proto maps "▁" to a space and <0xXX> pieces to their byte; tiktoken and
Tekken read their rank tables; the runtime wrappers hand it on (polymorphic, kept above). Tokenizers are the engine's
(core/module), not tensor mechanism. Measured: 140 580 to 140 649, +69.

| file | lines | what it holds |
|---|---|---|
| core/module/tokenizer/json_bpe.py | +32 | the decoder chain's per-token bytes (ByteLevel, ByteFallback, Replace, Metaspace; stop at Fuse) |
| core/module/tokenizer/sp_tokenizer.py, sp_proto.py, tekken_bpe.py, tiktoken_bpe.py | +16, +13, +4, +4 | each class's rule and the wrappers' |

## Constrained decoding: the request's grammar (2026-11-09): policy

Point 9a end to end: `response_format: json_schema` on chat completions compiled by our own grammar engine
(the nbxgrammar package, github.com/NeuroBrix/nbxgrammar, Rust std only, outside this repository: its C ABI loaded by the package's ctypes loader) and
enforced at every decode step through the seam above. The engine adds the request-side policy: the schema read and
refused by name (400 naming the construct; json_object and a schema beside tools refused), the grammar constraint over
the tokenizer's token_bytes (built once per tokenizer and logits width), handed to the decode. Measured: 140 649 to
140 721, +72.

| file | lines | what it holds |
|---|---|---|
| core/state/constraint.py | +36 | the schema compiled (a refusal by name when the engine is absent), the grammar constraint and its vocabulary cache |
| serving/openai_api/routes.py | +26 | response_format read on chat completions: json_schema compiled before the request is queued, the refusals |
| serving/engine.py, core/module/tokenizer/sp_proto.py | +8, +2 | the constraint made per request and handed to the decode; an id past the pieces has no bytes |

## Constrained decoding: the forced tool calls (2026-11-09): policy

tool_choice required / a named function and parallel_tool_calls false enforced: each template dialect's envelope
(agent/envelope.py: json, xml, json_list, the format its parser reads back) written as a grammar description our grammar
engine compiles; a marker the tokenizer writes as one id taken as a token step. Policy (which format, which tools, how
many calls), no tensor work. Measured: 140 721 to 140 836, +115.

| file | lines | what it holds |
|---|---|---|
| agent/envelope.py | +80 | the three dialects' envelopes, the markers as token steps, required / named / parallel |
| serving/openai_api/routes.py | +25 | tool_choice and parallel_tool_calls read; the envelope compiled before the request is queued (a reasoning prompt closes its thinking first); the refusals |
| core/state/constraint.py | +10 | a grammar description compiled (a refusal by name when the engine is absent) |

## Constrained decoding: the Dell's review (2026-11-09 23:06): refusals instead of silent looseness

What the review asked turned into refusals by name and one shared check: the step mask read and checked once for both
engines (an empty mask refused before either applies it; the compiled engine decodes the same words), a missing
eos_token_id refused, the grammar engine absent answered 501 by name, an xml string parameter whose constraints free
text cannot enforce refused, a decoder step token_bytes has no per-token rule for refused, the markers read from the
parser. Measured: 140 836 to 140 868, +32.

| file | lines | what it holds |
|---|---|---|
| core/state/constraint.py | +16 | step_mask and allowed_tokens (one check, one decode for both engines); the vocabulary keyed by the tokenizer itself and its EOS ids |
| serving/openai_api/routes.py, serving/engine.py | +9, +3 | the 501 for a machine without the grammar engine; a missing eos_token_id refused |
| agent/envelope.py, core/module/tokenizer/json_bpe.py, both generators | +3, +1, 0 | the xml string refusal and the parser's markers; a decoder step without a per-token rule refused; the shared step mask |

Point 9a's sections above were measured on their own base; on main c957cee3, rebased with the Mac's reasoning-close and
D4 commits, the engine measures 140 667 to 141 025 (+358): the same files, policy as stated in each section.

The served prefill chunk on an engine without certified times (core/prism/solver.py `_prefill_chunk`, +13, policy; 2026-11-10): 141 255 to 141 268.

A run touches only its plan's cards (2026-11-10): core/device_utils.py `plan_devices` / `sync_plan` (+19, policy: WHICH devices a run synchronizes; the synchronization itself is the runtime's), the call sites net -1: 141 282 to 141 300.

## The scan

`tools/engine_slim_scan.py` (standard library, ast) lists four kinds of finding in `src/neurobrix`, both branches:

| kind | what it finds |
|---|---|
| `dead` | a definition nothing outside the tests refers to (by name, attribute, import, keyword, string or data file; registered by a decorator counts as used, a call to itself does not) |
| `duplicate` | a function whose normalised body (docstring dropped, parameters and locals renamed in order) is another engine or library function's |
| `wrapper` | a function that only hands its parameters on; a class that only renames its base; an alias of an imported name; a library name other engine modules import through an engine module (a re-export); a parameter no call site passes |
| `fact` | a definition carrying the name of a public nbxtensor fact function (`dtype`, `shape`, `memory`, the device facts, the `*_cost` declarations) or of a recorded synonym, and literal dtype-size tables |

Its rules are syntactic: a finding is a question a person answers, never a deletion on the scan's word alone.
Two of its blind spots, met in practice: a parameter passed through an aliased import (`from m import f as _g`,
then `_g(..., p=...)`) reads as never passed, and so does one the tests pass — every `wrapper` finding is checked
against the callers by hand, tests included, before a parameter goes.

## Three fates

Every finding ends in exactly one of three fates:

1. **Deleted** — dead or duplicated: the code goes, the callers read the one that stays.
2. **Moved into nbxtensor** — tensor work the library did not yet do: a generic API with its tests (seen red), its
   documentation in `API.md` and its README example in the same commit; the engine then calls it and its copy goes.
3. **Kept in the engine** — policy (a decision of the engine: Prism, the flow, the formats, the families): one sentence
   says why, in the table below.

Open findings are listed in `docs/nbxtensor/engine_slim_list.tsv`, whose header records the engine's code-line
count and the number of open findings. `tests/unit/nbxtensor_boundary/test_the_engine_only_slims.py` holds the rules:

- a finding that is neither open in the list nor kept below fails — a new one was written: delete it, move it, or
  justify it here;
- a listed or kept finding the scan no longer finds fails — it was resolved: delete its line;
- the engine's code-line count (no docstring, comment-only or blank line — the brick ratchet's count, re-based
  2026-11-10 from 141 714 lines of text to 85 785 of code) and the number of open findings may only go down.

## Deleted

| when | what | lines |
|---|---|---|
| 2026-10-29 | `src/neurobrix/kernels/triton_kernels_ref/` — 1 097 files of vendored third-party reference kernels (flaggems for a dozen vendors, fla, attorch, gemlite, ...), imported by no code, excluded from both package targets — deleted outright: the sources live on their own repositories (Hocine Benkelaya, 2026-10-30) | 262 132 |
| 2026-10-29 | `kernels/metadata_ops.py`: 34 handlers no table maps (`_addmm`, `_bmm`, `_exp`, ... — those ops resolve through `torch.ops.aten` by name) and the two listing helpers nothing calls; `kernels/classification.py`: one dead helper | 297 |
| 2026-10-29 | `core/prism/`: 36 definitions nothing called — the multi-GPU strategy predicates (`is_block_scatter`, `is_multi_gpu`, ...), the profile accessors (`get_device_strings`, `has_fast_interconnect`, ...), the device-state block helpers, the import planner's two entry points, three Apple chip lookups and their name normaliser, `compute_tensor_gb/mb` | 248 |
| 2026-10-29 | `core/runtime/`: 42 definitions nothing called — the memory pool's `MemoryOptimizer` and `LivenessAnalyzer` (the Triton allocator replaced them), the native dispatcher's seven view-wrapper factories, the executor's metadata and intercepted-op paths and its unregister / summary helpers, the KV wrapper's three intercepts, the debug loggers, `SymInt`'s legacy parsers, and accessors no caller reads | 994 |
| 2026-10-29 | `core/flow/`, `core/memory/`: 6 definitions nothing called — the autoregressive handler's graph prefill / decode-step pair, the audio engine's sample-rate helpers, `get_component_output`, `MemoryManager.get_memory_stats` | 65 |
| 2026-10-29 | 8 definitions only tests reached (a test proves no use): `solver.memory_ladder_rung_mb` (a pass-through to `memory_budget.rung_down_mb`, which its tests now call), `memory_budget.is_shared` (the sharing predicate of the law before 2026-10-25, read by no budget since), `ActivationProfiler.growth_anomaly` (an instrument never wired into a decision), `InputConfig.to_symbol_map` (a refusal stub for an old name), `PrismProfile.node_of`, `Partition.peak_resident_mb`, and two unmapped metadata handlers; the two test cells that exercised only the deleted code went with them | 126 |
| 2026-10-29 | `core/dtype/itemsize.py`, the PyTorch branch's mirror of `nbxtensor.dtype.itemsize` (every public function identical; the float64 fact is an argument), its loader `core.config.loader.get_dtype_table`, and the test that held the two equal: the 11 engine modules read the library | 144 |
| 2026-10-29 | `core/module/tiling_sizes.py`, the PyTorch branch's mirror of `nbxtensor.ops.tiling_sizes` (all 30 functions identical; its value reader took a cached snapshot where the library reads the policy in force), the loader `core.config.loader.get_tiling_policy`, and the test that held the two equal: Prism, the TilingEngine, the attention scratch and the derived census read the library | 374 |
| 2026-10-30 | `core/prism/__init__.py`: six backward-compatibility aliases nobody imports (`GpuDevice`, `SmartSolver`, `SmartExecutionPlan`, `smart_solve`, `StructComponentAllocation`, `SolverExecutionPlan`) | 15 |
| 2026-10-30 | `core/strategies/`, `core/module/`: 76 definitions nothing called (to a fixpoint) — the strategies' unused helpers (both branches), a tensor-parallel shard loader nothing reached (`create_block_shards` and its five helpers), every scheduler's `add_noise` (no flow noises latents: its only caller was dead), the scheduler and generator factories' unused entries, tokenizer helpers | 1 049 |
| 2026-10-30 | the strategies' `handle_outputs`, declared by the base and implemented by all 16 strategies of both branches, called by nothing (a test's subclass had to stub it "unused") | 210 |
| 2026-10-30 | `nbx/`, `serving/`, `core/components`, `core/dtype`: 24 definitions nothing called, the build toolchain included (checked in its repository) — container accessors and writers no reader or builder uses (`add_file`, `compute_checksum`, `set_topology`, `get_weights`, `list_files`, ...), two loader helpers, NeuroTax's unused groupings, the text-encoder handler's two probes, `embed_record`, `get_onnx_dtype`, a session's `build_prompt` | 320 |
| 2026-10-30 | `core/validators/`: 36 definitions nothing called (to a fixpoint) — the fusion, memory, op, shape and sync validators' checks, the runtime guards' checks, the validation pipeline's management API; what stays is what the `validate` command and the iterative process read | 668 |
| 2026-10-30 | `core/validators/`: the layered validation pipeline whole — ten modules (base, config, fusion, memory, numerical, op, shape and sync validators, runtime guards, the pipeline) reached only by the package's re-exports, which the scan reads as a use; nothing in the engine, its tools, its tests or the build toolchain imported them. Then, to a fixpoint, `NBXContainer.get_config` / `set_config`, which only they named | 2526 |

## Moved into nbxtensor

| when | engine definition | nbxtensor API |
|---|---|---|
| 2026-10-30 | `traced_output_dtype_name` (triton/dtype.py), `traced_output_is_complex` and `amp_fp32_output_dtype` (twins in core/dtype/engine.py and triton/dtype.py, held equal by tests) | `nbxtensor.dtype.outputs` — one definition both engines import |
| 2026-11-02 | the strategies' duplicates live once in `shared_steps` (`execute_whole_component`, `unload_sharded_weights` with each branch's `_release_device_cache`, `_load_sharded_weights`); 8 findings closed. The RATCHET RAISED by the remainder, +66 lines: the Triton arena factor became a fact the engine measures at its first launch on a card no vendor file declares (`core/prism/arena_probe`, the owner 2026-11-02 15:01, a 0.0.1 criterion) — 178 lines added, 112 taken back here | -112 / +178 |
| 2026-11-03 | the expert store's miss-copy measurement, first written as `core/prism/facts_probe.expert_copy`, lives next to its kernel (`expert_miss_copy_kernel`). The RATCHET RAISED by the remainder, +38 lines: the store reads the launch measured on the machine and the first Triton launch measures it where the profile carries none (the supervisor 2026-11-03 10:32, the first item: the copy is 75-93 % of a MoE decode's GPU time on the V100s), replacing a launch written in the code | `nbxtensor.ops.moe_device_residency.measure_miss_copy` |
| 2026-11-03 | (no move) The RATCHET RAISED by +16 lines: the expert store's routing record (`core/runtime/routing_reuse`) carries each layer's routing frequency per expert and the run's routed sequence, prompt included — what an expert cache's policy is chosen on (the supervisor 2026-11-03 12:25: in the same record, no new mechanism) | — |
| 2026-11-03 | (no move) The RATCHET RAISED by +56 lines: the per-piece reload rate (`weight_reload_gbps`) became a fact the engine measures at the first launch for each stored dtype of the cache (`core/prism/arena_probe.reloads`, the supervisor 2026-11-03 22:50, list 1.12), and a profile's host facts lie over its vendor's key by key (`loader._over`, the Dell's rule); the streamed run's own measure of it is retired (one fact, one writer) — 63 lines added, 9 taken back, and 2 naming why a streamed reload counts an NBXTensor's bytes by its own count (Flex.1-alpha triton died on int(method), 2026-11-04) | — |
| 2026-11-04 | (no move) The RATCHET RAISED by +7 lines: the replay signature names the place of its owners' contributions once (`replay._CONTRIB_AT`) and checks it where the key is built — the growth of a KV pool had read one place too far since 2026-10-28 and retired no plan. |
| 2026-11-03 | (no move) The RATCHET RAISED by +27 lines: Prism prices an expert store's misses at the Triton store's measured copy kernel (`measured.expert_copy`), the compiled store's DMA, nothing on a unified device (`time_estimate._miss_bps`) — priced at the DMA, a V100's misses were under-priced by a third (the list of what remains 1.12) | — |
| 2026-11-03 | (no move) The RATCHET RAISED by +48 lines: the certifier times the fused MoE GEMM on the matrix unit beside its FMA configurations (`_fused_moe_unit_runs`: the launch's routing read back and aligned again at each unit tile's rows by the engine's `moe_align_block_size`), and `_fused_moe_pass` takes the unit where the certificate names it (the list of what remains 1.7: x2.5-5 on the V100's prefill projections) | — |
| 2026-11-03 | (no move) The RATCHET RAISED by +39 lines: Prism's time estimate prices a candidate's zero3 component with the weights its plan would keep resident (`_residency_priced`: the plan's own grant, on a provisional plan) — lazy_sequential was estimated 1 062 s for Janus-Pro-7B 16 GB and runs 223 s and counts the tokens the request decodes (`InputConfig.decode_tokens`: granite-speech's 32 tokens were the package's 448) (the list of what remains 3.14); hybrid()'s 64-piece bound, kept until the estimate was right, goes | — |
| 2026-11-03 | the matrix unit's candidate runs live in the library (`nbxtensor.ops.linalg_wrappers.matrix_unit_runs`). The RATCHET RAISED by the remainder, +29 lines: the certifier times the unit's tiles beside the FMA configurations of a GEMM the unit can take and files the faster (the list of what remains 1.8, the owner's rule 2026-11-03 17:48 — the route is the certificate's measurement, never a threshold) | `nbxtensor.ops.linalg_wrappers.matrix_unit_runs` |
| 2026-11-03 | (no move) The RATCHET RAISED by +34 lines: the certifier synthesises a fused MoE key (the key's experts routed at random, aligned by the engine's `moe_align_block_size`, an fp64 oracle per pair) — the grouped GEMM's tile became a certified configuration instead of a literal (the list of what remains 1.9, the owner's rule 2026-11-03 17:48) | — |
| 2026-11-03 | (no move) The RATCHET RAISED by +70 lines: the certifier synthesises a fused MoE key and the one-token MoE band's three launches (gateup, down partials, their sum; the band's hand table `_HALF_VEC` removed) (the key's experts routed at random, aligned by the engine's `moe_align_block_size`, an fp64 oracle per pair) — the grouped GEMM's tile became a certified configuration instead of a literal (the list of what remains 1.9, the owner's rule 2026-11-03 17:48) | — |
| 2026-11-03 | (no move) The RATCHET RAISED by +171 lines: the certifier synthesises a fused MoE key, the vector and the paged decode's split launches and the one-token MoE band's three launches (gateup, down partials, their sum; the band's hand table `_HALF_VEC` removed) (the key's experts routed at random, aligned by the engine's `moe_align_block_size`, an fp64 oracle per pair) — the grouped GEMM's tile became a certified configuration instead of a literal (the list of what remains 1.9, the owner's rule 2026-11-03 17:48) | — |
| 2026-11-04 | (no move) The RATCHET RAISED by the kernels lot, +321 lines over origin bd029dca: the certifier measures the matrix unit beside the FMA configurations and where a certificate made before the unit owes it (`autotune_certify.unit_owed`), the fused MoE and the decode attentions' synthesis for certification (their kernels in nbxtensor), Prism's residency-priced estimate (`solver._residency_priced`), the expert store's miss rate (`time_estimate`), the census' paged and MoE launches. |
| 2026-11-04 | (no move) The RATCHET RAISED by +31 lines: a replay records only where its slab — sized by the slab's own first fit over the measured run (`nbxtensor.memory.slab.first_fit_extent`) — and the verify pass beside it fit what the card has free, after the measured run's pool is returned (Allegro-TI2V's encoder tile overran a 32 GB card). |
| 2026-11-04 | (no move) The RATCHET RAISED by +11 lines: the replay hands a view output over by its values and verifies values, not storage bytes (`replay._heap_copy`, `_copy_into`; Allegro-TI2V's permuted encoder output came back scrambled). |
| 2026-11-04 | (no move) The RATCHET RAISED by +7 lines: a released component collects the heap once, through each engine's own flow release (`RuntimeExecutor._unload_component_weights`; the Dell 2026-11-04 08:14) — mochi-1-preview native on Metal met 1-2 GB of device memory held by Python cycles after its transformer, since the per-segment collect left `MemoryManager.unload_weights` (4d1187f9) | — |
| 2026-11-04 | (no move) The RATCHET RAISED by +18 lines: the certifier's MoE synthesis builds its expert-address tables with the runtime's builder and releases their Metal pins (`autotune_certify._expert_table`). |
| 2026-11-06 | (no move) The RATCHET RAISED by +5 lines: a run that frees the inputs its caller hands over (a layer-streamed piece, `_consumed_input_tids`) is not replayed (`replay.signature`) — granite-speech-3.3-8b triton on the M4 Pro read a freed input at the verify's re-run, then a stale one in a plan that kept it (the Mac's Metal gate, 2026-11-06). |
| 2026-11-06 | (no move) The RATCHET RAISED by +4 lines: the layer partition's op curves price the operand casts an op's precision wrapper makes (`ExecutedWidths.casts`, at the partitioner's own symbols, a pass at its slice) and the solver keeps them in the widths it hands it — Sana_1600M_4Kpx native on the M4 Pro died at the VAE's conv::62 asking a 4.00 GiB cast the sliced stretch had not priced (the Mac's Metal gate, 2026-11-06). |
| 2026-11-06 | (no move) The RATCHET RAISED by +4 lines: the strategy lets go of a value handed to a piece only where that piece is the last reader of the pieces as they run (`_run_pieces`): a stretch run in slices replaces its segment after the hand-over was decided — MiniCPM-o-4_5 triton on the M4 Pro found a value popped before its stretch read it (the Mac's Metal gate, 2026-11-06). |
| 2026-11-06 | (no move) The RATCHET RAISED by +10 lines: the replay's verify step says what its comparison observed and no more (`replay.PLAN_MATCHED_LINE`, the supervisor's decision 21:30) — a divergence guard that cannot see an error both passes share, never cited as proof; the lines are its statement in the module and the comment of the verify branch |
| 2026-11-06 | (no move) The RATCHET RAISED by +9 lines: the width model stores a GEMM under native bf16 by the launcher's rule on its operands at C (`runtime_widths._lower_precision` -> `mm_store`, a bmm's M its batch rows: a decode row fp32) where it returned C — the Apple census keyed orpheus-3b's decode query bf16 where the run reads fp32 (the Mac's Metal gate, 2026-11-06). |
| 2026-11-06 | (no move) The RATCHET RAISED by +24 lines: a run with no mode flag takes the engine that is installed and says it (`engines.default_mode`, read by `run` and `serve`; the owner via the supervisor 22:10: the light install was refused at its first run) — the function, its two callers and the help texts that name the default |
| 2026-11-06 | (no move) The RATCHET RAISED by +5 lines: a batched product's M is read through the census memo's signature (`runtime_widths._Rules.batch_rows`, `binding_signature` carries it): the width model's bmm read it beside the signature, which the memo's door refused |
| 2026-11-07 | (no move) The RATCHET RAISED by +33 lines: `neurobrix doctor` on a Mac (the M4 Pro, 00:13): the running interpreter's scripts directory first and an inactive environment told to activate itself (`cli/_path_helper`), the Triton engine's macOS route and the Apple GPU said as what it is, PyTorch's MPS read (`cli/commands/doctor`) |
| 2026-11-07 | (no move) The RATCHET RAISED by +26 lines: a first run on a busy storage reads serially and says so instead of ending in a Traceback (`workspace.declare_unmeasured_io_workers`, `autodetect._measure_io_workers` returns None and the profile keeps no count; the supervisor 02:45) |
| 2026-11-07 | (no move) The RATCHET RAISED by +28 lines: every directory under the engine's home follows a configured NEUROBRIX_HOME (`core.paths.engine_home`, read by the module constants of the replay cache, the license acceptances, the hardware profiles, the calibration store, the render checkpoints, the daemon, drift and the certifier's rig protocol; the library's replay cache told at import) and doctor's PyTorch line on a Mac (the Mac's new-user run 2026-11-07) |
| 2026-11-07 | (no move) The RATCHET RAISED by +19 lines: the simple half of the Windows audit (`cli._utf8_streams`; the install lock's liveness asked of the kernel on Windows in `nbx.atomic_install`; a debug dump's file name sanitized; text files in UTF-8) |
| 2026-11-07 | (no move) The RATCHET RAISED by +19 lines: the plan's resident term on Windows (`host_footprint.process_footprint_now` reads the working set through psapi; the supervisor 04:50, the audit's defect 3) |
| 2026-11-07 | (no move) The RATCHET RAISED by +21 lines: a first launch's arena probe that the device cannot hold bounds the measure instead of ending the run (`arena_probe.ProbeOutOfMemory`, the child's error said by name; the Mac's trial M of 0.6.1) |
| 2026-11-07 | (no move) The RATCHET RAISED by +51 lines: the arena probe scales only what the priced request scales and is taken on a spatial component, else a model's whole sequence walk (`arena_probe.scaled_names`, `smallest_container`; a fresh Mac measured 28.27 where its factor is about 1.5) |
| 2026-11-07 | (no move) The RATCHET RAISED by +11 lines: the arena factor declared by the device's vendor file, else its family (`core/config/loader.declared_arena_factor`, `vendors/family_facts.yml`), never probed at a user's first run (the supervisor 12:00) |
| 2026-11-07 | (no move) The RATCHET RAISED by +4 lines: the certifier's price door prices the key drawn on a unified device, not the key filed (`autotune_certify.synthesize`; the open element class refused at 2.2 TB on the Mac's re-proof) |
| 2026-11-07 | (no move) The RATCHET RAISED by +9 lines: the arena probe runs certified-only and a missing key bounds it (`arena_probe.ProbeUncertified`): a probe never sweeps |
| 2026-11-07 | (no move) The RATCHET RAISED by +34 lines: the certifier uploads a row-windowed key's draw once and keeps only the oracle's windows on the host (`autotune_certify._HostWindows`, `_oracle_rows`; price_key counts the windows): HAT-L/S and mochi keys refused at 20-33 GB of an 18 GB Mac |
| 2026-11-08 | (no move) The RATCHET RAISED by +61 lines: a language model's prompt is priced at its own length — the prompt's token count by the flow's own rule moved to a neutral module both engines and the derived census read (`core/runtime/prompt_length`, from `triton/flow/autoregressive`), bound by `request_input_config` and `FlowBindings.prompt_prefill_length` (TinyLlama priced ~0.01 GiB for any prompt where the run held up to 652 MB) |
| 2026-11-08 | The expert store's table pin holder (`TablePins`, `end_call_pins`: who ends a per-call pointer table's Metal pins, one held pin per new set of wraps) | `nbxtensor.memory.pointer_tables` (nbxtensor 0.0.6) — the owner's boundary rule of 2026-11-08 16:43: pins of wraps are mechanism; the engine keeps only the two decode-band call sites (+2 lines) |
| 2026-11-08 | (no move) The RATCHET RAISED by +61 lines, every line POLICY under the boundary rule of 16:43, the findings unchanged at 169 (decided between the Dell and the Mac, the count of findings never rises, each new line moved or justified here): Prism's refusal when an Apple profile loses its device's largest single allocation (+27, `core/prism/autodetect.py`, `loader.py`, `solver.py`: what Prism accepts as a profile fact); an expert store's miss on a unified device priced as a page-in at the measured cold load rate (+4, `core/prism/time_estimate.py`: a price); the compiled engine's measured growth as its one reserve (+6, `core/prism/solver.py`: the budget); the decode band's two calls to the library's pin holder (+2, `triton/moe.py`: when the flow ends them); the compiled engine on MPS recording its allocator's cache growth (+22, `core/device_utils.py`, `core/runtime/runtime_growth.py`, `cli/commands/run.py`: a torch-side reading of the compiled engine, which nbxtensor, torch-free, cannot hold) |
| 2026-11-08 | (no move) The RATCHET RAISED by +193 lines, every line POLICY under the boundary rule of 16:43, the findings unchanged at 169: a run that meets a device out-of-memory at a component re-plans once and re-enters its flow there (decided between the Dell and the Mac 18:51, release 0.6.3) — `core/runtime/replan.py` (+71: what of a plan decides how a component runs, compared before and after; which executor holds a flow's state between calls, refused by name; the recovery's contract), `core/runtime/executor.py` (+87: the flow's component boundary, the one re-plan per component and its named refusals, the new plan applied to the executors it moves, what the flow had loaded loaded again; setup's executor build and plan tiling factored so the re-plan reuses them), `cli/commands/run.py` (+36: each engine's facts given to it — its typed out-of-memory or the vendor file's untyped message, its growth record, Prism asked again). No tensor work: the release is each engine's existing one (`core.memory.manager.release_flow_memory`, `nbxtensor.memory.memory_pool.release_flow_memory`) |
| 2026-11-08 | (no move) The RATCHET RAISED by +13 lines, POLICY: `neurobrix autotune check` runs the library's generator-alias gate (`nbxtensor.ops.autotune_certified.validate_generator_aliases`) over each profile directory's alias file and says it — the command line's report of a library check (cli/commands/autotune.py); the mechanism lives in nbxtensor |
| 2026-11-10 | (no move) The RATCHET RAISED by +128 lines (141 025 -> 141 255), the findings unchanged: the mid-decode KV spill past the card's pages (point 10; the mechanism and the host math are nbxtensor's: memory/host_spill.py, ops/host_decode_partial.py, memory/paged_kv.py) — `core/state/kv_tier.py` (+33, POLICY: one host budget for the parked pages and the spill, a spilled conversation never parked, `attach_spill` deciding from the plan's cache and the model's window when a served cache spills), the two flows (+10, POLICY: the call at the served cache's creation), `core/runtime/graph/paged_kv_cache.py` and `kv_cache_wrapper.py` (+125, the COMPILED engine's own cache, torch tensors nbxtensor cannot hold: its R30 mirror of the spill, and the card partial of a spilled decode step and of a spilled prefill chunk in torch, merged with the library's host partial); `core/state/kv_tier.served_reach` and the serving door (+61, POLICY: the one served reach both doors read — card, then the host spill within the window — said at start, a request past it refused as 400 context_length_exceeded before it runs) |
| 2026-11-10 | (no move) The RATCHET RAISED by +14 lines (141 268 -> 141 282), POLICY: the compiled flow keeps ONE served cache per executor and, at a session switch, parks the conversation it held in the KV host tier before resetting it (`core/flow/autoregressive.py`, the R30 mirror of the Triton flow's executor-scoped cache; a new cache per session parked nothing, so the compiled engine never restored a returning conversation — the Dell's device cell 2026-11-10 11:41) |
| 2026-11-10 | (no move) The RATCHET RAISED from 141 300 to 141 582 lines (+163 for `run`, then the daemon's reload and doctor's lines), POLICY: an Intel GPU refused at run time (the experimental Level Zero backend; the owner 13:09) — `core/prism/intel_refusal.py` (what the engine does with nbxtensor's typed IntelDeviceError: the sentence, the record under the driver and nbxtensor versions, the one restart on the CPU, the expiry), detection listing a recorded chip as a refused device (`core/prism/autodetect.py`), the command line's entry (`cli/commands/run.py`), the daemon's one reload on the CPU plan (`serving/server.py`), doctor's lines (`cli/commands/doctor.py`). The mechanism — the Level Zero runtime, the driver, the typed error — is nbxtensor's |
| 2026-11-10 | (no move) The RATCHET RAISED by +13 lines (141 582 -> 141 595), the compiled mirror of a library mechanism (R30): a spilled prefill chunk's math in `core/runtime/graph/kv_cache_wrapper._prefill_over_spill` is torch's — it writes its fp32 scores — so its query rows are cut to one page of rows (or the plan's `attention_scores_budget_bytes` where the device writes full scores), refused by name when one row does not fit: the served plan's larger chunks (2 048 rows x 65 664 keys = 17 GB a matrix) made the uncut math impossible. The spilled positions themselves come from nbxtensor (`HostSpill.blocks`); the Triton engine's cut is the library's own attention route. Decided between the agents 17:07. |
| 2026-11-10 | (no move) The RATCHET RAISED by +119 lines (141 595 -> 141 714), all policy: Prism's pricing of the generation head at the one row the flows feed it (`core/prism/flow_bindings.FlowBindings.head_rows`, `InputConfig.scoring`, `run.scoring_request`), the daemon's request carrying the flow facts, and a new named decision component, `core/prism/served_prefill.py` (its contract at its head: the rows a serving session's prefill chunk is priced at, raised while the plan's residents hold). No tensor, kernel or device-memory code: the chunks themselves run through the existing prefill path. |
| 2026-11-10 | (no move) The header RE-BASED, not raised: `engine_lines` now counts CODE lines — no docstring, comment-only or blank line — the brick ratchet's own count (`tools/brick_ratchet.code_lines`), one definition for both ratchets. The same tree reads 141 714 lines of text and 85 785 lines of code; the header becomes 85 785. Why: the owner's rule of 2026-11-10 (code in bricks) has catch-all functions cut into named pieces, each with its one-sentence docstring, and this code carries its reasons in comments — under the old count both read as growth and the two ratchets pulled against each other (decided between the agents, 17:26). From here a row's figures are code lines. |
| 2026-11-10 | (no move) The RATCHET RAISED by +29 code lines (85 785 -> 85 814), POLICY, and it removes two hardcoded values: the batch and the dtype a request is planned at are one rule in one module with its contract (`core/prism/request_config.py`) — `run` kept it inline and the daemon restated it as the literals `batch_size=2`, `dtype="float16"` for every served family; and which engine reads a weight's transpose in place is asked of the profiler (`ActivationProfiler.weight_transposes_free`) from the flow's own fact that the text head runs with AMP off (`FlowBindings.head_runs_without_amp`). The same fact reaches the precision contract the widths are priced under (`PrecisionContract.amp`, `plan_time_contract(..., flow)`): a component run with AMP off is priced no operand copy. Under the brick ratchet the lot SHORTENS three listed offenders: `request_input_config` 106 -> 97, `PrismSolver._compute_memory` 242 -> 241, solver.py 6 280 -> 6 279. No mechanism: nothing here is nbxtensor's. |
| 2026-11-10 | (no move) The RATCHET RAISED to 85 853 code lines (from 85 814: +24 for Prism's pricing of the session chunk's attention reserve — `core/state/price.session_chunk_attention_bytes`, `core/prism/served_prefill.py` stating the cached keys, policy — and +15 for this row's share), the compiled mirror of the session chunk's attention (R30): `kv_cache_wrapper` reads a plain chunk with one attention per query group on the stored KV heads (the GQA copies are gone) and a spilled chunk in key blocks by spans of rows — `_partial`, `_merge_partials`, `_spill_cut` — holding one fp32 score matrix and one key block inside the reserve Prism priced (`price.session_chunk_attention_bytes`, read through `kv_tier.session_chunk_reserve`). torch code: the Triton engine's own path is nbxtensor's. |
| 2026-11-10 | (no move) The RATCHET RAISED by +20 code lines (85 853 -> 85 873), POLICY: a run holds a context only on the cards its plan names. `core/runtime/card_handoff.py` (a named component: the compiled engine's card-to-card hand-off through the host, because PyTorch >= 2.10 creates a context on every visible card at its first GPU-to-GPU copy and the library's raw peer copy is refused on PyTorch's expandable-segments memory — both measured), Prism's read of a card from the driver's tool before the runtime (`memory_budget.read_device_sharing`), the weight loader's pinned upload made on the card written. No tensor mechanism: the copies are torch's own `.to()`. |
| 2026-11-10 | (no move) The RATCHET RAISED by +42 code lines (85 873 -> 85 915), POLICY, a regroup with no behaviour change: the decision of what a chat request's decode is constrained by gets its own component with its contract (`serving/openai_api/constraint_policy.py`: two pure decisions and one typed refusal, no tokenizer, no HTTP status) and the execution seam owns its input types and builds what it executes (`core/state/constraint.Constrain` / `ForcedCall`, `compile_constraint`, `constraint_for`); `chat_completions` shrinks 134 -> 124 code lines and `complete_turn` loses its eight. The added lines are the two value types, the refusal type and the contracts' signatures. Proven by a recording of 400 request combinations taken before the move. |
| 2026-11-11 | (no move) The RATCHET RAISED by +8 code lines (85 915 -> 85 923), POLICY of the command line: two request flags no flow reads (`--reference-audio`, `--reference-image`) are refused by name before anything loads (`cli/utils.refuse_unread_flags`, a table of the unread flags with what each would have done) instead of being accepted and ignored — ZERO FALLBACK; the dead store of `global.reference_audio_path` leaves `cmd_run`. |
| 2026-11-11 | (no move) The RATCHET RAISED by +18 code lines (85 922 -> 85 940), POLICY: a generated hardware profile whose cards state no runtime total / context gets them measured at a launch (`core/prism/facts_probe.complete_runtime_card_facts`, asked by `autodetect.ensure_measured_facts` with its own reading) instead of refusing every plan for ever, each card read in its own short process (`card_facts_in_a_child`), and detection — run at every launch — no longer asks the runtime about any card; `ensure_measured_facts` does not grow and autodetect.py shrinks (1 900 -> 1 898). |
| 2026-11-11 | (no move) The RATCHET RAISED by +3 code lines (85 940 -> 85 943), POLICY, the command line's model lookup: a name that leads to an installed model without being its one name gets the single message naming it (`nbx/names.resolve`, the organisation-prefixed folder name included), the "installed models" lists print the names `--model` accepts (`nbx/names.accepted_names`), and `import` says when the hub's name is not the one typed (`cli/commands/registry._installed_line`; `_import_body` does not grow). No tensor code. |

## Kept in the engine

| kind | path | symbol | why it stays |
|---|---|---|---|
| fact | src/neurobrix/core/strategies/base.py | transfer_tensor | the strategy's method takes a DEVICE STRING (cpu, cuda:N, mps:N) and passes a non-tensor value through; the library's function takes a device index and an NBXTensor only — a different contract, not a copy (2026-11-02) |
| fact | src/neurobrix/core/strategies/triton/base.py | transfer_tensor | the same method on the Triton branch: device strings, cpu, pass-through of non-tensors (2026-11-02) |
| fact | src/neurobrix/triton/sequence.py | _parse_dtype | parses through the library's parse_dtype, then remaps bf16<->fp16 to the plan's compute dtype (Prism's decision, the engine's) (2026-11-02) |
| wrapper | src/neurobrix/core/module/tokenizer/sp_tokenizer.py | token_bytes | polymorphic: every runtime tokenizer answers token_bytes (constrained decoding's vocabulary), the grammar engine calls it whatever the class (2026-11-09) |
| wrapper | src/neurobrix/core/module/tokenizer/tiktoken_bpe.py | token_bytes | polymorphic: the tiktoken class's answer to token_bytes, its rank table read (2026-11-09) |
| wrapper | src/neurobrix/core/strategies/base.py | synchronize_device | polymorphic: both branches' bases define it (the Triton one through DeviceAllocator), and the shared steps call it on self (2026-11-02) |
| wrapper | src/neurobrix/core/strategies/shared_steps.py | cleanup | polymorphic: the flows call `cleanup()` on every strategy, this one empties the loaded set (2026-11-02) |
| wrapper | src/neurobrix/core/strategies/layer_streaming.py | _nbx_path | a test seam: six streaming tests stub it to run card-less (2026-11-02) |
| wrapper | src/neurobrix/core/prism/arena_probe.py | ProbeOutOfMemory | the arena probe's child process says by name that the device could not hold its request — the measure's bound, crossing a process boundary as text; the library's DeviceOOMError is raised in the child, its figures do not travel (2026-11-07) |
| wrapper | src/neurobrix/core/prism/arena_probe.py | ProbeUncertified | the arena probe's child says by name that its request formed a key the certified directory does not serve — the measure's bound, crossing a process boundary as text; the library's KeyNotCertified is raised in the child (2026-11-07) |
| duplicate | src/neurobrix/core/optim/passes/algebraic.py | _scalar | nbxtensor's twin is a function local to its residual-chain reader: the library exposes nothing to import, and the library never imports the engine |
| wrapper | src/neurobrix/core/cfg/engine.py | execute_component_with_cfg | the CFG engine's public entry, called by both iterative handlers; the Triton engine's converts its inputs at the boundary first, so the two are not one |
| wrapper | src/neurobrix/_nbxtensor_host.py | active | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| wrapper | src/neurobrix/_nbxtensor_host.py | record | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| wrapper | src/neurobrix/_nbxtensor_host.py | service | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| wrapper | src/neurobrix/_nbxtensor_host.py | shadow_run | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| fact | src/neurobrix/triton/dtype.py | storage_dtype | already reads nbxtensor's itemsize.storage_dtype at this branch's fp64 fact and returns the NBXDtype member: the engine's dtype-engine method over the library's table |
| wrapper | src/neurobrix/serving/engine.py | ContextLengthExceeded | the serving door's refusal BY CLASS: a request past the served reach is answered 400 context_length_exceeded (OpenAI's code) before it runs, any other engine failure a 500 — the class is what the door reads (2026-11-10) |
| wrapper | src/neurobrix/kernels/census.py | install(hardware_profile) | the tests install the census for a profile they name (tests/unit/kernels/test_a_one_row_convolution_keys_its_width_on_the_ladder) |
| wrapper | src/neurobrix/kernels/census_table.py | rows_for(root) | the tests read a table under a temporary root (tests/unit/kernels/test_the_census_is_one_table) |
| wrapper | src/neurobrix/agent/sandbox.py | SandboxViolation | a named refusal: the agent tells a sandbox refusal from a tool's own failure (exported by neurobrix.agent) |
| wrapper | src/neurobrix/core/workspace.py | SnapshotNotPresent | a named refusal: a missing snapshot, told apart from any other error by its callers |
| fact | src/neurobrix/core/runtime/graph/compiled_sequence.py | _parse_dtype | returns a torch.dtype with the sequence's Prism remap (compute_dtype); nbxtensor's parse_dtype returns an NBXDtype — two types |
| fact | src/neurobrix/core/runtime/precision_contract.py | kind | a different function under the same name: the precision class of an op (precision / casting / passthrough), not a dtype's kind |
| wrapper | src/neurobrix/core/runtime/graph/device_attention.py | _plain | the plain-SDPA entry of the attention dispatch table, called positionally like its other entries |
| wrapper | src/neurobrix/core/runtime/graph/kv_cache_wrapper.py | intercept_attention(enable_gqa) | the graph's own SDPA kwarg, handed in by the op dispatch from the traced op's kwargs (chatterbox t3) — the scan sees no Python caller |
| wrapper | src/neurobrix/core/runtime/graph/moe_fusion.py | detect_and_fuse_moe(refusals) | the tests read the refusals it collects (tests/unit/runtime/test_a_stacked_expert_block_routed_softmax_first_is_fused) |
| wrapper | src/neurobrix/core/runtime/graph_executor.py | _resolve_fp16_activation_policy | the executor's door to its precision contract, called at two compile points and by tests/unit/dtype/test_precision_contract |
| wrapper | src/neurobrix/core/runtime/graph_executor.py | execute | the executor interface's name (serving, strategies); run() takes a pre-op callback second, so it is not an alias of run |
| wrapper | src/neurobrix/core/runtime/lm_facts.py | decode_sequences(resolved) | the run's resolved values (a CLI cfg_scale) — the tests pass it (tests/unit/prism/test_the_plan_prices_every_cache_its_decoder_opens) |
| wrapper | src/neurobrix/core/runtime/phase_trace.py | first | the ledger's read door: _FIRST is the module's private state |
| wrapper | src/neurobrix/core/runtime/tensor_compat.py | aten | a named question (is torch loaded?) asked by the shared orchestration code, never importing it |
| wrapper | src/neurobrix/core/paths.py | PathNotConfigured | a named refusal: a caller tells a missing configured location from an ordinary directory it may create (tests/unit/core/test_one_door_for_the_machines_paths catches it by name) |
| wrapper | src/neurobrix/core/paths.py | BackendOnClearedStorage | a named refusal caught by name in cli run and cli autotune, beside the backend-not-installed errors, to refuse with its own sentence |
| wrapper | src/neurobrix/core/paths.py | installation_refusals(search_root) | the tests point it at a fake installation tree (tests/unit/triton/test_a_backend_built_on_ephemeral_storage_is_refused); the engine looks the package up on sys.path |
| fact | src/neurobrix/core/memory/manager.py | release_flow_memory | the compiled engine's own release: torch's device sync and cache flush (device_utils) on the flow's device; nbxtensor's syncs and empties the DeviceAllocator pool (no torch, R33) — the same discipline, two allocators |
| wrapper | src/neurobrix/core/flow/autoregressive.py | use_branch | the session's branch API, the pair of new_branch (the Triton session has the same): the next_token_diffusion flows hold the session, not its cache wrapper |
| duplicate | src/neurobrix/core/flow/next_token_diffusion.py | _require_tensor | each branch tests its own tensor type for the primary output (torch.Tensor in the compiled engine, NBXTensor in the Triton one): the same text, two types |
| duplicate | src/neurobrix/triton/flow/next_token_diffusion.py | _require | each branch tests its own tensor type for the primary output (torch.Tensor in the compiled engine, NBXTensor in the Triton one): the same text, two types |
| duplicate | src/neurobrix/core/flow/vlm.py | _span | a function nested in the VLM decode, closing over that decode's own locals (its ids, its tokenizer); the same text in both branches, shared only by lifting it with its captured state as parameters — the VLM decode is the Dell's in flight (logits projection, 2026-11-02 01:29) |
| duplicate | src/neurobrix/triton/flow/vlm.py | _span | a function nested in the VLM decode, closing over that decode's own locals (its ids, its tokenizer); the same text in both branches, shared only by lifting it with its captured state as parameters — the VLM decode is the Dell's in flight (logits projection, 2026-11-02 01:29) |
| duplicate | src/neurobrix/core/flow/vlm.py | _tok_str | a function nested in the VLM decode, closing over that decode's own locals (its ids, its tokenizer); the same text in both branches, shared only by lifting it with its captured state as parameters — the VLM decode is the Dell's in flight (logits projection, 2026-11-02 01:29) |
| duplicate | src/neurobrix/triton/flow/vlm.py | _tok_str | a function nested in the VLM decode, closing over that decode's own locals (its ids, its tokenizer); the same text in both branches, shared only by lifting it with its captured state as parameters — the VLM decode is the Dell's in flight (logits projection, 2026-11-02 01:29) |
| duplicate | src/neurobrix/core/flow/forward_pass.py | execute | each branch frees its own memory: the compiled one calls core/memory/manager.release_flow_memory, the Triton one nbxtensor.memory.memory_pool.release_flow_memory (two functions) — the text is the same, the call is not |
| duplicate | src/neurobrix/triton/flow/forward_pass.py | execute | each branch frees its own memory: the compiled one calls core/memory/manager.release_flow_memory, the Triton one nbxtensor.memory.memory_pool.release_flow_memory (two functions) — the text is the same, the call is not |
| duplicate | src/neurobrix/core/flow/iterative_process.py | _unload_component | each branch frees its own memory: the compiled one calls core/memory/manager.release_flow_memory, the Triton one nbxtensor.memory.memory_pool.release_flow_memory (two functions) — the text is the same, the call is not |
| duplicate | src/neurobrix/triton/flow/iterative_process.py | _unload_component | each branch frees its own memory: the compiled one calls core/memory/manager.release_flow_memory, the Triton one nbxtensor.memory.memory_pool.release_flow_memory (two functions) — the text is the same, the call is not |
| duplicate | src/neurobrix/core/flow/static_graph.py | _unload_component | each branch frees its own memory: the compiled one calls core/memory/manager.release_flow_memory, the Triton one nbxtensor.memory.memory_pool.release_flow_memory (two functions) — the text is the same, the call is not |
| duplicate | src/neurobrix/triton/flow/static_graph.py | _unload_component | each branch frees its own memory: the compiled one calls core/memory/manager.release_flow_memory, the Triton one nbxtensor.memory.memory_pool.release_flow_memory (two functions) — the text is the same, the call is not |
| duplicate | src/neurobrix/core/module/autoregressive/samplers.py | validate | each config type's own entry point (SamplerConfig / SchedulerConfig): the same fill-in of its DEFAULTS today, two unrelated configs — a shared base for three lines would tie them |
| duplicate | src/neurobrix/core/module/scheduler/config.py | validate | the scheduler config's entry point; same reason as SamplerConfig.validate |
| duplicate | src/neurobrix/core/module/autoregressive/samplers.py | __init__ | the compiled twin of triton/samplers.py's, which imports no torch (R33) and so cannot inherit from this module |
| duplicate | src/neurobrix/core/module/autoregressive/samplers.py | __call__ | each sampler class's own call (the dispatch picks a class by strategy); two classes' calls coincide today |
| wrapper | src/neurobrix/core/module/text/processor.py | tokenize_for_diffusion(encoder_name) | the encoder component the prompt is tokenized for; removing the parameter would only move the 'text_encoder' name into the body as a literal |
| wrapper | src/neurobrix/core/module/audio/output_processor.py | decode_snac_tokens(audio_token_start) | NOT a slimming question: the default is one model's constant (Orpheus's 128266, with vocab_size 156940) — both must come from the container (rule 3: zero per-model value); named to the Dell 2026-11-02 |
| wrapper | src/neurobrix/core/module/tiling_engine.py | make_unified_node_interceptor(_executor_ref) | a closure capture: the default binds the graph executor at definition, never a parameter a caller passes |
| wrapper | src/neurobrix/core/module/tiling_engine.py | make_unified_node_interceptor(_pending) | a closure capture: the default binds the pending chain at definition, never a parameter a caller passes |
| wrapper | src/neurobrix/core/module/tiling_engine.py | make_unified_node_interceptor(_registry) | a closure capture: the default binds the chain registry at definition, never a parameter a caller passes |
| wrapper | src/neurobrix/cli/commands/registry.py | download_resumable(chunk_size) | a test seam: the download tests pass small chunks to exercise resumption without megabytes |
| wrapper | src/neurobrix/nbx/atomic_install.py | installing(allow_break) | the one deliberate opening of the install lock (a lock held by another host broken on purpose), named in the refusal a user reads |
| wrapper | src/neurobrix/nbx/cache.py | extract(allow_replace) | the deliberate opening a refusal names (`extract(..., allow_replace=True)`, or its env): replacing a live model is said, never silent |
| wrapper | src/neurobrix/cli/commands/agent.py | complete_fn | two closures returned as one callback (daemon client, cold engine): the agent loop calls whichever the launch chose |
| wrapper | src/neurobrix/cli/commands/registry.py | _strip_build_stamp | a named rule (the build stamp a comparison ignores), pinned by its test |
| wrapper | src/neurobrix/cli/commands/run.py | _rt | the one door for every runtime dimension (request -> container defaults -> family config -> refusal), 8 callers |
| wrapper | src/neurobrix/cli/json_out.py | close | the writer's close of the JSON-lines protocol its callers hold |
| wrapper | src/neurobrix/core/components/handlers/transformer_handler.py | get_latent_scale | the override of the handler interface (core/components/base.py) the executor calls |
| wrapper | src/neurobrix/core/config/system.py | get_prism_defaults | a defensive copy of the defaults (3 callers): a caller editing it never edits the table |
| wrapper | src/neurobrix/nbx/component_flags.py | clear | the reset of the process's flag table the plan tests start from |
| wrapper | src/neurobrix/serving/client.py | new_chat | the daemon client's API (the CLI and the serving tests call it): each method is one request of the protocol |
| wrapper | src/neurobrix/serving/client.py | shutdown | the daemon client's API (the CLI and the serving tests call it): each method is one request of the protocol |
| wrapper | src/neurobrix/serving/client.py | status | the daemon client's API (the CLI and the serving tests call it): each method is one request of the protocol |
| wrapper | src/neurobrix/serving/client.py | template | the daemon client's API (the CLI and the serving tests call it): each method is one request of the protocol |
| wrapper | src/neurobrix/serving/protocol.py | endpoint | the module-level door to the daemon's one endpoint instance, used by the CLI and the serving tests |
| wrapper | src/neurobrix/cli/commands/registry.py | IncompleteDownload | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| wrapper | src/neurobrix/core/config/loader.py | UnsupportedArchitectureError | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| wrapper | src/neurobrix/core/module/autoregressive/factory.py | AutoregressiveConfigError | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| wrapper | src/neurobrix/core/module/scheduler/config.py | SchedulerConfigError | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| wrapper | src/neurobrix/core/optim/passes/invariant.py | InvariantPlanRefused | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| wrapper | src/neurobrix/nbx/atomic_install.py | InstallHeldByAnother | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| wrapper | src/neurobrix/nbx/weight_presence.py | AbsentWeightError | a named refusal: its type says what was refused, to the reader and to a caller that catches it |
| dead | src/neurobrix/cli/commands/registry.py | installing_path | a test seam: read by tests/unit/cli/test_import_speaks_ndjson.py, nothing else |
| dead | src/neurobrix/core/module/autoregressive/generator.py | AutoregressiveGenerator.num_generated | a test seam: read by tests/unit/serving/test_both_engines_read_the_stop_at_their_token_seam.py, nothing else |
| dead | src/neurobrix/core/module/tiling_engine.py | OpLevelTilingPlan.add_inplace_unary | a test seam: read by tests/unit/prism/test_the_tiling_rung_s_reason_reaches_a_reader.py, nothing else |
| dead | src/neurobrix/core/optim/passes/normalize.py | boundaries_present | a test seam: read by tests/unit/prism/test_a_segment_boundary_names_an_op_the_executor_still_has.py (the invariant a gate asserts), nothing else |
| dead | src/neurobrix/core/runtime/graph/paged_kv_cache.py | PagedKVCache.held_bytes | a test seam: read by the paged-cache tests of both engines (what was appended is held), nothing else |
| dead | src/neurobrix/core/state/mps_fence.py | MpsStateFence | a test seam: read by tests/unit/core/test_a_metal_page_comes_back_only_after_the_kernels_that_read_it.py, nothing else |
| dead | src/neurobrix/nbx/container.py | NBXContainer.get_topology | a test seam: read by tests/unit/prism/test_the_plan_prices_every_cache_its_decoder_opens.py, nothing else |
| duplicate | src/neurobrix/core/runtime/resolution/i2v_conditioning.py | _to_channels_first | the same text over each branch's tensor (torch.Tensor / NBXTensor: `dim`, `permute`, `contiguous` of its own class); one copy per branch, each branch's vace module reads it |
| duplicate | src/neurobrix/core/runtime/resolution/i2v_conditioning.py | build_condition | the same dispatch, to each branch's own builders (`_build_condition_wan` / `_cogvideox` / `_state_video_mask` of that module): the text is shared, the functions it calls are not |
| duplicate | src/neurobrix/triton/i2v_conditioning.py | _to_channels_first | the same text over each branch's tensor (torch.Tensor / NBXTensor: `dim`, `permute`, `contiguous` of its own class); one copy per branch, each branch's vace module reads it |
| duplicate | src/neurobrix/triton/i2v_conditioning.py | build_condition | the same dispatch, to each branch's own builders (`_build_condition_wan` / `_cogvideox` / `_state_video_mask` of that module): the text is shared, the functions it calls are not |
| duplicate | src/neurobrix/triton/samplers.py | __init__ | the Triton branch's twin of core/module/autoregressive/samplers.py, which imports torch at module level (R33) |
| wrapper | src/neurobrix/kernels/autotune_certify.py | log | certify's default log when no caller gives one: an unbuffered print, so an hours-long run is read while it runs |
| wrapper | src/neurobrix/triton/kv_cache.py | intercept(enable_gqa) | the SDPA signature the interceptor stands in for: a graph's call may carry the kwarg, and the interceptor must accept every argument of the op it replaces |
| wrapper | src/neurobrix/triton/replay.py | slab_refused | the allocator-interceptor protocol's hook: nbxtensor's slab allocator (nbxtensor/memory/slab.py) calls `slab_refused(reason)` on the interceptor in force |
| wrapper | src/neurobrix/triton/moe.py | _host_expert(_h) | the default argument binds the layer's host tables at definition — a closure capture, not a parameter a caller passes |
| wrapper | src/neurobrix/triton/generator.py | from_config | the factory interface every generator shares (core/module/autoregressive's generators and samplers have it): the flows construct through it |
| wrapper | src/neurobrix/triton/symbols.py | get | the symbol table's accessor (6 403 call sites): the bindings dict stays private, `is_bound` decides boundness |
| wrapper | src/neurobrix/triton/session.py | use_branch | the session is the flows' handle; the KV wrapper behind it is the session's to replace |
| wrapper | src/neurobrix/triton/sequential.py | bind_inputs | the interface every sequence implements (TritonSequence.bind_inputs, CompiledSequence.bind_inputs): the flows bind through it |
| wrapper | src/neurobrix/triton/kv_cache.py | replay_advance | the KV wrapper is the handle the Triton flows hold (triton/flow/*); the paged cache behind it is the wrapper's to replace — a caller reaching `.cache` would bind to that internal |
| wrapper | src/neurobrix/triton/kv_cache.py | replay_lens_restore | the KV wrapper is the handle the Triton flows hold (triton/flow/*); the paged cache behind it is the wrapper's to replace — a caller reaching `.cache` would bind to that internal |
| wrapper | src/neurobrix/triton/kv_cache.py | replay_lens_snapshot | the KV wrapper is the handle the Triton flows hold (triton/flow/*); the paged cache behind it is the wrapper's to replace — a caller reaching `.cache` would bind to that internal |
| wrapper | src/neurobrix/triton/kv_cache.py | replay_restore | the KV wrapper is the handle the Triton flows hold (triton/flow/*); the paged cache behind it is the wrapper's to replace — a caller reaching `.cache` would bind to that internal |
| wrapper | src/neurobrix/triton/kv_cache.py | skip_positions | the KV wrapper is the handle the Triton flows hold (triton/flow/*); the paged cache behind it is the wrapper's to replace — a caller reaching `.cache` would bind to that internal |
| wrapper | src/neurobrix/triton/kv_cache.py | get_cache_len | the interface both branches' KV wrappers implement (core/runtime/graph/kv_cache_wrapper.py has its own), read by 8 call sites through either |
| wrapper | src/neurobrix/triton/symbols.py | UnboundSymbolError | a typed refusal of the engine's error contract (162 references): callers and tests name it, where a bare RuntimeError would name nothing |
| fact | src/neurobrix/kernels/metadata_ops.py | _native_layer_norm | the PyTorch branch's ATen handler for native_layer_norm (torch tensors in, torch tensors out); the scan matched the name of the library's cost declaration, not its work |
| fact | src/neurobrix/kernels/metadata_ops.py | _softmax | the PyTorch branch's ATen handler for _softmax (torch tensors in, torch tensors out); the scan matched the name of the library's cost declaration, not its work |
| fact | src/neurobrix/kernels/metadata_ops.py | _parse_dtype | the PyTorch branch's ATen handler for a call's dtype keyword (a torch.dtype) (torch tensors in, torch tensors out); the scan matched the name of the library's dtype parser, not its work |
| wrapper | src/neurobrix/kernels/autotune_certify.py | _contenders(factor) | a test seam: tests/unit/kernels/test_autotune_certify_windowed_oracle.py passes it to observe or replace what the run does |
| wrapper | src/neurobrix/kernels/autotune_certify.py | _release_between_keys(allocator) | a test seam: tests/unit/kernels/test_release_between_keys_drains.py passes it to observe or replace what the run does |
| wrapper | src/neurobrix/kernels/autotune_certify.py | certify(bench) | a test seam: the certifier's tests (a stub bench: no card) passes it to observe or replace what the run does |
| wrapper | src/neurobrix/kernels/autotune_certify.py | certify(log) | a test seam: the certifier's tests (the log read back) passes it to observe or replace what the run does |
| wrapper | src/neurobrix/kernels/autotune_certify.py | synthesize(values) | a test seam: tests/unit/kernels/test_a_square_flash_key_states_its_k_layout.py passes it to observe or replace what the run does |
| duplicate | src/neurobrix/triton/scheduler/ddim.py | _init_step_index | the per-class hook the census shadows (kernels/census.py replaces `_init_step_index` on every scheduler class that defines it in its own body): one copy per scheduler class, by design |
| duplicate | src/neurobrix/triton/scheduler/unipc_multistep.py | _init_step_index | the per-class hook the census shadows (kernels/census.py replaces `_init_step_index` on every scheduler class that defines it in its own body): one copy per scheduler class, by design |
| dead | src/neurobrix/triton/state_fence.py | MetalStateFence | the Metal release fence the day launches stop waiting for their kernel (docs/nbxtensor/BACKENDS.md, 'Releasing state pages on Metal'; decision (b') of 2026-10-29 21:35): today only its refusal test reaches it |
| fact | src/neurobrix/core/dtype/config.py | parse_dtype | the PyTorch branch's reader: it hands a torch.dtype (the library hands NBXDtype, never torch) and applies the engine's half remap under the compute dtype Prism chose — the remap is the engine's decision |
| fact | src/neurobrix/core/dtype/engine.py | storage_dtype | the DtypeEngine's torch.dtype face of the library's rule (`nbxtensor.dtype.itemsize.storage_dtype`), with the device's fp64 fact this engine holds: the rule is the library's, the torch dtype is the PyTorch branch's |
| wrapper | src/neurobrix/core/prism/runtime_widths.py | storage_dtype_name | the compiled rules' half of the storage-ceiling interface the Triton rules implement with their own fp64 fact: which ceiling a branch applies is the engine's decision |
| wrapper | src/neurobrix/core/prism/chunked_region.py | _Mix | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `Exception` would name nothing |
| wrapper | src/neurobrix/core/prism/profiler.py | _Declined | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `Exception` would name nothing |
| wrapper | src/neurobrix/core/prism/solver.py | PlanNotComputable | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/core/prism/temporal_stream.py | StreamDeclarationError | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/core/prism/time_estimate.py | MissingFact | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `KeyError` would name nothing |
| wrapper | src/neurobrix/core/runtime/engines.py | PyTorchBranchNotInstalledError | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/core/runtime/graph_executor.py | GraphStructureError | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `Exception` would name nothing |
| wrapper | src/neurobrix/core/runtime/shape_resolver.py | ShapeResolutionError | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `Exception` would name nothing |
| wrapper | src/neurobrix/core/runtime/symexpr.py | ExpressionError | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `ValueError` would name nothing |
| wrapper | src/neurobrix/core/runtime/symexpr.py | RealInShapeSlot | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `ExpressionError` would name nothing |
| wrapper | src/neurobrix/core/runtime/symexpr.py | UnknownExpression | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `ExpressionError` would name nothing |
| wrapper | src/neurobrix/core/runtime_values.py | MissingRuntimeValue | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/core/state/policy.py | StatePolicyRefused | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/core/state/price.py | StateUndeclared | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/kernels/autotune_certify.py | UnreachableCensusKey | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/kernels/autotune_certify.py | WitnessDrift | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
| wrapper | src/neurobrix/kernels/census_table.py | EmptyCensus | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `ValueError` would name nothing |
| fact | src/neurobrix/core/prism/plan_advice.py | what_would_have_fit | a name collision: the plan's advice over its component verdicts, where the library's answers an allocator refusal — two facts, one name |
| fact | src/neurobrix/core/prism/structure.py | memory_gb | a name collision: a profile device's raw memory in GB, where the library's is the rounded memory class a certified entry is keyed by |
| fact | src/neurobrix/core/prism/runtime_widths.py | _aligned | a name collision: whether an op's operands are dtype-aligned as its wrapper launches them, where the library's rounds a byte count to an alignment |
| wrapper | src/neurobrix/core/module/autoregressive/generator.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/autoregressive/samplers.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/scheduler/consistency/lcm.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/scheduler/diffusion/ddim.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/scheduler/diffusion/dpm_solver_pp.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/scheduler/diffusion/euler.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/scheduler/diffusion/unipc_multistep.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/scheduler/flow/flow_euler.py | from_config | a scheduler / sampler / generator protocol's constructor from its config, called by name on whichever class the container names |
| wrapper | src/neurobrix/core/module/tokenizer/factory.py | decode | the tokenizer protocol's decode, called by name on whichever tokenizer the container names |
| wrapper | src/neurobrix/core/module/tokenizer/factory.py | encode | the tokenizer protocol's encode, called by name on whichever tokenizer the container names |
| wrapper | src/neurobrix/core/module/tokenizer/json_bpe.py | token_to_id | the tokenizer protocol's token lookup, called by name on whichever tokenizer the container names |
| wrapper | src/neurobrix/core/module/tokenizer/sp_tokenizer.py | decode | the tokenizer protocol's decode, called by name on whichever tokenizer the container names |
| wrapper | src/neurobrix/core/module/tokenizer/sp_tokenizer.py | vocab_size | the tokenizer protocol's vocabulary size, called by name on whichever tokenizer the container names |
| wrapper | src/neurobrix/core/module/tokenizer/tiktoken_bpe.py | token_to_id | the tokenizer protocol's token lookup, called by name on whichever tokenizer the container names |
| wrapper | src/neurobrix/core/prism/solver.py | ProfileFactMissing | the refusal of a fact the profile lacks (rule 1) as its own type, so the device reading's handler (a reading that cannot be taken is a shared pool) lets it through instead of swallowing it |
| wrapper | src/neurobrix/core/prism/solver.py | get_allocation | the plan's lookup the flows ask through hasattr on any plan object |
| wrapper | src/neurobrix/core/runtime/graph/execution_context.py | get_tensor | the execution context's protocol, called by name on either branch's context |
| wrapper | src/neurobrix/core/runtime/graph/kv_cache_wrapper.py | get_cache_len | the KV cache wrapper's protocol shared with the Triton cache, called by name on either |
| wrapper | src/neurobrix/core/runtime/symint.py | __radd__ | Python's reflected-operator protocol: 2 + s calls it, never by name |
| wrapper | src/neurobrix/core/runtime/symint.py | __rmul__ | Python's reflected-operator protocol: 2 * s calls it, never by name |
| wrapper | src/neurobrix/core/runtime/symint.py | __str__ | Python's str() protocol |
| dead | src/neurobrix/nbx/cache.py | NBXCache.list_cached | used by the build toolchain, which reads the engine's package from its own repository (outside this scan's roots) |
| dead | src/neurobrix/nbx/neurotax.py | NeuroTaxParser.normalize_parent_module | used by the build toolchain, which reads the engine's package from its own repository (outside this scan's roots) |
| dead | src/neurobrix/serving/engine.py | InferenceEngine.is_loaded | used by the build toolchain, which reads the engine's package from its own repository (outside this scan's roots) |
| fact | src/neurobrix/core/runtime/graph/kv_cache_wrapper.py | _chunk_mask | the compiled engine's prefill-chunk mask as a torch tensor in the cache's dtype; nbxtensor's `paged_kv.chunk_mask` builds the same 0/-inf values as an NBXTensor, and nbxtensor never holds torch (R33) (lot A, 2026-11-08) |
