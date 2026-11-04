# The engine slims: what it carries that nbxtensor makes unnecessary

Hocine's decision (2026-10-29 22:39, precised 22:45): the engine imports nbxtensor and keeps only its own job — no
duplication, no dead function, no adaptation layer, no over-engineering — and its size can be measured. nbxtensor
replaces torch for the whole project: tensor work the engine still carries (an operation, a shape or stride
computation, a conversion, a copy, a reduction, a memory placement, a launch utility, a device probe) moves INTO the
library as a generic API, and the engine calls it. The mechanism goes to the library; the policy stays in the engine
(rule 3); a piece that mixes the two is cut in two.

## The figures

Two figures, kept apart (the supervisor, 2026-10-29 23:40):

- **Third-party code taken out of the engine**: the vendored reference tree `kernels/triton_kernels_ref/`, 262 132
  lines, never imported and never packaged — a census of Triton kernels written elsewhere, which inspired the engine's
  own; the engine now holds every kernel it needs, and the sources are on their own repositories (Hocine, 2026-10-30
  00:46: deleted outright).
- **The engine's own cleaning** — the figure that says whether the engine slims, tracked from here: the Python lines
  of `src/neurobrix` and the open findings, both recorded in the header of `engine_slim_list.tsv` at every commit.
  At the ratchet's first commit (2026-10-29, on nbxtensor 6c059035): 146 646 lines, 842 open findings.
  On 2026-11-02 (nbxtensor ff5c55a3): 137 016 lines, 48 open findings — every one in a zone held elsewhere: core/strategies
  (29), core/prism (13), triton/weight_loader and triton/sequence (the Dell's lots in flight), and the sampling contract
  (`audio_utils.sample_token(top_p)`, assigned to the Dell on 2026-11-02 06:40: one sampling path for both engines).

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

Open findings are listed in `docs/nbxtensor/engine_slim_list.tsv`, whose header records the engine's Python line
count and the number of open findings. `tests/unit/nbxtensor_boundary/test_the_engine_only_slims.py` holds the rules:

- a finding that is neither open in the list nor kept below fails — a new one was written: delete it, move it, or
  justify it here;
- a listed or kept finding the scan no longer finds fails — it was resolved: delete its line;
- the engine's line count and the number of open findings may only go down.

## Deleted

| when | what | lines |
|---|---|---|
| 2026-10-29 | `src/neurobrix/kernels/triton_kernels_ref/` — 1 097 files of vendored third-party reference kernels (flaggems for a dozen vendors, fla, attorch, gemlite, ...), imported by no code, excluded from both package targets — deleted outright: the sources live on their own repositories (Hocine, 2026-10-30) | 262 132 |
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

## Kept in the engine

| kind | path | symbol | why it stays |
|---|---|---|---|
| fact | src/neurobrix/core/strategies/base.py | transfer_tensor | the strategy's method takes a DEVICE STRING (cpu, cuda:N, mps:N) and passes a non-tensor value through; the library's function takes a device index and an NBXTensor only — a different contract, not a copy (2026-11-02) |
| fact | src/neurobrix/core/strategies/triton/base.py | transfer_tensor | the same method on the Triton branch: device strings, cpu, pass-through of non-tensors (2026-11-02) |
| fact | src/neurobrix/triton/sequence.py | _parse_dtype | parses through the library's parse_dtype, then remaps bf16<->fp16 to the plan's compute dtype (Prism's decision, the engine's) (2026-11-02) |
| wrapper | src/neurobrix/core/prism/layer_partition.py | hybrid(max_pieces) | the 64-piece bound has no source, but removing it changes a working plan into a slower one: Janus-Pro-7B native on the 16 GB profile then takes layer_streaming (80 of 91 pieces held, estimated 1 048 s) over lazy_sequential (estimated 1 062 s), and the gate's own cell measured 1 555 s against 223 s (2026-11-03, campaigns/2026_11_03_slim_constants). Prism's time estimate is what is wrong (lazy_sequential's 1 062 s against 223 s measured); the bound goes when the estimate is right (the list of what remains, 2026-11-03) |
| wrapper | src/neurobrix/core/strategies/base.py | synchronize_device | polymorphic: both branches' bases define it (the Triton one through DeviceAllocator), and the shared steps call it on self (2026-11-02) |
| wrapper | src/neurobrix/core/strategies/shared_steps.py | cleanup | polymorphic: the flows call `cleanup()` on every strategy, this one empties the loaded set (2026-11-02) |
| wrapper | src/neurobrix/core/strategies/layer_streaming.py | _nbx_path | a test seam: six streaming tests stub it to run card-less (2026-11-02) |
| duplicate | src/neurobrix/core/optim/passes/algebraic.py | _scalar | nbxtensor's twin is a function local to its residual-chain reader: the library exposes nothing to import, and the library never imports the engine |
| wrapper | src/neurobrix/core/cfg/engine.py | execute_component_with_cfg | the CFG engine's public entry, called by both iterative handlers; the Triton engine's converts its inputs at the boundary first, so the two are not one |
| wrapper | src/neurobrix/_nbxtensor_host.py | active | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| wrapper | src/neurobrix/_nbxtensor_host.py | record | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| wrapper | src/neurobrix/_nbxtensor_host.py | service | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| wrapper | src/neurobrix/_nbxtensor_host.py | shadow_run | a host service registered into nbxtensor.host: it imports its engine module when the library first calls it, not at registration (no engine import, no torch, at nbxtensor's load) |
| fact | src/neurobrix/triton/dtype.py | storage_dtype | already reads nbxtensor's itemsize.storage_dtype at this branch's fp64 fact and returns the NBXDtype member: the engine's dtype-engine method over the library's table |
| wrapper | src/neurobrix/kernels/census.py | install(hardware_profile) | the tests install the census for a profile they name (tests/unit/kernels/test_a_one_row_convolution_keys_its_width_on_the_ladder) |
| wrapper | src/neurobrix/kernels/census_table.py | rows_for(root) | the tests read a table under a temporary root (tests/unit/kernels/test_the_census_is_one_table) |
| wrapper | src/neurobrix/core/state/pages.py | writable(fence) | the tests write at a fence (tests/unit/core/test_the_state_pages_grow_share_and_release_on_fences): the copy-on-write rule it pins |
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
| wrapper | src/neurobrix/core/runtime_values.py | resolve(default) | passed, through an aliased import the scan does not follow: cli/commands/run._rt (`resolve as _resolve_runtime`, family= at every call, default= at six) and serving/engine (default=None) |
| wrapper | src/neurobrix/core/runtime_values.py | resolve(extra) | passed, through an aliased import the scan does not follow: cli/commands/run._rt (extra= at five calls) and serving/engine (the container's own height/width/VAE scale, the request's max_tokens) |
| wrapper | src/neurobrix/core/runtime_values.py | resolve(family) | passed, through an aliased import the scan does not follow: cli/commands/run._rt (`resolve as _resolve_runtime`, family= at every call, default= at six) and serving/engine (default=None) |
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
| dead | src/neurobrix/core/state/pages.py | PagePool.pending_pages | a test seam: read by the state-page tests (a page released only past its fence), nothing else |
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
| dead | src/neurobrix/triton/paged_kv_cache.py | PagedTritonKVCache.capacity_for | the engine's prompt-aware sizing policy (S1) stated as one function of the prompt's end: the card-less instrument its tests pin the rule with (tests/unit/runtime/test_kv_prompt_aware_sizing.py), where the real path needs a card |
| dead | src/neurobrix/triton/paged_kv_cache.py | PagedTritonKVCache.held_bytes | what the paged cache costs now, from its granted pages: the observable the page-release tests read (test_the_paged_kv_cache_decodes_as_its_oracle, test_the_imposed_rung_...) |
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
| wrapper | src/neurobrix/core/state/pages.py | StateExhausted | a typed refusal of the engine's error contract: callers catch it or a traceback names it, where a bare `RuntimeError` would name nothing |
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
| wrapper | src/neurobrix/core/prism/solver.py | get_allocation | the plan's lookup the flows ask through hasattr on any plan object |
| wrapper | src/neurobrix/core/runtime/graph/execution_context.py | get_tensor | the execution context's protocol, called by name on either branch's context |
| wrapper | src/neurobrix/core/runtime/graph/kv_cache_wrapper.py | get_cache_len | the KV cache wrapper's protocol shared with the Triton cache, called by name on either |
| wrapper | src/neurobrix/core/runtime/symint.py | __radd__ | Python's reflected-operator protocol: 2 + s calls it, never by name |
| wrapper | src/neurobrix/core/runtime/symint.py | __rmul__ | Python's reflected-operator protocol: 2 * s calls it, never by name |
| wrapper | src/neurobrix/core/runtime/symint.py | __str__ | Python's str() protocol |
| dead | src/neurobrix/nbx/cache.py | NBXCache.list_cached | used by the build toolchain, which reads the engine's package from its own repository (outside this scan's roots) |
| dead | src/neurobrix/nbx/neurotax.py | NeuroTaxParser.normalize_parent_module | used by the build toolchain, which reads the engine's package from its own repository (outside this scan's roots) |
| dead | src/neurobrix/serving/engine.py | InferenceEngine.is_loaded | used by the build toolchain, which reads the engine's package from its own repository (outside this scan's roots) |
