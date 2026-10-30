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

## The scan

`tools/engine_slim_scan.py` (standard library, ast) lists four kinds of finding in `src/neurobrix`, both branches:

| kind | what it finds |
|---|---|
| `dead` | a definition nothing outside the tests refers to (by name, attribute, import, keyword, string or data file; registered by a decorator counts as used, a call to itself does not) |
| `duplicate` | a function whose normalised body (docstring dropped, parameters and locals renamed in order) is another engine or library function's |
| `wrapper` | a function that only hands its parameters on; a class that only renames its base; an alias of an imported name; a library name other engine modules import through an engine module (a re-export); a parameter no call site passes |
| `fact` | a definition carrying the name of a public nbxtensor fact function (`dtype`, `shape`, `memory`, the device facts, the `*_cost` declarations) or of a recorded synonym, and literal dtype-size tables |

Its rules are syntactic: a finding is a question a person answers, never a deletion on the scan's word alone.

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

## Moved into nbxtensor

| when | engine definition | nbxtensor API |
|---|---|---|
| 2026-10-30 | `traced_output_dtype_name` (triton/dtype.py), `traced_output_is_complex` and `amp_fp32_output_dtype` (twins in core/dtype/engine.py and triton/dtype.py, held equal by tests) | `nbxtensor.dtype.outputs` — one definition both engines import |

## Kept in the engine

| kind | path | symbol | why it stays |
|---|---|---|---|
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
