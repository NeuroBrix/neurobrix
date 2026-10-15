# The state engine — design (research and map; no code before the owner validates)

Ordered by the supervisor on 2026-10-15 17:25, validated by the owner: one sub-engine that owns all the memory a model
KEEPS between two steps. The transient activations of a step stay with Prism and the TilingEngine. Written on main
333bc9a7c. Dell: the engine map, CUDA and the proposal. Mac: the Metal, MLX and unified-memory section (§3), folded in as
written.

State, in this document, is:
- **(a)** the KV cache of autoregressive decoders (LLM, VLM, autoregressive TTS, DualAR);
- **(b)** constant cross-attention K/V and encoder outputs computed once and reused at every step (a diffusion's text
  condition, Whisper's encoder output);
- **(c)** recurrent states (RNNT, Mamba/SSM, streaming convolutions);
- **(d)** diffusion step caches (reuse between neighbouring steps).

---

## 1. Map of the engine today

Paths are under `src/neurobrix/`; line numbers are at main 333bc9a7c.

### Findings that shape the proposal

1. **Only the KV cache is priced.** Encoder outputs, recurrent state and the step cache hold memory that no Prism figure
   names: none of them appears in `core/prism/`.
2. **KV is priced where nothing is allocated.** VLM, audio_llm, tts_llm, DualAR and the omni speech talker re-run the
   whole context at every decode step, in both engines:
   - `core/flow/vlm.py:514,876,1380`, `triton/flow/vlm.py:579,910,1352`;
   - `core/flow/audio_llm.py:240-272`, `triton/flow/audio_llm.py:310`;
   - `core/flow/tts_llm.py:431-481`, `triton/flow/tts_llm.py:306-362`;
   - `core/flow/dual_ar.py:109,196-197`, `core/flow/speech.py:361`.

   Prism still prices a cache for them: through `has_kv_cache` in their family YAML, read at `core/prism/solver.py:1845-1852`
   and `:1872`, and through `lm_facts.decode_lm_component` (`core/runtime/lm_facts.py:75-82`). The cost is twofold:
   memory reserved for nothing, and decoding in O(n²).
3. **The runtime can allocate KV beyond the bytes priced.** The ceiling is raised to the model's context window:
   - compiled: `core/module/cache/factory.py:81-88`;
   - Triton: `triton/kv_cache.py:261-282` `_alloc_len_for`.
4. **There is no cross-attention K/V cache, no Mamba/SSM state, and no streaming-convolution state.**
   - Whisper's decoder re-projects the encoder states into K/V at every token: `core/flow/encoder_decoder.py:229`,
     `triton/flow/encoder_decoder.py:222`, classification at `core/flow/decoder_kv.py:70-80`.
   - Every diffusion transformer re-projects its constant text condition at every step.
5. **Every state container is written twice, once per branch, with different behaviour.** See the duplication list below.

### (a) KV cache

| | compiled | Triton |
|---|---|---|
| Container | `core/runtime/graph/kv_cache_wrapper.py:56-251` `KVCacheLayer`. `torch.zeros [B, kv_heads, len, head_dim]`; doubles at 80% (`_GROWTH_RATIO`, `:75`, `:185-188`) up to max_len; batch re-expanded lazily (`:225-246`) | `triton/kv_cache.py:20-201` `KVCacheLayer`. `NBXTensor.zeros` at max_len (`:33-37`); never grows in place; a layer too small at prefill is replaced and its replay plans are dropped (`:284-324`) |
| Sizing | `core/module/cache/factory.py:67-109` from `plan.kv_cache_plan`; the ceiling is raised to `max_position_embeddings` (`:81-88`) | `triton/flow/autoregressive.py:182-206` from the plan; the window and decode budget come from `:720-786`; `_alloc_len_for` can exceed max_cache_len |
| Lifetime | per session, dropped at close; buffers freed by garbage collection (`core/flow/autoregressive.py:336-338`) | memoized on the executor for its lifetime (`triton/flow/autoregressive.py:767-782`); no site frees it |
| CFG / negative branch | `kv_cache_wrapper.py:423` `new_branch` | `kv_cache.py:335-342,388-389` |
| Encoder-decoder (Whisper) | `core/flow/encoder_decoder.py:179-232`; dtype from `executor.dtype or "float16"` (`:206`), not from the plan | `triton/flow/encoder_decoder.py:175-224`; same dtype reading (`:200`) |

Shared, engine-neutral readers:
- `core/runtime/lm_facts.py`: `lm_config_of` `:15-39`, `decode_lm_component` `:54-83`, `decode_sequences` `:98-115`,
  `decoder_cache_facts` and `encoder_decoder_cache_from_plan` `:118-154`;
- `core/flow/decoder_kv.py:44-91`: self-attention told from cross-attention by dataflow, head geometry from the traced q;
- `serving/engine.py:480-488`: the history budget. Each turn re-prefills the whole conversation (`:424-425`).

The families' `kv_cache:` blocks (`enabled`, `strategy`, `injection_op`) are read by no code; only
`execution.has_kv_cache` is.

### Prism's KV pricing

| Function | Location | What it does |
|---|---|---|
| `_kv_per_token_bytes` | `solver.py:3984-3996` | layers × kv_heads × (k_dim + v_dim) × itemsize |
| `_estimate_kv_cache_bytes` | `solver.py:3963-3982` | (max_tokens + 128) × token bytes. Its docstring announces a serve branch the code lacks. |
| `_compute_kv_cache_plan` | `solver.py:4019-4093` | `min(remaining // token_bytes, bound)` |
| KV estimate added to the LM | `solver.py:3580-3583` | Added to the LM's activation bytes. It then passes through the ×1.2 attention correction (`:3586-3589`) and the Triton arena factor (`:1502-1517`); the check subtracts the unscaled estimate (`:855-856`, `:1042`). |
| Per-strategy check | `solver.py:2185-2400` | The source itself marks two defects as OPEN: the multi-card capacity sums the rungs of cards other than the LM's (`:2252-2258`), and the estimate subtracted from a MAX can come off the wrong component (`:2337-2344`). |
| `_kv_reserve_beside` | `solver.py:7013-7032,7206-7266` | layer_streaming's reserve |
| `_phase_peak(outliving=)` | `solver.py:909-927` | KV outlives the LM across phases |

The expert store's grant (`solver.py:989-1112`) charges the planned KV to the decoder it names. Unified placement
("Strategy 2.5", `:5686-5711`) carries the KV only inside the LM's activations.

### (b) Constant encoder outputs

- Whisper's encoder output is computed once per 30-second window and kept in the variable resolver until the request ends:
  `core/flow/encoder_decoder.py:131-162`, `triton/flow/encoder_decoder.py:128-161`.
- A diffusion's text-encoder outputs, with the CFG negative, are computed once and kept for the whole loop:
  `core/flow/iterative_process.py:210-414`, `triton/flow/iterative_process.py:317-517`, `core/flow/image_gen.py:142-169`.
- Neither is priced. `core/prism/flow_bindings.py:173-190` binds only the text axis's length.
- The cross-attention K/V projected from these outputs are recomputed at every step.

### (c) Recurrent state

- RNNT/TDT (Parakeet):
  - compiled: `torch.zeros(num_layers, 1, hidden)`, `core/flow/rnnt.py:366-370,416`;
  - Triton: numpy on the host, `triton/flow/rnnt.py:280-281,344-360`;
  - unpriced and duplicated.
- Mamba/SSM and streaming convolution state do not exist. VibeVoice re-decodes non-streaming at every step in place of the
  vendor's streaming caches (`core/flow/next_token_diffusion.py:7-17,36-40`).

### (d) Step caches

- One exists: `StepCache`, which reuses the whole prediction. It is opt-in (`defaults.step_cache`, `NBX_STEP_CACHE_*`).
  - `core/flow/step_cache.py:21-107` keeps one tensor.
  - `triton/flow/step_cache.py:21-110` keeps that tensor plus a numpy copy.
  - Unpriced, and an R30 mirror.
- No feature reuse inside the denoiser exists (DeepCache- or TeaCache-style).

### Duplicated across the two branches

- the KV containers and their growth rules;
- the ceiling rules: the factory's ceiling vs `_alloc_len_for`;
- lifetimes: per session vs memoized;
- the encoder-decoder cache builders;
- branch creation;
- `StepCache`;
- the RNNT state: torch vs numpy;
- the expert stores: torch slots `core/runtime/graph/expert_store.py:32-99` vs the NBXTensor arena `triton/expert_store.py:46-123`.

### Past defects from state sizing or pricing (git history)

- 2d8cd8043: serve budget mis-summed.
- f2ecd8caf: zero3 weights counted resident in the KV check.
- da2f17530: run-mode KV ignored the prompt.
- 15497e46e: Triton allocated the serve ceiling, +10.8 GB.
- 64dd753c3: VibeVoice re-prefilled every step.
- e820fed5c: layer_streaming ignored the resident KV.
- 3e27c6cf0: KV checked against the wrong budget.
- 51b5ea4d0: triton-sequential decoded without KV.
- e4af3d5c8: VibeVoice KV unplanned; Janus priced without its guidance batch.
- e8f3c5d0a: constant-sized fallback caches.
- 1e02ab2bb: Whisper's KV unpriced.
- c03d7dadb: `_needs_kv_cache` missed decode flows.
- f1cca1dcd: unified budget cut twice.
- 0cf52932f: KV check summed every component.
- 20d069686: compiled decode under layer_streaming without KV.
- 145d56bc8: zero3 residency ignored planned KV.
- 723af6fc0: served KV sized against the sum of rungs.
- 1daa03063: the store's graph constants unpriced.
- b69300bfe: decode flow planned without a KV plan.
- d364e04fe: the omni thinker's KV charged to no one.
- 9e1684d81: 6.3 GB of driver rounding in the expert store.
- 2bd3a2d84: MPS mapped pages kept counted.
- f034c7d65: drain cliff subtracted twice, store floor taken by the served cache.

**Twenty-three defects in seven weeks, every one in glue that sizes or prices state: none in a kernel.**

---

## 2. State of the art (sources read 2026-10-15)

- **vLLM PagedAttention and prefix caching.**
  [arXiv 2309.06180](https://arxiv.org/abs/2309.06180), [prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html).
  - KV lives in fixed blocks (16 tokens by default) mapped by a per-request block table.
  - Reference counts let parallel samples and beams share blocks, with copy-on-write.
  - Prefix caching hashes each full block (parent hash, tokens, extra keys) and evicts LRU from a free queue.
  - Existing systems use only 20.4-38.2% of their KV memory for real tokens; vLLM is near zero waste, at 2-4x throughput.
  - Block size 16-128 performs best; recompute costs at most 20% of a swap.
- **SGLang RadixAttention.** [arXiv 2312.07104](https://arxiv.org/abs/2312.07104).
  - A radix tree of token prefixes over a paged pool with one-token pages; LRU on leaves with reference counts.
  - Cache-aware scheduling; hit rates of 50-99%; overhead under 0.3% without reuse.
- **llama.cpp.**
  [completion README](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md),
  [PR 7412](https://github.com/ggml-org/llama.cpp/pull/7412), [issue 24485](https://github.com/ggml-org/llama.cpp/issues/24485).
  - K and V types are chosen separately: q8_0 is 8.5 bits per value, q4_0 is 4.5.
  - Perplexity: f16 6.2322; q8_0/q8_0 6.2344; q4_0/q4_0 6.4384. **K is more sensitive than V.**
  - A mismatched pair fell back silently to CPU attention, 25-45x slower: a policy taken without saying so.
  - Context shift keeps `--keep` tokens, drops half the rest and re-rotates RoPE on the cached K.
- **MLX / mlx-lm.** [cache.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py). Detailed in §3.
  - Growth in 256-token steps.
  - A rotating cache with 4 sinks.
  - A quantized cache (group 64, 8 or 4 bits).
  - `ArraysCache` for recurrent state.
  - A prompt cache saved to safetensors.
- **FlashInfer.** [KV layout](https://docs.flashinfer.ai/tutorials/kv_layout.html), [arXiv 2501.01005](https://arxiv.org/html/2501.01005).
  - CSR page tables (`indptr`, `indices`, `last_page_len`, int32).
  - Arbitrary page size; the engine owns allocation.
  - The latent-KV (MLA) layout has no head axis.
- **StreamingLLM.** [arXiv 2309.17453](https://arxiv.org/abs/2309.17453).
  - Keeps 4 sink tokens plus a rolling window; stable to 4M tokens; 22.2x faster than recomputing a sliding window.
  - It does not extend the context.
- **Eviction.** H2O [arXiv 2306.14048](https://arxiv.org/abs/2306.14048) keeps heavy hitters plus recent tokens.
  SnapKV [arXiv 2404.14469](https://arxiv.org/abs/2404.14469) compresses the prompt KV per head by an observation window:
  at 16K tokens, 3.6x speed and 8.2x memory.
- **KV quantization.**
  - KIVI [arXiv 2402.02750](https://arxiv.org/abs/2402.02750): 2-bit, K per channel and V per token, groups of 32, the last
    128 tokens kept in full precision; 2.6x memory.
  - KVQuant [arXiv 2401.18079](https://arxiv.org/abs/2401.18079): K quantized before RoPE, non-uniform types plus outliers;
    under 0.1 perplexity at 3 bits.
- **DeepSeek MLA.** [arXiv 2405.04434](https://arxiv.org/abs/2405.04434). The cache stores a latent of d_c + d_h^R per token
  per layer (576 for DeepSeek-V2, against 2·n_h·d_h for MHA): 93.3% less KV.
- **Diffusion.**
  - Cross-attention K and V depend only on the text, so computing them once per run is **exact**
    ([arXiv 2404.13766](https://arxiv.org/pdf/2404.13766)). This does not hold for joint attention (MM-DiT, Flux), where the
    text tokens attend to the image.
  - Step caches trade quality for speed and are never exact:
    - DeepCache [arXiv 2312.00858](https://arxiv.org/html/2312.00858): about 2.3-4.1x;
    - TeaCache [arXiv 2411.19108](https://arxiv.org/html/2411.19108): up to 4.41x at -0.07% VBench;
    - FasterCache [arXiv 2410.19355](https://arxiv.org/html/2410.19355): about 1.6x;
    - TGATE [arXiv 2404.02747](https://arxiv.org/abs/2404.02747): 10-50%.
- **SSM.** [mamba2.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py),
  [vLLM hybrid manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html).
  - The state is constant in sequence length: conv `[B, d, d_conv]` plus ssm `[B, heads, headdim, d_state]`.
  - vLLM forces one page size, padding the Mamba state up to an attention page (over 400 tokens, "too large").
  - Prefix caching then works only at block boundaries.
- **Whisper.** [model.py](https://github.com/openai/whisper/blob/main/whisper/model.py) computes the cross-attention keys and
  values "once and reuse[s them] in subsequent calls". Hugging Face's `EncoderDecoderCache` keeps two independent caches.

---

## 3. Metal, MLX and unified memory (the Mac's section)

Written 2026-10-15 by the Mac on main 333bc9a7c. The sources are the files themselves, fetched and read.

### 3.1 Where state lives on Metal today

On Apple silicon the device draws on the host's memory, so every byte of state below is also host memory.

| What | Where | Sizing today |
|---|---|---|
| Triton KV, one buffer pair per layer | `triton/kv_cache.py:34-37` | `NBXTensor.zeros((batch, kv_heads, max_len, head_dim))` at max_len, whole at the start |
| Compiled KV | `core/runtime/graph/kv_cache_wrapper.py:86-118` | `initial_len` grows up to max_len; `initial_len == 0` allocates max_len at once (legacy) |
| Prism's device plan on a unified device | `core/prism/host_footprint.py:59,114` | `device_bytes` added to the host figure under `has_unified_memory` |
| The working set Prism plans in | `kernels/metal_device.py:366` | `recommendedMaxWorkingSetSize()` (17 277 MB on the M4 Pro 24 GB) |
| How far the compiled MPS pool may grow | `core/runtime/torch_allocator.py:31,53`; `config/vendors/apple/apple_m4_pro.yml:232` | `compiled_allocator_memory_fraction: 1.0` |
| Attention scores the device writes | `config/vendors/apple/apple_m4_pro.yml:221` | `compiled_sdpa_full_scores: true`; the budget comes from the plan |

State-like defects on Metal last week:

- **Expert store growth outside the footprint.** Copies from a `torch.frombuffer` view kept the shard's pages, and swap grew
  6.8 GB. Fixed by staging plus `madvise` (2bd3a2d8).
- **The full attention scores on MPS.** The fused math path built the whole score matrix, adding 13 GB in one second at
  79 200 tokens. Fixed by chunking, waits and the pool bound (1d9b52dd1); the pieces got their budget in 6b8edbfd1.
- **Buffers in flight.** MPS reuses a freed buffer only after the device has run the command that used it. 672 chunks
  enqueued without a wait were all alive at once. On Metal, a "released" page is not free until the queue has drained past
  its last use.

### 3.2 State of the art on Apple silicon (read in source)

- **mlx-lm caches** (`mlx_lm/models/cache.py`, 1 741 lines):
  - `KVCache` (339-395) grows in 256-token steps;
  - `QuantizedKVCache` (251-337) uses `mx.quantize` with groups of 64 at 4 or 8 bits, and `to_quantized` converts in place;
  - `RotatingKVCache(max_size, keep)` (423) is a window plus sinks, with keep = 4 in `make_prompt_cache`;
  - `ArraysCache` (596) holds SSM state;
  - `ChunkedKVCache` (727) and `CacheList` (803) bound and compose caches;
  - `BatchKVCache` / `BatchRotatingKVCache` (880, 1101) serve batched decoding;
  - `LRUPromptCache` (1599) is a prefix trie with LRU eviction.
- **MLA on Metal** (`deepseek_v3.py`, `kv_lora_rank = 512`). The cache stores the latent plus its RoPE part, not per-head
  K/V. The container must declare it, and its size follows from `kv_lora_rank + qk_rope_head_dim`.
- **Mac clusters** (MLX `distributed.rst`):
  - Backends: MPI; RING over TCP; JACCL, RDMA over Thunderbolt 5 from macOS 26.2, "an order of magnitude lower" latency,
    needed for tensor parallelism; NCCL.
  - `mlx_lm/utils.py:696` `sharded_load` combines tensor parallelism and pipeline parallelism.
  - A pipeline-parallel model's KV stays on the node that owns its layers. Under tensor parallelism the KV heads are split
    across nodes.

### 3.3 What the state engine must hold on Metal

1. **One pool with the device.** State pages and Prism's activation budget come from the same working set and are priced
   once.
2. **Grow by pages, never allocate max_len at once.** Pages are sized from the profile's rung.
3. **Release is a queue event.** A page is reusable once the command buffer that last read it has completed. The engine
   tracks the last use per page, or fences a group of pages.
4. **Quantized KV and rotating windows are policies, chosen by score and stated in the plan.** On a 24 GB Mac they decide
   whether a 30B MoE's long context fits at all.
5. **Declared state types.** KV, latent KV, recurrent arrays, cross K/V and step caches are each sized from the container's
   declaration and the profile.
6. **Clusters as nodes and links, prepared now without a special case.**
   - Each node has its own working set and state pool; each link has its bandwidth and latency (RING/TCP,
     JACCL/RDMA over Thunderbolt 5).
   - A single Mac is the one-node case.
   - State follows its layers: a pipeline split keeps each node's own layers' KV; a tensor split gives each node its share
     of KV heads.
   - The price is per node.

### 3.4 What it replaces on Metal

Both whole-`max_len` allocations give way to paged growth. The KV term inside the unified device plan becomes the state
engine's own figure, read once by Prism. Nothing is added beside them: the two allocators go.

---

## 4. Proposal: one state engine

### 4.1 Declared state, sized by arithmetic

Forge writes a `state` section into `topology.json`, one entry per state the model keeps, derived from the graph and the
embedded config. No entry is ever keyed by a model name.

| kind | declared by | size per unit |
|---|---|---|
| `kv` | each self-attention whose K/V the decode step reads back | layers × kv_heads × (k_dim + v_dim) × dtype bytes, per token per sequence |
| `latent_kv` | an attention whose cache is a latent (MLA) | layers × (d_c + d_rope) × dtype bytes, per token per sequence |
| `cross_kv` | a cross-attention whose K/V depend only on loop-invariant inputs (the text condition, the encoder output) | layers × heads × (k_dim + v_dim) × dtype bytes × context length, per sequence, once per request |
| `invariant` | any other loop-invariant subgraph output the loop reads every step (its value does not depend on the step) | that tensor's bytes, once per request |
| `recurrent` | a state carried from one step's output to the next step's input (RNNT h/c, SSM conv and ssm states) | the declared shape × dtype bytes, constant |
| `step_cache` | a policy, not a declaration: the denoiser's output or residual kept to skip a step | one prediction's bytes |

`cross_kv` and `invariant` come from one generic analysis: a loop component's ops whose inputs are all loop-invariant (the
same at every step) are invariant. Forge marks them; the runtime evaluates them once and keeps their outputs as state. Its
exactness is in the definition, and in the cross-attention case it is sourced (arXiv 2404.13766). It does not apply to joint
attention, where the text tokens attend to the image: those ops read a step-variant input, so the analysis never marks them.

### 4.2 One sizing function, Prism asks it

`state_engine.price(state_declarations, request, profile, plan_inputs) -> {node: bytes, pages, policy}` is the only place
state bytes are computed. It is pure arithmetic and torch-free, so Prism and both engines import it.

It replaces:
- `_kv_per_token_bytes`, `_estimate_kv_cache_bytes`, `_compute_kv_cache_plan` and `_kv_reserve_beside`;
- the KV estimate folded into the LM's activations, together with its ×1.2 and arena scaling;
- the two OPEN defects of `solver.py:2252-2258` and `:2337-2344`. The price is per node and per owner component, so nothing
  is summed across cards and nothing is subtracted from a maximum.

The plan states the state budget per component and per node, as it states the attention budget today
(`ComponentAllocation.attention_scores_budget_bytes`). An executor, or a piece of one, reads it from the plan and never
computes its own.

### 4.3 Pages

- State is held in fixed-size pages, allocated as the request grows and never at max_len.
- The page size is derived per profile from one vendor value, `memory.state_page_bytes`, with its source:
  `tokens_per_page = state_page_bytes // token_bytes`.
- A sequence holds a page table. Pages carry reference counts, so a CFG pair's shared prompt and beam prefixes share pages
  with copy-on-write (vLLM's mechanism).
- The KV kernels read through the page table. The Triton branch gets a paged-KV append and attention read, written once for
  every backend. The compiled branch gathers through the same table.
- A page is released as a queue event. On an asynchronous device (the vendor states `memory.async_release: true`; MPS does,
  per §3.1) a page returns to the pool only after a fence past its last use.

### 4.4 Policies, chosen by score and always stated

When the full-precision, whole-context state does not fit the node, the engine offers policies, scored like Prism's
strategies:
- `offload` (discrete cards: cold pages to pinned host);
- `quantize_kv` (K at 8 bits, V at 8 or 4, groups from the vendor file; K is more sensitive, per the sources);
- `window` (sinks plus a rolling window; StreamingLLM / mlx-lm);
- `refuse`, by name.

The plan prints the policy taken and why, with its quality cost from the cited sources. A policy is never taken silently
(llama.cpp's silent CPU fallback is the counter-example). `step_cache` stays opt-in, off by default, with its quality cost
stated.

### 4.5 One pool per node, both engines

Each node has one state pool: a torch allocation in the compiled engine and an NBXTensor arena in the Triton engine.
- The pool's capacity is the node's state budget from the plan.
- The page table, the policies and the release queue are shared code. Only the raw allocation and the kernels are per
  engine (R33: the Triton side imports no torch).
- The expert store's slots are weights, not state, but they come from the same per-node ledger, so the grant and the state
  price are one subtraction.

### 4.6 What each family gains

| family | today | with the state engine |
|---|---|---|
| LLM (autoregressive) | KV priced; allocated beyond the price up to the window | paged KV at the price; prefix sharing for serving turns, instead of a full re-prefill per turn (`serving/engine.py:424-425`) |
| VLM, audio_llm, tts_llm, DualAR, omni talker | KV priced but never allocated: every decode step re-runs the whole context, O(n²) | real KV: O(n) decode, the largest speed gain in the catalogue |
| image-AR (Janus), next-token diffusion (VibeVoice) | KV for two branches, allocated twice | one page table, shared prompt pages |
| Whisper (encoder-decoder) | cross K/V re-projected every token; KV dtype from the executor | `cross_kv` once per window; dtype from the declaration |
| diffusion DiTs with cross-attention (PixArt, Sana, Wan, CogVideoX-style) | text K/V re-projected every step, every layer | `cross_kv` / `invariant` once per request; exact |
| MM-DiT / joint attention (Flux, SD3-style) | nothing to gain from cross K/V | unchanged: the analysis marks no op |
| RNNT (Parakeet) | h/c twice, unpriced | `recurrent`, priced, one code |
| future SSM / hybrid | absent | `recurrent` pages sized independently of the attention page; no padding to an attention page |
| Mac clusters | absent | per-node pools; state follows its layers |

### 4.7 What it removes

The state engine is a replacement, not an addition. These go:
- `core/runtime/graph/kv_cache_wrapper.py` (`KVCacheLayer`, `DistributedKVCache`);
- `triton/kv_cache.py`;
- `core/module/cache/factory.py` (`StateCacheFactory`);
- `triton/flow/autoregressive.py` `session_kv_params` and the memo;
- both `_decoder_kv_*` builders;
- both `step_cache.py`;
- the two RNNT state paths;
- the Prism KV functions listed in §4.2;
- the families' dead `kv_cache:` blocks and the `has_kv_cache` flag (state is declared by the container, not assumed by a
  family).

### 4.8 Order of work and proofs

1. **Declarations and price.** Forge declares `state`, and `state_engine.price` replaces Prism's KV functions with no runtime
   change.
   - Proof: plans identical where the figure is unchanged, and named where it moves.
   - Proof: the open defects closed by cells seen red.
2. **The pools and pages replace both containers.**
   - Proof: decode tokens bit-identical to the oracle (5-10 tokens); the gate.
3. **VLM, audio_llm, tts_llm, DualAR decode on the cache.**
   - Proof: tokens equal to the full re-forward; the speed measured.
4. **`cross_kv` / `invariant`.**
   - Proof: diffusion outputs bit-identical, by exactness; Whisper transcripts equal.
5. **Policies.**
   - Proof: quality measured against the vendor at the same request, reported in the plan's terms.
6. **Nodes and links in Prism.** A single machine is the one-node case; no cluster profile exists yet.

### 4.9 Questions for the owner

1. Is `invariant` (Forge marking loop-invariant subgraphs) acceptable as a build-time annotation? It adds a section to
   `topology.json`, not a field patched to work around a bug (R18).
2. Is the page size one vendor value per architecture (`memory.state_page_bytes`), or derived from the rung alone?
3. Quantized KV and windows change outputs. Are they allowed in the stable release as stated, scored policies, or held
   until after it?
