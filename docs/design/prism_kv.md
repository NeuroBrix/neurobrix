# Prism KV — a tiered, shared, warm KV cache, planned by Prism and run by one manager

Status: design, 2026-11-08, written on main 4cd7069e9. No code changes come with it. It extends
`docs/design/state-engine.md` (validated 2026-10-15) and does not replace it. The state engine's rules still hold: state
is declared by the container, it is priced in one place (`core/state/price.py`), it is held in fixed pages whose size is
derived, never a hand value, and only exact policies are taken without a measured quality cost. This document adds four
things to the KV case of that engine: tiers, prefix sharing, warmth across weight moves, and a runtime manager. It keeps
the Prism plan as the only place where decisions are made.

Paths are under `src/neurobrix/` unless written otherwise. Line numbers are at 4cd7069e9.

---

## 1. The current state (what exists, with file:line)

### 1.1 The plan: `KVCachePlan`

`core/prism/solver.py:319-340` holds these fields:

| field | meaning today |
|---|---|
| `max_cache_len` | positions per sequence; the ceiling Prism grants on the decoder's device |
| `num_layers`, `num_kv_heads`, `k_head_dim`, `v_head_dim` | the declared geometry (MLA: k = nope + rope) |
| `dtype` | the compute dtype the cache is held in |
| `memory_bytes` | `max_cache_len × per_token_bytes` |
| `per_token_bytes` | layers × kv_heads × (k + v) × itemsize × sequences (`core/state/price.py:44-49`) |
| `initial_cache_len` | serve plans only: the first request's size (`price.py:97`) |

What the plan does NOT carry:
- **No location.** The cache lives wherever the decoder's layers run (`_kv_devices`, `solver.py:965-975`). On a
  unified device, that includes the host.
- No tier, no host or disk capacity, no page size, no write policy, no prefix sharing, no rule for what happens to the
  cache while weights move.
- **A serialisation defect.** `initial_cache_len` is written by neither `to_dict` (`solver.py:613-622`) nor
  `from_dict` (`solver.py:679-691`). A plan read back from disk therefore loses the serve plan's initial size, and its
  `memory_bytes` makes a round trip through a float MB figure (`:690`).

### 1.2 Where the plan is sized

- **The size.** `_compute_kv_cache_plan` (`solver.py:5044-5070`) calls `core/state/price.kv_cache_size`
  (`price.py:75-98`). It takes as many positions as the remaining bytes allow, bounded by the request in run mode or
  by the declared window in serve mode. Below the minimum (`kv_min_tokens`, `price.py:57-62`: one turn, or two when
  serving) it refuses by name.
- **The room.** The per-strategy KV check (`solver.py:2941-3070`) computes "remaining" as the decoder's cards' capacity
  less what the strategy holds there:
  - `single_gpu` and `single_gpu_lifecycle`: the hot/cold sum or max, with the KV estimate taken out of the decoder's
    figures (`_kv_stripped`, `:952-963`);
  - `zero3`: activations only, because the weights sit on the host;
  - `layer_streaming`: the cut's window less its KV reserve (`_kv_reserve_beside`, `:9232-9256`);
  - on-demand strategies: the flow's dearest phase, because the cache outlives every phase (`_phase_peak`, `:1102`).
- **When the whole cache does not fit**, `_state_policy_for` (`solver.py:5024-5042`) scores `pages`, `quantize_kv` and
  `offload` (offload on a discrete card only). Every non-`pages` decision then rejects the strategy:
  "policy … has no runtime yet" (`solver.py:3063-3064`). **No host tier exists at run time.**

### 1.3 Pages (state engine step 2, built)

- **Bookkeeping, shared and torch-free.** `core/state/pages.py`:
  - `PagePool` (`:26-105`): reference counts and a fenced release queue;
  - `PageTable` (`:108-170`), with `fork` (`:131`) and copy-on-write `writable` (`:141`);
  - `block_table` (`:173-181`).
- **The page size is derived.** `price.page_tokens` (`price.py:101-124`) starts at the largest paged-read tile
  (`PAGED_DECODE_TILE_CANDIDATES = (16, 32, 64)`, `nbxtensor/src/nbxtensor/ops/tiling_sizes.py:33`) and doubles it
  until one layer's page reaches the vendor granule. The granule is `memory.state_page_granule_bytes`:
  `config/vendors/nvidia/volta.yml:121`, `ampere.yml:75`, `apple/apple_m4_pro.yml:124`, `cpu/x86_64.yml:18`. Release
  follows `memory.async_release`.
- **Triton engine.** `triton/paged_kv_cache.py:33` `PagedTritonKVCache`:
  - per-layer K and V pools on the layer's own card (`:106-121`);
  - the block table and lengths sit in fixed device buffers, so recorded replays keep their addresses (`:182-206`);
  - growth reallocates the pools and retires replay plans by generation (`:150-180`);
  - **prefill appends at the current length** (`:227-245`), so a prefill that starts at an offset is already possible
    mechanically.
- **Compiled engine.** `core/runtime/graph/paged_kv_cache.py:23` `PagedKVCache` uses the same pools as torch tensors
  and gathers through the table on every step (`:95-120`). `core/module/cache/factory.py:100-105` raises its ceiling to
  `max_position_embeddings`.

### 1.4 Lifetime: every chat turn re-prefills the whole conversation, on both engines

| | compiled | Triton |
|---|---|---|
| Who owns the cache | a new wrapper per request: `StateCacheFactory.create` at `core/flow/autoregressive.py:872` | memoised on the executor for its lifetime: `triton/flow/autoregressive.py:748-752` |
| End of a request | `cleanup` calls `reset_for_new_sequence`, which calls `cache.clear()` (`core/flow/autoregressive.py:335-339`, `core/runtime/graph/kv_cache_wrapper.py:335-341`) | `kv_cache.clear()` at the next request (`triton/flow/autoregressive.py:760`). The pages go back to the pool and the contents are lost (`triton/paged_kv_cache.py:360-367`) |

The serving path:
- `serving/session.py:1-13` documents "V1 Strategy: Re-prefill entire history each turn".
- `serving/engine.py:388-440` (`chat`) renders the whole history through the chat template (`session.build_token_ids`,
  `:433`) and runs a full request (`_generate_from_token_ids`, `:350-386`). `complete` (`:442-475`, the agent path)
  does the same.
- Turn n therefore costs a prefill of the whole conversation, O(total tokens), however short the new message.

Weights across requests:
- Only eager plans keep weights between requests. `loading_mode == "eager"` (`serving/engine.py:183-187`,
  `solver.py:9866-9888`) holds only for `single_gpu`, `single_gpu_lifecycle`, `component_placement`,
  `pipeline_parallel`, `block_scatter` and `weight_sharding` (`core/prism/structure.py:169-176`).
- Every rank-4 and rank-6 strategy reloads its weights per request, and the KV cache is gone as well.

### 1.5 The hard-coded 4096 and its neighbours in `serving/session.py`

- `serving/session.py:29`: `self._max_context = int(defaults.get("max_length", 4096))`. Only `get_context_info`
  (`:177-188`) reads it, to report `max_context` and `usage_pct` to clients. A container without `max_length` reports
  a context of 4096 that nothing in the plan supports. The real ceiling, `kv_cache_plan.max_cache_len`, is already
  read by `serving/engine.py:489-497` for `ensure_fits`.
- In the same file, three silent fallbacks contradict ZERO FALLBACK:
  - `except Exception: pass` around the summarisation (`:103-106`);
  - truncation when the summary is empty (`:108-118`);
  - a "~4 chars per token" estimate behind a second `except Exception` (`:190-198`).

### 1.6 The prism strategies (the real list)

`STRATEGY_RANK`, `solver.py:808-818`:

| rank | strategies | what they do with the weights during a pass |
|---|---|---|
| 1 | `single_gpu`, `single_gpu_lifecycle` | resident on one device |
| 2 | `component_placement`, `block_scatter`, `pipeline_parallel`, `weight_sharding` | resident, spread across cards |
| 3 | `op_level_tiling` | resident; the overflowing ops are cut |
| 4 | `zero3`, `component_placement_lazy`, `lazy_sequential`, `layer_streaming` | **cross the host link again**: per block, per phase, per layer per token |
| 6 | `cpu_execution`, `cpu_streaming` | computed on the host (compiled), or weights on the host with compute on the card (Triton with a card) |

---

## 2. What the literature teaches, and what maps onto NeuroBrix

All four projects are MIT or Apache 2.0. Ideas are taken. Code is not.

1. **vLLM PagedAttention.**
   [arXiv 2309.06180](https://arxiv.org/abs/2309.06180), [prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html).
   - The idea: KV lives in fixed blocks mapped by a per-sequence block table. Blocks are reference-counted and shared
     with copy-on-write. Prefix caching hashes each full block (its parent's hash, its tokens and extra keys such as
     multimodal hashes) and evicts the least recently used block from a free queue.
   - **Already ours** (step 2 of the state engine): fixed pages, block table, reference counts, copy-on-write.
   - **Kept**: the extra keys. Anything other than token ids that changes the KV must enter the key (§3.5).
   - **Rejected**: a block size chosen by hand. Ours is derived (`price.page_tokens`).
2. **SGLang RadixAttention and HiCache.**
   [arXiv 2312.07104](https://arxiv.org/abs/2312.07104), [HiCache blog](https://www.sglang.io/blog/hicache),
   [HiCache best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices).
   - The idea: a radix tree of token prefixes over a paged pool, with LRU on the leaves and reference counts.
   - HiCache adds tiers: GPU memory, host memory and shared storage. A tree node keeps its host copy when its device
     pages are evicted, so a later hit is reloaded instead of recomputed. Loading layer N+1 overlaps computing layer N.
   - Three write policies decide when the device tier is copied to the host:
     - `write_through`: every cached page also gets a host copy;
     - `write_through_selective`: only prefixes hit often enough are copied;
     - `write_back`: a page is copied only when it is about to be evicted, which puts the copy on the eviction path.
     The default and the hit threshold differ between versions of the docs; we take neither (see "Rejected").
   - **Kept**: the tree, the tiers, and the three policies by name.
   - **Rejected**: one-token pages. Our page is derived from the paged read's tiles, and that alignment is what makes
     the exactness contract provable (§5).
   - **Rejected**: a hand-set hit threshold. Ours is derived from estimated times (§3.4).
3. **Mooncake.**
   [arXiv 2407.00079](https://arxiv.org/abs/2407.00079), [FAST '25](https://www.usenix.org/system/files/fast25-qin.pdf).
   - The idea: a KVCache-centric architecture. The scheduler treats the KV cache as the resource to keep, and uses the
     cluster's otherwise idle host DRAM and SSD as a cache of KV ("trading more storage for less computation").
     Prefill and decode are disaggregated.
   - **Kept**: the inversion for our rank-4 strategies. The KV is what stays, and the weights are what transits. Under
     `layer_streaming` and `zero3` the weights already cross the link at every pass. The KV must never be what is lost
     between two turns (§4).
   - **Rejected for now**: prefill/decode disaggregation across nodes. Interconnect work comes after the stable
     release; a single machine is the one-node case of the state engine's per-node pools (§4.5 there).
4. **LMCache.**
   [arXiv 2510.09665](https://arxiv.org/abs/2510.09665),
   [local storage](https://docs.lmcache.ai/kv_cache/local_storage.html).
   - The idea: a KV layer outside the engine. It stores and loads in large chunks, not per layer per 16 tokens, which
     its paper finds is what lets it use the link's bandwidth. It spills host RAM to local disk, and a disk `prefetch`
     brings pages back to RAM ahead of use.
   - **Kept**: transfers in batches of pages, never one page per call; disk as the last tier, read back through the
     host; prefetch before the prefill that needs it.
   - **Rejected**: a separate service and default sizes such as "5 GB of CPU". Every capacity here is a profile fact
     (§3.2).
5. **Determinism under reuse.** This is the source the exactness contract rests on.
   [Thinking Machines, "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/),
   [SGLang deterministic inference](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/).
   - A cached prefix gives the same bytes as a full prefill only if every kernel on the path is invariant to how many
     rows it processes at once. That covers the matmul without an M-dependent split-K, the norms, and an attention
     with a fixed split size in place of a fixed number of splits.
   - The Thinking Machines post states that "handling the KV cache separately from the current KV values breaks batch
     invariance" through boundary conditions. The reduction order for a token must not depend on how many tokens of
     its sequence are processed with it.
   - SGLang reaches determinism with radix cache by aligning chunked-prefill truncation to the split size.
   - **Kept as the contract's mechanism (§5).**

---

## 3. The extended `KVCachePlan` (Prism decides; nothing here moves bytes)

The plan stays a dataclass in `core/prism/solver.py` and is computed by `core/state/price.py` (pure, torch-free). The
existing fields stay. New fields are grouped as follows. Every number carries the profile fact or container fact it
came from.

### 3.1 Geometry and pages (existing fields, plus the page)

`num_layers`, `num_kv_heads`, `k_head_dim`, `v_head_dim`, `dtype` and `per_token_bytes` keep their current meaning.

New:
- **`page`** = {`tokens`, `bytes_per_layer`, `derivation`}.
  - `tokens` comes from `price.page_tokens`, unchanged (state-engine §4.3: derived from the paged read's tile
    candidates and the vendor's `memory.state_page_granule_bytes`).
  - `derivation` names both inputs and their sources, and `--explain-plan` prints it.
  - One page size per node serves every tier, so a page moves between tiers as one unit of bytes and is never
    re-cut.

### 3.2 Tiers, each with a NAMED ceiling

`tiers` is an ordered list. Each entry has these fields:

| field | content |
|---|---|
| `tier` | `device`, `host` or `disk` |
| `where` | the device string (`cuda:i`, `mps`, `cpu`), or the engine data root for `disk` (from the workshop's tool roots configuration, never a path in code) |
| `role` | `active` for the pages the running sequence's attention reads, `retained` for the pages of prefixes kept for later turns |
| `pages` | the ceiling, in pages |
| `ceiling_bytes` | `pages × page bytes × layers on that device` |
| `reason` | one line that names every fact that decided the ceiling (below) |
| `link` | the measured bandwidth to the tier above: `h2d_gbps`/`d2h_gbps` from the vendor's `measured` block (e.g. `config/vendors/nvidia/volta.yml:180-186`), `storage_read_gbps` from the profile's `host_measured` block |

How each ceiling is derived (never a coefficient):
- **Device.**
  - The decoder's cards' rung, under the existing law: the profile's `runtime_total_mb` less `runtime_context_mb`,
    then the measured allocator reserves.
  - Less what the strategy holds there at its dearest phase. This is today's KV check (`solver.py:2941-3070`),
    unchanged.
  - Per card for a split decoder: each card holds the pages of its own layers (state-engine §4.5).
  - Example reason: `device cuda:0: 3 412 MB = rung 14 321 - decode phase 10 909 (single_gpu)`.
- **Host.**
  - The host budget `_host_budget_mb` (`solver.py:5266`), read from the profile's `cpu.ram_mb`.
  - Less what the plan holds on the host:
    - the pinned weights of `zero3`;
    - the expert store's host part;
    - the components of `cpu_execution` and `cpu_streaming`;
    - the runtime base (`cpu.runtime_base_mb`).
  - On a unified device the host tier is not a separate tier. It is the same pool (`_kv_devices` already says so), so
    the plan lists one tier and states why.
  - Pinned memory has no limit declared in the profile. Pinned is used up to the host ceiling; see Q4.
- **Disk.**
  - The profile carries `storage_read_gbps` (e.g. `config/hardware/v100-16g.yml:11`), but no write bandwidth and no
    capacity granted to the engine.
  - A disk tier exists only when the profile declares both: proposed `storage.kv_spill_bytes` and
    `storage_write_gbps`, measured by the same tool that measures `storage_read_gbps`.
  - Without them, the plan lists no disk tier and says so by name ("disk: not planned — the profile declares no
    storage.kv_spill_bytes"). It never reads the free space on the disk (engine rule 1).

The per-sequence ceiling `max_cache_len` stays. It now means "positions the ACTIVE sequence may hold". It is the
smallest of three bounds:
- the window the container declares (`state` entry `window`);
- the active tier's pages (the device tier, or the host tier under `offload`);
- the request (run mode).

The factory's "raise to the window" (`core/module/cache/factory.py:100-105`) goes. The ceiling is the plan's, and the
plan includes the window.

### 3.3 Prefix sharing

`prefix` = {`enabled`, `granularity`, `namespace`, `reuse_decode_kv`, `proof`}:
- **`granularity`**: whole pages. A radix edge always ends on a page boundary. A partial last page is never shared; it
  is copied on write (`PageTable.writable`).
- **`namespace`**: what must match for two prefixes to share bytes:
  - the container's sha;
  - the engine mode (compiled or triton): the two engines' bytes differ, and a page from one is never read by the
    other;
  - the KV dtype;
  - the hardware memory class: the certified directory's entries are per class, and so are the judged bytes;
  - the certified directory's revision for the KV-producing kernels;
  - the extra keys: a hash of every non-token input that the KV depends on (image or audio features of a VLM prefix,
    an adapter if one ever exists), computed by the same rule vLLM uses for its multimodal hashes.
- **`reuse_decode_kv`**: whether positions written by DECODE (M = 1) may be matched, or only positions written by a
  prefill. Prism decides it from the proof (§5).
- **`proof`**: `certified` or `unproven`, with the certificate's id. While it is `unproven`, `enabled` is false and the
  plan states why ("prefix reuse off: the KV-producing kernels of this plan are not certified row-invariant").

### 3.4 Write policy

`write_policy` = {`name`, `reason`}, with `name` one of `write_through`, `write_through_selective` or `write_back`.
Every policy is exact: a copy is bitwise. The choice therefore changes only time, and Prism takes it from the time
estimate (`core/prism/time_estimate.py`), as it does for strategies.

The costs:
- **`write_through`** costs `bytes_per_token / d2h_gbps` per generated token, on a copy that overlaps the next step.
  It is free when that is below the estimated time of a decode step. It is never free when the host link already
  carries the weights of a rank-4 strategy (§4).
- **`write_back`** puts the copy on the eviction path. It costs `evicted_bytes / d2h_gbps` at the moment pages are
  needed.
- **`write_through_selective`** copies a retained prefix when its expected saving exceeds the copy:
  `hits × est_prefill_time(prefix tokens) > copy_time(prefix bytes) + reload_time(prefix bytes)`. The "hit threshold"
  is that inequality, solved from the estimates, never a constant.

With no host tier (a unified device with one pool, or a profile with no host room left), the policy is `none` and the
plan states the reason.

### 3.5 Eviction

- Eviction is LRU over unreferenced radix leaves, tier by tier: device to host to disk to dropped.
- It never changes an output. An evicted prefix is reloaded, or it is recomputed by a prefill, which is the same bytes
  under the contract of §5.
- The plan states the order and the tiers it crosses. There is no tunable.
- Busy machine (engine rule 7): when the memory actually free at execution is below the plan's device ceiling, Prism
  re-plans on what is free and warns. The manager then evicts the retained pages first. Only when the active
  sequence's minimum no longer fits does the refusal come, from `PrismSolver._fail_error`. Pages never quietly move
  onto a host that is swapping: the host ceiling is the plan's, and it is never exceeded.

### 3.6 Warmth

`warm` = {`rule`, `reason`} says what happens to the device pages between two passes or two turns while the
strategy's weights move. The rule takes one of three values:
- `keep`: the pages stay on the device;
- `demote_on_phase_exit`: at the end of the decoder's phase, the pages are copied to the host tier, then prefetched at
  the next prefill, overlapping the weight load;
- `host_resident`: the active tier is the host.

§4 gives the rule per strategy.

---

## 4. Integration with every strategy in `STRATEGY_RANK`

`STRATEGY_RANK` (`core/prism/solver.py:808-818`) holds 13 strategy names; the table below has one row for each.

"Warm" means the KV cache is kept while the weights are re-streamed: turn n+1 finds turn n's pages without
recomputing them.

| strategy | where the pages live | device ceiling (what it is held against) | warm rule | link sharing |
|---|---|---|---|---|
| `single_gpu` | the card | rung less hot weights and peak activations (serve) or the cold peak (run) | `keep` | host link idle: `write_through` is free if the copy fits a step |
| `single_gpu_lifecycle` | the card | rung less persistent weights, the largest transient and its activations | `keep`; retained pages are evicted before a transient component would not fit | as above |
| `component_placement` | the decoder's card | that card's rung less its components | `keep` | as above |
| `block_scatter`, `pipeline_parallel` | each card holds the pages of its own layers (already true: `triton/paged_kv_cache.py:13-17`) | per card; the sequence ceiling is the smallest across the decoder's cards (no sum across cards: the open defect state-engine §1 names, closed by per-node pricing) | `keep` | each card's own link |
| `weight_sharding` | if attention heads are sharded, each card holds its head share (state-engine §3.3 point 6); if not, the decoder's card | per card | `keep` | to verify at step 2: how `weight_sharding` splits attention today |
| `op_level_tiling` | the card | rung less the tiled peak. The KV is never cut: attention's keys span the sequence, a global op, so no exact cut runs along it (the exact-cuts rule) | `keep` | as above |
| `zero3` | the card; the weights are on the host, pinned | activations only (today's check) | `keep`: the pages never leave the card between passes or turns | **shared**: the h2d link carries the blocks at every pass; host ceiling = host budget less the pinned weights; `write_back` preferred by the estimate, because d2h copies during streaming slow the next pass |
| `component_placement_lazy` | the decoder's group's card | that card's dearest phase, the cache outliving the decoder (`_phase_peak(outliving=)`) | `keep` when the dearest phase leaves room, else `demote_on_phase_exit` | the swap of phase weights uses the same link; the prefetch is scheduled after the decoder's weights |
| `lazy_sequential` | the decoder's card | as above; for an expert store, the store's grant and the KV are one subtraction (`solver.py:1196-1260`) | `keep` or `demote_on_phase_exit`, by the same test; a turn reloads the weights (lazy) but not the KV | idem |
| `layer_streaming` | the card, beside the segments; the reserve is `_kv_reserve_beside` (`solver.py:9232-9256`) | the cut's window less the segment peak; the reserve holds the active sequence (two turns when serving) | `keep` for the active reserve; retained prefixes beyond the reserve go to host only | **shared, and the heaviest**: every layer's weights cross per token. This is the Mooncake case: the KV is the resident thing. `write_back` only; the prefetch of a retained prefix runs before the first segment of the prefill |
| `cpu_execution` | compiled: the host (the K arrive on `cpu`), so device and host are one tier; Triton with a card: the card, the weights on the host | compiled: the host budget less the components; Triton: as `zero3` | compiled `host_resident`; Triton `keep` | compiled: no link; disk is the only lower tier |
| `cpu_streaming` | host | the host budget less the largest component | `host_resident`; the pages outlive each component's release | the disk tier matters most here: the host is the bottleneck, and the container's own reads use the same disk (`storage_read_gbps` shared) |

Three consequences in the code:
- **Session scope.** Under every strategy the KV pool belongs to the session, not to the executor. Today the Triton
  memo lives on the executor (`triton/flow/autoregressive.py:748`) and the compiled wrapper lives per request
  (`core/flow/autoregressive.py:872`). A lazy plan's executor that releases its weights must not release the pool. The
  pool is released only when the session ends, or by eviction.
- **Warm KV on lazy plans.** The serve path's "cold serving" (`serving/engine.py:228`) keeps reloading weights for lazy
  plans, which is correct. It no longer drops the KV.
- **Shared links.** The time estimate adds the KV traffic to the link it shares with the weight stream. Under rank-4
  strategies, the copy is never assumed free.

---

## 5. The contract: a cached turn gives the same bytes as a full prefill, on both engines

### 5.1 The statement

For a conversation of n turns, turn n served with the cached prefix gives byte-identical results to the same turn
fully re-prefilled from an empty cache: the same prompt token ids, the same seed and the same plan. "Byte-identical"
covers:
- the logits of the turn's prefill (the last position);
- the logits at every decode step;
- the generated token ids.

It holds on the compiled engine and on the Triton engine, each against itself. The two engines' bytes are not
compared with each other; the 4-mode oracle method covers that.

### 5.2 Why it is not free, and what makes it true

There are three places where the cached arm and the full arm compute differently.

1. **The prefix's K/V bytes.**
   - A cached position was produced by an earlier prefill of M1 rows, or by a decode with M = 1. In the full arm the
     same position comes from one prefill of M2 rows.
   - The projections, the norms and the MLP are row-wise ops whose kernels can choose a configuration by M class (a
     split-K, another BLOCK_K). If they do, the bits differ.
   - **Requirement R1 (row invariance)**: for the KV-producing path, each row's result depends only on that row and
     the weights, never on M.
2. **The suffix's attention over cached plus new keys.**
   - In the cached arm, a query at position i reads keys [0, P) from pages and keys [P, i] from the current chunk. In
     the full arm all keys come from one tensor.
   - **Requirement R2 (boundary invariance)**: the attention's reduction over keys follows the same tile order in both
     arms. That holds when the key tile size is fixed and the prefix boundary P falls on a tile boundary.
   - The page is the largest tile candidate times a power of two (`price.page_tokens`), and reuse is at page
     granularity (§3.3). **P is therefore always a multiple of every tile, which is exactly the alignment SGLang uses
     for chunked prefill.** The state engine's page derivation already gives this property; §3.3 forbids breaking it.
3. **Decode.** It is the same in both arms: the same length, the same block table contents (under R1 and R2), and the
   same split count. `paged_decode_splits` (`nbxtensor/src/nbxtensor/ops/paged_decode_attn.py:38-44`) is a function of
   the length alone: a fixed split size of 256 positions, with at most 64 splits. **Nothing to add.**

`reuse_decode_kv` (§3.3) follows from R1:
- If R1 is proven between M = 1 and the prefill classes, the reply tokens' decode-written KV may be matched.
- If not, only positions written by a prefill enter the radix tree. Turn n+1 then re-prefills the previous reply as
  part of its suffix, and still reuses the whole earlier prompt. This needs R1 only between prefill classes.

### 5.3 How R1 and R2 are established (certify once, never run a model to discover)

- **Triton engine (we own the kernels).** Row invariance becomes a certified property of the KV-producing kernel
  classes:
  - The certifier gains one check, run on synthetic tensors against itself: the same rows computed under every M
    class the census lists give the same bits.
  - A certified configuration that fails it (an M-dependent split-K) is marked `row_invariant: false`. The plan's
    `prefix.proof` is `certified` only when every KV-producing key of the plan is `row_invariant: true`.
  - R2 has its own probe on the paged prefill attention: a cached-plus-new read against a whole read, at every tile
    candidate.
  - The census already enumerates the keys, so this adds a column, not a pass.
- **Compiled engine (ATen and cuBLAS).**
  - We do not own the heuristics, and the Thinking Machines post shows that a default matmul is not batch-invariant.
  - The same probe measures the invariance of the plan's ATen ops on the certified shape classes of the profile.
  - A class that is not invariant makes the compiled plan's `prefix.proof` `unproven`, with the op named.
  - Whether the compiled engine may then carry `reuse_decode_kv = false` only, or must refuse reuse altogether, is
    Q1. R30 (mode universality) asks that the feature exists in both modes; the proof may differ.

### 5.4 The test (through the entry point)

The test is a two-arm run of the same conversation file (proposed CLI surface, Q5):
- Arm A, cached: turns 1..n served by one session.
- Arm B, the reference: turn n run alone from an empty cache, with the identical token ids that arm A's turn n
  rendered.
- Compared: the sha of the turn-n logits (prefill and each decode step) and the token ids. They must be equal.

How the test is seen failing:
- A deliberate injection: an M-dependent split-K forced into one projection key in a test directory must turn the row
  red.
- The register `docs/reference/vacuous-gates-register.md` receives the entry.

The models: one per declared KV kind, on both engines:
- a plain multi-head decoder;
- a grouped-query (GQA) decoder;
- a latent-KV (MLA) decoder;
- one VLM whose prefix holds an image, which checks the extra keys.

---

## 6. The runtime manager, separate from the plan

| | the plan (Prism) | the manager (execution) |
|---|---|---|
| where | `core/prism/solver.py`, `core/state/price.py`, `core/state/policy.py` | shared and torch-free: `core/state/kv_manager.py` (new), over `core/state/pages.py`; one data mover per engine |
| decides | tiers, ceilings and reasons, page size, write policy, prefix on or off, `reuse_decode_kv`, warm rule, eviction order | nothing; it applies the plan |
| owns | nothing at run time | the radix tree, the per-tier page pools and their ledgers, the reference counts, the LRU, the copy queue, the fences |
| refuses | when no plan fits (`_fail_error`) | by name, when a request would exceed a ceiling the plan set, e.g. `StateExhausted` (`pages.py:22`); never by enlarging a pool on its own |

The shared manager (torch-free, imported by both engines; R33):
- **Radix tree.** Nodes hold page-aligned runs of token ids under a namespace (§3.3).
- **Pages.** Each node holds, per tier, the page ids that carry it, and a reference count of the active sequences
  reading it.
- **Operations:**
  - `match(namespace, token_ids)` returns the longest page-aligned cached prefix, the tier it is in, and the pages.
  - `admit(sequence)` reserves the pages for the request's suffix and decode budget, evicting retained leaves if needed.
  - `promote(pages)` queues the copies from host or disk to the device, in batches, before the prefill that reads them.
  - `insert(sequence)` puts the finished sequence's prefill pages into the tree (and its decode pages too, if
    `reuse_decode_kv`). It applies the write policy and frees the rest.
  - `evict(bytes, tier)` moves LRU leaves one tier down, or drops them at the last tier.
- **Ledgers.** Each tier's held bytes are always equal to granted pages × page bytes. That is what the gate measures
  against the plan's ceiling.

The data movers, per engine (R33/R30: no shared compute code):
- **Triton.** Copies use `DeviceAllocator.memcpy`, as the growth path already does (`triton/paged_kv_cache.py:164-169`).
  It runs asynchronously on a copy stream, and the host side uses page-locked NBXTensor buffers. Promotion writes into
  pages whose addresses stay fixed, so recorded replays stay valid. The block table is rewritten in place under the
  existing fence (`_write_in_place`, `:182-190`).
- **Compiled.** torch copies on a side stream into a pinned host pool, with events for completion.
- **Disk (both engines).** Plain file I/O of raw page bytes under the engine data root. Each file has a header: the
  namespace key, the page geometry, the dtype, and a checksum.
  - A file whose header or checksum does not match is a MISS, logged by name, and the pages are recomputed. It is
    never read as data.
  - No vendor library is used to compute (R34).

How the session uses the manager (both engines):
- **The token-id history.** `ConversationSession` keeps the token ids the engine actually prefilled and generated, as
  well as the text.
- **Matching by ids, not by text.** At turn n+1 the chat template renders the history, and the manager matches the
  rendered ids. Whatever the template changes in earlier turns simply shortens the match, so nothing assumes the
  rendering is stable.
- **Prefill from the matched position.** The flow's prefill starts at the matched length: positions offset, pages read
  through the table. Both caches already append at their current length (`triton/paged_kv_cache.py:227-245`,
  `core/runtime/graph/paged_kv_cache.py:95-120`).
- **`complete()` (the agent path)** uses the same match, so a fixed system prompt and tools contract are prefilled
  once per session.

---

## 7. The 4096, named and replaced

- **The default.** `serving/session.py:29` defaults `max_length` to 4096. This default is to be removed.
- **What replaces it.** `ConversationSession` receives the plan's ceiling at construction: the active sequence's
  `max_cache_len` from the session's `KVCachePlan`, which `serving/engine.py:489-497` already reads. `get_context_info`
  reports that ceiling and the tiers' retained bytes. A session built without a plan is refused by name, as
  `_get_max_cache_len` already refuses.
- **The neighbouring fallbacks are removed in the same step:**
  - the `except Exception: pass` around summarisation (`:103-106`);
  - the truncation fallback (`:108-118`);
  - the chars/4 estimate (`:190-198`): the count is `len(build_token_ids())`, and a tokenizer that cannot render is
    refused.
- **Summarisation changes the conversation**, and so the output. It becomes a stated context policy, or a refusal by
  name when the conversation exceeds the ceiling (Q3).

---

## 8. Measurements and gate rows

Every cell writes these to the register (engine rule 3; the gate is the benchmark):

| measure | definition |
|---|---|
| `ttft_turn2_cached` | time to the first token of turn 2, prefix on the device tier |
| `ttft_turn2_full` | the same turn re-prefilled from empty (the reference arm) |
| `ttft_after_host_evict` | turn 2 after its prefix was evicted to the host tier |
| `ttft_after_disk_evict` | the same from the disk tier (where planned) |
| `prefill_tokens_computed` / `prefill_tokens_matched` | from the run-timings ledger (`core/runtime/run_timings.py`) |
| `kv_bytes_by_tier` | each tier's peak held bytes, equal to granted pages × page bytes, and its ceiling from the plan |
| `kv_copy_bytes` | d2h, h2d and disk bytes moved, with the estimate beside |
| `estimate` | the plan's estimated time for each of the above |

Gate rows (per family with declared KV; both engines; the hardware profile of the machine; certified-only):
1. **Exactness.**
   - The rows: a 2-turn conversation for each of the four models of §5.4.
   - Green: the cached arm's turn-2 logits sha and token ids equal the full arm's.
   - References are taken within one memory class.
2. **Speed.**
   - The row: `ttft_turn2_cached` against `ttft_turn2_full`.
   - Red if the cached arm is slower than the full arm.
   - Red if it is slower than its own last green by more than 1.5x (engine rule 3).
3. **Eviction.**
   - The rows: the same conversation under the existing budget door (a declared smaller rung, never an env var
     invented for this) so that the prefix must leave the device.
   - Green: the sha equals rows 1, and `ttft_after_host_evict` lies between `ttft_turn2_cached` and
     `ttft_turn2_full`. If it does not, the plan should not have retained the prefix, and the estimate is the defect.
4. **Warm under weight moves.**
   - The rows: one model forced by the door into each of `zero3`, `lazy_sequential`, `layer_streaming` and
     `cpu_execution`.
   - Green: the turn-2 sha equals the full arm, and `prefill_tokens_computed` equals only the suffix.
5. **Ceilings.**
   - The rows: no tier's held bytes exceed its ceiling.
   - Green: a profile without `storage.kv_spill_bytes` plans no disk tier, and `--explain-plan` says why.

The scope rule applies (engine rule 4). Only the cells whose plan, keys or code the change touches are replayed. The
cardless coverage check runs first: the plans explained, and the row-invariance column present in the certified store
for every KV-producing key.

---

## 9. Steps, in order, each with what it changes, its test and an estimate

Estimates are focused engineering days, without gate wall time.

| # | step | what it changes | its test (seen red first) | estimate |
|---|---|---|---|---|
| 0 | Plan hygiene | serialise `initial_cache_len` and integer `memory_bytes` (`solver.py:613-622`, `:679-691`); remove the 4096 and the three session fallbacks (§7) | plan round-trip equality test; a session without a plan refused by name; a container without `max_length` reports the plan's ceiling | 0.5 |
| 1 | Row and boundary invariance, certified | the certifier's row-invariance check and the R2 probe; a `row_invariant` column on KV-producing keys; the compiled engine's ATen probe on the profile's classes | the injected M-dependent split-K turns its key `false`; the probe on a known-invariant kernel stays `true` | 4 (+ certification time, one pass over the KV-producing keys) |
| 2 | Extended `KVCachePlan`, plan only | tiers, ceilings with reasons, page record, write policy, prefix decision, warm rule per strategy; `--explain-plan` prints them; no runtime change | per-strategy plan tests for the 13 strategies, each reason naming its facts; a missing profile fact refuses or omits the tier by name; run-mode plans unchanged except the new fields | 3 |
| 3 | The shared manager | `core/state/kv_manager.py`: radix tree, per-tier ledgers, LRU, namespace, write-policy scheduling (torch-free) | unit: no referenced page is evicted; a namespace mismatch never matches; LRU order; the ledger equals pages × bytes; two sessions over one pool | 4 |
| 4 | Prefix reuse on the device tier, both engines | session-scoped pool; the session's token-id history; prefill from the matched length; `chat` and `complete` through the manager; the per-request `clear()` replaced by `insert` | gate row 1 (exactness) on the four models, both engines; Triton replay plans survive a reuse (same uid and generation); gate row 2 | 5 |
| 5 | Host tier | pinned host pools and the two data movers; the three write policies; eviction to the host and promotion back | gate row 3 (host); the copied bytes equal the moved pages × page bytes; the sha equals the device-tier arm | 4 |
| 6 | Warm across weight moves | the per-strategy warm rule; `demote_on_phase_exit` and prefetch; link sharing in the time estimate | gate row 4 on `zero3`, `lazy_sequential`, `layer_streaming` and `cpu_execution` via the budget door | 5 |
| 7 | Disk tier | the profile facts `storage.kv_spill_bytes` and `storage_write_gbps` (measured, sourced); page files with header and checksum; prefetch to the host | gate row 3 (disk); a corrupted header is a named miss that recomputes, with the same sha; gate row 5 | 3 |
| 8 | The `offload` policy's runtime | the active sequence read from the host tier per layer, double-buffered (state-engine §4.4); this removes "no runtime yet" (`solver.py:3063`) | a model whose cache does not fit the card runs exact; the sha equals a larger-rung run in the same memory class | 5 |
| 9 | Gate and registers | the measures of §8 written by every cell; the rows added to the scope | the gate run seen red on the step-1 injection, then green | 2 |

Total: about 35 days, plus certification and gate wall time. Steps 0-4 give the main gain: turn n no longer re-prefills
the conversation. Steps 5-7 add the tiers, and step 8 extends the cache beyond the card.

---

## 10. Questions the owner decides

1. **Compiled-engine exactness.** If cuBLAS is not row-invariant on a class of the profile, does the compiled engine
   reuse only prefill-written KV (`reuse_decode_kv = false`)? Or does it refuse prefix reuse by name for that plan,
   while the Triton engine reuses? R30 asks for the feature in both modes; the proof cannot be the same.
2. **The cost of row invariance on Triton.** Pinning a KV-producing GEMM to an M-invariant configuration can slow
   prefill at small M. Is that cost accepted in exchange for exact reuse? The cost is measured at step 1 before
   anything lands.
3. **Summarisation.** Is the session's summarisation of old turns kept as a stated, output-changing context policy? Or
   is a conversation past the ceiling refused by name?
4. **Host and disk facts.** May the profile declare `storage.kv_spill_bytes`, `storage_write_gbps` and a pinned-memory
   bound? These are new fields, measured by the profile tool, each with its source. Is disk spill wanted at all for
   the stable release?
5. **The entry-point surface for multi-turn validation.** The rule wants validation through `python -m neurobrix run`,
   which is single-shot. Options:
   - a conversation input (e.g. `--turns <file>` from `tests/assets/`), whose reference arm runs each final turn from
     an empty cache;
   - or the gate drives `neurobrix serve` and `neurobrix chat`.
6. **Persistence across processes.** Do disk pages survive a restart of the engine? The namespace makes that safe, but
   it creates a cache to keep coherent with the certified directory's revision. The alternative is to drop it at
   session end.
7. **Priority against the stable release.** Steps 0-4 change serving latency for every LLM. Steps 5-8 extend capacity.
   Which belong to the stable release?
