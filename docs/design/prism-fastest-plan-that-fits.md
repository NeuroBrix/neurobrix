# The engine chooses right: the fastest plan that fits, and one precision contract

Status: design, 2026-10-25 (the owner's orders of 15:16 and 15:26, relayed 15:18 and 15:27). The measurement that
raised it: `registres/journaux/prism_choices_2026_10_25.tsv` (Dell, 294 plans from the entry point on 16 GB, 32 GB
and the 4-card NVLink profile) and `prism_choices_mac_2026_10_25.tsv` (Mac, 98 plans on the Metal profile).

## 1. What the literature teaches, and what we keep (fifteen lines)

1. **FlexGen** (arXiv 2303.06865): the latency of one layer is the MAX of the overlapped terms — compute, and each
   link's transfer (CPU↔GPU, disk↔CPU) — summed over layers, prefill and decode apart; the hardware constants
   (bandwidths, GEMM throughput) are fitted from a few profiled points. Kept: the shape of the cost and the measured
   constants. Rejected: placement by continuous percentages (we place whole components, blocks and shards) and the disk tier.
2. **Galvatron / Galvatron-BMW** (2211.13878, 2307.02031): communication = bytes / bandwidth of the link crossed;
   pipeline across the SLOW links first, tensor parallel inside the fast islands; compute from one profiled layer.
   Kept: islands by bandwidth and the per-boundary pricing. Rejected: training terms, power-of-two homogeneous devices
   (this rack mixes 16 and 32 GB).
3. **Alpa** (2201.12023): two levels — stages over sub-meshes (dynamic programming) and per-operator sharding (ILP),
   both pricing bytes / bandwidth of the mesh dimension crossed. Kept: the two levels are ours (component / block
   placement; weight sharding) and the resharding cost at a boundary. Rejected: an ILP per operator and compiling
   every candidate to profile it — we derive.
4. **vLLM**: fits one GPU → one GPU; fits one node → tensor parallel over the node's GPUs; else TP in the node × PP
   across nodes; without NVLink prefer pipeline parallel. Kept as the ORDER of ranks. **llama.cpp**: layer split by
   default, proportions by each card's memory (`--fit`), row split on request — kept: proportions by capacity.
5. **ZeRO-Inference**: streaming weights per layer is a throughput device for large batches; at batch 1 the fetch
   (bytes / host link) is not amortised, and latency work keeps the weights resident, across GPUs if needed. Kept:
   every plan that re-reads its weights per use ranks below every resident plan, and its cost is bytes / host link per use.

## 2. The rule of choice (Prism, both engines, one mechanism)

**(a) The budget is a fact of the profile.** A dedicated card: its capacity less the runtime's own context
(unchanged, `memory_budget.budget_mb`). A unified or shared pool: the profile's declared usable figure (the device's
`memory_mb`, on Apple `recommendedMaxWorkingSetSize`) less the pool's MEASURED host reservation (a `measured:` fact
written by `tools/profile_facts.py` on the idle machine) — never the free reading of the moment rounded onto the
ladder. The moment's reading becomes a DOOR at execution: free below the plan → refused by name (what holds the
memory), never a slower plan taken silently. The ladder survives for the census rungs (`NBX_PRISM_BUDGET_MB`).
The margin follows: `_margin_mb` hedged a volatile reading (max of 3 072 MB and 12 % of capacity); against a fixed
budget it becomes the measured allocator slack of the vendor file. Janus on the Mac: 692 MB of KV refused for a
3 072 MB hedge; the same plan on three runs is the test.

**(b) The order is physical.** Candidates are ranked by what they do with the weights, then by time inside a rank:

| rank | strategies | what it costs |
|---|---|---|
| 1 resident, one device | `single_gpu`, `single_gpu_lifecycle` | nothing moves |
| 2 resident, spread over a fast link | `component_placement`, `block_scatter`, `pipeline_parallel`, `weight_sharding` | activations (or partial sums) cross the link per use |
| 3 resident but the overflowing ops cut | `op_level_tiling` | the tile's halo and launches; the TilingEngine cuts ONLY what overflows (the owner, 10-21 and 10-23) |
| 4 weights streamed through the card per use | `zero3`, `expert_store` (component) | bytes / host link per use |
| 5 weights re-read per phase or per token | `component_placement_lazy`, `lazy_sequential`, `layer_streaming` | bytes / host link per phase or per layer per token |
| 6 host compute | `cpu_execution`, `cpu_streaming` | the host roofline |

The winner is the best candidate of the FIRST non-empty rank. Inside a rank the time estimate decides when the
profile carries measured facts (the estimate ranks, it no longer only says); without facts today's penalties
decide inside the rank only. The KV check stays where it is (a candidate that cannot hold its state is not a
candidate). Measured today: rank 3 is never chosen (0 of 294 plans) because `_try_op_level_tiling` asks the
detector only for upsample→conv pairs — Sana 4K (weights 11.4 GB, activations 23 GB) goes to streaming on every
profile; the detector must offer every op the runtime's tiling can intercept exactly (`tiling_exactness`).

**(c) The estimate prices the link.** `time_estimate.py` adds the device-to-device term: for each boundary a plan
crosses, the bytes crossing per use divided by `profile.topology.get_bandwidth(a, b)` (the profile's
`interconnect`, its `pcie_fallback` when the pair has no fast link — never a PCIe literal); `weight_sharding`
prices its partial sums per sliced GEMM per step. A profile with no bandwidth for a pair the plan crosses refuses
the estimate by name (`MissingFact`). On this rack: 4 cards full mesh at 300 GB/s, host link 8.1 GB/s measured.

**(d) Tests seen red first.** T1 a unified profile fixture: the plan is identical under free readings of
17 236, 16 902 and 15 000 MB (the door refuses the last at execution, the plan does not move). T2 a synthetic
component whose activation alone overflows and whose overflowing op the tiling can cut: `op_level_tiling` beats
`zero3`, `lazy_sequential`, `layer_streaming`. T3 the c4140 profile and a synthetic 60 GB routed LM: a spread
resident plan beats `lazy_sequential` (today Qwen3-Omni 65 GB, Wan2.2-I2V-A14B 66 GB and mochi 43 GB take
`lazy_sequential` on four cards whose memory holds them). T4 two spread plans over NVLink and over the PCIe fallback
differ by the bytes / bandwidth term in the estimate. T5 weights fit, KV 692 MB short of the flat hedge: a resident
plan with the measured margin.

## 3. One precision contract (the owner, 15:26)

On hardware that computes in bf16 (fp16 on V100) the model's tensors are half, weights and activations; fp32 only
where (a) the vendor traced the op fp32 inside a half graph (`core/dtype/traced_island`, 06:30), (b) the calibration
measured an overflow (`fp32_op_uids`), (c) a kernel accumulates in fp32 internally without storing it. The Triton
branch today keeps fp32 downstream of an island because its self-managed GEMMs and convolutions (`mm`, `bmm`,
`addmm`, `conv2d`, `_convolution` in `_SELF_MANAGED_OPS`) take their operands as they come and store wide: 2 017
policy sites on Sana's transformer (the Mac, 06:40), the compiled branch narrowing at the next AMP_FP16 op. The
rule: self-management covers the LAUNCH (in-kernel promotion, `launch_keys.mm_dtypes` and its kin), not the STORE —
under a half compute dtype `wrap_op` casts a self-managed op's non-island float operands to the compute dtype as it
does for every AMP_FP16 op, and the wrapper stores at the compute dtype; an island (traced or calibrated) keeps
fp32 as today. The width model (`runtime_widths._self_managed`, `_aligned`) and the key derivation follow the same
rule, the census is re-derived (the keys move from fp32 to half operands), the new keys certified. Proof: per-op
dtype of native and triton read from the plan dumps equal on V100 and on Metal, the Triton branch's peak lower,
outputs judged; test seen red: a half graph where an island's fp32 output feeds a non-island `mm` — Triton stores
fp32 today, the compute dtype after.

## 4. Order of work (one lot, `engine-chooses-right`, before the tag)

1. This design. 2. (a)+(b)+(c)+(d) in Prism and `time_estimate`, the tiling detector extended to the ops the
runtime can cut exactly, tests red then green. 3. The precision contract in `triton/dtype.py` and the wrappers'
store, the width model and the key derivation aligned, tests red then green. 4. Census re-derived by the frozen
tool, the new keys certified on both classes. 5. The gate (the scope replays what the change reaches — every cell:
a full gate), identity against lot52's base, R29 on what moved, landing. 6. The Mac validates the Metal cases
(Janus, Sana 4K, granite-speech; Sana, two LLMs, one video for the precision contract) and measures the peaks.

## 5. The gate on four cards (the owner's point 3)

The 16g cells run on the 16 GB cards when they are free (today they inherit by plan identity), four workers in
parallel; a class `4x-nvlink` for every model that does not fit one card (the MoE 30B, Qwen3-Omni, the Wan 14B),
launched by the entry point with the c4140 profile — the proof that Prism spreads, and the whole library under two
hours. Its layout is in `docs/design/interconnect-multigpu.md` §3; it follows this lot.
