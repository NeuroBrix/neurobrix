# Prism scores its candidate plans by estimated time

Status: design, 2026-10-24. Step B of the activation architecture decided on 2026-10-21 (A: exact liveness peak;
B: candidates scored by time; C: region cuts; D: offload or recompute of a crossing tensor).

## What exists

`PrismSolver._solve_at_rung` already generates every candidate: each strategy of the cascade (`single_gpu`,
`single_gpu_lifecycle`, `component_placement`, `pipeline_parallel`, `block_scatter`, `weight_sharding`,
`component_placement_lazy`, `lazy_sequential`, `zero3`, `op_level_tiling`, `layer_streaming`, `cpu_execution`,
`cpu_streaming`) is tried and every one that fits is kept (`_evaluate_all_strategies`), then the guidance split, the
token-axis slicing and the parked values add their own. The winner is the highest `_score_strategy`: a base score per
strategy name (`BASE_SCORES`, 1000 for `single_gpu` down to 5 for `cpu_streaming`) minus penalties for device
boundaries, lazy loads and zero3's PCIe. Those numbers are a hand-made ranking, not a measurement: they cannot say
whether `zero3` on a fast host link beats `layer_streaming`, nor by how much, and a new strategy needs a new number.

## What B changes

The score becomes the plan's ESTIMATED TIME for the request, computed from what the plan does and from measured
facts of the hardware. The fastest candidate that fits wins. Nothing else moves: the candidates are the same
functions, the peak that decides "fits" is A's exact liveness, and a plan that runs today stays a candidate, so a model
can only change plan for one the estimate says is faster.

### 1. Facts, measured, on the hardware profile, each with its source

| fact | measured by | used for |
|---|---|---|
| device matmul throughput per compute dtype (FLOP/s) | a GEMM at a few sizes, the certified kernel | compute-bound ops |
| device memory bandwidth (B/s) | a device copy | memory-bound ops, weight reads |
| host -> device and device -> host bandwidth, pinned (B/s) | a pinned copy each way | zero3, lazy loads, parked values, offload |
| device <-> device bandwidth (B/s) | exists (`topology`) | boundaries of multi-device plans |
| host compute throughput (FLOP/s) | a host GEMM | `cpu_execution` |
| storage read bandwidth (B/s) | a read of a weight file | `cpu_streaming`, cold loads |

On unified memory the host <-> device figures are the profile's own (no copy: the same pool), so a plan that "moves"
weights pays what the profile says it pays. A missing fact is refused by name, never defaulted.

### 2. Work per component, from the graph at the request

FLOPs and bytes of each component are read from its graph with the request's shapes — the same symbolic resolution the
peak uses: matmul / linear / conv / attention FLOPs from their operands, bytes from weights read once per use plus the
activations each op reads and writes. Uses per request come from the flow (denoising steps, decode tokens, codec
frames), the same counts the request already carries.

### 3. Time of a plan

For each component use: `max(FLOPs / throughput(dtype), bytes / memory_bandwidth)` on its device (a roofline), plus
what the strategy moves: weight bytes over the host link per load (lazy, zero3, layer streaming per segment), parked
values out and back, activations across device boundaries, the halo recompute of a tiled op, the host's own compute for
host placements. The sum over the request is the plan's time; the plan records it (`estimated_s`) beside its peak, and
`--explain-plan` prints every candidate's time and peak.

### 4. Rollout without breaking a running model

1. Facts tool and facts on the rack's profile (V100 16 and 32 GB) and the Mac's (M4 Pro).
2. Work model and time per candidate, computed and printed beside today's score; the choice is unchanged.
3. Across the catalogue (derivation only, no run): every cell where the time's choice differs from today's is listed
   with both times.
4. Each listed cell is run under both plans on the card; the estimate's error is measured. The switch to time lands
   when the estimate ranks every measured pair correctly; a pair it ranks wrongly is a fact or work-model defect,
   fixed at its source, not a weight added to a score.
5. `BASE_SCORES` and the penalties are removed with the switch.
