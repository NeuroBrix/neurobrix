# Interconnect and multi-GPU: the inventory before the workstream

Status: inventory, 2026-10-25 (the owner's order of 14:01, relayed 14:02). Written on the CPU from the code, the
registers and the gate rows of main 99e68714; nothing here is implemented before the stable tag. The workstream
itself ("interconnection and multi-GPU") opens after the stable.

The owner's observation: every validation runs a model on ONE card (the gate gives one card per cell), while the Dell
rack carries four V100 in a full NVLink mesh (300 GB/s per link) and the hardware profile already states it
(`interconnect:` groups / type / tech / bandwidth, `pcie_fallback`); other machines carry QSFP meshes (Tenstorrent),
networks, PCIe 5. The code must never assume PCIe.

## 1. The strategies: what covers each one, and whether a real model ever ran it

The cascade's thirteen strategies are the keys of `BASE_SCORES` (`core/prism/solver.py`); `expert_store` is a
COMPONENT strategy, not a rung (`tests/unit/prism/test_solver_registry_parity.py::test_expert_store_is_a_component_strategy_not_a_rung`).

Two facts frame the table. First, the gate pins ONE card per cell (`tools/gate_sched.py`: every cell runs with
`CUDA_VISIBLE_DEVICES=<card>`), so Prism plans every gate cell for a one-card profile (`default-ff6008b7` 16 GB,
`default-5ca6057c` 32 GB): a multi-device strategy can never be chosen at the gate. Second, the point B derivation
of 2026-10-25 (`registres/runs/pointB_derive_catalogue_2cd9034c_2026_10_25.tsv`) planned the same 98 cells under
the RACK profile (`c4140-4xv100-custom-nvlink.yml`, four cards, the measured facts): Prism chooses `single_gpu`
for 74, `pipeline_parallel` for 8, `block_scatter` for 7, `lazy_sequential` for 5, `component_placement` for 3,
`single_gpu_lifecycle` for 1. The 18 multi-device choices are plans the engine has never executed at a gate.

| strategy | tests that reach it (unit, synthetic profiles unless said) | executed on a real model |
|---|---|---|
| `single_gpu` | every gate cell; `test_a_model_that_fits_whole_is_planned_whole` | 74 of 98 gate cells on the rack profile; every one-card gate since 2026-10-14 |
| `single_gpu_lifecycle` | `test_the_force_door_reaches_every_strategy_it_guards`, registry parity | 1 cell on the rack profile; at the gate as the one-card plan of several multi-component models |
| `pipeline_parallel` | force door, registry parity, `test_execution_guarantee`, `test_the_kv_check_counts_what_a_plan_keeps_resident` | NEVER at a gate. Chosen on the rack profile for DeepSeek-Coder-V2-Lite, deepseek-moe-16b-chat, Ming-Lite-Omni-1.5, Wan2.1-I2V-14B-480P (both engines). One recorded run: the Triton path of Qwen3-30B (`kernels/nbx_tensor.py:1538`, the synchronous cross-device copy it needed), undated in the code; the collective study of 2026-09-03 measured the link on 3 cards (below). No judged output. |
| `component_placement` | force door, `test_a_host_placement_under_triton_is_planned_on_its_card`, kv check, planned-whole | NEVER at a gate. Chosen on the rack profile for Open-Sora-v2 (both) and Sana_1600M_4Kpx_BF16 native. One forced run on record (`NBX_FORCE_STRATEGY=component_placement`, two transformers on two cards, the pre-fusion tree; journal `dell_report_body_until_2026_10_25_0025.md`), unjudged by sha. |
| `block_scatter` | force door, registry parity, `test_symbol_map_names_its_hypothesis`, kv check, `kernels/test_prefill_determinism` | NEVER at a gate. Chosen on the rack profile for the three Qwen3-30B-A3B models (both engines) and Qwen3-Omni triton. |
| `weight_sharding` | force door, registry parity, `test_explain_plan`, kv check; `regression/test_serve_warm.py` (real models, serve shape) names it for Wan2.1-T2V-1.3B | No gate cell; the serve regression observed it unpinned on Wan-T2V (comment at its line 147). |
| `component_placement_lazy` | force door, registry parity | Never chosen today; the point B derivation names it as the estimate's fastest alternative for 22 cells (a 16 GB card at 789 GB/s against a 32 GB one at 779), on one card. |
| `lazy_sequential` | force door, registry parity, `test_a_component_over_the_rung_streams_inside_itself` | At every one-card gate for the large models (gate logs: Qwen3-30B, DeepSeek-V2-Lite, Wan2.1-I2V-14B "Strategy: lazy_sequential"); 5 cells on the rack profile (Wan2.2-I2V-A14B, mochi, Qwen3-Omni native). |
| `zero3` | force door, `test_a_streamed_llm_keeps_its_cache_inside_the_rung`, kv check | At the gate: Wan2.1-I2V-14B-480P's transformer (ratchet: 40 blocks, 28 resident, 12 streamed per pass, gate_msnv log); never chosen on the rack profile (pipeline_parallel wins there). |
| `op_level_tiling` | `test_execution_guarantee`, tiling tests | At the gate: the video decoders Prism tiles op-level (the tiling engine is single-device). |
| `layer_streaming` | force door, `test_layer_partition`, execution guarantee | Never at a gate (loses to zero3 / lazy on these cards). |
| `cpu_execution` | `test_execution_guarantee`, `regression/test_all_models.py` | Never (no CPU run, by order). |
| `cpu_streaming` | `test_cpu_streaming_places_everything_on_the_host`, execution guarantee | Never (same order). |
| `expert_store` (component) | 14 test files (`test_an_expert_store_leaves_its_card_one_drain_cliff`, `test_no_arena_is_planned_above_the_devices_largest_allocation`, the file-backed refusal, ...) | At every one-card gate of the routed models: Qwen3-30B-A3B ×4, DeepSeek-Coder-V2-Lite, deepseek-moe-16b-chat, Ming-Lite-Omni-1.5 — experts on pinned host memory (`core/runtime/expert_files.py`, `expert_residency.py`), streamed per call over the HOST link. |

The 2026-09-03 study (`registres/journaux/_docs_internal_old/tensor_parallel_study_2026_09_03.md`, verdict in
`validation_outputs/tp_collective_latency_2026_09_03/` of that day) measured the rig's collectives: the best shape
cost 124 µs per collective at a 4 KB decode payload on 3 cards with peer access on; 96 collectives a token =
11.9 ms, 47 % of the 25.1 ms token then measured; in every cell the device time equalled the host issue time (the
GPUs idle on the launch). It also found that a cross-device `memcpy` with peer access OFF is bounced through host
memory by the driver even though every pair reports NV2 — which is why `NBXTensor.enable_peer_access` exists.

## 2. Where the profile's `interconnect` is read, and where it should be and is not

Read today:

* `core/prism/loader.py` — parses `interconnect:` (groups, links, topology, `default_tech`, `bandwidth_gbps`) and
  `pcie_fallback` into `InterconnectTopology` (`structure.py`: `get_bandwidth(a, b)`, `devices_have_fast_interconnect`,
  `get_min_pairwise_bandwidth`; scoring reads the bandwidth figure, never the technology name).
* `core/prism/autodetect.py` — detects the topology per brand (NVLink; xGMI; Tenstorrent Ethernet mesh; MLULink) with
  a PCIe fallback (version, lanes, bandwidth) and writes the generated profile `default-<tag of the visible GPU set>`.
* `core/prism/solver.py` — spillover prefers the same group (1719-1764); a component chain's next device prefers a
  fast link to the last (4953); `pipeline_parallel` picks n devices that are all fast-interconnected (5219-5224); the
  score penalises transfers by the minimum pairwise bandwidth of the chosen devices and by topology completeness
  (6987-6997). The rack profile declares one `full_mesh` group of the four cards at 300 GB/s, PCIe 3.0 ×16 (32 GB/s)
  as the fallback.
* `kernels/nbx_tensor.py` — `can_access_peer` / `enable_peer_access`: a device-to-device copy over the interconnect
  instead of the driver's host bounce.

Not read, and should be:

* `core/prism/time_estimate.py` (point B): the movement term prices the HOST link only (`h2d_bps`: every component's
  weights cross it once per load; zero3 and layer_streaming per use; the lazy strategies per phase). A plan that
  spans devices moves activations across the device link at every boundary, and `block_scatter` / `weight_sharding`
  move partial results every step: there is no device-to-device term, so the estimate ranks `pipeline_parallel` for
  Wan2.1-I2V-14B at 55 s and `block_scatter` for Qwen3-30B at 15 s against measured one-card runs of 138 s and 174 s
  without ever having priced the link. The profile carries the figure (`get_bandwidth`); the measured block should
  carry a device-to-device figure as it carries `h2d_gbps`, from `tools/profile_facts.py`.
* the expert store (`core/runtime/expert_files.py`, `expert_residency.py`, the routed path of `graph_executor.py`):
  experts live on pinned host memory and cross the host link per call (measured h2d 8.1 GB/s, d2h 3.7 GB/s on card 0,
  PCIe 3.0). On this rack the four cards hold 96 GB of device memory: the 58-66 GB routed containers (Qwen3-30B-A3B
  ×4) fit resident across the mesh, their experts reached over NVLink. The store knows one device and one host.
* the tiling engine (`core/tiling*`): no interconnect read; a tiled component is a single-device component.
* serving (`serving/engine.py`): no read; one model per process, the profile handed to the wrappers as the CLI does.
* the gate itself (`tools/gate_run.py`, `gate_sched.py`): the class is the card's VRAM and a cell is one card.
* the width model and the derived census: a plan is projected per rung on one device; a multi-device plan's keys
  (per device, and the sliced shapes of `weight_sharding`) are not derived.

## 3. What the gate needs for a multi-card class (2 and 4 NVLink cards), inside the ten-minute regime

Today: one class per gate (the base gate's class, 16g or 32g), ONE card per cell, the generated one-card profile,
the plan-identical list (`tools/plan_listing.py`) so that a model with one plan across classes runs once.

1. **Classes.** Two more classes, named by what they are: `2x32g-nvlink` (cards 2 and 3, the same VRAM) and
   `4x-nvlink` (cards 0-3, mixed 16/32 GB). Their profiles are the generated ones autodetect already keys by the
   visible GPU set (`default-<tag>` for `CUDA_VISIBLE_DEVICES=2,3` and for `0,1,2,3`), the rack's `interconnect` read
   from the same detection: no hand-written profile, no new path.
2. **Which cells.** The plan listing extended to the new classes: a model whose plan on 2 (or 4) cards is the one-card
   plan (`single_gpu`) has nothing to prove there and gets NO cell; only the models whose plan CHANGES get one — on
   the rack profile that is 24 cells today (section 1), not 98. The regime's ten minutes hold: these plans exist to be
   faster than the one-card ones (their one-card gate times: 50-600 s), and the gate request stays the model's floor.
3. **The scheduler.** A multi-card cell takes every card of its class under the same hold ledger (the holds are per
   card today: take all or none, release all); nothing else runs on those cards; the cell's row records one `gpu`
   peak PER card and the per-card class it ran under.
4. **Identity and judgment.** The cell's output is compared with the model's one-card output of the same request
   (the rows already hold its sha): `pipeline_parallel`, `component_placement` and `block_scatter` keep every op whole
   on one device and must reproduce it byte for byte; `weight_sharding` changes the reduction order of the sliced
   matmuls and its output is judged by eye and against the one-card output (the register says which cells differ and
   why). A multi-card cell that is byte-identical inherits the one-card verdict (the regime of 2026-10-18).
5. **Certified-only.** The derived census must project the multi-card profile's plan (red line: "Prism's SOLVED plan
   per rung"): the per-device keys are the one-card keys for the three whole-op strategies; `weight_sharding` forms
   sliced shape classes that must enter the census table of the new class before its first cell — a missing key is
   an error naming the key, never a runtime sweep, on these cards as on one.
6. **The peer-access door.** A multi-card cell runs with peer access in force between its cards (`enable_peer_access`,
   both directions) or refuses by name: a plan that bounces through the host at 8 GB/s instead of the 300 GB/s link
   is not the plan Prism priced. The row records the link counters read from outside (`nvidia-smi nvlink`) so the
   proof that the link carried the traffic is not the engine's word.

## 4. The five models that gain most on four NVLink cards, and the measurement

Order: the routed 30B first (the owner), then the dense giants whose one-card plan streams.

| model | container | one-card plan today (gate) | plan on the rack profile | what four cards change |
|---|---|---|---|---|
| Qwen3-30B-A3B-Thinking-2507 | 58 GB | `lazy_sequential` + expert store on the host; 174 s native / 322 s triton at the gate | `block_scatter` (estimate 15 s, unpriced link) | every expert resident across 96 GB of device memory; no host streaming per call |
| Qwen3-Coder-30B-A3B-Instruct | 58 GB | same shape; 178 s / 283 s | `block_scatter` | same |
| Qwen3-VL-30B-A3B-Thinking | 58 GB | same shape; 395 s / 598 s | `block_scatter` | same, plus the vision tower on its own card |
| Wan2.2-I2V-A14B | 119 GB | `lazy_sequential` (two 14B denoisers loaded in turn); 118 s / 164 s | `lazy_sequential` still — the two denoisers, 2 × ~57 GB, fit two 32 GB cards only sharded | `component_placement` of the two denoisers once the weights are sharded across the mesh; the switch between them no longer reloads over PCIe |
| Wan2.1-I2V-14B-480P | 85 GB | `lazy_sequential` with the zero3 ratchet (28 of 40 blocks resident, 12 streamed per pass); 138 s / 174 s | `pipeline_parallel` (estimate 55 s) | the transformer whole across two 32 GB cards: no block streamed |

(DeepSeek-Coder-V2-Lite-Instruct, deepseek-moe-16b-chat and Ming-Lite-Omni-1.5 are the next three: 31-53 GB routed
containers that `pipeline_parallel` places on the rack profile; Qwen3-Omni-30B-A3B is the fourth 30B.)

The measurement, per model, when the workstream opens — nothing before the stable:

1. The confirmation request of the model (the one judged by sha in the registers), the same seed, through the entry
   point, on 1, 2 and 4 cards (`CUDA_VISIBLE_DEVICES` sets the class; the profile is the generated one of that set).
2. The output: byte-identical to the one-card output (the three whole-op strategies), or judged by eye with the
   difference explained (`weight_sharding`); the verdict by sha in the register.
3. Wall and exec time, load time, per-card peaks, from the harness rows; the NVLink counters read from outside
   during the run; the host link counters too, to show what moved where.
4. The estimate the time model gave against the measurement, so the device-to-device term is written from a
   measured figure, not before it (the rule of the measurement before the number).

## What this inventory does not do

It writes no code, changes no profile, runs no card. The order of work stays: lot52 lands, the confirmations end, the
final benchmark runs on the final main, main is rewritten on the supervisor's GO, the tag. Then this.
