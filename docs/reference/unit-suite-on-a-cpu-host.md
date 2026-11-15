# The unit suite on a CPU host

A map, read on 2026-11-13 at the head of `main` (commit 8ee0dcfb4), of what `tests/unit` and `nbxtensor/tests` do on a host with no card. No code and no test was changed to produce it. It is the list of follow-up tasks, not a verdict on the product: a CPU run is never a proof for a GPU path.

## Reproduce

```
# host: 32 cores, no GPU, CUDA runtime library present without a driver, no workshop configuration, empty model cache
# VENV_PYTHON = a Python 3.10 environment with the project's test requirements (pytest 9.2, torch 2.14); no xdist, no pytest-timeout
export PYTHONPATH=src:nbxtensor/src
# one pytest per folder, so one hang loses nothing; 6 folders at a time; a hard 1500 s limit per folder (`timeout`, as no per-test timeout plugin is installed)
for d in tests/unit/*/ nbxtensor/tests; do
  timeout 1500 $VENV_PYTHON -m pytest -p no:cacheprovider --continue-on-collection-errors -q -rfEs --junitxml=out/$(basename $d).xml $d > out/$(basename $d).log
done
```

`--continue-on-collection-errors` matters: without it a folder whose test modules fail at import (the workshop configuration) is interrupted and runs NOTHING (this happened on the first pass for census, kernels, prism, regression, tools: 0 tests ran). The five loose files of `tests/unit/` (shared helpers) hold no tests. No folder hit the time limit; the longest is `tests/unit/runtime` at 203 s. Wall times were taken with six folders running at once, so read them as upper bounds. A second full run of `nbxtensor/tests` gave the same 676 / 503 / 79 / 66 / 28.

## Totals

**7250 collected: 5225 passed, 228 failed, 122 errors, 1675 skipped** (sum of folder walls 1119 s). Of the 350 failures and errors: 169 need a GPU, 175 need a lab file or the workshop configuration, 3 are real defects, 3 are stale tests. Nothing hung.

## Per folder

"errors" are test modules or fixtures that raised before the test ran; the four right-hand columns classify failed + errors.

| folder | collected | passed | failed | skipped | errors | wall (s) | needs a GPU | needs a lab file / workshop config | real defect | stale test |
|---|---|---|---|---|---|---|---|---|---|---|
| nbxtensor/tests | 676 | 503 | 79 | 66 | 28 | 19 | 107 | 0 | 0 | 0 |
| tests/unit/agent | 62 | 62 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| tests/unit/audio | 91 | 87 | 0 | 4 | 0 | 7 | 0 | 0 | 0 | 0 |
| tests/unit/benchmarks | 25 | 23 | 2 | 0 | 0 | 5 | 0 | 2 | 0 | 0 |
| tests/unit/brick_ratchet | 19 | 18 | 1 | 0 | 0 | 41 | 0 | 0 | 1 | 0 |
| tests/unit/census | 66 | 55 | 7 | 3 | 1 | 35 | 1 | 7 | 0 | 0 |
| tests/unit/cfg | 12 | 7 | 0 | 5 | 0 | 4 | 0 | 0 | 0 | 0 |
| tests/unit/cli | 225 | 210 | 2 | 13 | 0 | 32 | 0 | 2 | 0 | 0 |
| tests/unit/config | 136 | 128 | 0 | 8 | 0 | 4 | 0 | 0 | 0 | 0 |
| tests/unit/container_symbols | 11 | 10 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| tests/unit/core | 64 | 58 | 0 | 6 | 0 | 16 | 0 | 0 | 0 | 0 |
| tests/unit/docs | 3 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| tests/unit/dtype | 311 | 303 | 0 | 8 | 0 | 12 | 0 | 0 | 0 | 0 |
| tests/unit/flow | 142 | 128 | 0 | 14 | 0 | 75 | 0 | 0 | 0 | 0 |
| tests/unit/harness | 13 | 13 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| tests/unit/io | 14 | 12 | 1 | 1 | 0 | 5 | 1 | 0 | 0 | 0 |
| tests/unit/kernels | 2031 | 896 | 21 | 1098 | 16 | 73 | 28 | 8 | 1 | 0 |
| tests/unit/module | 69 | 63 | 2 | 4 | 0 | 23 | 2 | 0 | 0 | 0 |
| tests/unit/nbx | 122 | 118 | 2 | 2 | 0 | 2 | 0 | 0 | 0 | 2 |
| tests/unit/nbx_tensor | 14 | 13 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| tests/unit/nbxtensor_boundary | 52 | 51 | 1 | 0 | 0 | 74 | 0 | 0 | 1 | 0 |
| tests/unit/packaging | 9 | 9 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| tests/unit/prism | 1192 | 944 | 23 | 210 | 15 | 60 | 16 | 22 | 0 | 0 |
| tests/unit/regression | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| tests/unit/runtime | 731 | 646 | 4 | 81 | 0 | 203 | 4 | 0 | 0 | 0 |
| tests/unit/scheduler | 82 | 59 | 1 | 22 | 0 | 4 | 0 | 1 | 0 | 0 |
| tests/unit/serving | 187 | 163 | 9 | 15 | 0 | 109 | 9 | 0 | 0 | 0 |
| tests/unit/state | 24 | 23 | 0 | 1 | 0 | 70 | 0 | 0 | 0 | 0 |
| tests/unit/strategies | 50 | 46 | 0 | 4 | 0 | 66 | 0 | 0 | 0 | 0 |
| tests/unit/tiling | 64 | 60 | 0 | 4 | 0 | 7 | 0 | 0 | 0 | 0 |
| tests/unit/tools | 457 | 323 | 70 | 3 | 61 | 52 | 0 | 130 | 0 | 1 |
| tests/unit/triton | 276 | 185 | 1 | 90 | 0 | 106 | 1 | 0 | 0 | 0 |
| tests/unit/workshop | 19 | 6 | 2 | 11 | 0 | 2 | 0 | 2 | 0 | 0 |
| **total** | **7250** | **5225** | **228** | **1675** | **122** | **1119** | **169** | **175** | **3** | **3** |

Skips (1675) are by name and honest: the largest reasons are "no GPU" (238), "CUDA required" (172), "no GPU visible to this process" (156), "the allocator's peak needs a device" (146), "needs a device" (118), "needs an Apple GPU with unified memory" (59), "the hardware profile declares no matrix_unit.mm" (49).

## What the table says

1. **One mid-state causes most of class A.** This host has the CUDA runtime library and no driver, so `cudaGetDeviceCount` answers error 35 and the library raises `DeviceQueryError` by name (correct, by the doctrine). The tests that skip with "no GPU" ask a narrower question (is a card visible) and do not skip on this error, so they fail instead. One shared skip condition, "the runtime refuses to count devices", clears nearly all 169; it is a test-side gate, no engine change. A subprocess variant of the same is in several tests that run a child Python (shadow, census, kernels, runtime, triton).
2. **Class B is one missing fixture: the workshop configuration.** Every tool imports it at module level, so a test module that imports a tool errors at collection (61 errors in tools, 15 in prism, 16 in kernels) and a test that runs a tool as a child fails with the same named error. A fixture that supplies a temporary configuration, or a skip by name when it is absent, clears about 150. The rest need a model container in the catalogue cache (two vision-language tests, one scheduler test that found zero flow containers) or a generated hardware profile (one census test).
3. **Cascade:** a failed first import of the campaign tool leaves a half-imported module, so three later tests fail with "cannot import name" for functions that exist in the file; they are class B, not stale.
4. **Class C and D are the real follow-ups** (6 tests). They fail on any host.

## Classes C and D, named

| class | folder | test | what |
|---|---|---|---|
| defect | tests/unit/kernels | test_peer_access_is_not_queried_by_hand_anywhere_else | peer access is queried in `memory/vmm.py` outside `DeviceAllocator` (library side) |
| defect | tests/unit/nbxtensor_boundary | test_the_engine_and_its_open_findings_only_go_down | the engine is 80880 code lines against a list header of 80879 |
| defect | tests/unit/brick_ratchet | test_the_limits_are_not_above_the_percentile_they_were_drawn_from | the `nbxtensor` limit is above its 95th percentile |
| stale | tests/unit/nbx | test_another_version_is_refused_by_name (2 cases) | expects `neurotax_rename`, the message now names the container builder |
| stale | tests/unit/tools | test_every_allowed_text_is_still_on_its_page | 8 allow-list texts no longer on their pages |

## Every failure and error, by test file

Counts are failures plus errors in that file. Class A and B rows share one cause each (stated in the class heading's first row).

### A. needs a GPU

| folder | test file | failures + errors | cause |
|---|---|---|---|
| nbxtensor/tests | `test_a_certification_harness_draws_measures_and_sweeps.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_device_name_is_read_once_for_every_backend.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_device_naming_that_moves_nothing_is_counted.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_device_speed_is_what_it_moves_over_its_time.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_dropped_tensor_is_freed_after_the_work_that_reads_it.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_layer_on_several_devices_is_the_sum_of_their_shares.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_one_element_sentinel_is_saturated_to_a_half_range.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_probe_leaves_no_context_on_a_card_nobody_selected.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_refused_allocation_names_who_holds_the_memory.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_sampler_s_filters_keep_what_transformers_keeps.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_sequence_decodes_past_the_card_s_pages.py` | 11 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_shadow_never_reads_a_host_copy_of_a_shadow_value.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_spilled_sequence_attends_as_one.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_tensor_is_compared_bit_for_bit_and_knows_its_allocation.py` | 7 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_token_bitmask_leaves_only_the_allowed_logits.py` | 6 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_view_of_freed_memory_is_copied_out_first.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_weight_block_uploads_on_its_copy_stream.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_a_weight_set_is_released_after_its_devices_drain.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_arena_counts_what_it_holds.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_array_enters_at_a_dtype_the_backend_has.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_array_is_placed_on_the_device_it_names.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_attention_returns_its_rows_log_sum_exp.py` | 17 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_axis_is_read_and_written_a_slice_at_a_time.py` | 7 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_expert_table_keeps_its_experts_alive.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_an_inspection_reads_every_value_once_on_the_host.py` | 5 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_keys_are_read_in_the_queries_order_and_one_head_per_group.py` | 3 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_the_allocator_states_its_pool_and_its_blocks.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_the_shadow_device_answers_without_a_driver.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| nbxtensor/tests | `test_tiles_are_cut_padded_and_crossfaded.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/census | `test_the_shadow_takes_what_from_numpy_takes.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/io | `test_the_loader_bounds_its_pinned_uploads.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `.py` | 12 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_a_card_s_free_memory_is_read_through_one_door.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_a_fused_moe_key_is_priced_and_windowed.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_a_one_row_convolution_keys_its_width_on_the_ladder.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_a_tensor_outliving_the_allocator_module_exits_clean.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_nbx_torch_bridge_names_its_device.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_the_census_paces_a_decode_to_its_buckets_top.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_the_census_records_the_op_that_formed_a_key.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_the_census_shadow_carries_the_profiles_target.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_the_census_shadow_records_the_launchers_key.py` | 3 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_the_chunk_is_measured_on_a_real_*.py` | 3 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/kernels | `test_the_conv_bias_rides_in_place.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/module | `test_an_activation_reuses_only_a_buffer_that_dies.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/module | `test_the_inplace_unary_interceptor_respects_the_tensor_kind.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/prism | `test_a_prompt_is_priced_at_its_own_length.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver"). Verdict (task 273): none stale, none defective; on a CPU host WITH the `triton` module the six files pass 28 of 28, without it 8 fail (`PlanNotComputable ... No module named triton`; the runtime-base test needs both engines): needs the `triton` module or a card, card job 19 |
| tests/unit/prism | `test_a_served_chunk_is_priced_with_the_cached_keys_it_attends.py` | 4 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver"). Verdict (task 273): none stale, none defective; on a CPU host WITH the `triton` module the six files pass 28 of 28, without it 8 fail (`PlanNotComputable ... No module named triton`; the runtime-base test needs both engines): needs the `triton` module or a card, card job 19 |
| tests/unit/prism | `test_a_session_chunk_is_chosen_from_certified_times.py` | 3 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver"). Verdict (task 273): none stale, none defective; on a CPU host WITH the `triton` module the six files pass 28 of 28, without it 8 fail (`PlanNotComputable ... No module named triton`; the runtime-base test needs both engines): needs the `triton` module or a card, card job 19 |
| tests/unit/prism | `test_a_session_reach_bounds_what_a_request_can_ask.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver"). Verdict (task 273): none stale, none defective; on a CPU host WITH the `triton` module the six files pass 28 of 28, without it 8 fail (`PlanNotComputable ... No module named triton`; the runtime-base test needs both engines): needs the `triton` module or a card, card job 19 |
| tests/unit/prism | `test_the_generation_head_is_priced_at_its_one_row.py` | 5 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver"). Verdict (task 273): none stale, none defective; on a CPU host WITH the `triton` module the six files pass 28 of 28, without it 8 fail (`PlanNotComputable ... No module named triton`; the runtime-base test needs both engines): needs the `triton` module or a card, card job 19 |
| tests/unit/prism | `test_the_profile_measures_the_runtimes_base.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver"). Verdict (task 273): none stale, none defective; on a CPU host WITH the `triton` module the six files pass 28 of 28, without it 8 fail (`PlanNotComputable ... No module named triton`; the runtime-base test needs both engines): needs the `triton` module or a card, card job 19 |
| tests/unit/runtime | `test_a_constrained_decode_samples_only_allowed_tokens.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/runtime | `test_stacked_experts_resolve_to_the_halves_the_build_split.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/runtime | `test_the_expert_store_prefetches_the_next_layers_experts.py` | 2 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/serving | `test_embeddings_are_pooled_as_the_container_declares.py` | 9 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |
| tests/unit/triton | `test_lazy_bind_bucket_boundary.py` | 1 | no card here: the CUDA runtime library is installed but there is no driver, so the first device question is refused (cudaGetDeviceCount error 35, set_device refused, a first allocation refused, or "no NVIDIA driver") |

### B. needs a lab file or the workshop configuration

| folder | test file | failures + errors | cause |
|---|---|---|---|
| tests/unit/benchmarks | `test_the_nbxtensor_bench_judges_right_before_fast.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/census | `.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/census | `test_a_census_record_names_no_machines_home.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/census | `test_the_census_keys_a_sessions_continued_prefill.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/census | `test_the_census_scope_is_what_the_rows_are_derived_from.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/census | `test_the_derived_table_is_the_walked_census.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/cli | `test_read_commands_speak_json.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/kernels | `.py` | 4 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/kernels | `test_the_census_is_one_table.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/kernels | `test_the_census_records_the_op_that_formed_a_key.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/kernels | `test_the_census_walks_every_audio_codec_extent.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/prism | `.py` | 15 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/prism | `test_a_refusal_names_why_layer_streaming_declined.py` | 3 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/prism | `test_a_streamed_plan_states_its_window.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/prism | `test_an_untouched_card_is_budgeted_less_the_context_the_run_will_make.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/prism | `test_mochi_triton_fits_a_16gb_card_with_its_states_parked.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/regression | `.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/scheduler | `test_unipc_flow_sigmas_follow_the_declared_diffusers_version.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `.py` | 57 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_a_derived_relation_break_is_reported_not_swallowed.py` | 4 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_a_dimension_carried_by_shapes_alone_is_not_never_carried.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_a_symbolic_zoo_gate_fails_only_on_a_clean_container_lost.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_campaign_entry_budget.py` | 16 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_campaign_is_recorded.py` | 3 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_campaign_lever_must_measure.py` | 5 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_campaign_refuses_a_live_tree.py` | 9 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_campaign_refuses_an_empty_selection.py` | 3 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_campaign_skip_done_needs_a_verdict.py` | 6 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_catalogue_report_never_borrows_a_measurement.py` | 6 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_census_table_retire_absent.py` | 4 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_public_pages_never_say_how_a_container_is_made.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_report_rotate_never_erases_the_report.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_symbol_collision_census.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_campaign_asks_for_the_voice_the_artefact_requires.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_census_coverage_asks_its_memory_class.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_census_opens_no_card.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_derivation_binds_the_multimodal_request_as_the_run_does.py` | 3 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_gate_is_one_script.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_harness_reads_the_cache_the_engine_reads.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/tools | `test_the_m3_exit_table_reads_the_plan_and_the_run.py` | 1 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |
| tests/unit/workshop | `test_the_regression_harness_names_its_engine_python.py` | 2 | no workshop configuration on this host (every tool that reads it raises the named error at import), or no model container in the catalogue cache, or no generated hardware profile |

### C. real defect on a CPU host

| folder | test file | failures + errors | cause |
|---|---|---|---|
| tests/unit/brick_ratchet | `test_the_brick_ratchet_refuses_what_gets_worse.py` | 1 | the brick ratchet's `nbxtensor` limit sits above the 95th percentile it was drawn from (holds on any host) |
| tests/unit/kernels | `test_peer_access.py` | 1 | `memory/vmm.py` queries peer access outside `DeviceAllocator`: the guard test is right and fails on any host (a library-side defect, not CPU specific) |
| tests/unit/nbxtensor_boundary | `test_the_engine_only_slims.py` | 1 | the engine is 1 code line over its list header (80880 vs 80879): the ratchet rule "it only slims" is broken on main |

### D. stale test

| folder | test file | failures + errors | cause |
|---|---|---|---|
| tests/unit/nbx | `test_a_container_of_another_neurotax_is_refused_by_name.py` | 2 | the test expects the text `neurotax_rename` in the refusal; the refusal now says "container builder" (public wording pass): the test lags the message |
| tests/unit/tools | `test_public_pages_never_say_how_a_container_is_made.py` | 1 | the allow-list names 8 texts their pages no longer hold (public wording pass): remove them from the allow-list |

## Not verified

- No GPU or Metal path: everything classed "needs a GPU" is unjudged until a card runs it. The Dell must rerun `tests/unit` and `nbxtensor/tests` to confirm those 169 pass there.
- Classes A and B were classified from the failure message and, for subprocess failures, the child's message; a file that holds both kinds shows the first one. Whether each class-B test then passes with a fixture is not tried.
- `tests/regression` was not run (it needs `NEUROBRIX_PYTHON` and model assets); only its unit folder was.
