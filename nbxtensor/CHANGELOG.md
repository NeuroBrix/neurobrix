# Changelog

All notable changes to nbxtensor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`backend.launcher.prewarm_compiler()`**: computes Triton's cache key (`triton_key`, the sha256 of libtriton and of Triton's sources, about 1.3 s on a V100 host) in a daemon thread, so a process's first kernel compile no longer waits for it; an application calls it at start, while it loads weights.

### Changed
- **The launch recorder receives each pointer's extent** (`backend.launcher.set_launch_recorder`: `fn(prepared, grid, params, extents)`): the byte extent of each pointer's tensor that a driver binding by extent is handed, None for a driver that binds by address — a replay can now launch exactly what was launched.
- **The README opens with what the library is and `pip install nbxtensor`**: the 0.0.1 README began its installation with `pip install -e nbxtensor` from the engine's repository, which PyPI showed as the package's description; it now opens with its title, what it does, the install from PyPI (the repository install is below, for contributors), a GPU example its tests run, why the library exists (no PyTorch, any Triton GPU, the V100's tensor cores back in use with their measured gains, planned memory, certified tuning) and the hardware tested and expected, and it links its API and backend references in the NeuroBrix repository, where they live (they were relative links to files the library's repository does not hold). The package's one-line description no longer names a CPU backend, which the library does not have.

## [0.0.1] - 2026-11-05

The first release of nbxtensor: an inference-first tensor library whose kernels are written in Triton and which never
imports torch. NeuroBrix's Triton engine is its first client: on 2026-11-05 the engine's final gate ran its 49 catalogue
models through the library on four V100 cards (both card classes, every output judged by eye), and the library's tests,
five reference models and an uncertified autotune sweep ran in an environment where torch is not installed, with no
attempt to import it.

**What the library holds**
- **A tensor without torch**: `nbxtensor.Tensor` with its dtype table (`dtypes.yml`: every width, safetensors code and
  key spelling in one place), strides counted in elements, ATen's view rule, broadcasting, Array API type promotion with
  weak Python scalars, seeded random streams, quantized weights (grouped int4 with scales and minima), and DLPack for
  exchange (`__dlpack__`, `from_dlpack`): a torch user writes `torch.from_dlpack(t)` on their side.
- **Triton kernels for the operators of inference graphs**, each declaring its cost (outputs, workspace and scratch with
  their lifetimes, bytes moved, flops) so a planner prices a model before running it: elementwise and comparison ops,
  reductions, softmax, layer, group, RMS and batch norms, matrix products and matrix-vector products (on the V100's
  m8n8k4 matrix unit where certified faster), 1-D, 2-D and 3-D convolutions and transposed convolutions, pooling and
  resampling, rotary embeddings, attention (flash, split decode, paged decode, vector decode), mixture-of-experts (the
  fused GEMM, the one-token band, the routing alignment), FFT, sorting, top-k, gathers and scatters.
- **Memory that is planned**: allocators per device behind one interface (a pool that splits and merges freed blocks; on
  CUDA with virtual memory management an expandable range), a slab planner (`memory.slab`), and the capture and replay
  of a repeated sequence of launches, verified byte-equal (CUDA graphs on NVIDIA).
- **Kernel settings proven once, per device class**: every autotuned kernel is certified per launch key against a
  float64 oracle and served from a store of proven settings; the library sweeps a key it has no setting for at first
  use, and an application may require certified settings only (a missing key is then an error naming it).
- **Every core of the host**: host work over large arrays (weight conversions at load) runs on a thread pool sized to
  the machine's compute cores, with the BLAS held to the same cores.

**Hardware**: NVIDIA V100 (CUDA) and Apple M4 Pro (Metal through triton-ext) are proven; NVIDIA Ampere and Hopper and
the other Apple silicon chips have device files and are not proven on our machines; AMD Instinct (HIP) and Intel GPUs
(Level Zero) have their interfaces written and are not tested for lack of hardware; there is no CPU kernel backend.

**Named refusals**: a missing fact is an error that names it, never a default — a dtype absent from the table, a
device fact absent from its device file, a strided slice, a signed integer promoted with `uint64`, a Python integer
outside an integer array's range, a pointer Metal declines to bind in place, a per-row bias of more than four rows in
`addmm`, an uncertified key where certified settings are required.

**Known limitations**: inference only (no autograd); slicing has step 1; no array namespace (`__array_namespace__`)
yet; `conv_transpose2d` runs on the FMA kernel only (no matrix-unit path; no catalogue model forms it); on Apple silicon,
two of NeuroBrix's video models do not complete on the M4 Pro yet — mochi-1-preview (the PyTorch engine's memory at its
decoder's temporal stream) and Allegro / Allegro-TI2V on the Triton engine with its replay (a Metal command buffer out of
memory; the fix is in test).

### Added
- **`nbxtensor.memory.slab.first_fit_extent`**: the slab a run needs, the highest offset the slab's own first fit
  reaches over the run's allocations and frees — between its live peak and the sum of its blocks.
- **nbxtensor describes the NVIDIA and AMD devices itself**: `backend/devices/nvidia/` (Volta, Ampere, Hopper) and
  `backend/devices/amd/` (CDNA, CDNA2, CDNA3) hold the facts its kernels read — tiles, shared memory, registers,
  warp size, the matrix unit, the autotune ladders and screen, the attention thresholds — as Apple's did; an
  application's vendor file only adds its own keys beside them (a key in both is refused).
- **The fused mixture-of-experts GEMM runs on the V100's matrix unit** (`fused_moe_m8n8k4_kernel`, `moe_wrappers.invoke_fused_moe_m8n8k4`): the FMA kernel's contract — the alignment's blocks, the expert pointer table, the top-k division, the routed weight — on the m8n8k4 unit at the profile's tiles, the routing aligned at the tile's rows. Its certificate decides per key against the FMA kernel (`fused_moe_unit_tile`); on a V100 the unit measured 2.5-5x faster on every prefill projection timed (DeepSeek-Coder-V2-Lite's 20-token gate 2.49 -> 0.96 ms, Qwen3-30B-A3B's 512-token gate 7.24 -> 1.40 ms).
- **nbxtensor names the device in force publicly**: `nbxtensor.backend.device_files.in_force()`, `for_arch(arch)` and `forget()`, and `nbxtensor.backend.launcher.bind_target(target)` for a derivation that launches nothing; the engine's key census and paged KV cache read them instead of the library's private configuration module.

### Changed
- **A certified file is parsed and validated once, then read from its compiled index** (`ops.certified_index`): the directory stays JSON, the source contributors edit; the first launch after a change validates a file and writes, per memory class, a binary index of the entries that class is served (length-prefixed keys and certifications, a SHA-256 trailer, never a pickle), named by the fingerprint of everything the validation read; later launches read the index and parse a certification only when its key is asked. A corrupt index is refused by name.
- **The launch recorder receives each pointer's extent** (`backend.launcher.set_launch_recorder`: `fn(prepared, grid, params, extents)`): the byte extent of each pointer's tensor that a driver binding by extent is handed, None for a driver that binds by address — a replay can now launch exactly what was launched.
- **Every device file states the tolerance an integer kernel output is certified at**: `autotune_screen_rtol.int64: 0.0` — met exactly (the MoE alignment's routing tables); the certifier invents no tolerance and refused every alignment key without it.
- **The MoE alignment's three stages take their chunk widths and warps from their certificates**: `moe_align_stage{1,2,3}_kernel` ran per-backend literal widths (128 or the experts' count on CUDA, 32-64 on Metal) at 4 warps; each is now certified per key (the block, the padded expert count, the layer's pairs in the profile's `elements` classes), its candidates pruned to the blocks that fit the profile's `memory.max_shared_memory_per_block`. Every candidate computes the same routing; the choice is speed only.
- **`nbxtensor.memory.pointer_tables.build_ptr_tables` is public**: the expert pointer tables a MoE kernel reads (on Metal captured under a pinned scope), built by the engine's MoE layer and by its certifier alike; it was the private `_build_ptr_tables`.
- **The split decode's reduce takes its warps from its certificate**: `flash_decode_reduce_kernel` ran at a literal 4 warps after both split decodes (vector and paged); it is now certified per key (the splits, the value head dim, the output dtype) through `attention_wrappers.decode_reduce`, and `decode_attn_vec.decode_vec_splits` names the vector decode's split rule for the wrapper, the census and the certifier alike.
- **The paged decode attention takes its position tile and warps from its certificate**: `paged_decode_attention` ran the judged tile (32, `PAGED_DECODE_JUDGED_TILE`, removed) at 4 warps; the tile is now certified per key among the candidates that divide the page, and the call takes no tile argument. A split's positions are whole pages' worth of the largest tile (`PAGED_DECODE_SPLIT_ALIGN`), so every candidate cuts the splits alike. The certificate is keyed by the splits the longest sequence makes (`paged_decode_attn.paged_decode_splits`), the same count a replayed decode step is bucketed by, so a replayed step and the step it is verified against run one configuration.
- **A matrix product the V100's matrix unit can take runs on the unit or on its FMA kernel as its certificate measured**: `mm` and `addmm` took the m8n8k4 unit whenever its smallest tile filled, at the first tile of the profile that fit, a launch the certification never compared with anything. The FMA launch now keeps its autotune key on the unit too; certifying it times every unit tile the profile declares beside the FMA configurations, against the same oracle, and the entry names the unit's tile only when it was faster (a 80 x 4096 x 4096 fp16 product: FMA 0.404 ms, the first tile 0.207, the second 0.191 — the second is served). A run with no certificate keeps the declared unit; a confirmation run refuses the missing key.
- **A 1-D transposed convolution runs as a GEMM and an overlap-add** (`conv_transpose_wrapper`, groups 1, dilation 1, no output padding): `x^T [L, C_in] @ W [C_in, C_out*K]` through `mm` (the length on the rows, which the autotune key classes) — the matrix unit where the hardware profile declares one — then `col2im1d_kernel` gathers each output position's taps. The scatter kernel looped over every input channel and tap per output element; the vocoders' upsamplers ran 20-307x slower on a V100. Same output dtype; the waveform of a chatterbox sentence agrees with the scatter kernel's at 49.5 dB SNR, its transcription exact.
- **The vector decode attention takes its key block and warps from its certificate**: `decode_attn_vec_split_kernel` ran 32 keys a step and 4 warps, read from environment variables (`NBX_DV_BLOCK_N`, `NBX_DV_WARPS`) with those defaults; both are now candidates certified per key (the keys' bucket, the head dims, the GQA group, the bias). The split (256 keys a program, at most 64) stays the kernel's: it defines the partials.
- **A one-token mixture-of-experts layer on half-precision experts takes its tiles from its certificate**: the gateup, the down projection's partials and their sum (`moe_decode_vec`) ran a hand table (32 x 128, 4 warps; 256 / 2 warps for the sum); each is now certified per key over the ladder the table was measured from (Qwen3-30B-A3B's 2048 x 768 experts on a V100: gateup 16 x 64, down 16 x 256, sum 1024).
- **The fused mixture-of-experts GEMM takes its tile from its certificate**: `fused_moe_kernel`'s column and reduction tiles, warps and stages were literals (64 x 32, 4 warps, 2 stages, `moe_wrappers._MOE_BN/_MOE_BK`); they are now candidates the certification measures per key — the GEMM's dims, the valid (token, expert) pairs' bucket and the layer's experts (`launch_keys.fused_moe_launches`). The row block stays the routing alignment's. A 512-token Qwen3-30B-A3B prefill's gate projection takes 128 columns.
- **A 1-D transposed convolution runs as a GEMM and an overlap-add** (`conv_transpose_wrapper`, groups 1, dilation 1, no output padding): `W^T [C_out*K, C_in] @ x` through `mm` — the matrix unit where the hardware profile declares one — then `col2im1d_kernel` gathers each output position's taps. The scatter kernel looped over every input channel and tap per output element; the vocoders' upsamplers ran 20-307x slower on a V100. Same output dtype; the waveform of a chatterbox sentence agrees with the scatter kernel's at 49.5 dB SNR, its transcription exact.
- **Depthwise convolutions read their input coalesced**: the depthwise kernel's tile put the channels on its fastest axis, H*W elements apart in memory, so no load coalesced; it now computes a (channels x rows x columns) output tile with the width fastest. On a V100: SANA-Video's 11200-channel 3x3 4.5-8 times faster (2.63 -> 0.33 ms, at or past torch's 0.38 ms), the vocoders' one-row depthwise 1.6-2.4 times faster (orpheus-3b's first audio 3.68 -> 1.85 s); every depthwise key re-certified on both memory classes.
- **An fp32 1 x 1 convolution runs on the V100's matrix unit**: the convolution's implicit GEMM carries fp32 operands as hi + lo through three MMAs (as the GEMM already did), but only where the hardware profile declares a measured row for the kernel window and the tile fills — on a V100 the 1 x 1 window (1.26-1.95 times faster on the video VAEs' shapes); a 3 x 3 measured slower than the FMA path at every tile and keeps it. The census forms no autotune key where the unit runs.
- **The matrix product and the linear layer declare what they allocate** (`nbxtensor.ops.linalg_wrappers.mm_cost` / `addmm_cost`, beside `bmm_cost`): the casts, the decode rows' GEMV and the weight copy a row-major weight costs there, the bias made contiguous, the output at its store dtype, an epilogue's output; each equal to the allocator's measured peak on the device, over the dtypes, the decode and tiled row counts, both weight layouts and the bias forms.

### Fixed
- **nbxtensor finds the NVIDIA card on a machine that has only the driver**: its CUDA backend opens the CUDA runtime library (`libcudart.so.12`), which the driver does not ship; on a clean Linux machine no GPU runtime was found and the card was read as a CPU. nbxtensor now depends on `nvidia-cuda-runtime-cu12` on Linux.
- **The README's examples run in their own interpreter**: the suite ran them in its own process, and the settings
  example's `set_store(None)` reset the certified store every later test of that process read (a test of the live
  directory found it empty in the suite and green alone).
- **The Metal runtime no longer invents its dispatch limits**: the command buffers a queue may hold in flight and the
  dispatches one buffer may carry fell back to 32 and 512 written in the runtime when the device file did not state them;
  they are read from the device file, and a missing one is refused by name.
- **The half-precision MoE decode band no longer tries configs past the device's shared memory**: `moe_gateup_vec_half_kernel` and `moe_down_split_vec_half_kernel` prune their tiles to the [BLOCK_K, BLOCK_N] fp32 blocks `memory.max_shared_memory_per_block` holds before any compile (on Apple's 32 KB a 256 x 64 block aborted the Metal emitter); the door is `_configs.fits_shared_memory`.
- **The library alone autotunes on NVIDIA**: without an application's hardware profile, a kernel's device facts
  (`memory.max_registers_per_block` and the like) were read through the profile and found absent, so a reduction or
  a softmax without a certified configuration was refused; they are now read from the device file of the card the
  launch targets. The backend conformance suite passes on CUDA with nbxtensor alone.
- **A request larger than a card's whole expandable range is refused as the out-of-memory it is**: on a device whose
  large blocks grow in an expandable (VMM) range, such a request raised a plain error without its figures; it now
  raises `DeviceOOMError` with the requested, live, pool-cached and driver-free bytes, as the driver path does, so a
  caller that reshapes the work on a refusal catches it.
- **A certified entry naming parameters its kernel lacks is no longer served**: the directory's check compared a
  key's arity and dtype with the kernel, never the configuration's parameter names, so the depthwise convolution's
  entries from its former tile (`BLOCK_HW`, `BLOCK_C`) were served to the new one (`BC`, `TH`, `TW`). Such a file is now
  refused by name; the Apple and V100 depthwise entries were certified again in the new space (1aa40649, f78b9c4b).
- **A certified-only launch whose key cannot be formed is refused by name**: the key builder raising left the key empty, which skipped the certified lookup and let Triton's autotuner sweep inside a run promised to serve the certified directory only; it is now a `KeyNotCertified` naming the kernel and the exception.
- **The Triton allocator's expandable range maps in chunks the engine measures on a real run, and a refusal remaps the
  memory it takes back**: the chunk was measured per mapping call (2 MiB on a V100), and a refusal released the chunks
  it reclaimed and created them again — SANA-Video_2B_720p on a V100-16GB mapped 89 GB over a run and was 33-44 % slower
  than without the range. The first launch now replays a real run's allocation trace, which the library carries, with
  the range off and at each candidate chunk, and keeps the smallest within the off replay's noise (32 MiB on the V100
  16 GB, 64 MiB on the 32 GB); a reclaim maps its chunks again where they are needed. SANA-Video 16g runs at the off figure, and on
  a 32 GB card the range holds 8 GB less at its peak.
- **The Triton allocator's large class grows in an expandable range on a CUDA device with virtual memory management**: one reserved address range per device, physical chunks mapped into it as the pool grows, and at a refusal one synchronisation, then every mapped chunk lying whole inside a free block unmapped and mapped again where the request needs it — a free remainder beside a live tensor no longer pins its pages (Wan2.1-I2V triton on a V100-16GB was refused a 731 MB attention product with 1.1 GB of 48 MB remainders pinned in driver segments). The method of PyTorch's `expandable_segments`; the chunk is a measured fact of the generated profile (`vmm_chunk_bytes`: the smallest multiple of the granularity whose map costs no more than an ordinary allocation, 2 MiB on V100), refused by name when absent; a device without VMM keeps its driver segments. The small class and the weight arenas are unchanged; `memory.pool.simulate` replays the same rules.
- **A refused small allocation is retried at the size of the segment the pool records**: a small request grows the device by a 2 MiB segment the pool then carves later small blocks from, but the retries after a refusal (the pool's flush, the idle slabs' release, the opt-in collection) asked the driver for the request's bytes — a 1 KiB driver allocation the pool treated as 2 MiB. Found while reading the allocator for the expandable segments (2026-11-02), on the path a full card takes; a test pins it.
- **The Triton allocator's pool no longer lets a scalar pin a multi-gigabyte block**: a request took "the smallest free block that holds it" in one class, so a 17 KB scalar carved a freed 2.5 GB block and pinned its remainder alive, a 632 MB request the next — Allegro triton on a V100-16GB (gate of 2026-11-01) was refused a 2.5 GB gelu output with 4.4 GB of free blocks in pinned segments (Allegro-TI2V, CogVideoX-5b-I2V, Wan2.1/2.2-I2V, MiniCPM-o the same). The pool now has two size classes (a request of at most 1 MiB is carved from 2 MiB small segments, torch's caching allocator's kSmallSize / kSmallBuffer) and the half rule (a large block is split only when what is left is at most the request); its wholly free segments stay parked across a growth and return to the driver at a refusal, as before 164e89dc, whose flush before every growth — which emptied the small free segments the scalars then carved large blocks for — is withdrawn. The real trace replayed through the pool's own rules (`memory.pool.simulate`, M3's replay) completes with the least held of every pooled rule tried, on Allegro 16g, mochi-1-preview 32g and Kokoro alike.
- **nbxtensor's `addmm` refuses a per-row bias `[M, N]` above four rows instead of computing it wrong**: its kernel adds one bias row to every output row, so such a bias was applied with its first row only. It is now refused by name (a bias `[N]` or `[1, N]`, which every catalogue model uses, is unchanged; a per-row bias at up to four rows is still served).

### Removed
- **`decode_attn_vec_grouped_kernel` and the environment switches of the vector decode** (`NBX_DV_GROUPED`, `NBX_DV_SEG`, `NBX_DV_BLOCK_N`, `NBX_DV_WARPS`): the grouped variant ran only behind its switch.
- **`flash_decode_split_kernel` and its route**: armed only by an environment switch (`NBX_FLASH_DECODE`, with `NBX_FD_BLOCK_N` and `NBX_FD_SEG`) and measured slower than the vector decode kernels; `flash_decode.py` keeps the fixed-order reduce they share.
- **`conv1d_forward_kernel`**: launched by no route (a 1-D convolution runs through `conv2d`).

[Unreleased]: https://github.com/NeuroBrix/nbxtensor/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/NeuroBrix/nbxtensor/releases/tag/v0.0.1
