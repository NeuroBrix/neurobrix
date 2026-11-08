# Changelog

All notable changes to nbxtensor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **A kernel served through a generator alias says so in the debug log, not on the terminal**: the line
  appeared once per kernel at default verbosity (6 to 9 lines on a plain run under a macOS 14 wheel); it is a
  developer line (`-v` / `NBX_LOG=debug`), and the plan's explanation still counts the kernels served that way.
- **An out-of-tree backend's certificate identity no longer includes its Python binding**: the CPython extension module
  tagged for one Python (`*.cpython-312-*.so`, `*.cp312-*.pyd`) is built per Python and generates no code, so the Metal wheels for Python
  3.10-3.13 share one identity when their code generator is the same bytes. Proven on 235-236 kernels per wheel set: the
  compiled .ttir, .ttgir, .msl and .metallib are byte-identical. The identity of an installed backend changes once:
  its certificates are re-labelled or re-proven under the new identity (`neurobrix autotune certify --reprove-generator`).

### Added
- **A certification carries one proof per code generator**: `autotune_certified` keeps an entry's (or memory-class
  variant's) `proof` and adds the proofs of the same configuration under other generators in `also_proven`; where the
  stored configuration failed the oracle under a new generator and a sweep chose another, that certification is filed
  in the class's `alternates`. A re-proof under a new generator therefore adds its proof and never un-serves another
  build (macOS 14 and 15 Metal wheels with byte-identical kernels are both served); a re-proof under the same
  generator replaces that generator's proof only. Serving picks the card's memory class, then the certification one
  of whose proofs names the running generator, within that class only. Both fields are additive: a reader that knows
  `proof` alone reads a file as before. New names: `certification_proofs`, `class_certifications`,
  `entry_certifications`, `entry_proofs`, `certification_for_generator`, `class_proven_since`, `reproof_source`,
  `ALSO_PROVEN`, `ALTERNATES`.

## [0.0.7] - unreleased (terminal lot)

### Added
- **`nbxtensor.terminal`, the one terminal rule of a command line built on the library** (`python -m nbxtensor`,
  `neurobrix`): one stderr handler, INFO by default, `-v` the developer lines, `-vv` an error's traceback too, and
  `NBX_LOG=debug|info|warning|error` from the environment (a name that is no level is refused by name).
- **`backend.install_help.compiler_missing`**: names a machine without the C (or C++) compiler Triton builds a kernel
  with, read from the frame Triton raises in (`triton/runtime/build.py`, `_find_compiler`), never from the text.
- **`backend.install_help.hidden_gpus`** (the GPUs the backend's visibility variable — the table's new `visible_env`,
  `CUDA_VISIBLE_DEVICES` on CUDA, ROCm's three on HIP — hides from this process, against the PCI bus) and
  **`backend.cpu_backend.triton_cpu_installed`**.
- **`io.read_rate.read_seconds`** (how long this storage takes to read so many bytes, measured now) and
  **`memory.trace_replay.first_replay` / `vmm_granularity`** (the chunk probe's first replay made ahead, its cost
  pricing the others).

### Changed
- **The library logs; it does not print**: every line goes through `logging`, one logger per module, and importing it
  installs no handler. Its developer lines are DEBUG; warnings and errors reach a command line's terminal.
- **`python -m nbxtensor doctor` lists the GPUs the process sees**: it listed every display controller on the host's
  PCI bus (four in a container given one); the others are counted as not visible here.
- **`python -m nbxtensor install-metal` downloads nothing when the pinned wheels are already installed**, and says so in
  one line (it re-downloaded 131 MB on every call); pip's own lines only with `NBX_LOG=debug`.

### Fixed
- **No Triton warning at the first compile of `masked_fill`**: a uint8 mask reached `tl.where` as its condition, which
  Triton calls deprecated; the condition is cast to a boolean, the compiled code identical.

## [0.0.6] - unreleased (the Apple lot)

### Added
- **`pointer_tables.TablePins` and `end_call_pins`**: a holder for the pins of per-call pointer tables built for a cache
  of experts that outlives the call. The cache ends them before it lets the pages they cover go, and at its release; a
  table over wraps already held ends its pins at once.

### Changed
- **Attribution**: author, maintainer and copyright holder Hocine Benkelaya (contact@neurobrix.es); a NOTICE file ships
  beside the LICENSE (supported and funded by Neural Network Holding Ltd, commercial distribution by WizWorks OÜ); the
  README carries the same attribution.

### Fixed
- **A mapped mixture-of-experts store no longer keeps every decoded expert resident on Apple GPUs**: a per-call
  table's pins ended at the call (re-wrapping the store's arena at a new address while it could still be read) or not at
  all (the decode path: a wrap left over every expert the store had let go). Held by `TablePins`, they end when the
  store says.

## [0.0.5] - 2026-11-07

### Fixed
- **The library runs on Windows**: weights are read at their offset without `os.preadv` (Windows' `ReadFile` with an
  OVERLAPPED offset), the BLAS's thread control is found beside numpy (`numpy.libs`) instead of through /proc, and the
  GPU libraries are opened by their Windows names (nvcuda.dll, cudart64_12.dll from the runtime wheel's `bin`, nvml.dll,
  amdhip64_6.dll, ze_loader.dll, OpenCL.dll). Written and tested with Windows simulated: the project has no Windows
  machine yet.
- **`python -m nbxtensor doctor` on a Mac without Triton exits 1**: it named the fix and exited 0, so a script could
  not tell that nothing could run yet.
- **The library imports on Windows**: its page size came from `os.sysconf`, which Windows does not have; and the code
  generator's identity now reads a backend's files as `/` paths with its `.pyd` and `.dll` binaries, so the same code
  has the same identity on every system.
- **A certified setting that is not served says why**: where the directory holds a key proven under another code
  generator than the running one, the line named a memory class that matched ("certified for 18 GB, this card is
  18 GB, not served"); it now names both generators and the re-proof command. A setting proven on another memory class
  is still said as such.

## [0.0.4] - 2026-11-07

### Added
- **Triton for the Apple GPU in one command**: `python -m nbxtensor install-metal` installs the Metal wheels of the
  project's fork NeuroBrix/triton-ext (Triton built with triton-ext, and imperatormk's AppleGPU backend as
  `nbx-triton-apple`, triton-lang/triton-ext#126), each by the release URL and sha256 written in the library,
  through pip with `--no-deps --require-hashes --no-index` — never an extra index nor a package name resolved on PyPI. The
  wheels are for Python 3.12 and macOS 15 or later (release apple-v0.1.0+triton3.8.0); it refuses by name off an
  Apple-silicon Mac, on another Python, or below the macOS the wheels are built for. `python -m nbxtensor doctor` on a Mac
  without it says so and names the command; the Metal backend's refusal names it too. On a Mac, nbxtensor depends on
  `pyobjc-framework-Metal`, through which it opens the Metal device; without it the run fell to a CPU profile and was
  refused for triton-cpu, it is now refused by name with its install line.

### Fixed
- **nbxtensor installs on a Mac**: it depended on `triton` on every platform, and PyPI publishes no Triton for macOS, so
  `pip install nbxtensor` — and with it the NeuroBrix engine, even its PyTorch engine on MPS — could not resolve on a
  clean macOS environment. Triton is now a dependency where PyPI has it; on a Mac it is built from source with its Metal
  backend (the README's install), and the library says by name what to build when it is absent.

## [0.0.3] - 2026-11-06

### Fixed
- **The backend is the one whose card is on the machine**: nbxtensor depends on the CUDA runtime wheel on Linux, and a runtime library loads without its card — on an AMD or Intel machine 0.0.2 named the CUDA backend and never tried HIP or Level Zero. Detection now tries only the runtime libraries of the GPU vendors present on the PCI bus (all of them where the bus cannot be read). The CUDA runtime wheel stays a Linux dependency: without an NVIDIA card it is not used.
- **The package summary says where it runs**: wherever Triton runs — NVIDIA, AMD, Intel and Apple GPUs (it still named NVIDIA and Apple only).

## [0.0.2] - 2026-11-06

### Added
- **`backend.launcher.prewarm_compiler()`**: computes Triton's cache key (`triton_key`, the sha256 of libtriton and of Triton's sources, about 1.3 s on a V100 host) in a daemon thread, so a process's first kernel compile no longer waits for it; an application calls it at start, while it loads weights.
- **Intel GPUs through Level Zero** (`backend.level_zero`, `backend.level_zero_driver`; the `xpu` backend, chosen when the system's Level Zero loader reports a GPU): a runtime over Level Zero that serves the allocator (device and page-locked host memory, in-order command lists as streams, timed events), and a launch driver for the SPIR-V kernels Intel's Triton build compiles, with Intel's own argument ABI; the compile target is built without torch or SYCL (Level Zero's counts, the OpenCL ICD's extensions, the device file's architecture name). Proven against stand-ins of libze_loader and the OpenCL ICD: no Intel card has run it yet.
- **What to install when a GPU is present and unusable** (`backend.install_help`, `python -m nbxtensor doctor`): for every NVIDIA, AMD or Intel GPU on the PCI bus, whether its runtime library and its Triton backend are installed, with the vendor's install commands when one is not; the refusal when no GPU runtime answers carries the same steps instead of a bare error. The README's "Install on AMD / Intel GPUs" section gives them.
- **Device files** for AMD CDNA4 (gfx950), RDNA3 (gfx11xx) and RDNA4 (gfx120x), and Intel Data Center GPU Max (pvc), Arc A-series and Flex (dg2), Arc B-series (bmg) and Lunar Lake (lnl), every value with its source.
- **A HIP launch driver for AMD GPUs** (`backend.launcher.HipDriver`, chosen when the runtime detects HIP): the HIP module API through the HIP runtime library the runtime already loaded, the launch Triton's own AMD launcher performs (hsaco code objects, `hipModuleLaunchKernel`, the kernel's arguments then the scratch pointers, a block of the code object's wavefront times the kernel's warps), one module per device and code object, the argument count read from the code object's own metadata and a list of another length refused, shared memory past the device's per-block maximum refused as Triton's own AMD launcher refuses it, the wavefront read from the code object and the device file (never defaulted), the launcher contract's `compile` surface shared with the CUDA driver, every missing symbol refused by name; page-locked host memory is asked of `hipHostMalloc` with its flags word. Its tests compile real Triton kernels for gfx90a, gfx942, gfx950, gfx1100 and gfx1201 on a machine without an AMD card and launch them through a stand-in HIP library: no AMD card has run it yet.

### Removed
- **Nine environment switches that changed a kernel**: `NBX_MM_MAX_OUTPUT_ELEMS` (the matrix product's output split, fixed by an int32 element offset), `NBX_FLASH_BLOCK_M_VOLTA` (Volta's flash Q tile, 64, the measured floor of correctness), `NBX_CONV3D_EAGER_FREE_BYTES` (when a 3-D convolution frees its buffers early), `NBX_FLASH_NUM_WARPS` and `NBX_FLASH_NUM_STAGES` (the flash kernel's launch, which its device file's flash rows state), `NBX_DGEMV_ADAPT` (the dequantising GEMV's width adaptation, set by `dequant_gemv.min_programs`), `NBX_D128_DETOUR` (the power-of-two head-dim detour, decided by `flash.pow2_head_dim_correct`; `launch_keys.flash_headdim_detour` loses its `enabled` argument), `NBX_DEPTHWISE_DISABLE` (the depthwise convolution route) and `NBX_GC_ON_OOM` (a garbage-collecting retry of a refused allocation, which is raised, never rescued). A value a kernel needs is a fact of its device file or of a certificate, never a host setting.

### Changed
- **A certified file is parsed and validated once, then read from its compiled index** (`ops.certified_index`): the directory stays JSON, the source contributors edit; the first launch after a change validates a file and writes, per memory class, a binary index of the entries that class is served (length-prefixed keys and certifications, a SHA-256 trailer, never a pickle), named by the fingerprint of everything the validation read; later launches read the index and parse a certification only when its key is asked. A corrupt index is refused by name.
- **The launch recorder receives each pointer's extent** (`backend.launcher.set_launch_recorder`: `fn(prepared, grid, params, extents)`): the byte extent of each pointer's tensor that a driver binding by extent is handed, None for a driver that binds by address — a replay can now launch exactly what was launched.
- **The README opens with what the library is and `pip install nbxtensor`**: the 0.0.1 README began its installation with `pip install -e nbxtensor` from the engine's repository, which PyPI showed as the package's description; it now opens with its title, what it does, the install from PyPI (the repository install is below, for contributors), a GPU example its tests run, why the library exists (no PyTorch, any Triton GPU, the V100's tensor cores back in use with their measured gains, planned memory, certified tuning) and the hardware tested and expected, and it links its API and backend references in the NeuroBrix repository, where they live (they were relative links to files the library's repository does not hold). The package's one-line description no longer names a CPU backend, which the library does not have.

### Fixed
- **CDNA3's shared memory per block is 64 KB**, not 160 KB (the device file had CDNA4's figure; Triton 3.8's AMD target features state 64 KB for gfx942): an autotune config needing more is no longer kept for an MI300.


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

[Unreleased]: https://github.com/NeuroBrix/nbxtensor/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/NeuroBrix/nbxtensor/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/NeuroBrix/nbxtensor/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/NeuroBrix/nbxtensor/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/NeuroBrix/nbxtensor/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/NeuroBrix/nbxtensor/releases/tag/v0.0.1
