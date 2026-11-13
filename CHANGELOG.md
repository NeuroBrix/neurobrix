# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`neurobrix run --profile-kernels` shows where the GPU time goes on Apple GPUs.** The run prints its kernels
  ranked by GPU time (and the copies as one row); `--profile-kernels PATH` also writes the whole table as JSON. Each kernel is
  timed on its own, so the table ranks kernels and the run is slower while it is measured; the answer is the same.

### Changed

- **A served conversation on Apple Silicon plans its prompt in chunks priced on the chip's own tables.** The serving
  session reads its chunk record and certified times under the chip's vendor profile, and a Mac's single memory pool
  is planned as the one cache it is, as on other hardware.

## [0.6.15] - 2026-11-13

With nbxtensor 0.1.7: Kokoro-82M speaks with its own voice on the Triton engine, `neurobrix upscale` runs on every host as
`neurobrix run` does, `neurobrix import` installs only the container the hub states, and large models run and serve within
the memory their plan priced.

### Changed

- **A model larger than its card runs on the Triton engine within its plan.** When a model's layers stream from host
  memory, the weights loaded on the card from the start (embeddings, norms) count against the layers the card keeps:
  Qwen2.5-Coder-14B runs on one 16 GB V100.
- **Long served prompts run on the PyTorch engine with the memory their plan priced.** The check that a model's attention
  mask is the causal pattern runs once per prefill chunk, reading the mask in bands no larger than the mask itself:
  Qwen3-30B-A3B-Thinking and Qwen3-Coder-30B-A3B serve every request with `--compiled` on one 32 GB V100.
- **Kokoro-82M speaks with its own voice on the Triton engine.** A model built in float32 and run in half precision keeps
  float32 wherever its own forward keeps it, unless the model's calibration record measured that op safe in half: the
  voice's harmonic source (a running phase of thousands of radians) is computed whole, and the Triton engine's voice
  matches the vendor's reference render and the PyTorch engine's.
- **`neurobrix import` installs only the container the hub states.** The downloaded `.nbx` is checked against the
  sha256 the hub publishes for it, whatever host served the bytes; another archive is refused by name with both
  values, kept beside the store as `<name>.nbx.unverified` and never extracted.
- **`neurobrix upscale` chooses its engine as `neurobrix run` does**, on every host: the same `--compiled`,
  `--sequential`, `--triton`, `--triton-sequential` flags, the PyTorch engine where it is installed and the Triton engine
  where it is not when no flag is given, said on the launch line. On a host without a GPU, `upscale` runs an upscaler on
  the PyTorch engine as `run --input-image` does. `--mode` stays accepted as the same choice.
- **A language model split across two cards scores every sequence of a batch.** When the last layer and the output head
  sit on different cards, each sequence's last position reaches the head through the tensor library's card-to-card
  transfer, laid out per sequence.

## [0.6.14] - 2026-11-13

With nbxtensor 0.1.6: served multimodal and vision-language models hold a whole conversation on the Triton engine, a
served mixture-of-experts model is ready about 30 % sooner, and the voice reads numbers and names as written.

### Changed

- **The voice reads every number and name as written.** Kokoro-82M now writes out numbers, years, versions, decimals,
  ordinals, percentages, currencies, units and known names before it speaks them ("a graphics card from 2017" is read
  in full); a character it cannot say is refused by name.
- **Each served build answers the modes it was built for.** Janus-Pro-7B built for images serves image generation; a
  request for a mode the container was not built for is refused by name, over HTTP as on `neurobrix run --mode`.
- **A served multimodal model keeps one copy of its weights for its whole life.** Janus-Pro-7B serves image after image,
  and any request after them, from the language model loaded at start.
- **Served vision-language models hold a conversation on the Triton engine.** Ming-Lite-Omni-1.5 and the models on the
  same path keep their attention cache across requests, so every turn is answered over its full context.
- **A served mixture-of-experts model is ready about 30 % sooner on the PyTorch engine.** Qwen3-Coder-30B-A3B on one
  32 GB V100 is ready in 545 s instead of 769 s: the model graph is prepared once per start.

## [0.6.13] - 2026-11-12

With nbxtensor 0.1.5: the Triton engine on Apple GPUs several times faster, and the engine's remaining tensor work moved
into its library.

### Changed

- **Triton models on Apple GPUs run several times faster.** nbxtensor runs each compiled kernel on its own Metal stream,
  in order, and waits for the GPU only where a result is read, instead of after every kernel. Measured on an M4 Pro,
  outputs byte-identical: TinyLlama 1.1B, 256 tokens, 46.2 s -> 8.1 s; granite-3.1-1b-a400m (mixture of experts), 128
  tokens, 22.5 s -> 4.7 s.
- **The engine is lighter: its tensor work is nbxtensor's.** The mixture-of-experts layer, the tiling engine's tile
  reads and blends, the allocator's deferred frees and release order, device and storage measurements, the samplers'
  arithmetic, the speech-to-text decoder's joint network (now on the device), the certification harness and the
  census's stand-in device all live in the library now; the engine calls it. 83 477 -> 80 938 code lines
  and 321 -> 172 places doing tensor work in the engine (measured, 0.6.11 -> 0.6.13), outputs unchanged.
- **A tensor handed from PyTorch to the Triton engine** (an input image, a conditioning, a tokenizer's ids) **arrives as
  the values it shows**, on every backend: a bfloat16 tensor on the CPU, a tensor on an Apple GPU in the image and video
  flows, and a non-contiguous one (an image in channels-last order), copied in the order it shows.

## [0.6.12] - 2026-11-12

With nbxtensor 0.1.4 unchanged: a large mixture-of-experts model served on one 32 GB card holds a whole conversation.

### Changed

- **A large mixture-of-experts model served on one 32 GB card answers every turn of a conversation.**
  deepseek-moe-16b-chat served with the Triton engine on one V100-32GB: the plan sizes the served context beside the
  engine's freeing queue, and a model that leaves no room for both keeps its experts in host memory behind a cache on
  the card, as on a 16 GB card.

## [0.6.11] - 2026-11-12

With nbxtensor 0.1.4 unchanged: the package carries every certified-settings table, and a missing one is named.

### Changed

- **The package carries every census table.** The wheel and the source archive hold the census tables, chunk records and
  refused-settings registers (the `.jsonl` files), so a served session on an installed engine finds the certified
  prefill chunk of its model. A release is built only when every file the engine tracks is in the built packages.
- **A missing census file is an error naming it**, never an empty table: a certified machine whose census files are
  absent is told to reinstall neurobrix; a machine or memory class that has no census is said so.

## [0.6.10] - 2026-11-12

The library takes back what is its own: nbxtensor 0.1.4 holds the engine's replay, its mixture-of-experts store and its
tensor helpers, and the engine calls it; plus long prompts tokenized in linear time and a server that reuses what this
machine already measured.

### Removed

- **The terminal coding agent (`neurobrix chat --agent`, `neurobrix run --mode agent`) leaves the engine**, with its
  flags (`--workdir`, `--max-turns`, `--bash-timeout`, `--allow-network`, `--approve-all`, `--yolo`,
  `--transcript-dir`). Coding agents will be their own project (NBX Code, on the roadmap), speaking to
  `neurobrix serve` through the OpenAI-compatible API, where forced tool calls stay honoured.

### Changed

- **`/v1/audio/speech` without a `voice`, for a model that ships several and declares no default, is a 400 naming
  `voice` and the voices**, before anything runs.
- **A new server process reuses the kernel settings this machine already measured**, for every kernel it loads: those
  loaded before the first computation and those a server loads at its first request (the paged attention of a
  conversation, the decode kernels of a mixture-of-experts model), so its first answers come without measuring again.
- **`neurobrix run` refuses by name a prompt longer than the model's context**, before anything loads, with the prompt's
  token count and the model's context, as `neurobrix serve` does.
- **A run whose card reports its error late exits with that error named**, rc 1 (`--compiled` on a Mac: a device-side
  index error reported at exit).
- **`neurobrix run --turns` states each turn's own timings**: every turn's `[Timings]` line gives that turn's time to
  first token, load time, rate and time per token.
- **A long prompt to a Llama-tokenizer model is tokenized in linear time, with the same tokens.** TinyLlama's
  12 265-token prompt: 25.7 s -> 0.03 s (4 078 tokens: 3.0 s -> 0.01 s), on every platform, in `run` and `serve`.
- **A model with thousands of added tokens reads a long prompt at full speed, with the same tokens.** orpheus (28 940
  added tokens), a 34 880-character prompt: 43.6 s -> 0.02 s, in `run` and `serve`.
- **`serve` keeps a bounded tokenizer cache.** A Llama-tokenizer model's prompt is one word for the tokenizer; the cache
  holds words under 256 bytes, at most 10 000, as the reference library's does. TinyLlama, 50 distinct 32 276-character
  prompts: 30.9 MiB held -> 0.0 MiB, the same tokens.
- **A driver error at start-up on a machine with several cards is said at once**: when the driver cannot answer which
  cards can read each other's memory, the run stops there, naming the call, the card and the driver's status.

## [0.6.9] - 2026-11-12

With nbxtensor 0.1.3 unchanged: a served mixture-of-experts model answers every turn of a conversation, and the server
reads GLM's and Mistral's tool calls, renders chat templates as their vendors do and answers each streamed request with
its own status.

### Added

- **Two more tool-call formats in the server: GLM's and Mistral's.** A model whose chat template writes
  `<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` (GLM-4.7-Flash) or
  `[TOOL_CALLS]NAME[ARGS]{…}` (Devstral Small 2) now has its calls returned as `tool_calls`, streamed or not. Each
  format is read from the template's own markers, as before. A forced call (`tool_choice: "required"` or a named
  function) is written in the form the parser reads back. Mistral's `[ARGS]` is a special token in its tokenizer, and
  the decode keeps it. A call that does not read as one is kept in `content`, whole. Tested against each vendor's own
  template: a call it writes reads back the same. After a thinking model's reasoning, a forced call or a JSON answer
  now starts with what the model's own template writes there (nothing for GLM-4.7-Flash; the two new lines of Qwen3's
  templates, as before), read from the template. The models themselves join the catalogue when their
  containers are built.

### Changed

- **A streamed request that stops before its first token answers with its own status.** The OpenAI-compatible server
  opens the event stream with its first event; an error before it is a plain JSON error with its status (`500` for an
  engine error). An error after tokens have streamed is an event in the stream, as the protocol requires.
- **A served mixture-of-experts model answers every turn of a conversation.** When its experts stream from host memory
  (one card too small to hold them all, e.g. Qwen3-Coder-30B-A3B on one 32 GB card), the server sees every change the
  decode made to the card's experts before a turn is read. Measured on a five-turn conversation: the answers are
  byte-identical to the same server run without its fast decode path.
- **A chat template renders the prompt its model was trained on.** Templates render in the environment their vendor's
  reference uses: `tojson` as `json.dumps` (characters as written, `ensure_ascii=False` honoured, as GLM-4.7-Flash's
  template passes it), the keys in the order the request gave them, `{% break %}` and `{% continue %}` as tags. A
  request's tool schemas and past calls reach the model as sent. The renderer is sandboxed: a template is data inside a
  container, and it reaches no Python object. Measured on the catalogue's 17 chat templates over one conversation with
  tools: 8 render differently from 0.6.8 and all 17 render; with no tools, all 17 render byte-identically. The new
  environment with 0.6.8's filter put back gives 0.6.8's text exactly, in all 34 renders, so the filter is the only
  difference.

## [0.6.8] - 2026-11-11

With nbxtensor 0.1.3: a mixture-of-experts coding model served at full speed request after request and spread over two
cards, a dense model split over two cards, and a model streamed layer by layer on a small card.

### Changed

- **A served mixture-of-experts model keeps its full speed from request to request.** `neurobrix serve` on a model run
  through the expert store (experts larger than the card): a decode step replayed from its recorded plan puts the dense
  weights back on the card first, and a request's later tokens bind the weights once. Measured on Qwen3-Coder-30B-A3B,
  one 32 GB card, three successive requests to one server: 0.067, then 0.057 s a token (17.7 s -> 0.057 s a token from
  the second request on).

- **An answer ends with its last sentence.** Through the OpenAI-compatible server, a bare opening tool-call marker
  (`<tool_call>`) written after the last sentence of a turn is left out of the content; a block with a body that does
  not read as a call is returned whole, as before.

- **A mixture-of-experts model whose plan spreads it over two cards runs on the Triton engine.** When the plan holds the
  whole model on two cards and a layer's experts sit on both (Qwen3-Coder-30B-A3B on two 32 GB cards), each card reads
  its own experts at their place, through nbxtensor 0.1.3's pointer tables, and the shares are added per token. The
  decode cache is sized against the part of the first card a component may hold, beside what that card already holds, on
  both engines.

- **A dense model split over two cards runs on the Triton engine.** GLM-4.1V-9B-Thinking on two 16 GB cards hands its
  tensors from card to card, through nbxtensor 0.1.3's expandable pool.

- **A model streamed layer by layer on a small card answers right.** TinyLlama on a 1 600 MB card, Triton engine: a
  replayed run mirrors exactly what its recorded run moved — the positions and the attention calls — so every layer's
  keys and values land in their own place.

- **The engine's messages speak of building containers.** A refusal or an error that named how a container was made
  says "build" where it used another word — "its build-time value", "a container built for --mode image", "rebuild its
  container": 77 messages reworded. No behaviour changes; a script that matched the old text of a message must follow.

## [0.6.7] - 2026-11-11

With nbxtensor 0.1.2: on Apple, every pinned address resolves inside its own buffer, and the expert store places its
tables on the card it names.

### Changed
- **Apple: the copying expert store reads every expert from its own buffer.** nbxtensor 0.1.2's Metal driver answers an
  address read through a pinned table from a buffer that covers it, and refuses by name one that none covers (its
  [changelog](https://github.com/NeuroBrix/nbxtensor/blob/main/CHANGELOG.md) has the mechanism). That store copies
  experts into an arena; on Apple the engine builds a unified device's store, which binds its experts in place. With the
  published 0.6.6 through `neurobrix serve`, on the two plans the mixture-of-experts model that fits a 24 GB Mac takes
  (resident, and streamed layer by layer), all two million pinned addresses observed resolved inside their buffer; the
  expert-store plan's models (31 GB) are beyond that Mac. CUDA is not concerned.
- **The expert store keeps its residency on the card where the backend's kernels read device addresses taken from the
  host** (CUDA and HIP): the backend answers that question by name, and the strategy asks it.
- **The Triton expert store places its five device tables on the card it names**, at the upload
  (`NBXTensor.from_numpy(..., device=)`); a test reading the code's syntax tree holds every upload to that form, a
  device held in a variable included. No result changes.

## [0.6.6] - 2026-11-11

### Changed
- **A long prompt sent to `neurobrix serve` is read several times faster on the PyTorch engine.** The daemon takes the
  largest slice of the prompt its memory plan holds without giving up anything it keeps on the GPU (128 tokens a slice
  in 0.6.5). The working memory a slice needs — it grows with the slice and with everything already cached — is one
  reserve the plan states and the engine keeps to, and the plan chooses the slice with it. Measured on two V100-32GB
  with Qwen3-Coder-30B-A3B: a 19 931-token prompt answered in 42 s instead of 322 s, a 79 931-token one in 9 min instead
  of 39 min, same answers. Starting the daemon takes up to six more planning passes — about four minutes for that model.
- **A prompt longer than the GPU's cache is read faster**: the part of a long prompt beyond the card's cache passes back
  through the GPU block by block (same result), within the reserve above. On the 79 931-token turn measured on two
  V100-32GB this takes off the CPU about half of each chunk past the card's 65 664 positions.
- **Reading a long prompt takes less GPU memory on the PyTorch engine**: each slice of the prompt reads the cached keys
  and values as stored, without a copy per attention head.
- **Intel Gen12 graphics and Wildcat Lake are tried** (experimental, not yet validated on hardware by the NeuroBrix lab
  — reports welcome): Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, DG1 and Wildcat Lake chips are described as
  devices when their Level Zero runtime is installed (the compile target's name does not change the compiled kernel:
  measured). Without the runtime they are named with what to install, and the model runs on the CPU. Gen9 to Gen11
  graphics are refused by name, under their own reason from the compute runtime.
- **An Intel GPU that refuses at run time is said by name, and the model runs without it** (the Intel backend is
  experimental, not yet validated on hardware by the NeuroBrix lab — reports welcome): when the Level Zero driver says
  the chip cannot do what was asked (a kernel it does not build, an unsupported feature or version), `neurobrix run`
  prints the refused call and the driver's reason (with its build log), remembers that chip model (its PCI id) as
  refused under that driver and nbxtensor version, and starts the request again once, planned without it (on the CPU
  when it was the only device), saying so in its first line and in the plan; `neurobrix serve` loads the model again
  once the same way (a request that met the refusal is answered with it and can be sent again). With `neurobrix run`, a
  profile given with `--hardware` is never replaced: the refusal says what to do (`serve` started with one answers the
  request with the refusal). Any other error (a lost device, memory) stays the run's own, named. Another driver or
  nbxtensor version tries the chip again. `neurobrix doctor` shows the label and what was refused; the new "Intel GPU
  report" issue template collects what happened on your card.
- **The grammar engine's source lives in its own repository only**,
  [NeuroBrix/nbxgrammar](https://github.com/NeuroBrix/nbxgrammar): the copy under `rust/` is gone from this one, and a
  source checkout uses the `nbxgrammar` wheel `pip install` brings, as an installed neurobrix does. The version
  installed with neurobrix is nbxgrammar 0.1.1 (pinned exactly, on the same five platforms): the same masks and the
  same speed as 0.1.0, see its [changelog](https://github.com/NeuroBrix/nbxgrammar/blob/main/CHANGELOG.md).
- Follows nbxtensor 0.1.1. Internal, no change of result: the rules that decide what constrains a chat request's
  answer (a JSON schema, a forced tool call) are grouped in one place, every request answered exactly as before (400
  combinations compared); the Triton engine's long-prompt path and its table of host expert addresses name the GPU
  they place data on in the call that places it.
- **NeuroBrix holds memory only on the GPUs its plan uses.** On a machine with several GPUs, a run's contexts and memory
  sit on the plan's cards alone, on both engines (about 300 MB left free on every other GPU; 306 MB on a V100 with the
  PyTorch engine): synchronisations name the plan's devices; the runtime's growth is read on the profile's cards; what
  crosses GPUs — a model's intermediate states — passes through host memory (since PyTorch 2.10, a tensor copied from
  one GPU to another initialises every visible GPU); the memory planner reads the driver's own tool; the weight loader
  prepares its uploads on the GPU written; and, without `--hardware`, the look NeuroBrix takes at the machine's GPUs at
  every launch touches none of them (what the runtime sees and takes on each GPU is read once, in a separate short
  process, when the generated profile is written or completed). Measured on four V100: a model across GPUs 2 and 3 holds
  memory on 2 GPUs (4 in 0.6.5); a model on GPU 2 of a two-GPU profile on 1 (2); without `--hardware` and two GPUs
  visible, a model planned on one on 1 (2). Results unchanged; a long prompt on a model split across two GPUs reads
  about 8 % slower (45.6 s against 42.3 s for 19 931 tokens), decoding is unchanged.
- **`neurobrix serve` plans memory for the request it will get**, by the same rule as `neurobrix run`: one item per
  request, doubled only where the model's guidance is in force, in the dtype the container declares — a chat model for
  one prompt, an upscaler for one image.
- **The memory plan reserves the projection onto the vocabulary for the last token only**, as both engines compute it:
  1.2 MB less per prompt token for a 151 936-token vocabulary, so a long prompt given to `serve` or `run` keeps the
  model's weights on the GPU. (A scoring request, `--set global.score_mode=1`, computes every position and is planned
  so.)
- **A chat model's output layer is counted once in the memory plan** on the PyTorch engine, as the engine holds it: a
  sixth of 0.6.5's reservation for a 151 936-token vocabulary.
- **A model has one name everywhere, and every message says it.** After `neurobrix import`, `run --model` with another
  name that leads to the installed model — its folder name, or the name typed at `import` when the hub records the model
  under another one (`Efficient-Large-Model/Sana_1600M_1024px_MultiLing` is
  `Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers`) — answers with one message naming the name to use. The
  "installed models" list prints the names `--model` accepts, and `import` says when the hub's name is not the one
  typed. Every model name in the documentation's commands is the name `run` accepts (the benchmark's commands, the
  guides' serving and removal examples, the upscaler table, and `neurobrix remove --help`).
- **`--reference-audio` and `--reference-image` are refused by name**, before anything loads: cloning a voice from a
  recording and conditioning on a style image are not wired in this version, and the command says so.
- **The "First launch on this machine: measuring it once" line is said at the first launch only.** A weight reload rate
  is kept from three reloads in a row that agree within 5 %; a later launch that has none kept yet for the model's
  stored weights (a machine that was not quiet, or weights of a kind the machine has not reloaded yet) says what it
  does: it measures the weight reload rate for that model's stored weights.
- **An older automatically generated hardware profile is completed at the next launch**: the two measured figures added
  since it was written (the GPU memory the runtime sees, and what the runtime itself takes on the card) are measured and
  added to the generated profile. A profile you wrote yourself is never modified.

## [0.6.5] - 2026-11-10
### Added
- **A served session states its context ceiling and reaches past the card into host memory**: `serve` prints "served
  context: up to N positions a session (... on the card + ... in host memory ...)" at start; past the card's KV cache
  the oldest pages move to host memory and decoding goes on (exact); a request longer than the ceiling is refused before
  it runs with the OpenAI error `400 context_length_exceeded`, naming the ceiling.
- **Constrained decoding on the OpenAI API: `response_format: json_schema`, `tool_choice: required` or a named function,
  `parallel_tool_calls: false` with a forced call**: the model can only write what the schema or the template's
  tool-call format allows, at every token, on both engines. JSON Schema: objects (properties, required; no other
  properties are generated), arrays (items, minItems, maxItems), strings, numbers, integers, booleans, null, enum,
  const, anyOf whose alternatives start differently, $ref into $defs; any other constraining keyword is refused by name
  (400), never ignored, and annotations (title, description, default, examples, ...) are read as annotations. A
  reasoning model answers after its reasoning ends, never inside it. `parallel_tool_calls: false` without a forced call
  is refused by name. Tool calls follow the format the model's chat template defines (`<tool_call>` JSON, Qwen3-Coder's
  XML, granite's list). The grammar engine is NeuroBrix's own, the `nbxgrammar` package (Rust, std only), installed with
  neurobrix where a wheel of it exists (Linux x86_64 and aarch64, macOS arm64 and x86_64, Windows x86_64); elsewhere
  these requests answer 501, by name.
- **A served conversation that leaves the KV cache comes back from host memory**: when another conversation takes the
  daemon's cache, the whole pages the first one held are parked in host memory and copied back when it returns, so its
  next turn prefills only what is new; the answers are the same tier on or off. The host memory it may hold is bounded
  by what the profile leaves (installed RAM less the system's share less the plan's host side), grown one conversation
  at a time, page-locked on a discrete card and pageable on unified memory; `serving.kv_host_tier_mb` (system.yml) caps
  it, 0 turns it off. The OpenAI usage reports the reused positions as `prompt_tokens_details.cached_tokens`.
- **An `embedding` model family and `/v1/embeddings`**, ready for the first embedding model: the vectors are the
  model's last hidden states pooled as its container declares (mean over the kept tokens, the first token, or the last
  token; normalised or not; its dimension); `input` a string or a list of strings, `encoding_format` float or base64,
  `dimensions` only the model's own. No model of the catalogue is an embedding model yet: the route answers every other
  family with a named refusal, and no vector is ever derived from a chat model.

### Changed
- **Contributions are made under the NeuroBrix Contributor Assignment Agreement** (docs/legal/ of the neurobrix
  repository, an individual and an entity version): signed once by a comment on the first pull request, for every
  NeuroBrix project; the `NeuroBrix CLA` check on every pull request says who has yet to sign. CONTRIBUTING.md explains
  it.
- **The hub's official address is neurobrix.org**: `neurobrix hub` and `neurobrix import` reach `https://neurobrix.org`
  by default, and the CLI texts, README and project URLs name it; the documentation is at docs.neurobrix.org, and the
  contact addresses (security, conduct, contributors, models, partners, the authors' e-mail in the package metadata) are
  @neurobrix.org. `NEUROBRIX_REGISTRY` still overrides it; the 0.6.x versions already installed keep working, the API
  under neurobrix.es stays served.
- **A long served prompt is prefilled in chunks on the compiled engine**: chunks of the rows the plan prices (128 for
  Qwen3-Coder-30B-A3B), so a prompt of 80 000 tokens never asks a 6 GB causal mask in one pass; it reaches past the
  card's cache into host memory and is answered: 39 minutes on two V100-32GB (625 chunks, the host's share of the
  attention included).
- **A mixture-of-experts language model on two cards runs at the cards' speed, served or run**: on two V100-32GB the
  plan holds the routed experts on both cards — Qwen3-Coder-30B-A3B decodes at 0.33 s a token through `run` (6.7 s a
  token with every block streamed from host memory). A served session keeps the model loaded between requests (a
  64-token answer: about 170 s -> 55 s), and its context cache is sized after the room the experts need (served: about
  31 s for 64 tokens; a served context of 65 664 tokens on two V100-32GB, with the experts on the cards). A hardware
  profile that does not state a card's measured runtime total and context is refused by name.
- **`neurobrix run` against a daemon ends at the daemon's answer**: when the daemon serving the asked model answers an
  error, that answer is the run's (exit 1). The daemon shows the model by its one name (`serve`'s line, `chat`'s
  status), and a socket path longer than the operating system's AF_UNIX addresses hold is refused naming the path and
  `NEUROBRIX_HOME`.
- **A video's VAE streamed in time returns each chunk's freed memory to the device**, chunk after chunk, so a first
  compiled run on an Apple GPU keeps its first plan (mochi-1-preview: 331 s -> 239 s). No output changes.
- **`neurobrix import` finds a model already installed from the same container**, by the container's sha256 the hub
  serves: the name record keeps it at install, and an entry installed under its container's own name is found before
  any byte is fetched. `--container FILE.nbx` installs from a container the machine already holds (or records the name
  of the entry installed from it), verified against the hub's sha256 before anything else is answered (a file of
  other bytes is refused naming both sha256, even when the model is already installed); nothing is downloaded and
  `--no-keep` never deletes that file.
- **A laptop with an NVIDIA card names its integrated GPU**: the detection goes through every backend, so an integrated
  Intel or AMD GPU beside an NVIDIA card is in the profile with its platform and the reason it is not planned (Intel's
  own reason for its chip, else: a plan runs one vendor's backend), and the NVIDIA card is planned as before.
- **A hybrid CPU's performance and efficiency cores are told apart on Linux and Windows**, as on macOS: Linux reads the
  kernel's hybrid PMU lists (`/sys/devices/cpu_core/cpus`, `cpu_atom/cpus`) and counts physical cores, Windows reads
  each core's efficiency class; the profile carries `performance_cores` and `efficiency_cores`, and the host's compute
  threads are its performance cores.
- **A request served by `neurobrix serve` re-plans after a device out-of-memory, as `run` does**: the component
  concerned is re-planned once on the memory really free and run again, the finished components' outputs kept, and the
  daemon then serves under the new plan (said in its log).
- **`neurobrix doctor` names macOS 14's Accelerate without thread control**, as `python -m nbxtensor doctor` does: the
  CPU pool's threads may each use Accelerate's own threads there, and `VECLIB_MAXIMUM_THREADS=1` is the user's choice.
- **A model whose output format the engine cannot resolve is refused by name on the warm path too**: the daemon and
  `neurobrix run` against a running daemon resolve the family's output format as the cold path does, so an image or an
  audio answer comes back as one, and a daemon that states no family is refused with the command that restarts it.

## [0.6.4] - 2026-11-09

### Changed
- `python -m nbxtensor install-metal` installs the macOS 14 wheels and CPython 3.10-3.13 that 0.6.3 announced, and the
  Triton engine runs on macOS 14: its CPU pool works without Accelerate's macOS 15 thread API (nbxtensor 0.0.9). Proven
  with Accelerate's macOS 14 BLAS simulated on a newer macOS (TinyLlama, whisper and Sana outputs equal by sha) and with
  all 238 Metal kernels of those three models compiled for macOS 14's Metal language (Metal 3.1,
  -mmacosx-version-min=14.0); no macOS 14 machine has run this release yet.
- The first launch with the Triton engine times a reload that holds a zero-size constant (Sana 1024's text encoder on a
  fresh machine): that constant's bytes are read through its count.

## [0.6.3] - 2026-11-09

### Added
- **An OpenAI-compatible HTTP API on `neurobrix serve`** (`--http`, default `127.0.0.1:8765`, from
  `config/system.yml`): `/v1/models`, `/v1/chat/completions` (streamed as Server-Sent Events, with tool calls parsed
  on the server from the format the model's own chat template declares, and a thinking model's reasoning in
  `reasoning_content`), `/v1/completions`, `/v1/images/generations`, `/v1/audio/speech` and
  `/v1/audio/transcriptions`, each served when the loaded model's family has that flow. It is the same daemon as the
  Unix socket — one model, one queue, one inference thread for both — so an OpenAI client and `neurobrix chat` share
  the warm weights. A parameter the engine does not honour (`stop`, `logprobs`, `n` > 1, a forced `tool_choice`, ...)
  is refused by name in the OpenAI error object, never ignored; a request beyond `serving.http_max_queued` waiting
  ones gets 429; an address other than loopback needs `--api-key-file`; a port already taken is refused naming the
  process that holds it. `/v1/embeddings` answers that no installed family declares an embedding output.
  Constrained decoding (a forced `tool_choice`, `parallel_tool_calls: false`) and embeddings are served from 0.6.5.
- **`neurobrix run --turns FILE`**: a JSON list of user messages played as one chat session through the same session
  code the daemon runs, each answer printed with the tokens its turn matched and prefilled — a multi-turn
  conversation validated through the entry point.

### Changed
- **A model has one name, its hub name** (`<org>/<name>`): `neurobrix import` records it beside the installed model —
  also when the model is already installed, without downloading — and `--model` (`run`, `serve`, `upscale`), `remove`,
  `neurobrix list` and the HTTP API use it. For a model whose name is recorded, the short directory name is refused,
  naming the hub name; there is no second name. A model installed before 0.6.3 keeps its directory name until `neurobrix
  import <org>/<name>` is run once for it, and `list` and every use say so in one line. `list --json` is schema 2:
  `name` is the one name, `entry` the directory. `calibrate`, `drift` and `inspect` take the one name too, and the
  documentation's examples use it.
- **Attribution**: author, maintainer and copyright holder Hocine Benkelaya (contact@neurobrix.es); a NOTICE file
  ships beside the LICENSE (supported and funded by Neural Network Holding Ltd, commercial distribution by WizWorks OÜ).
- **The PyTorch engine's memory reserve on a device whose runtime growth is measured is that measure**: where the
  compiled engine's growth during a run is a fact (measured for the model, or declared by the device's vendor file), it
  is the one reserve the plan keeps — the fixed private reserve is not taken a second time, and a component may use the
  whole budget instead of a fixed 92 % of it. Devices without the fact are planned as before.
- **A command prints what its user needs, and nothing else.** stdout carries the result only — the generated text, the
  transcript, `SAVED: <path>`, the table — and stderr a few status lines: the engine in use, the run time, a line when a
  first launch measures the machine once, a warning, an error in one line with its remedy. A successful `run` prints its
  2 or 3 lines of result (45 to 50 lines in 0.6.2); `import`, `hub`, `list`, `doctor`, `serve`, `chat` and `stop` print
  without banners, separators or an echo of the request, and `--help` lists the user's commands only (`--help -v` adds
  the developer ones). The engine's developer lines — the plan, the loaders, the replay, the kernel directory, the
  `[Timings]` and `[Peak]` lines — are shown with `-v` / `--verbose`, an error's full Python stack with `-vv`, both with
  `NBX_LOG=debug`; `import neurobrix` and `import nbxtensor` stay silent and install no logging handler. A serving
  daemon's log keeps every line.
- **`--json` prints its record on one line**; the content is unchanged.
- **The CLI reference is generated from the command-line parser**: `docs/reference/cli.md` is written by
  `tools/cli_reference.py` from `neurobrix`'s own parser — every command, the developer ones marked, every flag with its
  default and help — and a test holds the page and the parser equal. The sampling order, what a command prints, the
  storage layout and the debug variables are in `docs/cli_usage.md`.
- **`python -m nbxtensor install-metal` reuses the pinned Metal wheels already installed** (131 MB) and says so in one
  line.
- **The engine requires nbxtensor 0.0.8**: the paged KV cache, its page bookkeeping, a chat session's cache record and
  the prefill chunk's whole-page attention now live in the tensor library; the engine keeps the decisions (when a
  request is a session's turn, which chunk Prism plans).
- **The kernel census keys a chat session's later turns**, so a session's second turn runs under `--certified-only`: a
  turn after the first attends over the cached conversation, and the census derives those attention settings for the
  session's chunk (or, before Prism has chosen one, for every chunk-size candidate) at every cached length in whole
  cache pages up to the session's reach, and records for each candidate the settings Prism prices it from. Until a chunk
  is chosen, a later turn longer or shorter than a candidate names its missing setting.
- **Prism chooses a chat session's prefill chunk**: a session's turns can prefill in chunks of a fixed size, so a turn
  served from its cache gives the same bytes as the same conversation prefilled together. Prism picks the size from
  the certified kernel times of each candidate (whole cache pages, up to the prefill a session's plan prices), the
  cheapest whose kernels are all certified; with none certified a turn prefills in one chunk, and `--explain-plan`
  says why (`prefill_chunk_reason`). A one-shot `run` still prefills in one pass. The pricing asks the code generator
  once per plan.
- **A chat turn prefills only its new tokens**: the KV cache now belongs to the chat session (`serve` + `chat`,
  `complete`), on both engines. Each turn matches the conversation's token IDs against what the cache already holds
  and prefills only the rest (TinyLlama, a 421-token second turn: 158 rows prefilled instead of 421). A continued
  prefill attends over the cached keys read back as whole cache pages, so with a planned prefill chunk
  (`KVCachePlan.prefill_chunk`) a turn served from the session gives the same bytes as the same turns prefilled
  together. A daemon holds one model and one conversation's cache; its host tier comes in 0.6.5.
- **A conversation past its ceiling is refused, never summarised**: the session's ceiling is the plan's (its
  `session_reach`, else its KV cache), not the container's `max_length` or 4096. A turn that would pass it is refused
  with the ceiling and the plan fact it comes from; older turns stay whole, a tokenizer without a chat template is
  refused, and `/context` shows the exact token count, the ceiling and the tokens the cache holds (`tokens`,
  `cached_tokens`, `last_turn`; `estimated_tokens` is removed). The `chat` reply's `summarized` field is always `false`.
- **granite-3.1's tool calls are read**: its template's format, a marker then a JSON list, is parsed whether the marker
  comes as its special token or as plain `<tool_call>`.
- **A daemon serving a model that reloads its weights per request holds one copy of them**: the previous request's copy
  is released before the next request loads its own, so a daemon serving Qwen3-Coder-30B (58 GB of weights) holds one
  copy of them in host memory.
- **A daemon that stops says so**: an abnormal end writes every thread's stack to the daemon's log, and `neurobrix stop`
  names a daemon that was killed outright (which leaves no line of its own), with where its log is.
- **`neurobrix autotune certify --reprove-generator` adds its proof beside the others**: a certified directory re-proven
  under a new code generator keeps every entry's earlier proofs, and each generator is served its own certification, the
  published build included. `tools/certified_rename_generator.py`, `tools/migrate_conv_fp16_key.py`,
  `tools/merge_certified_class.py` and `tools/reproof_coverage.py` read every proof of an entry.
- **A run whose component meets a device out-of-memory re-plans it and goes on**: in `python -m neurobrix run`, on both
  engines, the run records the memory growth it measured, releases the device's cache, asks Prism for a plan again with
  that measure in force and runs the component again, keeping the outputs of the components that already finished (a
  video model's latents before its VAE); one warning line names the component and which plan replaces which. A component
  is re-planned once; when the new plan is the same or Prism refuses, the run stops with that refusal named. A first run
  on a new machine (mochi-1-preview on an M4 Pro) completes at its first launch. In `neurobrix serve` such a request is
  answered with the named refusal and the daemon stays loaded (its re-plan comes in 0.6.5).
- **A laptop whose GPU no backend drives runs on its CPU**: the first launch reads the machine with what the operating
  system ships — the PCI bus on Linux, Win32_VideoController on Windows, no vendor tool — and names each GPU with its
  platform and whether it is integrated. An Intel GPU a backend can drive (Lunar Lake, Meteor Lake, Arrow Lake, Panther
  Lake, Arc, Data Center Max, through Level Zero) is described as the device states it; an integrated one as unified
  memory, planned as one pool with the host the way Apple chips are. Any other GPU (Tiger Lake, Alder Lake, Raptor Lake;
  any GPU whose driver is not installed) is refused by name with its reason, and the plan says so and runs on the CPU
  with every core of the profile (a Tiger Lake laptop with Iris Xe graphics runs this way); nbxtensor's Level Zero
  runtime counts only the GPUs it drives. On Windows the machine's memory is read through PowerShell when `wmic` is
  absent (Windows 11 24H2). On an Intel Mac the integrated GPU is named from system_profiler and the engine runs on the
  CPU (the Mac's discrete GPU is refused by name the same way). A hybrid CPU's performance and efficiency cores are told
  apart on macOS (on Linux and Windows from 0.6.5), and on a laptop with an NVIDIA card the integrated GPU beside it is
  named from 0.6.5.
- **Every AMD GPU is planned with its real memory**, read from rocminfo's memory pools (stated in KB), from a 24 GB
  Radeon RX 7900 XTX to a 192 GB Instinct MI300X. An AMD APU (rocminfo "Memory Properties: APU": a Radeon 780M, a Strix
  Point or Strix Halo GPU) is described as unified memory, planned as one pool with the host like an integrated Intel
  GPU or an Apple chip; a GPU whose memory nothing states is refused by name.
- **`neurobrix serve` without `--hardware` serves on the GPU when it goes to the background on Linux**: the daemon
  resolves the machine's profile in its own process, after the fork.
- **`neurobrix import` of an installed model whose hub file carries a build stamp downloads nothing**: the check
  compares the installed model with the file's name without its stamp (28 GB saved for Wan2.2-I2V-A14B).
- **A mixture-of-experts model on an Apple GPU is planned with its expert misses priced**: a routed expert the store's
  cache misses is priced as the page-in it is, at the device's measured cold load rate, so Prism weighs the expert store
  against the other plans at its real cost (DeepSeek-Coder-V2-Lite).
- **A machine without a C compiler gets one line**: torch compiles some of its own kernels with Triton at their first
  use (torch 2.14.1 does it for `aten::bmm`), and Triton needs a C compiler to build them, which a slim container lacks.
  A run of the PyTorch engine there ends with one line naming the missing compiler, how to install one (`apt install
  gcc`), and `--triton`, whose engine needs none.
- **The first launch measures only what the requested model needs, and says so**: it times the weight reloads of the
  requested model's own stored types, and says in one line that this is a one-time measurement and how long it expects
  it to take, priced from what its first measure took on this machine. On a busy storage its reload probe reads
  serially, as the run does.
- **`import` of a model already installed downloads nothing**: the installed names are checked before any download, also
  when the hub record's name differs from the container's (`..._diffusers`; 9 GB saved for Sana).
- **`NEUROBRIX_HOME` (or `NEUROBRIX_CACHE`, `NEUROBRIX_STORE`) naming a directory that does not exist** is said in one
  line naming the directory and what to do, for every command, `--help` included (the full Python stack with `-vv`).
- **`doctor` with GPUs hidden by the environment** (`CUDA_VISIBLE_DEVICES=`) names the variable and how many GPUs the
  driver sees, says what each engine can run on (the CPU, or nothing), gives no torch advice, and exits non-zero only
  when neither engine can run (a torch that sees no CUDA beside a running Triton engine is advice, not a blocking
  problem).
- **`doctor` lists the GPUs the process can see**; the others on the host's PCI bus are counted as not visible here (one
  of four in a container given one).
- **A first run compiles without a Triton warning**: the masked-fill kernel gives `tl.where` a boolean condition (the
  compiled code is identical).
- **Cold serving / `run --turns`: a second request on a model released after the first computes as the first did**
  (Triton engine): a compile after the first keeps the precision contract the first compile resolved, so the model's
  calibration record holds and its activations keep their precision and speed; a model whose compile is kept binds again
  exactly the constants its first load held in fp32.
- **Cold serving / `run --turns`: a lazy plan's model is released through its plan at the end of each request**: a text
  model whose plan loads it on demand (a lazy or streamed plan) holds one load at a time, request after request, on a 16
  GB card as on a 32 GB one (deepseek-moe-16b-chat).
- **Cold serving / `run --turns`: the fp32 constants stay fp32 after a reload** (Triton engine). A component whose
  weights are released after a request and loaded again — every request after the first on a cold-serving plan (a lazy
  or streamed plan in `neurobrix serve`), each turn after the first of `run --turns`, any second request on one executor
  — binds the constants its plan holds in fp32 (normalisation weights) in fp32 again, under the plan's own precision
  contract, so every call reads them as they are and a replayed decode finds its slots (TinyLlama, the second turn of
  `run --turns`).
- **The kernel census writes no lock file into the package**: the table's write lock is taken on its directory and
  creates no file.
- **A prompt longer than the release gate's runs on certified kernel settings from its first run**: the plan states how
  far a session can reach on the device (`--explain-plan`: `session_reach`, within the model's declared window) and the
  census keys every prompt length from one token up to it. A saved plan keeps its KV cache's exact size and first
  request size when it is read back.
- **A language model's prompt is priced at its own length**: the prompt is counted by the flow's own rule (chat
  template, specials) and bound to the language model's sequence; the cache's prompt margin (128 tokens) is the bound
  only when a request states no prompt. A short prompt is priced below that margin, so a few plans of mixture-of-experts
  models grant their expert cache about a megabyte more.
- **Mixture-of-experts models on Apple GPUs stay within the expert cache Prism planned**: on unified memory the expert
  store reads a routed expert in place from the model's files, and the one-token decode path releases its pinned buffer
  over each expert it read before the store lets the pages go (DeepSeek-Coder-V2-Lite on an 18 GB Mac).
- **The macOS 14 Metal wheels (CPython 3.10-3.13) are published** at
  https://github.com/NeuroBrix/triton-ext/releases/tag/apple-v0.1.0%2Btriton3.8.0-macos14 and install by hand; in 0.6.3
  `python -m nbxtensor install-metal` installs the macOS 15 / CPython 3.12 build, and it selects the macOS 14 wheels
  from nbxtensor 0.0.9 (0.6.4).

## [0.6.2] - 2026-11-07

The Apple certified directory is complete under the published Triton build for Apple GPUs: 18 552 kernel settings are
certified, among them the settings 0.6.1 left to a first-run tuning, and the arena probe never tunes a kernel. The
kernels that compute a model on NVIDIA are 0.6.1's.

Eight Apple kernel settings are refused by size on an 18 GB Mac (the certifier's synthetic draw and its oracle do not
fit beside each other there): Real-ESRGAN x8's fp32 convolution at 8192 x 8192, Sana 4K's VAE convolutions, an RMS norm
of 16M x 128 rows, softmaxes of 25M x 1024 and 25M x 256 rows, and one bf16 convolution (48, 192, 480, 832). A model
that launches one of them tunes it once at its first run and keeps the setting; of the models the release trial runs,
Sana 4K is the one that does.

The Triton arena factor stays declared per hardware family (`vendors/family_facts.yml`): a measure on a user's machine
holds a fixed part and a part proportional to the activations, which one factor does not describe, so no measured
factor is written into a profile.

### Changed
- **The kernel certifier prices the size it draws**: on a unified-memory device (Apple) a key is drawn at the size its
  phase price fits, and the door before the draw prices that same size: the open element-wise class (its label, 68
  719 476 736 elements) is priced at the size it draws on an 18 GB Mac, not at its label.
- **The kernel certifier holds only what its oracle reads**: a row-windowed oracle (softmax, layer and RMS norms, group
  norm) reads three windows of rows; the draw is uploaded once and only its windows' rows stay on the host, which
  matters on a unified-memory Mac, where the draw and its device copy share one pool.
- **The Apple certified directory holds every setting the reference models launch on an 18 GB Mac but eight**: under the
  published Triton build for Apple GPUs, the 22 settings 0.6.1 tuned once at a first run and the second pass's are
  certified, 18 552 in all, with two of chatterbox's attention settings in the bf16 its run launches them in.
- **The arena probe never tunes a kernel**: run as a tool, it reads only certified kernel settings; a request that would
  need a setting the directory does not serve is the measure's bound, never a tuning at run time.

## [0.6.1] - 2026-11-07

A first run on a new machine ends with its result or a one-line message naming the cause, the package names none of the
project's machines, and the documentation says where NeuroBrix runs and how to start. The kernels that compute a model
are 0.6.0's: its final gate is carried, and five reference models through the entry point give the same outputs; what
changed is how a new machine starts (its first-run measures, its arena factor, its home), what the packages carry, and
Windows.

The Apple certified directory is proven again under the published Triton build for Apple GPUs (18 352 kernel settings);
a few Apple kernel settings for some models are certified in 0.6.2; until then they are tuned once at first run and
kept.

### Changed
- **The README and the documentation say where it runs and how to start**: Linux, Windows and macOS; on a Mac,
  NeuroBrix's Triton build for Apple GPUs (NeuroBrix/triton-ext); every `neurobrix import` example carries
  `--accept-license`, which the import asks of a model with a license.
- **The arena probe takes the device's refusal as its bound**: it measures the Triton arena's hold on the cache's
  smallest model by doubling its request while a prediction says the next fits; the largest request that fitted is kept,
  and the probe's own error is said by name.
- **A new machine's first run uses a declared Triton arena factor**: the factor is declared by the device's vendor file,
  else by its family (Apple 1.477, measured on an M4 Pro; NVIDIA 1.35, measured on Volta; AMD and Intel the NVIDIA value
  until measured), so a first run never probes it and Prism reserves the activation memory the machine needs from the
  start (on a fresh Mac the factor is about 1.5); a value an earlier first launch wrote into a profile is not read. The
  probe scales only what the priced request scales (a tool, until it is bounded to certified shapes).
- **A run starts on Windows**: the plan's resident term is read from the process's working set (GetProcessMemoryInfo),
  and a first-launch read-rate measure Windows cannot make (no way to read past its page cache) is declared unmeasured,
  so the run reads serially, said on its launch line. Written and tested with Windows simulated: the project has no
  Windows machine yet.
- **Text in UTF-8 on every platform**: a generated text written to a file, the vendor tokenizer files read at load, the
  agent's file edits and transcript, and the console output use UTF-8 where the platform's default code page is another
  (Windows' cp1252); a diagnostic dump's file name drops every character a file name refuses on some system; a stale
  install lock is asked of the kernel on Windows (signal 0 is Ctrl+C there).
- **The engine depends on nbxtensor 0.0.5**: a certified setting the library does not serve says why (another code
  generator, or another memory class).
- **A configured home is followed everywhere**: with `NEUROBRIX_HOME` set, the replay cache, the license acceptances,
  the hardware profiles, the calibration records, the render checkpoints and the serving daemon's files all follow the
  configured home.
- **`neurobrix doctor` on a Mac without PyTorch** says `pip install torch`, PyTorch's own macOS wheel.
- **The package names none of the project's machines**: the snapshots root is `~/hf_snapshots` (`NEUROBRIX_HF_SNAPSHOTS`
  overrides it); the census input records name the files they read relative to the engine's home (`home:`); the
  certified directory's proofs carry no host name (85 723 removed, and the certifier writes none).
- **A first run on a busy storage reads serially**: at its first launch the engine measures how many parallel reads the
  machine's model cache serves best; where no stable read rate is found (a storage another reader is using — a clean
  container whose cache is a network mount), the run reads serially, its launch line says so ("io workers: 1 for this
  run — ...; measured again at the next run"), and the profile keeps no count, so the next run measures again.

## [0.6.0] - 2026-11-07

The release that moves NeuroBrix's Triton engine onto nbxtensor (0.0.4). Its final gate ran the 49 catalogue models
through the entry point in both engines on both V100 card classes, every output judged by eye and every time in the gate
register; that gate is this release's benchmark, per model.

- **The Triton engine imports nbxtensor, not PyTorch**: `--triton` runs on nbxtensor, Triton and NumPy; PyTorch is an
  optional dependency, and the engine, the library and five reference models run in an environment where torch is not
  installed.
- **Every kernel setting is certified per launch key and per card class**: the keys a model forms are derived from its
  graph without running it, certified once against a float64 oracle, and served certified-only; a missing key is an
  error naming its census row.
- **The V100's tensor cores (m8n8k4) carry the matrix products where certified faster**: GEMMs, the mixture-of-experts
  prefill projections, fp32 1 x 1 convolutions and transposed 1-D convolutions.
- **Apple Metal and more GPU vendors on the Triton engine**: nbxtensor's Metal driver runs the engine on Apple GPUs with
  its own certified directory and a vendor file of measured facts, installed by `python -m nbxtensor install-metal`;
  nbxtensor's HIP and Level Zero drivers (written, not yet run on a card) come with vendor files for AMD CDNA4, RDNA3 and
  RDNA4 and Intel Data Center GPU Max, Arc A and B and Lunar Lake, and `neurobrix doctor` names, for every GPU on the PCI
  bus, the runtime library and Triton backend it lacks and the vendor's install commands.
- **The light install is the Triton engine alone**: `pip install neurobrix` is 1.48 GiB installed,
  `pip install "neurobrix[pytorch]"` adds the PyTorch engine (6.82 GiB); a run with no mode flag takes the engine that is
  installed and says which on its launch line.
- **Prism plans from the machine's facts**: the hardware profile and vendor files carry what it prices (measured
  reserves, reload rates, workspaces), a busy machine is re-planned on the memory really free with a clear warning, a
  model is refused from one place only when no plan fits, and a PyTorch-engine load is priced as the loader performs it.
  The TilingEngine cuts only where the graph proves the cut exact and refuses what it cannot place.
- **Mixture-of-experts models use every card**: a card that only holds experts takes every whole expert block that
  fits (Qwen3-30B-A3B on four 16 GB V100: 40 of 48 expert layers on the cards, 186 ms a token), the decode can replay its
  card-held layers (opt-in), and the expert store records each model's routing reuse beside the calibration records.
- **Certify your GPU and share it**: the README's four steps and the `certificate` pull request template.

## [0.5.4] - 2026-09-18

NVIDIA (CUDA) and CPU; Apple Silicon follows in 0.5.5.

- **Machine-readable output for clients such as NeuroBrix Studio.** `--json` on every read command (`info`, `list`, `hub`,
  `inspect`, `coverage`, `doctor`, `autotune status`, `autotune check`, `run --explain-plan`) writes one JSON record on
  stdout; `neurobrix import --json` reports its phases and byte counts as they happen.
- **A serving daemon a client can identify and follow.** Every response carries the protocol version and the engine, the
  daemon's `status` and `neurobrix info --json` share one identity record, and each token event carries its decoded text.
  A daemon instance named by `NBX_SOCKET_PATH` runs beside others, on Windows too.
- **`neurobrix run --explain-plan`** prints the placement decision (the strategy, the ones it beat or refused, each
  component's device and memory split, the KV-cache budget) and exits without loading anything.
- **Certified kernel settings served by memory class and generator.** `neurobrix autotune certify` records the card and the
  clock a proof was made on, with `--only-missing`, `--reprove-generator` and `--reprove-unclocked`; `autotune check
  --restamp` and `autotune status` read the directory by backend.
- **Tools to compare the two engines.** `neurobrix drift` locates where they part and classifies the origin; the planned
  ops, initial noise, final latent and decoded frames of a request can be written out; `NEUROBRIX_REGISTRY` names the
  registry a machine talks to; `neurobrix import` resumes an interrupted download.
- **Models.** Next-token-diffusion speech models (VibeVoice) run their language model as a KV-cached decoder; the Triton
  engine runs without loading torch, and image models on V100 keep the vendor's float32 islands, measured by
  `neurobrix calibrate`, now also on the Triton engine (`--triton`). The first Apple Metal device and allocator land.

## [0.5.3] - 2026-09-03

A clean `pip install neurobrix` runs a first model on the machines it targets, from install to first generation.

- **A first run on a new machine.** `neurobrix doctor` checks that the engine can use the GPU it finds, `CUDA_VISIBLE_DEVICES`
  chooses the GPUs, AMD Instinct MI200 and MI300 cards run, and a GPU the engine does not support is named.
- **Long recordings are transcribed in full** by the audio language models and the speech-to-text models, with RNN-T / TDT
  decoding following the vendor's greedy algorithm.
- **Upscalers run through the serving daemon**, with the same input preparation as `neurobrix upscale`; Flex.1-alpha
  generates images; Chatterbox and DeepSeek-MoE generate on the Triton engine.
- **Faster decoding on the Triton engine.** Mixture-of-experts generation, quantized single-token projections, Volta
  decode attention and `top_k` sampling run on purpose-built kernels; freed GPU blocks are cached for reuse; long prompts
  run on machines that mix card sizes.
- **`neurobrix hub`** shows a model's distribution state in its STATUS column; Parakeet-TDT-1.1B, Open-Sora-v2,
  SANA-Video-2B-720p and Orpheus-3B are public with their validated containers; a serving engine releases GPU memory on unload.

## [0.5.2] - 2026-08-20

- **Encoded weights on the Triton engine.** Builds whose weights are stored packed (int4) run on dense language models and
  on mixture-of-experts models through fused dequantize kernels: the 30B-A3B coding model runs on a single 32 GB V100
  with 17 GB of weights instead of 57 GB. A build whose encoding the engine does not support is refused by name before
  any weight is read.
- **Frozen-plan replay of autoregressive decode** (opt-in, `NBX_TRITON_REPLAY=1`): the per-token graph is recorded once
  and replayed as direct kernel launches, with a CUDA-graph launch option; warm serving keeps its KV-cache buffers.
- **Graph optimisation at load time**: operations no output reads and exact identities are removed; common-subexpression
  elimination, horizontal matmul fusion and constant folding are available as opt-in passes.
- **The serving daemon** streams tokens, takes an image input, answers vision-language and multimodal requests in text,
  and a daemon channel per instance (`NBX_SOCKET_PATH`) lets several daemons share a machine.
- **Teacher-forced scoring** with `--set global.score_mode=1`, turning a request into the score of a given continuation.

## [0.5.1] - 2026-08-10

- **Gated models install from scripts, CI and Docker.** `neurobrix import --accept-license` (or `NBX_ACCEPT_LICENSE=1`)
  accepts a model's license without a terminal; without one, the CLI states both options and the license URL, and relays
  the registry's message.

## [0.5.0] - 2026-08-03

- **Multimodal builds speak and draw, on all four execution modes.** `--mode audio` generates speech (WAV at the model's
  declared sample rate, with a default seed carried by the package), `--mode image` generates images through a
  flow-matching stack that lives inside the package, and a conditional-flow-matching voice synthesis chain
  (codec-token stage, flow encoder, flow-matching decoder, vocoder) runs from packaged components. Builds without the
  contract refuse by name.
- **Both engines share one seeded random stream**, so a pinned run is reproducible per engine, including the vocoder's noise
  on the compiled path; Ming-Lite-Omni decodes greedily and answers alike on text, image and audio inputs on every mode.
- **Warm serving keeps its weights loaded between requests**, large-vocabulary models serve on the Triton engine with a
  streaming softmax, and short-schedule video renders the same bytes run to run on V100.
- **Diagnostics for comparing engines**: `NBX_SPEECH_DUMP`, `NBX_DRAW_DIAG` and `NBX_DECODE_TOPK` log the speech legs,
  the sampling candidates and the top next-token logits step by step.

## [0.4.1] - 2026-07-27

Packaging release: shipped files name the build toolchain generically, and the optional per-component flag registry is
located through an environment variable or a pointer file for development setups.
A configured location that is missing is refused by name; deployed installs without a registry keep the documented
defaults. No execution path changes.

## [0.4.0] - 2026-07-27

Image, audio and video understanding for large multimodal models, a terminal coding agent, and image generation from
multimodal containers.

### Added

- **Multimodal understanding.** Qwen3-VL-MoE-class (30B-scale) vision-language models sharded across GPUs
  automatically; Qwen3-Omni-lineage models answer questions about an image (`--input-image`), an audio clip
  (`--input-audio`) and a video clip (`--input-video`, `--fps`) from one container, with text, image, audio and video
  understanding on a single build; MiniCPM-o-lineage models run text, image and audio understanding. GLM-4.1V-9B-Thinking
  is the first vision-language model: an image and a question (`--input-image` + `--prompt`) answer in text. The
  placeholder-mask convention is read from the container's graph.
- **Mixture-of-experts models.** Models with several routers per layer (Ming lineage, one gate per modality) run on the
  fused expert path on both engines; multi-head-latent attention (DeepSeek-V2 class) and sorting-based windowed vision
  transformers run on the Triton engine.
- **A terminal coding agent.** `neurobrix chat --agent` (interactive) and `neurobrix run --mode agent --prompt "<task>"`
  (one-shot) let a tool-capable chat model read and write files and run shell commands inside a sandboxed working
  directory. Tool-call syntax is read from the model's chat template (XML and JSON dialects). Works cold and warm, on both
  engines, with a markdown and JSONL transcript per session. Flags: `--workdir`, `--max-turns`, `--bash-timeout`,
  `--allow-network`, `--approve-all`, `--yolo`, `--transcript-dir`.
- **Image generation from multimodal containers on all four execution modes**, first Janus-Pro-7B, with `--cfg 1.0`
  guidance-free requests and `--seed` governing Triton-engine sampling.

### Changed

- **Mixture-of-experts models run faster** in the sequential mode and across several GPUs on the Triton engine, and
  under any family's container; large models keep a GPU plan when one exists.
- **Serving.** Repeated requests on the serving daemon are faster, and the Triton engine serves pre-tokenized requests
  and repeated requests of a session; generation stops at the template's stop tokens in every mode; warm agent runs
  name a mismatched `--model`.
- **Memory.** The Triton engine reclaims finished intermediates sooner near a card's limit; Sana 4K image generation
  completes on the Triton engine; batched matrix products issue one launch per batch.
- **Explicit errors.** A tiled merge that cannot complete, a non-finite diffusion state, a malformed flag registry,
  missing text-to-speech diffusion parameters and unresolved guidance inputs are each named and stop the run.

## [0.3.0] - 2026-07-07

The video release: the video generation family runs end-to-end, text-to-video and image-to-video, in all four execution
modes.

### Added

- **Ten video models**: Wan2.1-T2V-1.3B, CogVideoX-2b, SANA-Video-2B-720p, CogVideoX-5b-I2V, Mochi-1-preview,
  Wan-I2V-14B, Wan2.2-I2V-A14B (dual denoiser, 28B), Wan2.1-VACE-1.3B, Open-Sora-v2 and Allegro, plus Allegro-TI2V at
  720×1280, 88 frames, with classifier-free guidance where the reference pipeline uses it.
- **Conditioning, read from each container's data**: image-to-video through `--input-image`, VACE control video,
  FLUX-style packed-latent video denoisers, and a boundary switch between two denoiser experts.
- **Video commands**: `--num-frames` and `--fps`.
- **5D spatio-temporal tiling** of the video VAE decode and encode over (T, H, W), on both engines.
- **Triton kernels**: `max_pool3d_with_indices`, conv3d temporal chunk-streaming under memory pressure,
  `replication_pad{1,2,3}d`, complex dtypes for rotary embeddings, and flash attention with a broadcast key-padding bias.
- **Multi-GPU**: component placement of 14B+ video models across GPUs, pipeline placement that balances layers and
  reserves headroom, and placement strategies without torch on the Triton engine.
- **Schedulers**: Euler-family float-timestep parity with the reference pipelines, a runtime-pure
  `UniPCMultistepScheduler`, the CogVideoX DDIM variant and flow-matching `invert_sigmas` with a video timestep scale.
- **Seeded runs** thread one run-scoped generator through the initial noise and every scheduler draw, on both engines.

## [0.2.1] - 2026-06-09

The first publicly usable release of the universal runtime: the four-mode execution matrix, PyTorch (`--sequential`,
`--compiled`) and Triton (`--triton-sequential`, `--triton`), runs orpheus-snac, hat-l-x4, Flex.1-alpha and
deepseek-moe-16b, each validated in a clean environment without the vendor libraries and consistent across modes. The
audio, LLM, image and upscaler families run end-to-end. Qwen3-30B-A3B is served in compiled mode.

### Added

- **Audio.** VibeVoice-1.5B text-to-speech through a next-token-diffusion flow; Kokoro-82M speaks prompts of any length;
  OpenAudio (Fish-Speech dual-AR) speech; Chatterbox conditioning runs the model's own speaker and Perceiver encoder;
  the Orpheus SNAC vocoder and Kokoro's pronunciation lexicon are built into the `.nbx`, and the speech-to-text and
  text-to-speech tokenizers, mel front-ends and RNNT flow run without vendor libraries; audio-conditioned language
  models (Voxtral) run on the Triton engine.
- **Image and upscaling.** PixArt and Sana run in `--triton`; SwinIR classical super-resolution (x2, x4) runs through
  `neurobrix upscale` on all four modes.
- **Placement.** Hybrid CPU and GPU plans execute, components that overflow the GPU route to the CPU, and Prism places
  work on hosts without GPUs through a `cpu_execution` strategy.
- **Triton engine.** The Triton path is torch-free end to end, with its own scheduler subtree (Euler, Euler-Ancestral,
  DDIM, Flow-Euler), `NBXTensor` complex types, a non-power-of-two FFT, `stft`/`istft`, `im2col`, transposed
  convolutions, LSTM, dilated convolutions and a growing set of ATen ops.
- **Models.** Models with unmerged PEFT/LoRA adapters run with the runtime LoRA-agnostic.

## [0.2.0] - 2026-05-18

Sana 4K-pixel image generation, diffusion models on the Triton engine, and data-driven output families.

### Added

- **Sana 4K-pixel generation.** A 4096×4096 image is produced on a single 32 GB V100 in the sequential and Triton modes:
  band-streamed fused upsample and convolution, an in-place residual-add fusion in the op-level tiling engine, and a
  depthwise convolution specialization (453× faster on the Sana 4K-pixel VAE depthwise blocks).
- **Diffusion models on the Triton engine.** PixArt-Sigma-XL-2-1024-MS, PixArt-XL-2-1024-MS and Sana 1024 generate
  1024×1024 images; Janus-Pro-7B runs on the Triton engine.
- **Output families.** The runtime ships family configurations for the nine-family taxonomy (vlm, multimodal, tts, stt,
  audio_llm, upscaler, plus llm, image and video) and dispatches each model's output by its family, so a model produces
  the modality it was built for.
- **Memory.** The Triton engine bounds its peak memory over long op sequences and honours per-op interceptors as the
  compiled mode does; the per-component registry flags take effect at run time.

## [0.1.6] - 2026-04-20

The zero3 release: models larger than the card run with block-wise pipelining on both engines.

### Added

- **Zero3 block-wise pipelining** on the PyTorch and Triton engines: a sliding window of two blocks' weights on the
  card, so prefill and decode keep a bounded memory regardless of model size. Qwen3-30B-A3B-Thinking-2507 runs end-to-end
  with `--triton --hardware v100-16g`.
- **Prism.** An environment variable selects a placement strategy by name; two-V100-16GB hardware profiles, including
  reduced-memory profiles, serve matrix testing; a hardware-gated fp16 overflow protection reads the profile's native
  bf16 support.
- **NBXTensor.** A CPU form with pinned host memory, device memory accounting, asynchronous streams and events,
  `to_cuda_async`, and CPU promotion of expert weights in the fused mixture-of-experts path.
- **Kernels.** Matrix products promote fp16 weights to fp32 inside the kernel on pre-Ampere hardware, and every
  multi-GPU Triton launch selects its device.
- **Regression harness.** Automated flow-aware CLI dispatch for speech-to-text and text-to-speech models.

## [0.1.5] - 2026-04-15

Faster Triton decoding and a regression harness.

### Added

- **Regression harness** (`tests/regression/`): an automated model-by-mode matrix with golden output comparison, run with
  `--runslow` for the heavy models.
- **Three graph-level fusions for Triton decode**: causal-mask elimination (the kernel carries the causal rule),
  SwiGLU fusion into one kernel, and RoPE fusion into one kernel. TinyLlama on a V100 in fp16 steps in 94 ms instead of
  460 ms (4.9× faster), with 376 element-wise ops instead of 684.
- **Weight transpose elimination** on the Triton engine, with the structure of the compiled mode.

## [0.1.4] - 2026-04-14

Release hygiene: the benchmark script reads its Hugging Face token from the shell environment or a gitignored `.env`
file, and publishing to PyPI is a manual job that an operator starts after reviewing the build artefacts. The 0.1.3
source archive was withdrawn from PyPI and its credential rotated; `pip install neurobrix` resolves to 0.1.4.

## [0.1.3] - 2026-04-14

The Triton engine serves language models faster.

### Added

- **DeepSeek-MoE-16B on `--triton`**, joining TinyLlama-1.1B and Qwen3-30B-A3B as language models of the Triton engine.
- **Decode speedups on a V100 in fp16**: TinyLlama-1.1B steps in 160 ms instead of 443 ms (2.8× faster); a decode matrix
  product takes 0.20 ms instead of 2.02 ms; a grouped-query attention call takes 0.94 ms instead of 1.58 ms.
- **Decode-aware kernels**: matrix products accumulate in fp32 while one token is decoded, short-query attention uses a
  compact block size, and grouped-query attention reads the keys and values in place.
- **A profiling harness**, opt-in and off by default, and a reference benchmark against the Hugging Face and Accelerate
  `device_map=auto` baseline.
- **Timestep embeddings of any scalar index dtype** on the Triton engine, for PixArt-Sigma-class diffusion transformers.

## [0.1.2] - 2026-04-03

The Triton engine and Apple Silicon.

### Added

- **`--triton` and `--triton-sequential` modes**: autoregressive language-model generation on 136 pure `@triton.jit`
  kernel files from FlagGems, attorch, Liger-Kernel and Flash-Attention, with a KV cache for grouped-query attention,
  `NBXTensor` as the zero-PyTorch tensor descriptor, a CPU Triton backend on hosts without a GPU, and a cold start of
  about 5 seconds instead of 8+ minutes.
- **Apple Silicon**: M1 through M5 Ultra auto-detection with unified memory, a 20-variant chip database, bf16 on M2 and
  later with macOS 14 or later, and device-agnostic helpers for sync, cache, seed, memory and sampling.
- **`neurobrix doctor`** with OS-specific PATH instructions, and a GitLab CI/CD pipeline publishing to PyPI.
- **Licenses**: the hub is the single source of truth, and the CLI reads `gated`, `licenseName` and `licenseUrl` from
  the hub API.
- **Dependencies**: `pydantic`, `packaging`, `torchaudio`, `snac`, `phonemizer`, `imageio-ffmpeg`, `transformers`,
  `mistral-common` and `tiktoken`, so every family works out of the box.

## [0.1.0] - 2026-03-26

The first stable release: a universal deep learning inference engine.

### Added

- **The `.nbx` container format** (TensorDAG, topology, manifest) and 34 models across four families (LLM, image,
  audio, video), including all 11 audio models: Whisper, Whisper V3 Turbo, Parakeet, Orpheus, Canary-Qwen, Kokoro-82M,
  VibeVoice-1.5B, Voxtral, OpenAudio-S1, Granite Speech and Chatterbox, and SANA-Video 720p.
- **Prism** hardware solver with multi-GPU allocation across 11 strategies (single_gpu through zero3), a hot/cold
  budget split for serve and run, and 12+ GPU hardware profiles (RTX 20/30/40 series, A10, A100, H100, L40S, T4, V100).
- **Execution**: `CompiledSequence`, `DtypeEngine` with automatic mixed precision, mixture-of-experts fused dispatch, a
  data-driven KV cache, symbolic spatial dimensions, a universal `TilingEngine`, and zero3 layer-wise pipelining with
  dual CUDA streams.
- **Commands**: `run`, `serve`, `chat`, `stop`, `hub`, `import`, `list`, `remove`, `clean`, `inspect`, `validate`,
  `info` and `doctor`; multi-turn conversation with automatic summarization; license acceptance in `neurobrix import`;
  hardware auto-detection with `--hardware` optional.
- **Platforms**: Windows, macOS and Linux, with AF_UNIX or localhost TCP between client and daemon.
- **Security**: `weights_only=True` for `torch.load`, path validation on registry import, and a safe arithmetic parser
  for shapes.

[Unreleased]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.11...main
[0.6.11]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.10...v0.6.11
[0.6.10]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.9...v0.6.10
[0.6.9]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.8...v0.6.9
[0.6.8]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.7...v0.6.8
[0.6.7]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/NeuroBrix/neurobrix/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/NeuroBrix/neurobrix/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/NeuroBrix/neurobrix/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/NeuroBrix/neurobrix/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/NeuroBrix/neurobrix/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/NeuroBrix/neurobrix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/NeuroBrix/neurobrix/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/NeuroBrix/neurobrix/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/NeuroBrix/neurobrix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NeuroBrix/neurobrix/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NeuroBrix/neurobrix/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NeuroBrix/neurobrix/compare/v0.1.6...v0.2.0
[0.1.6]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.5...v0.1.6
[0.1.5]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.4...v0.1.5
[0.1.4]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.3...v0.1.4
[0.1.3]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.2...v0.1.3
[0.1.2]: https://gitlab.com/neurobrix/Neurobrix/-/compare/v0.1.0...v0.1.2
[0.1.0]: https://gitlab.com/neurobrix/Neurobrix/-/releases/v0.1.0
