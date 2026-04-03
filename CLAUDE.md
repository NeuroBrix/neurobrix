# NeuroBrix — Project Guidelines

## Core Philosophy
- **ZERO HARDCODE:** All values derived from the NBX container. Nothing hardcoded.
- **ZERO FALLBACK:** System crashes explicitly if data is missing. No silent defaults.
- **UNIVERSAL:** One engine handles any model. The engine KNOWS architecture (via NBX metadata) for optimization.
- **DATA-DRIVEN FLOWS:** Flow handlers orchestrate WHEN to run components. The graph handles HOW. No `executor._weights` access in flow handlers.

## Data-Driven Flow Rule (MANDATORY)
- **Tracer traces the FULL forward pass** per component: input → embedding → backbone → output (e.g. logits). Symbolic shapes handle dynamic dimensions (seq_len=23 at trace, any value at runtime).
- **Flow handlers NEVER access `executor._weights`** — all inference happens inside the graph via `executor.run()`.
- **variable_resolver + topology.json** manage all data flow between components — same pattern for ALL families (image, LLM, audio, video).
- **The diffusion flow (iterative_process.py) is the reference** — it never touches weights, only orchestrates scheduler steps + graph execution.
- **WRONG**: `embed = executor._weights["token_embed.weight"]; logits = hidden @ embed.T` (shortcut)
- **RIGHT**: `outputs = executor.run(inputs); logits = outputs["logits"]` (graph does everything)
- **Exception**: Input preprocessing (mel spectrogram, tokenization) happens before the graph — that's OK.

## NeuroTax Rule — RUNTIME MUST USE STANDARD NAMES (MANDATORY)
- **All weight keys in NBX safetensors are NeuroTax-normalized** — vendor names are gone after forge build.
- **Runtime code MUST match on NeuroTax standard names**, never on vendor names.
- **Standard names**: `block`, `attn`, `ffn`, `gate`, `up`, `down`, `q`, `k`, `v`, `out`, `norm`, `embed`, `head`, `enc`, `pred`, `joint`
- **WRONG**: `if "joint_net" in key` (vendor name), `if "q_proj" in key` (vendor name)
- **RIGHT**: `if key.endswith("q.weight")` (NeuroTax standard), `if "joint." in key` (NeuroTax standard)
- **Flow handlers** (`core/flow/*.py`) that iterate `executor._weights` MUST use NeuroTax patterns.
- **Why this exists**: Forge NeuroTax renames vendor keys → NBX stores standardized keys → runtime reads standardized keys. If runtime matches on vendor names, it breaks after rebuild.

## NBX Container Rule (MANDATORY)
- **NBX is a faithful transport/storage format** — reproduces vendor weights identically.
- **NEVER convert dtype in the builder** — vendor ships fp32 → NBX stores fp32. bf16 → bf16.
- **Runtime adapts** — DtypeEngine + Prism handle dtype conversion at execution time based on hardware.
- **Same for forge tracing** — trace in vendor's original dtype, never convert.
- The ONLY exception: bf16 on fp16-only hardware — if values exceed fp16 range, abandon for that hardware.

## Architecture

```
neurobrix/
  cli/           Command-line interface (run, serve, chat, hub, import, ...)
  core/          Runtime engine (executor, prism solver, dtype, flows)
  kernels/       Triton GPU kernels
  nbx/           .nbx container format
  serving/       Persistent model serving daemon
  config/        Hardware profiles, family configs
```

## CLI Commands

```bash
# Serving (recommended) — hardware auto-detected, --hardware optional
neurobrix serve --model <name>
neurobrix chat [--temperature T]
neurobrix stop

# Single-shot — hardware auto-detected, --hardware optional
neurobrix run --model <name> --prompt <text>

# Model management
neurobrix hub / import / list / remove / clean

# Inspection
neurobrix info / inspect / validate
```

## Families
- `image` — Diffusion and VQ image generation
- `llm` — Autoregressive language models
- `audio` — Speech-to-text
- `video` — Video generation

## Code Hygiene
- Delete unused code immediately (functions, imports, branches, files)
- No commented-out code (use git history)
- No "TODO: Remove" comments — remove it now
- Fix problems when detected, don't defer

## Triton Mode — Zero PyTorch Inference (MANDATORY)

### Architecture
- **Package**: `src/neurobrix/triton/` — 14 files, fully independent from PyTorch
- **Tensor type**: `NBXTensor` (in `kernels/nbx_tensor.py`) replaces `torch.Tensor` everywhere
- **Dispatch**: `kernels/dispatch.py` — ONE flat table (238 ops). Missing op = crash, no fallback.
- **Dtype**: `triton/dtype.py` — `TritonDtypeEngine` with same AMP rules as native mode, using `NBXDtype`
- **Weight loading**: `triton/weight_loader.py` — safetensors numpy API + `cudaMemcpy`, zero torch

| File | Purpose |
|------|---------|
| `executor.py` | Full pipeline: load graph → compile → bind weights → run |
| `sequence.py` | Compiled hot loop (ported from `compiled_sequence.py`) |
| `sequential.py` | Op-by-op debug dispatcher (`--triton-sequential`) |
| `arena.py` | `__slots__`-based O(1) tensor storage |
| `symbols.py` | Symbolic shape resolution (`input::id::dim_N` format) |
| `kv_cache.py` | Pre-allocated NBXTensor KV buffers with GQA support |
| `autoregressive.py` | Zero-torch LLM flow handler |
| `samplers.py` / `generator.py` / `session.py` | Sampling, decode loop, chat session |

### CLI Flags
```bash
neurobrix run --model <name> --triton --prompt <text>       # compiled (production)
neurobrix run --model <name> --triton-sequential --prompt <text>  # op-by-op (debug)
```

### Supported Flows
| Family | Flow | Status |
|--------|------|--------|
| LLM | Autoregressive | Supported |
| Image | Diffusion | Not supported |
| Audio | Encoder-decoder / RNNT / TTS | Not supported |
| Video | Generation | Not supported |

### KV Cache (`triton/kv_cache.py`)
- **Lazy allocation**: buffers created at first prefill, sized from `defaults.json`
- **GQA**: `num_kv_heads` independent of `num_heads` — repeat-interleave at SDPA time
- **Interceptor pattern**: `autoregressive.py` creates `TritonAttentionInterceptor`, registers it on the executor via `register_triton_interceptors()`. The compiled sequence replaces SDPA ops with the interceptor's `intercept` callable, which manages cache update + attention in one step.
- **Write**: `__setitem__` with slice indexing (cudaMemcpy under the hood)

### Key Bugs Fixed (Reference)
1. **`cat` + `is_contiguous()`**: NBXTensor cat output was non-contiguous after concat along non-last dim. Triton kernels require contiguous input — added `.contiguous()` call in dtype engine cast path.
2. **Symbolic promotion ambiguity**: Two symbols bound to same dim could conflict during `bind_from_inputs()`. Fixed by promoting the more-constrained symbol and dropping the ambiguous one.
3. **SDPA double-masking**: Causal mask applied twice (once explicit, once via `is_causal=True`) producing garbage attention. Fix: detect pre-applied mask and set `is_causal=False`.
4. **`__setitem__` strided scatter**: KV cache slice write on non-contiguous view silently corrupted memory. Fix: ensure target view is contiguous before cudaMemcpy, or use explicit scatter.

## Git Workflow (MANDATORY)

### Branch Strategy
- **main** = public branch (GitLab, PyPI). Clean, squashed commits only.
- Work directly on `main` locally. Commit frequently with clear messages.
- For large features: create a feature branch → PR → squash merge.

### Commit Rules
1. **Group related changes** into a single commit (don't commit every file separately)
2. **Commit message format**: `type: description` where type = fix, feat, security, docs, refactor, perf, build
3. **Push to origin/main** after each logical group of commits (security fixes, feature, docs update)
4. Never push broken code. Run basic validation before pushing.

### Release Workflow (Manual Grouped Releases)
Releases are grouped manually — NO automatic version bumping per commit.
1. Accumulate changes: add entries under `## [Unreleased]` in CHANGELOG.md with each commit
2. When ready to release (agreed between us):
   a. Bump version: `.claude/hooks/version-bump.sh [patch|minor|major]`
   b. Stamp CHANGELOG: rename `[Unreleased]` → `[X.Y.Z] - YYYY-MM-DD`, add fresh `[Unreleased]` above
   c. Commit: `git add pyproject.toml CHANGELOG.md && git commit -m "build: release vX.Y.Z"`
   d. Tag: `git tag vX.Y.Z`
   e. Push: `git push gitlab main --tags`
- **Release signal hook** (`.claude/hooks/neurobrix-release-signal.sh`) alerts after 10 source edits without a commit
- **Post-commit hook**: DISABLED (no more auto-bump)

### Version Bump Rules
- **patch** (`0.1.0` → `0.1.1`): Bug fixes
- **minor** (`0.1.0` → `0.2.0`): New features, new model families
- **major** (`0.1.0` → `1.0.0`): Breaking API changes

### Doc Sync
When modifying source code, check if corresponding docs need updating:
- `cli/*.py` → `docs/cli-reference.md`
- `core/runtime/*.py`, `core/flow/*.py` → `docs/architecture.md`
- `nbx/*.py` → `docs/nbx-format.md`

### Forge Git (SEPARATE REPO — MANDATORY)
- `forge/` has its OWN `.git` — completely separate from NeuroBrix
- Forge is gitignored in NeuroBrix's `.gitignore` (line 47: `forge/`)
- Remote: `https://gitlab.com/neurobrix/NeuroBrix_forge` (private)
- Auto-commit + auto-push hook runs on every forge file edit (`.claude/hooks/forge-git-autocommit.sh`)
- Doc-sync hook reminds to update forge docs (`.claude/hooks/forge-doc-sync.sh`)
- After large tasks: verify forge git is clean with `cd forge && git status`

## Output Convention
Always generate test images in project root (`/home/mlops/NeuroBrix_System/`), not `/tmp/`.

## Modification Tracking
All code modifications MUST be logged in `MODIFICATIONS.md` at the project root.

## CHANGELOG Maintenance (MANDATORY)

**Every commit that modifies source code MUST include a CHANGELOG.md update.** This is enforced by a pre-commit git hook.

### Rules
1. **When to update:** Every commit that touches `src/neurobrix/` or `forge/` source files
2. **Where to add:** Under `## [Unreleased]` section, in the appropriate category
3. **Categories:** `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
4. **Style:** Imperative mood, start with a verb ("Add X support", "Fix Y crash"), concise but informative
5. **Stage CHANGELOG.md** in the same commit as the code changes

### Format
```markdown
## [Unreleased]

### Added
- Universal hardware auto-detection for 10 GPU vendors

### Fixed
- RoPE overflow at 4K+ token sequences
```

### Release Workflow
When cutting a release (e.g., `0.2.0`):
1. Rename `[Unreleased]` to `[0.2.0] - YYYY-MM-DD`
2. Create new empty `[Unreleased]` section above it
3. Update comparison links at bottom of file
4. Commit: `docs: release changelog for v0.2.0`

### What NOT to log
- Internal refactors with zero user-visible impact
- Typo fixes in comments
- Test-only changes
- Changes to CLAUDE.md, MODIFICATIONS.md, or .gitignore
