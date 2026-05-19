# P-SYMBOLIC-DIMS-FAMILY-AWARE — verdict (2026-05-19)

Branch `p-symbolic-dims-family-aware` from `ef88513` (Ch5 HEAD).

## Section 1 — Goal & debt status

The symbolic dimension-naming convention that is serialized verbatim
into the public NBX container (each dimension name becomes a symbol
name in `graph.json`, and also gates which dimensions are emitted as
dynamic symbolic ints, propagating into `topology.json`) lived as a
single 95-line hardcoded heuristic with no externalized, auditable,
family-organized form. Debt status confirmed **LIVE** by code reading:
the convention was implicit, name-substring + ndim based, with no
per-family paradigm or axis-role documentation, and no test pinning
its output — a latent reproducibility/contract risk and a blocker for
disciplined coverage of the families whose dimensions currently fall
through to generic `dim_N` names.

Ch6 makes the convention **data-driven and family-canonical** while
guaranteeing the emitted contract is **byte-identical** for every
already-packaged model (R18 immutable container + R28 bit-identical
reproducibility preserved by construction). It is a formalization,
not a behavior change.

## Section 2 — Existing convention audit (statu quo)

`dim 0 → "batch"` always (rank ≥ 1; rank 0 → empty). Then:

| rank | sequence input | vision/latent input | attention 4D | generic |
|---|---|---|---|---|
| 1 | — | — | — | (batch) |
| 2 | (batch, seq_len) | — | — | (batch, features) |
| 3 | (batch, seq_len, hidden_dim) | — | — | (batch, dim_1, dim_2) |
| 4 | — | (batch, channels, height, width) | (batch, heads, seq_len, head_dim) | (batch, dim_1, dim_2, dim_3) |
| 5 | — | (batch, channels, time, height, width) | — | (batch, dim_1, dim_2, dim_3, dim_4) |

Resolution priority (rank 4): vision-pattern → else attention(seq) →
else generic. Rank 2/3: seq → else generic. Rank 5: vision → else
generic. "Sequence" = any of a 15-substring set in the (lowercased)
input name, AND the name is not an exact-match member of the 3-name
fixed-channels exclude set (mel/raw-waveform/pixel inputs whose
dim 1 is a fixed channel count, not a sequence). "Vision/latent" =
any of a 6-substring set. The overlap token (the common latent name,
in both the sequence and vision sets) is disambiguated purely by
rank — 3D ⇒ sequence, 4D/5D ⇒ spatial — which the externalized form
preserves exactly.

Family mapping (the R26 9 families → paradigm): sequence paradigm =
`llm`, `audio_llm`, `stt`, `tts`, `multimodal`, `vlm`; spatial
paradigm = `image`, `upscaler`, `video`. This mapping was previously
undocumented and unenforced.

## Section 3 — State of the art consulted (R16)

R16 mandates external research as the first reflex before *inventing*
an algorithm. Ch6 deliberately invents nothing: its core design
constraint is byte-identical reproduction of a pre-existing in-repo
heuristic, so the authoritative source is that heuristic itself, read
in full before any edit (R8). No external algorithm was adopted or
needed — adopting one would by definition break the byte-identity
guarantee. The dimension taxonomy used (batch / seq_len / hidden_dim
/ channels / height / width / time / heads / head_dim) is the
universal de-facto tensor-axis convention already encoded in-repo and
matches the standard transformer/vision/video axis ordering; nothing
in the ecosystem supersedes a faithful transcription here. R16 was
therefore considered and correctly results in zero external
dependency for this chantier.

## Section 4 — Implementation

Two new data files + one delegation, all in the offline build subtree
(its own private repository; the public runtime repo is unchanged —
see Section 5 scope):

- `conventions.yml` — the convention as data: dispatch literals
  (the batch axis role; the sequence-name substring set; the
  exact-match fixed-channels exclude set; the vision/latent substring
  set; per-rank axis-role vectors keyed by resolved paradigm) PLUS an
  informational R26 9-family → {paradigm, axis_roles} map. The
  `families` block is informational only — it documents and is
  schema-tested, but is **not** consulted by dispatch (family is not
  threaded into the symbolic layer; dispatch stays on the historical
  `(input_name, rank)` signature, which is what guarantees
  byte-identity). Per-family enrichment of currently-generic
  dimensions is explicitly future capability, **not** applied
  retroactively (that would change the container for the protected
  matrix).
- `conventions.py` — a process-cached loader that reproduces the
  historical control flow exactly; only the literals/vectors are
  data. `families()` exposes the informational map for the schema
  test.
- The dimension-name helper is now an 18-line delegate (was 90 lines
  of hardcoded branches) calling the loader. Signature and the
  consuming dynamic-dimension gate are unchanged.

Design choice: faithful-transcript over "cleaner reorganization".
Because each dimension name is serialized verbatim as a symbol name
(no normalization downstream — confirmed by reading the symbol-table
creation and JSON serialization) and also gates the dynamic-dimension
set, any reordering of pattern precedence or substring normalization
would be an R18/R28 regression. The control flow is therefore copied
branch-for-branch; only the data is externalized.

## Section 5 — Validation

1. **Byte-identity, exhaustive (by construction + test).** A unit
   suite embeds the pre-Ch6 helper verbatim as a reference oracle and
   asserts the data-driven loader returns an identical dimension-name
   mapping over the full cartesian product of an exhaustive input-name
   set (every sequence substring, every exact-exclude and non-exact
   superstrings of each, every vision substring, the overlap token,
   case variants, real model input names, no-match) × ranks 0–7.
   Plus targeted sentinels for the overlap token and the
   exact-vs-substring exclude distinction. **5/5 PASS.**
2. **Bit-identical container re-generation (integration).** The
   reference model `TinyLlama-1.1B-Chat-v1.0` had its container
   artefacts (`graph.json` ×2 components + `topology.json`)
   regenerated with the data-driven code and compared against the
   pre-Ch6 baseline. The symbol-name sets are empirically equal
   (`lm_head`: [batch, seq_len]; `model`:
   [batch, batch, seq_len, seq_len]). With the two Ch6-orthogonal
   environment fields normalized (the per-tensor device-placement
   string, and a build-provenance path line that differs only because
   the cached baseline predates an unrelated provenance change in the
   build subtree), the regenerated artefacts are **strictly
   bit-identical** to the baseline. R18/R28 preserved, proven
   empirically as well as by construction.
3. **Conventions schema coverage (Étape 5).** Test asserts all 9 R26
   families present, each with a known paradigm and well-formed
   axis-role vectors from a fixed role vocabulary (no typos / missing
   fields); and that every dispatch axis-role vector has length =
   rank with index 0 = batch. **PASS.**
4. **BL-1 upscaler arbitrary-size (Étape 4).** 1234×789 (non-pow2,
   not a multiple of window_size 8) into `swin2SR-classical-sr-x2-64`.
   The arbitrary size was **accepted and correctly handled at the
   symbolic-naming + preprocessing layer** (padded to 792×1240,
   single-GPU placement resolved). Execution then failed downstream
   at a window-partition `aten::view` whose shape argument is frozen
   to trace-time integers `[1, 8, 8, 8, 8, 180]` (the model was built
   at 64px → 64/8 = 8). This is a **pre-existing** symbolic-shape
   propagation gap on the window-partition reshape, orthogonal to
   Ch6 (the spatial dimensions *are* named/symbolic correctly — proven
   byte-identical; the freeze is on a derived reshape shape-arg in a
   different subsystem). Per the mandate this is exactly the
   "blocker beyond symbolic naming" case → **non-diagnostic for Ch6,
   latent observation D10** (Section 6). R29 artefacts under
   `validation_outputs/p_symbolic_dims_family_aware/upscaler_arbitrary_size/`.
5. **Anti-regression.** Ch6 modifies **zero** public runtime code
   and rebuilds **zero** packaged model (git-verified: the only
   public-repo working-tree entries are the gitignored validation
   artefacts and a pre-existing untracked venv dir; all four changed
   code files are in the private offline-build subtree). The
   already-packaged model matrix is therefore unchanged by
   construction; the bit-identical re-generation (point 2) is a
   stronger guarantee than a harness pass (it proves the artefact
   unchanged, not merely the behavior). A bounded runtime smoke
   (`TinyLlama` end-to-end, compiled) returns coherent output,
   EXIT 0. The full slow harness (`--runslow`) is deliberately
   **not** run: it would exercise runtime code Ch6 provably does not
   touch (non-diagnostic) and pull in the standing-INDETERMINATE
   4Kpx image model (which standing project memory forbids relaunching
   under longer budgets) plus multi-hour video — budget burn with
   zero Ch6 signal. No green→red is structurally impossible for this
   change.

## Section 6 — Latent observations (D10, NOT fixed — strict scope)

- **P-SWIN-WINDOW-VIEW-SYMBOLIC-DIMS** — the Swin window-partition
  `aten::view` shape argument is frozen to build-time integers
  instead of propagating as symbolic `floor(height/ws),
  floor(width/ws)`. Site: the view/reshape symbolic rule
  (`…/symbolic/rules.py:356 _view_reshape`, factor-partition
  heuristic). This is the BL-1 arbitrary-resolution blocker for
  window-attention SR models; a dedicated symbolic-rules chantier,
  not Ch6.
- Pre-existing unused-import / unused-variable warnings in the
  modified module (`Set`, `Tuple`, `field`, `make_symbolic_shape`,
  and several locals) were present before Ch6 and are left untouched
  (strict scope; note for a future hygiene pass).
- Family is informational in the convention because it is not
  threaded into the symbolic layer. Threading family/flow-type down
  to enable per-family emitted-name enrichment is a deliberate
  future chantier (it would change the container for currently-
  generic dimensions and must be gated against the protected
  matrix).

## Section 7

Commits: see closure report. Hocine validation: TODO (byte-identity
test + bit-identical re-generation are the sign-off evidence;
BL-1 artefacts present).
