# Vendor hardware configurations (R10 data-driven)

These YAMLs are the **runtime** source of truth for all
hardware-dependent parameters (block sizes, SDPA thresholds, shared-
memory limits, precision flags, the SDPA deterministic-routing
budget, …). They are consumed by `core/config/loader.py`
`get_vendor_config(vendor, architecture)`, which is **ZERO-FALLBACK**:
a missing file raises `FileNotFoundError`. Nothing hardcodes these
values in Python (R10 / R23 / R24).

Layout: `vendors/<vendor>/<architecture>.yml`
(`nvidia/volta.yml`, `nvidia/ampere.yml`, `nvidia/hopper.yml`,
`amd/cdna.yml`).

## Why this directory is tracked

It is committed for the same reason `config/families/*.yml` are:
both are runtime data-driven config the loader hard-requires. Before
2026-05-19 an over-broad `vendors/` `.gitignore` pattern accidentally
excluded this directory, so a fresh clone crashed on every
`get_vendor_config` call (P-TRITON-MOE-DETERMINISM-RESIDUAL absorbed
the fix: a targeted `.gitignore` negation now tracks **only** this
runtime-config directory; any other `vendors/`-named tree — e.g.
per-model build vendoring under R25 — remains ignored via its own
parent rule).

The per-machine *detected hardware profile* is a different artefact,
generated at first run by hardware autodetection and NOT one of these
files.

## Validation status (anti-bluff convention)

Each file declares a `validation_status` in its top comment block:

| status | meaning |
|---|---|
| `validated_empirical` | Values exercised end-to-end on real hardware; the comment lists the validated models + the chantier/commit. |
| `hand_curated_from_docs` | No such hardware available; values derived from vendor documentation or conservatively copied. NOT empirically validated. Refine when hardware access is obtained. |

Only `nvidia/volta.yml` is `validated_empirical` (4× V100 is the only
hardware available). `ampere`, `hopper`, `cdna` are
`hand_curated_from_docs`.

## Per-key source annotation

Every non-obvious key carries an inline comment with its rationale
or source. Empirically-derived keys cite the chantier/commit;
doc-derived keys cite the documentation basis. Never present a
doc-derived or copied-conservative value as if it were validated.

## Schema parity (R10 strict)

All vendor files MUST carry the **same key schema**; only the values
differ per architecture. If you add a key to one file, add it to all
four (with an architecture-appropriate value + rationale). Example:
`memory.sdpa_math_max_scores_bytes` exists in all four — `volta`
holds the empirically-derived budget; the others hold `0` with an
inline explanation that the routing it gates is a Volta-specific
deterministic-attention fallback, intentionally disabled elsewhere
(the value, not the key's presence, is the gate).

## Adding a new vendor / architecture

1. Create `vendors/<vendor>/<architecture>.yml` with the full key
   schema (copy an existing file as the template).
2. Add the `validation_status` header honestly.
3. Fill values from hardware measurement (`validated_empirical`) or
   documentation (`hand_curated_from_docs`, cite sources inline).
4. Ensure hardware autodetection maps the detected
   compute-capability/arch string to this file's name.
