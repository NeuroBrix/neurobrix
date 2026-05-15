# P-NEUROBRIX-UPSCALERS — v1 backlog

Tracked issues that require model-packaging / trace-upstream
changes (outside the NeuroBrix runtime repo scope) or deferred
follow-up.

## BL-1 — Upscaler profile.json missing `upscale` + `window_size` config

**Severity**: blocks arbitrary-input-size upscaling (works only at
the exact trace size, e.g. 64×64 for the swin2sr containers).

**Factual evidence**:
- `core/module/tiling_engine.py:from_component_config()` line 166
  reads `scale_factor = config.get("upscale")` from the
  component's `profile.json`. Line 174-176: if `scale_factor is
  None`, the method returns `None` → no `TilingEngine` is
  instantiated for that component.
- The swin2sr containers' `components/swin2sr/profile.json`
  `config` section contains only
  `{"hidden_size": 180, "num_layers": [6,6,6,6,6,6]}`. There is
  **no `upscale` and no `window_size`** key.
- Consequence: `should_tile()` is never reachable; inputs whose
  H/W exceed the trace size hit a hardcoded view-shape mismatch:
  `aten.view::18: shape '[128, 128, -1]' is invalid for input of
  size 24576` when feeding a 128×128 image to a graph traced at
  64×64.

**Root location**: the container's `profile.json` `config` does
not carry the super-resolution scale factor (`upscale`, =2/4/8
depending on variant) nor the Swin window size (`window_size`,
=8 for swin2sr). These values exist in the upstream model
`config.json` but are not propagated into the packaged container
profile. This is a **model-packaging** concern, not a runtime
bug — the runtime tiling logic is correct and data-driven; it is
starved of the config it needs.

**Workaround in v1**: validate the CLI + 4-mode runtime at the
exact trace size (64×64), which exercises the full
load→preprocess→execute→save→4-mode-dispatch path and proves the
integration. Arbitrary-size upscaling is deferred until the
container profile carries `upscale` + `window_size`.

**Resolution owner**: model-packaging pipeline (add `upscale`
and `window_size` to the profile.json `config` whitelist for the
upscaler family, sourced from the upstream model config). When
that lands, `TilingEngine.from_component_config` will
auto-instantiate and arbitrary input sizes will tile through the
64×64 graph transparently — no runtime change needed.

## BL-2 — manifest.model_type is None for upscaler containers

**Severity**: cosmetic / low. `manifest.model_type` reads `None`
for the swin2sr containers; family-based routing (`family ==
"upscaler"`) is the discriminator and works correctly, so this
does not block anything. Noted for completeness — a populated
`model_type` would aid diagnostics.
