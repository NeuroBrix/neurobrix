# The dtype per component

Rule of the owner, 2026-10-13. This page is the engine's side: what a container declares about dtype,
what the engine reads, and the one exception the engine owns.

## The rule

1. **Every component runs in the dtype its weights store.** A container declares that dtype, as it
   is, per component: `components/<c>/profile.json` `dtype`, `components/<c>/weights_index.json`
   `dtype` and `topology.json` `extracted_values.<c>.dtype` are one value, the dtype of the
   component's stored weights (for a component whose weights mix dtypes — norms in float32 beside
   bfloat16 matrices — the dtype holding the most bytes). There is no "dominant dtype": a vote over
   components, a tie, a traversal order decide nothing, and no such global value is written.
2. **The flow state's dtype is derived per component.** `runtime/defaults.json` `dtype` (the dtype the
   engine allocates the denoising latents or the decode state in) and `manifest.json` `dtype` are the
   stored dtype of the component the state passes through — the flow's loop components, which must
   agree — and, for a flow without a loop, the component holding the most weight bytes.
3. **The only exception is the engine's, at execution, and is never written.** A value beyond the
   hardware precision's margin (fp16 scale overflow on a V100) is computed in fp32; the DtypeEngine
   decides it from the hardware profile and the tensor itself, data-driven, never by model name. A
   bfloat16 component on hardware without bfloat16 is the same decision: the hardware policy, read
   from the profile, maps it at run time while the container still says bfloat16. The container stays
   portable: Apple and Hopper do not share a margin.
4. **No GPU on a known bug.** Before any new compute, the containers whose declared or state dtype
   differs from their stored dtypes are listed by reading their files (`runtime/defaults.json`,
   `topology.json`, the safetensors headers — zero execution); every one of them, validated or not, is
   corrected in its metadata (key/value, no retrace when the graph is right), its dated proof replayed
   at the CLI, and the hub receives the corrected container.

## What the engine reads

| file | field | reader |
|---|---|---|
| `runtime/defaults.json` | `dtype` | the state variables' allocation (`core/runtime/resolution/variable_resolver.py`) |
| `components/<c>/profile.json` | `dtype` | the component's compute dtype before the hardware policy (`nbx/container.py`, Prism) |
| `components/<c>/weights_index.json` | `dtype` | the weight loaders of both engines |

The three agree by construction under the rule. A container where they do not has not been
corrected yet; the audit of 2026-10-13 over the shared cache (49 containers, files only) found 23
such containers, among them Wan2.1-I2V-14B-480P, whose state dtype had moved float16 → float32 with
its retrace of 2026-10-12 and whose native video moved with it.
