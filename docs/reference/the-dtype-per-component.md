# The dtype per component

What a container declares about dtype, what the engine reads, and the one decision the engine makes itself.

## The rule

1. **Every component runs in the dtype its weights store.** A container declares that dtype per component:
   `components/<c>/profile.json` `dtype`, `components/<c>/weights_index.json` `dtype` and `topology.json`
   `extracted_values.<c>.dtype` are one value, the dtype of the component's stored weights (for a component
   whose weights mix dtypes, such as float32 norms beside bfloat16 matrices, the dtype holding the most
   bytes). There is no global "dominant dtype": no vote over components decides anything.
2. **The flow state's dtype is derived per component.** `runtime/defaults.json` `dtype` (the dtype the
   engine allocates the denoising latents or the decode state in) and `manifest.json` `dtype` are the stored
   dtype of the component the state passes through: the flow's loop components, which must agree, and, for
   a flow without a loop, the component holding the most weight bytes.
3. **The only exception is the engine's, at execution, and it is never written back.** A value beyond the
   hardware precision's range (float16 overflow on a V100) is computed in float32. The DtypeEngine decides
   it from the hardware profile and the operation, never by model name. A bfloat16 component on hardware
   without bfloat16 is the same decision: the hardware profile maps it at run time while the container
   still says bfloat16. The container stays portable: Apple and Hopper do not share a range.

## What the engine reads

| file | field | reader |
|---|---|---|
| `runtime/defaults.json` | `dtype` | the state variables' allocation (`core/runtime/resolution/variable_resolver.py`) |
| `components/<c>/profile.json` | `dtype` | the component's compute dtype before the hardware policy (`nbx/container.py`, Prism) |
| `components/<c>/weights_index.json` | `dtype` | the weight loaders of both engines |

The three agree by construction under the rule.
