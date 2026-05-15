# P0 — Upscaler .nbx loader resolution audit (P-NEUROBRIX-UPSCALERS)

## P0.1 — Path resolution

`src/neurobrix/cli/utils.py:find_model()` resolves a `model_name`
to its container by searching the runtime store
`~/.neurobrix/cache/<model_name>/` (pathlib.Path, cross-OS,
post path-leak-fix). It returns either the extracted directory
or `model.nbx` inside it.

**Finding**: no code change needed. The `models/<family>/<model>/`
build-output tree is NOT a default search path (by design — it
is build territory, not runtime territory). The canonical flow
is: container → extracted into `~/.neurobrix/cache/`. For
registry models `nbx import` does this; for locally-produced
containers the equivalent is a one-time unzip into the cache:

    for m in <model>; do
      mkdir -p ~/.neurobrix/cache/$m
      (cd ~/.neurobrix/cache/$m && unzip -q <models_dir>/$m/model.nbx)
    done

After extraction `find_model(<model>)` resolves correctly.
Verified for the 3 swin2sr variants:

| model_name | resolved path | exists |
|---|---|---|
| swin2SR-classical-sr-x2-64 | ~/.neurobrix/cache/swin2SR-classical-sr-x2-64 | ✓ |
| swin2SR-classical-sr-x4-64 | ~/.neurobrix/cache/swin2SR-classical-sr-x4-64 | ✓ |
| swin2SR-realworld-sr-x4-64-bsrgan-psnr | ~/.neurobrix/cache/...-bsrgan-psnr | ✓ |

## P0.2 — Basic load (no forward)

`NBXRuntimeLoader().load()` on swin2SR-realworld-sr-x4-64-bsrgan-psnr:

| field | value |
|---|---|
| manifest.family | `upscaler` |
| manifest.model_type | `None` (family is the discriminator, R34) |
| topology.components | `['swin2sr', 'upsample']` |
| topology.flow.type | `forward_pass` |

LOAD OK — manifest read, topology read, components enumerated,
no exception. Container is structurally valid and clean (zero
absolute-path leak residue confirmed by the prior cleanup pass).

## Conclusion

The loader requires no extension for upscalers. The container is
consumed as an opaque asset via the standard runtime store. U3
proceeds with the `nbx upscale` subcommand.
