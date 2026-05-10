# POINT 5 — fix halo bug `_tiled_conv2d_spatial_nbx` + `_fused_upsample_conv2d_nbx`
## P-SANA-4KPX-RUNTIME / phase 5 — partial progress

R29 inspectable artefact pour POINT 5 — progrès partiel non-final.

## ÉTAPE A — diagnostic op 635 conv::48

`/tmp/capture_conv48_both_modes.py`: capture inputs + outputs en seq/tri.

| | INPUT | OUTPUT |
|---|---|---|
| seq vs tri rel max_abs | 0.5% | 12.3% |
| seq vs tri max_d | 17.2 | 33.08 |
| frac elements diff > 1e-3 | **99.93%** | 99.95% |

conv::48 INNOCENT en kernel — son input arrive déjà divergent à 99.93%.
Walk-back vers upstream intercepted ops.

## ÉTAPE A walk-back — hypothèse TilingEngine kernel divergence

`/tmp/capture_conv36_runtime.py`: vérifier les premiers tiled convs.

**PRE-FIX** (mesures avant POINT 5):

| op | seq max | tri max | rel | max_d | frac>1e-3 |
|---|---|---|---|---|---|
| conv::36 | 129.2 | 129.2 | 0.0% | **52.78** | **98.72%** |
| conv::41 | 100.1 | 101.4 | 1.2% | 50.12 | 98.94% |
| conv::46 | 109.2 | 106.9 | 2.1% | 48.66 | 99.29% |
| conv::48 | 70.62 | 61.94 | 12.3% | 33.08 | 99.95% |

**ROOT CAUSE IDENTIFIÉ**: `_tiled_conv2d_spatial_nbx` et
`_fused_upsample_conv2d_nbx` appliquaient `padding=(pad_h, pad_w)` sur
TOUTES les bandes — y compris frontières internes — ce qui remplissait
de zéros là où le kernel devait lire les rangs réels de la bande
voisine (vrai halo). Le `_torch` backend utilisait l'algorithme halo
correct depuis toujours; le `_nbx` backend ne mirror pas.

## ÉTAPE B — confirmation cluster

Pattern observé sur conv::36, ::41, ::46 (tous tilés): même bug
zero-pad-au-frontier. conv::48 (non-tilé) hérite du résultat divergent
en cascade.

## ÉTAPE C — fix appliqué

| fichier | modification |
|---|---|
| `kernels/ops/fused_upsample_conv.py:_tiled_conv2d_spatial_nbx` | Réécrit avec halo-based algorithm matching `_tiled_conv2d_spatial_torch`. Use `constant_pad_nd_wrapper` pour pad asymétrique (top/bot indépendants). Conv runs avec `padding=(0, 0)` — halo + image-edge pad déjà appliqués. |
| `kernels/ops/fused_upsample_conv.py:_fused_upsample_conv2d_nbx` | Même fix. Le typo `band_pad_h = pad_h if (...) else pad_h` (les deux branches retournent pad_h) corrigé via halo + asymmetric pad. |

R33 zero-torch préservé (constant_pad_nd_wrapper est R33-pure).

**POST-FIX** (mesures):

| op | seq max | tri max | rel | max_d | frac>1e-3 |
|---|---|---|---|---|---|
| conv::36 | 129.2 | 129.2 | 0.0% | **0.7188** | 47.31% |
| conv::41 | 100.1 | 100.1 | 0.0% | **0.6875** | 52.61% |
| conv::46 | 109.2 | 109.3 | 0.06% | **1.062** | 58.45% |
| conv::48 | 70.62 | 70.62 | 0.0% | **1.385** | 86.88% |

**Réduction divergence ~50-75× sur max_d**. Les valeurs sont
maintenant dans le range fp16 ULP (max_d ~1 sur magnitudes 100+ =
~1% rel = mostly fp16 ULP).

## ÉTAPE D — anti-régression matrice

| modèle | mode | PNG | verdict |
|---|---|---|---|
| Sana 1024 | triton_sequential | sana1024_tri_seq_post_point5.png | **RED APPLE** ✓ |
| PixArt-XL | triton_sequential | pixart_xl_post_point5.png | **RED APPLE** ✓ |
| Sana 4Kpx | triton_sequential VAE-iso | sana4kpx_vae_iso_post_point5.png | **bandes RGB** (pattern persiste mais évolué visuellement, color reconstruction améliorée) |
| Sana 4Kpx | triton_sequential FULL pipeline | sana4kpx_full_post_point5.png | **bandes RGB** (similar to VAE-iso) |

## Verdict — progrès partiel, pas pomme rouge

Le halo bug FIX est un acquis structurel majeur:
- 50-75× réduction divergence conv tilés
- max_d post-fix = ~1 (fp16 ULP range, par construction
  équivalent au path non-tilé)
- Anti-régression Sana 1024 + PixArt-XL préservée

Mais **Sana 4Kpx ne produit pas pomme rouge cohérente**. Le pattern
de bandes RGB persiste. Possibilités résiduelles:

1. **D'autres tiled ops avec bug similaire**: tiled_rms_norm_spatial
   inspecté — pas de halo bug (rms_norm normalise par-row, pas de
   cross-band stat). Mais d'autres kernels (pixel_shuffle tilés,
   add_inplace, view interceptors) pourraient avoir des subtilités.

2. **L'effet cumulatif des ULP-level divergences résiduelles** sur
   les ~60 convs tilés produit un drift cumulé suffisant pour
   amplifier à travers le rms_norm/silu chain. fp16 ULP × 60 ops
   au shape 4Kpx = potentiellement plusieurs % rel cumulé.

3. **Bug subtil dans constant_pad_nd_wrapper** ou dans NBXTensor
   slicing (`input_tensor[:, :, ih_start:ih_end, :]`). Si le NBX
   slicing diffère de torch slicing dans certains cas, le band
   content diffère.

## Cumulative session statut

| | commit | verdict |
|---|---|---|
| POINT 1 | ea8e8e2 | input::z cast |
| POINT 2 | 331c611 | uniform AMP_FP32_OPS cast-back |
| POINT 2bis | 735e76e | registry plumbing |
| POINT 3 | 1810307 | 0 STRUCTURELLE_RACINE dtype |
| POINT 4 | 6224ac4 | relu::15 innocent |
| POINT 4-bis | b9d46ae | cross-variant Scénario 1 |
| **POINT 5** | this commit | **halo bug fix, 50× reduction, partial progress** |

## Awaiting Hocine arbitrage

Question pour POINT 6 / pivot:
- Investigate option 2 (cumulative ULP drift) avec measurement de
  drift cumulé après chaque tiled conv au 4Kpx?
- Investigate option 3 (NBX slicing/pad subtleties) avec microtest
  comparing tiled NBX path vs non-tiled path on same input?
- Pivot vers approche différente?

Le halo fix doit être committé indépendamment du reste — c'est un
bug fix valide même si pomme rouge pas atteinte.
</parameter>
