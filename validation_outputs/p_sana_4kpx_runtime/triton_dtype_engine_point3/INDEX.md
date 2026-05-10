# POINT 3 — Re-walk dtype_mirror + verdict classifié intermédiaire
## P-SANA-4KPX-RUNTIME / TritonDtypeEngine maturation phase 3/5

R29 inspectable artefact pour POINT 3 du plan TritonDtypeEngine.
Synthèse factuelle qui tranche entre divergences "structurelles vraies"
(à fixer en POINT 4-5) et divergences "cascade downstream" qui se
résoudront mécaniquement via leur ancêtre commun.

## Verdict synthétique — top-line

**0 STRUCTURELLE_RACINE / 0 CASCADE_DOWNSTREAM / 291 COMMUNE_1024 (100%)**

POINT 1 + 2 + 2bis ferment l'intégralité des divergences dtype
spécifiques à 4Kpx. **Toutes les 291 divergences dtype restantes sont
COMMUNE 1024** — présentes aussi sur Sana 1024 qui produit pomme rouge
cohérente. Donc:

1. **Aucune divergence dtype-RACINE n'est à fixer dans POINT 4-5.**
2. La grille de lignes fines entrelacées + bandes RGB observée sur
   Sana 4Kpx VAE-iso post-2bis n'est PAS causée par une divergence
   dtype.
3. Le plan POINT 4-5 doit pivoter: pas d'audit dtype kernels TilingEngine,
   mais audit VALUE-level de leur sortie numérique.

## Coverage (sans env override)

| variant | total ops | seq cap | tri cap | div(in+out) | div(out only) |
|---|---|---|---|---|---|
| 4Kpx | 737 | 686 | 670 | 291 | 264 |
| 1024 | 737 | 737 | 737 | 293 | 266 |

Les counts post-POINT 2bis matchent ceux post-POINT 2 + env. Le
plumbing factory→executor lit le registry correctement.

## Classification dtype divergences 4Kpx (291 totales)

| classe | count | % | interprétation |
|---|---|---|---|
| COMMUNE_1024 | **291** | **100%** | Divergence dtype présente aussi sur Sana 1024 (qui marche). NBX intrinsèquement diverge de PyTorch oracle de manière cohérente partagée à toutes les échelles. Pas une cible POINT 4-5. |
| CASCADE_DOWNSTREAM | 0 | 0% | Aucune cascade — pas de chaîne de propagation de divergence racine. |
| STRUCTURELLE_RACINE | **0** | **0%** | Aucune divergence dtype racine 4Kpx-spécifique. POINT 1+2+2bis a tout fermé. |

## Méthode

Pour chaque divergence du walk 4Kpx (`match_in_out=false`):
1. **COMMUNE_1024**: même op_uid divergente dans walk 1024 avec
   profil dtype identique (seq_in/out + tri_in/out tous égaux)
2. **CASCADE_DOWNSTREAM**: au moins un input vient d'un producer
   op_uid lui-même divergent dans walk 4Kpx
3. **STRUCTURELLE_RACINE**: divergente sans cause amont (inputs
   tous produits par ops non-divergentes ou inputs externes
   `input::*` / `param::*`)

Méthode validée par spot-check: les ops Famille B identifiées avant
fixes (silu::21,22,23 + rms_norm::21,22,23) sont MAINTENANT TOUS
en match_in_out=true au 4Kpx ET 1024. La Famille B au sens dtype
est dissoute par POINT 1+2+2bis.

## Pivot — résiduel VALUE-LEVEL au 4Kpx

Étant donné que toutes les divergences dtype sont COMMUNE et que
Sana 4Kpx triton_sequential VAE-iso est encore garbage, la résidue
DOIT être value-level (même dtypes, valeurs différentes).

`value_divergence_analysis.py` cross-référence les `max_abs` des
walks 4Kpx vs 1024:
- Pour chaque op où `match_in_out=true` au 4Kpx (dtypes identiques),
  on regarde `tri_max_abs` vs `seq_max_abs` au 4Kpx
- Si rel diff > 10% au 4Kpx ET rel diff < 5% au 1024 → 4Kpx-spécifique
  value divergence

| variant | counts |
|---|---|
| Total value-divergent (4Kpx rel > 10%) | 63 |
| 4Kpx-specific (1024 rel < 5%) | **53** |

## 13 clusters value-divergence 4Kpx-spécifiques

Liste chronologique (op_idx ranges):

```
cluster 519..525  ( 4 ops)  relu::15 + transpose/expand/view propagation
cluster 532..538  ( 7 ops)  bmm::15 + slices propagation
cluster 587..593  ( 4 ops)  relu::17 cluster
cluster 600..606  ( 7 ops)  bmm::17 + slices
isolé   626       ( 1 op)   silu::17
isolé   635       ( 1 op)   convolution::48
cluster 642..653  (11 ops)  conv::49, silu::18, conv::50, rms_norm::18, add::68 (LARGE)
cluster 658..661  ( 4 ops)
cluster 668..669  ( 2 ops)
cluster 675..677  ( 2 ops)  silu::21 cluster
cluster 683..693  ( 7 ops)  silu::22 + rms_norm::22 + permute::78-83 cluster
cluster 700..701  ( 2 ops)
isolé   707       ( 1 op)
```

**Première divergence value-level chronologique: op 519 relu::15**
au lieu de la Famille B 675-696 que j'avais initialement identifiée.
Cohérent avec le cross-variant report d'avant les fixes (relu::15
était la TOP rel_ratio à 2.88M×).

Les clusters 519+532, 587+600 montrent un pattern: relu/silu suivi
d'un bmm + slices. Probable: TilingEngine tile certaines
matmul-class ou activations spatiales à 4Kpx, et les valeurs
divergent du path non-tilé en val mais pas en dtype.

## Implications POINT 4-5

Le plan original POINT 4 était "audit dtype kernels TilingEngine".
**Doit pivoter à "audit VALUE kernels TilingEngine"**.

Cibles factuelles:
- 13 clusters identifiés ci-dessus
- 53 ops value-divergentes 4Kpx-spécifiques
- Premières racines: op 519 relu::15 puis op 532 bmm::15
- Cluster le plus large: 642..653 (11 ops, conv::49 + silu::18 + conv::50)

Pour chacun, audit attendu:
- Quel kernel NBX est invoqué au 4Kpx (tilé vs non-tilé)?
- Quel algorithme (band streaming, halo, splitK, fused)?
- Comparer à PyTorch path: même algorithme ou différent?
- Si différent, est-ce le différentiel d'algorithme qui cause la
  divergence numérique?

## Files

- `dtype_walk_4kpx_post_2bis.tsv` (737 rows)
- `dtype_walk_1024_post_2bis.tsv` (737 rows)
- `classification_4kpx.tsv` (291 divergent rows × classification)
- `value_divergence_4kpx.tsv` (63 value-divergent rows + 4Kpx_specific flag)
- `top5_root_cascade_map.md` (N/A pour STRUCTURELLE_RACINE=0)

## Awaiting Hocine arbitrage

Le plan POINT 4-5 nécessite réajustement: au lieu d'audit dtype
TilingEngine, audit VALUE TilingEngine. Cible probable: op 519
relu::15 puis op 532 bmm::15 puis cluster 642-653.

Brief proposé pour POINT 4 (à ton arbitrage):
"Identifier pour chacun des 13 clusters 4Kpx-spécifiques value-divergents
quel kernel TilingEngine est invoqué et quel algorithme il utilise vs
le path non-tilé. Pas de fix code, just le diagnostic du couple
kernel-algorithme par cluster, et hypothèse de cause numérique."

POINT 5 reste "fix kernels TilingEngine" mais sa portée précise sera
déterminée par les findings du POINT 4 révisé.
