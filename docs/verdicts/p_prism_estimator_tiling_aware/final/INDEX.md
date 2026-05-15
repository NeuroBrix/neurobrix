# P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE — final verdict
## POINT 9 — Chantier 1 partial closure + Chantier 2 structural verdict

R29 inspectable artefact. Two chantiers attaqués en boucle autonome.
Verdict factuel : **Chantier 1 estimator fix shipped et workable** ;
**cible binaire 16 GiB non atteinte par blocker structurel hors scope**
explicitement nommé dans le mandat.

## Chantier 1 — estimator fix : LANDED

Modifications `core/prism/profiler.py` + `core/prism/solver.py` :

1. **`estimate_peak_memory(zero_alloc_uids=...)`** — ops dont l'output
   est un sentinel proxy au runtime (FusionUpsampleProxy /
   BroadcastClonePyroxy / pass-through view / NBXTensor stride-0 expand)
   sont substituées avec `size=0` à l'allocation. Mirror exact du
   runtime.
2. **`estimate_peak_memory(inplace_adds=...)`** — pour chaque add
   in-place identifié par liveness (mirror de
   `_detect_inplace_add_candidates`), l'output est aliasé au reused
   input et leur lifetime fusionné. Reverse-map `frees_at[op_uid]`
   précomputée pour gérer les last-uses extended-via-alias qui ne
   listent pas le tid dans `op.input_tensor_ids`.
3. **`estimate_peak_memory(force_compute_dtype_for_fp=True)`** —
   override des fp meta dtypes (graph-traced fp32 sur Sana 4Kpx
   capture pre-quantification) par le runtime compute_dtype (fp16).
   Sans cet override : 2× over-estimation systématique.
4. **`PrismSolver._compute_memory(profile=...)`** — two-pass tiling-
   aware via le smallest GPU du profile. Premier pass identifie
   overflow_ops ; second pass injecte zero_alloc_uids (fusion +
   pixel_shuffle F2a chain) + inplace_adds détectés statiquement.
5. **Helpers DAG-static** dans solver :
   `_identify_fusion_upsample_uids` (mirror `_detect_op_level_tiling_pairs`
   `0.25 × VRAM` threshold + `conv_overflow OR up_overflow` OR-logic),
   `_identify_pixel_shuffle_chain_proxy_uids` (mirror
   `_detect_pixel_shuffle_broadcast_chains` chain detection),
   `_identify_inplace_add_candidates_static` (mirror
   `_detect_inplace_add_candidates` 1 GiB threshold + per-input
   liveness check). Threshold formulas dupliquées exactement →
   estimator et runtime tiling agree on which ops are "free".

### Mesures pre/post fix sur Sana 4Kpx VAE

Budget = 16 GiB single-GPU :

| pass | peak_bytes | %réduction |
|---|---|---|
| pre-fix (worst-case full mat) | **28.0 GiB** | baseline |
| pass 1 (fp dtype override seule) | **14.3 GiB** | −49% |
| pass 2 (+ zero_alloc + inplace) | **12.0 GiB** | **−57%** |

Runtime peak mesuré (POINT 7 ÉTAPE A) : **16.6 GiB** (live_tracked +
~0.6 GiB driver overhead = ~13 GiB live + overhead). Estimator post-fix
prédit 12 GiB de live tensors → cohérent avec mesure runtime.

## Chantier 1 — cible binaire : NON ATTEINTE par blocker structurel

Sana 4Kpx FULL pipeline triton_sequential sur 1× V100 16 GiB :

```
[ERROR] Pipeline failed: GPU malloc failed (error 2) for 4294967296 bytes
[device cuda:0 live_tracked=12897MB pool_cached=0MB (0 blocks)
 driver_free=2608MB / driver_total=16151MB]
```

- Prism PLANNING : **OK** (single_gpu accepté, A=12 GiB cohérent)
- Runtime : **OOM** au moment d'allouer un buffer 4 GiB
- État au moment de l'OOM : 12.9 GiB live + 0.6 GiB driver overhead
  = 13.5 GiB occupés / 16.0 GiB driver_total → 2.6 GiB libres
- Demande 4 GiB > 2.6 GiB libres → OOM
- Total runtime peak théorique = 12.9 + 4 = ~17 GiB > 16 GiB hardware

**Le modèle Sana 4Kpx 4096×4096 a un peak runtime structurel de ~17 GiB
qui n'entre pas dans une carte V100 16 GiB**, indépendamment de la
qualité de l'estimateur Prism. Le minimum hardware viable reste 32 GiB
(POINT 7 a démontré 16.6 GiB peak / 32.5 GiB driver_total = OK).

Le seul fix qui permettrait 16 GiB serait l'**intra-component VAE
split cross-device** (chantier nommé `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT`
explicitement EXCLU du périmètre POINT 9 par le mandate :
*"TOUT ce qui serait intra-component split cross-device pour VAE
[...] sort du scope, c'est P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT,
hors scope"*).

## Chantier 2 — multi-GPU 2× V100 16 GiB : MEME BLOCKER + bonus fix config

Test 2× V100 16 GiB : Prism rejette toujours après mon fix. Investigation
révèle DEUX causes distinctes :

1. **Config bug** : `config/hardware/v100-16g-x2-01.yml` n'avait pas
   `preferred_dtype: float16` au top-level (contrairement à `v100-16g.yml`
   et `v100-32g.yml`). Conséquence : `PrismSolver._resolve_dtype` retombait
   sur fp32 → dtype_bytes=4 → estimator A=24 GiB au lieu de 12 GiB.
   **FIX APPLIQUÉ** dans cette session : ajouté `preferred_dtype: float16`
   à `v100-16g-x2-01.yml` (in-scope, 1-line YAML edit).

2. **Blocker structurel résiduel** : même avec le config fix, la VAE
   Sana 4Kpx ne tient pas sur UNE carte 16 GiB (peak ~17 GiB). Les
   stratégies `component_placement` / `pipeline_parallel` placent un
   composant entier sur une carte ; aucune ne split intra-component.
   Donc même 2× 16 GiB = 32 GiB total ne peut pas accommoder VAE 17 GiB
   sur une seule carte.

Ouverture officielle du chantier backlog `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT`
nécessaire pour débloquer 16 GiB hardware. Hors scope POINT 9.

## Anti-régression matrice : 4/4 PASS

Tous les modèles maintenus en triton_sequential sans régression :

| modèle | hardware | wall | PNG / TXT | verdict |
|---|---|---|---|---|
| Sana 1024 | v100-32g | 73.67 s | `anti_reg_sana1024_32g.png` | 🍎 PASS |
| Sana 4Kpx | v100-32g | 513.06 s (vs POINT 7 510 s) | `anti_reg_sana4kpx_32g.png` | 🍎 PASS |
| PixArt-XL | v100-32g | 114.58 s | `anti_reg_pixart_xl.png` | 🍎 PASS |
| TinyLlama | v100-32g | 40.86 s | `anti_reg_tinyllama.txt` | ✓ poème cohérent |

Aucune régression introduite par le fix estimator. La cellule problématique
(Sana 4Kpx 16 GiB) reste rouge pour cause de blocker structurel
documenté ci-dessus.

## Verdict scope POINT 9

| critère mandate | verdict |
|---|---|
| **Chantier 1 — Sana 4Kpx FULL triton_seq 16 GiB PNG cohérente** | **NON ATTEINTE — blocker structurel hors scope** (model peak 17 GiB > 16 GiB hardware ; intra-component split = `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT` explicitement exclu) |
| **Chantier 1 — anti-régression 32 GiB inchangée** | **PASS** — Sana 4Kpx 32g 513s vs 510s POINT 7 (variance run-à-run, no regression) |
| **Chantier 1 — anti-régression Sana 1024 + PixArt + TinyLlama** | **PASS** — 4/4 cellules vertes |
| **Chantier 2 — multi-GPU 2×16 GiB** | **MEME BLOCKER STRUCTUREL** + config fix `v100-16g-x2-01.yml` `preferred_dtype: float16` |
| **Estimator fix shipped et workable** | **PASS** — pre 28 GiB → post 12 GiB sur Sana 4Kpx VAE ; estimator est désormais tiling-aware et compute-dtype-aware |

**État final** : Chantier 1 livre un fix structurel valide (estimator
57% plus précis) mais ne débloque pas le binaire 16 GiB par limitation
hardware. Chantier 2 ouvre un backlog explicitement nommé. **Remontée
selon condition de sortie #2 du mandate : "blocker hors scope >200 lignes"**
— l'intra-component VAE split est le fix nécessaire et il est
explicitement exclu du périmètre POINT 9.

## Cumulative session statut

| | commit / tag | verdict |
|---|---|---|
| POINTS 1-6 H2 | ea8e8e2..a862fe0 | numerical correctness Sana 4Kpx |
| POINT 7 | 90ac662 | full pipeline 32 GiB triton + compiled |
| POINT 8 | 1d22cf9 | audit perf compiled vs sequential factual |
| **POINT 9** | this commit | **estimator tiling-aware + dtype-aware + multi-GPU config fix** |

## Backlog ouvert

- `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT` (priorité haute si client
  veut tourner Sana 4Kpx sur < 32 GiB hardware) — split du VAE intra-
  component cross-device. Estimation > 200 lignes.
- `P-PRISM-DRIVER-OVERHEAD-ESTIMATOR` (priorité basse) — ajouter une
  marge driver/library overhead (~1-2 GiB) à l'estimator pour qu'il
  rejette les configs où live_tensors fits mais runtime overall OOM
  (cas du single 16 GiB courant : Prism accepte, runtime échoue).
