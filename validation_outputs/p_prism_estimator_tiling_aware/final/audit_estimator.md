# P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE — audit ÉTAPE A

R29 inspectable artefact. Audit pre-fix : où exactement Prism rejette,
quand OpLevelTilingPlan est construit, et pourquoi l'estimation peak
n'en tient pas compte.

## Q1 — Où le solver rejette quand `peak_bytes > vram_budget` ?

`src/neurobrix/core/prism/solver.py:505` :

```python
component_memory = self._compute_memory(container, neural_components,
                                         input_config, target_dtype_str)
```

`_compute_memory()` (ligne 930) calcule pour chaque composant
`activation_bytes = profile.peak_bytes` (ligne 979) **sans** passer
`vram_per_gpu_bytes` au profiler (ligne 975-978). Conséquence : pas
de calcul des `overflow_ops`, peak en worst-case full-materialization.

`total_bytes = weight_bytes + activation_bytes + overhead` est ensuite
utilisé par les 5 stratégies en cascade (`_evaluate_all_strategies`
ligne 570). Chaque stratégie compare ce `total_bytes` à la capacité
de la GPU candidate. Si aucune ne fit → `_fail_error()` ligne 617 →
`ZERO FALLBACK: No strategy can fit this model`.

Le rejet est donc **structurel à la passe d'estimation initiale**, pas
à une stratégie en particulier.

## Q2 — Quand OpLevelTilingPlan est construit, est-il utilisé pour DÉCIDER ou seulement pour CONFIGURER ?

`src/neurobrix/core/prism/solver.py:710-718` :

```python
# Step 7b: Op-level tiling — detect upsample→conv fusion pairs whose
# intermediate tensor would OOM the assigned GPU. Decision happens
# AFTER strategies are picked (allocations known) so we know the per-
# component VRAM budget. Plan stored in plan.runtime_op_tiling for
# the runtime executor to wire as op_uid interceptors.
plan.runtime_op_tiling = self._detect_op_level_tiling_pairs(
    container, neural_components, allocations, profile, input_config,
    target_dtype_str,
)
```

**Réponse claire** : seulement pour CONFIGURER le runtime. Le plan
est construit à l'étape 7b, **après** que la stratégie ait été choisie
à l'étape 4 (ligne 570). Donc le plan n'influence pas la décision de
stratégie. Quand la stratégie échoue (parce que peak_bytes
worst-case > budget), `_detect_op_level_tiling_pairs` n'est même pas
appelé : le ZERO FALLBACK est levé avant.

Le commentaire du code l'admet explicitement : *"Decision happens
AFTER strategies are picked"*.

## Q3 — Estimation peak per-tile pour les ops VAE Sana 4Kpx en overflow

Lecture `_detect_op_level_tiling_pairs` (ligne 724–920) :

- Détection des overflow_ops : appel `profiler.estimate_peak_memory(...,
  vram_per_gpu_bytes=comp_vram)` (ligne 762-768). Calcule
  `overflow_ops` au seuil `safety=0.85 × comp_vram`.
- Pour chaque upsample → conv adjacent dans `execution_order` :
  vérifie si le conv est dans overflow_ops OU si l'upsample produit
  > 25% du budget VRAM. Si oui, ajoute la fusion pair au plan avec
  un tile_factor calculé analytiquement (ligne 822-862).
- Pour les conv standalone overflow restants (ligne 870-886) : ajoute
  comme tiled_op avec tile_factor.
- Pour les rms_norm dont l'output > 20% VRAM (ligne 894-915) : ajoute
  comme tiled_op.

**Le plan capture la connaissance** que les ops overflow seront
matérialisées en bandes au runtime. **Mais cette connaissance n'est
pas re-injectée dans peak_bytes pour la décision de stratégie**.

## Q4 — Quantification du gap entre estimation et runtime

POINT 7 ÉTAPE C1 (1× V100 16 GiB rejected) : `_fail_error` rapporte
pour Sana 4Kpx :

```
vae: 31356MB (W=1191, A=28672)
```

A=28 GiB. Au runtime mesuré sur 1× V100 32 GiB (POINT 8 baseline) :
**peak runtime = 16.6 GiB**. **Gap = 11.4 GiB / 70% sur-estimation**.

Le gap est dominé par les upsample outputs en fusion pairs qui à
runtime sont des `FusionUpsampleProxy` (sentinel, `_nbytes = 0`),
donc 0 byte alloué. Estimation des upsample outputs full-materialization
Sana 4Kpx VAE :

| op | shape | bytes (fp16) |
|---|---|---|
| upsample_nearest2d::3 (1024→2048) | [1, 512, 2048, 2048] | 4 GiB |
| upsample_nearest2d::4 (2048→4096) | [1, 256, 4096, 4096] | 8 GiB |
| +autres upsamples in-block | | ~0-2 GiB |

Total éliminé par fusion ≈ 12-14 GiB. Cohérent avec le gap mesuré de
11.4 GiB (les upsamples in-block sont partiellement libérés par
last-use analysis de toute façon).

## Choix d'approche pour le fix

**Approche B (two-pass dans solver.py)** retenue :

1. Premier passe dans `_compute_memory` : `estimate_peak_memory` avec
   `vram_per_gpu_bytes = smallest_GPU_in_profile` → identifie
   overflow_ops.
2. Identification des upsamples-en-fusion-pair via une nouvelle
   méthode `_identify_fusion_upsample_uids` qui extrait la logique
   de détection upsample→conv adjacency de `_detect_op_level_tiling_pairs`
   (sans calculer les tile_factors — pas nécessaires pour l'estimation).
3. Deuxième passe : `estimate_peak_memory(fusion_upsample_uids=...)`
   substitue size=0 pour les outputs des upsamples en fusion (mirror
   du runtime FusionUpsampleProxy comportment).
4. `activation_bytes = ap2.peak_bytes`.

Pourquoi B vs A : A polluerait l'API du profiler avec un argument
OpLevelTilingPlan (couplage tiling_engine → profiler). B garde le
profiler agnostique de la plan structure — il accepte juste un set
de op_uid à zéro-allouer, qui est le comportement runtime exact
(FusionUpsampleProxy). Le solver garde l'orchestration two-pass
localement, à un seul endroit.

L'API du profiler change minimalement : nouveau paramètre kwarg
`fusion_upsample_uids: Optional[Set[str]] = None`. R30 / R32 / R33
non concernés (changement pur estimation, pas runtime).

## Impact attendu

| modèle | A estimé pre-fix | A estimé post-fix attendu | A runtime mesuré |
|---|---|---|---|
| Sana 4Kpx VAE | 28 GiB | ~14-16 GiB | 16.6 GiB |
| Sana 1024 VAE | inchangé | inchangé | inchangé (pas d'overflow) |
| PixArt-XL / Sigma | inchangé | inchangé | inchangé |
| TinyLlama | inchangé | inchangé | inchangé |

Pour Sana 4Kpx, le post-fix devrait permettre à `single_gpu` d'accepter
un budget 16 GiB. Anti-régression : pour les modèles qui ne montent
pas en overflow_ops, le second pass n'est pas déclenché (économise
le re-scan), comportement strictement identique au pre-fix.
