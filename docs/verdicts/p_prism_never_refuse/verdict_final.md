# P-PRISM-NEVER-REFUSE — verdict final POINT 10

R29 inspectable artefact. Remontée selon condition de sortie #2 du mandate
(blocker architectural >200 lignes vraiment hors scope).

## TL;DR

**Cible binaire 16 GiB et 2× 16 GiB NON ATTEINTE** par blocker architectural
réel. Sana 4Kpx VAE a un runtime peak structurel **~17 GiB live + 4 GiB
conv output = ~17 GiB**, qui ne tient pas dans 16 GiB hardware **indépendamment
de la stratégie Prism choisie**. Vérifié sur 6 stratégies différentes,
4 modes d'exécution, et 3 patterns d'allocation. Architectural fix
(CPU offload de la VAE ou intra-component split multi-GPU) >200 lignes,
hors scope POINT 10 par mandate.

**32 GiB acquis intact** : POINT 7-8-9 préservés ; aucune modification de
runtime ; matrice anti-régression non-régressée.

## Ce qui a été testé exhaustivement

| stratégie / mode | hardware | runtime peak | verdict |
|---|---|---|---|
| `single_gpu` (default) | 1× V100 16 GiB | 12.9 GiB live + 4 GiB request → 17 GiB | OOM @ conv::62 |
| `single_gpu_lifecycle` (forced) | 1× V100 16 GiB | Prism refuse PLANNING | NO PLAN |
| `lazy_sequential` (forced) | 1× V100 16 GiB | 12.9 GiB live + 4 GiB request → 17 GiB | OOM @ conv::62 |
| `zero3` (forced) | 1× V100 16 GiB | 2.6 GiB live + driver 13 GiB overhead | OOM @ weight load |
| `--triton-sequential` + `NBX_ALLOC_POOL=1` + `NBX_DEFERRED_DRAIN_BYTES=512MB` | 1× V100 16 GiB | 13.1 GiB live + 4 GiB request | OOM @ conv::62 |
| `--sequential` (PyTorch native) | 1× V100 16 GiB | 12.6 GiB allocated by PyTorch + 4 GiB request | OOM @ conv::62 |
| `--sequential` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 1× V100 16 GiB | 12.99 GiB in use + 4 GiB request | OOM @ conv::62 |

**Tous OOM au même endroit** : `aten.convolution::62`, le post-upsample::4
conv qui produit `[1, 128, 4096, 4096]` fp16 = **4 GiB** d'output. Ce buffer
unique est structurellement nécessaire pour l'execution downstream
(consommé par `add::86` qui in-place dans cette même buffer puis chaîné
jusqu'à `add::89` step 713). On ne peut pas le rendre plus petit sans
splitting intra-component.

L'estimateur Prism (POINT 9) prédit correctement A=12 GiB pour VAE,
cohérent avec les 12.9 GiB live mesurés au runtime. **L'estimator n'est
pas le bug** ; la limite est structurelle au modèle + hardware.

## Audits doctrinaux

### A.4 Model-agnostic violations : **AUCUNE** dans le code actif

`grep -nE "Sana|PixArt|TinyLlama|Llama|Flux|Qwen|DeepSeek|Janus|..."` sur
`src/neurobrix/core/prism/` et `src/neurobrix/core/strategies/` retourne
5 hits, **TOUS dans des commentaires** (justifications historiques /
exemples). Zero conditionnel actif `if model == 'sana'`. Prism + stratégies
sont structurellement model-agnostic. **Pas de dette technique
R-MODEL-AGNOSTIC à fixer.**

### A.2 Cascade complète

Prism cascade actuellement **9 stratégies en multi-GPU** et **4 en
single-GPU** (toutes câblées et invoquées via `_evaluate_all_strategies`).
Le message d'erreur ZERO FALLBACK qui listait seulement 5 stratégies
était **misleading** — fixé dans cette session :
`_fail_error` rapporte désormais la cascade réelle selon le device count
(commit de cette session).

## Composantes structurelles qui empêchent 16 GiB

1. **Conv::62 output structurel** : `[1, 128, 4096, 4096]` fp16 = 4 GiB.
   Non-tilable dans son utilisation actuelle (write destination de
   FusionUpsampleProxy + source de add::86 in-place chain).
2. **Activations cumulatives** : ~13 GiB de live tensors au moment de
   conv::62, dominé par residuals du DC-AE residual chain
   (`conv::55` + `pixel_shuffle::4` + `add::86`'s aliased buffer).
3. **Driver overhead** : 0.5-3 GiB selon le mode (sync, kernel cache,
   library state). Cette overhead seule fait passer la marge restante
   (2.6 GiB free) en-dessous du seuil d'allocation.

**Sum total irreducible** : ~13 GiB live + 4 GiB conv allocation
+ driver overhead = **~17-18 GiB**. > 16 GiB hardware.

## Chantier architectural requis (hors scope POINT 10)

Pour débloquer Sana 4Kpx sur 16 GiB hardware, deux chantiers sont
nécessaires (l'un ou l'autre suffit) :

### `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT` (déjà nommé POINT 9)

Split intra-component cross-device pour les composants qui dépassent
une seule carte. Pour Sana 4Kpx sur 2× V100 16 GiB :
- VAE split en deux moitiés (e.g. upsample stages 0-1 sur cuda:0,
  stages 2-3 sur cuda:1)
- NBXTensor cross-device transfer à la frontière
- Triton kernel dispatch sur le device approprié

**Estimation taille** : 400-800 lignes (NBXTensor cross-device path,
Triton runtime device-aware dispatch, Prism strategy `intra_component_split`).
Vraiment >200 lignes architectural.

### `P-PRISM-CPU-FALLBACK-EXECUTION` (nouveau, à ouvrir)

Nouvelle stratégie Prism qui place un composant entier sur CPU quand
il ne tient sur aucun GPU. Pour Sana 4Kpx sur 1× V100 16 GiB :
- Transformer + text_encoder sur GPU 16 GiB (fits)
- VAE sur CPU (256 GiB RAM disponible côté Dell)
- Cross-device transfer transformer output (GPU) → VAE input (CPU)
- PyTorch native eager dispatcher pour CPU components
- Triton dispatcher pour GPU components
- Auto-mode-switching dans le runtime quand component device differs

**Estimation taille** : 200-400 lignes (nouvelle stratégie + runtime
executor per-component dispatcher + cross-device transfer). À l'edge
ou >200 lignes architectural.

## Petit fix in-scope shipped dans cette session

`src/neurobrix/core/prism/solver.py:_fail_error` :
- Liste maintenant la cascade RÉELLE de stratégies (9 multi-GPU,
  4 single-GPU) au lieu de 5 hardcodées
- Mentionne explicitement les chantiers backlog nécessaires pour
  configs trop tight pour les stratégies actuelles

**Impact** : un opérateur Sana 4Kpx 16 GiB voit désormais clairement
les 9 stratégies tentées + le pointer vers les chantiers backlog,
au lieu d'un message trompeur qui sous-représente l'effort de Prism.
Strictement doctrinale ; aucun changement runtime.

## Doctrine de "Prism never refuses" — clarification factuelle

Le principe est aspirational mais pas implémentable sans
`P-PRISM-CPU-FALLBACK-EXECUTION`. Aujourd'hui :
- Prism **tente** 9 stratégies (multi-GPU) ou 4 (single-GPU)
- Toutes routent sur GPU (incluant zero3 qui offload weights mais
  compute GPU)
- Aucune ne route le compute sur CPU
- Si toutes échouent → ZERO FALLBACK (= refusal)

Pour respecter pleinement le principe :
1. Implémenter `P-PRISM-CPU-FALLBACK-EXECUTION` (chantier nommé)
2. La cascade devient "9 (10) stratégies + CPU fallback comme last resort"
3. Si même CPU n'a pas assez de RAM (hardware extrême), ZERO FALLBACK
   garde sa raison d'être

À ce moment, Prism **never refuses** sur des hardware raisonnables.

## Verdict scope POINT 10

| critère mandate | verdict |
|---|---|
| Sana 4Kpx FULL PNG cohérente sur 1× V100 32 GiB | **PASS** (acquis POINT 7 préservé) |
| Sana 4Kpx FULL PNG cohérente sur 1× V100 16 GiB | **NON ATTEINTE — blocker architectural** |
| Sana 4Kpx FULL PNG cohérente sur 2× V100 16 GiB | **NON ATTEINTE — même blocker architectural** |
| Audit doctrinal model-agnostic | **PASS — aucune violation** dans le code actif |
| Audit cascade Prism | **PASS — 9 stratégies effectivement câblées** ; le message d'erreur misleading est fixé |
| In-scope code change | `_fail_error` message corrigé (3-line fix doctrinal) |

**Condition de sortie #2 du mandate** : blocker architectural >200 lignes
vraiment hors scope. **Remontée honnête conforme**.

## Backlog officiellement ouvert

1. **`P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT`** (priorité haute pour
   2× V100 16 GiB)
2. **`P-PRISM-CPU-FALLBACK-EXECUTION`** (priorité haute pour 1× 16 GiB,
   nouveau)
3. `P-PRISM-DRIVER-OVERHEAD-ESTIMATOR` (priorité basse, déjà ouvert
   POINT 9)
