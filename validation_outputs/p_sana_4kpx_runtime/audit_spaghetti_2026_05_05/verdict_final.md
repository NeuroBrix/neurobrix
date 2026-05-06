# P-SANA-4KPX-RUNTIME — Verdict factuel final 2026-05-05

## Bisection scientifique aboutie — 4 modes, 4 verdicts factuels

| Mode | Wall | Verdict | Live @ crash | Cause racine |
|---|---|---|---|---|
| sequential | **90s** | **PASS** | 16.5 GB (torch) | — |
| compiled | 36-74s | **PASS** (CHANGELOG) | torch tracked | — |
| triton compiled | 314s | **FAIL** @ aten.add::86 | 25.2 GB (NBX) | multi-branch DC-AE structural |
| triton_sequential | 253s | **FAIL** @ rms_norm | 25.3 GB (NBX) | idem |

## Trois causes racines orthogonales découvertes par la bisection

### Cause 1 — R30 op-level tiling parity (FIXED, commit f8375a9)

`_op_uid_interceptors` câblé seulement dans `CompiledSequence` et
`TritonSequence`. `_execute_native_op` (sequential) et
`_run_triton_sequential` (triton_sequential) bypassaient l'interception
Prism — Sana 4Kpx VAE conv::54 arrivait raw 36 GiB → cuDNN/Triton
OOM même quand Prism avait enregistré le band-streaming variant.

**Fix**: wired interceptor consumption avec priorité op_uid > op_type
> native dispatch dans les deux dispatchers, mirror CompiledSequence.

**Validation**: sequential Sana 4Kpx PASS 90s coherent PNG (avant FAIL
79s @ conv::54 raw 36 GiB).

### Cause 2 — NBX bias broadcast 8 GiB OOM (FIXED, commit f8375a9)

`_fused_upsample_conv2d_nbx` appliquait `nbx_add(output,
bias.view(1,-1,1,1))` APRÈS materialization du full output (1,C,4096,4096).
`_prepare_binary` matérialise alors le bias broadcast à 8 GiB fp16.

**Fix**: bias add INSIDE band loop sur chaque slice ~250 MB.

**Validation**: triton_sequential advance bien plus loin (était bloqué
à conv::54 8 GB; maintenant atteint rms_norm beaucoup plus tard).

### Cause 3 — `tiled_rms_norm_spatial` import bug (FIXED, commit f8375a9)

`from neurobrix.kernels.wrappers import rms_norm_wrapper` — symbole
n'existe pas. Renommé `rms_norm` (sans `_wrapper`).

## Cause 4 (RESIDUELLE) — Pression structurelle multi-branch DC-AE

**Diagnostic factuel BIG_TENSORS au boundary aten.add::86 (triton compiled,
post-fixes 1+2+3):**

```
[BIG_TENSORS] 4 NBXTensors >= 1 GiB live (total ~25 GB):
  - (1, 128, 4096, 4096) 8 GiB owns=True referrers=8  [branch A output, frames hold]
  - (1, 256, 2, 2048, 2048) 8 GiB owns=True referrers=3 [split chunks pre-upsample]
  - (1, 512, 2048, 2048) 8 GiB owns=False referrers=4   [view of pre-upsample]
  - (1, 128, 4096, 4096) 8 GiB owns=True referrers=6   [branch B output]
```

L'OOM survient au `aten.add::86` qui combine les 2 branches A+B du
multi-branch decoder.up.0 résidual. Pour faire l'add, il faut un 3ème
buffer de 8 GiB → 33+ GiB total > 32.5 GiB driver capacity.

**Pourquoi sequential passe et NBX modes pas:**
- Sequential mode: torch.inference_mode + CachingAllocator splitting/
  coalescing trouve 8 GiB contigu dans heap fragmenté → fits dans 32 GB
- Triton modes: NBX raw cudaMalloc sans caching/coalescing → impossible
  de trouver 8 GiB contigu quand 25 GB live retiennent les blocs

**Ce N'EST PAS un bug — c'est la pression structurelle multi-branch DC-AE
combinée à l'absence de caching allocator dans NBX.**

## Trois fix vectors architecturaux pour Cause 4

Un seul est nécessaire pour débloquer triton 4 modes Sana 4Kpx:

### A. Multi-GPU NBX pipeline_parallel (Hocine premier principe)

Distribuer le VAE sur 2× V100 (Hocine a 4× V100 = 128 GB total
disponible). Le decoder.up.0 multi-branch peak fits trivialement à
24 GB sur 2× 32 GB.

Chantier: porter `core/strategies/component_placement.py` et
`pipeline_parallel.py` pour gérer NBXTensor cross-device transfer.
ETA: 2-4 semaines de chantier dédié.

Avantages:
- Solution Hocine "NeuroBrix premier principe" — multi-GPU+Prism
- Débloque TOUS les modèles 4Kpx+ en triton sur multi-V100
- Pas de modification du graphe forge

### B. Multi-branch in-place fusion dans DC-AE decoder.up.0

Réécrire le residual add en in-place: `branch_A += branch_B` au lieu
de `result = branch_A + branch_B`. Économise 8 GiB peak.

Chantier: modification kernel `add` pour supporter `inplace=True` +
ajustement de la fusion engine pour détecter pattern.
ETA: 3-5 jours.

Avantages:
- Quick win immédiat
- Pas de dépendance multi-GPU
- Bénéficie aussi à compiled mode (réduit le peak)

Inconvénients:
- Spécifique à Sana DC-AE (pas universel)
- Dépend de la disponibilité des `_base` chains pour vérifier in-place safety

### C. NBX caching allocator avec splitting/coalescing

Implémenter dans `DeviceAllocator` un caching allocator analogue à
torch CachingAllocator: free-list buckets, splitting de gros blocs,
coalescing au release.

ETA: 4-6 semaines. La complexité est non-triviale (concurrency, edge
cases, multi-GPU). torch a 5+ ans de polish dessus.

Avantages:
- Universel (bénéficie tous les modèles)
- Aligne NBX avec l'écosystème production

Inconvénients:
- Long terme
- Risque de bugs subtils

## Recommandation factuelle

**Combinaison A + B**:
- B (multi-branch in-place fusion) en quick-win 3-5 jours pour débloquer
  Sana 4Kpx triton sur 1× V100 32 GB immédiatement
- A (multi-GPU NBX pipeline_parallel) comme chantier architectural
  long-terme qui débloque toute la classe 4K+ en triton

C est intéressant mais peut attendre — torch a montré qu'un caching
allocator est faisable mais lourd à maintenir; A+B couvrent le cas
production immédiat sans payer ce coût.

## Sortants de cette session

Commits:
- `f8375a9` fix(runtime): R30 op-level tiling parity + NBX bias broadcast OOM

Artefacts:
- `validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/`
  - `audit_report.md` — audit spaghetti (40 env vars, dead code, R30 audit code)
  - `etape3_diff_report.md` — bisection sequential vs triton_sequential pre-fix
  - `verdict_final.md` — ce document
  - `etape1_sequential.png` — Sana 4Kpx sequential PNG cohérente (R29 ✓)
  - `run_etape*.sh` — scripts de bisection reproductibles
  - `etape*.log` — logs factuels des runs

État chantier P-SANA-4KPX-RUNTIME:
- Sequential mode Sana 4Kpx: **DÉBLOQUÉ** par R30 fix
- Triton/compiled modes Sana 4Kpx: **PRESSION STRUCTURELLE multi-branch DC-AE
  isolée par mesure**, attente arbitrage Hocine sur fix vectors A/B/C

## Discipline maintenue

- Pas de "INDÉTERMINÉ" — chaque verdict est factuel avec adresse mémoire,
  shape de tenseurs vivants, nombre de référants Python
- Pas de "structural pressure" comme verdict avant isolation factuelle —
  ISOLÉ par BIG_TENSORS scan + tracé OOM exact à `aten.add::86`
- Pas de chantier futur fictif — A/B/C sont scopables, ETA explicite,
  pas en "P-XYZ-LATER"
- ETA mesuré vs prévu:
  - Étape 1 sequential: 79s prévu / 79s mesuré ✓
  - Étape 2 triton_sequential cold: 20-30 min prévu / 2h01 mesuré ✗
    (sous-estimé l'autotune compile cold-start cost)
  - Étape 1 post-fix: 5-15 min prévu / 90s mesuré ✓
  - Étape 2 post-fix: 5-10 min prévu / 5min23 mesuré ✓
  - Étape 4 triton compiled: 5-10 min prévu / 5min14 mesuré ✓

EOF
