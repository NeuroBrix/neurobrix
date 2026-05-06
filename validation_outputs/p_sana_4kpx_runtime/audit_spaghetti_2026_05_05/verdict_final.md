# P-SANA-4KPX-RUNTIME — Verdict factuel final 2026-05-05/06

## Bisection scientifique aboutie — 4 modes, 4 verdicts factuels (post Fix B)

| Mode | Wall | Verdict | Live @ crash | OOM site | Cause racine |
|---|---|---|---|---|---|
| sequential | **90s** | **PASS** | 16.5 GB (torch) | — | — |
| compiled | 36-74s | **PASS** (CHANGELOG) | torch tracked | — | — |
| triton compiled (post Fix B) | 252s | **FAIL** | 25.3 GB (NBX) | conv::64 (chain retention) | structural multi-branch chain residue |
| triton_sequential (post Fix B) | similar | **FAIL** | ~25 GB (NBX) | rms_norm or conv::64 | idem |

**Fix Vector B (in-place residual add) implementé et fire correctement** —
la mesure factuelle montre que l'OOM site bouge de aten.add::86 (8 GiB
alloc refusée pré-fix) à aten.convolution::64 (différent op, même live
~25 GB). L'in-place add::86 succède (no new alloc, pixel_shuffle::4
freed at last use), mais la chaîne multi-branch globale retient
toujours 25 GB live à conv::64 — chaque add::XX::out_0 a 2 consumers
(next residual + downstream conv) donc reste alive jusqu'à ce que les
DEUX aient run.

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

### B. Multi-branch in-place fusion dans DC-AE decoder.up.0 (IMPLEMENTÉ commit 5b891f3)

**Implémenté et committé** comme universal detector dans
`OpLevelTilingEngine`. Détecte 26 candidats sur Sana 4Kpx VAE
(8 ≥ 4 GiB au 4096×4096, 4 au 2048×2048, 14 plus petits). Le fix
fire correctement: in-place add::86 succède (no new alloc), OOM site
moves de aten.add::86 → aten.convolution::64.

**Mais Fix B SEUL ne suffit pas** — la mesure factuelle post-fix montre
live_tracked toujours à 25 GB au conv::64. Raison: chaque
add::XX::out_0 a 2 consommateurs (next residual + downstream conv)
dans le multi-branch DC-AE, donc reste alive jusqu'à exécution des
DEUX. La chaîne 4Kpx retient 3-4 tenseurs de 8 GiB simultanément +
le 2048-level retient 4-5 de 4 GiB. Total ~24-30 GiB structurellement
alive AT conv::64.

**Fix B est donc NÉCESSAIRE mais pas SUFFISANT** pour Sana 4Kpx
triton sur 1× V100 32 GB. Il économise les nouvelles allocations
(8 GiB par add résiduel) mais ne réduit pas la rétention totale.
Universel — bénéficie tout modèle multi-branch décodeur.

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

## Recommandation factuelle (révisée post-Fix-B)

**Plus de quick win disponible** pour Sana 4Kpx triton 1× V100. Fix B
IMPLEMENTÉ et insuffisant (mesuré). Fix A ou C requis.

**A (multi-GPU NBX pipeline_parallel)** est maintenant le path le
plus court vers Sana 4Kpx triton fonctionnel:
- 24 GB peak fits trivialement sur 2× V100 32 GB = 64 GB
- Aligné philosophie NeuroBrix premier principe (multi-GPU+Prism)
- Débloque TOUTE la classe 4K+ en triton, pas que Sana
- Chantier P-MULTI-GPU-NBX-ADAPTER à ouvrir, ETA 2-4 semaines

**C (NBX caching allocator)** reste long terme. Long lift, peu
spécifique à Sana 4Kpx. Tabler sur A.

## Sortants de cette session

Commits:
- `f8375a9` fix(runtime): R30 op-level tiling parity + NBX bias broadcast OOM
- `9dcf588` docs(P-SANA-4KPX): bisection scientific verdict + 4-mode factual diff
- `5b891f3` feat(tiling): in-place residual add fusion (Sana 4Kpx peak ~8 GB savings)

Artefacts:
- `validation_outputs/p_sana_4kpx_runtime/audit_spaghetti_2026_05_05/`
  - `audit_report.md` — audit spaghetti (40 env vars, dead code, R30 audit code)
  - `etape3_diff_report.md` — bisection sequential vs triton_sequential pre-fix
  - `verdict_final.md` — ce document
  - `etape1_sequential.png` — Sana 4Kpx sequential PNG cohérente (R29 ✓)
  - `run_etape*.sh` — scripts de bisection reproductibles
  - `etape*.log` — logs factuels des runs

État chantier P-SANA-4KPX-RUNTIME (post Fix B):
- Sequential mode Sana 4Kpx: **DÉBLOQUÉ** par R30 fix (PNG cohérente)
- Compiled mode Sana 4Kpx: **DÉBLOQUÉ** (CHANGELOG existing PASS)
- Triton/triton_sequential modes Sana 4Kpx: **FAIL persistant** —
  Fix B implémenté et insuffisant. Fix A (multi-GPU NBX
  pipeline_parallel) requis comme chantier dédié pour débloquer.
  P-MULTI-GPU-NBX-ADAPTER ouvert, scope documenté.

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
