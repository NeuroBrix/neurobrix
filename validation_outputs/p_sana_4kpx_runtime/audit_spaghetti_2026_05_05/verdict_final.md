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

## Recommandation factuelle (révisée post-leak-diagnostic 2026-05-06)

**Diagnostic [BIG_TENSORS] avec tid identification au OOM révèle un
ORPHAN:** sur les 3 × 8 GiB live au conv::64, l'arena n'en track que
2 (silu::24::out_0 + add::86::out_0). Le 3ème tensor est ORPHAN
(`tid=ORPHAN(not in arena)`) — alive par Python refs hors arena.

C'est **case (b) liveness divergence** (non case (a) structural pur).
Si on libère cet orphelin, live drop 25 → 17 GB, conv::64 alloc 8 GB
→ 25 GB total ≤ 32 GB driver = **PASS Sana 4Kpx triton sans
multi-GPU**.

**Avant d'ouvrir P-MULTI-GPU-NBX-ADAPTER (2-4 semaines), un nouveau
chantier court P-TRITON-LIVE-LEAK-AUDIT** pour identifier la source
du leak ORPHAN:

1. Référants observés: `list[len=3] × 2 + tuple(len=2)` (inhabituel
   pour arena-managed)
2. Pas dans _deferred queue (drain récent confirmé par
   deferred_queue=0)
3. Suspects:
   - `other.contiguous()` / `other.to(dtype)` dans add_inplace_nbx
     créent NBXTensor temporaires dont le ref persiste
   - silu output capturant conv::63::out_0 via view `_base` chain
   - args_resolver retient un tuple ancien via Python frame lifecycle
4. Méthode: instrumentation par-op des `gc.get_referrers` avec
   tid identification (déjà en place commit en cours), bisection
   par désactivation de chaque suspect

ETA P-TRITON-LIVE-LEAK-AUDIT: 1-3 jours selon complexité du leak.

**Plus de quick win disponible avant ce diagnostic** — l'in-place add
fix B est nécessaire mais pas suffisant tant que le leak ORPHAN existe.

**A (multi-GPU NBX pipeline_parallel)** déclassé en backup au cas où
P-TRITON-LIVE-LEAK-AUDIT échoue. Si le leak est fixé, multi-GPU
n'est PAS nécessaire pour Sana 4Kpx (seul cas qui aurait justifié A).

**C (NBX caching allocator)** reste long terme.

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

État chantier P-SANA-4KPX-RUNTIME (post bisection 3-suspects 2026-05-06):
- Sequential mode Sana 4Kpx: **DÉBLOQUÉ** par R30 fix (PNG cohérente)
- Compiled mode Sana 4Kpx: **DÉBLOQUÉ** + dual-backend regression fix
  (PNG cohérente 82s post-fix)
- Triton/triton_sequential modes Sana 4Kpx: **FAIL — cause racine
  factuellement identifiée** par bisection des 3 suspects + NBX_MALLOC_TRACE
  + tid identification dans BIG_TENSORS:

  **Suspect 1 (silu _base chain) — RÉFUTÉ** par audit code: silu wrapper
  alloue output via `NBXTensor.empty_like(x)` qui crée NBXTensor avec
  `_base=None` (pas de view chain).

  **Suspect 2 (args_resolver lifecycle) — RÉFUTÉ** par audit code:
  closures capturent slot indices (int), pas tensors.

  **Suspect 3 (autre op du chain) — IDENTIFIÉ FACTUELLEMENT**:

  Pattern Sana DC-AE pixel_shuffle (exec[700-704]):
  ```
  unsqueeze::5 (1,256,1,2048,2048) — view
  expand::41 (1,256,2,2048,2048) — broadcast view (stride=0 dim 2)
  clone::5   (1,256,2,2048,2048) — ALLOC 8 GiB (matérialise broadcast)
  view::95   (1,512,2048,2048) — view de clone::5
  pixel_shuffle::4 (1,128,4096,4096)
  ```

  L'orphan tensor#3 de la BIG_TENSORS scan correspond exactement à
  `aten.clone::5::out_0` shape (1,256,2,2048,2048) owns=True 8 GiB.
  Tensor#1 view::95::out_0 (view de clone::5).

  Le clone matérialise un broadcast pour fournir un input contiguous à
  pixel_shuffle. Les 256 channels sont DUPLIQUÉS via expand, puis copiés
  en mémoire — 8 GiB alloués pour des données redondantes.

  **Confirmation par bisection NBX_DISABLE_INPLACE_ADD=1**: avec
  l'in-place add désactivé, les orphans deviennent EXPLICITEMENT
  `aten.clone::5::out_0` + `aten.view::95::out_0`. La cause est
  STRUCTURELLE au DAG forge'd, pas au runtime triton.

- **Next fix concret (intra-P-SANA-4KPX-RUNTIME, pas nouveau chantier):**

  **Fix structural pixel_shuffle broadcast elimination**:
  - **Option F1**: graph-level pass dans TritonSequence.compile qui détecte
    le pattern `expand → clone → view → pixel_shuffle` et élimine le clone
    en routant la stride directement à pixel_shuffle (qui en interne lit
    les channels broadcastés en re-mappant `ic >= C/2` vers `ic - C/2`).
    Économie: 8 GiB peak. Univeral pour tous les pixel_shuffle dans DC-AE.
    ETA: 60-90 min implémentation + test.

  - **Option F2**: pixel_shuffle wrapper détecte broadcast input via
    stride pattern et ajuste la lecture dans le kernel sans matérialiser.
    ETA: 30-60 min wrapper-level.

  Si F1+F2 résolvent: triton modes débloqués SANS multi-GPU.
  Si pas suffisant: P-MULTI-GPU-NBX-ADAPTER backup.

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
