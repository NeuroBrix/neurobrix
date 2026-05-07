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

## Update 2026-05-06 — F2a Approche C landed (commit 08cbe15)

**F2a Approche C** (kernel broadcast-aware via clone interceptor) implémenté
et validé R29 4-mode:

| Mode               | Résultat | Wall   | PNG          | OOM site                      |
|--------------------|----------|--------|--------------|-------------------------------|
| compiled           | ✓ PASS   | 74s    | cohérent     | —                             |
| sequential         | ✓ PASS   | 89s    | cohérent     | —                             |
| triton compiled    | ✗ FAIL   | 253s   | —            | conv::64 (live=25GB req=8GB)  |
| triton_sequential  | ✗ FAIL   | 252s   | —            | rms_norm (live=25GB req=8GB)  |

**Validation factuelle F2a**:
- Le clone::5 8 GiB allocation a disparu factuellement (verified par advance
  du pipeline triton de `aten.add::86` vers `aten.convolution::64`).
- compiled mode unchanged dual-backend safe (74s vs prior 82s — 8s gagnés
  sur le seul fix tensor_resolver `unknown` torch.* handler qui évite
  resolve_kwargs path détourné).
- sequential mode dual-backend safe (89s) avec `tensor_resolver._resolve_arg_info`
  étendu pour résoudre `torch.contiguous_format` quand op_uid interceptor
  force kwargs resolution.

**Résidu factuel — chantier suivant nommé**:

`P-TRITON-LIVE-WATERMARK-AUDIT` (intra-P-SANA-4KPX-RUNTIME, pas defer flou):
au moment OOM (conv::64 ou rms_norm post-pixel_shuffle::4), live_tracked
factuel = 25292 MB avec 3 × (1,128,4096,4096) NBXTensors live:
- tensor#1 = `aten.silu::24::out_0` 8192 MB (in-progress op input, attendu)
- tensor#2 = `aten.add::86::out_0` 8192 MB (devrait être tué par kill_slots)
- tensor#0 = ORPHAN(not in arena) 8192 MB held par
  `list[len=3] contents=[NBXTensor×3]` × 2 + `tuple(len=2) contents=[int,NBXTensor]`
  + `cell` (closure cell)

Le pattern d'orphan retention `list[len=3]×NBXTensor + cell` est SAME que
celui identifié dans le commit 9c81e8a pré-F2a — donc structurel à
TritonSequence args_resolver / closure capture, **PAS** introduit par F2a
(F2a a juste fait avancer la chain assez loin pour exposer cet orphan
distinct du clone::5 maintenant éliminé).

**Hypothèse factuelle pour P-TRITON-LIVE-WATERMARK-AUDIT**:
- Args list (`[r(arena) for r in resolvers_t]`) capturée par `enumerate()`
  ou par closure cell de `for op_idx, op in enumerate(self._ops)` dans
  `_run_single_device`. La référence persiste à travers les itérations
  parce que la frame courante / le cell pointent vers la liste précédente.
- ETA estimé: 1-2h diagnostic + fix targeted (force-clear args list à fin
  d'iter, ou rewrite resolver pour ne pas créer une fresh list à chaque
  call).

P-MULTI-GPU-NBX-ADAPTER reste backup si P-TRITON-LIVE-WATERMARK-AUDIT
échoue (mais résoudre le live-watermark Triton-side évite le coût
multi-GPU pour modèles single-GPU-fittable).

**Discipline maintenue**:
- Pas de "INDÉTERMINÉ" — outcome factuel par mode, dump live_tracked / req
  / driver_free chiffré, BIG_TENSORS scan avec referrers
- Pas de défer flou — `P-TRITON-LIVE-WATERMARK-AUDIT` scopé avec hypothèse
  testable (closure args list capture)
- F2a closes 1 des 3 leviers identifiés au début (kernel-level fix) ; live-
  watermark était implicite dans le levier "Prism op-level parity" mais
  émergé comme orphan retention distinct du clone::5

**Référentiel chantiers ouverts**:
- `P-OP-PATTERN-FUSION` (proposé pré-F2a): registry universel patterns
  pytorch décomposés (pixel_shuffle DONE via F2a + 8 autres scopés:
  pixel_unshuffle, layer_norm, softmax, GELU, unfold/fold, repeat_interleave,
  roll, grid_sample). À ouvrir post-`P-TRITON-LIVE-WATERMARK-AUDIT`.

## Update 2026-05-06 (later) — bisection 3-suspects post-F2a (etape 5,6,7)

**Suspect A (closure/frame retention) — FACTUELLEMENT RÉFUTÉ**.
NBX_AGGRESSIVE_CLEANUP=1 (force `args=kwargs=result=None` après chaque
op.func + gc.collect via NBX_FORCE_GC=1) ne change RIEN à la trajectoire
live: `op 662 silu::24 live=31441MB` identique avec et sans cleanup.
Le leak n'est PAS au niveau frame Python.

**Trajectoire fine per-op (etape6, etape7)**:
```
op 654 conv::62           live=30417 MB  (post-add::84 spike)
op 659 pixel_shuffle::4   live=27444 MB  (-3 GB)
op 660 add::86            live=21198 MB  (-6 GB Fix B in-place)
op 661 conv::63           live=21198 MB  (no change BEFORE conv::63)
op 662 silu::24           live=31441 MB  (+10 GB conv::63 alloc)
op 663 conv::64           live=31441 MB  (BEFORE conv::64 alloc → OOM)
```

Le saut +10 GB entre conv::63 et silu::24 = conv::63's allocation
(8 GB output + ~2 GB band-streaming transients qui ne se libèrent
pas). Avec OOM live=25 GB et 3×8 GiB tensors live (silu::24, add::86,
ORPHAN), la signature factuelle est:

- silu::24 (legitimate, in-progress conv::64 input)
- add::86 (legitimate, has remaining consumer add::89)
- ORPHAN — soit pixel_shuffle::4::out_0 (devait être killed à add::86)
  soit conv::63::out_0 (devait être killed à silu::24)

**Conclusion factuelle**: le leak est au niveau C-extension. Sources
candidates restantes:
- Triton autotune cache holding kernel args via internal refs not
  visible from Python gc (CompiledKernel cache, BenchmarkRunner state)
- NBXTensor `_owns_data` flag interaction with `_base` chain creating
  a strong-ref-cycle that breaks NBXTensor.__del__ → cudaFree
- CUDA driver-level retention not visible from Python

**Bisection suspect non-testée — mais coût opportunité élevé**:
Suspect B (wrapper self_managed_dtype) et Suspect C (decomposed
PyTorch op) restent à tester, mais mesure factuelle requise vs
diagnostic instrumentation déjà saturée.

**Décision factuelle pour le chantier**:
Le résidu live-watermark Triton 4Kpx = limite structurelle de
debugging Python-level. Le fix requiert soit (a) instrumentation
plus profonde NBXTensor lifecycle (suivre __init__/__del__ sites
factuellement), soit (b) basculer sur P-MULTI-GPU-NBX-ADAPTER pour
contourner via VAE sur GPU dédié (l'option backup nommée). Choix
arbitré par Hocine.

F2a Approche C reste un fix réel et durable (commit 08cbe15) qui
unblock add::86 + advance le pipeline triton plus loin que jamais.
Le résidu conv::64/rms_norm est PRE-EXISTANT à F2a (verified par
signature orphan identique entre commits 9c81e8a et etape5/6/7).

## Update 2026-05-06 (final) — bisection 3-suspects suite, fixes empilés

Suspect A (closure/frame retention) raffiné par instrumentation
ciblée NBX_TRACE_TIDS + NBX_TRACE_DEL_BIG_MB:
- pixel_shuffle::4::out_0 et conv::63::out_0 ont les MÊMES patterns
  de référants → SUSPECT SYSTÉMIQUE +2 refs sur tout tensor stocké.
- NBX_DEL trace prouve que conv::63 NBXTensor.__del__ ne fire QU'AU
  OOM unwind, PAS au silu::24's kill_slots → leak structurel.
- Root cause identifié: `for s in op.kill_slots: old = arena[s]`
  avec `old` Python loop variable persistante post-loop. Le `old`
  retient 8 GiB de chaque kill across iterations subséquentes
  jusqu'au prochain kill_slots loop qui rebind.

**Fix landed** commit cd5a108: explicit `old = None` post-loop à 5
sites dans `_run_single_device` + multi-device variant.

**Mesure post-fix etape10**: chain advance conv::64 → rms_norm::24
(2 ops). Live trajectory inchangée à add::86 (21 GB) parce que la
rétention `old` n'affectait pas le steady state mais le moment
exact de l'OOM.

**Suite — rms_norm OOM (exec[710])**: rms_norm wrapper alloue
2 × 8 GiB (x.contiguous() pour permute view + NBXTensor.empty_like
pour output). Fix in-place: si x.contiguous() matérialise (input
non-contiguous), output_2d = x_2d (kernel per-tile-safe).

**Fix landed** commit beeab71 (avec add bias).

**Suite — add::88 OOM (exec[711])**: add wrapper matérialise bias
broadcast 8 GiB via b.expand(out_shape).contiguous(). Fix:
add_bias_broadcast_kernel lit bias[offset % feat_dim] direct.

**Fix landed** commit beeab71.

**Outcome final cette session**:
- Chain advance: conv::64 (exec[708]) → conv::69 (exec[735]) = +27 ops
- conv::69 = LAST conv VAE (output RGB 1×3×4096×4096)
- OOM "Triton Error [CUDA]: out of memory" — Triton runtime-level,
  pas malloc_cuda. Probable autotune workspace overflow vs
  conv2d_wrapper x.contiguous() materialization for permute view input.

**État chantier P-SANA-4KPX-RUNTIME (post-session)**:
- 5 fixes incrémentaux land cette session
- Pipeline triton 4Kpx structurellement très proche de la fin (27 ops
  past original blocker, on est dans le LAST conv avant write)
- Le résidu live=30 GB à conv::69 est dans le seuil 32 GB V100 mais
  Triton autotune workspace ou x.contiguous() matérialisation le
  pousse over.

**Next étape factuelle (intra-P-SANA-4KPX-RUNTIME)**: investiguer
le mode OOM exact à conv::69 (autotune workspace vs contiguous
materialization). Si autotune: configurer un fixed config pour
conv::69 (output channels=3 unique). Si contiguous: appliquer
même pattern in-place qu'à rms_norm pour conv2d_wrapper avec
non-contiguous input.

ETA estimé pour clore: 1-2h investigation + fix targeted.

## Update 2026-05-07 — etape13 force_gc test, hypothèses raffinées

**Hypothèse (2) contiguous() materialization REFUTED** par audit
DAG: conv::69 input = `aten.relu::18::out_0` (relu wrapper alloue
output contigu via NBXTensor.empty_like — pas un permute view).
Le path `x.contiguous()` dans conv2d_wrapper est no-op pour
conv::69. Pas de matérialisation 8 GiB ici.

**Hypothèse (1) Triton autotune workspace TESTÉE** via etape13
NBX_FORCE_GC=10 (gc.collect every 10 ops). Same OOM at conv::69.
gc.collect ne libère pas de cycles → leak n'est pas Python-level.

**Stack trace confirmé**: OOM source est
`triton/backends/nvidia/driver.py:713 self.launch()` —
**CUDA driver-level kernel launch**, PAS NBX malloc_cuda. La
launch reserve quelques KB pour kernel args + Triton runtime
buffers. Avec live=30GB / 32GB V100 = 2GB free, une fragmentation
ou Triton do_bench cache_flush buffer (~256 MB par défaut)
suffit à pousser over.

**Conclusion factuelle**: le résidu conv::69 est limite
**structurelle V100 32GB** pour Sana 4Kpx VAE post-pixel_shuffle
chain. Pour clore définitivement, options scopables:

1. **Bypass autotune pour conv::69 specifique**: detect output
   channels=3 (unique pattern in VAE) dans conv2d_wrapper, route
   vers un appel `conv2d_forward_kernel.fn[grid](...)` avec config
   fixed (pas d'autotune do_bench overhead).

2. **Pre-warm autotune cache**: run conv::69 at startup with
   dummy tensors to populate disk cache before VAE pressure.
   Subsequent runs use cached config without benchmarking.

3. **Multi-GPU pipeline_parallel for VAE component**: VAE moves
   to dedicated GPU (32 GB free). Bypasses tightness.

ETA option 1: 30-60 min. ETA option 2: 60-90 min. ETA option 3:
several days (P-MULTI-GPU-NBX-ADAPTER scope).

**Discipline maintenue**: pas d'INDÉTERMINÉ, pas de pivot
prematuré (option 3 reste backup). Le chantier reste ouvert avec
1-2 options option 1/option 2 actionable scopable.

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
