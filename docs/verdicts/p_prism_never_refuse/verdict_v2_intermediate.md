# P-PRISM-NEVER-REFUSE v2 — verdict intermédiaire après B.3

R29 inspectable artefact. Synthèse cumulative au commit `e8c7efa` (B.3
Branche A `--compiled` mode CPU execution shipped).

═══════════════════════════════════════════════════════════════════
ACQUIS — Doctrine R35 implémentée pour `--compiled` mode
═══════════════════════════════════════════════════════════════════

**Cascade Prism complétée par `cpu_execution`** : tous les profils
(GPU et CPU-only) terminent par `cpu_execution`. Doctrine R35
implémentée structurellement.

| profil | cascade actuelle |
|---|---|
| Single-GPU | single_gpu → single_gpu_lifecycle → lazy_sequential → zero3 → **cpu_execution** |
| Multi-GPU | 9 stratégies GPU + **cpu_execution** |
| Pure-CPU (`cpu-only-x86.yml`) | **cpu_execution** (seul élément) |

**Validation concrète (R29 artefacts)** :

| modèle | mode | hardware | wall-time | output | artefact |
|---|---|---|---|---|---|
| TinyLlama-1.1B | `--compiled` | 40-core Xeon Gold 6230 + 256 GiB RAM (CPU-only) | 6.03 s / 39 tokens | "Fresh, golden, on frost-free trees, Apples grow atop a serene vine, Bringing apples to the mall, they say." | `tinyllama_cpu_compiled_haiku.txt` |
| Sana 1024 (diffusion) | `--compiled` | CPU-only | 43.69 s / 4 steps | 🍎 red apple coherent | `sana1024_cpu_compiled_apple.png` |

**Anti-régression GPU** :

| modèle | mode | hardware | wall-time | verdict |
|---|---|---|---|---|
| Sana 1024 | `--triton-sequential` | v100-32g | 73.67 s | 🍎 PASS (vs POINT 9 baseline 74.86 s) |
| Sana 1024 | `--compiled` | v100-32g | 15.29 s | 🍎 PASS |

═══════════════════════════════════════════════════════════════════
GAPS DOCUMENTÉS — sous-chantiers restants
═══════════════════════════════════════════════════════════════════

### B.3 Branche A `--sequential` (PyTorch legacy dispatcher CPU)

`--sequential` mode sur CPU échoue avec `IndexError: index 18 is out
of bounds for dimension 0 with size 1` dans `aten::index` au début
du autoregressive decode loop (`autoregressive.py:177` →
`_execute_native_op:2906`).

C'est un bug **pre-existing dans le legacy NativeATenDispatcher**
quand `device='cpu'` — non causé par mes changements (le code path
sequential n'a pas été modifié). Le mode `--compiled` (CompiledSequence
hot loop) couvre le même use case avec meilleures performances.

**À investiguer dans un sub-chantier dédié** : trace pourquoi
`aten::index` resolves to a size-1 tensor on CPU device. Probable
cause : un asset tensor (position embeddings ?) reste sur cuda:0 alors
qu'il devrait être sur cpu.

### B.3 Branche B (`--triton` et `--triton-sequential` CPU)

L'existant `cli/commands/run.py:365-368` détecte "no GPU" et set
`TRITON_CPU_BACKEND=1`. Mais la branche `triton/` runtime appelle
`DeviceAllocator.malloc_cuda` sans condition → fail avec
"GPU malloc failed (error 100) for 64 bytes" (CUDA_ERROR_NO_DEVICE).

**Intégration Triton-CPU upstream nécessaire** :
- `meta-pytorch/triton-cpu` (fork Meta) : compile les mêmes
  `@triton.jit` kernels pour CPU via LLVM MLIR + OpenMP backend.
- `OMP_NUM_THREADS=profile.cpu.cores` doit être set AVANT import
  triton-cpu (ordre OpenMP).
- Pour ops non couvertes par Triton-CPU upstream : fallback ATen CPU
  avec marker explicite + entrée dans `triton_cpu_coverage_gaps.md`.

**Estimation taille** : 200-400 lignes selon coverage Triton-CPU
upstream (variable). Sub-chantier dédié `P-TRITON-CPU-INTEGRATION`.

### B.1 — Tiling étendu pour Sana 4Kpx 16 GiB

Audit liveness POINT 10 v2 ÉTAPE A audit_p_prism_never_refuse_v2.md
+ commit c56be9b. Verdict B.1.a : **ZERO STALE TID** au moment du
problème (drain pre-hook ne peut rien faire). Bascule B.1.b évaluée :
"refactor contrat fusion band-streamed" pour la chaîne résiduelle
DC-AE (3× 4 GiB tensors simultanément alive) nécessite 300-500+
lignes architectural — condition #2 AT B.1 LEVEL.

**Workaround dispo** : l'utilisateur peut forcer `cpu_execution` sur
le profil `v100-16g` via `NBX_FORCE_STRATEGY=cpu_execution`. Le PNG
sortira (très lent — heures probables pour Sana 4Kpx CPU). Le
mécanisme existe doctrinalement. **Auto-cascade fallback runtime-OOM
→ cpu_execution non implémentée** (sub-chantier futur).

### B.4 — CPU offload partiel (hybrid VAE-on-CPU + reste-on-GPU)

Non-implémentée. La base existe (cpu_execution path opérationnel).
L'extension à faire : nouvelle stratégie `cpu_offload_component`
qui détecte le composant qui ne tient pas sur GPU et le place sur
CPU pendant que les autres restent sur GPU. Cross-device transfer
entre composants déjà géré par variable_resolver.

**Estimation taille** : 100-200 lignes. Faisable dans un
sous-chantier suivant.

### B.2 — Intra-component cross-device split

Non-implémentée. NBXTensor cross-device + Triton runtime device-aware
dispatch nécessaires. Cible 2× V100 16 GiB.

**Estimation taille** : 400-800 lignes. Au-delà du seuil mandate
condition #2 si pris seul.

═══════════════════════════════════════════════════════════════════
CIBLE BINAIRE — status
═══════════════════════════════════════════════════════════════════

| config | mode | status |
|---|---|---|
| 1× V100 32 GiB | les 4 modes | ✓ ACQUIS POINTS 1-9 |
| 1× V100 16 GiB | `--compiled` | ✓ via `NBX_FORCE_STRATEGY=cpu_execution` (manuel, slow). Auto-cascade non implémentée. |
| 1× V100 16 GiB | `--sequential` | ✗ blocked by B.3 Branche A `--sequential` IndexError |
| 1× V100 16 GiB | `--triton` / `--triton-sequential` | ✗ blocked by B.3 Branche B Triton-CPU non intégré |
| 2× V100 16 GiB | tous modes | ✗ blocked by B.2 (intra-component split) |
| CPU pur | `--compiled` | ✓ **VALIDÉ** TinyLlama + Sana 1024 |
| CPU pur | `--sequential` | ✗ même blocker que 1× V100 16 GiB sequential |
| CPU pur | `--triton` / `--triton-sequential` | ✗ même blocker Triton-CPU |

**Progrès** : la doctrine R35 est structurellement implémentée. Le
chemin `cpu_execution` fonctionne end-to-end pour `--compiled` mode
sur LLM + diffusion. Le sub-chantier Triton-CPU intégration est le
levier majeur pour compléter les autres modes.

═══════════════════════════════════════════════════════════════════
SUBS-CHANTIERS BACKLOG OUVERTS
═══════════════════════════════════════════════════════════════════

1. **`P-TRITON-CPU-INTEGRATION`** (priorité haute, débloque les
   modes `--triton` et `--triton-sequential` sur CPU) — intégration
   `meta-pytorch/triton-cpu` upstream dans `src/neurobrix/triton/`,
   wiring `OMP_NUM_THREADS` depuis profile.cpu.cores, fallback ATen
   CPU pour ops non couvertes via `triton_cpu_coverage_gaps.md`.

2. **`P-CPU-OFFLOAD-COMPONENT`** (B.4) — extension de l'existant
   pour placer un composant entier sur CPU pendant que d'autres
   tournent sur GPU. Auto-cascade détecte le composant overflow.

3. **`P-NATIVE-SEQUENTIAL-CPU-DEBUG`** (B.3 sequential gap) —
   investigation IndexError sur `aten::index` dans le legacy
   dispatcher quand `device='cpu'`. Probable un tensor reste sur
   cuda:0 alors qu'il devrait suivre la device override.

4. **`P-RUNTIME-OOM-REPLAN`** — quand le runtime catch une OOM,
   re-trigger Prism cascade avec budget mémoire réduit. Permettrait
   l'auto-fallback Sana 4Kpx 16 GiB single_gpu → cpu_execution.

5. **`P-PRISM-MULTI-GPU-INTRA-COMPONENT-SPLIT`** (B.2, déjà nommé)
   — split intra-component cross-device pour les 2× V100 16 GiB.

═══════════════════════════════════════════════════════════════════
DOCTRINE STATUS
═══════════════════════════════════════════════════════════════════

- **R34 model-agnostic** : audit clean (aucune violation active).
  CPUExecutionStrategy n'introduit aucun hardcode model-specific.
- **R35 Prism never refuses** : structurellement implémentée pour
  `--compiled` mode (CPU + GPU). Pour modes triton, l'intégration
  Triton-CPU restera le chemin de fermeture. Pour mode sequential,
  bug pre-existing à corriger.
- **R33 zero torch dans triton/** : préservée (no torch import added).
- **R30 mode universality** : respectée pour les modes existants.
  Les modes triton sur CPU sont "not yet supported" (clean failure
  vs silent garbage).
