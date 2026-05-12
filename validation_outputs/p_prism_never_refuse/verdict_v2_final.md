# P-PRISM-NEVER-REFUSE v2 — verdict final session

R29 inspectable artefact. Synthèse au commit `de5fb9e`. Phase B
substantielle livrée; runtime gaps documentés honnêtement.

═══════════════════════════════════════════════════════════════════
ACQUIS DURABLES (commit de5fb9e + commits précédents de la session)
═══════════════════════════════════════════════════════════════════

### Phase A — audit factuel complet

`audit_p_prism_never_refuse_v2.md` (commit `c56be9b`). Sourced
file:line. Architecture Prism saine; **R34 model-agnostic 100%
respectée** dans le scope audité (0 violation active dans code).

### B.1.a — liveness audit env var

`NBX_LIVENESS_AUDIT_AT_OP_UID=<op_uid>` dans `_run_triton_sequential`.
Diagnostic op-level générique (R34 conforme). Audit Sana 4Kpx 16 GiB :
**ZERO STALE TID** au moment du problème conv::62 — confirme que la
mémoire est genuinely needed par le graph DAG, pas une fuite. Liveness
analysis 100% correcte.

### B.1.b — bascule documentée + condition #2 at sub-chantier level

Audit factuel : la chaîne résiduelle DC-AE de Sana 4Kpx au-delà de
conv::62 maintient 3× 4 GiB tensors simultanément live (silu::24 op
707 : `add::86::out_0` + `conv::63::out_0` + `silu::24::out_0`).
Refactor "contrat fusion band-streamed" pour la chaîne entière =
300-500+ lignes architectural. **Condition #2 AT B.1 LEVEL** —
sub-chantier `P-DC-AE-RESIDUAL-CHAIN-TILING` ouvert au backlog.

### B.3 — CPU execution strategy (commit `e8c7efa` + `de5fb9e`)

Nouvelle stratégie `cpu_execution` cascadée en dernier recours dans
toutes les configs. Doctrine R35 implémentée structurellement.

**Validations end-to-end** :

| modèle | mode | hardware | wall | output |
|---|---|---|---|---|
| TinyLlama-1.1B | `--compiled` | cpu-only-x86 (40-core Xeon Gold 6230, 256 GiB) | 6.03 s / 39 tokens | 🍎 **coherent haiku about apples** |
| Sana 1024 | `--compiled` | cpu-only-x86 | 43.69 s / 4 steps | 🍎 **red apple PNG cohérente** |

Artefacts : `tinyllama_cpu_compiled_haiku.txt`,
`sana1024_cpu_compiled_apple.png`.

### B.4 — Hybrid CPU+GPU placement (commit `de5fb9e`)

`_place_component` Strategy 4 + driver overhead reserve (3 GiB) =
plan automatique correct pour Sana 4Kpx sur v100-16g :
```
Strategy: lazy_sequential
  vae          → cpu
  text_encoder → cuda:0
  transformer  → cuda:0
```
Mécanisme structurellement opérationnel; auto-cascade fonctionne; le
ZERO_FALLBACK historique sur Sana 4Kpx 16 GiB est éliminé au niveau
planning.

═══════════════════════════════════════════════════════════════════
GAP IDENTIFIÉ — runtime cross-device exécution
═══════════════════════════════════════════════════════════════════

Sana 4Kpx 16 GiB hybrid end-to-end : Prism plan accepté (voie B.4
correct), runtime démarre, mais **stuck à 99% CPU sur 1 thread après
50+ min** sans output ni progression GPU/RAM. Diagnostic :
- Engine COMPILED a démarré
- 96 threads alloués mais 95 dormants, 1 (main thread) tourne en pic
- VmRSS = 10.4 GiB (weights chargés)
- GPU 0 : 631 MB used (essentially empty)
- CPU memory : 12 GiB

Hypothèses du blocage (à investiguer dans un sub-chantier) :
- **Inter-component data transfer GPU↔CPU mal géré** : quand variable_resolver
  passe une activation de transformer (cuda:0) vers VAE (cpu) ou vice-versa,
  un `.to('cpu')` synchrone bloque indéfiniment ? Ou pire, infinite loop
  de transfert.
- **CompiledSequence hot loop pas device-aware** : la séquence compilée
  pour le composant transformer assume tous les inputs sur cuda:0 mais
  reçoit un input cpu pour le step suivant ?
- **MKL/oneDNN pas engagé** pour le CPU compute (seul 1 thread actif).
  `apply_cpu_config` peut-être pas appelé quand un mix de devices apparaît.

Le mécanisme est en place mais le runtime cross-device need debugging.
Sub-chantier `P-RUNTIME-HYBRID-DEVICE-DISPATCH` à ouvrir.

═══════════════════════════════════════════════════════════════════
ANTI-RÉGRESSION POST-CHANGES
═══════════════════════════════════════════════════════════════════

| modèle | mode | hardware | wall | verdict |
|---|---|---|---|---|
| Sana 1024 | `--triton-sequential` | v100-32g | 77.66 s | 🍎 PASS (vs 74.86 baseline POINT 9) |
| Sana 1024 | `--compiled` | v100-32g | 15.29 s | 🍎 PASS |
| TinyLlama | `--triton-sequential` | v100-32g | 16.15 s | ✓ poème cohérent |
| TinyLlama | `--compiled` | cpu-only-x86 | 6.03 s | ✓ haiku cohérent |
| Sana 1024 | `--compiled` | cpu-only-x86 | 43.69 s | 🍎 PASS |

**Aucune régression** sur les chemins GPU pré-existants. Les changes
de cette session touchent uniquement les stratégies cascadées en
DERNIER recours (cpu_execution, lazy_sequential Strategy 4). Le
chemin single_gpu prend une réserve overhead 3 GiB qui pourrait
borderline impacter les modèles très tight sur 32 GiB (pas vu en
matrice anti-régression).

═══════════════════════════════════════════════════════════════════
CIBLE BINAIRE — status par cellule
═══════════════════════════════════════════════════════════════════

|  | 1× V100 32G | 1× V100 16G | 2× V100 16G | CPU pur |
|---|---|---|---|---|
| compiled | ✓ acquis | ⚠️ plan OK, runtime hybrid stuck (gap doc.) | ✗ B.2 non implémenté | ✓ **validé** (Sana 1024 + TinyLlama) |
| sequential | ✓ acquis | ⚠️ même gap + IndexError preexisting | ✗ B.2 + IndexError | ⚠️ pre-existing IndexError |
| triton | ✓ acquis | ✗ Triton CPU non intégré | ✗ B.2 + Triton CPU | ✗ Triton CPU non intégré |
| triton_sequential | ✓ acquis | ✗ Triton CPU non intégré | ✗ B.2 + Triton CPU | ✗ Triton CPU non intégré |

**Progrès tangible** :
- Doctrine R35 structurellement implémentée (cascade complète vers
  CPU)
- 2 modes / 1 config (CPU pur `--compiled`) **complètement
  validés** end-to-end (TinyLlama + Sana 1024)
- 1 cible binaire (1× V100 16 GiB compiled) à **un debug runtime
  près** (plan automatique correct, runtime cross-device bloqué)

═══════════════════════════════════════════════════════════════════
SUBS-CHANTIERS BACKLOG OUVERTS
═══════════════════════════════════════════════════════════════════

Priorité haute (débloque cibles binaires) :

1. **`P-RUNTIME-HYBRID-DEVICE-DISPATCH`** (NOUVEAU) — debug du
   runtime cross-device (transformer GPU → VAE CPU). Probablement
   variable_resolver data transfer issue OU CompiledSequence
   device-aware refresh. ~100-300 lignes selon root cause.

2. **`P-TRITON-CPU-INTEGRATION`** — meta-pytorch/triton-cpu
   upstream intégré dans `src/neurobrix/triton/`. Débloque modes
   triton/triton_sequential sur CPU. Estimation 200-400 lignes.

3. **`P-NATIVE-SEQUENTIAL-CPU-DEBUG`** — IndexError sur
   `aten::index` dans le legacy NativeATenDispatcher quand
   `device='cpu'`. Probable tensor stays on cuda:0 alors qu'il
   devrait suivre device override.

Priorité moyenne :

4. **`P-DC-AE-RESIDUAL-CHAIN-TILING`** (= ancien B.1.b) — refactor
   contrat fusion band-streamed pour la chaîne résiduelle DC-AE.
   Permettrait Sana 4Kpx 16 GiB en mode triton sur GPU pur
   (alternative à la voie hybrid CPU+GPU).

5. **`P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT`** (B.2, déjà nommé
   POINT 9-10) — split intra-component cross-device pour 2× V100
   16 GiB. Estimation 400-800 lignes.

Priorité basse :

6. **`P-RUNTIME-OOM-REPLAN`** — auto re-plan when runtime hits OOM
   (would make `single_gpu` failures automatically cascade to
   `lazy_sequential` or `cpu_execution`).

═══════════════════════════════════════════════════════════════════
DOCTRINE STATUS
═══════════════════════════════════════════════════════════════════

- **R34 model-agnostic** : audit clean. Toutes les nouvelles
  stratégies (cpu_execution, _place_component Strategy 4) sont
  data-driven sur profile.cpu / profile.devices.
- **R35 Prism never refuses** : implémentée structurellement.
  Cascade single-GPU profile : `single_gpu → single_gpu_lifecycle →
  lazy_sequential (avec hybrid CPU placement) → zero3 →
  cpu_execution`. Cascade multi-GPU : 9 stratégies + cpu_execution.
  Cascade cpu-only profile : `cpu_execution` direct.
- **R33 zero torch dans `triton/`** : préservée (no torch import
  added).
- **R30 mode universality** : respectée pour les modes pré-existants
  + le nouveau cpu_execution path. Modes `--triton` / `--triton-
  sequential` sur CPU sont "not yet supported" (clean failure plutôt
  que silent garbage).

═══════════════════════════════════════════════════════════════════
COMMITS LANDÉS DURANT LA SESSION
═══════════════════════════════════════════════════════════════════

- `c56be9b` audit Phase A + B.1.a liveness diagnostic env var
- `e8c7efa` feat cpu_execution strategy (Doctrine R35 last-resort)
- `b696e53` docs intermediate verdict + R29 artefacts
- `de5fb9e` feat cpu_offload per-component cascade (B.4)

═══════════════════════════════════════════════════════════════════
RECOMMANDATION REMONTÉE
═══════════════════════════════════════════════════════════════════

Condition de sortie #2 du mandate (blocker architectural >300 lignes
hors scope) appliquée AU SUB-CHANTIER LEVEL :
- B.1.b residual chain tiling : >300 lignes (chantier dédié)
- P-RUNTIME-HYBRID-DEVICE-DISPATCH : debug profond probable
- P-TRITON-CPU-INTEGRATION : intégration upstream large
- B.2 intra-component split : 400-800 lignes

La session a livré un progrès structurel substantiel (Doctrine R35
implémentée + 2/16 cellules de la matrice complète validées) avec
les bases pour 4-5 sub-chantiers identifiés. Remontée pour
arbitrage Hocine sur la suite des sub-chantiers prioritaires.
