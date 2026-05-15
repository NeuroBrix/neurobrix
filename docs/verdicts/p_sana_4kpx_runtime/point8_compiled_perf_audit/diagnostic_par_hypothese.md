# POINT 8 — ÉTAPE B : diagnostic par hypothèse

Lecture code obligatoire (pas d'extrapolation depuis le nom).
Citations file:line des chemins exacts examinés.

## H1 — CompiledSequence hot-loop n'est plus emprunté (fallback silencieux)

**Verdict factuel : INVALIDÉE.**

Recherche `grep -nE "TritonSequentialDispatcher|fallback|sequential_dispatcher" src/neurobrix/triton/sequence.py` → **0 hits**.
`TritonSequence` (le hot-loop compiled) n'a aucune référence à
`TritonSequentialDispatcher` ni de chemin "fall back to sequential".

Le hot loop à `src/neurobrix/triton/sequence.py:2580–2625` est :

```python
for op_idx, op in enumerate(self._ops):
    args = op.args_resolver(arena)
    kwargs = op.kwargs_resolver(arena)
    # NOP propagation (deactivated MoE paths)
    if args and args[0] is None:
        ...
        continue
    else:
        ...
        result = op.func(*args, **kwargs)
```

`self._ops` est une liste pré-construite de `CompiledOp` au chargement.
Aucun dispatch dict-lookup, aucune string-parse, aucun kernel-registry
lookup par op dans la boucle. Pas de chemin fallback.

Corroboration mesures : si H1 était vraie (compiled ≡ sequential
silencieusement), le gap serait ≈0 sur les **deux** shapes. Au 1024
le gap −4.8 % est mesuré → compiled fait bien quelque chose de
différent et plus rapide → fallback exclu.

## H2 — CompiledOpResolver ne fait plus de fusion/pré-résolution

**Verdict factuel : INVALIDÉE structurellement (pas une régression).**

Lecture `src/neurobrix/core/runtime/graph/compiled_ops.py` et
`compiled_sequence.py` : le resolver pré-résout les wrappers à la
création de la sequence (closures sur les slots de l'arena, dispatch
par op-type **une seule fois** au compile-time). Aucune logique de
**fusion d'ops adjacentes** dans le path actuel (pas de mul+add → fma,
pas de cat+split fusion, etc.).

Ce n'est pas une régression : la fusion d'ops nécessite un compilateur
graph-level (style XLA / Triton compiler avec pattern matching) qui
n'est pas dans le périmètre actuel. Le hot-loop fait son travail
(pre-resolved closures, no per-op dispatch). La perf gap résiduel
viendrait de la fusion, qui sort du périmètre POINT 8 et du périmètre
ce-chantier en général (mandate interdit "Touches aux kernels Triton").

## H3 — Les fixes POINTS 1-6 ont inséré des matérialisations dans compiled

**Verdict factuel : INVALIDÉE.**

Le contiguous-guard POINT 6 H2 (commit `a862fe0`) ajoute des appels
`.contiguous()` dans `_fused_upsample_conv2d_nbx` et
`_tiled_conv2d_spatial_nbx`. `NBXTensor.contiguous()` à
`src/neurobrix/kernels/nbx_tensor.py:1644` :

```python
def contiguous(self) -> 'NBXTensor':
    if self.is_contiguous():
        return self       # early return ZERO COST
    ...
```

Early return zéro-coût si le tensor est déjà contigu. Les bandes
intermédiaires post-slice sont matérialisées (coût ~bande × dtype-size,
nécessaire et identique au path torch qui appelle aussi `.contiguous()`).
Pas d'amplification spécifique au mode compiled vs sequential — les
deux chemins traversent le même wrapper.

## H4 — Arena allocator en mode défensif sync-everywhere

**Verdict factuel : INVALIDÉE.**

`grep -nB 2 "DeviceAllocator.sync_device()" src/neurobrix/triton/sequence.py`
révèle 6 call sites, **tous gated** :

| line | gate |
|---|---|
| 2610 | `if _PROF:` (profiling only) |
| 2930 | `if _PROF:` (profiling only) |
| 3083 | periodic drain (deferred-free queue threshold ≥ `_drain_bytes_limit`) |
| 3089 | end-of-run drain (`if _deferred:`) |
| 3263 | periodic drain multi-device |
| 3269 | end-of-run drain multi-device |

Aucun sync_device par-op sur le happy path. Les drains périodiques
sont rares et amortissent les cudaFree. Le hot loop est sync-free.

Corroboration mesures : si H4 était vraie, le GPU util oscillerait
(periodic stalls par sync). Le profile mesuré au 4Kpx montre **67.5 %
du temps actif >90 % util sustained** — pas de creux par-op. Invalidée
par les deux côtés (code + mesures).

## H5 — Hot-loop de CompiledSequence a dégénéré

**Verdict factuel : INVALIDÉE.**

Même citation que H1 : la boucle à `src/neurobrix/triton/sequence.py:2580–2625`
est un `for op in self._ops` serré avec closures pré-résolues. Pas
de dispatch par op-type, pas de string-parse, pas de kernel-registry
lookup par itération.

Une petite source d'overhead identifiable côté Python : ligne 2615
`_per = _os_per.environ.get("NBX_LIVE_DUMP_EVERY", "0")` est appelée
par-op (env var lookup, ~1 µs). Avec ~5000 ops par forward × 12
steps = 60 000 itérations, l'overhead total = ~60 ms par run complet
— **négligeable vs 511 s mesurés (0.012 %)**. Pas un fix utile.

## Synthèse

| hypothèse | verdict | preuve dominante |
|---|---|---|
| H1 silent fallback | **INVALIDÉE** | code grep + measured gap @ 1024 |
| H2 no fusion | **INVALIDÉE** (hors scope) | code lecture (fusion = kernel-level work) |
| H3 contiguous-guard cost | **INVALIDÉE** | early-return zéro-coût @ nbx_tensor.py:1645 |
| H4 defensive sync | **INVALIDÉE** | tous syncs gated, GPU util sustained ≥90 % @ 4Kpx |
| H5 degenerate hot-loop | **INVALIDÉE** | hot-loop structurellement optimal @ sequence.py:2580 |

**Aucune hypothèse de régression validée.** Le compiled hot-loop est
déjà structurellement optimal pour ce que Python peut faire ; le gap
résiduel vs sequential reflète la (faible) marge dispatch disponible
au-delà du temps GPU.
