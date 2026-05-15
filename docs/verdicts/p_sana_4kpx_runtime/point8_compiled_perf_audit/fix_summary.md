# POINT 8 — ÉTAPE C : fix summary

## Aucun fix appliqué

Toutes les hypothèses H1–H5 sont invalidées par lecture code + mesures
profile-driven (voir `diagnostic_par_hypothese.md`). Le compiled hot-loop
est structurellement optimal pour ce que Python peut faire à dispatch-time.

Per mandate ÉTAPE C : *"Si le gap structurel ne permet pas d'atteindre
1.5× (par exemple si le VAE 4Kpx est massivement GPU-bound et l'overhead
Python sequential est déjà petit), tu rapportes factuellement le gap
incompressible plutôt que de forcer un fix qui n'apporte rien."*

Le profile mesure exactement ce cas : Sana 4Kpx GPU util sustained 83.5 %
avec 67.5 % du temps actif >90 % util → compute-bound. Forcer un fix
sur la couche Python n'apporterait pas le speedup ≥1.5× ciblé, et
risquerait d'introduire de la complexité sans bénéfice mesurable.

**Aucune modification de code n'est apportée. Closure factuelle.**

## Pistes d'optimisation hors scope POINT 8 — pour backlog

Pour atteindre un speedup significatif compiled vs sequential, il
faudrait travailler côté kernel (interdit par le périmètre POINT 8) :

1. **Fusion d'ops adjacentes** au compile-time (e.g. mul → add → silu
   → conv en un seul kernel). Pattern matching graph-level + génération
   de kernel Triton fusé. Coût d'implémentation : >200 lignes, ~chantier
   dédié (probable nom : `P-TRITON-FUSED-KERNELS`).
2. **CUDA Graphs** : capturer la séquence d'op pour la rejouer sans
   passer par Python à chaque inférence step. Speedup typique 1.5-2×
   sur shapes où le dispatch domine. Volta limité (CUDA Graphs ≥
   sm_70, plein support depuis sm_75). Chantier dédié.
3. **Autotune ON sur kernels matmul/conv** (déjà authorized par
   CLAUDE.md mais désactivé dans cette mesure pour matcher POINT 7).
   Speedup mesuré ~12 % sur Volta selon CLAUDE.md autotune policy.
   À mesurer comme baseline production séparée.

Aucune de ces pistes n'est dans le périmètre POINT 8. Reportées au
backlog.
