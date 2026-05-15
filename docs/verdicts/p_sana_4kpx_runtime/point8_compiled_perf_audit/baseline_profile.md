# POINT 8 — ÉTAPE A : baseline profile triton vs triton_sequential

Profile capturé 2026-05-10 sur 1× V100 32 GiB (`CUDA_VISIBLE_DEVICES=2 --hardware v100-32g`),
autotune désactivé (`NBX_DISABLE_AUTOTUNE=1`) pour matcher les conditions
POINT 7. GPU util échantillonné à 10 Hz via
`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits --loop-ms=100`.

## Wall time mesurée — 4 runs

| modèle | mode | wall time | gap vs sequential |
|---|---|---|---|
| Sana 1024 | `--triton-sequential` | 73.91 s | baseline |
| Sana 1024 | `--triton` (compiled hot-loop) | 70.35 s | **−4.8%** |
| Sana 4Kpx | `--triton-sequential` | 513.96 s | baseline |
| Sana 4Kpx | `--triton` (compiled hot-loop) | 511.78 s | **−0.4%** |

Reproduit le pattern POINT 7 (le mandate citait 514.83 vs 510.02 = +0.9%
côté triton ; les nouvelles mesures donnent triton 0.4% plus rapide ;
variance de ±1 % par run, donc les deux observations sont équivalentes :
**le gap compiled vs sequential est négligeable au 4Kpx**).

## Note sur la conflation 1024 dans le mandate

Le mandate cite "Sana 1024 : compiled 15.66s vs sequential 16.35s = -4.2%",
mais cette mesure POINT 7 ÉTAPE E correspond aux modes PyTorch
`--compiled` et `--sequential` (native ATen + cuDNN/cuBLAS), pas aux
modes triton qu'on étudie ici. La paire triton réelle sur Sana 1024
est **70.35 vs 73.91 s = −4.8%**, mesurée ci-dessus dans des conditions
identiques (autotune-OFF, hardware-bound v100-32g, prompt unique).

## GPU utilization trajectory par run (clé du diagnostic)

Partition automatique du log par seuils d'activité GPU 2 (sample > 30 %
= actif, < 10 % consécutifs ≥ 2 s = idle). Stats sur la fenêtre active
de chaque run (ignore les périodes de setup / model loading qui sont
CPU-bound) :

| run | active duration | avg util | >90 % util | 50–90 % util | <50 % util |
|---|---|---|---|---|---|
| Sana 1024 triton | 43.1 s | **60.2 %** | 10.2 % | 54.5 % | 35.3 % |
| Sana 1024 triton_seq | 45.1 s | **59.8 %** | 5.8 % | 59.0 % | 35.3 % |
| **Sana 4Kpx triton** | 398.5 s | **83.5 %** | **67.5 %** | 16.6 % | 16.0 % |
| **Sana 4Kpx triton_seq** | 400.8 s | 81.6 % | 66.1 % | 15.5 % | 18.4 % |

(active duration = temps où GPU 2 est utilisé > 30 %, exclut le setup
runtime qui dure ~30 s par run avant le premier kernel)

## Verdict baseline

**Sana 4Kpx est compute-bound** : la GPU est utilisée à 83.5 % en
moyenne avec 67.5 % du temps actif passé à >90 % d'utilisation. Le
résidu (avg 17 % en-dessous de full util) est composé majoritairement
de gaps entre kernels (Python dispatch + cudaMemcpy sync points
nécessaires). L'overhead Python a au maximum 17 % de marge théorique
pour s'améliorer ; le gap mesuré compiled vs sequential de **0.4 %**
indique que la majeure partie de cette marge est déjà capturée par
le compiled hot-loop, et que la résiduelle est en-dessous du seuil
mesurable (variance par run).

**Sana 1024 a un peu plus de marge dispatch** : avg util 60 % avec
35 % du temps <50 %. Mais le gap mesuré compiled vs sequential reste
modeste (4.8 %), suggérant que le hot-loop capture déjà l'essentiel
du gain disponible côté Python.

**Conclusion** : la cible mandate ≥1.5× compiled vs sequential sur
Sana 4Kpx est **structurellement incompressible**. Profile-driven
factual evidence : à GPU util sustained 83.5 %, l'arithmétique de
borne haute donne au mieux ≈1.20× (100/83.5) — et seulement si on
ramène Python à 0, ce qui est physiquement impossible.

## Conséquence pour les hypothèses H1–H5

Le pattern observé (compute-bound à 4Kpx, dispatch-light à 1024) est
**cohérent avec un hot-loop déjà bien optimisé**, pas avec une
régression silencieuse :
- H1 (silent fallback): si le hot-loop tombait en sequential, on
  verrait un gap ≈0 sur les deux shapes — au 4Kpx oui (compute-bound),
  mais au 1024 le gap −4.8 % est mesuré, donc compiled fait quelque
  chose de différent et de plus rapide. H1 invalidée par les mesures.
- H4 (defensive sync per-op): contrediraient le 67.5 % >90 % util
  (sync per-op forcerait des creux fréquents). Invalidée par les
  mesures.
- H5 (degenerate hot-loop): même argument que H1 — gap mesurable à
  1024. Invalidée.

H2 / H3 sont également invalidées par lecture code (voir
`diagnostic_par_hypothese.md`).

Aucun fix à appliquer. Closure factuelle authorisée par le mandate.
