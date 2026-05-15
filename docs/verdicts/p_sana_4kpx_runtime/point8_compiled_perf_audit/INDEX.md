# POINT 8 — audit perf triton compiled vs triton_sequential
## P-SANA-4KPX-RUNTIME / phase 8 — closure factuelle (no fix)

R29 inspectable artefact. Audit conclu par rapport factuel
profile-driven : le gap mesuré est **structurellement incompressible**
au 4Kpx (compute-bound), aucune des hypothèses H1–H5 ne tient sous
lecture code + mesure.

## TL;DR

| | Sana 1024 | Sana 4Kpx |
|---|---|---|
| triton (compiled hot-loop) | 70.35 s | 511.78 s |
| triton_sequential | 73.91 s | 513.96 s |
| **gap compiled vs sequential** | **−4.8 %** | **−0.4 %** |
| GPU util sustained | 60.2 % avg | **83.5 % avg, 67.5 % >90 %** |
| diagnostic | dispatch headroom modéré | **compute-bound** |

Mandate target 1.5× speedup compiled vs sequential sur Sana 4Kpx FULL
**n'est pas atteignable** à 83.5 % GPU util sustained — la borne haute
arithmétique 100/83.5 = 1.20× est physiquement impossible à atteindre
(requiert 0 % Python). Le gap mesuré 0.4 % est cohérent avec un hot-loop
déjà structurellement optimal pour ce que Python peut faire.

## Méthodologie

ÉTAPE A — baseline profile : 4 runs paired (Sana 1024 + Sana 4Kpx ×
triton + triton_seq) sous conditions identiques :
- `CUDA_VISIBLE_DEVICES=2 --hardware v100-32g` (single 32 GiB V100)
- `NBX_DISABLE_AUTOTUNE=1` (matche POINT 7 baseline)
- Prompt unique "a red apple", 12 diffusion steps
- GPU util échantillonné à 10 Hz pendant toute la durée

ÉTAPE B — diagnostic par hypothèse : lecture code obligatoire avec
citations file:line, corrélation avec les mesures.

ÉTAPE C — pas de fix (les 5 hypothèses sont invalidées).

ÉTAPE D — anti-régression : les 4 PNG de la baseline sont les
artefacts R29 finals (red apples sur toutes les 4 cellules).

## Détails

- `baseline_profile.md` — wall times + GPU util trajectory partitionnée
  par run, verdict compute-bound chiffré.
- `diagnostic_par_hypothese.md` — H1–H5 invalidées (file:line proofs).
- `fix_summary.md` — pas de fix, backlog des pistes hors scope.
- `gpu_util_full_run.log` — log brut nvidia-smi (10 Hz × 4 GPUs).
- `sana1024_triton.png` / `sana1024_triton_seq.png` — Sana 1024 R29.
- `sana4kpx_triton.png` / `sana4kpx_triton_seq.png` — Sana 4Kpx R29.

## Matrice anti-régression POINT 8

| modèle | mode | PNG | verdict |
|---|---|---|---|
| Sana 1024 | `--triton` | `sana1024_triton.png` | 🍎 PASS |
| Sana 1024 | `--triton-sequential` | `sana1024_triton_seq.png` | 🍎 PASS |
| Sana 4Kpx | `--triton` | `sana4kpx_triton.png` | 🍎 PASS |
| Sana 4Kpx | `--triton-sequential` | `sana4kpx_triton_seq.png` | 🍎 PASS |

Pas de changement code, donc pas de risque de régression sur les
autres modèles (PixArt-XL / PixArt-Sigma / TinyLlama). La matrice
POINT 7 (commit `90ac662`) reste l'oracle de référence pour ces
modèles, valide tant qu'aucun commit ne touche le hot-loop.

## Verdict scope POINT 8

| critère mandate | verdict |
|---|---|
| Compiled ≥1.5× sequential sur Sana 4Kpx FULL **OU** rapport factuel structural | **CLOSED via rapport factuel** — preuves chiffrées profile-driven (GPU util sustained 83.5 %), borne haute arithmétique 1.20× < cible 1.5× |
| Pomme rouge maintenue sur tous modèles | **PASS** — 4 R29 PNGs red apples, hot-loop intact (zero code change) |
| Aucune régression introduite | **PASS** — aucun changement code |

## Pistes backlog (post-POINT 8)

Pour atteindre un speedup significatif au-delà du dispatch :

1. **`P-TRITON-FUSED-KERNELS`** — fusion d'ops adjacentes au
   compile-time (mul+add+silu+conv → un kernel Triton fusé). Pattern
   matching graph-level + code-gen. Estimation > 200 lignes, chantier
   dédié.
2. **`P-CUDA-GRAPHS`** — capture/replay CUDA Graphs pour rejouer la
   séquence d'ops sans Python. Speedup typique 1.5–2× sur shapes
   dispatch-dominated. Compatible Volta (sm_70+).
3. **Autotune ON measurement** — re-mesurer le baseline avec
   `NBX_DISABLE_AUTOTUNE=0` (production-realistic). Le gap mesurable
   pourrait changer si le kernel time descend (Python overhead
   relatif augmente). Ne nécessite pas de code, juste une mesure.

Aucune de ces pistes n'est dans le périmètre POINT 8.

## Cumulative session statut

| | commit / tag | verdict |
|---|---|---|
| POINTS 1-4bis | ea8e8e2..b9d46ae | diag dtype + relu walk-back |
| POINT 5 | ad9b7a3 | halo bug fix tiled conv NBX |
| POINT 6 H2 (a) | dc7c3b7 | `add_inplace_nbx` contiguous guard |
| POINT 6 H2 (b) | a862fe0 | `_fused/_tiled_conv2d_nbx` `.contiguous()` |
| Tag clôture numérique | tag `p-sana-4kpx-runtime-closed` | sur a862fe0 |
| POINT 7 | 90ac662 | full pipeline 32 GiB triton + compiled red apples |
| Tag clôture totale | tag `p-sana-4kpx-runtime-fully-closed` | sur 90ac662 |
| Doc rules + CHANGELOG | f25e30a, 67d4f34 | règles A+B (CLAUDE.md local) + CHANGELOG entries |
| **POINT 8** | this commit | **closure factuelle audit perf compiled-vs-sequential** |
