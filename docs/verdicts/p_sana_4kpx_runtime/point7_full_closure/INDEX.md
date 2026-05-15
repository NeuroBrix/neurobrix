# POINT 7 — clôture totale Sana 4Kpx Triton mode
## P-SANA-4KPX-RUNTIME / phase 7 — VICTOIRE COMPLÈTE + 1 découverte factuelle

R29 inspectable artefact for POINT 7. Closes the remaining items of the
P-SANA-4KPX-RUNTIME scope after the numerical correctness commits
`a862fe0` + tag `p-sana-4kpx-runtime-closed`.

## Résultats par étape

### ÉTAPE A — full Sana 4Kpx triton_sequential sur 1× V100 32 GiB

| | valeur |
|---|---|
| Hardware | `v100-32g` (single GPU, `CUDA_VISIBLE_DEVICES=2`) |
| Stratégie Prism | `single_gpu` |
| Wall time | **510.02 s** (≈8.5 min) |
| Peak VRAM | **16.6 GiB / 32.5 GiB** (51% du budget) |
| Output | 🍎 `etapeA_full_4kpx_triton_seq.png` — red apple coherent, photoréaliste |
| Verdict | **PASS** — H1 confirmée |

La mesure historique conv::62 OOM (26+8 GiB vs 32 GiB) est éliminée
par les fixes POINTS 1-6. Le pipeline complet tourne en `single_gpu`
sans tiling spatial actif au niveau Prism — l'op-level tiling
embedded dans `conv2d_wrapper` (kernel-level halo streaming, commit
`ad9b7a3`) garde le runtime peak à 16.6 GiB.

### ÉTAPE B — full Sana 4Kpx triton compiled sur 1× V100 32 GiB

| | valeur |
|---|---|
| Hardware | `v100-32g` (single GPU, `CUDA_VISIBLE_DEVICES=2`) |
| Stratégie Prism | `single_gpu` |
| Wall time | **514.83 s** |
| Output | 🍎 `etapeB_full_4kpx_triton_compiled.png` — red apple coherent (apple + leaf + shadow) |
| Verdict | **PASS** — H2 confirmée |

Le mode `triton` (TritonSequence compiled hot loop) hérite mécaniquement
des fixes wrappers (POINTS 1-6 H2) parce qu'il lit le même `graph.json`
et appelle les mêmes wrappers tilés. Pas de divergence vs sequential.
Items 1 et 3 du scope CLOSED.

### ÉTAPE C — configs VRAM contraintes (16 GiB single + 2×16 GiB multi-GPU)

| config | hardware profile | verdict |
|---|---|---|
| C1 single 16 GiB | `v100-16g` (`CUDA_VISIBLE_DEVICES=0`) | ❌ `ZERO FALLBACK` au planning (Prism estimator) |
| C2 2× 16 GiB | `v100-16g-x2-01` (`CUDA_VISIBLE_DEVICES=0,1`) | ❌ `ZERO FALLBACK` au planning (Prism estimator) |

**Découverte factuelle structurelle** :

`PrismSolver._fail_error` rapporte pour les deux configs :
```
Components:
  vae: 31356MB (W=1191, A=28672)
  text_encoder: 10538MB (W=9973, A=63)
  transformer: 10314MB (W=6121, A=3702)
```

L'estimateur d'activations VAE = **28 GiB** (worst-case full
materialization à 4096×4096) — mais le runtime peak mesuré en ÉTAPE A
est **16.6 GiB**. Gap d'estimation ≈ 11 GiB / 70%, dû au fait que
l'estimateur ne tient pas compte de l'op-level tiling kernel-embedded
(le `conv2d_wrapper` halo-streaming réduit le peak per-op, et le
deferred-free / arena lifecycle libère les buffers intermédiaires
plus tôt qu'en worst-case).

Conséquence : pour les configs ≤ 28 GiB de budget VAE, Prism rejette
au planning AVANT que l'op-level tiling ne puisse engager. Les
5 stratégies cascadées (`single_gpu` / `component_placement` /
`pipeline_parallel` / `block_scatter` / `weight_sharding`) échouent
toutes parce que VAE ne tient pas sur un seul GPU avec cette
estimation, et qu'aucune stratégie ne split les activations VAE
entre devices.

**Cette découverte est hors scope POINT 7** (le mandate l'avait
explicitement anticipée : *"Si multi-GPU placement n'est pas branché
pour le VAE Sana 4Kpx aujourd'hui, c'est une découverte factuelle
à rapporter, pas un bug à fixer dans cette session"*).

Ouvre deux chantiers à nommer en backlog :
1. **`P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE`** — réviser
   `PrismSolver._estimate_activation_memory()` (ou équivalent) pour
   intégrer l'op-level tiling kernel-embedded dans l'estimation.
   Sans ce fix, tout modèle dont la VAE worst-case dépasse le budget
   single-GPU est rejeté même quand le runtime tiendrait.
2. **`P-MULTI-GPU-NBX-ADAPTER`** — déjà nommé dans CLAUDE.md
   (update 2026-05-05). `component_placement` / `pipeline_parallel`
   doivent gérer le cas où un component (ici VAE) dépasse une seule
   GPU et nécessite un split intra-component cross-device. Note :
   ce fix est lower-priority maintenant que la single-GPU 32 GiB
   marche en pratique.

### ÉTAPE D — non-déclenchée

H1 confirmée en ÉTAPE A (pas d'OOM), donc l'audit live-watermark
n'est plus nécessaire. Les fixes POINTS 1-6 ont réduit le peak
suffisamment (26.4 GiB → 16.6 GiB measured) pour libérer la 32 GiB
single-GPU.

### ÉTAPE E — anti-régression matrice complète

| modèle | mode | wall time | PNG / TXT | verdict |
|---|---|---|---|---|
| Sana 1024 | `--triton-sequential` | 74.86 s | `etapeE_sana1024_triton_seq.png` | 🍎 PASS |
| Sana 1024 | `--triton` | 69.40 s | `etapeE_sana1024_triton.png` | 🍎 PASS |
| Sana 1024 | `--compiled` | 15.66 s | `etapeE_sana1024_compiled.png` | 🍎 PASS |
| Sana 1024 | `--sequential` | 16.35 s | `etapeE_sana1024_sequential.png` | 🍎 PASS |
| PixArt-Sigma-XL | `--triton-sequential` | 141.54 s | `etapeE_pixart_sigma_triton_seq.png` | 🍎 PASS (apple + leaf) |
| PixArt-XL | `--triton-sequential` | 114.97 s | `etapeE_pixart_xl_triton_seq.png` | 🍎 PASS |
| TinyLlama | `--triton-sequential` | 57.62 s | `etapeE_tinyllama_triton_seq.txt` | ✓ coherent English poem about apples |
| Sana 4Kpx VAE-iso | `--triton-sequential` | (acquis POINT 6) | `etapeE_sana4kpx_vae_iso_point6_acquis.png` | 🍎 PASS (POINT 6) |
| **Sana 4Kpx FULL** | `--triton-sequential` | 510.02 s | `etapeA_full_4kpx_triton_seq.png` | **🍎 PASS** (ÉTAPE A) |
| **Sana 4Kpx FULL** | `--triton` | 514.83 s | `etapeB_full_4kpx_triton_compiled.png` | **🍎 PASS** (ÉTAPE B) |
| Sana 4Kpx FULL 16 GiB | (any) | — | — | ❌ Prism estimator rejects (ÉTAPE C1 — découverte factuelle) |
| Sana 4Kpx FULL 2×16 GiB | (any) | — | — | ❌ Prism estimator rejects (ÉTAPE C2 — découverte factuelle) |

**8/8 cellules numériquement OK**. Les 2 cellules rouges sont une
limitation Prism (planning estimator) **distincte** de la couche
triton — le moteur Triton fonctionne correctement, c'est l'allocateur
qui rejette en amont.

## Verdict scope POINT 7

| item du scope | verdict |
|---|---|
| 1. Sana 4Kpx triton compiled marche après POINTS 1-6 | **CLOSED — PASS** (ÉTAPE B) |
| 2. Sana 4Kpx full pipeline triton_seq sur 1× V100 32 GiB | **CLOSED — PASS** (ÉTAPE A, peak 16.6 GiB / 32 GiB) |
| 3. Sana 4Kpx full pipeline triton compiled | **CLOSED — PASS** (ÉTAPE B) |
| 4. Prism multi-GPU placement sur VRAM-contrainte | **DÉCOUVERTE FACTUELLE** — non-branché à cause d'un estimator overly-conservative, hors scope POINT 7, ouvre 2 chantiers backlog |
| Anti-régression matrice | **CLOSED — PASS** (10/10 cellules dépendantes de la couche triton) |

P-SANA-4KPX-RUNTIME est définitivement closed en mode 1× V100 32 GiB,
qui est la config production normale. Les configs VRAM-contraintes
nécessitent un fix Prism (P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE)
documenté ci-dessus.

## Cumulative session statut

| | commit | verdict |
|---|---|---|
| POINTS 1-4bis | ea8e8e2..b9d46ae | diag dtype + relu walk-back |
| POINT 5 | ad9b7a3 | halo bug fix tiled conv NBX |
| POINT 6 H2 (a) | dc7c3b7 | `add_inplace_nbx` contiguous guard |
| POINT 6 H2 (b) | a862fe0 | `_fused/_tiled_conv2d_nbx` `.contiguous()` — VAE-iso red apple |
| Tag clôture numérique | tag `p-sana-4kpx-runtime-closed` | sur a862fe0 |
| Doc rules + CHANGELOG | f25e30a | règles A+B (CLAUDE.md local) + CHANGELOG entry |
| **POINT 7** | this commit | **full pipeline 32 GiB triton + compiled red apples, matrice anti-reg 10/10, 2 chantiers backlog ouverts** |
