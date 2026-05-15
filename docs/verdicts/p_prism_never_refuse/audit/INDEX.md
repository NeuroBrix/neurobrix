# P-PRISM-NEVER-REFUSE — audit ÉTAPE A

R29 inspectable artefact. Audit factuel des stratégies avant tout fix.

## A.1 — Inventaire fichiers stratégies (`src/neurobrix/core/strategies/`)

8 stratégies concrètes + 1 base abstraite :

| fichier | LOC | description (docstring) |
|---|---|---|
| `base.py` | 298 | Abstract base class. Strategies handle HOW components execute. |
| `single_gpu.py` | 138 | All components execute on a single GPU. |
| `lazy_sequential.py` | 140 | Load one component to GPU, executes, unloads. Peak = max(single component). |
| `component_placement.py` | 157 | Whole components distributed across multi-GPU. |
| `block_scatter.py` | 176 | Block-level best-fit distribution across multi-GPU. |
| `pipeline_parallel.py` | 179 | Per-layer sequential fill across multi-GPU (Accelerate-style). |
| `weight_sharding.py` | 103 | Weight files split across multi-GPU (round-robin). |
| `tp_sharding.py` | 446 | Tensor parallel block sharding. |
| `zero3.py` | 764 | CPU offload + GPU compute. Last-resort cascade. |

## A.2 — Cascade solver (PrismSolver.solve)

**Single-GPU profile** (`len(devices) == 1`): 4 stratégies
1. `single_gpu` (score 1000)
2. `single_gpu_lifecycle` (score 900)
3. `lazy_sequential` (score 300)
4. `zero3` (score 1 — last resort)

**Multi-GPU profile**: 9 stratégies
1. `single_gpu` (1000) — uses largest single GPU
2. `single_gpu_lifecycle` (900) — single GPU with per-component lifecycle
3. `pipeline_parallel` (850)
4. `component_placement` (750)
5. `block_scatter` (700)
6. `weight_sharding` (680)
7. `component_placement_lazy` (400)
8. `lazy_sequential` (300)
9. `zero3` (1)

Le POINT 7 ÉTAPE C `_fail_error` message liste seulement
"single_gpu / component_placement / pipeline_parallel / block_scatter /
weight_sharding" — c'est un libellé hardcodé ligne 2826 qui n'inclut
pas lazy_sequential, single_gpu_lifecycle, component_placement_lazy ni
zero3. **Toutes les 9 stratégies ARE actually cascaded**. Le libellé
d'erreur est juste obsolète et trompeur.

`single_gpu` en cold mode (run mode) utilise le budget
`max(weight_mb + activation_mb)` (max-peak per component), pas la somme.
Ce qui signifie que `single_gpu` accepte tout plan où le pire composant
isolé tient — implicitement il assume du lifecycle au runtime. Mais sa
boucle d'exécution n'unload pas explicitement entre composants.
`single_gpu_lifecycle` est la stratégie qui configure activement
l'unloading lifecycle au runtime.

## A.3 — CPU support audit

**Triton CPU** (mai 2026) : `triton-cpu` est un fork séparé toujours
incomplet sur la couverture kernels mainstream. Le mainline Triton
ne supporte pas CPU end-to-end pour les workloads diffusion (conv2d
fused, attention, etc.) — soutien limité aux primitives.

**NeuroBrix CPU path** : ZeRO-3 (`zero3.py`) implémente une stratégie
de **CPU offload + GPU compute** : weights en RAM CPU (pinned pour DMA
rapide), bloc weights streamés vers GPU pour le compute d'un seul block
à la fois. Last-resort cascade — toujours tente avant ZERO FALLBACK.

**PyTorch eager CPU** : disponible via `--sequential` (PyTorch native
ATen dispatcher), exécute entièrement sur CPU si `device='cpu'`. Très
lent mais universel.

## A.4 — Audit doctrinal model-agnostic

`grep -nE "Sana|PixArt|TinyLlama|Llama|Flux|Qwen|DeepSeek|Janus|..."`
sur `core/prism/*.py` + `core/strategies/*.py` :

**Aucune violation hardcode** :
- 5 hits, tous dans des COMMENTAIRES (justifications historiques de
  thresholds, exemples de patterns rencontrés).
- 0 conditionnels code-actif `if "sana" in ...`, `if model_name == ...`,
  etc.

Prism + strategies sont structurellement model-agnostic. Les références
à Sana dans les commentaires sont pédagogiques (documentent pourquoi tel
threshold à 0.25 × VRAM existe via l'exemple du Sana 4Kpx VAE upsample).
**Pas de dette technique R-MODEL-AGNOSTIC à fixer dans cette session.**

## A.5 — Décision factuelle de stratégie pour Sana 4Kpx + 16 GiB

**Verdict initial** : `lazy_sequential` est le candidat parfait.
- Garanti par construction que peak runtime = max(single component live)
- Pour Sana 4Kpx : max(text_encoder=10.6 GiB, transformer=10 GiB,
  vae=13 GiB live) = **~13 GiB**
- Marge confortable sur budget 16 GiB

**Mais POINT 9 montre** : `single_gpu` gagne au scoring (1000 vs 300)
parce que sa max-peak cold-mode budget formula le rend "compatible"
avec 16 GiB. Au runtime, single_gpu ne fait PAS l'unloading explicite
entre composants → OOM.

**Voie B planifiée** :
1. **B.1 priorité** — probe `lazy_sequential` explicitement
   (`NBX_FORCE_STRATEGY=lazy_sequential`) pour Sana 4Kpx 16 GiB.
   Si PNG cohérente → fix scoring/cascade pour que lazy_sequential
   gagne sur les configs où single_gpu OOMerait au runtime.
2. **B.2 fallback** — si lazy_sequential aussi OOM (VAE seul peak
   >16 GiB), engager `zero3` qui offload les weights VAE sur CPU
   pendant qu'on streame les blocks. Réserver pour configs encore
   plus extrêmes (8 GiB, etc.).
3. **B.3 architectural** — si single_gpu doit perdre face à
   lazy_sequential sur 16 GiB, soit changer scoring (lazy_sequential
   monte si single_gpu's max-peak >0.8 × budget) soit fixer
   `_try_single_gpu` runtime pour qu'il fasse de l'unloading explicite
   (rapprochement avec single_gpu_lifecycle).

Probe en cours dans cette session ; verdict B suit.
