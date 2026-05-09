# P-SANA-4KPX-RUNTIME — verdict factuel (page blanche 2026-05-08)

Recadrage Hocine 2026-05-08 : nettoyer les artefacts accumulés (74→14
fichiers durables), garder uniquement ce qui est reproductible depuis
HEAD, et reposer le problème sur des bases factuelles propres avant
l'investigation suivante.

L'historique chronologique exhaustif (`final²`…`final¹¹`) est archivé
dans `_archive_pre_2026-05-08/verdict_final_accumulated.md` pour audit
trail.

## 1. Bug factuel observé

| Mode (Sana 4Kpx, BF16, prompt "a red apple", 12 steps) | Wall | PNG R29 |
|---|---|---|
| `--sequential` (PyTorch eager / cuDNN-cuBLAS) | 89s | ✅ red apple |
| `--triton-sequential` (NBX Triton-pure) | 1102s | ❌ structured green texture |

Sana 1024 (même architecture, smaller resolution) :
| Mode | Wall | PNG R29 |
|---|---|---|
| `--sequential` | ~50s | ✅ red apple |
| `--triton-sequential` | 51s | ✅ red apple |

→ Le bug est **shape-dependent** (Sana 4Kpx only) et n'apparaît que
sur le path Triton-pure NeuroBrix, pas sur le path PyTorch eager
exécutant le même DAG ATen.

## 2. Hypothèses ÉLIMINÉES factuellement (microtests + A/B)

Chaque kernel testé en isolation à la shape exacte de Sana 4Kpx
contre le reference torch — **bit-exact** (max diff ≤ fp16 ULP) :

- `conv2d_forward_kernel` @ `(2,32,128,128)→(2,2240,128,128)` bf16+fp16
- `add_bias_broadcast_kernel` @ `(2,16384,2240)+(1,16384,2240)`
- `bmm` cross-attention @ K=16384
- `softmax` fp32 @ scales 1, 1e6, 1e7 (sum=1.0, no NaN)
- `mm` self-attn Q-projection @ `(32768,2240)×(2240,2240)` fp32

A/B test diagnostic landed :
- **autotune éliminé** (Cas B): `NBX_DISABLE_AUTOTUNE=1` + Sana 4Kpx
  → PNG toujours garbage. Static config Volta-safe ne fixe rien.
- **VAE acquitté**: boundary diff transformer→VAE montre transformer
  output déjà corrompu (max_d=0.998 sur valeurs de magnitude ~1).

## 3. Outils factuels durables disponibles

- `tools/diff_dag_op_by_op.py` — analyse offline op-par-op des logs
  VALUE_TRACE existants. Classifie chaque op aligné par
  `(op_idx, op_uid)` en `{none, noise, borderline, real}` selon
  positions 0-3 du fingerprint (position-4 / n-1 exclue, artifact
  connu). Génère `diff_dag_op_by_op_report.md`.
- Env vars diagnostiques actifs dans `_autotune_policy.py` :
  - `NBX_DISABLE_AUTOTUNE=1` — pin tous les `@triton.autotune` à un
    config Volta-safe statique (BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    warps=4, stages=2).
  - `NBX_FORCE_FP32_ACCUM=1` — upcast tous les inputs fp16/bf16 à
    fp32 dans mm/bmm/addmm wrappers. Sacrifie 2× VRAM, sur 4Kpx
    OOM sur V100 32GB.
- Logs de référence pour analyse (post-fingerprint-fix commits
  `a2fd933` / `aadf5b0` / `8104c8d`) :
  - `sequential_4kpx_values_v2.log` (oracle PNG cohérent)
  - `sana4kpx_triseq_values_v5.log` (path Triton garbage)
  - `sana1024_triseq_values_v3.log` (Sana 1024 triton oracle PNG cohérent)

## 4. Premier diff factuel — narrative à reconsidérer

Tools/diff_dag_op_by_op.py sur Sana 4Kpx (sequential vs triton_seq) :

| Milestone | trace_line | op_uid | component | max_d | rel | sf |
|---|---|---|---|---|---|---|
| First real (rel≥5% AND abs≥0.01) | 2361 | `aten.pow::70` | text_encoder | 0.011 | 5.1% | 0 |
| First max_abs ≥ 1.0 | 3053 | `aten.pow::92` | text_encoder | 1.51 | 13% | 0 |
| First max_rel ≥ 50% | 6902 | `aten.convolution::0` | transformer iter 1 | 0.20 | 217% | 0 |
| **First sign flip** | 6953 | `aten.mm::0` (block.0.self_attn.query) | transformer iter 1 | 26.64 | 287% | 1 |

Distribution: 7976 `none` + 5237 `noise` + 3169 `borderline` + 18931
`real` + 450 `skip` = 35763 ops alignés. **18931/35763 (~53%) ops
classés "real" divergence.**

## 5. Question méthodologique critique non résolue

53% d'ops "real" est massif. Hypothèse à vérifier : seuils trop
sensibles fp16 noise normal entre PyTorch eager et NBX Triton, plutôt
que vraie "drift accumulée distribuée".

**Test discriminant requis (Phase 1 next session)** :

Run le même tool sur **Sana 1024** (sequential vs triton_seq, both
PNG cohérents).
- Si Sana 1024 ~0 ops "real" → seuils calibrés correctement,
  pow::70 + chain en text_encoder Sana 4Kpx sont vraies divergences.
  Investigation pow + `_to_copy` upstream pertinente.
- Si Sana 1024 ~50% "real" comme 4Kpx → seuils mal calibrés. Pivot
  méthodologie : stats globaux (min/max/mean/std) au lieu de 5
  positions samples.

## 6. État commits durables HEAD (commit 7b1bf80)

Fixes de production : R30 universalité, depthwise specialization,
F2a pixel_shuffle broadcast, in-place add interceptor, rms_norm
contiguity, add_bias_broadcast, loop-var retention fix,
fingerprint methodology (3 fixes : logical-to-physical, data_ptr
offset, drop n-1 sample).

Diagnostic infrastructure : tools/diff_dag_op_by_op.py,
NBX_DISABLE_AUTOTUNE, NBX_FORCE_FP32_ACCUM.

## 7. Phase 1 résultat (2026-05-08, commit `eda0629`)

Sana 1024 control diff (sequential vs triton_seq, both PNG R29 PASS) :

| Variant | none | noise | borderline | real | skip |
|---|---|---|---|---|---|
| Sana 4Kpx | 7976 | 5237 | 3169 | **18931 (~53%)** | 450 |
| Sana 1024 | 7841 | 5370 | 3251 | **18851 (~53%)** | 438 |

→ **Cas B confirmé**. Le seuil 5-position flagge cuDNN-vs-Triton fp16
noise sur LES DEUX variants identiquement. La narrative "drift
accumulée distribuée" Sana 4Kpx était un artifact d'analyse, pas un
signature de bug.

## 8. Pivot méthodologique : cross-variant analysis

`tools/diff_dag_cross_variant.py` compare **par (op_idx, op_uid)** la
divergence rel max entre 4Kpx (PNG garbage) et 1024 (PNG cohérent).
Filtre les ops à `rel_ratio = rel_4kpx / rel_1024 >= 10` ET `rel_4kpx
>= 0.5` — exclut le bruit baseline commun aux deux variants, isole
le signature shape-dependent.

Résultats (commit `eda0629`) :

- **126 ops avec rel_ratio ≥ 10** : signature shape-dependent claire.
- **First in trace order** : `aten.transpose::2` op_idx=67 line 9309
  = **transformer iter 2** (boundaries `[0, 3449, 6898, 9241, ...]`).
  rel_4kpx=5.05, rel_1024=0.0002, ratio 20621×. abs_4kpx=30.05,
  abs_1024=0.001.
- **TOP-20 par rel_ratio** : silu / relu / conv / pixel_shuffle ops
  avec op_idx 519-692 (pattern caractéristique VAE) avec ratios
  165-2.9M. La VAE 4Kpx montre les divergences les plus extrêmes.

Picture qui émerge :
1. text_encoder + transformer iter 0 + iter 1 : pas d'anomalie
   shape-specific (rel_4kpx ≈ rel_1024).
2. À iter 2 op_idx=67 transpose : drift accumulée à travers iter 1
   noise pred → scheduler latent update → début iter 2 visible en
   cross-variant.
3. VAE : ratios extrêmes (jusqu'à 2.9M) — soit VAE 4Kpx amplifie
   shape-dependent un input déjà corrompu par transformer, soit VAE
   a son propre bug shape-dependent.

## 9.5. Phase 1 path 1 attempts (VAE isolation discriminant)

Two attempts both blocked by V100 32 GB ceiling :

- **v1 (commit `4eb8df0`)** : monkey-patch `RuntimeExecutor._execute_component`
  to override `comp_inputs` at VAE call. **Phase A capture OK**
  (`vae_isolation_input.pt` 2 MB + sequential decode PNG cohérent
  red apple). **Phase B replay OOM** because the override only fires
  AT the VAE call, but the diffusion loop (text_encoder + 12
  transformer iters) still ran fully and exhausted memory before
  reaching VAE.

- **v2 (`tools/vae_isolation_probe.py --vae-only-decode`)** : public
  API path via `RuntimeExecutor(pkg, plan, mode).setup()` +
  `executor.executors["vae"].run({"z": saved_z})`. Bypasses
  text_encoder + transformer entirely. **Still OOM** even with
  `NBX_DISABLE_AUTOTUNE=1` :
  ```
  GPU malloc failed for 8 GiB: live_tracked=25172MB
  driver_free=6996MB
  ```
  
  Root cause: the standalone path bypasses the TilingEngine
  (`executor._component_tiling.get("vae")`) that the standard
  pipeline uses. At Sana 4Kpx, the VAE input `(1, 32, 128, 128)`
  spatially exceeds trace size, so without tiling the upsample
  chain allocates full 4Kpx output tensors immediately → > 32 GB.

### Path 1 next iteration plan

To engage TilingEngine in the standalone path, after `executor.setup()`:
```python
tiling = executor._component_tiling.get("vae")
output = tiling.tiled_execute(saved_z, lambda tile:
    executor.executors["vae"].run({"z": tile}))
```

This replays the same tiling logic that the full pipeline uses.
ETA next session: 30 min impl + 5-10 min run + R29 visual.

## 9. Non-fait, à faire (next session)

Pour discriminer "VAE input-corrupted vs VAE op-buggy" :

- **Test d'isolation VAE** : feed le final latent du pipeline
  sequential 4Kpx (qui produit PNG cohérent) à la VAE triton 4Kpx.
  Si output cohérent → VAE triton OK, bug en transformer. Si output
  garbage → VAE triton a un bug shape-dependent intrinsèque.
- **Audit op_idx=67 transformer iter 2** : identifier ce que
  `aten.transpose::2` opère (input shape, parent_module dans
  graph.json) et l'op upstream qui produit son input. Comprendre
  pourquoi iter 2 cross seuil mais pas iter 0/1.
- **Audit TOP-VAE ops** (silu::18-23, relu::15-16, pixel_shuffle::3,
  convolution::61) : vérifier si chaque kernel est bit-exact à shape
  4Kpx avec inputs aléatoires (microtest pattern de la session
  précédente).
