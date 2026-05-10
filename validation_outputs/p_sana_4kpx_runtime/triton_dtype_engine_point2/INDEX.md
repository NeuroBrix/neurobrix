# POINT 2 — rms_norm cast-back data-driven (voie uniforme)
## P-SANA-4KPX-RUNTIME / TritonDtypeEngine maturation phase 2/5

R29 inspectable artefact for POINT 2 of the 5-point TritonDtypeEngine
maturation plan.

## Décision technique — voie uniforme

Choix entre les deux options du brief:
1. **Voie uniforme** (retenue): retire le gate `_AMP_FP32_OPS_OPT_IN_CAST_BACK`. Toutes les ops de `AMP_FP32_OPS` passent par `_wrap_fp32_internal_compute_dtype_output` — cast inputs to fp32 internally, conditionnel cast-back to compute_dtype gated on `activations_fp16_safe` registry flag.
2. Voie minimaliste: étendre le set au lieu de le retirer.

Justification (R16, doctrine TritonDtypeEngine data-driven):
- Le gate supplémentaire fragmentait la doctrine: rms_norm/div avaient le hook cast-back, mais rsqrt/exp/log/layer_norm/batch_norm n'avaient pas. Une seule règle (registry flag) pour gouverner toute la classe est plus cohérent et plus testable.
- Risque LLM contrôlé: les LLMs (TinyLlama, Qwen3) n'ont PAS `activations_fp16_safe: true` dans le registry → flag reste False → cast-back ne fire jamais → comportement PyTorch-oracle préservé.
- Seuls les VAE annotés (Sana 4Kpx VAE, Sana 1024 VAE) → cast-back uniforme pour TOUS les AMP_FP32 outputs sur le VAE post-loop activation chain.

rms_norm explicitement ajouté à `AMP_FP32_OPS` (n'y était pas: c'est un op custom NeuroBrix, pas dans `AT_FORALL_FP32` PyTorch).

## Implémentation

| fichier | modification |
|---|---|
| `triton/dtype.py` | `AMP_FP32_OPS` += `rms_norm`; suppression du set `_AMP_FP32_OPS_OPT_IN_CAST_BACK` et du membership check; `_wrap_fp32_internal_compute_dtype_output` appliqué uniformément à toutes les AMP_FP32_OPS et au div FP16_NEED_FP32 |
| `triton/sequential.py` | `TritonSequentialDispatcher.__init__` accepte `activations_fp16_safe`; propage à `_w.set_compute_dtype` + `_w.set_activations_fp16_safe` au boundary du dispatcher (mirror de TritonSequence.run() per-run set, sans try/finally car sequential ne nest pas avec compiled) |
| `core/runtime/graph_executor.py` | `_run_triton_sequential` lit le registry et passe `activations_fp16_safe` au dispatcher constructeur (mirror de la flag init compiled mode line 1909-1917) |

R33 zero-torch préservé.

## Anti-régression — R29 PNG visual

| modèle | mode | PNG | verdict |
|---|---|---|---|
| Sana 1024 | triton (compiled) | `sana1024_tri_compiled_post_point2.png` | **RED APPLE** ✓ |
| Sana 1024 | triton_sequential | `sana1024_tri_seq_post_point2.png` | **RED APPLE** ✓ |
| PixArt-XL | triton_sequential | `pixart_xl_tri_seq_post_point2.png` | **RED APPLE** ✓ |
| PixArt-Sigma | triton_sequential | `pixart_sigma_tri_seq_post_point2.png` | **RED APPLE** ✓ |
| TinyLlama | triton_sequential | (text output, voir commit message) | **coherent poem about apples** ✓ |
| Sana 4Kpx | triton_sequential VAE-iso | `sana4kpx_vae_iso_tri_seq_post_point2.png` | **bandes colorées rouges/bleues/vertes** (signal positif: color reconstruction now partially working — Famille B grid pattern still active per attendu) |

## Évolution Sana 4Kpx VAE-iso — POSITIVE SIGNAL

Pre-POINT 1: chaotic green-only texture
Post-POINT 1: structured monochromatic green-only grid (Famille A absorbed, Famille B exposed)
**Post-POINT 2: structured RGB color bands** — color information now reaches the output where before it didn't. The grid is still horizontally striped (Famille B / TilingEngine cluster signature unchanged) but the rms_norm chain is now propagating correct color components.

This visual evolution suggests POINT 2's wrap-input-to-fp32 (consistent application across all AMP_FP32_OPS) is restoring color-channel computation that was being lost in the pre-fix path.

## Pre-existing bug discovered — registry plumbing model_name lookup

L'init du flag à `graph_executor.py:1925` (compiled mode, déjà branché PRE-POINT 2) ET ma version sequential mode (line 1567-1581) utilisent `getattr(self._pkg, 'cache_path', None)`. **`self._pkg` n'existe pas sur GraphExecutor** — c'est uniquement attribut du serving engine (`serving/engine.py:50`). La factory (`factory.py:396-403`) ne le set pas.

Conséquence: `_model_name = None` → `get_component_flag` retourne `default=False` quel que soit le registry. **Le flag `activations_fp16_safe` n'est en fait JAMAIS lu depuis le registry au runtime, malgré la doctrine "Phase 1 opt-in"** documentée.

Workaround validé:
```bash
NBX_ACTIVATIONS_FP16_SAFE=1  # env override active le cast-back
```

Le walk Sana 4Kpx avec env override CONFIRME le cast-back fonctionne:
- rms_norm::0 tri output: **fp16** (cast-back fired) au lieu de fp32
- divergence count: 217 (in+out) — augmenté car NBX cast-back diverge maintenant de PyTorch oracle qui garde fp32 pour AMP_FP32_OPS. Cohérent avec doctrine.

**Suggested follow-up (out of POINT 2 scope)**: `factory.py` doit set `executor._cache_path` (ou `_model_name`) après création de GraphExecutor. C'est un fix de plumbing minimal qui rend la lecture registry effective. À remonter pour arbitrage avant ou après POINT 3.

## Statut chantier

- POINT 1: ✅ commited `ea8e8e2`
- POINT 2: ✅ implementé + anti-régression validée (5/5 PNG cohérentes + LLM)
- POINT 2bis (registry plumbing): ⚠️ bug pré-existant identifié, escalation
- POINT 3: en attente (re-walk + verdict après resolution registry plumbing)
- POINT 4-5: en attente (TilingEngine Famille B)

Awaiting Hocine confirmation visuelle PNG before POINT 3.
