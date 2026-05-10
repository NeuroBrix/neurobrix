# POINT 2bis — registry plumbing fix (factory + graph_executor)
## P-SANA-4KPX-RUNTIME / TritonDtypeEngine maturation phase 2bis

R29 inspectable artefact for POINT 2bis (bloquant POINT 3).

## Cible

Faire que `GraphExecutor` accède effectivement au `cache_path` de
son composant pour que `get_component_flag` puisse résoudre
`_model_name` correctement. Le bug pré-existant: `self._pkg`
n'existait pas sur GraphExecutor (uniquement sur serving engine),
donc `getattr(self._pkg, 'cache_path', None)` retournait silencieusement
None → `_model_name=None` → registry retournait default → flag
`activations_fp16_safe` n'était jamais lu depuis le YAML au runtime.

## Implémentation — minimale (2 fichiers, 3 changements)

| fichier | modification |
|---|---|
| `core/runtime/factory.py` | Pose `executor._cache_path = cache_path` après création de l'executor (ligne ~412, juste avant attache du component_handler) |
| `core/runtime/graph_executor.py` `__init__` | Initialise `self._cache_path: Optional[str] = None` pour les instanciations manuelles (tests) |
| `core/runtime/graph_executor.py:1573` | `_run_triton_sequential` flag init lit `self._cache_path` au lieu de `getattr(self._pkg, 'cache_path')` |
| `core/runtime/graph_executor.py:1933` | `_ensure_triton_compiled` flag init même remplacement |

Fallback gracieux: si `self._cache_path` est None (instanciation hors
factory pour tests), `_model_name=None` → registry default → flag False.

Audit `grep self._pkg` dans graph_executor.py: 2 occurrences seulement,
toutes dans le flag init pré-existant + ma version POINT 2. Aucune
autre call site avec le même bug. Tu peux fermer la liste "follow-up
plumbing audit" — elle est vide.

## Validation factuelle critique

**Test sans env `NBX_ACTIVATIONS_FP16_SAFE`**:

`dtype_walk_4kpx_post_point2bis_NO_env.tsv` montre:

| op_uid | tri input | tri output | match | verdict |
|---|---|---|---|---|
| `custom.rms_norm::0` | [fp16, fp16] | **fp16** | match_out=true | cast-back fired ✓ |
| `custom.rms_norm::1` | [fp16, fp16] | **fp16** | match_io=true | cast-back fired ✓ |
| `custom.rms_norm::2` | [fp16, fp16] | **fp16** | match_out=true | cast-back fired ✓ |
| `custom.rms_norm::3` | [fp16, fp16] | **fp16** | match_io=true | cast-back fired ✓ |

Total divergence count: **291** (in+out) — IDENTIQUE à post-POINT 2
AVEC env override. Le pattern est désormais le bon DEFAULT pour les
modèles annotés. L'env override n'est plus nécessaire.

## Anti-régression — R29 PNG visual (TOUS sans env)

| modèle | mode | PNG | verdict |
|---|---|---|---|
| Sana 1024 | triton (compiled) | `sana1024_tri_compiled_post_point2bis.png` | **RED APPLE** ✓ |
| Sana 1024 | triton_sequential | `sana1024_tri_seq_post_point2bis.png` | **RED APPLE** ✓ |
| PixArt-XL | triton_sequential | `pixart_xl_tri_seq_post_point2bis.png` | **RED APPLE** ✓ |
| PixArt-Sigma | triton_sequential | `pixart_sigma_tri_seq_post_point2bis.png` | **RED APPLE** ✓ |
| TinyLlama | triton_sequential | (text output) | **coherent poem about apples** ✓ |
| Sana 4Kpx | triton_sequential VAE-iso | `sana4kpx_vae_iso_tri_seq_post_point2bis.png` | **bandes RGB structurées** (identique au pattern POINT 2 avec env) ✓ |

LLM non-régressé (registry n'annote pas TinyLlama → flag stays False
→ cast-back ne fire pas → comportement PyTorch-oracle préservé).

## Implication chantier

POINT 2 est désormais **permanent par défaut** pour les modèles
annotés `activations_fp16_safe: true` dans `model_registry.yml`:
- Sana 1024 transformer + VAE
- Sana 4Kpx transformer + VAE
- (autres modèles annotés futurs)

Plus besoin de l'env `NBX_ACTIVATIONS_FP16_SAFE=1`. Le runtime lit
le registry directement.

## Statut chantier

- POINT 1: ✅ commited `ea8e8e2`
- POINT 2: ✅ commited `331c611`
- POINT 2bis: ✅ implementé + validation factuelle dtype_walk + anti-régression 6/6
- POINT 3: en attente (re-run dtype_walk synthèse — maintenant mesure le bon système sans contamination env)
- POINT 4-5: en attente (TilingEngine Famille B)

Awaiting Hocine confirmation visuelle PNG + dtype_walk before POINT 3.
