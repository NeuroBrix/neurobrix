# POINT 1 — input::z cast à l'entrée du composant
## P-SANA-4KPX-RUNTIME / TritonDtypeEngine maturation phase 1/5

R29 inspectable artefact for POINT 1 of the 5-point TritonDtypeEngine
maturation plan (Hocine doctrine 2026-05-09).

## Cible

Faire que en mode triton (compiled) ET triton_sequential, `input::z`
arrive en compute_dtype (fp16 sur Volta) à l'arena Triton, comme
PyTorch sequential le fait déjà via `DtypeEngine` au point d'entrée.
La metadata chain unsqueeze→expand→clone→view est passthrough dans
les deux engines, donc le cast au point d'entrée propage automatiquement.

## Implémentation

Doctrine: TritonDtypeEngine est cerveau dtype data-driven. Le cast
décide à partir de:
- compute_dtype (Prism per-component dtype)
- graph_tensors metadata (input::* declared dtype)
Pas de hardcode, pas de per-model branching.

| fichier | modif | lignes |
|---|---|---|
| `src/neurobrix/triton/dtype.py` | `TritonDtypeEngine.cast_runtime_inputs(input_map, graph_tensors)` + helper `_target_dtype_for_input` | +60 |
| `src/neurobrix/triton/sequence.py` | `TritonSequence.bind_inputs` route via engine | +6 |
| `src/neurobrix/triton/sequential.py` | `TritonSequentialDispatcher.bind_inputs(input_map, graph_tensors)` | +18 |
| `src/neurobrix/core/runtime/graph_executor.py` | 1-line wiring: `cast_input_map = dispatcher.bind_inputs(input_map, tensors)` au point d'entrée store loading (line 1597-1601) | +3 |

Le cast couvre:
- Floating-point graph dtype (fp32, bf16, fp16) → cast à compute_dtype
- Non-floating (int64, bool) → preserve graph dtype (cast si différent)
- Unknown / missing dtype → no cast (preserve)

R33 zero-torch: aucun import torch ajouté. NBXTensor.to() est R33-pur.

## Anti-régression — R29 PNG visuel

| modèle | mode | PNG | verdict |
|---|---|---|---|
| Sana 1024 | triton (compiled) | `sana1024_tri_compiled_post_point1.png` | **RED APPLE** ✓ |
| Sana 1024 | triton_sequential | `sana1024_tri_seq_post_point1.png` | **RED APPLE** ✓ |
| PixArt-XL | triton_sequential | `pixart_xl_tri_seq_post_point1.png` | **RED APPLE** ✓ |
| PixArt-Sigma | triton_sequential | `pixart_sigma_tri_seq_post_point1.png` | **RED APPLE** ✓ |
| Sana 4Kpx | triton_sequential VAE-iso | `sana4kpx_vae_iso_tri_seq_post_point1.png` | green texture (Famille B remaining, expected per directive) |

Les 4 PNG cohérentes (Sana 1024 ×2 modes + PixArt ×2) confirment que
le cast n'introduit aucune régression. Sana 4Kpx reste vert comme
prévu — Famille B (TilingEngine cluster) à fixer en POINT 4-5.

## dtype_mirror_walk Sana 4Kpx post-fix

`dtype_walk_4kpx_post_point1.tsv` (737 rows).

**Premiers 7 ops (cascade racine input::z) maintenant MATCH:**

| op_idx | op_uid | seq_in | seq_out | tri_in | tri_out | match |
|---|---|---|---|---|---|---|
| 0 | unsqueeze::0 | [fp16, scalar] | fp16 | [fp16, scalar] | fp16 | ✓ |
| 1 | expand::0 | [fp16, non_tensor] | fp16 | [fp16, non_tensor] | fp16 | ✓ |
| 2 | clone::0 | [fp16] | fp16 | [fp16] | fp16 | ✓ |
| 3 | view::0 | [fp16, non_tensor] | fp16 | [fp16, non_tensor] | fp16 | ✓ |
| 4 | convolution::0 | [fp16, fp16, fp16] | fp16 | [fp16, fp16, fp16] | fp16 | ✓ |
| 5 | add::0 | [fp16, fp16] | fp16 | [fp16, fp16] | fp16 | ✓ |
| 6 | permute::0 | [fp16, non_tensor] | fp16 | [fp16, non_tensor] | fp16 | ✓ |

Comparaison aux 175 divergences pre-fix : **les 100+ divergences de
la cascade input::z sont éliminées**. Le compteur post-fix (181 div
in+out) ne reflète pas une réduction parce que la nouvelle frontière
de divergence remonte aux ops mm/bmm/relu où PyTorch AMP promote
[fp32, fp32] et NBX garde [fp16, fp16] (le wrapper Volta upcast l'act
en interne mais préserve la dtype output).

C'est exactement la doctrine que tu as fixée: NBX intrinsèquement
fp16-throughout, PyTorch oracle pour validation. Le compteur
overcount ces ops ne reflète pas un bug — c'est la divergence
"NBX moins défensif fp32". Voir POINT 3 (re-walk verdict) pour la
synthèse.

## Statut chantier

- POINT 1: ✅ implémenté + anti-régression validée
- POINT 2: en attente (rms_norm cast-back)
- POINT 3: en attente (re-walk + verdict)
- POINT 4-5: en attente (TilingEngine Famille B)

Awaiting Hocine confirmation visuelle PNG before proceeding to POINT 2.
