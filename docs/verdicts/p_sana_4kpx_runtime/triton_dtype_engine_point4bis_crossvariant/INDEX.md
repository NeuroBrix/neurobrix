# POINT 4-bis — Cross-variant walkback comparison ops 0..99
## P-SANA-4KPX-RUNTIME / TritonDtypeEngine maturation phase 4-bis

R29 inspectable artefact pour POINT 4-bis — verdict cross-variant
qui tranche entre Scénario 1 (COMMUNE 1024 → pivot Option C) et
Scénario 2 (4Kpx-spécifique dès op 25 → attaquer mm::0/1/2 racine).

## Méthode

`walkback_minimal_v2.py` paramétré sur MODEL env. Capture `max_abs`
+ 5-position fingerprint pour chaque op dispatchée (monkey-patch
`NativeATenDispatcher.dispatch` + `TritonSequentialDispatcher.dispatch`),
limité aux premières 100 ops par mode. Synthetic Gaussian fp32
latent pour Sana 1024 (shape (1, 32, 32, 32)), saved latent pour
Sana 4Kpx.

`compare_walkbacks.py` cross-référence rel_max_abs par op_idx entre
les deux variants. Verdict per op:

| verdict | critère |
|---|---|
| OK_BOTH | rel_4kpx < 0.005 AND rel_1024 < 0.005 |
| 4KPX_NEW | rel_4kpx > 0.05 AND rel_1024 < 0.01 |
| 4KPX_AMPLIFIED | rel_4kpx > 3 × rel_1024 AND rel_4kpx > 0.05 |
| COMMUNE_BASELINE | rel_4kpx ≈ rel_1024 |
| MIXED | autre |

## Verdict factuel — distribution

| verdict | count | % |
|---|---|---|
| OK_BOTH | 96 | 96% |
| COMMUNE_BASELINE | 4 | 4% |
| **4KPX_NEW** | **0** | **0%** |
| **4KPX_AMPLIFIED** | **0** | **0%** |

**ZÉRO divergence 4Kpx-spécifique dans ops 0..99**. Confirme
**SCÉNARIO 1**.

## 4 ops COMMUNE_BASELINE

| op_idx | op_uid | rel_4kpx | rel_1024 | observation |
|---|---|---|---|---|
| 1 | aten.expand::0 | 1.000 | 0.722 | Fingerprint artifact NBXTensor stride-0 broadcast (les deux variants pareils) |
| 25 | aten.split::0 | 0.224 | **0.443** | **1024 MORE divergent que 4Kpx** — pas amplification 4Kpx, baseline cohérente |
| 47 | aten.slice::5 | 0.543 | 0.563 | Quasi-identique (rel_ratio ≈ 1) |
| 93 | aten.split::2 | 0.236 | 0.286 | Quasi-identique |

Le pattern split/slice rel-divergent observé en POINT 4 walkback
4Kpx est PRÉSENT au même niveau (et plus fort à 1024 pour
op 25) dans Sana 1024. Les deux variants partagent la même
micro-numérique mm/PROMOTE_B/cat/split.

## Verdict scénario

**SCÉNARIO 1 CONFIRMÉ**: les divergences value-level early-chain
sont COMMUNE 1024 et donc non-causales du résiduel garbage 4Kpx.
Sana 1024 produit pomme rouge cohérente malgré ces mêmes divergences.

Implication: **la racine du résiduel garbage 4Kpx est shape-specific
TARDIVE**, pas dans les ops 0..99. Cohérent avec POINT 3 finding:
les 53 4Kpx-specific value divergences sont concentrées dans 13
clusters tardifs (op 519+).

## Recommendation Option C — pivot cluster 642..653

Cluster identifié au POINT 3 comme premier LARGE 4Kpx-specific
value-divergent:

```
op 642  aten.convolution::49      rel_4kpx 24.8%, rel_1024 0%
op 643  aten.silu::18             rel_4kpx 33.7%, rel_1024 0%
op 644  aten.convolution::50      rel_4kpx 26.2%, rel_1024 0%
op 645  aten.permute::72          rel_4kpx 26.2%, rel_1024 0%
op 646  custom.rms_norm::18       rel_4kpx 16.6%, rel_1024 0.1%
op 647  aten.add::68              rel_4kpx 16.6%, rel_1024 0.1%
op 648  aten.permute::73          ...
...
op 653  cluster terminus
```

Ces 11 ops sont par construction 4Kpx-spécifiques. La racine
amplificatrice est probablement à op 642 conv::49 ou op 635 conv::48
(isolé juste avant le cluster). C'est ICI que l'audit kernel
TilingEngine doit pointer.

## Files

- `INDEX.md` (ce verdict)
- `comparison_table.tsv` (100 op_idx × verdict)
- `comparison_output.txt` (terminal output)
- `walkback_1024_first100.tsv`
- `walkback_4kpx_first100.tsv`
- `walkback_minimal_v2.py` (script paramétré MODEL)

## Statut chantier

- POINT 1: ✅ ea8e8e2 (input::z cast)
- POINT 2: ✅ 331c611 (uniform AMP_FP32_OPS cast-back)
- POINT 2bis: ✅ 735e76e (registry plumbing fix)
- POINT 3: ✅ 1810307 (dtype divergence classification — 0 STRUCTURELLE_RACINE)
- POINT 4: ✅ 6224ac4 (relu::15 innocent + walk-back partial)
- POINT 4-bis: ✅ this commit (cross-variant verdict — Scénario 1 confirmé)
- **POINT 5: prêt à attaquer**, cible cluster 642..653 ou racine amplificatrice op 635 conv::48 / op 642 conv::49

Awaiting Hocine arbitrage final — POINT 5 cible exacte (cluster vs
racine isolée pré-cluster) + scope du diagnostic preliminaire avant
fix.
</parameter>
