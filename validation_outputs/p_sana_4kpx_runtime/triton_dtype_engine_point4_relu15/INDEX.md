# POINT 4 — Audit value-level cluster racine relu::15
## P-SANA-4KPX-RUNTIME / TritonDtypeEngine maturation phase 4/5

R29 inspectable artefact pour POINT 4 — audit value-level focused
sur op 519 `aten.relu::15` (TOP cross-variant rel_ratio 2.88M× et
première racine chronologique dans value_divergence_4kpx.tsv).

## ÉTAPE A — capture inputs/outputs relu::15 dans deux modes

Outil: `capture_relu15_both_modes.py` (op_uid_interceptor sur
relu::15, dispatch mode-correct, captures input + output CPU).

**Verdict ÉTAPE A**:

| tenseur | seq dtype | seq max_abs | tri dtype | tri max_abs | rel | max_d | frac > 1e-3 |
|---|---|---|---|---|---|---|---|
| INPUT | fp32 | 47.28 | fp16 | 54.38 | **13.05%** | **55.07** | **99.98%** |
| OUTPUT | fp32 | 32.75 | fp16 | 33.56 | 2.41% | 15.94 | 20.50% |

**relu::15 est INNOCENT**. Les inputs divergent massivement
(rel 13%, max_d 55, 99.98% des éléments diffèrent en plus avec
mismatch dtype fp32 vs fp16). L'output diverge moins que l'input
(relu écrête le négatif, lissant les différences) — comportement
attendu du wrapper relu sur input divergent.

Cause amont, **walk-back requis**.

## ÉTAPE A walk-back — chronological scan ops 0..500

Outil: `walkback_minimal.py` (monkey-patch dispatchers seq+tri,
capture max_abs + 5-position fingerprint par op, no full-tensor
storage pour fitter en RAM).

**Premières divergences VALUE-level (compute ops, rel > 5%)**:

```
op_idx 25  aten.split::0       seq=79.79 tri=102.8 rel=22.4%
op_idx 47  aten.slice::5       seq=8.27e7 tri=1.81e8 rel=54.3%
op_idx 93  aten.split::2       rel=23.6%
op_idx 115 aten.slice::11      rel=19.9%
op_idx 161 aten.split::4       rel=17.5%
...
op_idx 471 aten.slice::41      rel=40.1%
op_idx 491 aten.mul::32        rel-fp 2.2% (compute, not metadata)
op_idx 492 aten.convolution::37 rel-fp 7.1% (compute)
op_idx 519 aten.relu::15       rel-fp 13% (où la cible POINT 4 commence)
```

## Pattern observé

1. Les **divergences max_abs au split/slice** apparaissent dès op 25
   et reviennent périodiquement (toutes les ~50-70 ops). Pattern
   indique: split sur un tensor where seq vs tri ont déjà des
   contenus légèrement différents → chunks extraits avec max_abs
   différents (le pic d'un chunk dépend de quelle moitié/partie
   on prend).

2. Les **divergences rel-fp** au compute-level (mul::32, conv::37)
   sont plus modestes (~2-7%) avant de cascade en direction de
   relu::15 où l'input arrive à 13% rel.

3. Les **divergences fp_max_d** sur les ops metadata (permute,
   slice, view) sont des **artefacts de fingerprint**. Mes
   fingerprints lisent à des positions absolues du flat tensor;
   après permute, les valeurs aux mêmes positions absolues
   correspondent à des éléments différents du tensor original.
   Les max_abs restent IDENTIQUES (rel < 0.001). Pas une vraie
   divergence value-level.

## Hypothèse de cause numérique partielle

Les divergences max_abs au split/slice (op 25 et suivants)
suggèrent que les inputs aux split sont déjà divergents en
**contenu** entre modes même si le **max_abs global** matche.
La vraie première op où seq et tri produisent des outputs avec
max_abs significativement différents (pas juste fingerprint
artifact) est:

**op_idx 25 `aten.split::0`** — seq=79.79, tri=102.8, rel=22.4%

**Ses inputs** (op 23 cat::1) ont max_abs 102.8 dans les deux
modes (rel=0). Donc split prend des chunks différents → seq's
chunk[0] a max_abs 79.79, tri's chunk[0] a max_abs 102.8.

Cela peut être expliqué par:
- (A) Les CONTENUS du cat::1 output ont des valeurs disposées
  différemment entre modes, malgré max_abs égal au global. Le
  chunk[0] (premier chunk) capture la moitié-tête du tensor qui
  a des valeurs distribuées différemment.
- (B) Le split lui-même opère différemment (chunk size,
  ordering) entre torch.ops.aten.split (seq) et la version
  NBX (tri).

L'hypothèse (A) est plus probable: le cat::1 input
(mm::0+mm::1+mm::2 outputs concaténés) a des valeurs qui
dépendent des accumulations matmul. Si NBX et torch matmul
produisent des micro-différences (sub-ULP) à des positions
différentes, le résultat concaténé a même max_abs global mais
des distributions par-chunk différentes.

## Limitations partielles de POINT 4

Time/scope budget atteint. Findings partiels:
- **relu::15 est INNOCENT**, root cause amont confirmé
- Walk-back identifie premier diff value-level réel à op 25 split::0
- Le mécanisme exact (A vs B) nécessite capture full-tensor à
  op 23 cat::1 et split::0 pour confirmer

## Avant POINT 5 — réflexion sur la racine

Les divergences à op 25 split::0 et op 47 slice::5 sont probablement
**COMMUNE 1024** (le pattern split/cat existe dans les deux
variants, et les max_abs diffèrent dans les deux). Mon walk-back
ne fait pas le cross-variant; pour confirmer COMMUNE il faudrait
un walk Sana 1024 équivalent et comparaison.

Si les premiers diff ops sont COMMUNE 1024, alors **les divergences
value-level fondamentales sont elles aussi non-causales du garbage
4Kpx** (puisque 1024 marche). La cascade aboutit à relu::15 mais
n'est pas la SOURCE du défaut spécifique au shape 4Kpx.

Dans ce scénario, le défaut spécifique 4Kpx serait **plus tard**:
au moment où une op subit un changement de comportement
shape-dependent (TilingEngine activation, kernel choice, etc.) qui
ajoute une nouvelle divergence par-dessus la baseline COMMUNE.

## Fichiers livrés

- `INDEX.md` (ce verdict partiel)
- `capture_relu15_both_modes.py` (ÉTAPE A capture)
- `walkback_minimal.py` (walk-back chronological 500 ops)
- `walkback_chain.txt` (extrait des STRONG divergences)

## Awaiting Hocine direction

POINT 4 partiel demande arbitrage:

**Option A** — Accepter l'innocence relu::15 confirmée et pivoter
vers POINT 4-bis: cross-variant walk-back pour distinguer
COMMUNE_1024 des divergences 4Kpx-spécifiques, et identifier la
PREMIÈRE op 4Kpx-spécifique value-divergente (probablement plus
tard dans la chaîne).

**Option B** — Continuer POINT 4 strict sur relu::15 chain en
remontant à op 25 split::0 / cat::1 pour examiner les outputs
contenu-par-contenu et confirmer hypothèse (A) vs (B). Implique
capture full-tensor sur cat::1 (~67MB) et split::0 chunks.

**Option C** — Sauter directement à un cluster value-divergent
4Kpx-spécifique du POINT 3 (e.g., cluster 642..653 conv::49 +
silu::18 + conv::50 + rms_norm::18 + add::68) qui est par
construction 4Kpx-spécifique (rel_4kpx > 10%, rel_1024 < 5%) et
donc directement causal du résiduel garbage 4Kpx.

Recommandation: **Option C**. La cluster 642..653 cible directement
les ops 4Kpx-spécifiques value-divergentes. Plus économique en
walk-back vs Option A/B qui passeraient par COMMUNE_1024 ops.
</parameter>
