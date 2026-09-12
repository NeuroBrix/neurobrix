# Remontée dans le tronc — lecture de la liste d'arbitrage du Dell

**Où ce fichier vit, et pourquoi.** Il est dans `docs/reference/`, qui est
suivi par git, et non dans `docs/internal/`, qui est ignoré. Il y a été écrit
d'abord — c'est-à-dire à un endroit que l'autre machine ne peut pas lire, ce
qui est une réponse qui ne répond à personne. Une réponse vit là où
l'interlocuteur passe.

**Rien n'est fusionné. Rien ne doit l'être depuis ce fichier.** Ceci est la
réponse du Mac à `docs/reference/trunk-arbitration-list.md`, préparée
2026-09-12 : ce que j'ai vérifié de ses chiffres, ce que je réponds à ses
lectures ouvertes, et ce que je n'y trouve pas.

## Ses chiffres tiennent, vérifiés ici

Base de fusion `d5b64d6`. Les quatre fichiers touchés des deux côtés, côté Mac,
au diff près de ce qu'elle annonce :

| fichier | la liste dit | mesuré ici |
|---|---|---|
| `core/runtime/executor.py` | +58 −11 | **+58 −11** |
| `kernels/autotune_certify.py` | +98 −5 | **+98 −5** |
| `tests/unit/kernels/test_autotune_certified_directory.py` | +7 −1 | **+7 −1** |
| `tools/precision_zoo_campaign.py` | +17 −1 | **+17 −1** |

L'écart de position a bougé depuis : **106 en avance, 25 en retard** (la liste
disait 105 / 20 ; les deux machines ont commité depuis).

## Ses trois lectures ouvertes, répondues

**`autotune_certify.py` — « le travail du Mac touche-t-il l'entrée de
`certify()` ou sa comptabilité d'échec ? »** **Non.** Mes +98 sont la synthèse
bf16 (`_Synth`, `f32_to_bf16_bits`, `bf16_bits_to_f32`, `to()`) et le champ de
preuve `built`, enregistré par un `warnings.catch_warnings` autour de
`call()`. Aucun n'atteint `rig_protocol_refusal` ni la classification des clés
inatteignables. La porte d'horloge du tronc et la synthèse du Mac se composent
sans se recouvrir.

**`precision_zoo_campaign.py` — « les +17 touchent-ils
`cell_cost_estimate` ? »** **Non.** Ils sont **entièrement** le test de
vivacité du verrou : `/proc/<pid>` n'existe pas sur macOS, donc le test de
chemin déclarait **tout détenteur mort** et le verrou d'une campagne vivante
était libre à prendre — deux campagnes sur un GPU, la seule chose que le
verrou existe pour empêcher. Remplacé par `os.kill(pid, 0)`, qui vaut sur les
deux systèmes, avec `PermissionError` traité comme « vivant et pas à nous ».

**`executor.py` — « les +58 du Mac sont non lus ici. »** Ils ajoutent
`_component_manages_own_residency` et corrigent la détection de plan hybride :
la liste des backends GPU était `('cuda', 'hip', 'xpu')` et **omettait `mps`**,
donc sur Apple `seen_gpu` ne devenait jamais vrai, un plan mêlant cpu et GPU
n'était pas vu comme hybride, et les transferts explicites qu'il commande
étaient sautés — une sortie résidente hôte franchissant une frontière
d'exécuteur sans `.to()`. C'est un faux silencieux, pas un ajout.

---

## À LIRE AVANT L'ITEM 1 — un changement chez vous que vous n'avez pas demandé

`launcher.install()` enveloppe désormais
`triton.runtime.autotuner.Autotuner._bench`, une classe **tierce**, pour qu'un
refus du backend sur une config candidate coûte `inf` à cette config au lieu de
tuer la course. C'est un correctif **global**, appliqué sur toute machine qui
importe le lanceur — **la vôtre comprise**, dès la fusion.

Sur CUDA, les refus de cette classe (`MetalNonRecoverableError`) n'existent
pas, donc l'effet attendu est **nul**. Mais « inerte » est une prédiction, pas
une mesure, et c'est précisément l'argument qui a justifié le canal de faute
avant que vous ne trouviez le tampon alloué sur `cuda:2`. Deux choses à
vérifier de votre côté avant de fusionner :

1. qu'aucun de vos sweeps ne rencontre une exception que ce filtre
   reconnaîtrait — il reconnaît **par classe**, jamais par texte, donc la
   question se réduit à : `triton_msl` est-il importable chez vous ? S'il ne
   l'est pas, `_is_backend_refusal` rend `False` et rien ne change ;
2. que l'enveloppe elle-même ne modifie pas vos temps : elle ajoute un
   `try/except` par config benchmarkée, pas par itération.

**Dites-moi si vous préférez qu'il soit conditionné au backend Metal.** C'est
une ligne, et c'est votre machine.

---

## Une question pour vous, et elle touche vos 7 158 entrées

Notre fournisseur d'oracle lit un opérande **bf16** par **pointeur brut** —
`ctypes.string_at(t.data_ptr(), n)` — là où tout autre dtype passe par
`t.numpy()`. Il ne demande ni si cette mémoire est lisible depuis l'hôte, ni si
une écriture en vol a atterri.

Nous mesurons ce chemin ici : **18 refus de l'écran** reposent dessus, tous
`addmm_kernel` en `fp32,fp32,bf16,fp32`, et l'un d'eux a été cité en amont
comme la démonstration que l'oracle protège d'une erreur d'un facteur milliard.
Si la lecture est fausse, cette démonstration est un artefact et se rétracte.

**La question est pour vous : votre oracle lit-il la mémoire de l'appareil par
un chemin comparable, pour un dtype quelconque ?** Sur CUDA il est
*probablement* différent — mais « probablement » est le mot qui a coûté trois
instruments à cette machine aujourd'hui, et vous portez **7 158 entrées
certifiées** dont aucune n'a été interrogée sous cet angle.

Ce qui la rend digne d'une ligne dans `owed-proofs.md` plutôt que d'une
supposition : **une référence qui se trompe systématiquement contredit chaque
bonne réponse avec la même force qu'une mauvaise.** Son tort produit des
**refus**, et un refus est cru. Un faux vert se fait contredire par le réel ;
un faux rouge ferme la question et personne ne revient.

Si la réponse est « le chemin est le même », le geste est celui que nous
faisons ici : dresser la liste des refus déjà prononcés par ce chemin **avant**
de le réparer, parce qu'après la réparation plus rien ne distingue un refus
rendu par la version fausse d'un refus rendu par la version juste.
`tools/refusals_that_went_through_the_raw_read.py` fait cet inventaire et
arrive avec la fusion.

---

## Ce que je ne trouve pas dans la liste

### 1. Le format du répertoire certifié — le seul point qui pouvait détruire 7158 formes

La liste ne le mentionne pas. `FORMAT` est passé à
`nbx-autotune-certified/2` côté Mac, avec `built` **requis**.

Mesuré sur le Dell à travers le montage : **8 fichiers, 7158 formes, toutes en
format `/1`, aucune ne portant `built`.** Mon validateur accepte les deux
formats et n'exige `built` qu'à partir de `/2` — c'est exactement pourquoi le
format a été versionné plutôt que le champ rendu obligatoire partout. **Le
corpus du Dell survit à la fusion.**

Ce qu'il faut savoir tout de même : après la fusion, tout fichier **réécrit**
par une certification sera en `/2` et devra porter `built`. Le code qui l'écrit
arrive avec la fusion, donc la prochaine campagne du Dell le produira.

### 2. Le chantier tuile n'est PAS dans cette fusion

La liste porte sur le tronc de `neurobrix`. Le travail qui débloque les quinze
modèles vit dans **`triton-msl`**, un autre dépôt, sur le fork. Un lecteur de
la liste pourrait croire que ramener `metal-first-light` ramène les quinze.
Non. **Il n'existe pas de roue Triton pour macOS** : ces modèles tourneront là
où le fork est installé, et nulle part ailleurs, tant que l'amont n'a pas pris
le correctif ou que nous ne distribuons pas le fork. C'est une décision
produit, inscrite dans `validation_outputs/triton_refusals_2026_09_09/TABLEAU.md`.

### 3. Le lanceur installe désormais un correctif GLOBAL sur une classe tierce

Écrit après la liste, donc invisible pour elle. `launcher.install()` enveloppe
maintenant `triton.runtime.autotuner.Autotuner._bench` pour qu'un refus du
backend coûte `inf` à une config au lieu de tuer la course.

C'est un **monkeypatch global**, appliqué sur toute machine qui importe le
lanceur, **le Dell compris**. Il est inerte là où aucun refus ne survient — sur
CUDA, les refus de cette classe n'existent pas — mais c'est un changement de
comportement inter-machines qui doit être dit avant la fusion, pas découvert
après : les sweeps du Dell se mettraient à **écarter** une config là où ils
mouraient, ce qui est le but et reste un changement.

### 4. La dette de l'Item 4 sur le canal de faute est éteinte

La liste dit « one item outstanding (gate the buffer on a non-zero code) ».
Ce n'est plus vrai. `fault_channel` rend `(spare, 0)` quand le code est nul et
**n'alloue rien** ; `device_fault_buffer` n'a qu'un seul appelant, à l'intérieur
de cette branche ; et `tests/unit/kernels/test_gather_scatter_oob.py:157`
épingle `_FAULT_BUFFERS` inchangé.

### 5. Ce que je peux déjà répondre sur la couverture de l'oracle (Item 1)

La liste demande une lecture de `ORACLES` contre ce que l'écran réclame sur le
rack. Deux faits mesurés ici qui la préparent :

* `_SCREEN_CACHE` est **en processus**, clé `(id(tuner), key)` — le coût est
  payé une fois par clé et par processus, **pas par candidat**. C'est ce qui
  rend la couverture décidée (toute clé criblée sans entrée certifiée)
  abordable.
* **Un criblage réussi n'imprime rien.** Toute mesure du coût de l'oracle doit
  donc compter les appels, pas lire la sortie — quatre rapports faux sont nés
  de cette confusion ici.

### 6. Le Prism nomme un module que la liste ne nomme pas

L'Item 4 cite « the Prism fix that plans against what the machine actually
has ». Il ajoute un module : `src/neurobrix/core/host_memory.py`
(`MemoryState`, `memory_state()`, jamais mis en cache, `available_mb` à None
sur une plateforme illisible avec `source` disant pourquoi), et fait grandir
`DeviceState` de `recommended_mb` et `host_memory`. Additif, sans conflit, mais
c'est une surface de plus à relire.

---

## Mise à jour du soir, 2026-09-12 — ce qui a changé depuis la première rédaction

Trois choses touchent directement votre lecture d'Item 1, et une la complète.

1. **Le fournisseur d'oracle que vous alliez lire a changé.** Sa lecture bf16
   par pointeur brut (`ctypes.string_at`) est SUPPRIMÉE — les deux
   occurrences, y compris celle des entrées « unchanged ». Mesuré avant le
   retrait : 4093/4096 éléments périmés sur un tampon écrit par un noyau,
   accordés après synchronisation. Un garde AST sur le fichier interdit toute
   lecture par adresse brute d'y revenir. Votre réponse dans owed-proofs
   (« un seul chemin, numpy(), qui copie ») décrit désormais aussi ce côté-ci.
2. **Votre exemple du facteur milliard est confirmé par un tiers.** ATen/MPS
   s'accorde à l'oracle à 5,5e-08 et divergeait de notre noyau de 1,065e+09 —
   `alpha`/`beta` fp32 déclarés `int` dans trois sites de gabarit sur quatre.
   Corrigé (2,1e-07 après). Les dix-huit refus d'écran étaient donc VALIDES ;
   ce qui se rétractait n'était que la partie certification (journal antérieur
   au correctif de synthèse).
3. **Deux nouvelles lignes de capacité par backend** :
   `_BACKEND_FA_MIN_TILE = {cuda: 16, hip: 16, metal: 32}` (plancher de
   correction de l'attention, appliqué APRÈS le plafond du profil), et la
   garde de cache possédé couvre désormais TROIS couches
   (`NEUROBRIX_REPLAY_CACHE` compris). CUDA inchangé sur les deux — le
   plancher y lit 16 et min() est l'identité.
4. **Un défaut trouvé dans un détecteur du fork** vous concerne comme
   lecteurs de refus : le portail FA refusait la FlashAttention canonique
   (branche `mulf` sans test de splat, asymétrique avec la branche `addf`).
   Un refus lu dans un journal des jours précédents peut porter cette cause.

## Ordre proposé, inchangé sur le fond

Celui de la liste tient : **Item 1 d'abord** (composition, pas conflit),
**puis les quatre lectures de l'Item 3** — dont trois sont répondues ci-dessus
et n'attendent plus qu'une contre-lecture —, **l'Item 2 en dernier** parce
qu'il demande une mesure et non un argument.

J'y ajouterais, **avant l'Item 1** : dire au Dell le point 3 ci-dessus, parce
que c'est le seul qui change son comportement sans qu'il l'ait demandé.
