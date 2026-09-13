# Un nombre qui nomme une quantité doit nommer son dénominateur

Trois fois en une semaine, deux dénominateurs portant le même nom nous ont
coûté une décision. Ce n'est plus une distraction, c'est une classe.

| la quantité | les deux dénominateurs | ce que ça a coûté |
|---|---|---|
| « les modèles du hub » | **56** artefacts mesurables depuis ce poste / **47** modèles du catalogue | deux tableaux posés côte à côte qui ne comptaient pas la même population |
| « les entrées certifiées » | **971** / **1 509** | une comparaison entre deux machines qui ne portait pas sur la même chose |
| « la mémoire libre » | **0,1 Go** (`Pages free`) / **9,4 Go** (`available_mb`) | une attente décidée sur le mauvais motif |

## La règle

**Un nombre qui nomme une quantité nomme son dénominateur, dans la même
phrase.** « 22 tournent » ne veut rien dire ; « 22 des 56 artefacts mesurables
depuis ce poste » en veut un.

## Et le corollaire qui tranche quand deux sources existent

**Quand un de nos propres modules calcule cette quantité, c'est LUI l'autorité,
pas l'outil système.**

Un module à nous porte la question dans sa définition. `vm_stat` répond à la
sienne, qui n'est pas la nôtre : `Pages free` sur macOS est presque toujours
minuscule parce que le système ne garde pas de pages libres — l'espace réel est
dans l'inactif et le purgeable. Lire `Pages free` et l'appeler « la mémoire
libre », c'est emprunter le mot d'un outil pour une question qu'il ne pose pas.

`core/host_memory.py` calcule `available_mb`, et c'est le nombre contre lequel
Prism planifie. C'est donc celui qui décide, et c'est celui qu'on cite.

**Deuxième piège du même endroit :** la taille de page de cette machine est
**16 384 octets**, pas 4 096. Un calcul en pages qui suppose 4 Ko se trompe
d'un facteur quatre en plus de se tromper de question. Un module qui lit la
page depuis `sysctl` ne peut pas faire cette erreur ; un calcul à la main la
fait chaque fois.

## La conduite

1. **Écrire le dénominateur avec le numérateur.** Toujours, y compris dans une
   phrase de passage.
2. **Préférer notre module à l'outil système** quand les deux existent, et dire
   lequel on a lu.
3. **Quand un nombre motive une décision, dire ce qu'il motive.** « J'attends
   faute de mémoire » et « j'attends parce que la mesure est sérialisée » se
   règlent différemment : le premier en coupant quelque chose, le second en
   attendant son tour. On ne coupe pas le travail de quelqu'un pour la mauvaise
   raison.

Ce troisième point est ce qui a failli arriver ici : la machine virtuelle
tourne, 5,3 Go et trois quarts d'un cœur, et il reste **9,4 Go disponibles**.
Ce qui faisait attendre était la **sérialisation du GPU**, pas la mémoire.
