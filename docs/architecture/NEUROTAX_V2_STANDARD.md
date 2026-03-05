# NeuroBrix Nomenclature V2.1 — Standard de Nommage Universel des Tenseurs IA

## 1. Objectif du standard

V2.1 définit un système universel, cohérent et lisible pour nommer les tenseurs dans les modèles d’intelligence artificielle, toutes architectures confondues (LLM, Diffusion, Vision Transformers, AudioLM, VideoML, modèles multimodaux, etc.).

L’objectif est de :
*   Standardiser le langage utilisé par les ingénieurs, les chercheurs, et les runtime engines.
*   Rendre la lecture des logs, profils mémoire et dumps de tenseurs immédiate et intuitive.
*   Créer une taxonomie inspirée du cerveau biologique (sensory, internal, modulatory, motor).
*   Garantir l'unicité des noms même dans les architectures complexes (MoE, SwiGLU, U-Net).

---

## 2. Structure du nommage

Un nom de tenseur suit systématiquement la structure :

`BIO.MODALITY.FUNCTION.POSITION`

### Le concept d'Identité Intrinsèque
Un tenseur doit être nommé selon **ce qu'il est**, pas uniquement selon l'endroit où il se trouve. Cependant, le contexte d'utilisation (pipeline) peut influencer la couche `BIO` (voir section 3.1).

---

## 3. Couche BIO — Rôle biologique

Les quatre familles s’inspirent des grandes classes neuronales du cerveau.

| Nom | Rôle | Description |
| :--- | :--- | :--- |
| **sensory** | Entrée | Données brutes ou embeddings d’entrée (yeux, oreilles du modèle). |
| **internal** | Calcul | Représentations internes, attention, raisonnement, FFN (cortex). |
| **modulatory** | Modulation | Informations de contrôle : position, temps, conditioning, instructions (hippocampe/thalamus). |
| **motor** | Sortie | Logits, génération finale, reconstruction (cortex moteur). |

### 3.1 Règle de Relativité (Le "Context Switch")
La classification `sensory` vs `modulatory` dépend de l'architecture globale :
*   **Dans un LLM (ex: Llama) :** L'encodeur de texte est `sensory` (c'est le flux principal).
*   **Dans un T2I (ex: PixArt/SD) :** L'encodeur de texte est `modulatory` (il guide la génération visuelle, qui est le flux principal).

---

## 4. Couche MODALITY — Modalité sensorielle

Définit la nature du signal traité.

| Nom | Domaine |
| :--- | :--- |
| **text** | Tokens, embeddings textuels. |
| **vision** | Images, latents 2D, patches visuels. |
| **audio** | Spectrogrammes, formes d’onde. |
| **video** | Patches spatio-temporels. |
| **latent** | Latents de diffusion purs (timestep, bruit) ou représentations abstraites compressées. |
| **multimodal** | Fusion cross-modalité (ex: attention visuelle + textuelle). |
| **generic** | Pour les composants agnostiques (ex: LayerNorm générique). |

---

## 5. Couche FUNCTION — Fonction opérationnelle (Atomicité & Synonymes)

Cette couche définit l'opération mathématique. V2.1 introduit l'**Atomicité** pour éviter les collisions (ex: FFN vs Gate/Up/Down).

### 5.1 Le Registre des Synonymes (Mapping Standard)
Tout terme propriétaire (Vendor) doit être mappé vers un terme standard.
*   `MLP`, `Dense`, `Linear` → **FFN** ou **PROJ**
*   `TransformerBlock`, `EncoderLayer` → **INTERNAL** (implicite par la couche BIO)

### 5.2 Catalogue des Fonctions

**Embeddings & Encodage**
*   `embed` : Embedding principal.
*   `pos` : Encodage positionnel.
*   `rope` : Rotary Positional Embedding.
*   `time` : Timestep embedding (Diffusion).
*   `patch` : Patch embedding (ViT).

**Attention**
*   `attn` : Module d'attention global.
*   `q`, `k`, `v` : Projections Query, Key, Value atomiques.
*   `qkv` : Projection fusionnée.
*   `out` : Projection de sortie de l'attention.

**Feed-Forward (FFN/MLP)**
*   `ffn` : Bloc FFN générique (si monolithique).
*   `ffn_gate` : Projection de "Gating" (SwiGLU, GeGLU).
*   `ffn_up` : Projection ascendante (dim model -> dim hidden).
*   `ffn_down` : Projection descendante (dim hidden -> dim model).

**Normalisation & Modulation**
*   `norm` : LayerNorm, RMSNorm, GroupNorm.
*   `cond` : Conditioning projection (AdaLayerNorm, FiLM).
*   `scale`, `shift` : Paramètres de modulation affines.

**Sortie**
*   `logits` : Prédiction de tokens.
*   `recon` : Reconstruction d'image/audio (pixels/spectro).
*   `head` : Tête de classification ou de décodage.

---

## 6. Couche POSITION — Topologie Hiérarchique

V2.1 abandonne la linéarité stricte (`layerN`) pour supporter les architectures complexes (U-Net, MoE). La position doit refléter la topologie du graphe.

### 6.1 Format Standard
`[STAGE].[BLOCK].[INDEX]`

*   **STAGE (Optionnel) :** Pour les U-Nets ou modèles multi-échelles.
    *   `down`, `mid`, `up` (ex: `down_block2`).
    *   `stage1`, `stage2` (ex: ResNet).
*   **BLOCK (Recommandé) :** L'unité répétitive.
    *   `layerN` (ex: `layer12`).
    *   `blockN`.
*   **INDEX (Sub-Location) :** Pour les composants internes au bloc.
    *   `headN` (Tête d'attention).
    *   `expertN` (Mixture of Experts).

### 6.2 Exemples de Positions
*   Transformer simple : `layer12`
*   U-Net (SDXL) : `down_block1.layer0`
*   MoE (Mixtral) : `layer4.expert2`

---

## 7. Exemples Concrets (V2.1)

### 7.1. Modèle LLM Moderne (Llama 3 / Mistral) - SwiGLU
*   `sensory.text.embed.layer0` (Embedding)
*   `internal.text.attn.layer4` (Bloc Attention)
*   `internal.text.q.layer4` (Matrice Query)
*   `internal.text.ffn_gate.layer4` (SwiGLU Gate)
*   `internal.text.ffn_down.layer4` (SwiGLU Down Proj)
*   `internal.text.norm.layer4` (RMSNorm)
*   `motor.text.logits.final` (Sortie)

### 7.2. Modèle Diffusion Transformer (PixArt / DiT)
*   `sensory.latent.embed.layer0` (Patch Embed des latents)
*   `modulatory.text.embed.layer0` (T5 Encoder - Context)
*   `modulatory.latent.time.layer0` (Timestep Embed)
*   `internal.vision.attn.layer9` (Self-Attention)
*   `internal.multimodal.attn.layer9` (Cross-Attention Text/Image)
*   `internal.vision.ffn.layer9` (MLP classique)
*   `motor.vision.recon.final` (Reconstruction finale)

### 7.3. Modèle U-Net (Stable Diffusion XL)
*   `internal.vision.attn.down_block1.layer0` (Attention dans la descente)
*   `internal.vision.ffn.mid_block.layer0` (FFN au goulot d'étranglement)
*   `internal.vision.conv.up_block2.layer1` (Convolution remontante)
*   `modulatory.text.proj.down_block1` (Injection du prompt)

---

## 8. Règles d'Implémentation (Pour les Parsers)

1.  **Priorité au Synonym Registry :** Toujours consulter le dictionnaire de synonymes avant de tenter une déduction heuristique.
2.  **Défensivité :** Si un terme est inconnu, le marquer `unknown` mais NE PAS CRASHER. Logger l'anomalie.
3.  **Atomicité FFN :** Si le parser détecte `w1`, `w2`, `w3` (conventions Llama), il doit les mapper respectivement vers `gate`, `up`, `down`.
4.  **Conservation Topologique :** Ne pas écraser la structure `down/up` des U-Nets en une liste plate `0..N`.