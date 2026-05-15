# Audit des dettes résiduelles — 2026-05-16

**Mandat** : analyse pure, lecture seule. Aucun fix, aucun commit
de code, aucune fermeture de follow-up. Angle = *tout ce qu'on a
touché et pas fini, fermé en contournant, déféré silencieusement,
ou laissé dormir dans le code/les verdicts*. Complément (pas
doublon) de `docs/audit_technical_debt_2026_05_15.md`.

**Discipline** : chaque affirmation est sourcée à un `fichier:ligne`
ou un `SHA` lu pendant cet audit (8 passes : git des 2 dépôts,
follow-ups, verdicts + sous-dossiers, lessons, audits du dépôt de
packaging, grep code, tests skip/xfail, commits grep'd).

**Convention** : le dépôt de packaging privé est désigné « PKG »
et ses internes par des descripteurs neutres (contrainte de
confidentialité du dépôt public ; SHAs et numéros de ligne
conservés pour le sourcing). Un lecteur ayant accès au dépôt PKG
peut mapper les descripteurs.

**Convention REOPEN** (renfort Hocine) : items à fort potentiel
de correction / gain majeur marqués **[REOPEN-CANDIDATE]**. Ce
rapport ne corrige rien ; il hiérarchise.

---

## SECTION 1 — Dettes correctness critiques (bugs silencieux / faux, tout mode/hardware)

### 1.1 [REOPEN-CANDIDATE] `linspace` Triton retourne de la mémoire NON INITIALISÉE
- **Localisation** : `src/neurobrix/kernels/dispatch.py:336-344`. `_create_linspace` fait `out = NBXTensor.empty((steps,), ...)` puis `# TODO: proper Triton linspace kernel` / `return out`. Importe `fill_kernel` ligne 340 mais **ne l'appelle jamais**.
- **Reachabilité CONFIRMÉE** : `linspace` ∈ `_METADATA_OPS` (`src/neurobrix/kernels/classification.py:122` `"arange", "linspace",`) → `aten::linspace` routé METADATA → résolu via `_OP_MAP.get()` (`dispatch.py:742`) → `_create_linspace`.
- **Impact réel** : tout modèle dont le DAG contient `aten::linspace` (échéanciers de timesteps diffusion/vidéo, grilles positionnelles) en `--triton`/`--triton-sequential` reçoit des **valeurs aléatoires**, pas de crash, pas de NaN garanti. Sortie silencieusement fausse. Touche directement vidéo/diffusion qu'on s'apprête à attaquer.
- **Comment on est passé outre** : `# TODO` inline jamais relu ; jamais sourcé dans un verdict ; le `fill_kernel` importé donne l'illusion d'un kernel câblé.

### 1.2 [REOPEN-CANDIDATE] `index_put` / `index_put_` = lambdas identité (write silencieusement perdu)
- **Localisation** : `src/neurobrix/kernels/dispatch.py:700-701` dans `_build_op_map()` : `"index_put": lambda x, indices, values, acc=False: x,  # TODO` (idem `index_put_`). Retourne l'entrée **non modifiée** → l'écriture scatter est entièrement abandonnée.
- **Reachabilité À CONFIRMER** : `index_put` n'est PAS dans `_METADATA_OPS` (classification.py n'a que `"index"`). Mais `_OP_MAP` est la table générale nom→callable (`dispatch.py:737-742`). `aten::index_put` est référencé dans le chemin MoE-fusion v2 (`core/runtime/graph/moe_fusion.py:426`, `compiled_sequence.py:52`). **Interprétation à confirmer avec Hocine** : si un DAG résout `index_put` via `_OP_MAP` (masked scatter, écriture KV-cache indexée, MoE v2 `index_put+sum`), sortie silencieusement fausse, zéro signal.
- **Impact réel potentiel** : familles audio_llm / multimodal / LLM MoE — exactement la coverage visée. Sévérité maximale SI reachable.
- **Comment on est passé outre** : `# TODO` inline, jamais tracké en follow-up nommé.

### 1.3 Kokoro-82M `::native` — régression `aten::cudnn_batch_norm` non corrigée, cause seulement SUPPOSÉE
- **Localisation** : `docs/follow-ups/kokoro_cudnn_batch_norm_regression.md` ; xfail `tests/regression/conftest.py:135`. Symptôme : `aten.cudnn_batch_norm::0 ... sym_strides() called on an undefined Tensor`. Vert à `a64aa4b` (v0.1.5), rouge à `be5c7b8`.
- **Impact réel** : modèle TTS **régressé dans son mode par défaut** (`::native`, pas triton). Bloque la complétion TTS.
- **Comment on est passé outre** : "Narrow bisect ... was not run in this session" → marqué xfail (`e971900` : "this commit marks the test xfail"). 3 suspects rankés non confirmés (`be5c7b8` ratchet / `d4c229d` batch_norm wrapper / `22e9ff2` stride). **Pas de band-aid** : grep `cudnn_batch_norm`/`sym_strides` dans `compiled_sequence.py` = 0 hit → bug réellement ouvert, non masqué. **[REOPEN-CANDIDATE]** (gain : débloque Kokoro natif ; bisect borné à ~8 commits).

### 1.4 Bare-except qui désactive silencieusement le clamp VAE
- **Localisation** : `src/neurobrix/core/module/output_processor.py:87-88` : `except Exception: / pass  # Silently continue with defaults`. Enveloppe le lookup registre `config.clamp_before_normalize`.
- **Impact réel** : toute erreur de parse/lookup → fallback silencieux `clamp_before_normalize=False` → sortie VAE hors-plage (image fausse), aucun diagnostic. Touche image/vidéo.

### 1.5 Synthèse d'input inconnue silencieusement ignorée (viole ZERO-FALLBACK)
- **Localisation** : `src/neurobrix/core/runtime/resolution/input_synthesizer.py:248` : `pass  # Unknown synthesis method - skip silently`.
- **Impact réel** : règle de synthèse inconnue ne produit **aucun tenseur** ; composant aval consomme un slot manquant/non-init → inférence silencieusement fausse au lieu d'un crash clair. Violation directe ZERO-FALLBACK.

### 1.6 Bare-except masquant écritures cache prefetch / provenance conteneur
- `src/neurobrix/core/io/memory.py:262` : `except:` nu (attrape même KeyboardInterrupt/SystemExit) autour de `queue.put_nowait`+écriture cache → composant silencieusement absent du cache prefetch, servi périmé/non-init sur le warm path.
- `PKG/importer/builder.py:607` : `except: pass` autour de l'extraction regex vendor/URL → provenance (`origin.vendor/url`) silencieusement absente du conteneur immuable (perte de donnée au build, irrécupérable au runtime).

---

## SECTION 2 — Dettes correctness LATENTES (cassent sur un autre target/config)

| Item | Localisation | Déclencheur |
|---|---|---|
| Dropout training non implémenté (raise) | `kernels/wrappers.py:2076,2084` | mode entraînement Triton — inference identité OK, raise sinon (fail-fast) |
| `scatter_reduce` mode non implémenté (raise) | `wrappers.py:3722` | reduce-mode non câblé en `--triton` |
| `pad` mode non implémenté (raise) | `wrappers.py:4825` | mode pad non standard `--triton` |
| `interpolate` mode/ndim non implémenté (raise) | `wrappers.py:5604` | interpolation non standard `--triton` |
| `complex(real,imag)` TODO | `wrappers.py:5579` | tenseurs complexes (audio/STFT) `--triton` — **pertinent TTS/STT à venir** |
| `full`/`ones` non-zero fill TODO | `kernels/nbx_tensor.py:1847,1945` | branche fill scalaire non-zéro via ce chemin |
| op-level tiling non-conv non implémenté | `core/prism/solver.py:915` `# only conv tiling implemented for now` | configs tiling non-conv |
| triton-CPU fallback gap | `solver.py:2736` (TODO) | runtime triton-CPU absent (R25 escalade upstream) |
| Kokoro `aten::lstm` / VibeVoice DDPM decoder | `core/flow/stages/kokoro.py:398`, `stages/vibevoice.py:282` | builds nécessitant retrace décodeur (R33-temporaire) |
| `_set_device` historiquement absent 163/211 sites | SHA `8854dee` "163 missing guards ... latently broken: bmm, addmm, softmax, rms_norm ... every RNG wrapper" | multi-GPU silencieusement cassé avant ce commit — **vérifier qu'aucun site n'a re-régressé** |
| `requires_fp32_compute` = opt-in manuel, PAS auto-detect famille | `solver.py:2936,2978` `_components_force_fp32` | PixArt VAE redescend fp16 sur V100 par défaut (overflow→±inf) sauf flag manuel — **latent image/vidéo non-flaggée** |
| `_transfer_tensor` strides (corrigé `22e9ff2`) | SHA `22e9ff2` "computed act @ W instead of act @ W.t() ... no crash, no NaN" | Qwen3-30B zero3 triton silencieusement faux — corrigé ; garder comme classe de bug |

---

## SECTION 3 — Dettes architecturales (parité native↔triton, optimisations déférées, Option-B-avec-TODO)

### 3.1 [REOPEN-CANDIDATE] Symbolisation des dims spatiales/temporelles — dette de fond récurrente sur 13 mois
- **Sources** : leçon PKG « VQ-autoregressive trace fix » §7:173 ("seq_len non symbolique — graph a seq_len=704 en dur", priorité Moyenne) ; doc PKG « vision symbolic-shapes » §6 **MANDATORY** "Every input dimension that CAN vary MUST be symbolic" → rendre le module de propagation symbolique `_infer_dim_names()` *family-aware*.
- **Gap vs code** : Fix1 (slice negative-end symbolique, PKG `symbolic/rules.py:2261,2318-2336`) et Fix2 (SymInt const-fold, PKG `symbolic/shapes.py:196-221`) **FAITS**. Mais le §6 *mandatory* (le vrai correctif de coverage) **PAS FAIT** : PKG `symbolic/tracker.py:555 _infer_dim_names()` reste l'heuristique nom-pattern (prend `input_name`, jamais `family`). Conséquence auto-documentée : VAEs image 0-20 %, encodeurs audio 0-1 %, vision 0 % de coverage symbolique. **C'est exactement BL-1** (containers `profile.json` sans `upscale`/`window_size`) et la cause profonde du blocage tiling arbitraire. Gain système majeur (vidéo/audio/multimodal ont des dims variables).

### 3.2 Lessons 002 / 004 = doctrine PÉRIMÉE vs code (DtypeEngine)
- `docs/lessons/002-dtype-engine-prism-master.md:89-95,109` prescrit "PAS de AMP_FP32_OPS / _make_fp32_wrapper / amp_cast_inputs". Code actuel `core/dtype/engine.py:79,227,274,455,470` ré-introduit exactement ça.
- `docs/lessons/004-fp16-matmul-overflow-inf-fix.md:147-151` déclare `_make_inf_fix_wrapper` ✅ / `_make_fp32_wrapper` ❌ ; règle "never clamp". Code : pas de `_make_inf_fix_wrapper`, garde overflow = `inp.clamp(-65504,65504).to(dtype)` (`engine.py:420,446`) — le `clamp()` que la lesson interdit.
- **Contradiction doctrine/code** dangereuse avant d'ajouter des modèles fp16-sensibles. **Interprétation à confirmer avec Hocine** : laquelle est la doctrine courante ?

### 3.3 Parité R30 native↔triton — gaps déférés
- HAT triton/triton-seq bloqué `aten::im2col` (OCAB `nn.Unfold`) — `docs/follow-ups/p-triton-im2col-kernel.md` OPEN ; no silent fallback (R33 hard crash confirmé). 2/4 modes seulement.
- `P-CONTAINER-EMBED-ORPHAN-SCALARS` : `compiled_sequence.py:2537` matérialise les `constant_T_*` non résolus en `torch.empty` 0-dim — **workaround approximatif auto-déclaré**, "not guaranteed at larger tiled sizes". Lié à 3.1/BL-1.
- Tiled conv2d R30 : SHA `1dff885`/`c9d2581` "skip standalone-tile / chain wrapper on triton modes" — gaps R30 interimaires, superseded par `8a6daf2` (root fix `NBXTensor.__getitem__`) mais le wrapper Triton-natif tilé "correct à toutes résolutions" n'a jamais été écrit.
- `_adapt_seq_dependent_weights` : code dormant "no-op for TinyLlama, reserved for future families" (`p_prism_never_refuse_s2/INDEX.md:20-21`) — non exercé.

### 3.4 Optimisations explicitement déférées post-dev (acceptées)
- **P-AUTOTUNE-OFFLINE** : gap structurel Volta sm_70 mm/bmm/addmm vs cuBLAS ~12 % plafond, irréductible cette année même avec autotune (`d514bdb`). Ne pas re-tester (`396fef1` revert = do-not-retry list).
- **POINT 8** : speedup compiled-vs-sequential 1.5× Sana 4Kpx FULL "physiquement impossible" (plafond 1.20×) — closure factuelle sans fix. P-TRITON-FUSED-KERNELS / P-CUDA-GRAPHS déférés.
- R33 Level-2 (kernels fusionnés) déférable par doctrine SI nommé en backlog.

### 3.5 Layer 6.bis — mécanisme Volta SDPA jamais expliqué
- SHA `dcec6a3` : "mechanism remains unexplained ... four open hypotheses for a future Layer X investigation". **"Layer X" jamais ouvert nommément.** Dette d'investigation orpheline.

---

## SECTION 4 — Dettes documentaires (verdicts contradictoires, docs périmées, plans non suivis)

### 4.1 Contradiction majeure : Sana 4Kpx 16 GiB "structurellement irréductible" vs "fermé ✓"
- `docs/verdicts/p_prism_estimator_tiling_aware/final/INDEX.md:73-77` (POINT 9) : "Sana 4Kpx peak ~17 GiB n'entre pas dans une V100 16 GiB ... minimum hardware viable reste 32 GiB" + "Chantier 1 NON ATTEINTE — blocker structurel".
- `docs/verdicts/p_sana_4kpx_runtime/point7_full_closure/INDEX.md:114-132` : "définitivement closed en mode 1× V100 32 GiB" seulement, 16g/2×16g `❌ ZERO FALLBACK`.
- `docs/verdicts/p_prism_never_refuse_v2_closed.md:8-9,99-102` : ferme 16g ET 2×16g ✓ sur les 4 modes via S5 depthwise + chain wrapper + root fix `8a6daf2`.
- **Verdict** : la thèse "17 GiB irréductible / 32 GiB minimum" de POINT 9 est **empiriquement renversée** par la clôture v2. Trois verdicts coexistent avec conclusions opposées. **Interprétation à confirmer avec Hocine** : POINT 9 à marquer SUPERSEDED ?

### 4.2 Tags/closures contradictoires P-PRISM-NEVER-REFUSE v2
- `p_prism_never_refuse_v2_session_close.md:179-180` : "Tag NOT posted ... victory criterion not met" (10/16).
- `p_prism_never_refuse_v2_closed.md:161` : "Tag posted" (14/16).
- Les deux fichiers vivent dans l'arbre ; `session_close` périmé non marqué tel. Idem `p_prism_never_refuse_v2_matrix_progress.md` (premature-closure-via-épuisement, 12/16).

### 4.3 Chaîne de closures rescindées P-SANA-4KPX-RUNTIME (auto-documentée)
- `p_sana_4kpx_runtime/sana_4kpx_post_phase15/verdict.md:25,83` : "(RESCINDED) Closure" + "REOPENED AGAIN 2026-05-05". Séquence `ede5d59`→`6631637`→`7d99b03`→`17e6d01` (2 closures prématurées rescindées). Clos par POINT 6/7 + v2, mais l'historique de relitigation reste sans index "lequel fait foi".

### 4.4 Plans du dépôt PKG écrits mais NON suivis (écart plan↔code)
- **Plan d'alignement capture↔runtime (PKG)** : **ABANDONNÉ** — pas de `graph_runner_v2.py`, aucun `normalize_graph`/`_convert_native_to_makefx` ; direction make_fx abandonnée au profit du DAG natif. Seul `torch_dtype` a atterri. Plan ~11h jamais exécuté, doc jamais marquée abandonnée.
- **Audit d'alignement « variable embarquée » (PKG)** : **ABANDONNÉ/SUPERSEDED** — plan 15-problèmes, abstraction "Enhanced Layer 1" absente du code (réécriture archetype).
- **Audit de sharding/clés (PKG)** : problème résolu mais **PAS via l'Option B recommandée** (reverse-lookup via la table de clés inverse) — résolu par réconciliation clé au build (`PKG/importer/builder.py:83`) + fallback runtime suffix-match (`graph_executor.py:1061`). Doc recommande une solution non implémentée.
- **Audit de capture-graph (PKG)** : partiel — versions diffusers/transformers SoT faites ailleurs, mais path racine des modèles bruts toujours hardcodé en défaut (CLI PKG), pas de config unifiée, orphan-attr-scan non fait — et **contredit** l'audit « legacy capture » qui endosse `constant_T_` (deux audits PKG en contradiction).
- **Audit des familles upscalers (PKG)** : partiel — U1.2/U1.3/U1.4 écrites comme plan ; le code a depuis **dépassé** (U5/U6/U7 ont livré real-esrgan/swinir/hat) — doc périmée vs état réel.
- **Doc « vision key-norm » (PKG)** : V3 (annotation layer + classifieurs bio/modality) **non démarré**, auto-déclaré roadmap, gated sur un système LoRA inexistant.
- **Leçon PKG « 5 fixes capture »** : "Fix 3 option CLI removed as dead code" mais la doc de séparation PKG la liste encore comme valide — incohérence doc↔doc (**à confirmer**).

### 4.5 Verdicts avec "Hocine validation: TODO" jamais back-annotés
- Tous les `docs/verdicts/runtime_alignment_phase5/*/verdict.md` finissent "Hocine validation: TODO" alors que `runtime_alignment_phase5/INDEX.md:52-61` enregistre la revue faite. Incohérence de tracking. Idem `p_nbx_tiled_conv2d_small_scale/verdict.md:97`, `p_triton_live_watermark_audit/verdict.md:82`, `p_prism_never_refuse_s2`.

---

## SECTION 5 — Dettes opérationnelles (modèles non publiés / .nbx hub périmés)

### 5.1 Backlog de publication (hub LIVE, .nbx convertis non publiés)
Hub `neurobrix.es` opérationnel, 15 modèles publiés. Convertis localement mais **NON publiés** : les 10 upscalers (real-esrgan ×3, swin2SR ×3, swinir ×2, hat ×2), parakeet, whisper-large, Kokoro, VibeVoice, openaudio, canary-qwen. Source : `docs/audit_technical_debt_2026_05_15.md` §A + requête live `?category=UPSCALER` = NONE. Opération `publish` de routine.

### 5.2 [REOPEN-CANDIDATE] .nbx du hub potentiellement PÉRIMÉS — path-leak + métadonnées d'ops
- **Path-leak** : l'audit path-leak PKG (`:188-219`) — fix appliqué `3349c94/61e40a8`, script de nettoyage rétroactif `4a1684e` sur 173 fichiers. **Question non tranchée** : les .nbx déjà publiés sur le hub *avant* le fix portaient le path absolu `/home/mlops/...`. Le cleanup a-t-il touché les blobs du stockage objet du hub, ou seulement les .nbx locaux ? `b848d4e` "release patch decision" — **interprétation à confirmer avec Hocine** : les 15 modèles du hub sont-ils pré- ou post-fix ? Si pré-fix → path absolu embarqué → à re-uploader. Gain : intégrité du hub.
- **Régression métadonnées d'ops (R28)** : SHA `ec4b385` "parent_module était vide pour 100% des ops sur TOUS les nouveaux graphs capturés" — fixée seulement en R28.a. Tout .nbx buildé entre R28 et R28.a a des métadonnées d'ops dégradées. **À confirmer** : quels modèles du hub sont dans cette fenêtre.

### 5.3 Hooks de commit déconnectés → travail hors version-control
- SHA `62ad908` "auto: sync uncommitted changes (hooks were disconnected)" ; `4b6d721` "orphan ... never committed". Du travail a accumulé hors git côté PKG ; rattrapé par sync mais période de non-traçabilité.

### 5.4 WIP/alpha sur main
- SHA `f42970c` "KNOWN INCOMPLETE: openaudio DualAR still crashes ... Perf not measured ... Ampere+ path not verified" — WIP commité sur main NeuroBrix.
- SHA `75b482c` "TinyLlama triton compiled: correct output, 2 tok/s (perf WIP)" — alpha sur main.

---

## SECTION 6 — Tests skip/xfail jamais dé-skippés (exhaustif)

Tous dans `tests/regression/{conftest.py,test_all_models.py}` (le reste = tutorial vendored, exclu). **Aucun `strict=True`** : tous `xfail(strict=False, run=True)` → un blocker fixé mais non retiré masquerait silencieusement la régression (le conftest l'avertit lui-même, `conftest.py:100-101`).

| Cellule | Mark@conftest | Raison (verbatim) | Blocker |
|---|---|---|---|
| whisper-large::triton | xfail:108 | "Triton encoder_decoder audio flow not validated end-to-end yet" | OPEN |
| whisper-large-v3-turbo::triton | xfail:109 | idem | OPEN |
| parakeet-tdt-1.1b::triton | xfail:110 | "Triton rnnt flow not validated end-to-end yet" | OPEN |
| canary-qwen-2.5b::triton | xfail:111 | "triton/flow/audio_llm.py missing — handler never ported" | OPEN (fichier confirmé absent) |
| granite-speech-3.3-8b::triton | xfail:112 | "triton/flow/audio_llm.py missing + CFormer projector native-stage mix" | OPEN |
| Voxtral-Mini-3B-2507::triton | xfail:113 | "triton/flow/audio_llm.py missing" | OPEN |
| chatterbox::triton | xfail:119 | "triton/flow/tts_llm.py doesn't wire audio_path reference voice" | OPEN-PARTIAL (lit `global.ref_audio`, pas `audio_path`) |
| openaudio-s1-mini::triton | xfail:120 | "triton/flow/dual_ar.py doesn't wire audio_path reference voice" | OPEN (confirmé) |
| Kokoro-82M::triton | xfail:134 | "_execute_native_text_encoder passes NBXTensor to torch.nn.functional.embedding" | OPEN (`stages/kokoro.py:424`) |
| Kokoro-82M::native | xfail:135 | "aten::cudnn_batch_norm ... regression v0.1.5→be5c7b8" | OPEN (cf. §1.3) |
| VibeVoice-1.5B::native + ::triton | xfail:145 | "TensorDAG contract violation — DDPM + ConvNext1d outside graph; needs re-capture" | OPEN (2 cellules) |
| conftest.py:211 | skip | "skipped; pass --runslow" | n/a (politique opt-in image/vidéo) |
| test_all_models.py:211 | skip | "no golden on record — re-run" | n/a (transitoire 1er run) |

**Lecture clé** : 12 cellules xfail = quasi toute la coverage triton audio/STT/TTS/audio_llm à venir est **non validée, trackée seulement par xfail non-strict**. `triton/flow/audio_llm.py` **n'existe pas** (jamais porté de `core/flow/audio_llm.py`) — dette structurelle bloquant Voxtral/canary/granite en triton.

---

## SECTION 7 — Chantiers fermés AVEC dette résiduelle annoncée, non trackée séparément

| Chantier | Closure SHA/tag | Phrase exacte | Statut aujourd'hui |
|---|---|---|---|
| **P-PRISM-NEVER-REFUSE v2** | `77571b7` / tag `p-prism-never-refuse-v2-closed` | "14/16 cells validated + 2 upstream-blocked ... CPU triton + CPU triton_sequential remain upstream-blocked on `triton-cpu` (not on PyPI; fp16 gap issue #147)" (`p_prism_never_refuse_v2_closed.md:18-19`) | **Présente** — 2 cellules CPU OPEN, externe (R25, non forké). |
| **P-PRISM-NEVER-REFUSE v2** | idem | "Backlog: P-OP-LEVEL-CROSS-DEVICE-SPLIT (Gap B)" (`:130`) | **Oubliée** — pas de follow-up doc dédié ; vit dans le verdict seul. |
| **P-NEUROBRIX-UPSCALERS-V1** | `a92ee00` / tag `p-neurobrix-upscalers-v1-closed` | "HAT 2/4 modes ... blocked by aten::im2col" + "U8 DRCT deliberately skipped" + "BL-1 ... arbitrary-size tiling stays gated" | **Présente** — im2col + orphan-scalars trackés OPEN, DRCT/swinir-realworld non convertis, BL-1 OPEN. |
| **P-SANA-4KPX-RUNTIME** | tag `p-sana-4kpx-runtime-fully-closed` (`b66423a`) | "Blocker mémoire full pipeline reste sous P-TRITON-LIVE-WATERMARK-AUDIT" + "Backlog: P-PRISM-ACTIVATION-ESTIMATOR-TILING-AWARE" + POINT 7 "16/2×16 GiB ❌ ZERO FALLBACK" | **Partielle** — 16g/2×16g ensuite fermés en v2 (contradiction §4.1) ; live-watermark résorbé POINT 6 ; `P-MULTI-GPU-NBX-ADAPTER` jamais ouvert en follow-up nommé. |
| **Layer 6.bis** | `dcec6a3` | "mechanism remains unexplained ... future Layer X investigation" | **Oubliée** — "Layer X" jamais ouvert. |
| **Layer 7 / 8 / 9** | `a736689` / `b6eb3de` / (Sana VAE) | follow-ups OPEN ; Layer 7 fix archi (auto-detect family fp32) **non implémenté** — seul opt-in manuel | **Présente** — PixArt/Sana 4Kpx triton bloqués ; Layer 8 fin-de-but contredit par Layer 9. |
| **pixart_triton_arena_inter_run_bug** | archivé "SUPERSEDED BY LAYER 7" | Layer 7 lui-même OPEN sans fix ; diagnostic promis `NBX_ARENA_DUMP` **confirmé absent du code** | **Chaîne supersedor non terminée** — PixArt triton effectivement bloqué, ré-attribué. |
| **POINT 9 / estimator-tiling-aware** | `p_prism_estimator_tiling_aware/final/INDEX.md` | "Chantier 1 NON ATTEINTE — blocker structurel (17 GiB > 16 GiB)" | **Renversée** par v2 (§4.1) — à re-statuer. |
| **POINT 8** | `point8/fix_summary.md:19` | "Aucune modification de code. Closure factuelle." | **Acceptée** — dette perf assumée, fused-kernels/cuda-graphs déférés. |
| **P-NBX-TILED-CONV2D-SMALL-SCALE** | `verdict.md:11-15` | "16g triton remains ⏳ ... OOM at conv::64" + "Hocine validation: TODO" + "wall-time 2.6× baseline, to be confirmed" | **Traitée plus tard** (`8a6daf2`, 16g triton ✓ v2) ; verdict ⏳ non back-annoté. |
| **runtime_alignment_phase5** | `INDEX.md:30-34` | "Orpheus GQA wrapper pre-existing bug" + "Voxtral hallucination — processor-multimodal chantier" + "Chatterbox charabia — out of scope" + "Janus color fidelity off — out of scope" | **Présente** — GQA wrapper, Voxtral hallucination, Chatterbox decoding, Janus color : 4 dettes "out of scope" jamais ouvertes en follow-up nommé. **Très pertinent coverage audio/multimodal à venir.** |
| **PKG taxonomy P2.2** | `84c895f` | "Régression P2.2 oubliée ... publish oublié ... Doctrine candidate (à formaliser plus tard)" | **Doctrine non formalisée.** |

---

## SECTION 8 — Recommandation d'ordre d'attaque (strict : correctness > universalité HW > parité native↔triton > cosmétique)

**Avant d'attaquer la prochaine famille**, solder dans cet ordre. Top 10 :

1. **D1 — `linspace` Triton mémoire non-init** (§1.1). *Scope* : petit (câbler le `fill_kernel` déjà importé). *Prérequis* : aucun. *#1* : reachable confirmé, corrompt silencieusement diffusion/vidéo — pile la famille visée. **[REOPEN-CANDIDATE]**, gain majeur, coût minime.

2. **D2 — `index_put`/`index_put_` lambdas identité** (§1.2). *Scope* : petit-moyen. *Prérequis* : confirmer reachability (`_OP_MAP` vs classification pour MoE-v2/KV-cache). *#2* : si reachable = perte silencieuse d'écriture audio_llm/MoE/multimodal. **[REOPEN-CANDIDATE]**. Action 1 = trancher reachability avec Hocine.

3. **D3 — Bare-except correctness** (§1.4/1.5/1.6) : `output_processor.py:87`, `input_synthesizer.py:248`, `memory.py:262`, `PKG/importer/builder.py:607`. *Scope* : petit (crash explicite ou log+raise, doctrine ZERO-FALLBACK). *Prérequis* : aucun. *#3* : 4 trous silencieux, fix mécanique.

4. **D4 — Kokoro `::native` cudnn_batch_norm** (§1.3). *Scope* : moyen (bisect borné ~8 commits). *Prérequis* : harness regression. *#4* : modèle TTS régressé en mode défaut ; bloque complétion TTS. **[REOPEN-CANDIDATE]**.

5. **D5 — `triton/flow/audio_llm.py` manquant** (§6). *Scope* : moyen-grand (porter `core/flow/audio_llm.py`). *Prérequis* : D2 (KV-cache index_put). *#5* : débloque Voxtral/canary/granite triton (5 cellules xfail) — coverage audio_llm directe.

6. **D6 — Symbolisation family-aware `_infer_dim_names`** (§3.1). *Scope* : grand, fort levier. *Prérequis* : décision archi Hocine. *#6* : cause racine BL-1 + tiling arbitraire + coverage symbolique vision/audio/codec ~0 %. **[REOPEN-CANDIDATE]** (gain système max ; session dédiée).

7. **D7 — Réconcilier doctrine DtypeEngine** (§3.2). *Scope* : petit (doc) ou moyen (retour lesson). *Prérequis* : arbitrage Hocine. *#7* : ambiguïté doctrine/code avant modèles fp16-sensibles audio/vidéo.

8. **D8 — Intégrité hub : .nbx pré-fix path-leak / R28** (§5.2). *Scope* : moyen (audit des 15 .nbx : date build vs fix ; re-upload si pré-fix). *Prérequis* : accès hub (acquis). *#8* : universalité — path absolu casse chez un tiers. **[REOPEN-CANDIDATE]**.

9. **D9 — Layer 7 auto-detect fp32 family-aware** (§3.3). *Scope* : moyen. *Prérequis* : D7. *#9* : PixArt/Sana image triton bloqués ; parité native↔triton image.

10. **D10 — Hygiène verdicts** (§4) : marquer SUPERSEDED (POINT 9 §4.1, session_close §4.2, chaîne rescindée §4.3), back-annoter "Hocine validation" (§4.5), ouvrir en follow-ups nommés les dettes orphelines (Gap B, Layer X, GQA wrapper, Voxtral hallucination, Chatterbox decoding). *Scope* : petit (doc). *#10* : évite que la prochaine session reparte sur des verdicts contradictoires.

**Hors Top 10, déférés assumés (NE PAS attaquer)** : P-AUTOTUNE-OFFLINE, POINT 8 perf, R33 Level-2, P-OP-LEVEL-CROSS-DEVICE-SPLIT, triton-CPU upstream (R25 externe). Cosmétique (~150 `# type: ignore`/`# noqa`/commentaires post-mortem "silently") : aucune action, non-dette.

---

## Synthèse pour Hocine

La dette correctness la plus dangereuse n'est pas dans les chantiers fermés mais dans deux `# TODO` inline jamais relus : `linspace` Triton retourne de la **mémoire non initialisée** (reachable confirmé, corrompt diffusion/vidéo silencieusement) et `index_put` est une **lambda identité** (write scatter perdu, reachability à confirmer). Plus 4 `except: pass` désactivant silencieusement clamp VAE / synthèse d'input / cache. Côté chantiers : P-PRISM-NEVER-REFUSE v2 "14/16 + 2 upstream-blocked" et la contradiction Sana-4Kpx-16GiB (POINT 9 "impossible" vs v2 "fermé ✓") sont exactement le type de fermeture-en-contournant visé. La dette de fond (symbolisation non family-aware, récurrente sur 13 mois) sous-tend BL-1 et toute la coverage vidéo/audio à venir. Recommandation : solder D1-D5 (correctness, scope petit-moyen) **avant** d'ouvrir la prochaine famille ; D6 en session dédiée ; D10 en continu.

— Fin du rapport. Lecture seule, aucun code/commit modifié pendant l'audit.
