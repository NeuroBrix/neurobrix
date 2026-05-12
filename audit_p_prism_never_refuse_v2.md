# P-PRISM-NEVER-REFUSE v2 — Phase A audit factuel

Audit sourcé fichier+ligne, lu pendant la session. Point de départ
commit `af2f858`. Doctrine R34 (model-agnostic strict) + R35 (Prism
never refuses) appliquées. Remontée Hocine avant Phase B.

═══════════════════════════════════════════════════════════════════
A.1 — INVENTAIRE STRATÉGIES EXISTANTES
═══════════════════════════════════════════════════════════════════

**8 stratégies concrètes + 1 base abstraite + 1 sous-classe lazy** dans
`src/neurobrix/core/strategies/` :

| fichier | classe principale | execute_component |
|---|---|---|
| `base.py:92` | `ExecutionStrategy(ABC)` | `:119` abstract |
| `single_gpu.py:20` | `SingleGPUStrategy` | `:47` |
| `lazy_sequential.py:24` | `LazySequentialStrategy` | `:41` |
| `component_placement.py:26` | `ComponentPlacementStrategy` | `:53` |
| `component_placement.py:127` | `ComponentPlacementLazyStrategy(ComponentPlacementStrategy)` | hérite |
| `block_scatter.py:23` | `BlockScatterStrategy` | `:76` |
| `pipeline_parallel.py:22` | `PipelineParallelStrategy` | `:78` |
| `weight_sharding.py:19` | `WeightShardingStrategy` | `:49` |
| `tp_sharding.py` | utilities only (block sharding helpers, pas de stratégie cascadée) |
| `zero3.py:97` | `Zero3Strategy` | `:692` |

**Hypothèses hardware sous-jacentes par fichier** (lues dans docstrings + signatures) :

- `single_gpu.py` — single GPU only. Cold mode budget = `max(component peak)`.
- `lazy_sequential.py` — single GPU only. `_place_component` cascade GPU intrinsique (single_gpu → fgp → zero3 GPU-side).
- `component_placement.py` — multi-GPU strict. Place chaque composant sur un GPU différent.
- `block_scatter.py` — multi-GPU strict. Best-fit-decreasing par block.
- `pipeline_parallel.py` — multi-GPU. Layers 0..N sur GPU0, N+1..M sur GPU1.
- `weight_sharding.py` — multi-GPU. Shard de poids round-robin entre GPUs.
- `zero3.py` — single ou multi-GPU. Weights pinned CPU + compute GPU (block-wise ratchet). **Activations restent sur GPU** (cf. ÉTAPE B.3 nécessitera étendre).

**Aucune stratégie ne route le compute vers CPU**. La cascade R35 nécessite une nouvelle stratégie `cpu_execution` (priorité Phase B.3).

### Dispatcher inventory (les deux branches R30/R33)

**Branche A — PyTorch ATen native**
- `src/neurobrix/core/runtime/graph/sequential_dispatcher.py` — dispatcher eager.
  - Signature `__init__(self, device: Optional[str] = None, ...)` (`:43`)
  - `self._runtime_device = device` (`:47`) utilisé pour résoudre les `device` attrs du graph
  - Ligne 78 : `torch.device(self._runtime_device)` — **device-aware par construction**, accepte 'cpu'
  - Ligne 206 : reroutage natif `_scaled_dot_product_flash_attention_for_cpu` — CPU-aware ATen
- `src/neurobrix/core/runtime/graph/compiled_sequence.py` — CompiledSequence côté PyTorch (mature)

**Branche B — Triton pure (R33)**
- `src/neurobrix/triton/sequential.py` — `TritonSequentialDispatcher`. `device_idx: int = 0` — **GPU-only par construction** (kernel dispatch via Triton + NBXTensor en CUDA).
- `src/neurobrix/triton/sequence.py` — `TritonSequence` hot loop. Idem GPU-only.
- **Triton-CPU non intégré** dans la branche. Grep `triton.cpu|triton_cpu|cpu_backend` retourne 0 hit pertinent dans `triton/` (1 hit dans `kernels/triton_kernels_ref/fla/nvidia/utils.py:381` — référence tiers fla, pas le runtime NeuroBrix).
- NBXTensor (`src/neurobrix/kernels/nbx_tensor.py`) supporte `device='cpu'` (lignes 1126, 1212, 1384, 1389, 1408, 1428, 1455, 1468, 1485, 1647, 1868, 1874, 1876). La représentation CPU existe ; ce qui manque c'est le **dispatcher Triton-CPU** côté `triton/`.

═══════════════════════════════════════════════════════════════════
A.2 — CASCADE ACTUELLE DANS LE SOLVER
═══════════════════════════════════════════════════════════════════

`src/neurobrix/core/prism/solver.py`, méthode `solve()` :

**Single-GPU profile** (`len(devices) == 1`, ligne `519-524`) — **4 stratégies** :
1. `single_gpu`
2. `single_gpu_lifecycle`
3. `lazy_sequential`
4. `zero3`

**Multi-GPU profile** (ligne `526-536`) — **9 stratégies** :
1. `single_gpu`
2. `single_gpu_lifecycle`
3. `component_placement`
4. `pipeline_parallel`
5. `block_scatter`
6. `weight_sharding`
7. `component_placement_lazy`
8. `lazy_sequential`
9. `zero3`

### ZERO_FALLBACK call sites

`grep -nE "ZERO FALLBACK|_fail_error|raise RuntimeError" core/prism/solver.py` :

| line | context |
|---|---|
| `173` | `RuntimeError(f"ZERO FALLBACK: Cannot allocate ...")` — per-device allocator failure |
| `514` | `raise RuntimeError("ZERO FALLBACK: No GPU devices ...")` — profile vide |
| `552/558` | strategy filter failures via NBX_FORCE_STRATEGY |
| `584` | NBX_FORCE_STRATEGY mode |
| **`623`** | **`self._fail_error(sorted_components, devices)` — point d'émission principal** quand `not candidates` post-cascade |
| `700` | FGP cascade fallthrough |
| `1390/1403/1436` | KV cache budget |

Le `_fail_error` final (ligne `623`) est précédé de :
- Cascade `_evaluate_all_strategies` (ligne `570`)
- FP32 fallback retry si bf16 model (ligne `605-614`)
- Serve mode → cold mode dégradation (ligne `591-602`)

**Hook d'extension** : ajouter `cpu_execution` et `cpu_offload` dans la liste `strategies` ligne `519-536`, avant `zero3`, ne nécessite aucun refactor du cascade — `_evaluate_all_strategies` itère sur tout ce qu'on lui donne.

### POINT 10 v1 in-scope fix (déjà landed dans af2f858)

`_fail_error` lignes `2820-2842` :
- Message d'erreur dynamique selon `len(devices)` (corrige le hardcode "5 strategies tried")
- Mentions explicites des chantiers backlog `P-MULTI-GPU-NBX-INTRA-COMPONENT-SPLIT` et `P-PRISM-CPU-FALLBACK-EXECUTION` (renommé ici en P-PRISM-NEVER-REFUSE-v2)

═══════════════════════════════════════════════════════════════════
A.3 — TILING ENGINE — CAPACITÉ ACTUELLE
═══════════════════════════════════════════════════════════════════

`src/neurobrix/core/module/tiling_engine.py` (**965 lignes**, pas 377 — mandate approximatif) :

### Classes
- `TilingEngine:32` — spatial tiling COMPONENT-LEVEL (input H/W dépasse trace size)
  - `from_component_config:80` — construction depuis graph.json + profile.json
  - `should_tile:202` — décide tile vs single-pass
  - `tiled_execute:223` — execute par tiles overlappés, accumulate-and-divide
- `OpLevelTilingPlan:399` — data class du plan OP-LEVEL
  - `fusion_pairs:413` — `(upsample_uid, conv_uid, tile_factor)`
  - `tiled_ops:417` — `(op_uid, op_type, tile_factor)` — convs ou rms_norms standalone
  - `inplace_adds:425` — `(op_uid, reuse_input_index)` — résidual adds in-place
- `OpLevelTilingEngine:449` — implémentation des interceptors
  - `_detect_pixel_shuffle_broadcast_chains:469` — détection F2a (expand→clone→view→pixel_shuffle)
  - `_detect_inplace_add_candidates:613` — détection in-place adds par liveness
  - `register_into_graph_executor:713` — **point d'entrée principal** ; wire les interceptors

### Ops actuellement interceptées par op_uid
Lecture `register_into_graph_executor:713-963` :

| op type | mécanisme | source |
|---|---|---|
| `aten.upsample_nearest2d::*` (en fusion pair) | `make_upsample_proxy_interceptor` retourne `FusionUpsampleProxy` (sentinel, 0 byte) | `:739` |
| `aten.convolution::*` (en fusion pair) | `make_conv_interceptor(tf)` → `fused_upsample_conv2d` band-streamed | `:744-756` |
| `aten.convolution::*` (standalone tiled) | `make_tiled_conv(tf)` → `tiled_conv2d_spatial` halo-streamed | `:765-781` |
| `custom.rms_norm::*` (output > 20% VRAM) | `make_tiled_rms(tf)` → `tiled_rms_norm_spatial` | `:782-791` |
| `aten.add::*` (in-place résidual) | `make_inplace_add_interceptor(reuse_idx)` dual-backend NBX/torch | `:794-845` |
| `aten.expand::*` / `aten.clone::*` / `aten.view::*` / `aten.pixel_shuffle::*` (F2a chains) | proxy interceptors broadcast-aware | `:854-951` |

### Adding a new tiled op type — R34 mécanisme

`register_into_graph_executor` ligne `765` :
```python
for op_uid, op_type, tile_factor in self.plan.tiled_ops:
    cl = op_type.split("::")[-1]
    if cl in ("convolution", "conv2d", "_convolution"):
        ...
    elif cl == "rms_norm":
        ...
    else:
        logger.debug(f"[OpLevelTilingEngine] No tiled implementation for op_type={op_type} ...; skipping.")
```

Ajouter une op tilable :
1. Le `OpLevelTilingPlan.add_tiled_op(uid, op_type, tile_factor)` est appelé par `_detect_op_level_tiling_pairs` dans `solver.py:870-887` (convs en overflow) et `solver.py:894-915` (rms_norms en overflow). Pour étendre à d'autres op_types, étendre cette détection.
2. Implémenter le kernel tilé dans `src/neurobrix/kernels/ops/` (les deux backends NBX + torch).
3. Ajouter la branche `elif cl == "<new_type>":` dans `register_into_graph_executor`.

**R34 conforme** : la sélection se fait par `op_type` (générique) + critère mémoire (`output_bytes_estimated > threshold`), pas par nom de modèle.

### conv::62 (POINT 10 v1 bottleneck) — déjà dans le périmètre ?

Le mandate dit "Confirmer si `aten.convolution::62` (bottleneck 16 GiB POINT 10 v1 : output [1,128,4096,4096] fp16 = 4 GiB + workspace cuDNN) est dans le périmètre potentiel d'interception".

**Réponse factuelle** : conv::62 EST DÉJÀ intercepté par la fusion pair `upsample::4 → conv::62` (POINT 6). L'interception route vers `_fused_upsample_conv2d_nbx` (et `_torch`), qui :
1. **Pré-alloue le buffer output FULL** : `output = NBXTensor.empty((N, out_c, conv_out_h, conv_out_w), ...)` — c'est le 4 GiB structurel
2. Itère sur tile_factor bands, chaque band lit halo + écrit dans son slice de `output`
3. Le buffer FULL est nécessaire parce qu'il est consommé par `add::86` in-place chaîné

Donc l'interception est faite, mais **le 4 GiB output reste structurel** parce que la consommation downstream (add::86) le requiert. L'option B.1 du mandate (tiling étendu) nécessite **changer le contrat** : au lieu d'écrire dans un buffer FULL, streamer l'output band-par-band directement vers le consommateur. C'est faisable mais c'est un refactor non-trivial du contrat de fusion (modifier `add::86` pour qu'il consomme un proxy de stream au lieu d'un buffer complet).

**Alternative B.1 plus simple** : tile encore plus aggressively à l'amont — réduire les autres live tensors au moment de conv::62 (~9 GiB sont composées de conv::55 output + permute + clones préalables). Si on libère certains avant conv::62 (via deferred-free drain), le 4 GiB allocation pourrait fit.

POINT 10 v1 expérimentation : NBX_DEFERRED_DRAIN_BYTES=512MB n'a pas suffi. La majorité de la 12.9 GiB live au moment de conv::62 n'est pas dans la deferred queue mais dans des tensors réellement alive selon le graph DAG. Un nouveau mécanisme est nécessaire (ex: forcer un sync+drain juste avant conv::62 via interceptor pre-hook).

### Tiling engine et les deux branches

Le tiling engine est **branche-agnostic dans sa philosophie** : les interceptors sont registrés dans `graph_executor._op_uid_interceptors` qui est lu par les deux dispatchers (CompiledSequence native + TritonSequence + sequential_dispatcher + triton_sequential). Les kernels tilés eux-mêmes ont des branches NBX/torch (`_fused_upsample_conv2d_nbx` vs `_fused_upsample_conv2d_torch`) — donc R30 préservé.

═══════════════════════════════════════════════════════════════════
A.4 — AUDIT R34 (model-agnostic violations)
═══════════════════════════════════════════════════════════════════

### Grep par nom de modèle (large)

`grep -rEni "sana|pixart|flux|qwen|tinyllama|deepseek|janus|kokoro|voxtral|whisper|chatterbox|orpheus|vibevoice|gemma|llama|mistral|mixtral"` sur `core/prism/`, `core/strategies/`, `core/module/tiling_engine.py`, `core/runtime/graph_executor.py`, `core/runtime/graph/`, `triton/` :

**Classification des hits** :

| catégorie | nombre | exemples (fichier:ligne) |
|---|---|---|
| COMMENT (justification historique / exemple) | majorité | `tiling_engine.py:13` "VAE decoders (Sana 4K DC-AE: window attention seam artifacts)" — comment explicatif |
| GENERIC_FEATURE (mention dans une feature universelle) | quelques | `block_detector.py:27` regex `r'single_transformer_blocks[._](\d+)'` matching "Flux style" — **regex pattern matching pour BLOCK STRUCTURES, pas pour le modèle Flux spécifiquement** ; ce regex match tout modèle qui utilise cette convention de nommage. |
| ACTIVE_VIOLATION | **0** | aucune |

### Grep par pattern `if model_name == ...` etc.

`grep -rEn "if.*(model_name\|self\.model\|family)\s*==|if.*\"(sana|pixart|...)\"|family ==|model_type =="` :

| hit | fichier:ligne | verdict |
|---|---|---|
| `if category == "diffusion":` | `core/prism/solver.py:1307` | **GENERIC_FEATURE** : `category` set par `self._model_category` depuis `manifest.family`, données du container, pas hardcoded |
| `elif category == "diffusion":` | `core/prism/solver.py:1636` | Idem |

**Verdict A.4** : **AUCUNE active violation R34 détectée** dans le périmètre Prism + strategies + tiling + runtime dispatchers. Les mentions de noms de modèles sont :
- Commentaires historiques (R34 OK — commentaires ne changent pas le code path)
- Regex de pattern matching pour conventions de nommage de blocks (R34 OK — match tout modèle utilisant la convention)
- Discriminations par `family/category` data-driven depuis le manifest (R34 OK — généralisable)

**Aucun fichier `audit_model_agnostic_violations.md` à créer** pour ce périmètre. La pureté model-agnostic est préservée.

═══════════════════════════════════════════════════════════════════
A.5 — SUPPORT CPU ACTUEL DU RUNTIME
═══════════════════════════════════════════════════════════════════

### Branche A — PyTorch ATen native

**NBXTensor CPU support** : `src/neurobrix/kernels/nbx_tensor.py` ligne `1126-1876` — `device='cpu'` géré natif (alloc pinned ou heap, cudaMemcpy DtoH/HtoD, conversion CPU→GPU).

**sequential_dispatcher.py** : ligne `43-77`, `device` parameter accepted; `_runtime_device` overrides graph-hardcoded devices. **Device-aware par construction** — accepte 'cpu' sans modification.

**compiled_sequence.py** (CompiledSequence côté PyTorch) : non audité en détail mais utilise les mêmes primitives `torch.device` paramétrables.

**Thread configuration depuis cpu.cores** :
- `src/neurobrix/core/prism/cpu_config.py:91-115` — `apply_thread_config(cpu, involves_cpu)` câble :
  - `torch.set_num_threads(cpu.cores)` (intra-op)
  - `torch.set_num_interop_threads(max(1, min(4, cpu.cores // 4)))` (inter-op)
- Câblage au CLI : `src/neurobrix/cli/commands/run.py:266-274` :
  ```python
  if hw_profile.cpu:
      from neurobrix.core.prism.cpu_config import apply_cpu_config
      apply_cpu_config(cpu=hw_profile.cpu, strategy=execution_plan.strategy,
                      device_count=hw_profile.device_count,
                      preferred_dtype=hw_profile.preferred_dtype)
  ```
- Câblage au serving : `src/neurobrix/serving/engine.py:115-116` idem.

**`apply_cpu_config` est gated par `involves_cpu`** (cf. `cpu_config.py:91`), qui est True si la stratégie inclut le CPU (à vérifier dans `apply_cpu_config` ligne 207-220 — code lu : il appelle `apply_thread_config(cpu, involves_cpu)` mais le calcul de `involves_cpu` n'a pas été tracé exhaustivement). Vérification Phase B.

**MKL/oneDNN** : auto-tiré par PyTorch selon `cpu.features` — pas de wiring explicite nécessaire dans NeuroBrix. AVX512 / AMX_BF16 / AMX_INT8 captured par `autodetect.py` (cf. `autodetect.py:235-237` et alentours).

**Verdict Branche A** : **Le path PyTorch CPU est viable et largement déjà câblé**. Ce qui manque pour victoire :
1. Une stratégie Prism `cpu_execution` qui ALLOUE toutes les composantes à `device='cpu'`
2. Vérification `apply_cpu_config(involves_cpu=True)` est appelé quand `cpu_execution` est choisie

### Branche B — Triton pure (R33)

**Statut Triton-CPU** : non intégré dans `src/neurobrix/triton/`. Grep `triton.cpu|triton_cpu|cpu_backend|backend.*cpu` retourne 0 hit dans `triton/`, 1 hit dans `kernels/triton_kernels_ref/fla/nvidia/utils.py:381` (référence tiers, pas le runtime).

**Triton-CPU upstream (meta-pytorch/triton-cpu)** — recherche web obligatoire selon le mandate. **Non effectuée dans cette session** (pas de web_search tool actif dans la trousse courante). À effectuer avant Phase B.3 :
- Version stable 2026 ?
- Coverage des ops critiques NeuroBrix (matmul, conv2d, attention/SDPA, rms_norm, upsample_nearest2d, pixel_shuffle, layer_norm) ?
- Build : pip vs source LLVM ?

**Hypothèses prudentes (à confirmer)** :
- Triton-CPU compile la SAME source `@triton.jit` que Triton-GPU mais cible un backend CPU MLIR-based + OpenMP. Permettrait théoriquement de réutiliser tous les kernels NeuroBrix Sana/PixArt/TinyLlama sans réécriture.
- Coverage probablement incomplète sur certains kernels customs (flash attention V2 GPU-specific, ...) — fallback PyTorch CPU temporaire avec `triton_cpu_coverage_gaps.md` documenté.

**Verdict Branche B** : **Travail d'intégration significatif** pour Phase B.3. Estimation > 100 lignes mais probablement <300 lignes si Triton-CPU upstream couvre 80%+ des kernels NeuroBrix. À ré-évaluer post-web-search.

### Memory estimator et RAM CPU

`src/neurobrix/core/prism/solver.py` :
- Ligne `2408` : `cpu_cores = profile.cpu.cores if profile.cpu else 4` — utilisation cores
- Ligne `2501-2505` (dans `_try_zero3`) : validation `weights fit 70% of RAM` — **seule utilisation actuelle de `cpu.ram_mb` comme contrainte de budget**
- Ligne `2809` : `cpu_ram_mb=profile.cpu.ram_mb if profile.cpu else 0` — passe au plan
- **Le memory_estimator (POINT 9 fix f7f2d9b) ne raisonne PAS sur la RAM CPU comme budget total** ; il raisonne sur la VRAM par GPU.

Phase B.3 nécessitera étendre `_compute_memory` ou `_try_cpu_execution` pour valider :
`sum(component peaks at compute_dtype) <= cpu.ram_mb * safety` où le peak compute_dtype côté CPU est typiquement fp32 (cuDNN/MKL fp32) pour les modèles bf16 et fp16 pour fp16 natif. R34 conforme — pas de hardcode model-specific.

═══════════════════════════════════════════════════════════════════
A.6 — VERDICT AUDIT ET PLAN PHASE B
═══════════════════════════════════════════════════════════════════

### Verdict synthétique

1. **Architecture saine** : Prism cascade 9 stratégies multi-GPU + 4 single-GPU, hook d'extension propre (`_evaluate_all_strategies`).
2. **R34 préservée** : aucune violation model-agnostic active. Audit clean.
3. **Tiling engine extensible** : ajouter une op tilable = critère générique d'overflow + kernel tilé + branche dans `register_into_graph_executor`. **conv::62 déjà intercepté** par fusion mais le 4 GiB buffer output est structurel à la fusion actuelle.
4. **Branche A (PyTorch CPU)** : largement câblée déjà. Nouvelle stratégie `cpu_execution` + vérification thread wiring = travail léger.
5. **Branche B (Triton CPU)** : non intégrée. Intégration Triton-CPU upstream nécessaire. Risque taille >200 lignes selon coverage.

### Plan Phase B priorisé selon cascade R35

**B.1 — Tiling engine étendu pour les configs 16 GiB**

*Cible* : Sana 4Kpx 1× V100 16 GiB produit PNG sur les 4 modes.

*Approche* :
- POINT 10 v1 a montré que conv::62 alloc 4 GiB sur fond de live 9 GiB → OOM. Le problème n'est pas l'absence d'interception (fait), c'est le buffer FULL pré-alloué.
- **Option B.1.a (simpler)** : un pre-hook interceptor sur conv::62 (et par symétrie sur toute conv en overflow) qui force `DeviceAllocator.sync_device()` + drain deferred queue + `gc.collect()` immédiatement avant l'allocation. Vérifier si la 12.9 GiB live de POINT 10 v1 contient des refs Python orphelines (tensors retained from earlier ops alors que graph.last_uses dit ils sont morts). Si oui, sync+collect peut libérer assez pour passer.
- **Option B.1.b (deeper)** : refactor le contrat de fusion conv→add_inplace pour streamer band-par-band sans buffer FULL. Le band sortant est consommé par add::86 directement (le buffer FULL n'est plus nécessaire). Estimation 100-200 lignes, R30 dual-backend obligatoire.
- **Option B.1.c (parallel)** : aussi intercepter les convs DOWNSTREAM (conv::63, ::64, ...) pour réduire le live au moment de conv::62. Si add::86's input1 (conv::55::out_0 = 8 GiB) peut être tilée en lecture pour add::86 (consume-as-you-stream), le live drop.

Préférence post-audit : **B.1.a en premier** (court, fenêtre de test 1-2h) ; **B.1.b** si B.1.a insuffisant.

**B.2 — Intra-component cross-device split**

*Cible* : Sana 4Kpx 2× V100 16 GiB produit PNG sur les 4 modes.

*Approche* :
- Nouvelle stratégie `intra_component_split` câblée dans solver entre `weight_sharding` et `component_placement_lazy`.
- Critère d'activation R34 : `component_max_peak > max_single_device_vram AND sum(devices_vram) >= component_max_peak`.
- Pour la VAE Sana 4Kpx : split sur upsample stage boundary (e.g. stages 0-1 sur cuda:0, stages 2-3 sur cuda:1). La frontière naturelle est entre upsample::3 (output 2048×2048) et upsample::4 (output 4096×4096). Critère générique : couper sur la première op dont l'output dépasse 25% du budget single-device.
- NBXTensor cross-device transfer (déjà partiellement supporté via `_transfer_to_device` dans `kernels/wrappers.py:add_inplace_nbx` cf POINT 6 H2) à étendre proprement.
- Estimation : 200-400 lignes (stratégie + cross-device wrapper extensions + Triton runtime device-aware dispatch). À l'edge du seuil mandate.

**B.3 — CPU execution path (les deux branches)**

*Cible binaire minimale* : TinyLlama generate sur les 4 modes en cpu_execution = texte cohérent.
*Cible bonus* : Sana 4Kpx CPU → PNG (acceptable si plusieurs heures wall-time).

*Approche Branche A (PyTorch CPU)* :
1. Nouvelle stratégie `core/strategies/cpu_execution.py` : alloue toutes les composantes à `device='cpu'`.
2. Câbler dans `solver.py:519-536` après `zero3` dans les deux cascades (single-GPU et multi-GPU).
3. Vérifier que `apply_cpu_config(involves_cpu=True)` est appelé quand `cpu_execution` est choisie. Si pas — fix le gate.
4. Wiring `torch.set_num_threads(cpu.cores)` + `torch.set_num_interop_threads` au plus tôt dans la pipeline (probablement dans `apply_cpu_config` qui est déjà appelé en CLI). MKL auto-tiré.
5. Test TinyLlama generate sur CPU 84 cores Supermicro.

*Approche Branche B (Triton CPU)* :
1. **Web search obligatoire avant code** : Triton-CPU upstream version stable, install method, coverage.
2. Intégrer Triton-CPU comme dépendance vendored (R25 strict) ou cherry-pick selon stratégie projet.
3. `OMP_NUM_THREADS=profile.cpu.cores` AVANT import triton-cpu (ordre OpenMP).
4. Faire tourner les kernels @triton.jit existants sur Triton-CPU. Pour ops non couvertes, fallback PyTorch CPU + entrée `triton_cpu_coverage_gaps.md`.
5. Test TinyLlama generate sur les modes triton/triton_sequential sur CPU.

*Estimation taille* :
- Branche A : 100-150 lignes (stratégie courte + vérifs câblage).
- Branche B : variable, 100-400 lignes selon coverage Triton-CPU upstream.

**B.4 — CPU offload partiel (hybrid)**

*Cible* : Sana 4Kpx avec VAE sur CPU et reste sur 1× V100 16 GiB → PNG.

*Approche* :
- Nouvelle stratégie `cpu_offload` câblée entre `intra_component_split` et `cpu_execution`.
- Critère : `largest_component_peak > vram_budget AND ram_budget >= largest_component_peak * compute_dtype_factor`. Place les composantes overflow sur CPU, autres sur GPU.
- Cross-device transfer GPU↔CPU à la frontière inter-composantes (via `variable_resolver`).
- Per-component dispatcher selection : GPU components → Triton ou native GPU, CPU components → native PyTorch CPU ou Triton-CPU selon la branche du mode CLI.

*Estimation* : 100-200 lignes en plus de ce qui aura été construit par B.2 + B.3.

### Estimation cumulée vs seuil mandate

Mandate condition #2 = "blocker architectural >300 lignes vraiment hors scope". Estimation cumulée Phase B :
- B.1 : 50-200 lignes (selon sous-option)
- B.2 : 200-400 lignes
- B.3 : 200-550 lignes (variable selon Triton-CPU)
- B.4 : 100-200 lignes (incrémentale)

**Total estimé Phase B** : 550-1350 lignes réparties sur 4 commits indépendants. **Chaque chantier B.X individuellement peut tenir <300 lignes**, donc reste sous le seuil mandate condition #2 quand pris séparément. Si un sous-chantier dépasse seul 300 lignes (probablement B.2 si Triton runtime device-aware nécessite refactor profond), condition #2 sera levée ponctuellement avec remontée immédiate.

═══════════════════════════════════════════════════════════════════
REMONTÉE FIN PHASE A
═══════════════════════════════════════════════════════════════════

**État** : Phase A complète. Audit factuel sourcé. Conclusion :
- Architecture Prism saine, R34 préservée, hooks d'extension propres.
- conv::62 déjà intercepté mais buffer FULL structurel — B.1 nécessite soit drain forcé pre-hook (B.1.a) soit refactor contrat fusion (B.1.b).
- Branche A CPU largement câblée — B.3 branche A léger.
- Branche B Triton CPU non intégrée — web_search prérequis avant B.3 branche B.
- Estimations B.1-B.4 cumulées 550-1350 lignes, chaque sous-chantier <300 lignes en isolation.

**Demandes validation Hocine avant Phase B** :
1. Ordre des sous-chantiers : ordre proposé B.1 → B.2 → B.3 → B.4. OK ?
2. Pour B.1 : préférence B.1.a (drain pre-hook, court) ou B.1.b (refactor fusion contract, plus deep) en première tentative ? Recommandation audit : B.1.a en premier.
3. Pour B.3 branche B : autorisation explicite de faire web_search Triton-CPU (et Triton-Shared, Triton-Linalg pour cherry-picks) avant coder ? Recommandation : oui obligatoire selon mandate "web_search obligatoire après 2-3 itérations".
4. Pour B.2 : confirme que NBXTensor cross-device extensions + Triton device-aware dispatch sont dans le périmètre permis "kernels/ops/ et triton/" ?

Sans validation, audit en attente. Avec validation explicite ou directive de continuer en autonomie pleine, Phase B démarre.
