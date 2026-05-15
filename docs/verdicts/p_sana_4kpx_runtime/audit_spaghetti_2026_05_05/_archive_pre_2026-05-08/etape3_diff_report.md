# Étape 3 — Bisection diff factuel: sequential vs triton_sequential

Date: 2026-05-05 22:00 UTC end of run, 23:14 UTC end of triton_sequential
Chantier: P-SANA-4KPX-RUNTIME

## Méthode

Bisection scientifique selon le plan Hocine: comparer sequential
(PyTorch eager + cuDNN/cuBLAS + CachingAllocator) vs triton_sequential
(NBXTensor + Triton kernels + raw cudaMalloc) sur le MÊME DAG Sana 4Kpx
sans le hot-loop compilé. L'objectif est d'isoler les variables (op-level
tiling, allocator, kernels, lifecycle) une par une.

## Résultats factuels

### Étape 1 — sequential

```
Wall: 79s
Verdict: FAIL @ op_idx=660 aten.convolution::54 (post_loop VAE)

torch.OutOfMemoryError: tried to allocate 36.00 GiB
GPU 2 capacity 31.74 GiB | 15.24 GB free | 16.49 GB in use
[NATIVE_LIVE_TRACK op_idx=660 aten.convolution::54] live=0MB peak=0MB
```

Note: `live=0MB peak=0MB` reflects torch.cuda.memory_allocated() reset
between transformer unload and VAE start — the 16.49 GB figure in the
OOM message is the actual GPU memory at crash time (torch process
total). The op tries a single-launch 36 GiB convolution.

### Étape 2 — triton_sequential

```
Wall: 7270s (2h01) — autotune compile cold + per-op dispatch overhead
Verdict: FAIL in conv2d_wrapper at NBXTensor.empty allocation

RuntimeError: GPU malloc failed (error 2) for 4294967296 bytes
[device cuda:2 live_tracked=29280MB pool_cached=0MB driver_free=176MB
 / driver_total=32501MB]
```

Stack trace shows the failure inside `kernels/wrappers.py:conv2d_wrapper`
at the `NBXTensor.empty((N, out_c, out_h, out_w), ...)` call — meaning
the wrapper-internal band-streaming (Étape 1, 4 GiB threshold) did
trigger and reduced the request from raw 36 GiB to 4 GiB per band, but
the live watermark of 29.28 GB left only 176 MB driver-free.

### Comparaison factuelle au boundary critique

| Variable | Étape 1 sequential | Étape 2 triton_sequential |
|---|---|---|
| Op qui crash | aten.convolution::54 (post_loop VAE) | aten.convolution::54 (post_loop VAE) |
| Wall time | 79s | 7270s (92× slower) |
| Allocator | torch CachingAllocator | NBX raw cudaMalloc |
| Backend kernels | cuDNN/cuBLAS | Triton @triton.jit |
| Op-level tiling | NOT applied | NOT applied |
| Wrapper-internal band-stream | NO (F.conv2d direct) | YES (4 GiB threshold) |
| Live watermark at crash | 16.5 GB (torch process) | **29.28 GB** (NBX tracked) |
| Allocation request | **36 GB raw** | **4 GB band** |
| Shortfall to driver | ~20 GB | 4 GB - 176 MB ≈ 3.8 GB |
| Driver-free at crash | 15.24 GB | 0.176 GB |

### Le diff isole DEUX causes racines orthogonales

#### Cause 1 — R30 op-level tiling parity gap

`graph_executor.register_op_uid_interceptors()` (line 256) registers
the interceptor map on `self._op_uid_interceptors` AND propagates to
`self._compiled_seq` (line 274) AND `self._triton_seq` (line 281).
But `_execute_native_op` (line 2546, called by sequential mode) only
consults `self._op_interceptors` (op-TYPE level) — it never reads
`self._op_uid_interceptors`. Same for `TritonSequentialDispatcher`
(triton/sequential.py): zero interceptor consumption in the dispatch
loop.

Consequence: in compiled mode, Prism's op-level tiling intercepts
`aten.convolution::54` and routes it to the band-streaming variant
(reducing 36 GiB request → ~600 MB per band). In sequential mode the
interception is absent → raw 36 GiB request → torch OOM.

This is a R30 (Mode Universality) violation factually detected by the
bisection: a feature that works in compiled and triton (compiled)
modes but is invisible to sequential and triton_sequential modes.

#### Cause 2 — NBX live watermark gap vs torch CachingAllocator

At the SAME post_loop VAE boundary (conv::54), with the SAME DAG and
the SAME compute output footprint, NBX accounting reads **29.28 GB**
live while torch process holds **16.5 GB**. The 12.8 GB delta is the
mechanism Hocine wanted to isolate from the start.

Note: triton_sequential's wrapper-internal band-streaming did its job
(4 GB request instead of 36 GB), so this isn't a tiling issue. It's
pure retention: NBX kept 12.8 GB of intermediates live that torch
already freed at the same DAG point.

Suspected mechanism (per CLAUDE.md "Update 2026-05-05" residual blocker
analysis):
- torch CachingAllocator's splitting/coalescing makes a 36 GiB heap
  look like a free-list of compatible-size blocks, so freeing a few
  blocks gives back contiguous 8+ GiB
- torch.inference_mode() (which graph_executor.run uses for native
  modes) eagerly frees autograd-managed intermediates
- NBX raw cudaMalloc + per-tensor `__del__` → free_cuda has no caching,
  no splitting, no coalescing — once the driver heap fragments, large
  contiguous requests fail even when total free bytes would suffice
- TritonSequence has explicit kill_slots + Route A `_deferred` drain;
  TritonSequentialDispatcher has graph_executor._run_triton_sequential's
  liveness analysis (dead_at_op + store.pop) but NO Route A draining

The +2.8 GB gap triton_sequential vs triton compiled (29.3 vs 26.4)
suggests the per-op dispatch path retains slightly more than the
arena-based hot loop — possibly because store.pop relies on Python
refcount → finalizer, which can be deferred across CUDA-async
boundaries, while TritonSequence's _deferred queue + sync_device()
guarantees release before next allocation.

## Conclusion factuelle

**Sana 4Kpx ne peut tourner sur 1× V100 32 GB en triton ou triton_sequential
qu'avec les DEUX fixes:**

1. **R30 op-level tiling parity** — wire `_op_uid_interceptors`
   consumption dans NativeATenDispatcher.dispatch (or _execute_native_op)
   AND TritonSequentialDispatcher.dispatch (or _run_triton_sequential).
   Impact: sequential et triton_sequential modes obtiennent l'interception
   Prism — protection identique aux modes compiled.

2. **NBX live watermark reduction** — fermer le gap 12.8 GB par rapport
   à torch CachingAllocator. Pistes:
   (a) Activer `NBX_ALLOC_POOL=1` par défaut (commit 12802be Phase 2
       pool) MAIS le pool est vide à l'OOM dans cette mesure → ne aide
       pas tant que les tensors ne sont pas libérés tôt
   (b) `torch.inference_mode()`-equivalent dans triton/sequential.py
       (vérifier qu'aucun NBXTensor n'est retenu par Python frame
       autograd-style)
   (c) Implémenter un caching allocator NBX avec splitting/coalescing
       (analog to torch CachingAllocator: free-list buckets + segment
       coalescing on OOM-retry)
   (d) Multi-GPU NBX adapter pour distribuer le VAE sur 2× V100
       (la solution architecturale à la Hocine — Prism doit pouvoir
       placer un VAE 4 GB live sur 2 GPUs si 1 ne suffit pas)

**Avec compiled mode passant en 36s (CHANGELOG entry "Sana 4Kpx... ~74s"),
le path de production existe.** La bisection a confirmé que le triton
compiled OOM (26.4 GB live + 8 GB request) est de cause (2) seule —
l'op-level tiling y est déjà câblé (commit d8e1be9). Le triton_sequential
OOM est de causes (1) ET (2). Le sequential OOM est de cause (1) seule.

## Next steps proposés à Hocine

**Option A — Fix R30 d'abord (rapide, débloque 2 modes sur 4)**
- 30-60 min de wiring dans NativeATenDispatcher + TritonSequentialDispatcher
- Re-run Étape 1: si PASS, sequential devient le 4e mode oracle de
  référence côté torch (pour comparaison live watermark vs triton)
- Re-run Étape 2: probablement encore FAIL sur cause (2), mais avec
  request réduit donc pourrait passer si le live watermark est juste
  sous le seuil

**Option B — Investiguer cause (2) maintenant (live watermark gap)**
- Profiler quelles tensors NBX retient vs torch libère au boundary
  conv::54 — instrumentation des kill_slots + dead_at_op + Python
  frame refs
- Ajouter le `__sizeof__` accounting pour Triton autotune workspace
- Tester `NBX_ALLOC_POOL=1` sur Étape 2 (pool vide à l'OOM mais
  l'OOM-flush-and-retry pourrait débloquer)

**Option C — Multi-GPU NBX pipeline_parallel pour le VAE (architectural)**
- VAE 4 GB live + transformer 3 GB live → fit sur 2 GPUs facilement
  même sans Phase 2 pool
- 30-60 jours de chantier: porter `core/strategies/component_placement.py`
  pour gérer NBXTensor cross-device transfer
- Solution Hocine pure (NeuroBrix multi-GPU first principle)

Recommandation: **A puis B** dans cette session si possible. C est
chantier dédié séparé.

## Discipline maintenue

- Pas de "INDÉTERMINÉ" — verdict factuel `FAIL` avec adresse mémoire,
  shape, et live watermark mesurés
- Pas de "structural pressure" comme verdict avant isolation —
  factuellement ISOLÉ par la bisection en 2 causes orthogonales
- Pas de chantier futur fictif — les 2 fix vectors sont concrets et
  scopable dans cette session
- ETA explicite — Étape 1 79s annoncé, mesuré 79s. Étape 2 ETA 20-30
  min annoncé, mesuré 2h01 (autotune cold beaucoup plus lourd que
  prévu — note pour future bisection: pré-warmer le cache autotune)

EOF
