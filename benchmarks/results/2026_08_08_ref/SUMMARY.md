# Benchmark matrix — 2026_08_08_ref

## What only NeuroBrix runs on this hardware

Every cell below is a documented does-not-run verdict for a competitor at its pinned best on the V100 rig.

- **omni_ming_t2i** [machine] — `vendor_transformers` **cannot run it on this rig**: The vendor's official recipe cannot run on V100: their single-device loading path needs ~42 GB bf16 (model card) > 32 GB, with NO published multi-GPU HF recipe; official snippets hard-pass attn_implementation= 'flash_attention_2' installed from an sm_80+ cu12/torch2.4 wheel — impossible on sm_70 (Dao-AILab #524); vendor validation hardware is H800-80GB/H20-96G, bf16-only (inclusionAI/Ming v1.5 requirements + model card). → NeuroBrix RUNS it (neurobrix_pytorch, neurobrix_triton).
- **omni_qwen3omni** [machine] — `vendor_transformers` **cannot run it on this rig**: The vendor's official recipe cannot run on this rig: documented theoretical minimum 78.85 GB measured WITH flash_attention_2 (Qwen3-Omni memory-table footnote); FlashAttention-2 does not exist on sm_70 (Dao-AILab flash-attention#524/#148/#228 — Ampere+ only), and the sdpa/eager fallback costs MORE than that minimum on the 96 GB heterogeneous pool; vendor-validated dtype is bf16, emulated (not native) on Volta (pytorch#124996). Weights alone: 70.5 GB across 15 shards. → NeuroBrix columns pending.
- **vlm_glm41v** [pinned] — `ollama` **cannot run it on this rig**: Ollama has NO glm4v support at the v0.24.0 pin (or any version): model request ollama#11858 closed as duplicate with no development; no published GGUF carries the vision mmproj (unsloth GLM-4.1V-9B-Thinking-GGUF discussion#1: 'currently only text is supported'); upstream llama.cpp only gained GLM4V vision on 2025-12-16 (PR #18042) and Ollama's vendored llama.cpp does not expose it. → NeuroBrix RUNS it (neurobrix_pytorch, neurobrix_triton).
- **vlm_glm41v** [pinned] — `vllm` **cannot run it on this rig**: vllm==0.7.3 (era pin, 2025-02) predates GLM-4.1V (released 2025-07) — no glm4v model class exists at this pin. → NeuroBrix RUNS it (neurobrix_pytorch, neurobrix_triton).
- **vlm_qwen3vl** [machine] — `vllm` **cannot run it on this rig**: vllm==0.7.3 (era pin, 2025-02) predates the qwen3_vl architecture entirely — no model class exists at this pin (qwen3 family support landed in vLLM 0.8.x). → NeuroBrix RUNS it (neurobrix_pytorch, neurobrix_triton).
- **vlm_qwen3vl** [pinned] — `vllm` **cannot run it on this rig**: vllm==0.7.3 (era pin, 2025-02) predates the qwen3_vl architecture entirely — no model class exists at this pin (qwen3 family support landed in vLLM 0.8.x). → NeuroBrix columns pending.

## omni_ming_t2i — config: machine

| column | result |
|---|---|
| neurobrix_pytorch | 11.94s · 0.40s/step · 50.3G |
| neurobrix_triton | 110.87s · 3.70s/step · 43.5G |
| vendor_transformers | **DNR** — The vendor's official recipe cannot run on V100: their single-device loading path needs ~42 GB bf16 (model card) > 32 GB, with NO published multi-GPU HF recipe; |

## omni_minicpmo_voice — config: pinned

| column | result |
|---|---|
| neurobrix_pytorch | 3.61s · 17.3G |
| neurobrix_triton | 28.84s · 17.5G |
| vendor_transformers | 1.15s · 21.3G |

## omni_qwen3omni — config: machine

| column | result |
|---|---|
| neurobrix_pytorch | error — `RuntimeError('Daemon error: Failed at op aten.mm::0 (aten::mm): CUDA out of memory. Tried to allocate 20.00 MiB. GPU 1 h` |
| neurobrix_triton | error — `RuntimeError('Daemon error: ZERO FALLBACK: vlm flow requires a modality input — provide --input-image, --input-video or ` |
| vendor_transformers | **DNR** — The vendor's official recipe cannot run on this rig: documented theoretical minimum 78.85 GB measured WITH flash_attention_2 (Qwen3-Omni memory-table footnote); |

## vlm_glm41v — config: pinned

| column | result |
|---|---|
| neurobrix_pytorch | 1.43 tok/s · TTFT 0.90s · 20.6G |
| neurobrix_triton | 0.38 tok/s · TTFT 4.25s · 24.1G |
| ollama | **DNR** — Ollama has NO glm4v support at the v0.24.0 pin (or any version): model request ollama#11858 closed as duplicate with no development; no published GGUF carries t |
| vllm | **DNR** — vllm==0.7.3 (era pin, 2025-02) predates GLM-4.1V (released 2025-07) — no glm4v model class exists at this pin. |

## vlm_qwen3vl — config: machine

| column | result |
|---|---|
| neurobrix_pytorch | 0.63 tok/s · TTFT 1.69s · 73.2G |
| neurobrix_triton | 0.25 tok/s · TTFT 5.86s · 62.8G |
| ollama | 54.81 tok/s · TTFT 0.98s · 90.8G |
| vllm | **DNR** — vllm==0.7.3 (era pin, 2025-02) predates the qwen3_vl architecture entirely — no model class exists at this pin (qwen3 family support landed in vLLM 0.8.x). |

## vlm_qwen3vl — config: pinned

| column | result |
|---|---|
| neurobrix_pytorch | error — `RuntimeError('Daemon error: Failed at op custom.rms_norm::0 (custom::rms_norm): Expected all tensors to be on the same d` |
| ollama | 7.90 tok/s · TTFT 5.14s · 30.9G |
| vllm | **DNR** — vllm==0.7.3 (era pin, 2025-02) predates the qwen3_vl architecture entirely — no model class exists at this pin (qwen3 family support landed in vLLM 0.8.x). |

