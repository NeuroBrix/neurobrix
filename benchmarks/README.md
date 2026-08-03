# NeuroBrix Benchmarks — the three-column harness

Reproducible comparison of NeuroBrix against the reference open-source
serving stacks, on the same hardware, the same weights, the same
prompts, the same precision. The goal is a number the reader can
re-derive, not a marketing table: every published figure carries the
exact command, environment manifest and raw timing artifact that
produced it.

## Columns

| column | what runs |
|---|---|
| vLLM | upstream vLLM serving the reference weights |
| Ollama | upstream Ollama serving an f16 GGUF of the same weights |
| NeuroBrix / PyTorch | `neurobrix serve` — compiled engine (mode 1) |
| NeuroBrix / Triton | `neurobrix serve --triton` — pure-Triton engine (mode 2) |

Rows where a competitor column cannot run the workload at all (e.g.
diffusion image generation) record `not applicable` for that column —
never a silently missing cell.

## Fairness rules (all mandatory)

1. **Same GPU, one column at a time** — each measurement pins one
   physical GPU (`CUDA_VISIBLE_DEVICES`), no co-tenant compute; the
   harness refuses to start if the target GPU has any running process.
2. **Same weights lineage, fp16 end-to-end** — every column serves the
   same upstream checkpoint in half precision. Ollama uses an f16 GGUF
   conversion of the identical checkpoint (no quantization anywhere).
3. **Warm serving for every column** — each backend runs its own
   persistent server; one warmup request precedes measurement; server
   start-up cost is reported separately (cold-start column), never
   mixed into throughput.
4. **Same prompts, same generation contract** — the fixed prompt set
   in `config/rows.yml`, greedy decoding, identical `max_new_tokens`;
   streaming enabled so first-token latency is measured at the wire.
5. **Repetition** — N=5 timed requests per cell; report median and
   min–max spread; raw per-request timings kept in the artifact.
6. **Pinned environment** — every dated run writes an environment
   manifest (backend versions, torch/CUDA versions, driver, GPU model,
   git commit) next to its results; version pins live in
   `config/backends.yml` once sourced.

## Metrics

- **tokens/s** — decode throughput: generated tokens / (last-token
  time − first-token time), per request.
- **TTFT (ms)** — request sent → first streamed token.
- **peak GPU memory (MiB)** — sampled via `nvidia-smi` during the
  measurement window (1 Hz), maximum over the cell.
- **cold start (s)** — server launch → first successful response
  (reported separately, rule 3).

## Rows (phase 1 — see `config/rows.yml`)

- LLM dense: TinyLlama-1.1B-Chat
- LLM MoE: deepseek-moe-16b-chat
- VLM: Qwen3-VL row (image + prompt)
- Diffusion image: Sana 1600M 1024px (NeuroBrix columns only —
  competitor columns `not applicable`)

## Artifacts

```
benchmarks/results/<YYYY_MM_DD>/
├── env_manifest.json
├── <row>_<column>.json      # raw per-request timings + metrics
└── SUMMARY.md               # the four-column table for the date
```

Published numbers reference a dated artifact directory, always.

## Status

- 2026-08-03 — methodology + phase-1 rows recorded (this lot).
  Backend availability on the reference rig: vLLM and Ollama not yet
  installed; their version pins must be sourced against the rig's
  hard ceilings (CUDA driver 535 → torch ≤ 2.5.1+cu121, V100 sm_70)
  before install, and recorded in `config/backends.yml`. The runner
  lands with the pins.
