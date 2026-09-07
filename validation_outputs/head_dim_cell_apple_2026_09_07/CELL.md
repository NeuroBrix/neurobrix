# The head_dim cell — every decoder at a context length equal to its head dimension

Generated **2026-09-07T22:24:32+02:00** by `tools/head_dim_length_cell.py` on Darwin 25.6.0 arm64.

The length is computed from each model's own config, not chosen: it is the one length at which a K in (b, h, head_dim, seq) and a K in (b, h, seq, head_dim) have the same shape, so an engine that reads the layout off the shape has no answer there — and gave a wrong one, in silence, on every model.

Every arm is compared against the **sequential oracle** (`--sequential`, the ATen op-by-op path) on the same exact token ids; llama-like decoders are also compared against `tools/llama_fp64_oracle.py`.

## `after` — source tree `e6d8058`

| model | head_dim | `--compiled` | `--sequential` | `--triton` | `--triton-sequential` | float64 oracle |
|---|---:|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | 64 | `8138fa86` identical | `8138fa86` identical | `8138fa86` identical | `8138fa86` identical | argmax 29871 @ 7.154 — 4 arms agree |

## `before` — source tree `e6d8058 + uncommitted changes`

| model | head_dim | `--compiled` | `--sequential` | `--triton` | `--triton-sequential` | float64 oracle |
|---|---:|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | 64 | **void** — changing the last token id did not change the generated ids, on THIS tree and THIS machine. Two readings fit and this row cannot choose between them: the engine did not run on the ids given, or its answer at this length does not depend on that token. Decide it by comparing a tree or a rig where the same control DOES move — measured 2026-09-07, the same before-engine left the answer unmoved on Apple and moved on V100, so an unmoved control is not a signature of the defect, only a reason this row proves nothing. | — | — | — | argmax 29871 @ 7.154 — no arm agrees, 4 disagree |

## What changed between the trees

### `after` against `before`

| model | arm | after | before | verdict |
|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | `--compiled` | `8138fa86` | `6459ce0b` | **changed — it was wrong at this length** |
| TinyLlama-1.1B-Chat-v1.0 | `--sequential` | `8138fa86` | `6459ce0b` | **changed — it was wrong at this length** |
| TinyLlama-1.1B-Chat-v1.0 | `--triton` | `8138fa86` | `e05d4d35` | **changed — it was wrong at this length** |
| TinyLlama-1.1B-Chat-v1.0 | `--triton-sequential` | `8138fa86` | `f1b4f3a7` | **changed — it was wrong at this length** |

