# Red cells — every cell that does not confirm, its class and what closes it

The canonical record of the matrix's red cells (the owner's method, 2026-09-28, point 5): a finding
is written here once, with its evidence, and a cell leaves this list when a certified-only
confirmation run is green. Append; a closed row keeps its closing line.

Classes: **trace** (a dim frozen at its trace value, fixed in Forge at the source, then re-censused,
certified, confirmed) · **census** (a key a run forms that the census table does not hold) ·
**prism** (a plan that cannot run on the card it is placed on, or a refusal where the rule is a
last-resort offload) · **engine cost** (a real, measured time, not an autotune) · **kernel**.

| model | mode(s) | class | what the cell does | cause, as measured | closes it | evidence |
|---|---|---|---|---|---|---|
| Wan2.2-I2V-A14B | all | trace | fails at `aten.div::9` (vae_encoder), `(1,384,…)` vs `(1,7296,…)` | **read from the graph (2026-09-28 20:56):** `aten.expand::9` broadcasts a channel norm `(1,1,1,28,44)` back to 384 channels, and the traced size argument writes the channel count 384 as an arithmetic EXPRESSION of the `time` symbol (s1, trace 9) — a trace-value collision: at another frame count the expression gives 7 296 (= 384 x 19). The channel is a constant; the tracer must not bind it to a symbol | Forge's collision guard at the source, then the retrace (4-card VMM window, with Wan2.1-I2V) | queue-12/13 gate rows; `vae_encoder/graph.json` expand::9 |
| Wan2.1-I2V-14B-480P | all | trace | fails at `aten.div::62`, `(1,7296,44,104)` vs `(1,384,44,104)` | NOT the same op: its `expand::9` carries a literal channel 384 — and a LITERAL batch 1 (`[1, 384, 1, s2, s3]`), against the rule that batch is always a symbol; the mismatch appears later, at div::62 — to be read the same way | the same window | queue-12/13 gate rows; `vae_encoder/graph.json` expand::9 |
| Wan2.1-VACE-1.3B | all | trace | fails at `aten.div::9` (vae_encoder) | **the same class as Wan2.2-I2V, read from the graph:** `expand::9`'s channel dim is bound to `s1` (time) — target dims per symbol `[s0, s1, 1, s2, s3]`, where the channel must be a literal | the same Forge collision guard, then its retrace | `vae_encoder/graph.json` expand::9 |
| Open-Sora-v2 | — | trace | frame-causal mask frozen | static scan | Forge fix, retrace | static scan |
| Sana_1600M_4Kpx_BF16 | — | trace | a frozen dimension | static scan | retrace | static scan |
| CogVideoX-2b | triton | trace | fails at `native_group_norm::20` | frozen pos_embed table (index_select arange on the trace grid) | Forge fix, retrace | queue-12/13 gate rows; session 2026-09-26 |
| mochi-1-preview | triton, tseq | engine cost | stops at the gate's 900 s cap | long-cap run: green at 2 801 s / 3 394 s | the Wan/mochi Triton step-cost item | longcap 2026-09-28 |
| Flex.1-alpha | triton, tseq | engine cost | stops at the 900 s cap | long-cap run: triton 1 797 s, tseq 396 s, apples | the same item | longcap 2026-09-28 |
| Qwen3-Omni-30B | triton | engine cost | 911 s on the queue-13 gate (cap 900) | output byte-identical three times, generation equal or faster; the gap is load time under host load | a confirmation at the smallest request | q13_attrib 2026-09-28 20:18 |
| Ming-Lite-Omni-1.5 | 16 GB, all | prism | refused at every 16 GB rung: "no strategy can fit model + KV cache" | a refusal where the owner's rule is CPU offload as a last resort | Prism last-resort branch | census_main 16g, 2026-09-28 |
| Allegro | triton, 16 GB | prism | out of memory on the 16 GB card | the Prism budget class | Prism | verify_16g 2026-09-28 |
| granite-speech-3.3-8b | Apple native | prism | swapped 16 GB, killed | a streamed segment's load keeps a host copy on unified memory (the Mac's loader probe) | loader + Prism unified branch | archives/mac/granite_plan_2026_09_28/ |
| chatterbox, openaudio-s1-mini, orpheus-3b-0.1-ft | triton, tseq | census | 32 / 26 / 22 keys formed that no census held | the census never walked the codec's generated length | the audio extent walk (deb76f2e, 76823c51) + the one-row W ladder, then the table | verify_16g / verify_32g 2026-09-28 |
| Qwen3-VL-30B-A3B-Thinking | 32 GB | census | 2 keys formed that no census held | the census shadow diverges at `aten.index_put::0` | the census tool | verify_32g 2026-09-28 |
| Janus-Pro-7B | layer_streaming | prism | a layer_streaming defect | not reproducible on 16 GB | a 32 GB reproduction | 2026-09-27 |
