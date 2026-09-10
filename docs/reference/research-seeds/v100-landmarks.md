# Published V100 throughputs — landmarks, not bench columns

**These are not measurements of this engine and they are not comparable to one.**
They exist for one purpose: to check that the gaps this project reports have a
credible order of magnitude. A number below that sat beside one of ours in a
table would be a lie by layout.

Every figure was measured under conditions this file does not know and cannot
reconstruct — batch size, quantisation, context length, clock state, driver and
framework version, whether the run was warm. Two of them disagree by a factor of
eight on the same card, which is itself the argument: the spread is the
conditions, not the hardware.

Consulted **2026-09-10**.

| figure | model / condition as stated by the source | source |
|---|---|---|
| 98.8 tok/s | Qwen 3.6, MoE 35B, llama.cpp, V100 32GB | miyagadget.page, "Tesla V100 32GB Local LLM Benchmark", 2026-07-15 |
| 32.9 tok/s | dense 27B, Q4_K_M quantisation, llama.cpp, V100 32GB | same source |
| 12.32 tok/s | Qwen3 30B at maximum context (70 000 tokens), V100 32GB | hardware-corner.net, "I Tested the Tesla V100 32GB for Local LLM" |
| multi-GPU scaling figures | 3×V100, vLLM | databasemart.com, "3×V100 vLLM Benchmark" |
| cross-accelerator survey | several frameworks and accelerators, V100 included | arXiv:2411.00136, *LLM-Inference-Bench* |

## How to use them, and how not to

**Use:** if this engine reported 900 tok/s on a dense 27B on one V100, that
number would be wrong and these landmarks would say so before any gate did. That
is the entire service they render.

**Do not use:** never as a target, never as a baseline, never in the same table
as a measured column, never as the "before" of a speedup. A comparison against a
number whose conditions are unknown is not a bench — it is an intuition wearing a
bench's clothes.

The engine's own comparisons are the paired arms of a campaign, on one machine,
with the clocks locked, both arms cold on the same three things, and the repeats
interleaved. That protocol exists precisely because a single number carries none
of it.
