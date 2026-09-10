# Research seeds — numbers found outside, and what they are worth

Nothing in this directory is a proof, and nothing in it may become a certified
entry. The certified directory is
`src/neurobrix/config/autotune/<vendor>/<profile>/`, every entry there carries
its own measurement against the fp64 oracle at the profile's tolerance, and the
distance between that and what lives here is the whole point of the directory.

## The rule, once, for every number found outside

1. **An external number is an order-of-magnitude control, never a reference.**
   It was measured under conditions nobody here knows — batch size,
   quantisation, context length, clocks, stack version — so a comparison
   against it is not a bench, it is an intuition.
2. **It may be a research seed**: a starting neighbourhood for a sweep, which
   is worth real time on hardware we do not own.
3. **It may never enter a table of hardware limits.** The driver gives the
   real one. `config/vendors/KNOWN_TARGETS.md` says why, and a test enforces it.

---

## vLLM tuned Triton MoE configurations

**Harvested 2026-09-10** from `vllm-project/vllm`, directory
`vllm/model_executor/layers/fused_moe/configs`, **Apache-2.0** — compatible with
this project's licence. Index with a URL and a size per file:
`vllm_fused_moe_index.json`. **334 files** at harvest.

Each file is named by the number of experts, the per-rank intermediate size, the
device name and sometimes the dtype (`E=128,N=768,device_name=NVIDIA_B200.json`),
and gives, per batch size, `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps` and
`num_stages`.

### What they are worth

**They carry no proof.** No entry was validated against an oracle, none carries
a tolerance, and none states the conditions it was measured under.

Upstream says so itself. `vllm-project/vllm#25858`, opened **29 September 2025**,
records that the shipped file
`E=16,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json` **crashed
Jamba** and had to be re-tuned. A configuration that ships is not a configuration
that was validated, and the file that proves it is one of these files.

Nor is it an isolated accident: the `illegal memory access` class recurs across
that tracker through 2025 — #26720, #29361, #26558, and the MoE-Marlin alignment
defect #47769.

**They are tuned for vLLM's kernel, not ours.** The knobs share names because
both are Triton. What happens between the knobs does not.

**So they never become certified entries.** They become a *starting
neighbourhood* for a sweep: on hardware we do not own, a user certifying for the
first time can begin from a point someone has already found workable instead of
from the middle of the space. They become certified only if our own
certification validates them against the fp64 oracle at the profile's tolerance,
at which point what is certified is our measurement, not their file.

### Coverage — and what it means for the campaign running now

The catalogue's device names, counted from the harvest:

| device | files |
|---|---|
| NVIDIA H100 (80GB HBM3 and plain) | 67 |
| NVIDIA H200 | 51 |
| NVIDIA B200 | 35 |
| AMD Instinct MI300X | 34 |
| NVIDIA A100 (SXM4 40GB and 80GB) | 35 |
| AMD Instinct MI325X / MI325_OAM | 22 |
| NVIDIA H20 / H20-3e | 32 |
| NVIDIA GB200 | 11 |
| AMD Instinct MI350/MI355 (OAM, X) | 15 |
| NVIDIA A800 | 6 |
| NVIDIA L40S | 4 |
| NVIDIA RTX PRO 6000 Blackwell | 6 |
| NVIDIA B300 | 3 |
| AMD Radeon R9700 | 2 |
| NVIDIA GB10 | 2 |
| NVIDIA GeForce RTX 4090 | 2 |

**There is nothing for Volta. Zero files of 334.** Checked directly against the
repository listing, not inferred: `volta_entries` in the index is an empty list.

So this harvest **does not accelerate the campaign running on this rig by one
second**, and no plan may be founded on it. Its value is entirely for a future
user certifying on H100, H200, B200, B300, MI300 or a Blackwell consumer part —
hardware this project does not own.
