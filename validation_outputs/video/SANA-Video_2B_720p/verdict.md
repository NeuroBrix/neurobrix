# SANA-Video_2B_720p — 4-mode matrix verdict

Agent verdict: **4/4 COHERENT** — sharp red fox walking in snow, temporally
consistent, in sequential / compiled / triton-sequential / triton, with
cross-mode agreement <= 2/255 vs the sequential oracle. Clean runs (no env
overrides): dtype handled by the registry `requires_fp32_compute` flag on the
transformer (vendor prescribes bf16; V100 fp16 overflows at step 1); text
encoder and LTX2 VAE run fp16.

Eight root causes were fixed for this closure (forge: 9bfb448 config mirror,
c69c2a1 symbolic rope, 90328c4 de-collision + view redistribution + unbind
rule, 36fc554 functional rotary + constant-aware de-collision, 52c3f9f dtype
flag; NBX: e09b8bc alias-preserving copy, d229f9a strided slice + promotion
guard). See INDEX.md rows for the per-mode narrative.

Relaunch (any mode):
neurobrix run --model SANA-Video_2B_720p_diffusers --prompt "a red fox walking in snow" \
  --height 704 --width 1280 --num-frames 9 --steps 30 --seed 42 --<mode> --output out.mp4

Hocine validation: TODO
