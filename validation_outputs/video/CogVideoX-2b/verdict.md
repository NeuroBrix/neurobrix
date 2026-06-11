# CogVideoX-2b — 4-mode matrix verdict

Agent verdict: **4/4 COHERENT** — red fox walking in snow, treeline,
temporally consistent (f9 = 480x720x9, 30 steps) in sequential / compiled /
triton-sequential / triton. Closure required the dead-output liveness rule
in all four engines (NBX f724772 / e9278fb / 80dc152), compiled dead
arena-slot kills + native_group_norm derived scalars, and the CogVideoX DDIM
scheduler variant in both engines (bcf09dd, vendor-oracle gated).

Relaunch (any mode):
neurobrix run --model CogVideoX-2b --prompt "a red fox walking in snow" \
  --height 480 --width 720 --num-frames 9 --steps 30 --cfg 6.0 --<mode> --output out.mp4

Hocine validation: TODO
