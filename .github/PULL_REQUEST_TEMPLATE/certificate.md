## Certificates for a GPU class

<!-- Thank you for certifying your GPU. Open this pull request with ?template=certificate.md in its URL.
     The steps are in the README, "Certify your GPU and share it". -->

### The GPU

- GPU model (as `nvidia-smi`, `rocm-smi`, `xpu-smi` or `system_profiler` names it):
- Memory:
- Driver version:
- The device file it ran under (`--profile`, e.g. `volta`):
- Operating system, Python, Triton and nbxtensor versions (`python -m nbxtensor doctor` prints them):

### The commands you ran

<!-- The commands of steps 1 to 3 as you typed them: the models you ran, the certify command, the certified-only run. -->

```
```

### The certified-only run (step 3)

<!-- Paste its output: the `[autotune] certified directory:` line must say `0 swept at runtime`, and the model's output
     must be the same as in step 1. -->

```
```

### The files

- [ ] The files of `./my-certificates/<vendor>/<profile>/` are added under `src/neurobrix/config/autotune/<vendor>/<profile>/`
- [ ] No other file is changed
