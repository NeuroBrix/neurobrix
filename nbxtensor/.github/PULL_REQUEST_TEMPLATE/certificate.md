## Certificates for a GPU class

**GPU model** (as `nvidia-smi -L` or the system reports it):
**Driver version:**
**Operating system / Python:**
**Versions** (`pip show nbxtensor neurobrix triton`):

**Models run** (the `neurobrix run` commands whose keys were certified):

```
neurobrix run ...
```

**Certification command:**

```
neurobrix autotune certify --profile <profile> --keys <...> --out <dir>
```

**Check — the certified-only run, served from these certificates** (paste its `[autotune] certified directory:` line and
the output's sha256):

```
NEUROBRIX_AUTOTUNE_CERTIFIED_DIR=<dir> neurobrix run ... --certified-only
```

- [ ] the files are added under `src/neurobrix/config/autotune/<vendor>/<profile>/` of the NeuroBrix repository
- [ ] the certified-only run above served every key (0 swept at runtime) and its output matches the uncertified run
