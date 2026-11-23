# CLI commands — the list the engine ships

Generated from `neurobrix --help` and **pinned against it** by
`tests/unit/cli/test_documented_commands_match_the_cli.py`.

Every command the engine offers is named here. The full options of each are in the
[CLI reference](cli.md).

| command | what it does |
|---|---|
| `run` | Run inference using NBX Engine |
| `calibrate` | Measure the precision calibration record of a model |
| `autotune` | The certified autotune directory: certify a profile, |
| `drift` | Name the first op where the Triton engine drifts from |
| `coverage` | Which installed containers actually reach a symbol |
| `info` | Display system information |
| `inspect` | Inspect a .nbx file or an installed model |
| `import` | Download model from NeuroBrix registry |
| `processor` | A model's sub-components: fetch them ahead, carry them to an offline machine, list them |
| `list` | List installed models |
| `remove` | Remove a model (cache, store, or both) |
| `clean` | Wipe all downloaded models (store and/or cache) |
| `hub` | Browse models on the NeuroBrix registry |
| `serve` | Start persistent model serving daemon |
| `chat` | Interactive chat with running daemon |
| `stop` | Stop the serving daemon and free VRAM |
| `doctor` | Diagnose installation problems (PATH, PyTorch/CUDA, |
| `validate` | Validate NBX file integrity |
| `upscale` | Upscale an image using a super-resolution model |

## `coverage` — the command that says whether a test can prove anything

`coverage` says which installed containers carry an operation, a symbol or a
request field, so you know whether a model run can reach the code you want to test.

```
neurobrix coverage aten::tril      # the installed containers that carry it, named
neurobrix coverage aten::var       # 0 means no installed model run can reach it
neurobrix coverage --rarest 25     # what the installed models barely exercise
neurobrix coverage --unreached     # what it classifies and never carries
neurobrix coverage --field temperature
```

It reads the containers, so it answers questions about the **graph**. It cannot
answer a question about a **runtime decision** — which attention branch ran,
which autotune config was seated, whether a fallback fired. Those are chosen
while running and are not nodes in any graph, so this command cannot answer
about them.

A count of zero means "no container on THIS machine carries it", never "dead
code". The hub holds more than one machine does.
