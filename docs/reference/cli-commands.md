# CLI commands — the list the engine ships

Generated from `neurobrix --help` and **pinned against it** by
`tests/unit/cli/test_documented_commands_match_the_cli.py`.

It exists because the two lists had drifted with nobody comparing them: a user
reported `neurobrix hub` as undocumented, and the audit found the opposite gap —
`autotune` and `drift` shipped and were described nowhere. A command the engine
offers and no page names is a command nobody will use.

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

A test that exercises nothing is green, and this project has paid for that three
times in three days. `coverage` answers, before a cell is written, whether a
model run can reach the code the cell claims to validate.

```
neurobrix coverage aten::tril      # 22 of 56 installed containers, named
neurobrix coverage aten::var       # 0 of 56 — no model run can reach it
neurobrix coverage --rarest 25     # what the catalogue barely exercises
neurobrix coverage --unreached     # what it classifies and never carries
neurobrix coverage --field temperature
```

It reads the containers, so it answers questions about the **graph**. It cannot
answer a question about a **runtime decision** — which attention branch ran,
which autotune config was seated, whether a fallback fired. Those are chosen
while running and are not nodes in any graph; asking this command about them
would return a confident wrong answer. That class needs its own instrument, on
the execution path.

A count of zero means "no container on THIS machine carries it", never "dead
code". The hub holds more than one machine does.
