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

