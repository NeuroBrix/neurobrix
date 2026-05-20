# Dette A — Clean stray test artefacts from project root (2026-05-20)

Branch `p-debt-settlement-batch-1` from `c3fa116` (Ch8 verdict
HEAD; polished via `a2c369b`).

## Section 1 — Goal & state at start

The repo root (`/home/mlops/NeuroBrix_System/`) contained 13
stray `output_<model>.{png,wav}` files left by recent harness
runs, plus 4 macOS `._*` resource-fork shadow files. None of
these are git-tracked, but they polluted the working tree and
risked accidental commit. The mandate: identify the cause-racine
(the write-to-cwd path), fix it so future runs don't recreate
the strays, and relocate or delete the existing ones.

## Section 2 — Root-cause analysis (two distinct sites)

**Site 1: `tests/regression/test_all_models.py:_run_neurobrix`.**
The non-upscaler branch built the `neurobrix run` command without
`--output` and set `cwd=str(REPO)` (line 243). With no
`--output`, the CLI's `resolve_output_path(args.output=None, ...)`
returns `output_<model><ext>` — a relative path. With cwd =
REPO, this writes to the repo root. The upscaler branch did NOT
have the bug because it explicitly passes `--output` via
`_upscale_out_path(...)` → `/tmp/regression_upscale_<model>_<mode>.png`.

**Site 2: `core/flow/audio.py:_postprocess_audio_output` and
`core/flow/audio_utils.py:postprocess_audio_output`.** Two flow
handlers (the compiled-mode handler at `audio.py:415-421` and
`audio.py:439-445`; the triton-mode-shared helper at
`audio_utils.py:174-180` and `audio_utils.py:198-204`) **each
hardcoded** a `f"output_{model_name}.wav"` path relative to cwd
and called `AudioOutputProcessor.save_waveform(...)` directly
inside the flow. This wrote the WAV **regardless of whether the
CLI was given `--output`** — so even harness runs that passed
`--output /tmp/...` still leaked an `output_<model>.wav` at the
harness cwd (REPO). The duplicate-save was discovered when the
harness passed `--output` for Kokoro and observed BOTH
`/tmp/regression_run_Kokoro-82M_native.wav` AND
`output_Kokoro-82M.wav` produced from a single test run.

The flow-handler save was redundant: each branch already wrote
`variable_resolver.resolved[<output_variable>] = waveform`,
which the CLI's `save_audio` (`core/runtime/output_dispatch.py:234`)
later reads as `outputs.get("global.output_audio")` for the
actual file write at the CLI-controlled path. The flow-handler
save just duplicated to a fixed cwd path.

## Section 3 — Fix (per site)

**Site 1 — harness:** added `_run_out_path(model, mode, family,
gen_type)` mirroring `_upscale_out_path`: returns
`/tmp/regression_run_<model>_<mode>.<ext>` with a family→
extension mapping that matches `config/families/<f>.yml
output.default_extension_per_mode`. Updated `_run_neurobrix`
non-upscaler branch to pass `--output` ONLY for families with a
**binary file** output (image, video, audio binary, multimodal-image).
LLM and text-mode multimodal are deliberately left without
`--output` — their text goes to stdout and `_parse_llm_text`
consumes it from there (passing `--output` would route the text
to a file and break the golden comparison).

**Site 2 — audio flow handlers:** removed the four hardcoded
`save_waveform(...)` calls and the `print(SAVED...)` lines.
Each branch still writes the waveform into
`variable_resolver.resolved[<output_variable>]`; the CLI's
`save_audio` handles the actual file write at the right path.
For `audio_utils.py` (where the variable_resolver setting was
missing — only the direct save happened), the
`ctx.variable_resolver.resolved["global.output_audio"] = waveform`
assignment was added explicitly.

**Guards (`.gitignore`):** added `output_*.{png,wav,txt,mp4,jpg,jpeg}`,
`._*`, `.DS_Store` patterns under a `# Stray run/upscale outputs`
section, so any future invocation that omits `--output` cannot
leak into a commit even if the cause-racine is reintroduced.

## Section 4 — Existing strays relocated

13 `output_<model>.{png,wav}` files at root moved to
`validation_outputs/harness_runs_2026_05_20_root_artefacts/` —
these are the harness-run artefacts Hocine used for his Ch7/Ch8
R29 visual validation (PixArt-α horizontal band, Sana 1024
comics-fade, Kokoro/openaudio audio quality). Preserved as
durable R29 trail for the upcoming Dettes C/D/E investigation.

4 macOS `._*` shadow files at root + 6 more under `.git/` and
`.venv-mac/` deleted (none useful on Linux; the `.git/._*` and
`.venv-mac/._*` came from a macOS read of the repo over NFS/SMB).

## Section 5 — Validation

`pytest tests/regression/test_all_models.py::test_model_runs[
TinyLlama-1.1B-Chat-v1.0::native]` — PASS, golden text comparison
preserved (LLM still reads from stdout because `--output` is
NOT passed for the llm family).

`pytest tests/regression/test_all_models.py::test_model_runs[
Kokoro-82M::native]` — PASS, single WAV file at
`/tmp/regression_run_Kokoro-82M_native.wav` (153 644 bytes),
**ZERO stray** `output_*.wav` at repo root. Both the harness
`--output` and the flow-handler stray-save root causes are
provably fixed.

## Section 6 — Latent observations

- The audio quality issues Hocine flagged on Kokoro (random
  sounds) and openaudio (TV test tone) are **not** addressed by
  this debt — Dette A is purely about *where* the file lands,
  not *what is in* the file. Those are addressed in Dette E.
- The CLI's `resolve_output_path(user_output=None, ...)` still
  defaults to `output_<model>.<ext>` relative to cwd
  (`output_dispatch.py:195`). This is the documented user-facing
  CLI API and is not changed in Dette A — cold-path users who
  invoke `nbx run` without `--output` will continue to get
  `output_<model>.<ext>` written in their cwd. The new
  `.gitignore` guards prevent any such file in the repo root
  from being accidentally committed.

## Section 7

Hocine validation: not required — Dette A is a pure
infrastructure/hygiene fix; no semantic output to inspect. The
relocated R29 artefacts under
`validation_outputs/harness_runs_2026_05_20_root_artefacts/`
will be consumed by Dettes C/D/E.
