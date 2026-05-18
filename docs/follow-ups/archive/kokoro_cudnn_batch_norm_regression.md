# Kokoro-82M `::native` regression — `aten::cudnn_batch_norm` undefined Tensor

## Symptom (verbatim stderr at HEAD)

Running `Kokoro-82M` with the `::native` path via the regression harness
(under a venv Python that has `neurobrix`, `transformers`, and
`mistral_common` all importable — see the harness `pytest_configure`
hook added alongside this doc):

```
File "/home/mlops/NeuroBrix_System/src/neurobrix/core/runtime/graph/compiled_sequence.py",
  line 3078, in _run_inner
    raise RuntimeError(f"Failed at op {op.op_uid} ({op.op_type}): {e}") from e
RuntimeError: Failed at op aten.cudnn_batch_norm::0 (aten::cudnn_batch_norm):
  sym_strides() called on an undefined Tensor
```

`sym_strides()` is a `torch.Tensor` API; "undefined Tensor" is
PyTorch-speak for a `None` or sentinel tensor. One of the positional
args `aten::cudnn_batch_norm` receives from the compiled sequence is
not a real tensor.

## Was-green / was-red

| Commit     | Kokoro-82M `::native` | Notes                                               |
|------------|:--------------------:|-----------------------------------------------------|
| `a64aa4b`  | **PASSED**           | v0.1.5 release tag. Confirmed this session under venv python. |
| `be5c7b8`  | **FAILED**           | First commit seen failing with this exact stderr. Confirmed this session. |
| HEAD (`194ec9a`) | FAILED         | Same stderr.                                        |

Narrow bisect across the ~8 commits between `a64aa4b` and `be5c7b8`
was not run in this session (per session scope). The suspect window
is:

| Commit     | Title                                                                     |
|------------|---------------------------------------------------------------------------|
| `228be08`  | build(sdist): exclude venvs, regression harness, media artefacts          |
| `d4c229d`  | fix: zero torch in triton/, flow-aware audio harness, Janus family-guard, kernel coverage for TTS vocoders |
| `f42970c`  | wip: hardware-gated fp16 overflow architecture                            |
| `c70ed46`  | fix(triton): bind-time fp16→fp32 weight upcast on pre-Ampere — step 2    |
| `857378a`  | fix(memory): sync before clear in MemoryManager.unload_weights            |
| `ea90d66`  | fix(zero3): correct mat2-on-CPU crash via per-op slow-path forcing        |
| `2b1b0d4`  | feat(zero3-triton): universality + metadata-op retention leak fix         |
| `22e9ff2`  | fix(triton): preserve strides in _transfer_tensor for non-contiguous CPU sources |
| `be5c7b8`  | feat(zero3): block-wise ratchet pipelining (native + triton) + MoE fusion output-sweep |

Higher-priority suspects on prior reading:

- `be5c7b8` (zero3 ratchet) touched `compiled_sequence.py` extensively
  (per-op device override, CPU-backed arg detection, new callback
  plumbing). An `aten::cudnn_batch_norm` with a `running_mean` or
  `running_var` that became a zero3-offloaded CPU tensor may now be
  fed `None` to the native compile path if the device-remapping
  skipped a None-check.
- `d4c229d` — touched batch_norm wrapper (per commit title: "kernel
  coverage for TTS vocoders"). A wrapper-level change may have
  altered how `aten::cudnn_batch_norm` is dispatched to match the
  "universal" kernel, and a Kokoro-specific arg shape may now fall
  into an untested branch.
- `22e9ff2` — preserve strides in `_transfer_tensor`: if
  `running_mean` / `running_var` are view-backed, stride preservation
  could change how they're resolved into the compiled sequence.

## Investigation tasks (DO NOT PERFORM IN THIS DOC; dedicated session)

1. **Narrow bisect** between `a64aa4b` and `be5c7b8`. Use
   `NEUROBRIX_PYTHON=/home/mlops/ml/venv/bin/python pytest
   tests/regression/test_all_models.py::test_model_runs
   -k "Kokoro-82M and native" -v -s` as the test command. Report the
   single breaking commit.

2. **Read the compiled sequence batch_norm dispatch** at
   `core/runtime/graph/compiled_sequence.py:3078` and follow back to
   the `CompiledOp` that represents `aten::cudnn_batch_norm::0`.
   Identify which positional arg resolves to the undefined tensor —
   typically one of `running_mean`, `running_var`, `weight`, or
   `bias`. The culprit is probably an input slot whose producer was
   dead-op-eliminated or device-remapped without a replacement.

3. **Inspect Kokoro's `graph.json`** to see which op slots feed
   `aten.cudnn_batch_norm::0` and whether any of them reference a
   weight that would have been CPU-offloaded under zero3. Kokoro is
   a 82M model, well under any zero3 trigger, so zero3 *should not*
   activate — but if the per-op device override logic in the zero3
   change still applies in the general case, a Kokoro op that
   legitimately had a None arg (e.g. optional bias) may now crash
   where it used to silently no-op.

4. **Propose the fix** at the identified breaking commit. Two
   shapes likely:
   - If the break is in `compiled_sequence.py`: a None-guard in
     the op dispatch path for `aten::cudnn_batch_norm`'s optional
     args.
   - If the break is in a wrapper under `kernels/`: a mirror of
     the native ATen dispatcher's None-arg handling.

5. **Validate**: Kokoro-82M `::native` passes, and the full harness
   does not regress on any of the other 11 currently-passing models
   (Voxtral + whisper-large + whisper-large-v3-turbo also pass
   after the harness `pytest_configure` interpreter auto-detect
   lands in this commit).

## Not in scope

- **Kokoro `::triton`**: orthogonal bug (`_execute_native_text_encoder`
  passes `NBXTensor` to `torch.nn.functional.embedding`), tracked
  separately in `KNOWN_FAILURES` at `tests/regression/conftest.py`.
- **Phonemizer CLI install**: the "`Phonemizer model requires
  'kokoro', 'phonemizer', or 'espeak-ng' CLI`" error is a
  system-dependency on machines lacking the phonemizer binary —
  orthogonal to the `aten::cudnn_batch_norm` regression. Under the
  venv python the phonemizer path doesn't raise that error because
  the venv has a Python `phonemizer` package importable.

## Session rules (apply in the follow-up)

- No fix before the narrow bisect identifies the breaking commit.
- Audit R1 on whichever file gets touched.
- Native + triton parity: any fix to the native `aten::cudnn_batch_norm`
  path must preserve the triton mode path's behaviour on models that
  don't use batch norm.
- Zero torch in triton path (`src/neurobrix/triton/`) — unchanged constraint.

## RESOLVED — 2026-05-16

Bisect (P-KOKORO-NATIVE-CUDNN-BATCH-NORM session) : breaking commit `ea90d66`
"fix(zero3): correct mat2-on-CPU crash via per-op slow-path forcing".

Cause racine : InstanceNorm dans Kokoro encode.norm1.norm reçoit
légitimement `running_mean=None, running_var=None` (sémantique torch
standard pour training=True). La logique zero3 per-op-device override
de ea90d66 s'appliquait au general single-device path (Kokoro 82M
n'active pas zero3) et cassait cette signature légitime.

Fix indirect : entre `be5c7b8` et `3fb4430`, les refactors zero3/device
des chantiers P-PRISM-NEVER-REFUSE v2 et P-SANA-4KPX-RUNTIME ont
restructuré la per-op device override d'une façon qui ne mute plus les
args None légitimes. Le commit fixeur exact n'a pas été épinglé
(forward-bisect ~100+ commits, coût élevé / valeur nulle vs le
livrable). Aucun commit ne nomme cudnn_batch_norm — side-effect.

Validation : HEAD 2a72b31 + bisect endpoints a64aa4b GREEN, be5c7b8 RED
confirmés cette session ; HEAD GREEN runtime direct ×3 + harness XPASS.

Garde-fou pour le futur : tout chantier touchant zero3/device override
(notamment P-OP-LEVEL-CROSS-DEVICE-SPLIT Gap B) doit valider non-régression
sur Kokoro-82M::native et plus généralement sur les ops InstanceNorm/
LayerNorm/BatchNorm dont running_mean/running_var sont légitimement None
en mode training.

Verdict : docs/verdicts/p_kokoro_native_cudnn_batch_norm/verdict.md.
