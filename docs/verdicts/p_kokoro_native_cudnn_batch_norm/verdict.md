# P-KOKORO-NATIVE-CUDNN-BATCH-NORM — Verdict

**Chantier:** diagnose & fix the Kokoro-82M `::native`
`aten::cudnn_batch_norm` "sym_strides() called on an undefined
Tensor" regression.
**Branch:** `p-kokoro-native-cudnn-batch-norm` from `2a72b31` (HEAD
of `p-harness-upscaler-dispatch`).
**Date:** 2026-05-16.

---

## 1. Original mandate and its invalidation

The mandate assumed Kokoro-82M `::native` is red at HEAD and asked
for a None-guard fix on `aten::cudnn_batch_norm`'s optional args
(Étape 3). The Étape 1-2 audit, performed BEFORE writing any code,
invalidated that premise: **the bug no longer exists at HEAD**. It
was a real regression (breaking commit pinned by bisect) but was
fixed indirectly, weeks ago, by the zero3/device refactors of
later chantiers. Writing the None-guard would fix a non-existent
bug — the fictitious-work / R5 anti-pattern. Escalated; the
maintainer confirmed option (a): pivot this chantier to clean
hygiene (un-xfail + archive + verdict). No runtime/src code was
written.

---

## 2. Bisect — breaking commit

Endpoints verified empirically this session (direct runtime, venv
python, cached `.nbx` constant):

| Commit | Kokoro-82M::native | Evidence |
|---|---|---|
| `a64aa4b` (v0.1.5) | **GREEN** | `SAVED: output_Kokoro-82M.wav`, RC 0, 13.13 s |
| `be5c7b8` | **RED** | `RuntimeError: Failed at op aten.cudnn_batch_norm::0 (aten::cudnn_batch_norm): sym_strides() called on an undefined Tensor`, RC 1 |

Method: `git bisect run` on the **runtime symptom** (signature →
BAD, `SAVED` → GOOD, unrelated failure → SKIP) — deliberately
harness-independent, because the pytest harness itself drifted
across the bisect window (the venv auto-probe landed mid-window);
only `src/` varies, venv + `.nbx` constant.

**First bad commit: `ea90d66`** — *"fix(zero3): correct mat2-on-CPU
crash via per-op slow-path forcing"*. It is suspect #6 in the
follow-up's window list (the §3-class suspect: per-op device
override leaking into the general path), NOT the follow-up's #1
rank `be5c7b8`, and NOT an out-of-window commit — no architectural
remontée was warranted at the bisect stage.

---

## 3. Causal diagnosis

Kokoro `aten.cudnn_batch_norm::0`, `parent_module = encode.norm1.norm`
(an InstanceNorm). From `graph.json` `attributes.args` and a runtime
probe (both concordant):

```
[ input=T(1,514,128), weight=T(514,), bias=T(514,),
  running_mean=None (slot 3), running_var=None (slot 4),
  training=True, momentum=0.1, eps=1e-5 ]
```

`running_mean=None, running_var=None` with `training=True` is the
**legitimate** torch semantics for InstanceNorm (no running stats).
Verified independently:
`torch.cudnn_batch_norm(x,w,b,None,None,True,0.1,1e-5)` and
`F.batch_norm(x,None,None,w,b,True,…)` both run fine on the
production torch. So this is NOT a dead-eliminated producer (the
follow-up's hypothesis A) — the `input/weight/bias` slots are real
tensors, the None slots are correct.

`git show ea90d66` contains **zero** occurrences of `batch_norm`
or `cudnn` — the commit never touches the op directly. The
regression is therefore **indirect**: `ea90d66`'s zero3 per-op
device-override machinery (`mark_cpu_weighted_ops_for_transfer`,
`recompute_op_devices_for_slots`, the `compute_op_devices` changes)
applied in the **general single-device path** (Kokoro is 82M —
well under any zero3 trigger; `_is_multi_device=False`, runs via
plain `_run_inner`) and mutated/derailed the legitimate
InstanceNorm `None` signature so the op received an undefined
tensor → `sym_strides()`. This is exactly the follow-up §3
hypothesis ("if the per-op device override logic still applies in
the general case, a Kokoro op that legitimately had a None arg may
now crash where it used to silently no-op").

---

## 4. Death of the bug

Kokoro-82M::native is GREEN at:

- `3fb4430` (P-CORRECTNESS base) — spot-check this session: `SAVED`,
  RC 0.
- `2a72b31` (this branch HEAD) — direct runtime ×3 (3 distinct
  prompts) all `SAVED` RC 0, deterministic (the InstanceNorm-None
  path is fixed, not MoE-style flaky); pytest harness cell
  `test_model_runs[Kokoro-82M::native]` → **XPASS**.
- Maintainer independent verification at `2a72b31`:
  `neurobrix run --model Kokoro-82M --prompt 'Hello world' --output
  /tmp/kokoro_native_test.wav --compiled` → full pipeline
  (bert → bert_encoder → text_encoder native → predictor native →
  decoder), `SAVED /tmp/kokoro_native_test.wav`, 12.05 s, RIFF PCM
  16-bit mono 24 kHz, 153 KB. No `sym_strides`.

The fix landed **indirectly within `be5c7b8..3fb4430`** — the
zero3/device-override refactors of P-PRISM-NEVER-REFUSE v2 and
P-SANA-4KPX-RUNTIME restructured the per-op device override so it
no longer mutates legitimate `None` args. No commit in that window
names `cudnn_batch_norm` (side-effect). The exact fixing commit
was **not pinned**: a forward bisect across `be5c7b8..3fb4430`
crosses two multi-week chantiers (~100+ commits, skip-heavy for
unrelated breakages) — high cost, zero value versus the actual
deliverable. R23 priority: correctness > readability > historical
exhaustiveness. The P-CORRECTNESS verdict's unattributed
"1 xpassed" is now identified as exactly this Kokoro-82M::native
cell (stale-xfail-but-passing, noted-but-not-drilled then).

---

## 5. Deliverables (hygiene pivot)

1. **Un-xfail** `tests/regression/conftest.py`: removed the stale
   `("Kokoro-82M", "native", "aten::cudnn_batch_norm fails …")`
   `KNOWN_FAILURES` entry + rewrote the comment block. The
   conftest's own doctrine is verbatim: *"When a fix lands, REMOVE
   the entry — leaving stale xfails masks regressions."* The
   `("Kokoro-82M", "triton", …)` entry stays — orthogonal
   `_execute_native_text_encoder` NBXTensor→torch embedding bug,
   out of this chantier's scope. Commit `e9a56af`.
2. **Archive** the follow-up:
   `docs/follow-ups/kokoro_cudnn_batch_norm_regression.md` →
   `docs/follow-ups/archive/…`, with an appended `RESOLVED —
   2026-05-16` section (bisect result, root cause, indirect-fix
   window, validation, forward guard-rail). Commit `ebd3121`.
3. **Forward guard-rail** (in the archived doc): any future
   chantier touching zero3 / per-op device override — notably
   **P-OP-LEVEL-CROSS-DEVICE-SPLIT (Gap B)** — must validate
   non-regression on Kokoro-82M::native and, more generally, on
   InstanceNorm/LayerNorm/BatchNorm ops whose
   `running_mean`/`running_var` are legitimately `None` in
   training mode.

---

## 6. Anti-regression matrix

The change is **conftest + docs only — zero runtime/src
modification**, so no cell's runtime behaviour can change. The
only harness-level effect: `Kokoro-82M::native` moves from the
xfail bucket (XPASS) to the normal bucket (PASS).

| Scope | Before de-xfail (HEAD 2a72b31, P-HARNESS post-fix run) | After de-xfail (run `bptyn678k`, 31m37s) |
|---|---|---|
| Kokoro-82M::native | xfail entry present → **XPASS** | **PASS** ✓ (absent from failed/xpass/xfail; clean normal pass) |
| Global fast | 4F / 34P / 12skip / 11xf / **1xp** | **5F / 34P / 12skip / 11xf / 0xp** |

`xpassed 1→0` ✓ (the stale Kokoro::native XPASS is gone — now a
normal PASS). `xfailed 11→11`, `skipped 12→12` unchanged. `passed`
nets 34 = 34 + Kokoro::native(now counted PASS, was XPASS so not in
"passed") − Qwen3-30B::triton(flaked red this run).

`failed 4→5`: the +1 is **`Qwen3-30B-A3B-Thinking-2507::triton`** —
the documented non-deterministic MoE-triton flaky cell
(P-CORRECTNESS verdict §5.6: green in the P-HARNESS run, red here,
green in P-CORRECTNESS-era runs; its DAG has no index_put/linspace,
the harness llm-invocation is byte-identical, untouched by every
chantier). **This is provably NOT a regression of this chantier**:
the only changes are conftest (de-xfail) + a docs archive — **zero
runtime/src modification** — which cannot alter any model's runtime
output. The other 4 failures (`hat-s-x4::triton`,
`hat-l-x4::triton` — P-HARNESS verdict §5, HAT 2/4 / im2col;
`orpheus-3b-0.1-ft::native`, `::triton` — P-CORRECTNESS §5.2, SDPA
0-dim) are pre-existing and documented in prior verdicts.

**Criterion SATISFIED:** Kokoro-82M::native is a clean **PASS** (no
longer xpassed/xfailed); no previously-green cell regressed because
of this chantier (the lone delta vs the P-HARNESS baseline,
Qwen3-30B::triton, is the documented pre-existing flaky cell, and a
conftest+docs-only change cannot causally affect MoE-triton runtime
output).

---

## 7. Commits

Branch `p-kokoro-native-cudnn-batch-norm` (3 commits atop `2a72b31`):

| SHA | Subject |
|---|---|
| `e9a56af` | test(regression): un-xfail Kokoro-82M::native (bug fixed in zero3 refactor era) |
| `ebd3121` | docs(follow-ups): archive kokoro cudnn_batch_norm (resolved indirectly) |
| _(verdict)_ | docs(verdict): P-KOKORO-NATIVE-CUDNN-BATCH-NORM |

---

## 8. Latent observations

- This is the **third chantier in two days** that closed as "the
  requested work was not the real work" (P-UPSCALER-CLI-WIRING →
  architectural escalation; P-KOKORO-NATIVE-CUDNN-BATCH-NORM → bug
  already dead). The exhaustive pre-prompt reading BEFORE writing
  code is what surfaced both — kept as the working discipline.
- The P-CORRECTNESS verdict's "1 xpassed" was left unattributed;
  it is this cell. Minor documentary debt in that verdict, now
  resolved by cross-reference.

Hocine validation: TODO
