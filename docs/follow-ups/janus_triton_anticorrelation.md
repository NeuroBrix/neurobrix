# Janus-Pro-7B `::triton` — numerical anti-correlation at step 0

## Status: **CLOSED** — root cause identified and fixed

Resolved in a follow-up commit after `bc8b9b8`. Image is now a coherent
cat matching the prompt; step-0 logits match native to 4 decimal places
with `cos = 1.000000` on both conditional and unconditional batches.

See [Resolution](#resolution) below.

## Original symptoms (for reference)

Collected in `/tmp/janus_parity/` during the diagnostic session that
wrote this doc's first version:

```
native cond   argmax= 2122  max=  19.066  L2= 1517.85
native uncond argmax= 3736  max=  23.770  L2= 2140.07
triton cond   argmax=14025  max= -909.93  L2=182662.88
triton uncond argmax= 5725  max=-1068.61  L2=204235.51

cos(native cond,   triton cond)   = -0.986152
cos(native uncond, triton uncond) = -0.995283
cos(native cond,   native uncond) =  0.994400  (baseline)
cos(triton cond,   triton uncond) =  0.999263  (!)
```

Two signals: near-perfect anti-correlation with ~120× magnitude on both
CFG batches, AND triton cond ≈ triton uncond while native distinguishes
them. Multiple hypotheses were listed (SDPA softmax polarity, RMSNorm
rsqrt path on fp16 V100, `_normalize_sdpa_scaling` not applied to
efficient-attention variant, `TritonAttentionInterceptor` batch leak,
attention-mask broadcast, RoPE position_ids batch handling).

## Diagnostic method

The "zoom into the first diverging layer" playbook narrowed the window
fast. Using the existing `NBX_DUMP_TIDS=/path NBX_DUMP_TIDS_FILTER=...`
infrastructure to dump head-10 + full L2 for each layer's residual-add
output (`aten.add::{7+n*7}` for n in 0..29):

```
uid               n_L2           t_L2           ratio  cos_head10
aten.add::7       93.7036        93.7033        1.000  1.0000
aten.add::14      22446.7734     22449.6991     1.000  1.0000
aten.add::21      22464.8438     22467.7685     1.000  1.0000
...
aten.add::210     18100.1738     18101.4043     1.000  1.0000
```

**Every single one of the 30 LM block outputs matched native to four
decimals on both L2 magnitude and head10 direction.** The anti-correlation
was entirely downstream of the `language_model` component.

## Root cause

`TritonLMSession._extract_hidden` (`src/neurobrix/triton/session.py`,
pre-fix) scanned the LM executor's graph outputs for a tensor whose
last dim equals `hidden_dim` (4096), with a fallback to "first output":

```python
def _extract_hidden(self, outputs: dict) -> Optional[NBXTensor]:
    for name, tensor in outputs.items():
        if hasattr(tensor, 'shape') and tensor.shape[-1] == self.hidden_dim:
            return tensor
    # Fallback: return the first output
    for tensor in outputs.values():
        if hasattr(tensor, 'shape'):
            return tensor
    return None
```

Janus's `language_model` graph **includes the text-vocab `lm_head` as
its final op chain** (`aten.view::422 → aten.mm::210 → aten._unsafe_view::301`)
so its declared graph output has shape `(2, S, 102400)` — text-vocab
logits, not hidden states. The `shape[-1] == hidden_dim` check fails;
the fallback then returns these text logits to the session as "hidden
states".

Downstream, `TritonImageStrategy.get_logits` selected the last seq
position from the returned tensor (giving `(2, 1, 102400)`), narrowed
the CFG batch, and passed `(1, 1, 102400)` to `gen_head`. `gen_head`'s
first op is `aten::view` with target shape `(s0, 4096)` — 102400 /
4096 = 25, so the view silently reinterpreted the text logits as
**25 batches of 4096 features**, then ran the two gen_head addmms +
gelu on that reinterpretation. The result has nothing to do with
hidden states, which exactly matches the observed profile: negative
sign (gelu flip on junk inputs), huge magnitude (wrong values through
two addmms), and near-identical across CFG batches (both go through
the same nonsense rearrangement).

Native's `core/flow/autoregressive.py::GraphLMSession.prefill` handles
this correctly by calling `enable_hidden_states_capture()` **before**
`executor.run()`; that method marks the first input of the last
`aten::mm` (the pre-`lm_head` hidden state) as a persistent tid, and
`get_hidden_states()` retrieves it from the `ExecutionContext.tensor_store`
after the run. The triton session never called this path.

## Resolution

Three changes:

1. **`src/neurobrix/core/runtime/graph_executor.py::_ensure_triton_compiled`**
   now pre-processes the DAG to add any tids in `self._persistent_tensor_ids`
   to the DAG's declared `output_tensor_ids` before the `TritonSequence`
   is constructed. `TritonSequence`'s liveness analysis already protects
   tids in `dag.output_tensor_ids` from being freed — this routes the
   native capture mechanism into the triton path without touching the
   triton sequence itself.

2. **`src/neurobrix/core/runtime/graph_executor.py::get_hidden_states`**
   gains a triton-mode branch that reads from
   `self._triton_seq.gather_outputs(...)` instead of
   `ExecutionContext.tensor_store`. The new `_get_hidden_states_triton`
   mirrors the same 2-strategy cascade as the native path (graph output
   is already hidden-dim → use it; else find last `aten::mm` and pull
   its first input from the arena).

3. **`src/neurobrix/triton/session.py::prefill` and `decode_step`** call
   `self.executor.enable_hidden_states_capture()` before each run (only
   the first call on prefill actually installs the capture — subsequent
   calls are no-ops because the tid is already in the set and the
   sequence is already compiled with the protected slot), and prefer
   `self.executor.get_hidden_states(...)` over the old `_extract_hidden`
   shape-guessing fallback.

The fallback `_extract_hidden` is kept for backwards compatibility
(models where the graph output is already hidden states and
`enable_hidden_states_capture` returns None).

## Post-fix measurements

Same prompt (`"a cat"`), same hardware (auto-profile → cuda:2 on V100-32g):

```
native cond   argmax= 2122  max=19.066  L2(1517.852)
native uncond argmax= 3736  max=23.770  L2(2140.073)
triton cond   argmax= 2122  max=19.064  L2(1517.457)
triton uncond argmax= 3736  max=23.769  L2(2139.993)

cos(native cond,   triton cond)   = 1.000000
cos(native uncond, triton uncond) = 1.000000
cos(native cond,   native uncond) = 0.994400  [baseline]
cos(triton cond,   triton uncond) = 0.994395
```

All three user gates from the session brief met:
- Gate 1 (image matches prompt): photorealistic cat, indistinguishable
  from native output by eye (see `/tmp/janus_parity/triton_v2.png`
  vs `/tmp/janus_parity/native.png`).
- Gate 2 (`cos(native cond, triton cond) ≥ 0.95`): 1.000000, exceeds by
  5 orders of magnitude on the residual.
- Gate 3 (`cos(triton cond, triton uncond) ≤ 0.998`): 0.994395 — within
  `5 × 10⁻⁶` of the native baseline 0.994400.

Decode time: native 192 s, triton 158 s on cuda:2 (triton faster by
~17 %, unchanged from pre-fix — the wrong tensor that was feeding
gen_head happened to execute in the same time as the right one).

## What this session explicitly does NOT change

- Zero-torch rule in `src/neurobrix/triton/` — still respected.
- Regression baselines (TinyLlama-1.1B, Qwen3-30B-A3B, DeepSeek-MoE)
  verified green after the fix.
- Native path for Janus and other image-AR models — untouched.
