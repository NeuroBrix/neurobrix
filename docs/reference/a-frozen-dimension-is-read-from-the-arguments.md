# A claim about a frozen dimension is read from the operation's ARGUMENTS

**Rule.** A claim that a dimension is frozen is read from the operation's arguments —
`attributes.args` in `graph.json` — and **never** from `output_shapes`. Whoever makes the claim
**names the field they read**, in the sentence that makes it.

## Why, with the instance that cost the most

`D-MOCHI` said: *"at `aten._unsafe_view::1` the VAE's H and W are the literals 28 and 44"*. On
2026-09-18 that was checked against the container:

```
aten._unsafe_view::1  output_shapes  = [[1, 27, 28, 44, 2048]]
aten._unsafe_view::1  attributes.args[1] =
  {"type":"list","value":[1, {"type":"mul",
                              "left":{"type":"symbol","id":"s1","trace":9},
                              "right":3,"trace":27}, ...]}
```

The literals are real. They are in `output_shapes`, **which records what the shapes WERE at trace
time** — that is the field's entire job. The argument the runtime evaluates is `s1*3`, and it is
symbolic. The **07-07 container carries the same expression**, so the op was never frozen, in either
container.

The cost: mochi was re-traced on that premise — 2.7 h of two 32 GB cards plus hours of NFS reads —
and the re-trace could not have changed anything, because there was nothing at that op to change.
A thirty-second look at `attributes` would have refused the premise. The engine rule that forces
that check (*establish the smallest remedy before re-tracing a validated container*) was followed,
and it was run **on the wrong field**, which is why the rule now names the field.

## How to read it

`output_shapes` and `input_shapes` are a RECORD of the trace. Literals there are expected and carry
no claim about symbolic coverage.

`attributes.args` is the EXPRESSION. A dimension is frozen when the argument that produces it is a
plain integer where it should be a `symbol` or a `mul`/`add` over one:

```python
frozen  : {"type": "list", "value": [1, 27, 28, 44, 2048]}
symbolic: {"type": "list", "value": [1, {"type":"mul","left":{"type":"symbol","id":"s1"},"right":3}, ...]}
```

Ops that take no sizes carry no symbols by construction and prove nothing either way —
`aten::permute` takes a permutation order, and a scan that flags it produces false positives (87 of
them on mochi's VAE the first time this was measured).

## The instrument

`tools/frozen_dim_report.py <model> [component]` reads the arguments, never the shapes, and prints
per shape-op whether its size argument is symbolic, literal, or not-applicable. Use it before
filing, restating or acting on a frozen-dimension debt, and quote it.

`tools/frozen_dim_scan.py` is the catalogue-wide scan (2026-10-04). It reads each tensor's symbolic
annotation (`symbolic_shape.dims`) — the field the derivation and the placement read — and never
`output_shapes`, and for every hit it reads the producer's arguments too and says which field holds
the literal: *in the arguments* (the runtime evaluates it, the claim above) or *annotation only*
(the arguments are symbolic or `-1`, and what is frozen is the annotation the derivation keys
from — Qwen3-VL's expert view `[128, -1, 2048]` annotated `[128, 230, 2048]`). Both are defects of
the trace; the field named tells the retrace which one it fixes.
