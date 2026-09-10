# P-VACE-IMAGE-CONDITIONED-CONTROL

## Status: **OPEN** — named follow-up (opened 2026-09-10, certified-directory proof campaign)

## Problem

`Wan2.1-VACE-1.3B-diffusers` runs the all-generate (pure text→video) path and
only that path. Given a still reference image it takes a route that produces a
video tensor with a temporal extent of **0**, and the request fails inside the
VAE encoder.

The all-generate synthesis in `cli/commands/run.py` is guarded on the image
being ABSENT:

```python
if "global.image" not in inputs:
    if _gcf(model, "transformer", "vace_control_conditioning", default=None):
        _nf = int(getattr(args, "num_frames", 0) or 1)
        inputs["global.image"] = zeros(1, 3, _nf, _h, _w)   # zeros control clip
```

So supplying `--input-image` does not *add* image conditioning — it *disables*
the only implemented control path. The still image (4-D, no temporal axis) then
reaches a `vae_encoder` whose graph declares a 5-D input
`[batch, 3, time, height, width]`, and `time` resolves to 0.

## How it surfaced

The precision-zoo campaign adds `--input-image` to any video/image model that
declares an image input (`precision_zoo_campaign.py`, `_declares_image_input`),
which VACE does. The arm then failed as:

```
Failed at aten.convolution::60 (aten::convolution): GPU malloc failed
for -4860000 bytes [device cuda:2 live_tracked=1081MB driver_free=31000MB]
NBX args: arg0 ptr=0x0 shape=(1, 3, 0, 450, 450)
```

A negative allocation with 31 GB free — `_conv3d_via_conv2d` reshapes by
`B * T_out`, and `T_out` derived from `T=0` goes negative. The run was recorded
as an OOM.

Two engine defects on the path have been fixed since, and neither is this one:

* the symbol binder read its constraints from the wrong key, so `time = 0` bound
  cleanly against a declared `constraints: {min: 1}` (fixed 2026-09-10 — the
  failure now names the symbol, its source and the violated bound at bind time
  instead of surfacing 60 ops later as a byte count);
* `--skip-done` treated the resulting crash file as a verdict, so the model was
  skipped by every later campaign (fixed 2026-09-10).

With those in place the request now fails immediately and legibly. It still
fails.

## What to do

Implement image-conditioned VACE control, or state that the reference-image
input is not part of this model's contract and make the request say so rather
than composing a degenerate clip. Deciding between those needs the vendor
semantics read first (R16): VACE takes a reference image AND a control video,
and how a still is meant to seed the control clip (repeat to `num_frames`, pad
with zeros after frame 0, or feed the reference through a separate path) is the
vendor's answer, not one to infer.

Requires a GPU to validate: the arm produces an `.mp4` that must be inspected
(R29), not just a shape that stops raising.

## Scope

`Wan2.1-VACE-1.3B-diffusers` (the only container declaring
`vace_control_conditioning`). Not a blocker for the all-generate path, which is
what the model is otherwise measured on.
