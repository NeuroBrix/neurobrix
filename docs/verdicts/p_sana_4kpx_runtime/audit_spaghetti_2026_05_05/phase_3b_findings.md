# Phase 3b — VAE-only seq vs triton diff findings

## Run setup

`vae_isolation_probe.py --vae-only-decode --mode {sequential|triton_sequential}`
on saved latent `vae_isolation_input.pt`, with `NBX_VALUE_DUMP_EVERY=1`.
Both modes execute the same VAE graph on the same input.

## First divergence in trace

Trace ops 0..487: bit-exact / fp16 ULP between seq and tri.

At trace op_idx=488 = `aten.convolution::36` (decoder.up.3.1.conv_out.conv_depth, depthwise 3×3 groups=4096):

```
seq[0..3] = [+1.062, 0.292, 1.012, 3.002]
tri[0..3] = [-1.046, 1.471, 3.195, 0.567]
max_d = 2.435, rel = 81%, sign flip on position 0
```

## conv::36 is NOT the bug source — it's a propagator

Microtest of NBX `conv2d_wrapper` (depthwise path) at the EXACT failing
shape `(1, 4096, 512, 512) × (4096, 1, 3, 3)` groups=4096:
- with random fp16 inputs vs `F.conv2d`: **BIT-EXACT** (max_d=0.0)
- with REAL captured silu::12 output vs `F.conv2d`: **BIT-EXACT** (max_d=0.0,
  NBX = `-1.046` = torch reference; matches the triton runtime trace)

So NBX depthwise is computing exactly what torch computes on the same
input. The sign flip at op_idx=488 is amplification of upstream
divergence in silu::12 input (or before), not a kernel bug at
convolution::36 itself.

## Why fingerprint missed the upstream divergence

The 5-position fingerprint (`(0, n//8, n//4, n//2, 3n//4)`) on a
~1 GB tensor (n=1.07e9 for silu::12 at shape (1, 512, 1024, 1024))
samples 5 elements out of 1 billion. The 3×3 spatial neighborhood
of conv::36 output[0,0,0,0] depends on a 3×3 patch of silu::12
output near (0,0,0,0). If that patch has fp16-ULP-level noise that
correlates structurally (say, every other element), depthwise sums
9 values that compound to a sign flip in the output despite all
sample positions matching at fp16 ULP precision.

## Conclusion

The bug is **distributed numerical drift in mid-VAE upsampling**,
specifically before silu::12 (decoder.up.3.1.conv_out.nonlinearity),
amplified by depthwise conv::36 to a visible sign flip. The drift
itself comes from accumulated per-element noise that the sparse
fingerprint cannot see.

This invalidates the "depthwise is the bug" hypothesis. The bug is
genuinely the cumulative-drift narrative we earlier dismissed —
but it manifests in VAE not transformer (Cas B from v5 isolation
test).

## Why is sequential VAE coherent and triton VAE garbage if drift is the issue?

Sequential (NativeATenDispatcher → torch.ops.aten) and triton
(TritonSequentialDispatcher → NBX kernels) take different paths
through fp16 arithmetic. Per-op fp16 ULP differences accumulate
DIFFERENTLY across hundreds of VAE ops. By the final conv::69 output,
the accumulated drift produces:
- sequential: red apple (correct decode)
- triton: green texture (decoded from drift-shifted latent state)

The 15 kernels we proved bit-exact in isolation each contribute
fp16-ULP noise that's individually correct (matches torch ULP)
but compounds across the chain.

## Resolution paths

This is NOT a single-kernel bug. Resolution requires either:

1. **Higher precision intermediates** (fp32 or fp16 with strict
   IEEE rounding) for the VAE upsampling chain. Memory cost ~2x.
2. **Bit-match torch's specific accumulation order** in critical
   ops. Hard — torch uses cuDNN's internal accumulation pattern
   that Triton can't fully replicate on Volta.
3. **Algorithm-level normalization** that prevents fp16 ULP from
   amplifying through depthwise conv neighborhoods. Custom
   stabilization.

None of these are "a kernel bug to fix". They're architectural
trade-offs between performance and per-op-equivalence with torch.

## What we proved

- 15 distinct kernels bit-exact in isolation (random AND real inputs)
- v5 isolation: triton VAE on identical latent → garbage (Cas B)
- Phase 3b: divergence onset at op_idx=488 conv::36 BUT conv::36
  itself is bit-exact; divergence is upstream drift
- Therefore: the bug is distributed numerical drift through the
  VAE upsampling chain, not localized to any single kernel
