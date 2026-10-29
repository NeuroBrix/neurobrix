"""The launch configuration of every element-wise kernel, certified (chantier 1).

An element-wise kernel reads and writes each element once: its time is the memory's, and the program's tile and warps
move it. Measured on V100 (2026-10-23, `campaigns/2026_10_12_mq25_gate/stable/c1rest/ew_bench.json`: add, mul, exp,
silu, gelu, erf, pow and a copy, fp32 and fp16, 4 096 to 16 777 216 elements, 15 tiles x warps): at 16 M elements the
best of the 15 is within 0.3 % (max 1.4 %) for every operation of the configuration that is best for the copy; below
65 536 elements the launch itself dominates. The OPERATION does not move the optimum — the element count does, by
class. So the element-wise kernels share ONE key: the class of the launch's element count (`autotune.buckets.elements`,
the profile's ladder, else its `default`), certified on the copy below, and every element-wise launch reads its tile
and warps through `ew_meta` — one memoized resolution per class and process, through the autotuner's own door
(`ops/_configs.nbx_autotune`: the census record, the certified directory, the refusal of a confirmation run).

The copy is in fp32 (the widest element a launch of this class commonly moves); a tile changes no bit of any
element-wise result — each element is computed alone — so its oracle is identity.
"""

import triton
import triton.language as tl

from nbxtensor.ops._configs import nbx_autotune


def _ew_configs():
    """Tile x warps; the configuration every element-wise launch ran before certification (1 024 x 4) is among them."""
    return [triton.Config({'BLOCK_SIZE': block}, num_warps=warps)
            for block in (256, 512, 1024, 2048, 4096) for warps in (2, 4, 8)]


@nbx_autotune(configs=_ew_configs(), key=['N_BUCKET'], cache_results=True)
@triton.jit
def ew_launch_kernel(x_ptr, y_ptr, n_elements, N_BUCKET, BLOCK_SIZE: tl.constexpr):
    """y = x over n_elements: the element-wise class's representative."""
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    tl.store(y_ptr + offset, tl.load(x_ptr + offset, mask=mask), mask=mask)


_MEMO = {}          # {the element count's class: {'BLOCK_SIZE': ..., 'num_warps': ...}} — one resolution per process


def ew_bucket(n_elements: int) -> int:
    """The class of an element-wise launch of `n_elements` (`autotune.buckets.elements`, else the profile's
    `default`): the key `launch_keys.elementwise_launches` forms for the census."""
    from nbxtensor.ops.autotune_bucket import bucket_of
    return bucket_of("elements", max(1, int(n_elements)))


def ew_meta(n_elements: int) -> dict:
    """The tile and warps of an element-wise launch of `n_elements`: its class's certified configuration (a
    confirmation run refuses a class the directory lacks, by name; another run sweeps it once), read once per class
    and process. The resolution launches the copy above at the class's floor — the fewest elements that form it, so
    never more than the launch that asks — on a scratch buffer of the current device, through the autotuner's door."""
    nb = ew_bucket(n_elements)
    hit = _MEMO.get(nb)
    if hit is not None:
        return hit
    from nbxtensor.ops.autotune_bucket import class_floor
    from nbxtensor.tensor import NBXTensor
    from nbxtensor.dtype.nbx_dtype import NBXDtype
    from nbxtensor.ops.launch_observer import launch_observer
    _census = launch_observer()                 # the application's launch observer (dry run), if any
    from nbxtensor.ops.autotune_cache import key_of
    rep = class_floor("elements", nb)
    buf = NBXTensor.empty((rep,), dtype=NBXDtype.float32, device='cuda')
    args = (buf, buf, rep, nb)
    ew_launch_kernel[lambda META: (triton.cdiv(rep, META['BLOCK_SIZE']),)](*args)
    cfg = ew_launch_kernel.cache.get(key_of(ew_launch_kernel, args, {}))
    if cfg is None:
        if _census.active():         # a census shadow launches nothing: no configuration exists, none is needed
            first = ew_launch_kernel.configs[0]
            return {'BLOCK_SIZE': first.kwargs['BLOCK_SIZE'], 'num_warps': first.num_warps}
        raise RuntimeError(f"element-wise class {nb}: the copy's autotuner ran and holds no configuration for it")
    hit = {'BLOCK_SIZE': int(cfg.kwargs['BLOCK_SIZE']), 'num_warps': int(cfg.num_warps)}
    _MEMO[nb] = hit
    return hit


def ew_grid(n_elements: int, meta: dict) -> tuple:
    """The 1-D grid of an element-wise launch at its configuration."""
    return (triton.cdiv(int(n_elements), meta['BLOCK_SIZE']),)
