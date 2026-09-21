"""Power — pure @triton.jit kernel."""

import triton
import triton.language as tl

@triton.jit
def pow_forward_kernel(
    input_ptr, output_ptr,
    n_elements,
    exponent,
    BLOCK_SIZE: tl.constexpr,
    INT_EXP: tl.constexpr = 0,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # 64-bit from the program id: the product itself wraps past 2^31 elements (register 58)
    mask = offset < n_elements

    x = tl.load(input_ptr + offset, mask=mask)
    x_fp32 = x.to(tl.float32)
    # `tl.extra.CUDA.libdevice.pow` used to stand here. It is NVIDIA's device
    # library, and asking any other backend to lower it fails at compile time —
    # measured on triton-ext, Kokoro-82M dying at aten.pow::0 (1, 55, 2048) with
    # "PassManager::run failed", and reproduced with a bare kernel containing no
    # engine code (repro_pow_libdevice_cuda_namespace.py):
    #     tl.extra.cuda.libdevice.pow   FAILED   PassManager::run failed
    #     portable exp/log/sign         COMPILED max rel err 2.522e-07
    # Triton 3.8.0 exposes no backend-neutral `pow`, so the semantics are built
    # here rather than borrowed from one vendor.
    #
    # libm semantics, which is what the libdevice call was chosen for:
    #   x > 0                     exp(e * ln x)
    #   x < 0, e an integer       sign * exp(e * ln|x|), sign = -1 when e is odd
    #   x < 0, e not an integer   NaN          (no real root)
    #   x = +-0                   the THREE cases below, not one
    #
    # The zero lane read `0.0` for every exponent until 2026-09-19. That is
    # wrong for two thirds of it, and the Dell measured it on CUDA against an
    # fp64 oracle: `0.0 ** 0.0` gave 0.0 where the oracle gives 1.0, and
    # `0.0 ** -1.0` gave 0.0 where it gives inf. Mine, and the shape of the
    # mistake is worth naming: replacing a vendor intrinsic means reimplementing
    # its whole DOMAIN, and I reimplemented the interesting part and not the
    # edges. C99 / IEEE-754 pow at zero:
    #
    #   pow(+-0, 0)                1              (even 0**0, and even nan**0)
    #   pow(+-0, e) e > 0          +-0, negative only for -0 with e an odd int
    #   pow(+-0, e) e < 0          +-inf, same sign rule
    #
    # The sign of the zero has to be read from the SIGN BIT, because `x < 0` is
    # false for -0.0 and `x == 0` cannot tell the two apart. A bitcast to int32
    # is negative exactly when the sign bit is set, which covers -0.0.
    _ax = tl.abs(x_fp32)
    _mag = tl.exp(exponent * tl.log(_ax))
    _is_int = tl.floor(exponent) == exponent
    _is_odd = _is_int & ((tl.floor(exponent * 0.5) * 2.0) != exponent)
    _sign = tl.where(_is_odd, -1.0, 1.0)
    _neg = tl.where(_is_int, _sign * _mag, float("nan"))

    _signbit = x_fp32.to(tl.int32, bitcast=True) < 0      # true for -0.0 too
    _zsign = tl.where(_signbit & _is_odd, -1.0, 1.0)
    _zero = tl.where(exponent == 0.0, 1.0,
                     tl.where(exponent > 0.0,
                              _zsign * 0.0,               # +-0
                              _zsign * float("inf")))     # +-inf
    out = tl.where(x_fp32 > 0, _mag, tl.where(x_fp32 < 0, _neg, _zero))
    # THE EXACT ROUTE FOR A SMALL INTEGER EXPONENT — the branch the comment
    # below records as "available rather than taken". Taken on 2026-09-20,
    # because the Dell measured the general route on CUDA against the fp64
    # oracle and it is not a rounding footnote: x**2 lands up to 15 ulps from
    # the oracle where main's libdevice call sat within 2 (38 147 lanes, |x| a
    # standard normal with zero lanes), x**0.5 at 5 vs 2, x**-1 at 7 vs 2. A
    # square computed as exp(2 log|x|) is not a square. Repeated multiplication
    # is correctly rounded at e=2 and within two roundings at e=3, needs no
    # vendor library, and carries IEEE's own sign rules for free: (-0)**2 is
    # +0, (-0)**3 is -0, 1/(+-0) is +-inf, so none of the select logic above is
    # needed on this route. The wrapper sets INT_EXP when the exponent is an
    # integer-valued scalar with 1 <= |e| <= 8; INT_EXP == 0 keeps the general
    # route, which still owns e == 0, non-integers and large integers.
    if INT_EXP != 0:
        _n: tl.constexpr = INT_EXP if INT_EXP > 0 else -INT_EXP
        _p = x_fp32
        if _n >= 2:
            _p = _p * x_fp32
        if _n >= 3:
            _p = _p * x_fp32
        if _n >= 4:
            _p = _p * x_fp32
        if _n >= 5:
            _p = _p * x_fp32
        if _n >= 6:
            _p = _p * x_fp32
        if _n >= 7:
            _p = _p * x_fp32
        if _n >= 8:
            _p = _p * x_fp32
        if INT_EXP < 0:
            _p = 1.0 / _p
        out = _p
    # ---------------------------------------------------------------------
    # THE ACCURACY COST OF THIS ROUTE, AND WHAT IT IS AND IS NOT INHERENT TO.
    #
    # `exp(e * log|x|)` loses precision as |x| moves away from 1, because the
    # error in `log|x|` is multiplied by `e` before `exp` amplifies it. The Dell
    # measured pow-to-the-third at 19x worse relative error than main's
    # libdevice call on CUDA. Measured here against fp64, on the ranges a
    # vocoder actually sees (probe_pow_error.py):
    #
    #   range                    e    this kernel   x*x(*x)     ratio
    #   |x| in [0.5, 2]          2      2.56e-07   5.94e-08      4.3x
    #   |x| in [0.5, 2]          3      3.95e-07   1.14e-07      3.4x
    #   |x| in [1e-3, 1e3]       3      3.58e-06   1.15e-07       31x
    #   |x| in [1e-6, 1e-3]      3      7.24e-06   1.13e-07       64x
    #
    # So the 19x is real and it is range-dependent: 3.4x on typical activations,
    # 64x on small ones.
    #
    # WHAT IT COSTS ON THE MODELS THAT USE IT. `aten::pow` appears 194 times
    # across the 16 cached containers, and only in the two vocoder TTS models:
    #   exponent 2    182x   Kokoro-82M 49, chatterbox 133
    #   exponent 3.0   12x   Kokoro-82M only
    # Nothing else in the catalogue calls pow at all, and no site uses a
    # non-integer exponent. Both models are judged by third-party ASR and both
    # transcribe correctly; the error is ~1e-7 relative on their activation
    # range, which is below fp32 round-off for the accumulations downstream of
    # it. So the cost TODAY is not audible and not measurable at the artefact.
    #
    # BUT IT IS AVOIDABLE, and that is the part worth recording rather than
    # accepting: the cost is inherent to `exp(e * log|x|)`, NOT to `pow`. Every
    # exponent the catalogue uses is a small integer, and for those, repeated
    # multiplication is 3-64x MORE accurate and no more expensive — one
    # multiply for e=2 against a log, a multiply and an exp. A `POW_SMALL_INT`
    # constexpr branch would take the exact route at compile time and leave this
    # path for the general case.
    #
    # It is NOT done here, deliberately: it changes the numerics of two SHIPPED
    # models, so it owes a fresh set of judged cells and a fresh CUDA column,
    # and 0.5.4 has just been cut. Recorded as available rather than taken.
    tl.store(output_ptr + offset, out, mask=mask)
