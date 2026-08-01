"""Shared seeded CPU fp64 draw frontier (R30).

ONE code path draws sampled tokens for BOTH engines: logits cross the
boundary as fp64 numpy arrays (D2H at the caller), the vendor-faithful
filter chain (repetition penalty -> temperature -> top-k -> top-p) and
the draw run in CPU fp64, and the uniform stream comes from one seeded
generator. Compiled and triton therefore produce bit-identical draws
whenever the incoming logits agree; near-ties remain the only residual
class, adjudicated by margins (NBX_DECODE_TOPK doctrine).

Torch-free by construction: numpy is CPU glue, legal in both modes
(R33/R34), so this module is importable from core/ AND triton/ without
duplication. Precedent: the dual_ar flow's byte-identical duplicated
`_sample_token_np` pair — this module is that prototype promoted to a
single shared frontier. Scope: activated by a registry-declared speech
sampling contract; existing zoo samplers are untouched (their byte
gates hold by construction).

Filter semantics mirror the transformers LogitsProcessor chain:
- repetition penalty: seen ids with logit > 0 divided by the penalty,
  logit <= 0 multiplied by it (RepetitionPenaltyLogitsProcessor).
- top-k: THRESHOLD keep — every logit >= the k-th highest survives
  (ties at the threshold may keep more than k, exactly like
  TopKLogitsWarper's `< topk(...)[-1]` mask).
- top-p: smallest set of highest-probability tokens whose cumulative
  probability reaches top_p, at least one token kept
  (TopPLogitsWarper); ties broken by ascending vocab index for
  run-to-run determinism.
- draw: inverse-CDF walk over the filtered, renormalized fp64 probs in
  vocab-index order against one uniform from the stream. The draw
  count per step is the cross-engine coupling contract: both engines
  MUST consume draws in the same order.
"""

import json
import os

import numpy as np

__all__ = ["SeededDrawStream", "apply_repetition_penalty",
           "seeded_gaussian"]

# NBX_DRAW_DIAG=<jsonl path>: per-draw top-4 candidate ids/logits + the
# top-2 margin + the chosen id, appended as one JSON line. The speech
# twin of NBX_DECODE_TOPK — at any cross-engine code divergence the
# FIRST differing line localizes the flip and its margin adjudicates
# near-tie vs large-margin (both engines instrumented by construction:
# this module IS the shared frontier). Default off, zero cost off.
_DIAG_PATH = os.environ.get("NBX_DRAW_DIAG")


def apply_repetition_penalty(logits: np.ndarray, seen_ids, penalty: float) -> np.ndarray:
    """RepetitionPenaltyLogitsProcessor semantics on fp64 logits (in place)."""
    if not penalty or penalty == 1.0 or seen_ids is None:
        return logits
    ids = np.unique(np.asarray(list(seen_ids), dtype=np.int64))
    if ids.size == 0:
        return logits
    ids = ids[(ids >= 0) & (ids < logits.shape[0])]
    vals = logits[ids]
    logits[ids] = np.where(vals > 0, vals / penalty, vals * penalty)
    return logits


class SeededDrawStream:
    """One seeded uniform stream; the draw ORDER is the R30 contract."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        self.draws = 0

    def draw(self, logits_fp64, temperature: float = 1.0, top_k: int = 0,
             top_p: float = 1.0, seen_ids=None,
             repetition_penalty: float = 1.0) -> int:
        """Sample one token id from a single 1-D fp64 logits vector."""
        z = np.array(logits_fp64, dtype=np.float64).reshape(-1)
        apply_repetition_penalty(z, seen_ids, repetition_penalty)
        _diag_rec = None
        if _DIAG_PATH:
            _t4 = np.argsort(z)[::-1][:4]
            _diag_rec = {
                "n": self.draws,
                "top4": [[int(i), round(float(z[i]), 6)] for i in _t4],
                "margin": round(float(z[_t4[0]] - z[_t4[1]]), 6)
                            if _t4.size > 1 else None,
            }
        # temp <= 0 is the GREEDY contract (argmax), never a silent
        # unscaled multinomial — the Ming sampling-class trap (2026-07-27).
        if temperature is not None and temperature <= 0:
            _g = int(np.argmax(z))
            if _diag_rec is not None:
                _diag_rec["chosen"] = _g
                with open(_DIAG_PATH, "a") as _f:
                    _f.write(json.dumps(_diag_rec) + "\n")
            return _g
        if temperature and temperature != 1.0:
            z = z / float(temperature)

        keep = np.ones(z.shape[0], dtype=bool)
        if top_k and 0 < top_k < z.shape[0]:
            kth = np.partition(z, -top_k)[-top_k]
            keep &= z >= kth  # threshold keep — ties survive (HF semantics)

        # softmax over the surviving set only, fp64-stable
        zk = np.where(keep, z, -np.inf)
        zk = zk - zk.max()
        p = np.exp(zk)
        p /= p.sum()

        if top_p and 0.0 < top_p < 1.0:
            # descending prob, ties by ascending vocab index (lexsort keys
            # are last-key-major): smallest high-prob set reaching top_p.
            order = np.lexsort((np.arange(p.shape[0]), -p))
            csum = np.cumsum(p[order])
            cut = int(np.searchsorted(csum, top_p, side="left")) + 1
            nucleus = order[:cut]
            mask = np.zeros(p.shape[0], dtype=bool)
            mask[nucleus] = True
            p = np.where(mask, p, 0.0)
            p /= p.sum()

        u = self._rng.random()
        self.draws += 1
        chosen = int(np.searchsorted(np.cumsum(p), u,
                                     side="right").clip(0, p.shape[0] - 1))
        if _diag_rec is not None:
            _diag_rec["chosen"] = chosen
            with open(_DIAG_PATH, "a") as _f:
                _f.write(json.dumps(_diag_rec) + "\n")
        return chosen

    def draw_chattts(self, logits_fp64, temperature: float, top_p: float,
                     top_k: int, min_tokens_to_keep: int = 3,
                     seen_ids=None, repetition_penalty: float = 1.0,
                     penalty_window: int = 16,
                     eos_masked: bool = False, eos_id: int = -1) -> int:
        """Vendor-exact ChatTTS-class chain (MiniCPM-o tts contract,
        modeling generate :4425-4462 + gen_logits :4994-5008):
        temperature -> window-frequency penalty -> TopP (min_keep) ->
        TopK (min_keep) -> optional eos -inf mask (min_new_token) ->
        softmax -> ONE multinomial from the stream. Same seeded stream,
        same draw-order contract as draw() (R30 coupling)."""
        z = np.array(logits_fp64, dtype=np.float64).reshape(-1)
        if temperature is not None and temperature <= 0:
            return int(np.argmax(z))          # greedy contract
        if temperature and temperature != 1.0:
            z = z / float(temperature)
        z = _chattts_window_penalty(z, seen_ids, repetition_penalty,
                                    penalty_window)
        _mk = max(1, int(min_tokens_to_keep))
        # TopP FIRST (the vendor warper order), HF keep-set semantics +
        # min_tokens_to_keep floor; ties by ascending vocab index.
        if top_p and 0.0 < top_p < 1.0:
            zs = z - z.max()
            pf = np.exp(zs)
            pf /= pf.sum()
            order = np.lexsort((np.arange(pf.shape[0]), -pf))
            csum = np.cumsum(pf[order])
            cut = max(int(np.searchsorted(csum, top_p, side="left")) + 1,
                      _mk)
            mask = np.zeros(pf.shape[0], dtype=bool)
            mask[order[:cut]] = True
            z = np.where(mask, z, -np.inf)
        # TopK with the min_tokens_to_keep floor (HF: k = max(k, mk)).
        if top_k and top_k > 0:
            _k = min(max(int(top_k), _mk), z.shape[0])
            kth = np.partition(z, -_k)[-_k]
            z = np.where(z >= kth, z, -np.inf)
        if eos_masked and 0 <= eos_id < z.shape[0]:
            z[eos_id] = -np.inf                # min_new_token contract
        _diag_rec = None
        if _DIAG_PATH:
            _t4 = np.argsort(z)[::-1][:4]
            _diag_rec = {
                "n": self.draws, "class": "chattts",
                "top4": [[int(i), round(float(z[i]), 6)] for i in _t4],
                "margin": round(float(z[_t4[0]] - z[_t4[1]]), 6)
                if _t4.size > 1 else None,
            }
        zk = z - z[np.isfinite(z)].max()
        p = np.exp(zk)
        p[~np.isfinite(zk)] = 0.0
        p /= p.sum()
        u = self._rng.random()
        self.draws += 1
        chosen = int(np.searchsorted(np.cumsum(p), u,
                                     side="right").clip(0, p.shape[0] - 1))
        if _diag_rec is not None:
            _diag_rec["chosen"] = chosen
            with open(_DIAG_PATH, "a") as _f:
                _f.write(json.dumps(_diag_rec) + "\n")
        return chosen


def seeded_gaussian(seed: int, shape) -> np.ndarray:
    """Seeded fp32 standard-normal draw — the latent-init twin of
    SeededDrawStream. ONE stream (PCG64(seed), native-fp32 ziggurat) on
    the CPU; both engines consume the SAME array, so generative
    diffusion legs start from bit-identical noise whenever seed and
    shape agree. This is the cross-engine coupling contract for
    image/video latent init, exactly as the draw stream is for sampled
    tokens: the engines may only diverge through compute numerics,
    never through RNG provenance."""
    return np.random.default_rng(int(seed)).standard_normal(
        size=tuple(int(d) for d in shape), dtype=np.float32)


def _chattts_window_penalty(z: np.ndarray, seen_ids, penalty: float,
                            window: int) -> np.ndarray:
    """ChatTTS-class repetition penalty (vendor
    CustomRepetitionPenaltyLogitsProcessorRepeat): alpha = penalty^freq
    over the last `window` draws; negative logits multiply by alpha,
    non-negative divide. Distinct from the HF processor (seen-set,
    freq-blind) — declared per contract as repetition_penalty_class."""
    if not penalty or penalty == 1.0 or not seen_ids:
        return z
    ids = np.asarray(list(seen_ids)[-int(window):], dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < z.shape[0])]
    if ids.size == 0:
        return z
    freq = np.bincount(ids, minlength=z.shape[0]).astype(np.float64)
    alpha = np.power(float(penalty), freq)
    return np.where(z < 0, z * alpha, z / alpha)
