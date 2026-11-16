"""The routing op's operands for its tests, drawn in float32 on the host: hidden [T, H] standard normal, the router
weight [E, H] normal with deviation H^-1/2 (router logits of unit size, as a trained router gives), and a choice bias
[E] normal of mean `bias_mean` and deviation `bias_sd` when asked (None: no bias). A negative mean drives some choice
values below zero, where a dropped group's `fill` of 0 outranks them."""
import numpy as np


def draw(T, E, H, seed, bias_mean=None, bias_sd=0.1):
    r = np.random.default_rng(seed)
    hidden = r.standard_normal((T, H)).astype(np.float32)
    weight = (r.standard_normal((E, H)) / np.sqrt(H)).astype(np.float32)
    bias = None if bias_mean is None else (bias_mean + bias_sd * r.standard_normal(E)).astype(np.float32)
    return hidden, weight, bias
