"""Every text condition has its negative — the pooled vector as well as the hidden states.

Classifier-free guidance runs the denoiser twice, conditioned on the prompt and on the negative prompt. A denoiser
can read a prompt through two outputs of its text encoders: a sequence of hidden states (T5, cross- or joint-
attended) and a pooled vector (CLIP's pooler_output, added to the timestep embedding). The vendors encode the
negative prompt through EVERY text encoder and feed both of its outputs to the unconditional pass (Open-Sora v2:
opensora/utils/sampling.py I2VDenoiser.prepare_guidance builds `text + neg + neg`, and `prepare` encodes the whole
list with T5 and CLIP alike). The engine encoded the negative only for the hidden-state encoder and repeated the
prompt's pooled vector into the unconditional row as if it were a shared condition. Measured 2026-10-04 on
Open-Sora-v2 (192x336, 51 frames, the vendor's code fed the engine's noise): the unguided step-0 output equals
the vendor's conditional row at cos 0.99992, while the guided output stood at cos 0.78 — the recovered
unconditional prediction lay 0.16 (relative) from the conditional one where the vendor's lies 0.43.

A shared condition stays shared: an image embedding (Wan-I2V's CLIP image encoder) is passed unchanged to both
passes by its vendor, and its encoder reads no token ids.

The rules, shared by both engines (R30) and torch-free (R33):

    pooled_text_outputs   the outputs of a text encoder (it reads `global.input_ids*`) that feed a loop component,
                          other than the hidden state the CFG split already carries
    negative_port         where the negative of an output is recorded: `<component>.negative_<output>`
    negative_swaps        for a loop component, each conditioning port whose negative the flow recorded, with it
"""
from __future__ import annotations

from typing import Any, List, Mapping, Tuple

_TOKEN_IDS = "global.input_ids"


def negative_port(from_port: str) -> str:
    """The resolver key holding the negative of an encoder output port `<component>.<output>`."""
    comp, out = from_port.split(".", 1)
    return f"{comp}.negative_{out}"


def _loop_components(topology: Mapping[str, Any]) -> List[str]:
    loop = (topology.get("flow") or {}).get("loop") or {}
    return list(loop.get("components") or []) if isinstance(loop, Mapping) else []


def _reads_token_ids(topology: Mapping[str, Any], comp_name: str) -> bool:
    return any(str(c.get("from", "")).startswith(_TOKEN_IDS) and str(c.get("to", "")).startswith(f"{comp_name}.")
               for c in topology.get("connections", []))


def pooled_text_outputs(topology: Mapping[str, Any], comp_name: str) -> List[str]:
    """Outputs of the text encoder `comp_name` that condition a loop component besides its hidden states."""
    if not _reads_token_ids(topology, comp_name):
        return []
    loop = set(_loop_components(topology))
    outs: List[str] = []
    for c in topology.get("connections", []):
        src, dst = str(c.get("from", "")), str(c.get("to", ""))
        if not src.startswith(f"{comp_name}.") or "." not in dst or dst.split(".", 1)[0] not in loop:
            continue
        out = src.split(".", 1)[1]
        if "hidden_state" in out.lower():
            continue
        if out not in outs:
            outs.append(out)
    return outs


def negative_swaps(topology: Mapping[str, Any], resolved: Mapping[str, Any], comp_name: str,
                   skip: Tuple[str, ...] = ()) -> List[Tuple[str, Any]]:
    """(port, its negative) for every port feeding `comp_name` whose negative the flow recorded.

    A recorded negative whose shape is not its positive's is refused by name: the two rows of one batch."""
    swaps: List[Tuple[str, Any]] = []
    for c in topology.get("connections", []):
        src, dst = str(c.get("from", "")), str(c.get("to", ""))
        if not dst.startswith(f"{comp_name}.") or "." not in src or src in skip:
            continue
        key = negative_port(src)
        if key not in resolved:
            continue
        neg, pos = resolved[key], resolved.get(src)
        if pos is not None and tuple(neg.shape) != tuple(pos.shape):
            raise RuntimeError(f"ZERO FALLBACK: '{key}' has shape {tuple(neg.shape)}, '{src}' has "
                               f"{tuple(pos.shape)}: the unconditional condition is not the conditional's kind.")
        if all(s != src for s, _ in swaps):
            swaps.append((src, neg))
    return swaps
