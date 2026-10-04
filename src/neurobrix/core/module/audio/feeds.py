"""What an audio flow feeds its first stage for one recording — computed without running it.

The flows feed the recording's OWN extent (never the trace's, `core/flow/input_extent`), so the
extent a component runs at is a fact of the request: Prism prices the largest feed
(`core/prism/flow_bindings.FlowBindings.audio_feeds`) and the derived census keys every feed
(`tools/derived_census.py`) from this ONE function. It calls what the flows call: the shared
loader and numpy extractors (`mel_dsp` — the Triton flows' own; the compiled rnnt flow computes
its mel in torch from the same loader's samples, so the frame count is the same number), the RNNT
feed plan (`stt_longform.rnnt_feed_plan`), the admission door (`input_extent.admit`). Numpy +
stdlib only: both engines' planning paths import it (R33).

A container that cannot take the recording's extent is refused BY NAME here, at plan time, before
a weight is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def first_audio_stage(topology: Optional[dict]) -> Optional[str]:
    """The component an audio flow feeds the recording's features to — by the flows' own rule:
    the rnnt flow always takes a recording; every other audio flow takes one when its input
    modality is audio, a modality it does not declare being read off its direction (`stt` is
    audio in, `core/flow/audio.py::_preprocess_input`). None for every other flow."""
    flow = (topology or {}).get("flow") or {}
    audio = flow.get("audio") or {}
    stages = audio.get("stages") or []
    if not stages:
        return None
    if flow.get("type") != "rnnt":
        modality = (audio.get("input") or {}).get(
            "modality", "audio" if audio.get("direction") == "stt" else "text")
        if modality != "audio":
            return None
    return stages[0].get("component")


def request_feeds(topology: dict, cache_path, audio_path, dag: dict,
                  container: Optional[str] = None, family: Optional[str] = None
                  ) -> List[Dict[str, tuple]]:
    """Every distinct `{input name: shape}` the flow feeds its first audio stage (`dag`, that
    stage's graph) for the recording at `audio_path`, in feed order — one entry for a recording
    that goes through in one pass, one per distinct window extent of a long-form RNNT run."""
    from neurobrix.core.flow import input_extent as IE
    from neurobrix.core.module.audio import mel_dsp
    comp = first_audio_stage(topology)
    if comp is None:
        return []
    flow = topology.get("flow") or {}
    audio = flow.get("audio") or {}
    cfg = mel_dsp.model_config_dir(Path(cache_path))
    if flow.get("type") == "rnnt":
        from neurobrix.core.config.loader import get_family_config
        from neurobrix.core.module.audio.stt_longform import rnnt_feed_plan
        params = mel_dsp.nemo_mel_params(cfg)
        feats = mel_dsp._nemo_mel(str(audio_path), cfg, None)               # the Triton flow's call
        feat_in, len_in = IE.features_and_length(dag)
        if not family:
            raise RuntimeError(
                f"container {container!r}: the RNNT feed plan reads the family profile's long-form "
                f"window, and no family was given.")
        plan, _overlap = rnnt_feed_plan(feats.shape[2], get_family_config(family).get("long_form"),
                                        params["sr"], params["hop"], family)
        feeds: List[Dict[str, tuple]] = []
        for _start, valid in plan:
            feed = {feat_in: (1, int(feats.shape[1]), int(valid)), len_in: (1,)}
            if feed not in feeds:
                IE.admit(container, comp, dag, feed)
                feeds.append(feed)
        return feeds
    inp = audio.get("input") or {}
    variable = inp.get("variable", "global.input_features")
    axes = IE.input_axes(dag, IE.fed_input(topology, comp, variable))
    if axes is None:
        raise IE.FrozenTraceExtent(
            f"container {container!r}, component {comp!r}: the graph has no input for the flow's "
            f"{variable!r}.")
    prep = mel_dsp.resolve_preprocessing(inp.get("preprocessing"), axes.trace_shape)
    feats = mel_dsp.extract_features_np(prep, str(audio_path), cfg, axes.trace_shape)
    feed = {axes.name: tuple(int(d) for d in feats.shape)}
    IE.admit(container, comp, dag, feed)
    return [feed]


def declared_extent_seconds(family: Optional[str]) -> Dict[str, float]:
    """The family profile's `audio_extent` block — the recording lengths an any-length encoder of
    the family is planned and censused over. Refused by name when the family declares none."""
    from neurobrix.core.config.loader import get_family_config
    if not family:
        raise RuntimeError("the audio extent range is a family profile's value, and no family was given.")
    block = get_family_config(family).get("audio_extent")
    if not block or "min_seconds" not in block:
        raise RuntimeError(
            f"ZERO FALLBACK: {family}.yml audio_extent.min_seconds missing — the shortest recording "
            f"an any-length encoder of this family is censused at is the family profile's value "
            f"(with its source), never a number in a tool.")
    return {k: float(v) for k, v in block.items()}


def extent_feeds(topology: dict, cache_path, dag: dict, container: Optional[str] = None,
                 family: Optional[str] = None) -> Optional[tuple]:
    """`(component, lo feed, hi feed)` — what the flow feeds its first audio stage for the SHORTEST
    and the LONGEST recording the family profile declares, each `{input name: shape}`; None for a
    flow with no audio stage. The two differ on the frame axis alone (an any-length encoder), or
    are one feed (an extractor that pads to its own window: Whisper's 30 s).

    The longest feed is what a SERVED plan prices (no request exists when it is solved) and where
    the derived census's enumerated frame extent ends; the shortest is where it starts. Computed
    by the flows' own functions: the extractor run on silence of the declared duration
    (`mel_dsp.feature_shape_at`), the RNNT window (`stt_longform.rnnt_window_frames` — the flow
    never feeds more than one window, so the family's window IS its longest feed). Both ends pass
    the admission door: a container that cannot take them is refused by name."""
    from neurobrix.core.config.loader import get_family_config
    from neurobrix.core.flow import input_extent as IE
    from neurobrix.core.module.audio import mel_dsp
    comp = first_audio_stage(topology)
    if comp is None:
        return None
    flow = topology.get("flow") or {}
    cfg = mel_dsp.model_config_dir(Path(cache_path))
    declared = declared_extent_seconds(family)
    if flow.get("type") == "rnnt":
        from neurobrix.core.module.audio.stt_longform import rnnt_window_frames
        params = mel_dsp.nemo_mel_params(cfg)
        feat_in, len_in = IE.features_and_length(dag)
        lo_shape = mel_dsp.feature_shape_at("nemo_mel", declared["min_seconds"], cfg)
        window, _overlap = rnnt_window_frames(get_family_config(family).get("long_form"),
                                              params["sr"], params["hop"], family)
        lo = {feat_in: lo_shape, len_in: (1,)}
        hi = {feat_in: (*lo_shape[:2], int(window)), len_in: (1,)}
    else:
        inp = (flow.get("audio") or {}).get("input") or {}
        variable = inp.get("variable", "global.input_features")
        axes = IE.input_axes(dag, IE.fed_input(topology, comp, variable))
        if axes is None:
            raise IE.FrozenTraceExtent(
                f"container {container!r}, component {comp!r}: the graph has no input for the "
                f"flow's {variable!r}.")
        prep = mel_dsp.resolve_preprocessing(inp.get("preprocessing"), axes.trace_shape)
        lo_shape = mel_dsp.feature_shape_at(prep, declared["min_seconds"], cfg, axes.trace_shape)
        if "max_seconds" in declared:
            hi_shape = mel_dsp.feature_shape_at(prep, declared["max_seconds"], cfg, axes.trace_shape)
        else:
            # no declared top: admissible only for an extractor whose output does not depend on
            # the recording's length (it pads to its own window) — proven on a second duration
            hi_shape = mel_dsp.feature_shape_at(prep, 2 * declared["min_seconds"], cfg, axes.trace_shape)
            if hi_shape != lo_shape:
                raise RuntimeError(
                    f"ZERO FALLBACK: {family}.yml audio_extent.max_seconds missing — container "
                    f"{container!r} feeds {comp!r} the recording's own length ({lo_shape} at "
                    f"{declared['min_seconds']} s, {hi_shape} at twice that), and the longest "
                    f"recording it is planned for is the family profile's value.")
        lo, hi = {axes.name: lo_shape}, {axes.name: hi_shape}
    moved = [(k, i) for k in lo for i, (a, b) in enumerate(zip(lo[k], hi[k])) if a != b]
    if len(moved) > 1:
        raise RuntimeError(
            f"container {container!r}, component {comp!r}: the shortest and the longest feed differ "
            f"on more than one axis ({lo} / {hi}) — the extent is not one frame axis.")
    for feed in ([lo] if lo == hi else [lo, hi]):
        IE.admit(container, comp, dag, feed)
    return comp, lo, hi


class BeyondServedExtent(RuntimeError):
    """A served request whose recording is longer than the extent the served plan was priced at."""


def refuse_beyond_served(served_feed: Dict[str, tuple], request: List[Dict[str, tuple]],
                         container: Optional[str], component: str, family: Optional[str]) -> None:
    """A served plan priced `served_feed` (the family's longest declared recording); a request
    whose flow would feed `component` more than that runs at an extent no plan priced — refused by
    name, never run on a plan made for less."""
    import math
    for feed in request:
        for name, shape in feed.items():
            top = served_feed.get(name)
            if top is not None and math.prod(shape) > math.prod(top):
                raise BeyondServedExtent(
                    f"container {container!r}, component {component!r}: this recording feeds "
                    f"{name!r} at {tuple(shape)}, beyond the {tuple(top)} this served plan was "
                    f"priced at — the longest recording the family profile declares "
                    f"({family}.yml audio_extent / long_form). A served plan is solved before any "
                    f"request and is not re-solved per request: run this recording with "
                    f"`neurobrix run`, which plans the request itself.")
